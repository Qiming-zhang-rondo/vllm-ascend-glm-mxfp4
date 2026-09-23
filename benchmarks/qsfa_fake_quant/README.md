# QSFA 单算子低精度实验

默认只在 CPU 上验证低精度组合的数值误差，复用现有 Python 和 torch。没有 pip 安装、编译、框架 patch 或模型启动，也不需要 NPU、torch_npu 或 CANN。

后续的真实 Ascend C Q8/C4/O8 设备原型已放在 [qsfa_q8c4_o8](../qsfa_q8c4_o8/README.md)，使用独立编译/测试入口，尚待 A5 验证。本目录继续保留 CPU 伪量化行为。

## 一条命令

在已激活 PyTorch 环境的终端运行，A3/A5 容器或有 CPU torch 的机器均可：

```bash
git -C /workspace/vllm-ascend-glm-mxfp4 pull --ff-only && bash /workspace/vllm-ascend-glm-mxfp4/benchmarks/qsfa_fake_quant/run_qsfa.sh
```

脚本直接在 CPU reference 上运行低精度伪量化实验，默认不调用任何 NPU 算子。当前官方 QSFA 没有待测的 C4/Q8/P8/O8 compute 接口，因此不会把这些伪量化案例伪装成真实低精度 NPU kernel。

`--reference-only` 仍可使用，与默认行为相同。也可将 `--python /path/to/python` 作为脚本的第一组参数指定现有解释器。脚本使用独立 Python 导入路径，避免仓库 `tools/bisect` 等文件遮蔽标准库，并在 Python 启动前关闭已知的 FLA `.pth` 注入。

仅在专门排查原生 QSFA 时显式添加 `--native-baseline`：它会先执行一组 **INT8 cache / BF16 Q** 的 NPU 正确性基线，需现有 torch_npu 和 CANN。2026-09-23 的 A5 用户运行在该原生调用处发生 507014 AI Core timeout，CPU 伪量化尚未开始；超时根因未确认。该原生检查不是数值模拟的必要前提，默认已移除。保留当次 `plog/` 供后续排查，无需为本实验重复运行失败的原生基线。

默认形状：单序列、Q=1、KV=8192、单卡本地 Q heads=8、KV heads=1、NoPE=512、RoPE=64、选中 KV=2048。合成实验运行三个固定 seed，Q/K 源数据为标准正态分布后转 BF16，attention scale 为 `1/sqrt(576)`。这不是采自 GLM 的真实张量。

**输出 1K token 不代表单次 QSFA 的 Q=1K。** 此处默认测试 decode 的一次 Q=1 调用；heads=8 也只是默认单卡形状，不是已核实的模型配置。伪量化形状可以改，例如：

```bash
bash /workspace/vllm-ascend-glm-mxfp4/benchmarks/qsfa_fake_quant/run_qsfa.sh --heads 8 --key-tokens 57344 --query-tokens 1
```

原生 INT8 smoke 的形状固定，以上参数调整的是伪量化实验。reference 每次只展开一个 Q 的选中 KV，不构建完整 Q×K attention 矩阵；大量 Q 行仍会增加 CPU 计算时间。

## 比较内容

| 案例 | Q | KV NoPE | P（softmax 后、PV 前） | 输出 |
|---|---|---|---|---|
| BF16 reference | BF16 | BF16 | BF16 | BF16 |
| `kv4` | BF16 | MXFP4 | BF16 | BF16 |
| `q8_kv4` | MXFP8 | MXFP4 | BF16 | BF16 |
| `q8_kv4_o8` | MXFP8 | MXFP4 | BF16 | MXFP8 |
| `q8_kv4_p8_o8` | MXFP8 | MXFP4 | MXFP8 | MXFP8 |

- Q 默认整个 D576 量化；`--keep-q-rope-bf16` 可只量化 Q NoPE。KV 的 RoPE 始终保留 BF16。
- MLA 的 V 复用同一份 K NoPE，不能独立重新量化 V 来美化结果。
- QK/PV reference 的累加和 softmax 都是 FP32。官方基线在 PV 前将 P 舍入为 BF16；P8 案例改为 MXFP8，不在量化后再次归一化 P。
- MXFP4：E2M1，group32/E8M0，`scale_alg=0`，ties-away。MXFP8：E4M3FN，group32/E8M0，`scale_alg=0`，RNE、饱和到 ±448。该合同不同于 per-token FP32 `absmax/448` scale。
- 输出 O8 从 FP32 累加结果直接量化；基线输出则舍入 BF16。软件 roundtrip 的返回值是反量化后的 FP32 张量，不是伪称 FP4/FP8 的 NPU tensor。
- **PV 的 scale 轴仍是 kernel 设计问题：** cache 的 group32 沿 latent D，PV 的归约轴却是选中 token；P8 在该 token 轴量化。这里模拟数值误差，不能证明缓存能直接被 MX Cube 消费，也不能证明 FP8×FP4 是原生混合指令。

## 结果怎么看

每次保存到仓库 `.qsfa-fake-quant/<时间>-<PID>/`：`results.json`、`run.log`、`plog/`。

终端和 JSON 同时保留各案例相对 **BF16** 和相对 **C4** 的 cosine、relative RMSE；JSON 还包含最大绝对误差、每个 query/head 的 P99 和最差相对 RMSE，以及相邻阶段的增量误差。

默认工程筛选阈值为 cosine ≥ 0.99、relative RMSE ≤ 0.10，可显式指定 `--min-cosine` / `--max-relative-rmse`。它们不是官方算子容差，也不能证明 GLM 模型精度不掉。总体通过要求两个参照都通过；`incremental_screening_passed_vs_c4` 单独表示新增量化相对 C4 的误差是否过门槛。

- 退出码 **0**：数值筛选通过。
- 退出码 **1**：实验算完，但至少一个精度门槛未通过；查看指标，不是运行崩溃。
- 退出码 **2**：原生 baseline 失败、入参错误或运行异常；查看 `stage` 和 `error`。

**本实验不报告伪量化的性能加速比。** `performance.measured=false`：CPU 模拟速度和新 NPU kernel 的性能没有可比性。没有加载模型，也没有测试端到端精度。

本地 CPU 默认三 seed 的初步结果：C4 vs BF16 的 relative RMSE 约 10.5%～10.9%，超过暂定 10% 门槛；Q8+C4+O8 vs C4 约 4.3%，加入 P8 后约 4.8%～5.0%。保留失败结果，不为通过而修改阈值。A5 原生 baseline 本次未通过，不能据此宣称验证了原生算子。

## 可选：重放实际输入

`--input /path/to/inputs.pt` 支持 `torch.save` 的以下字典，只读取张量和基础类型（`weights_only=True`）：

```python
{
    "query": query,          # [Q, local_heads, 576], CPU BF16/FP32
    "kv": kv,                # [K, 576], 逻辑顺序、已反量化；前512维也是V
    "indices": indices,      # [Q, selected_tokens], int32/int64，逻辑token索引
    "scale_value": scale,    # 实际模型attention scale，必须提供
}
```

仅支持单序列、单 KV head、右下角 causal mask，Q 对应 K 前缀最后 Q 个位置；无 sinks。源数据统一舍入 BF16后比较。`-1` 只能在每行末尾填充；不接受重复索引。不能直接传 packed/PA cache，也不能把多序列拼接后当单序列。脚本不会自动抓取模型张量或修改服务。

## 源码依据与本地验证

- 官方 golden 固定 `cann/ops-transformer@55498d91634277d4eec912499c027818a8c167fb`，文件 `attention/kv_quant_sparse_flash_attention/tests/pytest/kv_quant_sparse_flash_attention_golden.py`。原样摘录 `gatherKV`、`softmax`、`_t_increattention_bnsd`，逐函数 SHA 和原 License 保存在 `vendor/`。移除了未使用的 TensorFlow 数据生成依赖。
- 原生 bridge 使用同 ref 的 `examples/test_npu_kv_quant_sparse_flash_attention.py` 的公开 API；调整 shape 和输入范围的内容记录在 `vendor/SOURCES.json`。这是沿用官方 calling path/golden 的适配用例，不是原封不动运行完整官方 pytest 套件。
- roundtrip 对齐 `cann/ops-nn@2a77283db46e6648ff47bc8277442cf9c721e3c2` 的 `quant/dynamic_mx_quant/tests/assets/golden.py`。CPU 交叉验证覆盖 FP16/BF16/FP32 两格式共 122,880 个数值，结果及符号位一致。
- CPU 回归覆盖舍入边界、饱和、零值/极值、尾块、causal mask、空稀疏行、P 注入位置、禁用量化的中性对照、拒绝全零原生输出、官方源码 SHA 和 CLI 失败结果保留。
