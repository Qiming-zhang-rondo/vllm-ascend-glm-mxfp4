# A5 shared expert QBMM：MXA8W4 vs MXA8W8

独立验证 `torch_npu.npu_quant_matmul` 的**算子精度、量化损失和 CANN Task Duration**。
对应 vLLM 0.26.0 / vLLM-Ascend v0.26.0rc1 的独立 shared expert Linear 路径。
这里 W8 基线是 **MXFP8 E4M3 × MXFP8 E4M3**，W4 是 **MXFP8 E4M3 × MXFP4 E2M1**；输出 BF16，E8M0 scale，group size 32。

## 一条命令更新并运行

在 A5 已配置好 CANN、torch、torch_npu 的容器内、已有 `vllm-ascend-glm-mxfp4` 仓库目录执行：

```bash
(set -euo pipefail; git fetch origin qbmm-shared-mxfp-bench; qbmm_src=$(mktemp -d /tmp/qbmm-a5.XXXXXX); git archive origin/qbmm-shared-mxfp-bench benchmarks/shared_expert_qbmm | tar -x -C "$qbmm_src"; bash "$qbmm_src/benchmarks/shared_expert_qbmm/run_qbmm.sh")
```

该命令下载分支并把测试目录提取到新的临时目录，保留当前 checkout；结果保存到当前目录的 `qbmm-results/<时间>-<PID>/`。测试源码留在打印/日志可追溯的临时目录，未自动删除。首次获取也可以：

```bash
git clone --depth 1 --branch qbmm-shared-mxfp-bench https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4.git qbmm-a5
bash qbmm-a5/benchmarks/shared_expert_qbmm/run_qbmm.sh
```

**脚本不安装任何包、不编译扩展、不更换运行库。** 原版 CANN golden 依赖容器已有 `numpy`、`torch`、`ml_dtypes`、`en_dtypes`；缺失时明确失败。NPU Python 接口需要支持 `npu_dynamic_mx_quant` 的 FP4 输出和 `npu_format_cast` 的 `customize_dtype/input_dtype` 参数。

必须经 `run_qbmm.sh` 启动：在 Python 启动前设置 `FLA_NPU_DISABLE_PTH=1`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；`ASCEND_LAUNCH_BLOCKING` 未设置时默认为 `1`，已经设置时保留原值。每个量化 case 都使用新进程，首个失败立即停止，保留日志和 plog。

## 自动执行的内容

1. **官方精度用例门禁**：3 条 CANN 原始 CSV 用例，包含 ND、NZ 和 M=10 场景。采用原 shape/dtype/range/tolerance；本 harness 做固定随机种子的输入分配，调用原版 `customize_inputs`、`TestSpec.golden` 和 `isclose_compare`。通过官方 `torch_npu.npu_quant_matmul` 桥调用，未运行完整 TTK runner，也没有自建 ctypes 桥。
2. **W8 派生对照门禁**：把最后一条用例的可精确表示输入改为 W8 ND，使用原版 V3 golden。它是本测试的派生用例，不冒充原始官方 W8 ST 用例。
3. **GLM 形状测试**：每个形状分别运行 W4、W8；量化后输入先通过算子精度门禁，再对同一组已准备好的 NPU 输入测速。相邻形状交替 W4/W8 顺序。

默认 M：`1,16,128,512`；TP：`1,8,16`，共 24 个形状 × 2 种格式 + 4 个门禁进程。

| shared expert 层 | X[M,K] 中 K | W[K,N] 中 N |
| --- | ---: | ---: |
| gate_up | 6144 | 4096 / TP |
| down | 2048 / TP | 6144 |

这是 hidden=6144、shared intermediate=2048 的形状假设；**没有读取用户 GLM-5.2 checkpoint**。M 是本 rank 收到的 token 数，TP 仅用于生成本 rank 的矩阵尺寸，不启动分布式通信。若 shared expert 是复制执行，使用 `--tp-sizes 1`；不要直接套 routed expert 的 TP。实际 config 不同请用 `--shape M K N`。

```bash
# 单个自定义形状，仍然先过官方门禁
bash benchmarks/shared_expert_qbmm/run_qbmm.sh --shape 128 6144 4096
# 只验证门禁
bash benchmarks/shared_expert_qbmm/run_qbmm.sh --smoke-only
# 不依赖 NPU 查看计划
bash benchmarks/shared_expert_qbmm/run_qbmm.sh --dry-run
```

## 精度怎样判断

- `operator_error` / `operator_check`：实际输出对比**已量化输入**的原版 CANN golden，golden 包含 BF16 输出舍入。API smoke 按原 CSV 的 `rtol=0.001, ptol=0.001, atol=0.001`；NZ smoke 和派生 GLM case 使用 `rtol=0.001, ptol=0.001, atol=1e-8`。ptol 是允许不满足逐元素 `isclose` 的比例。额外拒绝 NaN/Inf；不放宽阈值。
- `quantization_loss`：量化后 golden 对比原始 BF16 输入的 FP32 matmul，报告 relative L2、cosine、RMSE、max abs。它衡量量化损失，不作为算子实现正确性门禁。
- `total_error_vs_bf16_inputs`：实际输出相对同一原始浮点参考的总误差。

GLM 形状使用固定种子的 Gaussian BF16 A，W 按 `1/sqrt(K)` 缩放。两格式使用完全相同的原始输入并记录 SHA256。A 用官方 `npu_dynamic_mx_quant` 转 FP8，W 分别转 FP4/FP8；这属于合成数据量化实验，**不代表已校准模型权重或模型任务精度**。精度参考覆盖全部输出元素，未抽样。

## 性能口径

- W4 按 vLLM 的 FP4 packed `[N,K/2]` → `npu_format_cast(...,29,customize_dtype=FP8,input_dtype=FP4)` → transpose；scale `[N,K/32]` → `[K/64,N,2]`，不额外 contiguous，`group_sizes=[0,0,32]`。
- W8 按 vLLM 的 FP8 `[N,K]` → transpose + contiguous，scale 同样重排后 contiguous，`group_sizes=[1,1,32]`。
- 默认 warmup 20 次、正式采集 50 次。所有数据生成、CPU golden、H2D、量化、NZ 转换、首次调用和 warmup 在采集前完成。计时区只有预准备的 QBMM 调用和最终同步。
- 只从 NPU profiler 的 `op_summary*.csv` 读取 **`Task Duration(us)`**，给出 p50、p90、mean、全部样本。Python wall time 仅作为包含启动/setup 的诊断字段，不用于加速比。默认保留 launch blocking，使用热缓存连续调用；不是整模型并发负载。
- 记录实际 `Op Type`、Op Name、dtype、stride、格式。只有单一 `QuantBatchMatmulV3` 或 `QuantBatchMatmulV4` 且总任务数等于正式调用次数才出 QBMM 性能结论；类型改名、额外计算 kernel、混合 V3/V4、拆分多任务、缺失采集都保留 trace 并失败。不会把未知 kernel 静默求和或仅凭 dtype 声称命中某个特化。
- `w8_over_w4_speedup = W8_p50 / W4_p50`，**大于 1 表示 W4 更快**。`latency_reduction_percent` 为正表示 W4 延迟更低。

W4 NZ 对 W8 ND 是这版 vLLM 的实际 Linear 路径对比。这个结果包含两条路径的布局差异。单算子结果不包含 dynamic quant、SwiGLU、通信、shared/routed overlap，不能直接当作端到端吞吐提升。

## 输出与排查

`summary.json`：总体状态、每个 case 的精度和性能、输入哈希与版本。
`comparison.csv`：每个形状的 W4/W8 延迟和加速比。
每个 case 目录包含 `spec.json`、`result.json`、`run.log`、`plog/`；性能 case 包含原始 `trace/`，若可用保存 `npu-smi.txt`。数值门禁失败另存 `accuracy_failure.npz`（actual/expected）。`--timeout` 默认每个子进程 1200 秒。

原 CSV 包含负范围的 E8M0 scale，官方 `customize_inputs` 可能打印“清洗 NaN 并重新生成”的警告，这是原版测试输入准备行为。正式输出若有非有限值依然失败。

## 源码与本地验证

- vLLM-Ascend tag `v0.26.0rc1`，commit `f2f74a16c3c50a76f4349d807918e83edec1e35c`：[W4 Linear](https://github.com/vllm-project/vllm-ascend/blob/f2f74a16c3c50a76f4349d807918e83edec1e35c/vllm_ascend/quantization/methods/w4a8_mxfp4.py)、[W8 Linear](https://github.com/vllm-project/vllm-ascend/blob/f2f74a16c3c50a76f4349d807918e83edec1e35c/vllm_ascend/quantization/methods/w8a8_mxfp8.py)。
- CANN ops-nn commit `2a77283db46e6648ff47bc8277442cf9c721e3c2`；原文件未经修改，来源与 SHA256 见 `vendor/cann-ops-nn/MANIFEST.json`。每次使用 golden 验证哈希；保留 CANN Open Software License Agreement 2.0，见同目录 `LICENSE`。
- 新增 harness 代码为 Apache-2.0；上游 vendor 文件按各自原许可。
- 本地 CPU 验证包括原版 golden、全部官方输入的 packing/scale roundtrip、W4/W8 参数和布局 mock、日志/失败处理、CANN CSV 解析。运行 `python -m unittest discover -s benchmarks/shared_expert_qbmm -p 'test_*.py' -v`。

**尚未在 A5 实机运行，仓库不包含虚构的 NPU 精度或性能结果。**
