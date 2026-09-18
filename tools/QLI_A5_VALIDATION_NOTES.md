# A5 QLI V2 MXFP4 单算子验证复盘

记录日期：2026-09-15。数据来自用户在 A5 容器执行后提供的日志，不是开发机上的模拟结果。
测试代码版本：本仓 `a3f53d4487ff070140acd28701bb4a3f63ad080b`；CANN 9.1.0，
torch 2.9.0+cpu、torch_npu 2.9.0.post7，设备 Ascend950DT_9582，runtime SOC 260。
开发机没有 NPU；本地回归只能验证测试工具，不能代替设备验证。

## 已得到的结果

官方小 shape 和旧脚本大 shape 都已经完成真实 MXFP4、FP8 QLI 计算及计时。
**旧脚本总状态仍是 `failed`：算子正确性通过，但 MXFP4 的量化损失门槛未通过。**
不能把“完成计算”“算子正确”“量化损失达标”“模型精度达标”合并为一个结论。

| 配置 | 官方 PA 对照 | 旧脚本默认配置 |
| --- | --- | --- |
| B / Q tokens / K tokens | 1 / 1 / 128 | 1 / 1 / 8192 |
| Q heads / K heads / D | 1 / 1 / 128 | 64 / 1 / 128 |
| Q 逻辑 shape | `[1, 1, 128]` | `[1, 64, 128]` |
| K cache 逻辑 shape | `[1, 128, 1, 128]` | `[64, 128, 1, 128]` |
| Q/K layout | TND / PA_BBND | TND / PA_BBND |
| block size / TopK / mask | 128 / 64 / 0 | 128 / 2048 / 3 |
| max_seqlen_q | 1 | -1 |
| 精度参考 | 官方输入和 golden | 同一 FP16 输入生成两种量化 payload，分别检查算子计算与量化损失 |
| 计时 | CANN compute Task Duration | Python/ACLNN 准备、workspace、提交和同步的 wall latency |

上表为逻辑维度。MXFP4 每字节承载两个 E2M1 值，D128 的 payload 最后一维占 64 字节，
不能把存储字节数与逻辑 head_dim 混为一谈。

### 官方 PA 对照：均通过

两例均预热 5 次、采集 20 次。`return_value=0`，验证 TopK 索引集合，不单独验证输出 scores。

| 用例 | 模式 | TopK 检查 | P50 / μs | P90 / μs | Mean / μs |
| --- | --- | --- | ---: | ---: | ---: |
| `MXFP4_PA_20` | MXFP4，mode 5 | 通过，索引集合与 golden 一致 | 8.377 | 8.791 | 8.344 |
| `FP8_PA_04` | FP8，mode 1 | 通过，索引集合与 golden 一致 | 9.257 | 9.904 | 9.306 |

这两次运行中，C4 的 P50 减少 0.880 μs，即 9.51%，FP8/C4 P50 比值为 1.105。
这是小 shape、同步启动、profiler 下的设备任务观测，不是 GLM 长上下文吞吐结论。
两例使用各自官方合成数据，不是同一原始输入的 C4/C8 量化损失实验。
两组 META 用例的完整性能结果不在这次已核实范围内。

### 旧脚本大 shape：计算通过，C4 量化门槛未通过

固定 seed=20260911，预热 5 次、采集 20 次，`check_cache_layout=false`。

| 指标 | MXFP4 | FP8 |
| --- | ---: | ---: |
| `operator_correctness_passed` | true | true |
| 对 decoded payload 的 TopK recall | 99.3164% | 99.5117% |
| 对 decoded payload 的 score NMAE | 0 | 0.00139910 |
| 对原始 FP16 输入的 TopK recall | **89.6484%** | 97.7051% |
| 对原始 FP16 输入的 score cosine | 0.99931943 | 0.99996370 |
| 对原始 FP16 输入的 score NMAE | 0.03501829 | 0.00695632 |
| `quantization_thresholds_passed` | **false** | true |
| wall P50 / ms | 0.241010 | 0.222950 |
| wall P90 / ms | 0.253430 | 0.229120 |
| wall Mean / ms | 0.243478 | 0.223792 |

总状态失败的具体原因是 C4 的 recall `0.896484375 < 0.9`，差 0.3515625 个百分点；
cosine 和 NMAE 门槛均满足。不是再次发生 AI Core timeout，也不是算子计算检查失败。
`check_case()` 最后将两个独立 gate 合并为总状态；保留失败，不降低阈值制造 Pass。
这里的原始输入参考用 FP16 Q/K 做 FP32 计算，因此差异包含输入量化和计算舍入的合并影响，
不是隔离出来的纯 MXFP4 编码误差，也不能直接推导模型任务精度。

算子正确性使用返回 score 与 decoded reference 的接近程度，以及选中 score 是否满足
TopK 边界容差，并不要求索引 recall 必须为 100%。本次 C4 两项条件均为 true；
旧脚本的 `rtol=0.02, atol=0.02` 是临时实验容差，不等同于官方用例的判据。
BF16 舍入、同分或近分项可能交换索引；本 JSON 只能证明差异满足所设边界容差，
不能断言这 14 个索引差异全是精确同分，也不能仅由非 100% recall 认定算子错误。
JSON 中 cosine `1.000000119...` 是浮点误差，应理解为约 1，不代表超出完美一致的精度。

旧脚本中 C4 wall P50 比 FP8 高约 8.10%，**不能宣称大 shape 的 C4 更快**。
这不与官方小 shape 的 9.51% 下降冲突：shape、调用桥和计时口径均不同。
两份结果不能拼成纯 kernel 加速结论。

## 遇到的错误和处理

| 问题或现象 | 核实与处理 | 依据 / 状态 |
| --- | --- | --- |
| 单算子验证反复尝试安装、下载依赖 | 验证应首先复用容器里的 torch、torch_npu、CANN；缺算子时才离线构建所需 QLI，缺依赖则报告 | `5a22f6771`；`build_qli_v2_mxfp4_a5.py::check_prerequisites` |
| CANN 9.1 的 `acl_base.h` 存在，却报告所有 enum 缺失 | 旧 Python parser 只看转发头文本，没有展开 include；改用已有编译器预处理后解析 | `614d687ad`；`qli_container_backend.py::parse_acl_enums` 及头文件加载逻辑 |
| 生成脚本报 `No module named regex` | 生成脚本改用 Python 标准库 `re`，没有安装额外 Python 包 | `046e425b7` |
| 报“不支持 NZ/FRACTAL storage”，实际是基础格式 | 自写 descriptor 的基础格式白名单不完整，五维 scale 的 NCDHW 被误拒绝；按官方转换接受 ND/NCHW/NHWC/NCDHW，仍拒绝不透明格式 | `1053bfc7f`；`QLIBackend._descriptor`；不靠 dtype 强转解决 |
| MXFP4 score 与自写 FP32 golden 不一致 | 参考算术没有匹配该路径的 BF16 QK/ReLU 结果、BF16 weights 与逐 head BF16 FMA；修正 decoded reference，保留原始 FP16 输入参考用于衡量量化损失 | `6d9ee60a8`；`test_qli_v2_mxfp4_a5.py::reference_scores` |
| 输入生成、量化、metadata、compute 混在一起，异步错误难定位 | CPU 准备原始输入和 golden，分阶段同步与记录；旧脚本后来也在 CPU 准备量化 payload | `947717c9a`、`7fe4b14ee`、`a5f245bda`；这些改动是隔离问题，不能都记作内核修复 |
| 加性能后报 507014 / AI Core timeout | 曾同时改变同步启动设置和多用例运行方式，破坏了对照；恢复 `ASCEND_LAUNCH_BLOCKING=1`，性能模式每例独立进程，精度先通过才预热和计时 | `eb2e57583`；隔离后仍出现过超时，故此项不是已证实的最终根因 |
| 日志里 `fla_npu` vendor 自动排到私有包前面 | 查到 FLA 的 Python `.pth` 启动注入机制；子进程启动前设置 `FLA_NPU_DISABLE_PTH=1`，关闭第三方后端自动导入，恢复私有路径，并在真实 dlsym 位置打印 dladdr 来源 | `a3f53d448`；官方 PA 与旧大 shape 在修正环境后的实测中都完成计算 |
| `17.1875%` / `18.75%` 却显示 Pass | 这是未排序索引逐位置接近的比例，不是 TopK 集合 recall；官方比较先排序检查集合，两份 PA 日志均显示集合相同 | `result_compare_method.py::check_result`；没有修改官方阈值 |
| profiler 提示停止时仍处于 RECORD | 不凭 warning 猜测成败；本次导出完成，汇总器要求恰好 20 条有效 QLI compute 记录，两例满足 | `qli_official_perf_summary.py::summarize_case`；不能推广为所有采集都完整 |
| 大 shape 末尾 `Accuracy gate failed` | 计算和计时已完成；C4 相对原始输入的 recall 未达 90% | 最新用户 JSON；仍为失败，不改成通过 |
| host tiling 库缺 `Ops::Base::ToString(gert::Shape const&)` | 早期两次官方通过日志仍有这个加载错误；9 月 18 日新容器离线构建中，它导致 `liboptiling.so` 加载失败，随后 tiling struct 未注册、`QLIV2TilingData` 未定义、`.o` 未生成。现将 53 处 shape 日志格式化改为局部实现，待 A5 重新构建验证 | 仅改变日志字符串的生成方式，保持格式及所有检查条件、tiling 和 kernel 不变；不能据此宣称历史运行的 OPP 来源或 ABI 冲突已确定 |

### 2026-09-18：QLI 构建期间的 tiling 库加载失败

A5 默认启用 `BUILD_WITH_3_8_PACKAGE`（`csrc/CMakeLists.txt`）；
`csrc/cmake/custom_build.cmake` 的该分支未链接 `opsbase`。
QLI 的 `op_host/quant_lightning_indexer_v2_tiling.cpp` 却有 53 处日志调用依赖
`Ops::Base::ToString(gert::Shape const&)`。本次日志明确报告该符号无法解析。
修复用同文件已有的 `ToStringRaw` 加局部 `FormatShape` 保持 `[dim0, dim1]`
格式，移除这项外部依赖；没有给 device kernel 强行包含 host 头文件。

生成关系可对照 `ops-transformer@55498d916` 的
`tests/ut/framework_normal/op_kernel/scripts/gen_tiling_head_file.py`：
先加载 tiling 库并注册，随后获取 tiling 信息并写入 kernel 所需头文件。
因此应处理更早的库加载失败，而不是手工补齐最终缺失的 `.o`。

本地 CPU 回归编译、链接并加载真实 formatter helper，覆盖空 shape、动态维、
普通维和 64 位维度，同时阻止外部 shape formatter 调用重新引入。
这不代表完整 CANN 编译或 A5 执行已通过。
容器中需重新运行 `tools/test_qli_v2_mxfp4_a5.sh --build-op --check-only --jobs 8`，
构建成功后 source `tools/activate_qli_mxfp4_a5.sh`；本修复不需要重新安装 VA。

## 关于 tiling、kernel 和加载来源，纠正此前的推断

`MXFP4_PA_20` 是用例名，不是报错类型，也不能仅凭它失败认定该场景不受支持。
小用例通过不代表大用例通过；TND+PA 和 BSND+BSND 更不能直接互相证明。
CANN `QuantLightningIndexerV2Tiling::DoTiling()` 将 Q/K dtype、输出 dtype、分页标志、
Q/K layout 编入 tiling key；其他参数也可能影响 kernel 内的运行分支。

**本轮从官方精度入口到加载隔离的修正没有修改 QLI device kernel 或 tiling 算法。**
`git diff 48ddbd2cd..a3f53d448 -- csrc` 为空。此前对 tiling、metadata、block 数量的判断
是排查假设，不能沉淀成“已确认并修复的 tiling bug”。

已经确认的是失败日志中的 vendor 优先级污染，以及隔离后的成功运行。
FLA 的公开启动钩子与日志中的路径组合一致；但未取得失败进程完整的加载溯源，
不能把所有历史 timeout 的根因定论为 FLA 串包。
特别是旧 ctypes 路径的早期失败报告已经显示 API 来自私有库：
仅凭 API 所属库相同，不能排除 OPP/tiling/设备 kernel 注册来源的差异。

`QLIV2_API_LIBRARY` 的四行 dladdr 只证明 host ACLNN 入口归属。
最终 tiling 注册、设备 kernel `.o` 来源仍需实际 plog/profiler 与二进制对应证据；
manifest 中的预期路径和按故障 PC 找到的反汇编候选不能代替它。
此次保留已有成功结果，不为补推断要求用户反复重跑。

## 以后采用的验证顺序

1. 复用容器，先跑官方最小精度用例、原始 golden 和官方 bridge。Metadata 或 API 存在不能替代真实 mode-5 compute 完成。
2. Python 启动前处理 `.pth` 和后端自动导入。确认实际库来源；不能只打印启动配置就声称来源已核实。
3. 精度通过后，复用同一批输入和调用测性能。保留同步设置，记录实际采样数和计时单位。
4. 再测试目标工作负载。记录 Q/K 逻辑与物理 shape、dtype、scale、layout、stride、offset、TopK、mask、metadata 和量化模式；不能只检查 dtype。
5. 失败先区分准备阶段、metadata、compute、数值 gate、profiler；先读本次日志、JSON 和 plog，再决定改什么。一次只改变有证据支持的因素。
6. 单独报告执行完成、算子正确性、量化损失、cache 覆盖、性能和模型精度。未跑 padded/offset cache、NPU quantizer、cache producer 或模型时，不宣称这些已通过。

已核实的官方 PA 对照入口：

```bash
git -C /workspace/vllm-ascend-glm-mxfp4-optest pull --ff-only && python3 /workspace/vllm-ascend-glm-mxfp4-optest/tools/run_qli_official_a5.py --perf --cases MXFP4_PA_20,FP8_PA_04
```

旧下载脚本的大 shape 复测入口（两个开关必须放在 Bash/Python 启动前）：

```bash
FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 bash test_qli.sh --update
```

这些是复用方法，不表示本次需要再跑。旧脚本 `--update` 更新 checkout，不替换正在执行的
下载版 Bash；环境变量前缀可直接由其所有 Python 子进程继承。

## 源码与运行证据

- 本仓固定版本 [a3f53d448](https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/tree/a3f53d4487ff070140acd28701bb4a3f63ad080b)：[官方入口](run_qli_official_a5.py)、[旧 benchmark](test_qli_v2_mxfp4_a5.py)、[旧 descriptor/ACLNN adapter](qli_container_backend.py)、[性能汇总](qli_official_perf_summary.py)。
- CANN `ops-transformer@55498d91634277d4eec912499c027818a8c167fb`：`attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_stc.py::{BASE,TestCases,QUANT_PROFILES}` 定义 PA 两例；同目录 `result_compare_method.py::check_result` 与 `test_quant_lightning_indexer_v2_single.py::test_qliv2` 定义精度检查。文件映射与 SHA256 见 [TEST_SOURCES.json](vendor/cann_qli_v2/TEST_SOURCES.json)。
- 同一 CANN ref：`torch_extension/cann_ops_transformer/common/aclnn_common.h::{GetCustOpApiHandlers,GetOpApiFuncAddr}` 定义库选择顺序；`attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp::QuantLightningIndexerV2Tiling::DoTiling` 定义 key。桥接仅添加 dladdr 日志的差异见 [BRIDGE_SOURCE_MANIFEST.json](vendor/cann_qli_v2/BRIDGE_SOURCE_MANIFEST.json)。
- FLA `v26.6.0`：[fla_npu_opp_env.pth](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.6.0/torch_custom/fla_npu/fla_npu_opp_env.pth) 导入 [fla_npu_opp_env.py::_setup](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.6.0/torch_custom/fla_npu/fla_npu_opp_env.py#L24)，在 site 初始化时前置 vendor，提供 `FLA_NPU_DISABLE_PTH` 开关。不能假定 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 会同时关闭 Python `.pth`。

用户提供的两次官方运行编号分别是 `20260915-152025-114486`（C4）与
`20260915-153247-124268`（FP8）；旧大 shape 的输出文件为 `qli_a5_results.json`。
下列为收到的原始粘贴文件 SHA256，仅用于区分证据；不将包含容器路径的完整日志提交到仓库。

| 证据 | SHA256 |
| --- | --- |
| 官方 C4 PA 日志 | `ae8dedc261838c5e622a3a7d665dc593ad1d95bc62f42aa4696f3ca1d0307a22` |
| 官方 FP8 PA 日志 | `57fb2191c1f9aa6ae57e0da3a6bf058f277890cb724d718393e6a5fe27cde8d5` |
| 旧大 shape JSON | `4a908d71e493ec626cbcb49637c38a7388e2759d7acb95ca1105816276cd179a` |
