# A5 Q8/C4/O8 QSFA 算子原型

这是 **Ascend C 设备代码**，通过独立 Torch `.so` 执行。当前是五个串行 kernel 的研究原型，已在用户的 A5/CANN9.1.1 容器完成默认 shape 的计算正确性与同步 wall 耗时验证，不是已完成优化的融合 QSFA。只做单算子；没有修改 VA/AV、模型或现有 CANN OPP。

## 在现有 A5 容器中执行

激活已有 PyTorch/CANN 环境后运行：

```bash
git -C /workspace/qsfa-a5-optest pull --ff-only && bash /workspace/qsfa-a5-optest/benchmarks/qsfa_q8c4_o8/run_qsfa.sh
```

脚本复用或编译局部 `.so`，默认对比新原型与容器现有的 **BF16 Q/O、FP8 KV QSFA**：共享同一份原始输入、稀疏索引和 attention scale，分别在独立进程中通过计算正确性检查后计时。复用容器的 Python、torch、torch_npu、CANN、CMake 和 C++ 编译器；缺依赖直接说明，不下载、不 pip install、不重编 VA。源码、编译环境和产物摘要匹配时复用已有 `.so`；本次只新增 Python 对照测试，不改变已编译算子的源码或构建摘要。

构建支持已安装的 `ASCConfig.cmake` / `FindASC.cmake`，也支持缺少该 CMake 包但有原生 `bisheng` 的容器。后者自动使用同一 CANN 目录内的编译器、头文件和库；先编译、链接一个包含 Cube、SIMT VF 和 host launch 的最小样例，**不执行设备代码**，再编译实际算子。只存在旧 `ascendc.cmake` 不证明原生 `.asc` 语法可用，须由这个编译检查确认；失败的编译器输出保存在 `.build/<摘要>/toolchain_probe.log`。同一 CANN 内若有多个不同的 `bisheng`，检查失败后尝试下一个；实际算子编译失败则直接停止。不会跨 CANN 版本混用工具链或安装 SDK。

这两份源码使用 SIMD/Cube 和 SIMD 内调用 SIMT VF，构建不使用纯 SIMT 的 `--enable-simt`。参见[官方原生编译说明](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.html)。最小样例通过仅证明基本编译、链接路径；实际 MXFP8 API 兼容性仍由后续算子编译检查，精度与性能仍须 A5 实测。

构建后还会用当前 Python/torch_npu 加载 `.so` 并检查算子注册，成功后才写入构建 manifest；这一检查不调用设备 kernel。加载成功也不等于精度通过。

默认：**Q=1、H=8、K=8192、selected=2048、D=576（NoPE512+RoPE64）**，seed=20260921。H 是单卡本地 head 数。当前仅支持单序列、单 KV head、decode，H∈{8,16,32,64}，selected∈[128,8192] 且为128的倍数。K 可以更长；selected 是实际参与 attention 的稀疏 token 数。

可选参数：`--build-only`、`--candidate-only`、`--jobs 4`、`--python /path/to/python`、`--profile`、`--heads 16`、`--key-tokens 57344`、`--selected-tokens 2048`。`--profile` 为两组各额外采一轮完整调用的 CPU/NPU trace；`--candidate-only` 保留此前只测新原型的行为。这里不运行此前超时的 INT8 baseline；新对照使用已核对的 A5 FP8 cache 合同，但仍需要容器实测，不能据此宣称此前 timeout 根因已解决。

每次运行记录在本目录 `.runs/<时间-PID>/`：`run.log`、`build.json`、`results.json`、`candidate.json`、`native_baseline.json`、共享输入快照、`plog/`，可选 profiler 子目录。编译产物为 `.build/<摘要>/libqsfa_q8c4_o8.so`。启动 Python 前设置 `FLA_NPU_DISABLE_PTH=1`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`ASCEND_LAUNCH_BLOCKING=1`。

## 原生 FP8 QSFA 对照

基线直接调用容器的 `torch_npu.npu_kv_quant_sparse_flash_attention`，复现 `A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention` 与 `custom_kv_rmsnorm_rope` 的缓存合同：

- Q/O BF16，Q 布局 TND；K/V 共用 PA_BSND cache，block_size=256，单 KV head。
- 每 token 656 字节：512字节 E4M3FN NoPE、128字节 BF16 RoPE、16字节 FP32 scale（每128通道一个）。这是 **block128 FP8**，不是新原型的 D32/E8M0 MXFP8 格式，不能混用 scale。
- `attention_mode=2`、`key_quant_mode=value_quant_mode=2`、`quant_scale_repo_mode=1`、`tile_size=128`、`rope_head_dim=64`、`sparse_mode=3`。
- CPU 依照 DynamicBlockQuant 默认规则构造 absmax/448 scale、E4M3 RNE payload，字节解码建立官方 golden；page 尾部补零并由 actual length 屏蔽。准备与 H2D 不计时，不调用 INT8 替代，也不把 CPU golden 当作 NPU 基线。

两组都使用完整算子调用的同步 wall latency，warmup/iterations 一致。`COMPARISON` 输出两组 p50、`baseline_p50 / candidate_p50` 加速比与 `(1 - candidate_p50 / baseline_p50) * 100%` 延迟变化；新原型更慢时如实报告小于1的加速比和负百分比。这是同一 workload 下不同量化/缓存实现的对比，不代表仅 Q8/O8 的收益，也不是纯 kernel 时间或端到端加速。

任一组计算错误都不生成加速比；候选量化筛选失败但计算正确时仍完成对比，最终保留 `quantization_failed` 和非零退出码。基线失败时保留已经完成的新原型结果及失败阶段，不回退到参考实现。

## 实际计算路径

1. **GatherKernel / SIMT**：根据给定稀疏索引读取 C4 cache；E2M1 数值无损展开成 E4M3，保留原 D32 E8M0 scale；同时生成 PV 使用的 BF16 V 和 BF16 RoPE。
2. **QkKernel / Cube**：NoPE 使用 `mx_fp8_e4m3_t` 的 `LoadData + Mmad`，FP32 L0C 累加；BF16 RoPE 乘积加到同一 FP32 L0C。
3. **SoftmaxKernel / SIMT**：FP32 scale、max、exp、sum、归一化，P 舍入 BF16。
4. **PvKernel / Cube**：BF16 P × BF16 V，沿 selected token 归约，FP32 累加并输出。
5. **OutputKernel / SIMT**：从 FP32 PV 结果直接生成 O8 payload 和 E8M0 scale，不额外插入 BF16 输出舍入。

Q 的 D576 全部以 MXFP8 存储；RoPE 部分解码为 BF16 后计算。K/V NoPE 共用同一份 C4 数据，K RoPE 保留 BF16。**这里没有 FP8 累加，没有 P8，也不假设硬件直接支持 FP8×FP4。** C4 的 D32 scale 不能直接充当 PV token 归约轴的 scale。

首版显式物化 gather、score、P 和转置 V，使用保守的单缓冲 Cube 搬运和同步。额外 GM 读写、kernel 启动以及小 M 利用率可能抵消低精度 QK 的收益；此版的耗时不能代表将来融合版本的性能上限。

2026-09-24 原生对照已由用户在A5跑通：原型p50=452.614µs，原生FP8-cache QSFA p50=131.050µs；当前原型耗时为原生的3.4538倍，整体量化筛选仍失败。对照官方A5流水、具体优化位置及下一版buffer/精度约束见[源码性能调研](OPTIMIZATION_REVIEW.md)。调研没有改动当前可执行算子。

## 输入、输出合同

调用 `torch.ops.qsfa_q8c4_o8.forward(q, qs, kv, ks, rope, indices, scale)`，输入均为同一 NPU 的连续 ND tensor，storage offset=0。

| 参数 | shape / 存储 | 含义 |
|---|---|---|
| q | uint8 `[H,576]` | E4M3FN 原始字节 |
| qs | uint8 `[H,18]` | E8M0，每32个通道一个 scale |
| kv | uint8 `[K,256]` | E2M1，偶数 D 在低 nibble，奇数 D 在高 nibble |
| ks | uint8 `[K,16]` | E8M0，沿 NoPE D32 |
| rope | BF16 `[K,64]` | K RoPE |
| indices | int32 `[selected]` | 唯一、有效的逻辑 token 索引，无 `-1` padding |
| scale | 正的有限 FP32 可表示值 | 实际 attention scale |
| output | uint8 `[H,512]` | E4M3FN 原始字节 |
| output_scale | uint8 `[H,16]` | E8M0，沿输出 D32 |
| status | int32 `[1]` | 0正常；1索引越界，此时结果不可消费 |

字节载体由本算子解码；**不是**把 uint8 直接当 ACLNN FP8/FP4 tensor 使用。这里是分离 payload/scale/RoPE 的逻辑 cache，尚未适配框架 PA cache、block table 或656字节 cache 行。输出必须连同 scale 一起消费。

量化合同：E8M0 group32，`scale_alg=0`；Q/O 使用 E4M3 RNE并饱和到±448，C4 使用 E2M1 ties-away。输入量化在 CPU 完成并逐字节解码核查后才复制到 NPU，避免把测试输入构造混进待测 kernel。

## 精度和耗时如何判定

- `operator_correctness_passed`：真实输出对 decoded Q8/C4 输入的官方 QSFA golden（保留 BF16 P/V 边界，最后 O8），暂定 cosine≥0.999、relative RMSE≤0.02。失败立即停止，不跑性能。
- `quantization_screening_passed`：真实输出相对原始 BF16 Q/K/V 的差异，暂定 cosine≥0.99、relative RMSE≤0.10。
- `incremental_screening_passed_vs_c4`：真实输出相对 BF16 Q+C4 cache+BF16 O 的新增误差，阈值仍为0.99/0.10。
- 每项同时报告全局和逐 head 最差误差；阈值是工程筛选，不是官方新算子认证或 GLM 模型精度保证。

`status=quantization_failed` 表示算完、计算正确性通过，但相对 BF16 的量化筛选未过，仍保留性能。现有随机输入的 C4 本身就可能超过10%门槛，不会放宽阈值来制造 PASS。`status=failed` 要检查 `stage/error`；`compute_verified` 只有真实 NPU 数值检查通过才为 true。

候选性能测 **完整自定义调用的同步 wall latency**：包括 host dispatch/分配、五个设备 kernel 和同步，排除 CPU 量化/golden、H2D、加载、warmup、输出读回。它不是纯 kernel Task Duration；`--profile` 可以查看五个设备任务，不能只取 QK 时间代表 QSFA。默认对照还会测原生 QSFA 的完整调用；只有两组实际运行并通过各自的计算正确性检查，才报告加速比。

可用 `--input /path/inputs.pt` 重放 `torch.save` 的 `{query:[1,H,576], kv:[K,576], indices:[1,selected], scale_value:float}`，张量在 CPU。输入先舍入 BF16；无自动模型抓取。单次 decode 的 Q=1 与生成1024个输出 token 是两个概念。

## 源码依据和验证边界

- CANN skill：`cann/cannbot-skills@3074681ba927916f99f5a9ac808b4e6797934ce5` 的 `ascendc-direct-invoke-template`、`torch-ascendc-op-extension/routes/direct-invoke.md`、`ascendc-simt-best-practices`。采用 ASC CMake 或原生 bisheng 编译，加直接 launch；没有自制 ACL ctypes binding。
- [QLI V2 Cube reference](https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h)：`LoadQScaleToL1`、`LoadKScaleToL1`、`LoadQueryToL0a`、`LoadKeyToL0b`、`ComputeL0c` 提供 MX L0 类型、scale pair 的 Dn2Nz 和 NT LoadData 合同。
- [QSFA 官方 golden](https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/tests/pytest/kv_quant_sparse_flash_attention_golden.py)：复用此前固定的 `gatherKV/softmax/_t_increattention_bnsd`，见相邻 `qsfa_fake_quant/vendor/SOURCES.json`。
- [原生 QSFA 合同检查](https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp)：`CheckFeatureMlaAntiquantDtype/Attr` 检查 FP8/BF16、mode2 与 tile128；[VA A5 调用](https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/blob/360151909ac763ec2ed515cab6771bf6a7cdbff4/vllm_ascend/device/device_op.py) 的 `A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention` 直接调用 public torch_npu API；[DynamicBlockQuant](https://gitcode.com/cann/ops-nn/blob/2a77283db46e6648ff47bc8277442cf9c721e3c2/quant/dynamic_block_quant/docs/aclnnDynamicBlockQuant.md) 定义 FP8 payload 和 FP32 block scales。
- [QuantMatmulWeightNz](https://gitcode.com/cann/ops-nn/blob/2a77283db46e6648ff47bc8277442cf9c721e3c2/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md)：MXFP8×MXFP8、FP32输出合同；同ref依赖的 `ops-tensor@781745c86312b478009742851b6cfc1967da8dda` 中 `KernelMatmulMixWeightPrologue::CopyConvertStoreWeight` 用 `ShiftW4ToW8`，配合 `BlockMmadWeightPrologueMx::CopyCL0c2Gm` 的×64补偿。本原型直接使用真实 E2M1→E4M3 数值映射，不照搬该位移技巧。
- [SIMT 执行空间规则](https://asc.gitcode.com/api/SIMT-API/SIMD_SIMT_hybrid_programming_intro/extended_syntax/function_execution_space_qualifier.html)：VF 的 codec helper 使用 `__simt_callee__`。

本机完成 CPU payload/golden/失败门槛测试、实际 `codec.h` 的宿主 C++ 编译及全部 FP8码/舍入边界检查、支持shape的分块地址核对、源代码同步检查。同步脚本最初无法解析泛型 `HardEvent event`，报两条未知方向候选；改为六个显式 Set/Wait 方向函数后候选为0。flow分析仅覆盖两处 DataCopy，跨函数的 Mmad/LoadData/Fixpipe 生命周期另行人工核对，静态检查不证明设备执行正确。

2026-09-23 本地验证：相关4个测试文件共32项及23个子测试通过，Ruff check/format、`bash -n`通过。完整仓库 `bash format.sh ci` 因本地未安装 pre-commit 而未执行，不额外安装依赖。

2026-09-24 构建入口修复：相关5个测试文件共43项及31个子测试通过，新增多工具链目录、符号链接环、编译检查失败后回退与实际内核编译失败即停止的本地回归；构建命令用 mock 验证流程。Ruff、shell语法检查通过；本机没有 CANN 编译器/A5，尚不能宣称实际 CANN 编译通过。仓库完整格式检查仍因缺少 pre-commit 未执行。

2026-09-24 用户 A5 日志确认：CANN9.1.1 的原生 bisheng 完成 `.so` 编译、链接；随后 `torch.ops.load_library` 报 `NPUBridge::GetNpuStorageImplDesc` 未定义符号，未进入 NPU 计算。布局检查已改为通过 dispatcher 调用 [torch_npu v2.10.0 注册的 `npu::get_npu_format(Tensor) -> int`](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/npu_native_functions.yaml)，保留严格 ND 检查，避免依赖没有导出标注的 NPUBridge 内部接口；同时让未解析的外部符号在链接阶段报错。

加载修复本地验证：49项及31个子测试通过，包含实际编译、链接、运行 CPU libtorch 的 C++ Dispatcher 测试（ND接受、NZ拒绝、int64返回值、schema缺失及随后注册），以及加载失败时不生成成功 manifest 的回归。该测试没有 torch_npu/A5，不代表设备验证。

随后用户提供的 `link.txt` 确认本容器使用 ASC CMake 编译 `.asc`、使用 `/usr/bin/c++` 最终链接，而不是直接 bisheng 后备路径，因此没有 `toolchain_probe.log`。原链接命令缺少 Ascend C runtime 的完整依赖；容器内 `libprofapi.so` 导出 `MsprofReportApi`，`libmmpa.so` 导出 `mmGetTid`，库本身不缺失。构建现按 [CANN 9.1 内置库清单](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation_basic_usage.md) 显式链接静态 `ascendc_runtime` 及其后的共享依赖 `runtime/profapi/unified_dlog/mmpa/ascend_dump/c_sec/error_manager/ascendcl`。两条构建路径使用同一列表，保留未定义符号检查及构建后的真实加载检查，不通过忽略链接错误来绕过问题。

2026-09-24 用户 A5 实测默认 shape：候选输出与 decoded-payload golden 完全一致；同步 wall p50=0.500839ms、mean=0.473957ms（warmup5/iters20）。相对原始 BF16 的 relative RMSE=0.115499，量化筛选仍失败；相对 C4/BF16 reference 的新增 relative RMSE=0.042727，通过增量筛选。这份旧日志没有原生 QSFA 耗时，不能推导加速比。

新增原生对照的本地验证：8个相关测试文件共65项和42个子测试通过，覆盖 PA656 字节/scale/页边界、官方 golden、原生数值失败禁止计时、共享输入、两侧量化失败保留状态，以及减速和基线失败的汇总行为。Ruff、shell语法及 diff 检查通过；完整仓库格式检查仍因缺少 pre-commit 未运行。CPU 与 mock 测试不代表原生基线已在 A5 执行。

**待 A5 验证**：新增 FP8 原生 QSFA 对照与加速比、更多输入及设备任务耗时。尚未支持 torch.compile/ACLGraph、prefill、多batch、框架接入或端到端模型验证。包含 CANN 派生代码的 `csrc/matmul.asc` 保留 CANN2.0许可，见 `CANN_LICENSE`。
