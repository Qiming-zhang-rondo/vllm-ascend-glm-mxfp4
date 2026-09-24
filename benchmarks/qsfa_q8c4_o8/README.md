# A5 Q8/C4/O8 QSFA 算子原型

这是 **Ascend C 设备代码**，通过独立 Torch `.so` 执行。当前是五个串行 kernel 的研究原型，尚未在 A5 编译、验证或测得加速，不是已完成优化的融合 QSFA。只做单算子；没有修改 VA/AV、模型或现有 CANN OPP。

## 在现有 A5 容器中执行

激活已有 PyTorch/CANN 环境后运行：

```bash
git -C /workspace/vllm-ascend-glm-mxfp4 pull --ff-only && bash /workspace/vllm-ascend-glm-mxfp4/benchmarks/qsfa_q8c4_o8/run_qsfa.sh
```

脚本会编译局部 `.so`，然后做精度检查，通过计算正确性门槛后再计时。复用容器的 Python、torch、torch_npu、CANN、CMake 和 C++ 编译器；缺依赖直接说明，不下载、不 pip install、不重编 VA。源码、编译环境和产物摘要匹配时复用已有 `.so`。

构建支持已安装的 `ASCConfig.cmake` / `FindASC.cmake`，也支持缺少该 CMake 包但有原生 `bisheng` 的容器。后者自动使用同一 CANN 目录内的编译器、头文件和库；先编译、链接一个包含 Cube、SIMT VF 和 host launch 的最小样例，**不执行设备代码**，再编译实际算子。只存在旧 `ascendc.cmake` 不证明原生 `.asc` 语法可用，须由这个编译检查确认；失败的编译器输出保存在 `.build/<摘要>/toolchain_probe.log`。同一 CANN 内若有多个不同的 `bisheng`，检查失败后尝试下一个；实际算子编译失败则直接停止。不会跨 CANN 版本混用工具链或安装 SDK。

这两份源码使用 SIMD/Cube 和 SIMD 内调用 SIMT VF，构建不使用纯 SIMT 的 `--enable-simt`。参见[官方原生编译说明](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.html)。最小样例通过仅证明基本编译、链接路径；实际 MXFP8 API 兼容性仍由后续算子编译检查，精度与性能仍须 A5 实测。

默认：**Q=1、H=8、K=8192、selected=2048、D=576（NoPE512+RoPE64）**，seed=20260921。H 是单卡本地 head 数。当前仅支持单序列、单 KV head、decode，H∈{8,16,32,64}，selected∈[128,8192] 且为128的倍数。K 可以更长；selected 是实际参与 attention 的稀疏 token 数。

可选参数：`--build-only`、`--jobs 4`、`--python /path/to/python`、`--profile`、`--heads 16`、`--key-tokens 57344`、`--selected-tokens 2048`。`--profile` 额外采一轮完整算子的 CPU/NPU trace。这里不运行此前超时的原生 INT8 QSFA baseline。

每次运行记录在本目录 `.runs/<时间-PID>/`：`run.log`、`build.json`、`results.json`、`plog/`，可选 profiler 子目录。编译产物为 `.build/<摘要>/libqsfa_q8c4_o8.so`。启动 Python 前设置 `FLA_NPU_DISABLE_PTH=1`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`ASCEND_LAUNCH_BLOCKING=1`。

## 实际计算路径

1. **GatherKernel / SIMT**：根据给定稀疏索引读取 C4 cache；E2M1 数值无损展开成 E4M3，保留原 D32 E8M0 scale；同时生成 PV 使用的 BF16 V 和 BF16 RoPE。
2. **QkKernel / Cube**：NoPE 使用 `mx_fp8_e4m3_t` 的 `LoadData + Mmad`，FP32 L0C 累加；BF16 RoPE 乘积加到同一 FP32 L0C。
3. **SoftmaxKernel / SIMT**：FP32 scale、max、exp、sum、归一化，P 舍入 BF16。
4. **PvKernel / Cube**：BF16 P × BF16 V，沿 selected token 归约，FP32 累加并输出。
5. **OutputKernel / SIMT**：从 FP32 PV 结果直接生成 O8 payload 和 E8M0 scale，不额外插入 BF16 输出舍入。

Q 的 D576 全部以 MXFP8 存储；RoPE 部分解码为 BF16 后计算。K/V NoPE 共用同一份 C4 数据，K RoPE 保留 BF16。**这里没有 FP8 累加，没有 P8，也不假设硬件直接支持 FP8×FP4。** C4 的 D32 scale 不能直接充当 PV token 归约轴的 scale。

首版显式物化 gather、score、P 和转置 V，使用保守的单缓冲 Cube 搬运和同步。额外 GM 读写、kernel 启动以及小 M 利用率可能抵消低精度 QK 的收益；此版的耗时不能代表将来融合版本的性能上限。

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

性能默认测 **完整自定义调用的同步 wall latency**：包括 host dispatch/分配、五个设备 kernel 和同步，排除 CPU 量化/golden、H2D、加载、warmup、输出读回。它不是纯 kernel Task Duration；`--profile` 可以查看五个设备任务，不能只取 QK 时间代表 QSFA。没有通过同环境、同 shape 的对照组前，不报告加速比。

可用 `--input /path/inputs.pt` 重放 `torch.save` 的 `{query:[1,H,576], kv:[K,576], indices:[1,selected], scale_value:float}`，张量在 CPU。输入先舍入 BF16；无自动模型抓取。单次 decode 的 Q=1 与生成1024个输出 token 是两个概念。

## 源码依据和验证边界

- CANN skill：`cann/cannbot-skills@3074681ba927916f99f5a9ac808b4e6797934ce5` 的 `ascendc-direct-invoke-template`、`torch-ascendc-op-extension/routes/direct-invoke.md`、`ascendc-simt-best-practices`。采用 ASC CMake 或原生 bisheng 编译，加直接 launch；没有自制 ACL ctypes binding。
- [QLI V2 Cube reference](https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h)：`LoadQScaleToL1`、`LoadKScaleToL1`、`LoadQueryToL0a`、`LoadKeyToL0b`、`ComputeL0c` 提供 MX L0 类型、scale pair 的 Dn2Nz 和 NT LoadData 合同。
- [QSFA 官方 golden](https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/tests/pytest/kv_quant_sparse_flash_attention_golden.py)：复用此前固定的 `gatherKV/softmax/_t_increattention_bnsd`，见相邻 `qsfa_fake_quant/vendor/SOURCES.json`。
- [QuantMatmulWeightNz](https://gitcode.com/cann/ops-nn/blob/2a77283db46e6648ff47bc8277442cf9c721e3c2/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md)：MXFP8×MXFP8、FP32输出合同；同ref依赖的 `ops-tensor@781745c86312b478009742851b6cfc1967da8dda` 中 `KernelMatmulMixWeightPrologue::CopyConvertStoreWeight` 用 `ShiftW4ToW8`，配合 `BlockMmadWeightPrologueMx::CopyCL0c2Gm` 的×64补偿。本原型直接使用真实 E2M1→E4M3 数值映射，不照搬该位移技巧。
- [SIMT 执行空间规则](https://asc.gitcode.com/api/SIMT-API/SIMD_SIMT_hybrid_programming_intro/extended_syntax/function_execution_space_qualifier.html)：VF 的 codec helper 使用 `__simt_callee__`。

本机完成 CPU payload/golden/失败门槛测试、实际 `codec.h` 的宿主 C++ 编译及全部 FP8码/舍入边界检查、支持shape的分块地址核对、源代码同步检查。同步脚本最初无法解析泛型 `HardEvent event`，报两条未知方向候选；改为六个显式 Set/Wait 方向函数后候选为0。flow分析仅覆盖两处 DataCopy，跨函数的 Mmad/LoadData/Fixpipe 生命周期另行人工核对，静态检查不证明设备执行正确。

2026-09-23 本地验证：相关4个测试文件共32项及23个子测试通过，Ruff check/format、`bash -n`通过。完整仓库 `bash format.sh ci` 因本地未安装 pre-commit 而未执行，不额外安装依赖。

2026-09-24 构建入口修复：相关5个测试文件共43项及31个子测试通过，新增多工具链目录、符号链接环、编译检查失败后回退与实际内核编译失败即停止的本地回归；构建命令用 mock 验证流程。Ruff、shell语法检查通过；本机没有 CANN 编译器/A5，尚不能宣称实际 CANN 编译通过。仓库完整格式检查仍因缺少 pre-commit 未执行。

**待 A5 验证**：CANN9.1实际编译兼容、MX NoPE→BF16 RoPE连续累加、完整数值结果、设备任务和 wall 性能。尚未支持 torch.compile/ACLGraph、prefill、多batch、框架接入或端到端模型验证。包含 CANN 派生代码的 `csrc/matmul.asc` 保留 CANN2.0许可，见 `CANN_LICENSE`。
