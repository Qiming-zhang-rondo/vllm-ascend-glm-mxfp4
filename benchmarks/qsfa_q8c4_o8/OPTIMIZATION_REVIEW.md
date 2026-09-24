# A5 QSFA：官方流水与 Q8/C4/O8 原型的性能差距

2026-09-24。结论：有明确的结构性优化空间，首要目标是复用官方按 tile 的 Cube/Vector 流水和片上缓存，而不是继续只调整 dtype 或增加 launch 核数。本文件是源码分析和下一版设计依据，尚未实现融合版本，也没有新的 A5 性能结果。

## 实测基线与证据范围

用户在 Ascend950DT_9582、CANN9.1.1、torch2.10/torch_npu2.10 环境运行同一份输入的两个独立进程：Q=1、H=8、KV=8192、selected=2048、D=576（NoPE512/RoPE64），warmup5/iters20。

| 指标 | 已安装原生 QSFA | 当前 Q8/C4/O8 原型 |
|---|---:|---:|
| 同步 wall p50 | 0.131050ms | 0.452614ms |
| 同步 wall mean | 0.129601ms | 0.450560ms |
| 对各自 decoded-payload golden | 通过，relative RMSE 0.002478 | 通过，误差0 |
| 对原始 BF16 relative RMSE | 0.023485 | 0.115499 |
| 量化筛选 | 通过 | 失败，保留10%门槛 |

原型 p50 为原生的3.4538倍。这是包含调度、分配、全部设备任务及完成同步的调用延迟，不能写成纯 kernel 慢3.4538倍；原型的 status 读回位于计时区间外。两组保留 `ASCEND_LAUNCH_BLOCKING=1`，不通过只给候选改启动环境来制造收益。

原生接口来自容器当前安装的 `torch_npu.npu_kv_quant_sparse_flash_attention`。下面研究的是固定的公开源码，尚未证明容器二进制恰好对应这个提交；具体设备任务、活跃核数和耗时占比仍需 profiler 确认。

- 官方：`cann/ops-transformer@55498d91634277d4eec912499c027818a8c167fb`。
- 原型：`Qiming-zhang-rondo/vllm-ascend-glm-mxfp4@b4d5b77fba862a9287e8837e60115377f076d389`。
- 分析方法：`cannbot-skills@3074681ba927916f99f5a9ac808b4e6797934ce5` 的 `ascendc-perf-optimize`、`ascendc-tiling-design` FlashAttention 路由及量化合同。

## 官方实现为什么值得复用

以下文件位于官方 `attention/kv_quant_sparse_flash_attention/`，链接固定到上述 ref。

| 部分 | 公开 A5 实现 | 原型差异 | 依据 |
|---|---|---|---|
| 执行组织 | 一个 mixed kernel 内按128个 selected tokens 推进，当前 tile 做 Vec0/QK、前一 tile 做 softmax/PV、前两 tile 做输出更新 | gather、QK、softmax、PV、O8 分成5次 launch | [kernel::ProcessMainLoop，565、654–690行][official-kernel]；[prototype forward，105–113行][prototype-binding] |
| Q | 首个 S2 tile 全载 L1，后续 tile 复用 | 每个 QK N tile、每个 K128小块重复调用 Q 拷贝 | [cube::PrepareLeftMatrixBmm1QSFA，292–310行][official-cube]；[AccumulateMx，158–160行][prototype-matmul] |
| KV | 按稀疏地址 GM→UB；反量化成 BF16 后直接 UB→L1，保留到 PV 完成 | 先把整个 selected 集展开成 GM 中的 FP8 K 与转置 BF16 V | [vector::ProcessSparseKv，560–630行][official-vector]；[GatherVf，57–72行][prototype-vector] |
| QK结果 | FP32 L0C 经 Fixpipe 直接进入 UB | FP32 scores 写到 GM 后 softmax 再读 | [cube::IterateBmm1QSFA，260–287行][official-cube]；[CubeBuffers::Store，196–207行][prototype-matmul] |
| softmax | UB 内在线维护 FP32 max/sum/exp，输出 BF16 中间 P 到 L1 | 对整段2048计算，再写 GM 概率矩阵 | [vector::ProcessVec1，635–671行][official-vector]；[SoftmaxVf，99–125行][prototype-vector] |
| P/V复用 | P 写入同一 L1 tile 已使用完的 RoPE 区域；PV读取同一 tile 的 NoPE V，之后才释放 KV | PV读取独立 GM `vt[512,S]` 和 P | [cube::IterateBmm2QSFA，323–333行][official-cube]；[PvKernel，274–290行][prototype-matmul] |
| 输出 | FP32 UB在线重缩放、累计，最后归一化并输出 | 完整 FP32 acc 写 GM 后再启动 O8 kernel | [vector::ProcessVec2/Bmm2DataCopyOut，728–794行][official-vector]；[OutputVf，137–155行][prototype-vector] |
| 流水缓冲 | KV/P L1三缓冲，QK UB双缓冲，PV UB单缓冲，L0A/B/C双缓冲 | 单缓冲，每次搬运、LoadData、Mmad 后保守串行等待 | [kernel，116–132行][official-kernel]；[cube，128–139行][official-cube]；[CubeBuffers，136–224行][prototype-matmul] |

官方不是“完全没有 GM 中间数据”。`IS_SPLIT_G` 时，Vec0的 BF16 KV tile 经三缓冲 GM 共享，再进入 L1；`GetKVPhyAddr` 也会生成 GM 地址表。当前 H8 不切G，按这个 ref 的条件推导会生成16KiB稀疏物理地址表。应区分这类有目的的局部共享与原型完整 K/V/score/P 的多次写回。

最有价值的复用点是：**保留同一 tile 的 KV，让 QK 和 PV 接续消费；score、P、输出累计尽量留在片上。**

## 小 Q 分核：一个容易判断错的地方

不能说官方是靠 split-KV 用满核心才更快：

- [host tiling.h，353、358行][official-tiling-h] 的 `splitKVFlag_` 为 false、`kvSplitPart_` 为1；该路径没有开启 split-KV 的赋值。
- [GenTilingKey，320–331行][official-tiling] 仅在 G>64 时切G。
- [kernel::InitCalcParamsEach，242–272行][official-kernel] 固定 `actBatchS2=1`，按实际 Q 数分核。B1/Q1/G8 在这个 ref 推导主计算使用1个AIC及对应2个AIV，其余主循环退出。
- 原型 H8补成M16，QK为32个 blocks、PV为8个 blocks；PV每个block沿2048个token做16轮K128累加。[原型 launch 与循环][prototype-matmul]

因此官方的关键参考价值是融合与数据复用。对小 Q 加入 split-KV 可以作为后续独立实验，但需要额外的部分输出/max/sum合并，增加同步和改变浮点归约顺序，不能先验断言越多核心越快。

## 原型中可以明确定位的开销

以下都是源码静态计算，不能冒充 profiler 测得的 HBM 流量或瓶颈占比。

1. **缓存节省与中间数据。** 原生 FP8 PA656 为512B NoPE+128B BF16 RoPE+16B FP32 scale；原型 C4 分离缓存为256B NoPE+128B BF16 RoPE+16B E8M0 scale，合计400B/token。selected=2048理想有效cache数据量少0.5MiB。但 `kn` 为1MiB、`vt` 为2MiB，一次写出再读取就有6MiB逻辑搬运，尚未计score/P/其他缓冲。缓存命中、重复加载和内存事务粒度会影响实际总线流量。
2. **转置写入不连续。** `GatherVf` 相邻 lane 沿 d 前进，`vt[d*S+row]` 在 S=2048时，相邻地址间隔4096字节。不能同时把行方向的 K读取和该 V转置存储都当作连续访问。推荐直接复用官方 L1/NZ布局，避免完整 GM 转置；如需保留原型对照，单独采用分块转置进行验证。
3. **单缓冲缺少重叠。** `AccumulateMx/AccumulateBf16/Multiply` 的每轮copy/load/mmad等待保证了正确性，但不具备官方的多槽流水。不能简单删除等待；应按每个buffer的最后消费者配置双缓冲和事件生存期。
4. **多次主机开销。** 每次 forward 有13次临时/输出张量申请、格式检查、stream bookkeeping及5次launch。allocator可复用内存块，但这些API调用仍在wall计时范围。后续预分配工作区或图执行必须单独报告口径；不能仅改变候选计时范围后宣称设备kernel提速。
5. **输出量化的收益很小、开销却单独发生。** H8/D512时BF16输出为8192B，O8 payload+scale为4224B，只少3968B。把O8做成独立kernel未必划算；应合入最后的Vector输出阶段。其对下游消费的潜在收益不在这次单算子测试内。

## 推荐下一版：基于官方流水的独立 Q8/C4/O8 分支

保留当前可运行原型作为回归参照。下一版仍只做单算子，不替换容器默认 QSFA，不修改 VA/AV 或模型。

### 文件与改造点

| 官方文件/函数 | 需要改造的内容 |
|---|---|
| `...service_vector_mla_arch35.h::ProcessSparseKv/DequantKv` | 接入 E2M1/E8M0 D32 cache；按tile解码，分别给 NoPE QK准备MXFP8 operand/scale、给PV准备BF16 V；不生成完整GM K/V转置 |
| `...service_cube_mla_arch35.h::PrepareLeftMatrixBmm1QSFA/IterateBmm1QSFA` | Q payload/scale驻留；NoPE使用MXFP8计算，RoPE使用BF16，保持同一FP32累加语义 |
| `...service_cube_mla_arch35.h::IterateBmm2QSFA` | 保留BF16 P/V和FP32累加，复用片上V；先不引入P8和额外重复量化 |
| `...service_vector_mla_arch35.h::ProcessVec1/ProcessVec2/Bmm2DataCopyOut` | 保留在线softmax和FP32输出累加；只在最后除sum之后产生O8 payload+scale，替代独立OutputKernel |
| `...kernel_mla_arch35.h::InitMMResBuf/ProcessMainLoop` | 重新规划多槽buffer、scale布局、QK/PV不同dtype的生存期及跨核生产/消费握手 |
| host/binding及tiling | 使用独立入口，显式携带Q/K/O scale和cache布局；不能让现有656B、D128 FP32 scale的consumer读400B C4布局。若采用注册算子，还需更新op_host定义、推导和op_api参数校验；独立direct-invoke则在自身binding实施相应合同 |

### 三个必须先解决的约束

**类型不能整体替换。** 官方 `Q_T` 同时控制Q、反量化KV、softmax P和两个Matmul。全局改成FP8会同时改变RoPE、P、PV。应拆开Q输入、QK operand、PV operand及FP32 accumulator类型，保持阶段之间的明确合同。

**L1预算需要重算。** 现有官方分配为三份 `128*576*2` KV槽和一份 `64*576*2` Q，合计516096B；512KiB仅剩8192B。直接再放MX K及BF16 V不可行。可评估两种方案：QK消费完成后复用槽位，或缩小S2 tile/首版M容量。

一个待验证的首版Q1/H8方案是物理M16、S2 tile64：每槽保守同时计入BF16 V=65536B、FP8 K=32768B、Kscale=1024B、RoPE=8192B、BF16 P=2048B，共109568B；三槽328704B，Q/scale/Qrope10496B，合计339200B。这仅证明这些L1数据项在容量上有余量，不证明完整buffer布局、UB/L0用量、API对齐和流水性能已验证。tile64增加循环数，必须与128方案实测比较。

**C4 scale不能直接拿来做低精度PV。** C4 scale是 `scale[token,d/32]`；PV沿token求和，scale随归约轴变化，不能移到求和之外，也不能当成沿token每32项的MX scale。逐tile解码为BF16再PV，可保持现有数值含义，不必为了C4 cache引入P8。

### 精度合同保持

- 目标仍是Q8/C4/O8；中间BF16 P/PV和FP32累加是当前方案的一部分。
- 当前原型Q的576维全部先量化，Q RoPE再解码为BF16。融合优化应保持同一份decoded Q，不能偷偷换成原始BF16 Q RoPE后声称仅改了性能。
- 在线softmax与当前整段softmax的舍入顺序不同，需要重新对decoded-input golden做数值检查；不要求bit-exact，不放宽现有门槛。
- O8只在完整输出最终归一化之后量化。不能每个tile先O8再相加，不能把FP32累加改成FP8。
- 原始BF16对照的11.55%量化误差仍然存在；减少启动和访存不会自动解决它。继续保留计算正确性、量化筛选两个独立结果。

## 验证顺序和缺失证据

按skill的四步分析：Step1已经对照shape、tiling和buffer预算；Step2为卡间通信，本单卡算子不适用；Step3从官方cross-core握手源码完成结构审查，但无时间线；Step4目前只有源码候选开销，没有设备流水profiling，因此不标注已确认的memory-bound/Cube-bound/Vector-bound。

下一步先用现有 `run_qsfa.sh --profile` 保留同环境的两组trace：查看候选5个kernel的Task Duration、调用之间的主机空隙、原生kernel组成；需要进一步硬件计数器才能判断真实访存瓶颈。raw launch与torch_npu调度在blocking环境下的行为也应从timeline核对，不能仅由环境变量名推断每次launch是否阻塞。

融合版应依次验证解码/布局、QK与FP32累加、在线softmax/PV、最终O8；继续用同一冻结输入、官方golden和容器原生FP8 QSFA作对照。只有完整计算正确性通过后才报告wall加速比，并另列设备任务时间。首先复测已通过的默认shape，再覆盖H16/32/64、不同selected长度和真实dump输入；未验证的shape直接拒绝而非静默切路径。

本次仅完成源码调研和设计，没有改动可执行算子，因此无需用户重新编译来查看结论。

[official-kernel]: https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla_arch35.h
[official-cube]: https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_service_cube_mla_arch35.h
[official-vector]: https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_service_vector_mla_arch35.h
[official-tiling]: https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp
[official-tiling-h]: https://gitcode.com/cann/ops-transformer/blob/55498d91634277d4eec912499c027818a8c167fb/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.h
[prototype-binding]: https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/blob/b4d5b77fba862a9287e8837e60115377f076d389/benchmarks/qsfa_q8c4_o8/csrc/torch_binding.cpp
[prototype-matmul]: https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/blob/b4d5b77fba862a9287e8837e60115377f076d389/benchmarks/qsfa_q8c4_o8/csrc/matmul.asc
[prototype-vector]: https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/blob/b4d5b77fba862a9287e8837e60115377f076d389/benchmarks/qsfa_q8c4_o8/csrc/vector.asc
