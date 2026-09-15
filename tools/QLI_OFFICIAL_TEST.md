# A5：运行 CANN 官方 QLI V2 MXFP4 精度和性能测试

已有私有 QLI 算子包（`.qli-op-build/install.json`）的容器中执行：

```bash
git -C /workspace/vllm-ascend-glm-mxfp4-optest pull --ff-only && python3 /workspace/vllm-ascend-glm-mxfp4-optest/tools/run_qli_official_a5.py
```

入口直接运行官方 pytest、输入生成、golden 和比较逻辑，通过官方
`cann_ops_transformer` C++ bridge 调用已经安装的私有算子。
不运行自写 benchmark/smoke，不重新编译 NPU kernel。
首次运行会用容器现有 C++ 编译器和 ninja 编译一个官方 host bridge，后续复用缓存。
入口在子进程启动前设置 `FLA_NPU_DISABLE_PTH=1`，关闭 fla_npu 的 Python `.pth`
路径注入钩子，并关闭 PyTorch 第三方后端自动导入。显式导入容器的 torch_npu，随后恢复私有
`ASCEND_CUSTOM_OPP_PATH`，避免第三方包在导入时把自己的 vendor 插到测试算子之前。
这些设置仅作用于测试子进程。桥接解析接口时打印 `QLIV2_API_LIBRARY`，记录实际
函数所属的库；启动时打印的 manifest 路径本身不能证明最终命中的库。
需要容器已有 torch、torch_npu、numpy、pandas、pytest、编译器及 CANN 开发头文件；
缺失时停止并报告，不执行 pip、不下载依赖、不操作镜像。

默认顺序运行两个官方 mode 5 用例，首个失败即停止：

| 用例 | 主要覆盖 |
| --- | --- |
| `MXFP4_PA_20` | TND + 分页 K，Q=1、K=128、H=1、D=128、TopK=64 |
| `MXFP4_META_70_002` | BSND，Q=4、K=641、H=64、D=128、TopK=64，检查 indices 和 scores |

不加 `--perf` 时仅检查官方精度，子进程启用详细 plog 和同步执行。
2026-09-15 用户 A5 实测：`MXFP4_PA_20`、`FP8_PA_04` 精度和各 20 次性能采样通过，
设备任务 P50 分别为 8.377 μs、9.257 μs。两组 META 的完整性能尚未核实。
旧大 shape 也已完成计算与计时，但 C4 量化损失门槛未通过；不能记录为整体 Pass。
详细错误复盘、结果口径和遗留问题见 [A5 验证记录](QLI_A5_VALIDATION_NOTES.md)。
完整 56K prompt、每块 8192 个 Q token 的目标负载测试见 [chunked prefill 用法](QLI_CHUNKED_PREFILL.md)。

## 同时测精度和性能

```bash
git -C /workspace/vllm-ascend-glm-mxfp4-optest pull --ff-only && python3 /workspace/vllm-ascend-glm-mxfp4-optest/tools/run_qli_official_a5.py --perf
```

默认增加相同 shape 的 `FP8_PA_04` 和 `FP8_META_70_002`，共四个官方用例。
按指定顺序执行，默认先 C4 后 FP8；每个用例启动独立 Python 进程，共用已有 JIT 磁盘缓存。
首个失败即停止，避免前一个用例的进程内 profiler 状态影响后续用例。
每个用例先执行完整官方精度检查，通过后复用该次实际输入、metadata 和原始 C++ 调用，
预热 5 次，再用容器已有的 `torch_npu.profiler` 采集 20 次计算。
profiler 导出使用容器已有的 CANN `msprof`；入口会查找现有安装路径，不会下载或安装。
用 `--warmup N --iters N` 调整次数，用 `--cases` 选择官方 STC 用例。
官方用例、golden、比较阈值和 C++ 调用入参未因性能功能而修改。

终端打印每个用例的 p50、p90、平均耗时（微秒），以及两组 FP8/C4 的 p50 比值。
比值大于 1 表示该 shape 下 C4 的设备任务耗时更短。
完整数据保存在本次目录的 `performance.json`；各用例的日志、报告、原始 CSV
分别位于 `01/`、`02/` 等子目录，CSV 的完整路径记录在报告中。

统计值取自 CANN `op_summary*.csv` 的 QLI compute `Task Duration(us)`。
这是 profiler 下的设备任务耗时，包含设备任务调度、执行和完成响应，不能称为纯指令执行时间。
输入生成、H2D、量化、metadata、首次 JIT、精度比较、预热及 CPU 提交间隔不计入这个字段。
性能子进程保留已通过精度检查时的 `ASCEND_LAUNCH_BLOCKING=1`，降低调试日志级别。
因此结果是同步启动条件下的设备任务耗时，不代表生产异步吞吐；退出不改变容器原有环境。
采样为空、异常或数量不符时，保留 trace 并报错，不输出伪造的性能值。
每例报告区分 `accuracy_failed` 和 `performance_failed`，并在 QLI 调用前记录
`compute_arguments`：实际 tensor shape、dtype、stride、offset 和标量参数。
该记录不读取 tensor 数值、不增加 NPU 同步；它不包含可用于逐字节复现的 Q/K payload。

两种量化模式使用相同 shape、各自的官方合成数据。这不衡量 C4 相对原始高精度输入的量化损失，
也不验证 GLM 端到端精度；上述小 shape 的性能不能直接代表 GLM 生产负载。

## 输出和诊断

每次输出保存在 `.qli-official/<时间>/`：
精度模式的日志在该目录；性能模式每例的以下日志位于 `01/`、`02/` 等子目录。

- `preflight.log`：容器依赖检查。
- `official.log`：官方 pytest 的完整输出。
- `plog/`：本次进程的 CANN 日志。
- `run.json`：用例、进程 PID 和指定的私有算子库。
- `diagnostic.json`：失败时提取的 kernel 信息、错误日志和故障 PC 附近反汇编（工具可用时）。

比较通过与失败的运行时，应核对 `QLIV2_API_LIBRARY`、完整用例参数及 plog 的
kernel 名称、tiling key。相同 dtype 或同名算子不等于同一 kernel specialization。
诊断中的私有包反汇编候选也不能单独证明该二进制就是实际执行的 kernel。

失败后提供本次打印的 `official.log` 和 `diagnostic.json` 即可；完整 plog 仍在上述目录。
单独查看已有 benchmark 超时的日志，无需再次执行算子：

```bash
python3 /workspace/vllm-ascend-glm-mxfp4-optest/tools/collect_qli_timeout.py
```

## 来源

代码固定来自 [CANN ops-transformer，55498d91634277d4eec912499c027818a8c167fb](https://gitcode.com/cann/ops-transformer/tree/55498d91634277d4eec912499c027818a8c167fb)。
`tools/vendor/cann_qli_v2/TEST_SOURCES.json` 和 `BRIDGE_SOURCE_MANIFEST.json`
记录上游文件路径与 SHA256，原始 CANN 许可证保存在同目录。
测试文件唯一兼容性修改是删除 golden 中未使用的 `import test`，
避免要求容器安装 CPython 自身的测试包；用例、golden 和精度阈值均未修改。
最小 Python 包入口只导入 QLI，复用原始 builder 和 C++ bridge。
公共头文件只增加实际 QLI API 归属库的 `dladdr` 日志，不改变解析顺序或算子行为；
`BRIDGE_SOURCE_MANIFEST.json` 保留原始和修改后的 SHA256。更新此头文件后，ninja
可能重新编译 host bridge，仍不重新编译 NPU kernel，也不下载依赖。
