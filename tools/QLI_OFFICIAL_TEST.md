# A5：运行 CANN 官方 QLI V2 MXFP4 精度用例

已有私有 QLI 算子包（`.qli-op-build/install.json`）的容器中执行：

```bash
git -C /workspace/vllm-ascend-glm-mxfp4-optest pull --ff-only && python3 /workspace/vllm-ascend-glm-mxfp4-optest/tools/run_qli_official_a5.py
```

入口直接运行官方 pytest、输入生成、golden 和比较逻辑，通过官方
`cann_ops_transformer` C++ bridge 调用已经安装的私有算子。
不运行自写 benchmark/smoke，不重新编译 NPU kernel。
首次运行会用容器现有 C++ 编译器和 ninja 编译一个官方 host bridge，后续复用缓存。
需要容器已有 torch、torch_npu、numpy、pandas、pytest、编译器及 CANN 开发头文件；
缺失时停止并报告，不执行 pip、不下载依赖、不操作镜像。

默认顺序运行两个官方 mode 5 用例，首个失败即停止：

| 用例 | 主要覆盖 |
| --- | --- |
| `MXFP4_PA_20` | TND + 分页 K，Q=1、K=128、H=1、D=128、TopK=64 |
| `MXFP4_META_70_002` | BSND，Q=4、K=641、H=64、D=128、TopK=64，检查 indices 和 scores |

这是官方精度检查，不提供性能基线，也不验证 GLM 端到端精度。
为定位错误，子进程启用详细 plog 和同步执行；这些设置不影响容器外部的进程。
本地已检查调用流程、用例选择及日志隔离；实际 C++ 编译和算子执行仍需 A5 验证。

每次输出保存在 `.qli-official/<时间>/`：

- `preflight.log`：容器依赖检查。
- `official.log`：官方 pytest 的完整输出。
- `plog/`：本次进程的 CANN 日志。
- `run.json`：用例、进程 PID 和指定的私有算子库。
- `diagnostic.json`：失败时提取的 kernel 信息、错误日志和故障 PC 附近反汇编（工具可用时）。

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
最小 Python 包入口只导入 QLI，复用原始 builder、C++ bridge 和公共头文件。
