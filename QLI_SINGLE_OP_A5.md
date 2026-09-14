# A5 Indexer C4 单算子测评：复用容器依赖

这个入口只测 `QuantLightningIndexerV2`：C4 为 `quant_mode=5`、FP4 E2M1 Q/K + K32 E8M0 scale；FP8 对照为 `quant_mode=1`。不需要模型权重，不安装 vLLM 或 vLLM-Ascend，不创建或拉取容器，不运行 pip。

## 更新旧的 test_qli.sh 并执行

在现有 A5 容器中运行：

```bash
curl -fsSL https://raw.githubusercontent.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/main/tools/test_qli_v2_mxfp4_a5.sh -o test_qli.sh && bash test_qli.sh --update
```

这里只下载测试代码。原来默认的 `/workspace/vllm-ascend-glm-mxfp4-optest` checkout 会继续复用，`--update` 做 fast-forward 更新；有本地修改则停止，不搬走或覆盖目录。可以用 `--workdir PATH` 指定另一处测试 checkout。

如果已经拉取完整仓库，可以完全不联网地启动：

```bash
bash tools/test_qli_v2_mxfp4_a5.sh
```

所有启动方式复用当前容器的 Python、torch、torch_npu 和 CANN。可用 `--python /path/to/python` 或 `--cann-root /path/to/cann` 明确选择已有环境。

## 容器不一定已经有 C4 算子

脚本先解析容器真实 ACL dtype 定义、检查 API 来源，再执行小规模 **mode5 compute**。Metadata 能运行或函数名存在，都不算 C4 支持通过。

- 已有算子通过探针：直接做精度与性能测评。
- 缺少 QLI V2 API：尝试只编译本仓 QLI V2 compute 与 Metadata。若 C4 在准备执行时被拒绝，但同形状 FP8 compute 成功，也会保留错误证据后尝试一次补装；这只是修复尝试，不把错误号当成确定的能力证明。
- kernel 执行、同步或数值检查错误：报告原始失败，不自动降为 FP8 或用伪量化替代。
- 容器缺少所需 torch_npu 接口、FP4 类型定义或 CANN 开发工具链：列出缺项，停止，不下载或升级依赖。

选择性编译只使用现有 CANN 工具链和已安装的头文件/库。QLI 的共享 `lightning_indexer_v2` 源码依赖由原构建系统处理；不构建整个 VA 的 Python/C++ 扩展，不拉取 protobuf、absl、JSON 或 makeself。JSON 头文件若位于自定义路径，可加 `--json-include /existing/include`；编译并行度可加 `--jobs 8`。

算子安装在仓库内 `.qli-op-build/`，不覆盖系统 CANN。成功产物记录在 `.qli-op-build/install.json`，后续按源码、CANN、SoC 指纹检查后复用。日志记录实际加载的 compute/metadata DSO，显式选择自定义库时不会静默使用旧库里的同名函数。安装成功仍须通过真实 C4 探针。

可选命令：

```bash
# 只检查容器已配置的算子，不编译
bash tools/test_qli_v2_mxfp4_a5.sh --installed-only --check-only

# 明确用仓库内 C4 源码重新编译，然后自动测评
bash tools/test_qli_v2_mxfp4_a5.sh --build-op

# 使用自己已经安装的算子包；请先设置它的 ASCEND_CUSTOM_OPP_PATH
bash tools/test_qli_v2_mxfp4_a5.sh --opapi-lib /existing/vendor/op_api/lib/libcust_opapi.so

# 改变测试规模
bash tools/test_qli_v2_mxfp4_a5.sh --query-tokens 4 --key-tokens 16384 --iterations 50
```

`--soc` 接受 Ascend950 的型号字符串，选择性 CANN 构建使用 `ascend950` 架构族；设备入口仍检查实际 A5 硬件。A3 不使用这条真实 MXFP4 路径。

## 测评内容与结果

默认 T=1、64 个 Q heads、D128、K=8192、TopK=2048。C4/FP8 使用相同原始输入、权重及 shape：

- 对实际 FP4/FP8 payload 和 scale 独立反量化，检查算子 score 与 TopK 选择，分离实现错误与量化损失。
- 另报告相对原 FP16 输入的 FP32 参考的 TopK recall、score cosine/NMAE；显式拒绝 NaN/Inf。当前阈值为合成用例门槛，不是 GLM 模型精度验收。
- 比较连续 K cache 与带块间隔、非零 storage offset 的 cache，要求输出一致；也核对仅返回 indices 与同时返回 values 两条路径。
- 记录 C4/FP8 的同步调用 P50/P90/mean 与比值。**这是包含 Python、ACLNN 准备、workspace、dispatch 和同步的 wall latency**，不称为纯 kernel 时间；不包含输入量化和 Metadata，也不代表模型吞吐。

结果默认保存为 `qli_a5_results.json`；控制台日志为 `qli_a5_*.log`；探针为 `.qli-op-build/probe-installed.json` / `probe-custom.json`。出错会保留 JSON 中的环境、API/库路径与错误；返回非零退出码。可用 `--output PATH` 指定完整测评结果位置。

## 这次修复的依据

旧版本 `edb145c2f` 的 `tools/test_qli_v2_mxfp4_a5.sh` 每次搬走目录并重新 clone，再执行 `pip install --no-deps -e`。`--no-deps` 没有关闭隔离构建；`pyproject.toml::build-system.requires` 包含 torch、torch_npu 等，因此仍下载构建依赖。参见 [pip 官方构建隔离说明](https://pip.pypa.io/en/stable/reference/build-system/#disabling-build-isolation)。新测试完全移除了这条安装路径。

同版本 Python 入口直接运行在 `tools/` 下，仓库 `tools/bisect` 遮蔽标准库 bisect，torch 导入链会失败。新入口消除该路径遮蔽，并移除所有 VA import。

接口依据为 CANN `ops-transformer@632dddba712a4e6cace3f8b44f198aef8a82ce3e` 的 `attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md` 和 Metadata 公开头文件。新 `tools/qli_container_backend.py::QLIBackend` 直接调用容器里的两段式 ACLNN API，FP4 的 shape、stride、offset 和 storage size 都以逻辑 nibble 单位描述。

**本地验证范围：50 项 CPU 契约/错误分流/脚本启动/构建调度测试通过，改动 Python 文件的 ruff 与两个 shell 入口的语法检查通过。未在 CANN 或 A5 上编译运行。** 仓库完整 `bash format.sh ci` 已尝试，但本机缺少 `pre-commit`，未通过该完整检查。依赖复用问题和可复现的脚本启动错误已经修复，设备上的 API/SDK 兼容性、算子精度与性能仍需上述命令实测。完整模型安装脚本是另一条部署流程，此单算子入口不会调用它。
