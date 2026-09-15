# 56K prompt 的 QLI chunked prefill 验证

这里的 56K 是完整 prompt 的 token 数：`57344 = 56 × 1024`。
每块处理 `8192` 个 Q token，共 7 块，Q heads=64、K heads=1、D=128、TopK=2048。
每块独立调用 QLI；K cache 只包含截至当前块的前缀，块内保留因果 mask。

在已有 A5 容器、已构建私有算子和旧 `test_qli.sh` 的环境中，一条命令：

```bash
FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 bash test_qli.sh --update --prefill-tokens 57344 --chunk-size 8192 --reference-rows 16 --output qli_prefill_56k_chunk8192.json
```

复用既有 Python/CANN 和已通过单算子验证的调用 adapter，不下载 Python 依赖或镜像。
`--chunk-size` 控制每次计算的 Q 数，`--prefill-tokens` 控制总 prompt 长度；
这两个长度均须按 128-token cache block 对齐。总长度不能整除 chunk 时，最后一块取剩余长度。

| 块 | 全局 Q token 范围（含首尾） | Q 数 | 可见 K 前缀长度 |
| --- | --- | ---: | ---: |
| 1 | 0–8191 | 8192 | 8192 |
| 2 | 8192–16383 | 8192 | 16384 |
| 3 | 16384–24575 | 8192 | 24576 |
| 4 | 24576–32767 | 8192 | 32768 |
| 5 | 32768–40959 | 8192 | 40960 |
| 6 | 40960–49151 | 8192 | 49152 |
| 7 | 49152–57343 | 8192 | 57344 |

输入 layout 为 TND + PA_BBND，mask_mode=3、cmp_ratio=1。
当前块 `cu_seqlens_q=[0,Q_chunk]`，`seqused_k=[prefix_end]`；
局部行 i 的有效 K 数是 `prefix_end - Q_chunk + i + 1`，而不是整个最终 56K。
第一块前 2047 行的有效 K 数不足 TopK，剩余输出应为 index=-1、value=-inf。
不能为了满足旧脚本的 TopK 约束额外增加真实 K，这会改变 prefill 场景。

## 精度和性能分别覆盖什么

- 两种量化模式共享同一份合成 Q/K、weights。每块完整 Q 和 K 前缀都在 A5 上执行。
- 对完整输出检查有效索引数量、重复索引、因果范围、有效 score 的有限性和 padding。
- 数值精度默认每块抽样 16 个 Q 行，覆盖首尾和 TopK 边界；JSON 记录具体全局行号。
  CPU golden 仅计算这些行对应的 score，保留它们的真实因果位置，避免完整 56K×56K 参考矩阵。
- 算子正确性与相对原始输入的误差门槛分别报告。原始输入误差包含量化和计算舍入。
  数值 gate 未通过时保留失败结果并继续收集各块耗时；执行异常或输出结构错误则立即停止。
- 每块每种模式预热 5 次、测量 20 次，沿用现有 Python/ACLNN 同步 wall latency。
  输入生成、量化、H2D、metadata、CPU golden 和精度比较位于计时区间外。
- 总耗时指标是 **7 块 mean wall latency 的求和**，不是完整请求逐次测量的 P50，
  也不包含真实模型计算、KV cache 写入或 prefill 调度开销。报告名称为 `sum_chunk_mean_wall_ms`。

`--reference-rows N` 可调整每块抽样数量（至少 5，最多为该块实际 Q 数）；
`--warmup N --iterations N` 可调整计时次数。
不加 `--prefill-tokens` 时，原单次 decode benchmark 行为保留。

本功能在开发机进行 CPU contract 与执行流程回归；56K/8192 的 A5 实测仍由运行结果确认。
已通过的旧 decode 和官方小 shape 数据见 [验证复盘](QLI_A5_VALIDATION_NOTES.md)，
不能把它们作为本 prefill 场景已通过的证据。
