# Validation before publication

Date: 2026-09-15. Host: macOS arm64, CPU only. No A5/NPU measurements have been made.

Passed:

- 23 unittest cases, including original CANN golden execution for all three official source cases, both golden paths on identical representable inputs, FP4 nibble/scale round trips, vLLM bridge arguments and layout, profiler CSV validation, child failure logging, and a mocked worker pass/fail run proving accuracy gates profiling.
- `bash -n run_qbmm.sh`.
- Python compile checks for harness and vendor sources.
- `run_qbmm.py --dry-run`: 4 accuracy gates and 48 GLM-shape format cases.
- SHA256 verification of all 11 source/license entries in the vendor manifest.

These checks do not establish that the installed A5 CANN/torch_npu versions accept every interface or that any kernel passes accuracy. The runtime gates perform that validation on the user's A5 environment, retain failures, and prevent reporting performance after a numerical failure.
