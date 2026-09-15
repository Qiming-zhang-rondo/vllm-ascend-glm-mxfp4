# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "tools/qli_official_perf_summary.py"
SPEC = importlib.util.spec_from_file_location("qli_official_perf_summary", SOURCE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class OfficialPerfSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.trace = Path(temporary.name).resolve()

    def write_csv(self, name, columns, rows):
        path = self.trace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_type_takes_priority_and_multiple_csvs_are_summarized(self):
        columns = ["OP Type", "Op Name", "Task Duration(us)"]
        first = self.write_csv(
            "PROF_1/mindstudio_profiler_output/op_summary_1.csv",
            columns,
            [
                ["QuantLightningIndexerV2", "kernel_1", "10"],
                ["QuantLightningIndexerV2Metadata", "QuantLightningIndexerV2", "invalid-but-not-compute"],
                ["Memcpy", "QuantLightningIndexerV2_copy", "999"],
                ["QuantLightningIndexerV2", "kernel_2", "20"],
            ],
        )
        second = self.write_csv(
            "PROF_1/mindstudio_profiler_output/op_summary_2.csv",
            columns,
            [["QuantLightningIndexerV2", "kernel_3", "30"], ["QuantLightningIndexerV2", "kernel_4", "40"]],
        )
        result = summary.summarize_case(self.trace, 4)
        self.assertEqual(result["samples_us"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(result["p50_us"], 25.0)
        self.assertEqual(result["p90_us"], 37.0)
        self.assertEqual(result["mean_us"], 25.0)
        self.assertEqual((result["min_us"], result["max_us"]), (10.0, 40.0))
        self.assertEqual((result["sample_count"], result["expected_calls"]), (4, 4))
        self.assertEqual(result["files"], [str(first), str(second)])
        self.assertIn("compute", result["scope"])

    def test_name_fallback_normalizes_prefix_without_including_metadata_or_copy(self):
        self.write_csv(
            "op_summary.csv",
            ["Op Name", "Task Duration(us)"],
            [
                ["QuantLightningIndexerV2", "1"],
                ["quant_lightning_indexer_v2_abc_high_performance_1", "2"],
                ["QuantLightningIndexerV2Metadata", "100"],
                ["quant_lightning_indexer_v2_metadata_abc", "100"],
                ["QuantLightningIndexerV2_copy", "100"],
                ["MemcpyAsync", "100"],
                ["QuantLightningIndexer", "100"],
            ],
        )
        self.assertEqual(summary.summarize_case(self.trace, 2)["samples_us"], [1.0, 2.0])

    def test_invalid_compute_durations_fail_and_keep_trace(self):
        for value in ("nan", "inf", "-inf", "-1", "0", "", "not-a-number"):
            with self.subTest(duration=value):
                path = self.write_csv(
                    "op_summary.csv", ["Op Name", "Task Duration(us)"], [["QuantLightningIndexerV2", value]]
                )
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "Invalid QLI compute Task Duration"):
                    summary.summarize_case(self.trace, 1)
                self.assertEqual(path.read_bytes(), before)

    def test_missing_data_schema_and_sample_count_are_explicit_failures(self):
        with self.assertRaisesRegex(ValueError, r"No op_summary"):
            summary.summarize_case(self.trace, 1)
        path = self.write_csv("op_summary.csv", ["Op Name"], [["QuantLightningIndexerV2"]])
        with self.assertRaisesRegex(ValueError, "Task Duration"):
            summary.summarize_case(self.trace, 1)
        self.write_csv("op_summary.csv", ["Op Name", "Task Duration(us)"], [["QuantLightningIndexerV2Metadata", "10"]])
        with self.assertRaisesRegex(ValueError, "Expected 1 .* found 0"):
            summary.summarize_case(self.trace, 1)
        self.write_csv("op_summary.csv", ["Op Name", "Task Duration(us)"], [["QuantLightningIndexerV2", "10"]])
        with self.assertRaisesRegex(ValueError, "Expected 2 .* found 1"):
            summary.summarize_case(self.trace, 2)
        self.assertTrue(path.is_file())

    def test_single_sample_and_invalid_expected_calls(self):
        self.write_csv("op_summary.csv", ["Op Name", "Task Duration(us)"], [["QuantLightningIndexerV2", "2.5"]])
        result = summary.summarize_case(self.trace, 1)
        self.assertEqual((result["p50_us"], result["p90_us"], result["mean_us"]), (2.5, 2.5, 2.5))
        for count in (0, -1, 1.5, True):
            with self.subTest(expected_calls=count), self.assertRaisesRegex(ValueError, "positive integer"):
                summary.summarize_case(self.trace, count)


if __name__ == "__main__":
    unittest.main()
