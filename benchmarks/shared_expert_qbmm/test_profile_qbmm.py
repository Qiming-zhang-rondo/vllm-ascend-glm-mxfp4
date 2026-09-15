"""CPU-only trace parser tests. Run: python -m unittest test_profile_qbmm -v."""

import csv
import tempfile
import unittest
from pathlib import Path

from profile_qbmm import ProfileValidationError, profile_call, summarize_trace


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trace = Path(self.temp.name)

    def write_csv(self, rows, *, name="PROF_001/device_0/summary/op_summary_0.csv", headers=None):
        path = self.trace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers or ["Op Name", "Op Type", "Task Duration(us)"])
            writer.writerows(rows)
        return path

    def test_v4_statistics_raw_locations_and_memory_exclusion(self):
        source = self.write_csv([
            ["qbmm0", "QuantBatchMatmulV4", "1"],
            ["copy", "Memcpy", "999"],
            ["qbmm1", "QuantBatchMatmulV4", "3"],
            ["qbmm2", "QuantBatchMatmulV4", "9"],
            ["qbmm3", "QuantBatchMatmulV4", "7"],
        ])
        result = summarize_trace(self.trace, 4)
        self.assertEqual(result["samples_us"], [1, 3, 9, 7])
        self.assertEqual(result["actual_op_types"], ["QuantBatchMatmulV4"])
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["p50_us"], 5)
        self.assertAlmostEqual(result["p90_us"], 8.4)
        self.assertEqual(result["mean_us"], 5)
        self.assertEqual(result["min_us"], 1)
        self.assertEqual(result["max_us"], 9)
        self.assertEqual(result["excluded_memory_op_types"], {"Memcpy": 1})
        self.assertEqual(result["source_paths"], [str(source.resolve())])
        self.assertEqual(result["raw_samples"][1]["csv_row"], 4)
        self.assertEqual(result["raw_samples"][1]["source_path"], str(source.resolve()))

    def test_v3_single_sample_and_header_spelling(self):
        self.write_csv(
            [["QuantBatchMatmulV3", "2.5"]], headers=[" OP TYPE ", "Task Duration (us)"]
        )
        result = summarize_trace(self.trace, 1)
        self.assertEqual(result["actual_op_types"], ["QuantBatchMatmulV3"])
        self.assertEqual(result["p50_us"], 2.5)
        self.assertEqual(result["p90_us"], 2.5)

    def test_multiple_shards_retained_in_file_order(self):
        first = self.write_csv([["x", "QuantBatchMatmulV3", "5"]], name="b/op_summary_b.csv")
        second = self.write_csv([["x", "QuantBatchMatmulV3", "2"]], name="a/op_summary_a.csv")
        result = summarize_trace(self.trace, 2)
        self.assertEqual(result["samples_us"], [2, 5])
        self.assertEqual(result["source_paths"], [str(second.resolve()), str(first.resolve())])

    def test_missing_trace_fails(self):
        with self.assertRaisesRegex(ProfileValidationError, "No CANN op_summary"):
            summarize_trace(self.trace, 1)

    def test_host_duration_is_not_a_device_duration(self):
        self.write_csv([["QuantBatchMatmulV3", "100"]], headers=["Op Type", "Duration(us)"])
        with self.assertRaisesRegex(ProfileValidationError, "requires Op Type"):
            summarize_trace(self.trace, 1)

    def test_op_name_fallback_is_not_permitted(self):
        self.write_csv([["QuantBatchMatmulV3_0", "10"]], headers=["Op Name", "Task Duration(us)"])
        with self.assertRaisesRegex(ProfileValidationError, "Op Name guessing"):
            summarize_trace(self.trace, 1)

    def test_mixed_versions_fail(self):
        self.write_csv([
            ["a", "QuantBatchMatmulV3", "2"], ["b", "QuantBatchMatmulV4", "3"]
        ])
        with self.assertRaisesRegex(ProfileValidationError, "Expected exactly one"):
            summarize_trace(self.trace, 2)

    def test_missing_and_duplicated_tasks_fail(self):
        self.write_csv([["a", "QuantBatchMatmulV3", "2"]])
        with self.assertRaisesRegex(ProfileValidationError, "found 1"):
            summarize_trace(self.trace, 2)
        self.write_csv([["a", "QuantBatchMatmulV3", "2"]], name="copy/op_summary_duplicate.csv")
        with self.assertRaisesRegex(ProfileValidationError, "found 2"):
            summarize_trace(self.trace, 1)

    def test_unknown_split_kernel_fails_without_summing(self):
        source = self.write_csv([
            ["a", "QuantBatchMatmulV4", "2"],
            ["a_reduce", "QuantBatchMatmulV4Reduce", "1"],
        ])
        with self.assertRaises(ProfileValidationError) as caught:
            summarize_trace(self.trace, 1)
        diagnostic = caught.exception.diagnostics
        self.assertEqual(diagnostic["observed_op_types"]["QuantBatchMatmulV4Reduce"], 1)
        self.assertEqual(diagnostic["unrecognized_device_rows"][0]["csv_row"], 3)
        self.assertTrue(source.exists())

    def test_extra_preprocessing_fails(self):
        self.write_csv([["a", "QuantBatchMatmulV4", "2"], ["q", "DynamicMxQuant", "3"]])
        with self.assertRaisesRegex(ProfileValidationError, "extra device operation"):
            summarize_trace(self.trace, 1)

    def test_invalid_durations_fail(self):
        for value in ["", "oops", "NaN", "inf", "-inf", "0", "-0.5"]:
            with self.subTest(value=value):
                self.write_csv([["a", "QuantBatchMatmulV3", value]])
                with self.assertRaisesRegex(ProfileValidationError, "Invalid QBMM Task Duration"):
                    summarize_trace(self.trace, 1)

    def test_invalid_counts_and_warmup_fail_before_npu_import(self):
        for count in [0, -1, 1.1, True, "2", None]:
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "expected_calls"):
                    summarize_trace(self.trace, count)
                with self.assertRaisesRegex(ValueError, "iterations"):
                    profile_call(lambda: None, self.trace, 1, count)
        for warmup in [-1, True, 1.1]:
            with self.subTest(warmup=warmup):
                with self.assertRaisesRegex(ValueError, "warmup"):
                    profile_call(lambda: None, self.trace, warmup, 1)

    def test_existing_trace_is_rejected_before_npu_import(self):
        self.write_csv([["a", "QuantBatchMatmulV3", "2"]])
        with self.assertRaisesRegex(ValueError, "must be new or empty"):
            profile_call(lambda: None, self.trace, 1, 1)


if __name__ == "__main__":
    unittest.main()
