"""Tests for the benchmark-saturation analysis."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_measurements.benchmark_saturation import (
    _saturation_scale_bounds,
    analyze_benchmark,
    compute_saturation,
    web_payload,
    write_web_output,
)


class SaturationCalculationTests(unittest.TestCase):
    def test_scale_bounds_preserve_declared_numeric_range(self) -> None:
        self.assertEqual(
            _saturation_scale_bounds("ordinal", "{-2, -1, 0, 1, 2}"),
            (-2.0, 2.0),
        )
        self.assertEqual(_saturation_scale_bounds("binary", "correctness"), (0, 1))

    def test_incoherent_scales_have_no_bounds(self) -> None:
        for response_type in ("continuous_unbounded", "error_presence", "mixed"):
            with self.subTest(response_type=response_type):
                self.assertIsNone(
                    _saturation_scale_bounds(response_type, "{0, 1}")
                )

    def test_best_subject_mean_determines_verdict(self) -> None:
        responses = pd.DataFrame(
            {
                "subject_id": ["a", "a", "b", "b"],
                "response": [0.0, 0.0, 0.8, 1.0],
            }
        )
        self.assertEqual(
            compute_saturation(responses, "item", "binary", "{0, 1}"), 1.0
        )
        responses.loc[3, "response"] = 0.8
        self.assertEqual(
            compute_saturation(responses, "item", "binary", "{0, 1}"), 0.0
        )

    def test_unsupported_analysis_is_nan(self) -> None:
        responses = pd.DataFrame({"subject_id": ["a"], "response": [1]})
        self.assertTrue(
            math.isnan(compute_saturation(responses, "aggregate", "binary", "{0, 1}"))
        )
        self.assertTrue(
            math.isnan(compute_saturation(responses, "item", "mixed", "{0, 1}"))
        )


class SaturationArtifactTests(unittest.TestCase):
    def test_analyze_benchmark_reads_without_mutating_collection_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "example"
            folder.mkdir()
            benchmark_path = folder / "benchmarks.parquet"
            pd.DataFrame(
                [
                    {
                        "benchmark_id": "example-id",
                        "name": "Example",
                        "granularity": "item",
                        "response_type": "binary",
                        "response_scale": "{0, 1}",
                    }
                ]
            ).to_parquet(benchmark_path, index=False)
            pd.DataFrame(
                {"subject_id": ["a", "a"], "response": [1, 1]}
            ).to_parquet(folder / "responses.parquet", index=False)
            before = benchmark_path.read_bytes()

            row = analyze_benchmark(folder)

            self.assertEqual(row["slug"], "example")
            self.assertEqual(row["benchmark_id"], "example-id")
            self.assertEqual(row["saturation"], 1.0)
            self.assertEqual(benchmark_path.read_bytes(), before)

    def test_web_output_is_slug_keyed_strict_json(self) -> None:
        results = pd.DataFrame(
            {
                "slug": ["saturated", "not_saturated", "inconclusive"],
                "saturation": [1.0, 0.0, float("nan")],
            }
        )
        self.assertEqual(
            web_payload(results),
            {"saturated": True, "not_saturated": False, "inconclusive": None},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark-saturation.json"
            path.write_text(
                json.dumps({"untouched": True, "saturated": False}),
                encoding="utf-8",
            )
            write_web_output(results, path)
            self.assertEqual(
                json.loads(path.read_text()),
                {**web_payload(results), "untouched": True},
            )
            self.assertNotIn("NaN", path.read_text())

    def test_web_output_rejects_an_existing_non_object_payload(self) -> None:
        results = pd.DataFrame({"slug": ["example"], "saturation": [1.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark-saturation.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "boolean/null object"):
                write_web_output(results, path)

    def test_web_payload_rejects_a_non_verdict_value(self) -> None:
        results = pd.DataFrame({"slug": ["example"], "saturation": [0.5]})
        with self.assertRaisesRegex(ValueError, "invalid saturation verdict"):
            web_payload(results)


if __name__ == "__main__":
    unittest.main()
