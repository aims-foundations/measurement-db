"""Fixed grading descriptions are optional, validated metadata."""

import copy
from datetime import date
from pathlib import Path
import unittest

from scripts.build_measurement_tables.validate_benchmark_metadata import (
    BenchmarkMetadataError,
    load_benchmark_metadata,
    validate_benchmark_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class GradingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = load_benchmark_metadata(ROOT / "benchmarks/real_webagents/metadata.yaml")

    def test_grading_is_optional_for_existing_builders(self):
        del self.metadata["grading"]
        validate_benchmark_metadata(self.metadata)

    def test_common_rule_does_not_require_a_static_verifier_description(self):
        self.metadata["grading"] = {"rule": "All released tests must pass."}
        validate_benchmark_metadata(self.metadata)
        self.metadata["grading"]["rule"] = " "
        with self.assertRaises(BenchmarkMetadataError):
            validate_benchmark_metadata(self.metadata)

    def test_invalid_grading_structure_is_rejected(self):
        for grading in (
            None,
            {},
            {"verifiers": {}},
            {"verifiers": {"checker": "Use a structured description"}},
            {"verifiers": {"checker": {}}},
            {"verifiers": {" ": {"logic": "exact match"}}},
            {"verifiers": {"checker": {"logic": "exact match"}}, "fallback_rule": " "},
            {"verifiers": {"checker": {"logic": "exact match"}}, "fallbak_rule": "typo"},
        ):
            with self.subTest(grading=grading):
                metadata = copy.deepcopy(self.metadata)
                metadata["grading"] = grading
                with self.assertRaises(BenchmarkMetadataError):
                    validate_benchmark_metadata(metadata)

    def test_descriptions_must_serialize_to_json(self):
        for value in (date(2026, 9, 21), float("nan"), float("inf"), {"not JSON"}):
            with self.subTest(value=value):
                self.metadata["grading"]["verifiers"] = {"checker": {"value": value}}
                with self.assertRaisesRegex(BenchmarkMetadataError, "JSON-compatible"):
                    validate_benchmark_metadata(self.metadata)


if __name__ == "__main__":
    unittest.main()
