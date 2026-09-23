"""Metadata keeps static builder inputs typed and available to the shared loader."""
import copy
from datetime import date
from pathlib import Path
import unittest

from build_base import BenchmarkBuild, BuildContractError
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    BenchmarkMetadataError, load_benchmark_metadata, validate_benchmark_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


class BuildParametersTests(unittest.TestCase):
    def setUp(self):
        self.metadata = load_benchmark_metadata(ROOT / "benchmarks/researchcodebench/metadata.yaml")

    def test_parameters_are_optional(self):
        del self.metadata["build"]["parameters"]
        validate_benchmark_metadata(self.metadata)

    def test_groups_accept_verbatim_strings(self):
        self.metadata["build"]["parameters"] = {
            "prompt": {"prefix": "\n  Indented text\n\n", "suffix": ""},
            "patterns": {"id": r"^task-(\d+)$"},
        }
        validate_benchmark_metadata(self.metadata)

    def test_malformed_parameter_groups_are_rejected(self):
        for parameters in (
            None, {}, [], {" ": {"file": "data.json"}},
            {"archive": {}}, {"archive": "data.json"},
            {"archive": {" ": "data.json"}},
            {"archive": {"file": 1}}, {"archive": {"file": None}},
            {"archive": {"file": ["data.json"]}},
            {"archive": {"file": {"nested": "data.json"}}},
            {"archive": {"file": date(2026, 9, 23)}},
        ):
            with self.subTest(parameters=parameters):
                metadata = copy.deepcopy(self.metadata)
                metadata["build"]["parameters"] = parameters
                with self.assertRaisesRegex(BenchmarkMetadataError, "build.parameters"):
                    validate_benchmark_metadata(metadata)

    def test_loader_exposes_parameters_without_running_transformations(self):
        class Fixture(BenchmarkBuild):
            def build_tables(self):
                raise AssertionError("No transformation should run while loading metadata")

        configured = Fixture(str(ROOT / "benchmarks/researchcodebench/build.py"))
        self.assertEqual(configured.build_parameters, self.metadata["build"]["parameters"])
        unconfigured = Fixture(str(ROOT / "benchmarks/real_webagents/build.py"))
        self.assertEqual(unconfigured.build_parameters, {})

    def test_inline_parameters_cannot_override_metadata(self):
        class Fixture(BenchmarkBuild):
            build_parameters = {"archive": {"file": "different.json"}}

        with self.assertRaisesRegex(BuildContractError, "build_parameters.*metadata.yaml"):
            Fixture(str(ROOT / "benchmarks/researchcodebench/build.py"))


if __name__ == "__main__":
    unittest.main()
