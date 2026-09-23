"""Selection, failure propagation, and missing-data behavior of dataset tests."""
import contextlib
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as runner


def expectations():
    return {
        "format_version": 1,
        "tables": {name: {"rows": 0, "logical_sha256": "0" * 64}
                   for name in ("subjects", "items", "benchmarks", "responses")},
        "source_claims": {"source_responses": {
            "kind": "count", "origin": "derived_from_data",
            "source": "https://example.org/results.json", "locator": "records",
            "scope": "All released attempts", "expected": 2,
        }},
    }


class DatasetRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "benchmarks/example"
        self.directory.mkdir(parents=True)
        (self.directory / "characterization.yaml").write_text(yaml.safe_dump(expectations()))
        (self.root / "tests/benchmarks").mkdir(parents=True)

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = runner.main(self.root, argv=list(args))
        return status, output.getvalue()

    def test_discovery_excludes_templates_and_unreviewed_folders(self):
        template = self.root / "benchmarks/_template"
        template.mkdir()
        (template / "characterization.yaml").touch()
        (self.root / "benchmarks/unreviewed").mkdir()
        self.assertEqual(runner.benchmark_names(self.root), ["example"])
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            runner.dataset_suite(self.root, ["unreviewed"])
        self.assertEqual(self.run_cli("--list"), (0, "example\n"))

    def test_cli_requires_data_unless_explicitly_allowed_and_restores_environment(self):
        before = (self.directory / "characterization.yaml").read_bytes()
        with patch.dict(os.environ, {"MEASUREMENT_DB_FULL_TEST": "previous"}):
            status, output = self.run_cli("example")
            self.assertEqual(status, 1)
            self.assertIn("generated benchmark table(s) are absent", output)
            self.assertEqual(os.environ["MEASUREMENT_DB_FULL_TEST"], "previous")
            status, output = self.run_cli("example", "--allow-missing")
            self.assertEqual(status, 0)
            self.assertIn("skipped", output)
        self.assertEqual((self.directory / "characterization.yaml").read_bytes(), before)
        self.assertFalse((self.directory / "formatted_tables").exists())

    def test_source_specific_failures_are_preserved(self):
        (self.root / "tests/benchmarks/test_example.py").write_text(
            "import unittest\n"
            "class SourceTests(unittest.TestCase):\n"
            "    def test_source_mapping(self):\n"
            "        self.assertEqual('wrong subject', 'upstream subject')\n")
        status, output = self.run_cli("example", "--allow-missing")
        self.assertEqual(status, 1)
        self.assertIn("test_source_mapping", output)
        self.assertIn("upstream subject", output)

    def test_source_import_error_does_not_hide_other_benchmarks(self):
        (self.root / "tests/benchmarks/test_example.py").write_text("raise ValueError('broken source check')\n")
        other = self.root / "benchmarks/other"
        other.mkdir()
        (self.root / "tests/benchmarks/test_other.py").write_text(
            "import unittest\n"
            "class OtherTests(unittest.TestCase):\n"
            "    def test_other_source(self): pass\n")
        status, output = self.run_cli("--all", "--allow-missing")
        self.assertEqual(status, 1)
        self.assertIn("broken source check", output)
        self.assertIn("test_other_source", output)

    def test_shared_source_claims_compare_independent_observations(self):
        def audit(directory):
            self.assertEqual(directory, self.directory)
            return {"source_responses": 2}
        suite = runner.benchmark_suite(self.root, "example", source_audit=audit)
        tests = list(suite)
        self.assertEqual(sum(t._testMethodName == "test_complete_dataset" for t in tests), 1)
        case = next(t for t in tests if t._testMethodName == "test_independent_provider_claims")
        case.expected = expectations()
        case.tables = {"responses": pd.DataFrame({"response": [0.0, 1.0]})}
        case.test_independent_provider_claims()
        case.tables = {"responses": pd.DataFrame({"response": [0.0]})}
        with self.assertRaises(AssertionError):
            case.test_independent_provider_claims()

    def test_inherited_shared_checks_are_not_duplicated(self):
        (self.root / "tests/benchmarks/test_example.py").write_text(
            "from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared\n"
            "class SourceTests(shared.DatasetTests):\n"
            "    def test_source(self): pass\n")
        tests = list(runner.benchmark_suite(self.root, "example"))
        self.assertEqual(sum(t._testMethodName == "test_complete_dataset" for t in tests), 1)
        self.assertEqual(sum(t._testMethodName == "test_source" for t in tests), 1)

    def test_characterization_alone_does_not_count_as_a_source_audit(self):
        suite = runner.benchmark_suite(self.root, "example")
        case = next(t for t in suite if t._testMethodName == "test_independent_provider_claims")
        with self.assertRaisesRegex(AssertionError, "independent source checks"):
            case.test_independent_provider_claims()


if __name__ == "__main__":
    unittest.main()
