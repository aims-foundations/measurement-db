"""Shared, read-only dataset tests and benchmark selection.

Run through ``python tests/test_benchmark_datasets.py [slug ...]`` in either
repository. Source-specific tests live in tests/benchmarks/test_<slug>.py.
This runner never downloads data, rebuilds tables, or updates expectations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import unittest

import pandas as pd
import pyarrow.parquet as pq

from measurement_db.scripts.build_measurement_tables import (
    validate_dataset, validate_parquet_schema,
)
from measurement_db.scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
)
from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims, check_tables, load_characterization,
)


class DatasetTests(unittest.TestCase):
    """Common checks; the runner supplies the benchmark directory."""

    directory = None
    tables_directory = None

    @classmethod
    def setUpClass(cls):
        directory = cls.directory
        expected_path = directory / "characterization.yaml"
        if not expected_path.is_file():
            raise RuntimeError(f"{expected_path} is absent; review the dataset first")
        cls.expected = load_characterization(expected_path)
        cls.table_dir = cls.tables_directory or directory / "formatted_tables"
        if cls.tables_directory is None and not cls.table_dir.is_dir():
            cls.table_dir = directory  # Flat downloaded snapshots.
        missing = [name for name in cls.expected["tables"]
                   if not (cls.table_dir / f"{name}.parquet").is_file()]
        if missing:
            message = f"{directory.name}: generated benchmark table(s) are absent: {missing}"
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.tables = {p.stem: pd.read_parquet(p)
                      for p in sorted(cls.table_dir.glob("*.parquet"))}

    @classmethod
    def tearDownClass(cls):
        # Do not retain one complete dataset per benchmark while testing a repo.
        if "tables" in cls.__dict__:
            del cls.tables

    def test_metadata(self):
        metadata = load_benchmark_metadata(self.directory / "metadata.yaml")
        self.assertEqual(metadata["build"]["contract_version"], 2)
        self.assertEqual(self.tables["benchmarks"].iloc[0]["name"],
                         metadata["benchmark"]["name"])

    def test_complete_dataset(self):
        validate_dataset(self.tables, expected_benchmark_id=self.directory.name)

    def test_reviewed_measurements(self):
        check_tables(self.expected, self.tables)

    def test_canonical_storage_types(self):
        for name in self.tables:
            validate_parquet_schema(name, pq.read_schema(self.table_dir / f"{name}.parquet"))


def _source_claim_test(self):
    observed = self.source_audit(self.directory)
    check_source_claims(self.expected, observed)
    if "source_responses" in observed:
        self.assertEqual(len(self.tables["responses"]), observed["source_responses"])
    if "source_successes" in observed:
        self.assertEqual(float(self.tables["responses"].response.sum()),
                         observed["source_successes"])


def _missing_source_test(self):
    self.fail(f"{self.directory.name}: implement independent source checks under tests/benchmarks/")


def benchmark_names(root):
    """Only reviewed datasets or explicitly implemented source suites qualify."""
    root = Path(root)
    names = {p.parent.name for p in (root / "benchmarks").glob("*/characterization.yaml")}
    names.update(p.stem.removeprefix("test_")
                 for p in (root / "tests/benchmarks").glob("test_*.py"))
    return sorted(name for name in names if not name.startswith("_"))


def benchmark_suite(root, slug, *, source_audit=None, tables_directory=None):
    """Combine shared validation and the selected benchmark's source checks."""
    root = Path(root).resolve()
    if isinstance(source_audit, Mapping):
        source_audit = source_audit.get(slug)
    directory = root / "benchmarks" / slug
    if slug not in benchmark_names(root) or not directory.is_dir():
        raise ValueError(f"No reviewed dataset or source tests registered for {slug!r}")
    source_path = root / "tests/benchmarks" / f"test_{slug}.py"
    classes = []
    if source_path.is_file():
        spec = importlib.util.spec_from_file_location(f"benchmark_source_{slug}", source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes = [value for value in vars(module).values()
                   if isinstance(value, type) and issubclass(value, unittest.TestCase)
                   and value.__module__ == module.__name__]
        if not classes:
            raise ValueError(f"{source_path} defines no source tests")
    if (directory / "characterization.yaml").is_file() and not any(
            issubclass(cls, DatasetTests) for cls in classes):
        classes.insert(0, DatasetTests)
    has_source_tests = source_path.is_file()
    suite = unittest.TestSuite()
    for cls in classes:
        if issubclass(cls, DatasetTests):
            attributes = {"directory": directory, "tables_directory": tables_directory}
            if source_audit is not None:
                attributes.update(source_audit=staticmethod(source_audit),
                                  test_independent_provider_claims=_source_claim_test)
            elif not has_source_tests:
                attributes["test_independent_provider_claims"] = _missing_source_test
            cls = type(f"{slug}_{cls.__name__}", (cls,), attributes)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    return suite


def dataset_suite(root, names=None, *, source_audit=None):
    available = benchmark_names(root)
    selected = available if names is None else list(dict.fromkeys(names))
    if not selected:
        raise ValueError("No reviewed benchmark datasets found")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"No reviewed dataset or source tests registered for: {', '.join(sorted(unknown))}")
    suite = unittest.TestSuite()
    for slug in selected:
        try:
            suite.addTests(benchmark_suite(root, slug, source_audit=source_audit))
        except Exception as exc:
            # An import failure must fail this benchmark without hiding the rest.
            def import_failure(error=exc):
                raise error
            suite.addTest(unittest.FunctionTestCase(import_failure,
                                                    description=f"{slug}: load source tests"))
    return suite


def main(root, *, source_audit=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", help="Benchmark slugs; defaults to all reviewed datasets")
    parser.add_argument("--all", action="store_true", help="Test all reviewed datasets")
    parser.add_argument("--list", action="store_true", help="List registered benchmarks without reading data")
    parser.add_argument("--allow-missing", action="store_true", help="Skip absent tables on a code-only checkout")
    args = parser.parse_args(argv)
    if args.all and args.benchmarks:
        parser.error("choose benchmark names or --all")
    if args.list:
        print("\n".join(benchmark_names(root)))
        return 0
    previous = os.environ.get("MEASUREMENT_DB_FULL_TEST")
    os.environ["MEASUREMENT_DB_FULL_TEST"] = "0" if args.allow_missing else "1"
    try:
        suite = dataset_suite(root, args.benchmarks or None, source_audit=source_audit)
        count = len(args.benchmarks) if args.benchmarks else len(benchmark_names(root))
        print(f"Checking {count} reviewed benchmark(s); unregistered folders are not included.", flush=True)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return int(not result.wasSuccessful() or (bool(result.skipped) and not args.allow_missing))
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        if previous is None:
            os.environ.pop("MEASUREMENT_DB_FULL_TEST", None)
        else:
            os.environ["MEASUREMENT_DB_FULL_TEST"] = previous
