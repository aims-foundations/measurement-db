#!/usr/bin/env python3
"""Read-only checks; record reviewed expectations in characterization.yaml.

Implement source_observations() independently of build.py. See the repository's
characterization.md for the schema, citation rules, and review workflow.
"""
from pathlib import Path
import os
import sys
import unittest

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR.parents[2]))

from measurement_db.scripts.build_measurement_tables import validate_dataset
from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    TABLE_NAMES, load_characterization, check_tables, check_source_claims,
)


def source_observations():
    """Return {claim_id: measured_value} from an independent upstream audit.

Read the archived inputs; reconcile the curated records with them. For a
reported aggregate, reconstruct that statistic over the matching curated scope.
Include every claim in characterization.yaml, using the same grouping keys.
Never return the expected values or call the builder to obtain observations.
"""
    raise NotImplementedError("Implement the benchmark's independent source audit")


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = ("items", "subjects", "benchmarks", "responses")
        missing = [name for name in required
                   if not (BENCHMARK_DIR / f"{name}.parquet").is_file()]
        if missing:
            message = f"generated benchmark table(s) are absent: {missing}; run build.py first"
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        path = BENCHMARK_DIR / "characterization.yaml"
        if not path.is_file():
            raise RuntimeError(f"{path} is absent; review the build and record its characterization")
        cls.expected = load_characterization(path)
        cls.tables = {name: pd.read_parquet(BENCHMARK_DIR / f"{name}.parquet")
                      for name in TABLE_NAMES
                      if (BENCHMARK_DIR / f"{name}.parquet").is_file()}

    def test_complete_dataset(self):
        validate_dataset(self.tables, expected_benchmark_id=BENCHMARK_DIR.name)

    def test_reviewed_tables(self):
        check_tables(self.expected, self.tables)

    def test_source_claims(self):
        check_source_claims(self.expected, source_observations())

    # Add named tests for source-specific parsing, coverage, and trace attribution.


if __name__ == "__main__":
    unittest.main(verbosity=2)
