#!/usr/bin/env python3
"""Read-only checks of the reviewed release and its cached provider records."""
import hashlib
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_dataset

DIRECTORY = Path(__file__).resolve().parent


def nonempty(value):
    return value if isinstance(value, str) and value.strip() else None


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


def table_digest(frame):
    rows = []
    for values in frame.itertuples(index=False, name=None):
        values = [None if pd.isna(v) else v for v in values]
        rows.append(json.dumps(values, ensure_ascii=False, sort_keys=True, default=str))
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DIRECTORY / "responses.parquet").is_file():
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError("Run build.py before testing the complete release")
            raise unittest.SkipTest("Run build.py to populate the released tables")
        cls.tables = {name: pd.read_parquet(DIRECTORY / f"{name}.parquet")
                      for name in ("subjects", "items", "benchmarks", "responses", "traces")}
        cls.expected = json.loads((DIRECTORY / "testdata/characterization.json").read_text())

    def test_complete_dataset(self):
        validate_dataset(self.tables, expected_benchmark_id=DIRECTORY.name)

    def test_reviewed_measurements(self):
        for name in ("subjects", "items", "responses", "traces"):
            self.assertEqual(len(self.tables[name]), self.expected["counts"][name], name)
            self.assertEqual(table_digest(self.tables[name]), self.expected["digests"][name], name)

    def test_provider_outcomes_and_traces(self):
        expected = self.provider_records()
        traces = self.tables["traces"].set_index("response_id").trace.to_dict()
        actual = Counter((float(r.response), text_hash(traces.get(r.response_id)))
                         for r in self.tables["responses"].itertuples())
        self.assertEqual(actual, expected)

    def provider_records(self):
        result = Counter()
        rows = pd.read_parquet(DIRECTORY / "raw/openhands_trajectories.parquet")
        for row in rows.itertuples():
            if not pd.isna(row.resolved):
                result[float(int(row.resolved) == 1), text_hash(nonempty(row.model_patch))] += 1
        return result

    def test_repeated_trials_are_contiguous_per_instrument(self):
        for _, group in self.tables["responses"].groupby(["subject_id", "item_id"]):
            self.assertEqual(sorted(group.trial), list(range(1, len(group) + 1)))


if __name__ == "__main__":
    unittest.main()
