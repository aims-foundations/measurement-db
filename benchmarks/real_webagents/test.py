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
        data = json.loads((DIRECTORY / "raw/model_data.json").read_text())
        for model in data["aiModels"]:
            if not model.get("name"):
                continue
            for site in model.get("websites", []):
                for task in site.get("tasks", []):
                    if task.get("id") is None or task.get("accuracy") is None:
                        continue
                    failed = task.get("evalsFailed")
                    grade = float(not failed) if failed is not None else float(task["accuracy"] >= 100)
                    trace = nonempty(task.get("retrievedAnswer"))
                    if trace is not None and trace.strip() in ("Done", "No response"):
                        trace = None
                    result[grade, text_hash(trace)] += 1
        return result

    def test_missing_v2_definitions_are_explicit(self):
        items = self.tables["items"]
        v2 = items[items.raw_item_id.str.startswith("v2.")]
        self.assertEqual(len(v2), 121)
        for row in v2.itertuples():
            self.assertIsNone(json.loads(row.grading_criterion).get("reference_answer"))
            self.assertIn("unavailable", json.loads(row.grading_criterion)["rule"])
            self.assertEqual(json.loads(json.loads(row.verifier)["spec"])["kind"], "provider_result_mapping")


if __name__ == "__main__":
    unittest.main()
