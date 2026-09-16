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
        import yaml
        metadata = yaml.safe_load((DIRECTORY / "metadata.yaml").read_text())
        import re
        from measurement_db.scripts.build_measurement_tables.validate_benchmark_metadata import declared_source_artifacts
        paths = {re.sub(r"_x([0-9a-f]{2})_", lambda m: chr(int(m[1], 16)), d["file"]):
                 DIRECTORY / "raw" / d["file"]
                 for d in declared_source_artifacts(metadata["sources"], benchmark_dir=DIRECTORY)}
        result = Counter()
        lfs_pointers = 0
        for original, path in paths.items():
            if not original.startswith("results/"):
                continue
            record = json.loads(path.read_text())
            successes = set(record.get("resolved") or record.get("resolved_ids") or [])
            failures = set(record.get("unresolved_ids") or record.get("unresolved") or [])
            pred_path = paths.get("preds/" + Path(original).stem + ".jsonl")
            patches = {}
            if pred_path is not None:
                text = pred_path.read_text()
                if text.startswith("version https://git-lfs.github.com/spec/v1\n"):
                    lfs_pointers += 1
                    text = ""  # This captured file contains no released patch bytes.
                # JSON strings can contain Unicode line separators; JSONL
                # records are separated by ASCII newlines only.
                for line in text.split("\n"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    patch = nonempty(row.get("model_patch"))
                    if patch is not None:
                        patches[row["instance_id"]] = patch
            for raw_id in successes | failures:
                result[float(raw_id in successes), text_hash(patches.get(raw_id))] += 1
        self.assertEqual(lfs_pointers, 1)
        return result

    def test_repeated_trials_are_contiguous_per_instrument(self):
        for _, group in self.tables["responses"].groupby(["subject_id", "item_id"]):
            self.assertEqual(sorted(group.trial), list(range(1, len(group) + 1)))


if __name__ == "__main__":
    unittest.main()
