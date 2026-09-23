#!/usr/bin/env python3
"""Read-only checks of the reviewed release and its cached provider records."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "swe_rebench"


def nonempty(value):
    return value if isinstance(value, str) and value.strip() else None


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY


    def test_provider_outcomes_and_traces(self):
        expected = self.provider_records()
        traces = self.tables["traces"].set_index("response_id").trace.to_dict()
        actual = Counter((float(r.response), text_hash(traces.get(r.response_id)))
                         for r in self.tables["responses"].itertuples())
        self.assertEqual(actual, expected)
        check_source_claims(self.expected, {
            "released_responses": sum(expected.values()),
            "released_traces": sum(n for (_, trace), n in expected.items() if trace is not None),
        })

    def provider_records(self):
        result = Counter()
        source = DIRECTORY / "raw/trajectories.parquet"
        if not source.is_file():
            source = DIRECTORY / "raw/openhands_trajectories.parquet"
        rows = pd.read_parquet(source, columns=["resolved", "model_patch"])
        for row in rows.itertuples():
            if not pd.isna(row.resolved):
                result[float(int(row.resolved) == 1), text_hash(nonempty(row.model_patch))] += 1
        return result

    def test_repeated_trials_are_contiguous_per_instrument(self):
        for _, group in self.tables["responses"].groupby(["subject_id", "item_id"]):
            self.assertEqual(sorted(group.trial), list(range(1, len(group) + 1)))


if __name__ == "__main__":
    unittest.main()
