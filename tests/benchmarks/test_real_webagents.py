#!/usr/bin/env python3
"""Read-only checks of the reviewed release and its cached provider records."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "real_webagents"


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
