#!/usr/bin/env python3
"""Read-only checks of the reviewed release and its cached provider records."""
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "onlycodes_artifact"
MODELS = {"claude": "claude-sonnet-4-6", "codex": "gpt-5.5"}
HARNESSES = {"claude": "claude_code", "codex": "codex_cli"}
TOOL_MODES = {"tool_rich": "baseline", "bash_only": "bash_only", "code_only": "code_only"}


def features(value):
    return dict(part.split("=", 1) for part in value.split(";")) if value else {}


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY

    def provider_records(self):
        rows = pd.read_csv(DIRECTORY / "raw/all_results.csv")
        return rows.loc[rows.benchmark.eq("artifact")]

    def released_responses(self):
        """Key each table response by the released attempt it represents."""
        subjects = self.tables["subjects"].set_index("subject_id")
        items = self.tables["items"].set_index("item_id").raw_item_id
        result = Counter()
        for row in self.tables["responses"].itertuples():
            subject = subjects.loc[row.subject_id]
            result[(subject.display_name, subject.harness, subject.harness_version,
                    features(subject.subject_features_extra)["tool_mode"], items[row.item_id],
                    int(row.trial), float(row.response))] += 1
        return result

    def test_provider_outcomes(self):
        records = self.provider_records()
        expected = Counter(
            (MODELS[r.agent], HARNESSES[r.agent], re.search(r"\d+\.\d+\.\d+", r.agent_version)[0], TOOL_MODES[r.arm],
             r.instance_id, int(r.seed), float(r.verdict == "PASS"))
            for r in records.itertuples()
        )
        self.assertEqual(self.released_responses(), expected)
        rates = (records.assign(passed=records.verdict.eq("PASS"))
                 .groupby(["agent", "arm", "instance_id"]).passed.mean()
                 .groupby(["agent", "arm"]).mean())
        categories = Counter(
            yaml.safe_load(path.read_text())["category"]
            for path in sorted((DIRECTORY / "raw/tasks/problems/artifact").glob("*/*/task.yaml"))
        )
        check_source_claims(self.expected, {
            "released_responses": len(records),
            "released_passes": int(records.verdict.eq("PASS").sum()),
            "reported_pass_rates": {f"artifact:{agent}:{arm}": rate for (agent, arm), rate in rates.items()},
            "reported_category_counts": dict(categories),
        })

    def test_agent_never_receives_grader_files(self):
        # The harness copies workspace/ only; grader/ and reference outputs stay hidden.
        for manifest in self.tables["items"].asset_manifest.dropna():
            for attachment in json.loads(manifest):
                self.assertTrue(attachment["path"].startswith("workspace/"))
                self.assertNotIn("reference_output", attachment["path"])

    def test_reference_answers_only_where_grader_reads_them(self):
        for row in self.tables["items"].itertuples():
            spec = json.loads(json.loads(row.verifier)["spec"])
            reference = json.loads(row.grading_criterion).get("reference_answer")
            self.assertEqual(reference is not None, spec["reference_output"] is not None)
            if reference is not None:
                self.assertIn(Path(spec["reference_output"]).name, spec["hidden_grader"])

    def test_trials_are_released_seeds_split_only_by_harness_version(self):
        # Trials keep the released seed. A Claude task whose seeds ran on two Claude Code versions has its
        # seeds divided between two subjects; together they still cover seeds 1-3 exactly once.
        subjects = self.tables["subjects"].set_index("subject_id")
        responses = self.tables["responses"].assign(
            system=lambda frame: frame.subject_id.map(
                lambda subject: (subjects.loc[subject, "display_name"], subjects.loc[subject, "subject_features_extra"])))
        for _, group in responses.groupby(["system", "item_id"]):
            self.assertEqual(sorted(group.trial), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
