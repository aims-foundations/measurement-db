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

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "onlycodes_swe"
MODELS = {"claude": "claude-sonnet-4-6", "codex": "gpt-5.5"}
HARNESSES = {"claude": "claude_code", "codex": "codex_cli"}
TOOL_MODES = {"baseline": "baseline", "bash_only": "bash_only", "onlycode": "code_only"}


def features(value):
    return dict(part.split("=", 1) for part in value.split(";")) if value else {}


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY

    def provider_records(self):
        rows = pd.read_csv(DIRECTORY / "raw/all_results.csv")
        return rows.loc[rows.benchmark.eq("swebench")]

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
        repos = Counter()
        for path in sorted((DIRECTORY / "raw/tasks/problems/swe").glob("*/*.yaml")):
            if path.stem in set(records.instance_id):
                task = yaml.safe_load(path.read_text())
                repos[f"{path.parent.name.removeprefix('swebench-')}:{task['repo'].split('/')[1]}"] += 1
        check_source_claims(self.expected, {
            "released_responses": len(records),
            "released_passes": int(records.verdict.eq("PASS").sum()),
            "reported_pass_rates": {f"swebench:{agent}:{arm}": rate for (agent, arm), rate in rates.items()},
            "reported_repository_counts": dict(repos),
        })

    def test_reference_patches_match_swebench_test_patches(self):
        # Every gold patch was taken from a SWE-bench row whose held-out tests equal Onlycodes' own.
        for row in self.tables["items"].itertuples():
            spec = json.loads(json.loads(row.verifier)["spec"])
            patch = (DIRECTORY / "raw/tasks/patches" / f"{row.raw_item_id}_tests.patch").read_text()
            self.assertEqual(spec["test_patch"], patch)
            self.assertTrue(json.loads(row.grading_criterion)["reference_answer"].strip())

    def test_repeated_trials_are_contiguous_per_instrument(self):
        for _, group in self.tables["responses"].groupby(["subject_id", "item_id"]):
            self.assertEqual(sorted(group.trial), list(range(1, len(group) + 1)))


if __name__ == "__main__":
    unittest.main()
