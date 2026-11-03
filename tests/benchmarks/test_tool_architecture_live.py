#!/usr/bin/env python3
"""Read-only checks of the reviewed release and its cached provider records."""
import hashlib
import json
import sys
import unittest
import zipfile
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "tool_architecture_live"
MODELS = {"kimi-k2.5": "kimi-k2.5", "qwen3-coder-30b": "Qwen3-Coder-30B-A3B-Instruct",
          "sonnet4.5": "claude-sonnet-4-5"}
FOLDERS = {"kimi-k2.5": "Kimi-K2.5", "qwen3-coder-30b": "Qwen3Coder-30B", "sonnet4.5": "Sonnet-4.5"}
TOOL_MODES = {"bashonly": "bashonly", "atomic-common": "atomic", "bash-search": "nlsearch",
              "bash-hypo": "hypotrack", "bash-scratchpad": "scratchpad", "py-interface": "python"}
SETUP_FOLDERS = {"bashonly": "BashOnly", "atomic-common": "Atomic", "bash-search": "NLSearch",
                 "bash-hypo": "HypoTrack", "bash-scratchpad": "Scratchpad", "py-interface": "Python"}


def features(value):
    return dict(part.split("=", 1) for part in value.split(";")) if value else {}


def grade(value):
    return None if value is None else float(value)


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY

    def archive(self):
        return zipfile.ZipFile(DIRECTORY / "raw/tool-arch-study-data.zip")

    def keyed_responses(self):
        """Key each table response by the released attempt it represents."""
        subjects = self.tables["subjects"].set_index("subject_id")
        items = self.tables["items"].set_index("item_id").raw_item_id
        traces = self.tables["traces"].set_index("response_id").trace
        result = {}
        for row in self.tables["responses"].itertuples():
            subject = subjects.loc[row.subject_id]
            key = (subject.display_name, subject.harness, features(subject.subject_features_extra)["tool_mode"],
                   items[row.item_id], int(row.trial))
            result[key] = (None if row.response != row.response else float(row.response), traces.get(row.response_id))
        return result

    def test_provider_outcomes_and_traces(self):
        archive = self.archive()
        released = json.loads(archive.read("data/processed/figure4/inputs.json"))["trajectories"]
        table2 = json.loads(archive.read("data/processed/table2/inputs.json"))["per_instance"]
        actual = self.keyed_responses()
        expected = {}
        for row in released:
            harness = "simple_agent" if row["setup"] == "py-interface" else "trae_agent"
            suffix = f"__{row['repeat']}" if row["repeat"] else ""
            member = (f"data/raw_traj/main_live/{FOLDERS[row['actor']]}/{SETUP_FOLDERS[row['setup']]}/"
                      f"{row['instance']}{suffix}.json")
            expected[(MODELS[row["actor"]], harness, TOOL_MODES[row["setup"]], row["instance"], row["repeat"] + 1)] = (
                grade(row["resolved"]), hashlib.sha256(archive.read(member)).hexdigest())
        observed = {key: (value, hashlib.sha256(trace.encode()).hexdigest()) for key, (value, trace) in actual.items()}
        self.assertEqual(observed, expected)
        passes = Counter()
        for row in table2:
            passes[f"{row['actor']}:{row['setup']}"] += row["n_pass"]
        check_source_claims(self.expected, {
            "released_responses": len(released),
            "released_resolved": sum(row["resolved"] is True for row in released),
            "unfinished_attempts": sum(row["resolved"] is None for row in released),
            "reported_trajectories": len(released),
            "reported_tasks": len({row["instance"] for row in released}),
            "reported_all_attempt_resolve_rates": {key: count / 650 for key, count in passes.items()},
        })

    def test_items_match_the_selected_tasks(self):
        selected = {row["instance_id"]: row
                    for row in json.loads((DIRECTORY / "raw/swebench_live_verified_mini.json").read_text())}
        self.assertEqual(set(self.tables["items"].raw_item_id), set(selected))
        for row in self.tables["items"].itertuples():
            spec = json.loads(json.loads(row.verifier)["spec"])
            task = selected[row.raw_item_id]
            self.assertEqual(row.content, task["problem_statement"])
            self.assertEqual(json.loads(row.grading_criterion)["reference_answer"], task["patch"])
            for field in ["test_patch", "test_cmds", "FAIL_TO_PASS", "PASS_TO_PASS", "log_parser"]:
                self.assertEqual(spec[field], task[field])
            # Only the duplicated verified definition carries a conflicting test command.
            expected = [["python -m pytest -rA ."]] if row.raw_item_id == "conan-io__conan-18153" else None
            self.assertEqual(spec["conflicting_test_cmds"], expected)

    def test_trials_are_released_repeats(self):
        for _, group in self.tables["responses"].groupby(["subject_id", "item_id"]):
            self.assertEqual(sorted(group.trial), list(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
