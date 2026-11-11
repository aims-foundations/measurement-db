#!/usr/bin/env python3
"""Curate the Tool Architecture study's released SWE-bench Live attempts into measurement tables."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher

# Released actor labels map to model-registry keys; the archive's trajectory folders use the study's names.
ACTORS = {"kimi-k2.5": "kimi-k2.5", "qwen3-coder-30b": "Qwen3-Coder-30B-A3B-Instruct",
          "sonnet4.5": "claude-sonnet-4-5"}
ACTOR_FOLDERS = {"kimi-k2.5": "Kimi-K2.5", "qwen3-coder-30b": "Qwen3Coder-30B", "sonnet4.5": "Sonnet-4.5"}
SETUPS = {"bashonly": "bashonly", "atomic-common": "atomic", "bash-search": "nlsearch",
          "bash-hypo": "hypotrack", "bash-scratchpad": "scratchpad", "py-interface": "python"}
SETUP_FOLDERS = {"bashonly": "BashOnly", "atomic-common": "Atomic", "bash-search": "NLSearch",
                 "bash-hypo": "HypoTrack", "bash-scratchpad": "Scratchpad", "py-interface": "Python"}
HARNESSES = {setup: "simple_agent" if setup == "py-interface" else "trae_agent" for setup in SETUPS}
TASKS, REPEATS = 65, 10
DEFINITION = ["repo", "base_commit", "patch", "test_patch", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS",
              "test_cmds", "log_parser"]
RULE = (
    "Apply the agent's patch and the held-out test_patch, run test_cmds, and parse the log with log_parser; the "
    "attempt is resolved when every FAIL_TO_PASS and PASS_TO_PASS test passes. An attempt that produced no "
    "patch is not evaluated."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    """Compare parquet arrays and JSON lists as the same Python values."""
    return [plain(item) for item in value] if isinstance(value, (list, np.ndarray)) else value


class ToolArchitectureLive(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("archive", "tasks", "harness", "swebench_live_verified")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform released main-study attempts into subjects, items, responses, and traces."""
        archive = zipfile.ZipFile(self.raw_dir / "tool-arch-study-data.zip")

        # 1. Load per-attempt outcomes; resolved None marks an unfinished attempt that was never evaluated.
        attempts = pd.DataFrame(json.loads(archive.read("data/processed/figure4/inputs.json"))["trajectories"])
        keys = ["actor", "setup", "instance", "repeat"]
        require(len(attempts) == len(ACTORS) * len(SETUPS) * TASKS * REPEATS, "Main cohort size changed")
        require(not attempts.duplicated(keys).any(), "Duplicate released attempt")
        require(set(attempts.actor) == set(ACTORS) and set(attempts.setup) == set(SETUPS), "Unexpected design")
        cells = attempts.groupby(["actor", "setup", "instance"]).repeat.agg(set)
        require(len(cells) == len(ACTORS) * len(SETUPS) * TASKS and cells.eq(set(range(REPEATS))).all(),
                "An (actor, setup, task) cell lacks a repeat")
        require(attempts.resolved.map(lambda value: value is None or isinstance(value, bool)).all(),
                "Invalid resolved status")
        # Table 2's all-attempt outcomes count unfinished attempts as failures; they must otherwise agree.
        table2 = {(row["actor"], row["setup"], row["instance"]): row["resolved_by_repeat"]
                  for row in json.loads(archive.read("data/processed/table2/inputs.json"))["per_instance"]}
        require(all(table2[row.actor, row.setup, row.instance][row.repeat] == (row.resolved is True)
                    for row in attempts.itertuples()), "Table 2 and Figure 4 outcomes no longer reconcile")

        # 2. Describe each actor model, agent harness and tool architecture as one subject.
        attempts["subject_key"] = attempts.actor + "|" + attempts.setup
        subjects = attempts.drop_duplicates("subject_key")[["subject_key", "actor", "setup"]].reset_index(drop=True)
        subjects["raw_label"] = subjects.actor.map(ACTORS)
        subjects["features"] = [{"harness": HARNESSES[row.setup], "tool_mode": SETUPS[row.setup]}
                                for row in subjects.itertuples()]

        # 3. The bundled task file selects the tasks; the runners graded them against the HF verified split.
        selected = pd.DataFrame(json.loads((self.raw_dir / "swebench_live_verified_mini.json").read_text()))
        require(len(selected) == TASKS and set(selected.instance_id) == set(attempts.instance), "Task cohort changed")
        verified = pd.read_parquet(self.raw_dir / "swebench_live_verified.parquet")
        definitions, conflicts = [], {}
        for record in selected.to_dict("records"):
            rows = verified.loc[verified.instance_id.eq(record["instance_id"])].to_dict("records")
            matches = [row for row in rows if all(plain(row[field]) == plain(record[field]) for field in DEFINITION)]
            require(len(matches) == 1, f"{record['instance_id']} lacks exactly one matching verified definition")
            # The grader keeps duplicate rows, so a conflicting definition leaves the verdict's source ambiguous.
            others = [plain(row["test_cmds"]) for row in rows if row is not matches[0]]
            require(all(all(plain(row[field]) == plain(matches[0][field]) for field in DEFINITION if field != "test_cmds")
                        for row in rows), f"{record['instance_id']} has conflicting task content")
            if others:
                conflicts[record["instance_id"]] = others
            definitions.append(matches[0])
        tasks = pd.DataFrame(definitions)
        require(set(conflicts) == {"conan-io__conan-18153"}, "Duplicate verified definitions changed")

        # 4. Record the content, reference patch and grading protocol of each task.
        grader = next(source for source in self.source_manifest["upstream"] if "github.com/microsoft" in source["url"])
        protocol = f"{grader['url']}/blob/{grader['revision']}/swebench/harness/grading.py"
        verifiers = [
            ExactMatcher(json.dumps({
                "kind": "swebench_live_harness", "protocol": protocol, "dataset": "SWE-bench-Live/SWE-bench-Live",
                "split": "verified", "namespace": "starryzhang", "instance_image_tag": "latest",
                "repo": row["repo"], "base_commit": row["base_commit"], "test_patch": row["test_patch"],
                "test_cmds": plain(row["test_cmds"]), "log_parser": row["log_parser"],
                "FAIL_TO_PASS": plain(row["FAIL_TO_PASS"]), "PASS_TO_PASS": plain(row["PASS_TO_PASS"]),
                "conflicting_test_cmds": conflicts.get(row["instance_id"]),
            }, sort_keys=True))
            for row in definitions
        ]
        items = tasks.assign(
            item_key=tasks.instance_id, raw_item_id=tasks.instance_id, content=tasks.problem_statement,
            grading_criterion=[{"reference_answer": row["patch"], "rule": RULE} for row in definitions],
            verifier=verifiers,
            features=[{"repo": row["repo"], "base_commit": row["base_commit"]} for row in definitions],
        )

        # 5. Map outcomes to responses; repeats are trials and unfinished attempts stay ungraded.
        attempts["response_key"] = (attempts.subject_key + "|" + attempts.instance + "|"
                                    + attempts.repeat.astype(str))
        responses = attempts.assign(
            item_key=attempts.instance, trial=attempts.repeat + 1,
            response=attempts.resolved.map({True: 1.0, False: 0.0}),
        )

        # 6. Each attempt's complete released trajectory is its trace.
        statements = tasks.set_index("instance_id").problem_statement
        traces = []
        for row in attempts.itertuples():
            suffix = f"__{row.repeat}" if row.repeat else ""
            member = (f"data/raw_traj/main_live/{ACTOR_FOLDERS[row.actor]}/{SETUP_FOLDERS[row.setup]}/"
                      f"{row.instance}{suffix}.json")
            text = archive.read(member).decode("utf-8")
            trajectory = json.loads(text)
            require(trajectory["model"] == ACTOR_FOLDERS[row.actor], f"{member} records a different model")
            # Trae trajectories record the problem statement with \r\n line endings normalized to \n.
            statement = statements[row.instance]
            require(trajectory["task"] in (statement, statement.replace("\r\n", "\n")),
                    f"{member} records different task text")
            traces.append(text)
        traces = pd.DataFrame({"response_key": attempts.response_key, "trace": traces})

        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ToolArchitectureLive(__file__).main_from_args()
