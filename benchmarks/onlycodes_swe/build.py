#!/usr/bin/env python3
"""Curate Onlycodes SWE-bench Mini's released per-attempt results into measurement tables."""

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher

MODELS = {"claude": "claude-sonnet-4-6", "codex": "gpt-5.5"}
# The released CSV keeps the harness's legacy arm label; the paper calls it code_only.
ARMS = {"baseline": "baseline", "bash_only": "bash_only", "onlycode": "code_only"}
SUBSETS = {"swebench-verified-mini": 50, "swebench-datasci-mini": 50}
SEEDS = {1, 2, 3}
RULE = (
    "After the agent exits, apply the held-out test patch, then run test_cmd. PASS requires the patch to "
    "apply, pytest --collect-only to collect at least one test, test_cmd to exit 0, and at least one "
    "executed, non-skipped test; otherwise FAIL. test_cmd: "
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def harness_table(source: str, name: str) -> dict:
    """Read one literal per-instance table from the pinned harness without executing it."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise ValueError(f"harness.py no longer defines {name}")


class OnlycodesSWE(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results", "tasks", "harness", "swebench_verified", "swebench_full")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform released SWE-bench Mini attempts into subjects, items, and responses."""
        # 1. Load the released SWE attempts; any change to the released design stops the build.
        results = pd.read_csv(self.raw_dir / "all_results.csv")
        results["response_key"] = results.index  # Row in the released CSV.
        attempts = results.loc[results.benchmark.eq("swebench")].reset_index(drop=True)
        keys = ["agent", "arm", "instance_id", "seed"]
        require(len(attempts) == 1800 and not attempts.duplicated(keys).any(), "SWE attempt identity changed")
        require(set(attempts.verdict) == {"PASS", "FAIL"}, "Unexpected SWE verdict")
        require(set(attempts.agent) == set(MODELS) and set(attempts.arm) == set(ARMS), "Unexpected agent or arm")
        require(set(attempts.seed) == SEEDS and set(attempts.run) == {1}, "Repeat design changed")
        require(attempts.groupby("dataset").instance_id.nunique().to_dict() == SUBSETS, "SWE task cohort changed")
        cells = attempts.groupby(["agent", "arm", "instance_id"]).seed.agg(set)
        require(len(cells) == 600 and cells.eq(SEEDS).all(), "An (agent, arm, task) cell lacks a seed")

        # 2. Describe each model, harness, harness version and tool surface as one subject.
        attempts["harness_version"] = attempts.agent_version.str.extract(r"(\d+\.\d+\.\d+)", expand=False)
        require(attempts.harness_version.notna().all(), "Unparsed harness version")
        attempts["subject_key"] = attempts.agent + "|" + attempts.arm + "|" + attempts.harness_version
        subjects = attempts.drop_duplicates("subject_key")[
            ["subject_key", "agent", "arm", "agent_surface", "harness_version"]
        ].reset_index(drop=True)
        subjects["raw_label"] = subjects.agent.map(MODELS)
        subjects["features"] = [
            {"harness": row.agent_surface, "harness_version": row.harness_version, "tool_mode": ARMS[row.arm]}
            for row in subjects.itertuples()
        ]

        # 3. Join attempted tasks to the task files the harness read.
        tasks = attempts[["dataset", "instance_id"]].drop_duplicates().reset_index(drop=True)
        records = [
            yaml.safe_load((self.raw_dir / "tasks/problems/swe" / row.dataset / f"{row.instance_id}.yaml").read_text())
            for row in tasks.itertuples()
        ]
        tasks = tasks.join(pd.DataFrame(records)[["instance_id", "repo", "base_commit", "test_cmd", "patch_file",
                                                  "problem_statement"]].rename(columns={"instance_id": "task_file_id"}))
        require(tasks.instance_id.eq(tasks.task_file_id).all(), "Task file names a different instance")
        require(tasks.patch_file.eq("patches/" + tasks.instance_id + "_tests.patch").all(), "Unexpected test patch path")
        require(tasks.problem_statement.str.strip().str.len().gt(0).all(), "A task lacks its problem statement")
        tasks["test_patch"] = [(self.raw_dir / "tasks" / path).read_text() for path in tasks.patch_file]

        # 4. Take each gold patch from the one SWE-bench row matching the task's repository, commit and tests.
        columns = ["instance_id", "repo", "base_commit", "patch", "test_patch"]
        verified = pd.read_parquet(self.raw_dir / "swebench_verified.parquet", columns=columns)
        full = pd.read_parquet(self.raw_dir / "swebench_full.parquet", columns=columns)
        released = pd.concat([verified.assign(swebench_dataset="SWE-bench_Verified"),
                              full.loc[~full.instance_id.isin(verified.instance_id)].assign(swebench_dataset="SWE-bench")])
        tasks = tasks.merge(released, on="instance_id", how="left", suffixes=("", "_swebench"),
                            validate="one_to_one", indicator=True)
        require(tasks._merge.eq("both").all(), "An attempted task has no SWE-bench row")
        require(tasks.repo.eq(tasks.repo_swebench).all() and tasks.base_commit.eq(tasks.base_commit_swebench).all(),
                "A task's repository or base commit differs from SWE-bench")
        require(tasks.test_patch.eq(tasks.test_patch_swebench).all(), "A held-out test patch differs from SWE-bench")
        require(tasks.patch.str.strip().str.len().gt(0).all(), "A matched SWE-bench row lacks its gold patch")

        # 5. Record the grading protocol and pre-agent environment patches from the pinned harness.
        harness = (self.raw_dir / "harness/swebench/harness.py").read_text()
        seeds = harness_table(harness, "_INSTANCE_SOURCE_SEEDS")
        pytest_args = harness_table(harness, "_INSTANCE_EXTRA_PYTEST_ARGS")
        env = harness_table(harness, "_INSTANCE_ENV")
        revision = next(source["revision"] for source in self.source_manifest["upstream"]
                        if source.get("name") == "harness")
        protocol = f"https://github.com/hyang0129/onlycodes/blob/{revision}/swebench/harness.py"
        verifiers = [
            json.dumps({"kind": "onlycodes_swe_harness", "protocol": protocol, "repo": row.repo,
                        "base_commit": row.base_commit, "test_patch": row.test_patch, "test_cmd": row.test_cmd,
                        "extra_pytest_args": pytest_args.get(row.instance_id),
                        "extra_env": env.get(row.instance_id)}, sort_keys=True)
            for row in tasks.itertuples()
        ]
        # A source seed patch changes the repository the agent starts from, so it is part of the stimulus.
        attachments = [
            [{"source_path": f"tasks/{seeds[task_id]}", "path": Path(seeds[task_id]).name,
              "media_type": "text/x-diff", "role": "source_seed_patch"}] if task_id in seeds else None
            for task_id in tasks.instance_id
        ]
        items = tasks.assign(
            item_key=tasks.instance_id, raw_item_id=tasks.instance_id, content=tasks.problem_statement,
            attachments=attachments,
            grading_criterion=[{"reference_answer": row.patch, "rule": RULE + row.test_cmd} for row in tasks.itertuples()],
            verifier=[ExactMatcher(spec) for spec in verifiers],
            features=[{"subset": row.dataset, "repo": row.repo, "base_commit": row.base_commit,
                       "reference_source": row.swebench_dataset} for row in tasks.itertuples()],
        )

        # 6. Map released verdicts to responses; seeds are trials.
        responses = attempts.assign(
            item_key=attempts.instance_id, response=attempts.verdict.eq("PASS").astype(float), trial=attempts.seed
        )
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "grading_criterion",
                            "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
        }


if __name__ == "__main__":
    OnlycodesSWE(__file__).main_from_args()
