#!/usr/bin/env python3
"""Curate SWE-rebench's captured task and trajectory tables."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher

MODEL_LABEL = "Qwen3-Coder-480B-A35B-Instruct"
SUBJECT_FEATURES = {"harness": "OpenHands", "harness_version": "v0.54.0"}


class SWERebench(BenchmarkBuild):

    def download(self):
        return self.fetch_sources('tasks', 'trajectories')
    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform captured tables into subjects, items, responses, and traces."""
        # 1. Load task definitions and retain the released graded trajectories.
        task_files = sorted(name for name in self.source_files if name.startswith("instances/"))
        task_files = task_files or ["instances.parquet"]  # Archived combined table or local import.
        tasks = pd.concat([pd.read_parquet(self.raw_dir / name) for name in task_files], ignore_index=True).reindex(columns=[
            "instance_id", "problem_statement", "patch", "test_patch",
            "FAIL_TO_PASS", "PASS_TO_PASS", "docker_image"
        ])
        trajectory_file = "trajectories.parquet" if "trajectories.parquet" in self.source_files else "openhands_trajectories.parquet"
        attempts = pd.read_parquet(self.raw_dir / trajectory_file, columns=["instance_id", "resolved", "model_patch"])
        attempts = attempts.loc[attempts.resolved.notna()].reset_index(drop=True)
        attempts["response_key"] = attempts.index

        # 2. Describe the released subject, or the subject recorded by a fresh run.
        settings_path = self.raw_dir / "subject_settings.json"
        labels = list(json.loads(settings_path.read_text())) if settings_path.exists() else [MODEL_LABEL]
        if len(labels) != 1:
            raise ValueError("SWE-rebench input contains trajectories for exactly one declared subject")
        subjects = pd.DataFrame({
            "subject_key": labels, "raw_label": labels, "features": [SUBJECT_FEATURES]
        })

        # 3. Join distinct attempted task IDs to their definitions in first-attempt order.
        items = attempts[["instance_id"]].drop_duplicates().merge(
            tasks, on="instance_id", how="left", sort=False, validate="one_to_one", indicator=True
        )
        missing = items.loc[items._merge.eq("left_only"), "instance_id"].tolist()
        if missing:
            raise ValueError(f"Attempted tasks lack definitions: {missing[:5]}")
        items = items.drop(columns="_merge").set_index("instance_id")

        # 4. Assemble complete test lists, grading rules, and harness inputs for each item.
        tests = items[["FAIL_TO_PASS", "PASS_TO_PASS"]].reset_index().melt(
            id_vars="instance_id", var_name="kind", value_name="test_name"
        )
        tests = tests.loc[tests.test_name.str.len().gt(0)].explode("test_name")
        tests["test_name"] = tests.test_name.astype(str)
        test_lists = tests.groupby(["instance_id", "kind"], sort=False).test_name.agg(list).unstack("kind")
        test_lists = test_lists.reindex(index=items.index, columns=["FAIL_TO_PASS", "PASS_TO_PASS"])
        test_lists = test_lists.map(lambda names: names if isinstance(names, list) else [])
        failed_tests, passed_tests = test_lists.FAIL_TO_PASS, test_lists.PASS_TO_PASS
        missing_tests = failed_tests.str.len().eq(0) & passed_tests.str.len().eq(0)
        if missing_tests.any():
            raise ValueError(f"Attempted tasks lack grading tests: {items.index[missing_tests].tolist()[:5]}")

        # Blank artifacts are missing; nonempty text is preserved byte for byte.
        artifacts = items[["patch", "test_patch", "docker_image"]].replace(r"^\s*$", None, regex=True)
        artifacts = artifacts.astype(object).where(artifacts.notna(), None)
        rules = (
            "apply test_patch; FAIL_TO_PASS (" + failed_tests.str.len().astype(str) + ") must pass and "
            "PASS_TO_PASS (" + passed_tests.str.len().astype(str) + ") must still pass: "
            "FAIL_TO_PASS=" + failed_tests.astype(str) + ", PASS_TO_PASS=" + passed_tests.astype(str)
        )
        criteria = pd.DataFrame({"reference_answer": artifacts.patch, "rule": rules})
        verifiers = artifacts.assign(
            kind="swebench_harness", FAIL_TO_PASS=failed_tests, PASS_TO_PASS=passed_tests
        )
        verifier_specs = pd.Series(verifiers.to_dict("records"), index=items.index).map(
            lambda spec: json.dumps(spec, sort_keys=True)
        )
        items = items.assign(
            item_key=items.index, raw_item_id=items.index, content=items.problem_statement,
            grading_criterion=criteria.to_dict("records"), verifier=verifier_specs.map(ExactMatcher)
        )

        # 5. Preserve trajectory order and the released binary outcome mapping.
        responses = attempts.rename(columns={"instance_id": "item_key"}).assign(
            subject_key=labels[0], response=attempts.resolved.astype(int).eq(1).astype(float)
        )
        # Shared registration numbers trials after canonical item identities resolve.
        traces = responses.loc[
            responses.model_patch.str.strip().fillna("").ne(""), ["response_key", "model_patch"]
        ].rename(columns={"model_patch": "trace"})
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    SWERebench(__file__).main_from_args()
