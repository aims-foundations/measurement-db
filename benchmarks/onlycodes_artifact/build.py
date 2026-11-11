#!/usr/bin/env python3
"""Curate Onlycodes Artifact's released per-attempt results into measurement tables."""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher

MODELS = {"claude": "claude-sonnet-4-6", "codex": "gpt-5.5"}
# The released CSV keeps the harness's legacy arm label; the paper calls it baseline.
ARMS = {"tool_rich": "baseline", "bash_only": "bash_only", "code_only": "code_only"}
CATEGORIES = {"algorithmic": 8, "data_engineering": 15, "data_processing": 8, "data_science": 15, "enumeration": 8,
              "iterative_numerical": 8, "ml_engineering": 15, "stateful_reasoning": 8, "verification_heavy": 8}
SEEDS = {1, 2, 3}
MEDIA_TYPES = {".py": "text/x-python", ".json": "application/json", ".jsonl": "application/x-ndjson",
               ".csv": "text/csv", ".yaml": "application/yaml"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class OnlycodesArtifact(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results", "tasks", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform released artifact-suite attempts into subjects, items, and responses."""
        # 1. Load the released artifact attempts; any change to the released design stops the build.
        results = pd.read_csv(self.raw_dir / "all_results.csv")
        results["response_key"] = results.index  # Row in the released CSV.
        attempts = results.loc[results.benchmark.eq("artifact")].reset_index(drop=True)
        keys = ["agent", "arm", "instance_id", "seed"]
        require(len(attempts) == 1674 and not attempts.duplicated(keys).any(), "Artifact attempt identity changed")
        require(set(attempts.verdict) == {"PASS", "FAIL"}, "Unexpected artifact verdict")
        require(set(attempts.agent) == set(MODELS) and set(attempts.arm) == set(ARMS), "Unexpected agent or arm")
        require(set(attempts.seed) == SEEDS and set(attempts.run) == {1}, "Repeat design changed")
        require(attempts.groupby("dataset").instance_id.nunique().to_dict() == CATEGORIES, "Artifact task cohort changed")
        cells = attempts.groupby(["agent", "arm", "instance_id"]).seed.agg(set)
        require(len(cells) == 558 and cells.eq(SEEDS).all(), "An (agent, arm, task) cell lacks a seed")

        # 2. Describe each model, harness, harness version and tool surface as one subject.
        # Claude runs span two Claude Code versions; each attempt keeps the version it recorded.
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
        tasks["task_dir"] = [Path("tasks/problems/artifact", category, task_id.split("__", 1)[1])
                             for category, task_id in zip(tasks.dataset, tasks.instance_id)]
        specs = [yaml.safe_load((self.raw_dir / task_dir / "task.yaml").read_text()) for task_dir in tasks.task_dir]
        require([spec["instance_id"] for spec in specs] == tasks.instance_id.tolist(), "Task file names a different instance")
        require([spec["category"] for spec in specs] == tasks.dataset.tolist(), "Task file names a different category")
        require(all(spec["hidden_grader"] == "grader/hidden.py" for spec in specs), "Unexpected grader location")

        # 4. The agent starts from a copy of workspace/ without its generator; the harness then runs the
        #    generator with a seed derived from the task ID to write the bulk input data.
        attachments, contents = [], []
        for task_dir, spec in zip(tasks.task_dir, specs):
            contents.append((self.raw_dir / task_dir / spec["problem_statement"]).read_text())
            workspace = task_dir / spec["workspace_dir"]
            generator = task_dir / spec["workspace_generator"] if spec.get("workspace_generator") else None
            files = sorted(path.relative_to(self.raw_dir) for path in (self.raw_dir / workspace).rglob("*") if path.is_file())
            require(generator is None or generator in files, "Declared workspace generator is missing")
            require(all(path.suffix in MEDIA_TYPES for path in files), "Unreviewed workspace file type")
            attachments.append([
                {"source_path": str(path), "path": path.relative_to(task_dir).as_posix(),
                 "media_type": MEDIA_TYPES[path.suffix],
                 "role": "workspace_generator" if path == generator else "workspace_file"}
                for path in files
            ] or None)

        # 5. Grading runs grader/hidden.py on the agent's scratch directory; only four graders read their
        #    committed reference output, which is then the reference answer.
        revision = next(source["revision"] for source in self.source_manifest["upstream"]
                        if source.get("name") == "harness")
        protocol = f"https://github.com/hyang0129/onlycodes/blob/{revision}/swebench/artifact_grade.py"
        criteria, verifiers = [], []
        for task_dir, spec in zip(tasks.task_dir, specs):
            grader = (self.raw_dir / task_dir / spec["hidden_grader"]).read_text()
            reference = spec.get("reference_output")
            read_by_grader = reference is not None and Path(reference).name in grader
            criteria.append({
                "reference_answer": (self.raw_dir / task_dir / reference).read_text() if read_by_grader else None,
                "rule": f"grader/hidden.py grade(scratch_dir) must return passed=True for {spec['output_artifact']}",
            })
            verifiers.append(ExactMatcher(json.dumps({
                "kind": "onlycodes_hidden_grader", "protocol": protocol, "output_artifact": spec["output_artifact"],
                "hidden_grader": grader, "reference_output": reference if read_by_grader else None,
            }, sort_keys=True)))
        require(sum(criterion["reference_answer"] is not None for criterion in criteria) == 4,
                "Graders reading their committed reference output changed")

        items = tasks.assign(
            item_key=tasks.instance_id, raw_item_id=tasks.instance_id, content=contents, attachments=attachments,
            grading_criterion=criteria, verifier=verifiers,
            features=[{"category": spec["category"], "difficulty": spec.get("difficulty")} for spec in specs],
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
    OnlycodesArtifact(__file__).main_from_args()
