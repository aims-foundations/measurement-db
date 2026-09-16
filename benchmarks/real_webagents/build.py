#!/usr/bin/env python3
"""Curate REAL's pinned leaderboard and task definitions without filling source gaps."""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge

API_URL = "https://realevals.xyz/api/getVerifiedModelData"

# Upstream harness repo: one JSON file per task (goal, evals[], ...) and the
# checker that grades those evals against a recorded environment-state diff.
GITHUB_TASKS_BASE = (
    "https://raw.githubusercontent.com/agi-inc/REAL/main/"
    "src/agisdk/REAL/browsergym/webclones/tasks"
)
EVALUATOR_URL = (
    "https://raw.githubusercontent.com/agi-inc/REAL/main/"
    "src/agisdk/REAL/browsergym/webclones/evaluate.py"
)

# WebCloneEvaluator.evaluate() is the same for every task: it ANDs each entry
# in the task's evals[] (the per-item reference_answer). Fixed JSON descriptor,
# not per-task, so it is computed once rather than in a loop.
VERIFIER_SPEC = json.dumps(
    {
        "kind": "external_judge",
        "source": EVALUATOR_URL,
        "class": "WebCloneEvaluator.evaluate",
        "logic": (
            'ANDs every entry of the task\'s evals[]. type="jmespath": runs '
            "`query` against the recorded environment-state diff and exact-"
            'matches `expected_value`. type="llm_boolean": sends '
            "(model_response, `rubric`) to an LLM judge and passes if the "
            "returned similarity score > 0.8."
        ),
    },
    sort_keys=True,
)



# The v2 task definitions were not included in the captured release. Describe
# only the deterministic mapping from the provider's released verdict fields;
# do not infer v2 rubrics from similarly named older tasks.
PROVIDER_MAPPING = json.dumps({
    "kind": "provider_result_mapping",
    "source": API_URL,
    "logic": "1 iff evalsFailed is empty; if absent, 1 iff accuracy >= 100; otherwise 0",
    "upstream_task_verifier": "not released in the captured v2 task snapshot",
}, sort_keys=True)


class REALWebAgents(BenchmarkBuild):
    def build_subject_item_response_rows(self) -> None:
        data = json.loads((self.raw_dir / "model_data.json").read_text())
        task_evals = {}
        for path in sorted((self.raw_dir / "tasks").glob("*.json")):
            evals = json.loads(path.read_text()).get("evals")
            if evals:
                task_evals[path.stem] = evals
        items = {}
        trials = Counter()
        for model in data["aiModels"]:
            if not model.get("name"):
                continue
            subject = self.add_subject(model["name"])
            for website in model.get("websites", []):
                website_id = website.get("websiteId") or website.get("name")
                for task in website.get("tasks", []):
                    if task.get("id") is None or task.get("accuracy") is None:
                        continue
                    raw_id = str(task["id"])
                    key = (raw_id, website_id)
                    if key not in items:
                        evals = task_evals.get(raw_id)
                        if evals:
                            criterion = {"rule": json.dumps(evals, sort_keys=True)}
                            verifier = (Judge(spec=VERIFIER_SPEC, judged_by="llm")
                                        if any(e.get("type") == "llm_boolean" for e in evals)
                                        else ExactMatcher(spec=VERIFIER_SPEC))
                        else:
                            criterion = {"rule": "Success requires all provider task checks to pass; task-specific checks are unavailable in the captured release."}
                            verifier = ExactMatcher(spec=PROVIDER_MAPPING)
                        items[key] = self.add_item(
                            raw_item_id=raw_id, content=task.get("prompt") or raw_id,
                            grading_criterion=criterion, verifier=verifier,
                            features={"website": website_id} if website_id else None,
                        )
                    failed = task.get("evalsFailed")
                    success = float(len(failed) == 0) if failed is not None else float(float(task["accuracy"]) >= 100)
                    retrieved = task.get("retrievedAnswer")
                    trace = (retrieved if isinstance(retrieved, str) and retrieved.strip()
                             and retrieved.strip() not in ("Done", "No response") else None)
                    item = items[key]
                    trials[subject, item] += 1
                    self.add_response(subject_id=subject, item_id=item, trial=trials[subject, item],
                                      response=success, trace=trace)


if __name__ == "__main__":
    REALWebAgents(__file__).main()
