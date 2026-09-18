#!/usr/bin/env python3
"""Curate the cached SWE-rebench release; see metadata.yaml and curation_record.md."""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge

MODEL_LABEL = "Qwen3-Coder-480B-A35B-Instruct"
SUBJECT_FEATURES = {"harness": "OpenHands", "harness_version": "v0.54.0"}



def _to_list(x) -> list[str]:
    """Normalize a parquet list-cell (numpy array / None / NaN) to a python list."""
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    try:
        return [str(v) for v in list(x)]
    except TypeError:
        return []


def _clean_str(x) -> str | None:
    if isinstance(x, str) and x.strip():
        return x
    return None


def _grading_spec(bank_row) -> tuple[str | None, str | None]:
    """Build the (reference_answer, verifier) pair for one SWE-rebench instance.

    reference_answer states the rule the upstream `resolved` flag was computed
    from: after applying `test_patch`, every FAIL_TO_PASS test must pass and
    every PASS_TO_PASS test must still pass. verifier carries the artifacts
    needed to actually run that check (gold patch, test-injection diff, the
    two test lists, and the eval docker image).
    """
    fail_to_pass = _to_list(bank_row.get("FAIL_TO_PASS"))
    pass_to_pass = _to_list(bank_row.get("PASS_TO_PASS"))
    if not fail_to_pass and not pass_to_pass:
        return None, None  # instance carries no recoverable grading rule
    reference_answer = (
        f"apply test_patch; FAIL_TO_PASS ({len(fail_to_pass)}) must pass and "
        f"PASS_TO_PASS ({len(pass_to_pass)}) must still pass: "
        f"FAIL_TO_PASS={fail_to_pass}, PASS_TO_PASS={pass_to_pass}"
    )
    verifier = json.dumps(
        {
            "kind": "swebench_harness",
            "patch": _clean_str(bank_row.get("patch")),
            "test_patch": _clean_str(bank_row.get("test_patch")),
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "docker_image": _clean_str(bank_row.get("docker_image")),
        },
        sort_keys=True,
    )
    return reference_answer, verifier



class SWERebench(BenchmarkBuild):
    def build_subject_item_response_rows(self) -> None:
        bank = pd.read_parquet(self.raw_dir / "instances.parquet").set_index("instance_id").to_dict("index")
        settings_path = self.raw_dir / "subject_settings.json"
        labels = list(json.loads(settings_path.read_text())) if settings_path.exists() else [MODEL_LABEL]
        if len(labels) != 1:
            raise ValueError("SWE-rebench input contains trajectories for exactly one declared subject")
        subject = self.add_subject(labels[0], features=SUBJECT_FEATURES)
        items = {}
        trials = Counter()
        for row in pd.read_parquet(self.raw_dir / "openhands_trajectories.parquet").itertuples(index=False):
            if pd.isna(row.resolved):
                continue  # Preserve the released graded-attempt selection.
            if row.instance_id not in items:
                record = bank[row.instance_id]
                rule, spec = _grading_spec(record)
                items[row.instance_id] = self.add_item(
                    raw_item_id=row.instance_id,
                    content=record["problem_statement"],
                    grading_criterion={"reference_answer": _clean_str(record.get("patch")), "rule": rule},
                    verifier=ExactMatcher(spec=spec),
                )
            item = items[row.instance_id]
            trials[item] += 1
            self.add_response(subject_id=subject, item_id=item, trial=trials[item],
                              response=1.0 if int(row.resolved) == 1 else 0.0,
                              trace=_clean_str(row.model_patch))


if __name__ == "__main__":
    SWERebench(__file__).main_from_args()
