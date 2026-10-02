#!/usr/bin/env python3
"""Tabulate the selected WeaveBench gallery without running its agents or checks."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class WeaveBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("gallery", "harness", "dataset_notice")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths, labels = self.build_parameters["paths"], self.build_parameters["labels"]

        # 1. Read the original gallery index and result JSON directly from the archive.
        with tarfile.open(self.raw_dir / paths["archive"], "r:gz") as archive:
            members = {member.name: member for member in archive if member.isfile()}
            manifest = pd.read_json(archive.extractfile(paths["root"] + "/" + paths["manifest"]),
                                    dtype=False, convert_dates=False, precise_float=True)
            result_names = sorted(name for name in members if name.startswith(paths["root"] + "/" + paths["records"])
                                  and name.endswith(".json"))
            payloads = [json.load(archive.extractfile(name)) for name in result_names]
        records = pd.DataFrame(payloads).assign(source_member=result_names, source_record=payloads)

        # 2. Reconcile the manifest and preserve final scores without guessing missing grades.
        if records.task_id.duplicated().any() or manifest.id.duplicated().any() or set(records.task_id) != set(manifest.id):
            raise ValueError("WeaveBench requires exactly one released rollout per manifest task")
        joined = records.merge(manifest, left_on="task_id", right_on="id", validate="one_to_one", suffixes=("", "_index"))
        for field in ["model", "harness", "score"]:
            if not (joined[field].eq(joined[field + "_index"]) | (joined[field].isna() & joined[field + "_index"].isna())).all():
                raise ValueError("WeaveBench manifest disagrees with the native result: " + field)
        actions = joined.steps.map(lambda steps: sum(step["kind"] in ("cli", "gui") for step in steps))
        if not joined.is_hack.eq(joined.hack).all() or not actions.eq(joined.steps_index).all():
            raise ValueError("WeaveBench manifest has inconsistent fabrication flags or step counts")
        valid = records.source_record.map(lambda row: row["score"] is None or
            type(row["score"]) in (int, float) and 0 <= row["score"] <= 1)
        if not valid.all():
            raise ValueError("WeaveBench requires finite native scores in [0, 1], or explicit null")

        # 3. Separate recorded model/runtime pairs and keep complete task grading evidence.
        records["subject_key"] = records.model + "/" + records.harness
        subjects = records[["subject_key", "model", "harness"]].drop_duplicates().copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.model + " / " + subjects.harness
        subjects["features"] = [dict(harness=row.harness, recorded_model_label=row.model,
            settings_status=labels["settings_status"]) for row in subjects.itertuples()]
        items = records.rename(columns={"task_id": "item_key", "task_prompt": "content"}).copy()
        items["raw_item_id"] = items.item_key
        items["features"] = items[["category"]].to_dict("records")
        items["grading_criterion"] = [dict(rule=json.dumps(dict(pass_rule=self.grading["rule"],
            checks=row.checks, deliverables=row.deliverables), ensure_ascii=False)) for row in items.itertuples()]
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["published_judge"], sort_keys=True), judged_by="llm")

        # 4. Apply the paper's threshold; missing assessments remain ungraded attempts.
        responses = records.rename(columns={"task_id": "item_key"})[["subject_key", "item_key"]].copy()
        responses["response_key"] = responses.item_key
        responses["response"] = records.score.ge(self.grading["verifiers"]["published_judge"]["pass_threshold"]).astype(float)
        responses.loc[records.score.isna(), "response"] = None
        responses["trial"] = 1
        responses["test_condition"] = labels["selection"]

        # 5. Retain every published step, judgment and linked screenshot inside the original archive.
        records["screenshot_members"] = [[paths["root"] + "/" + paths["screenshots"].format(task_id=row.task_id, shot=step["shot"])
            for step in row.steps if step.get("shot")] for row in records.itertuples()]
        if not set(records.screenshot_members.explode().dropna()).issubset(members):
            raise ValueError("A WeaveBench trace references an absent screenshot")
        traces = records[["task_id"]].rename(columns={"task_id": "response_key"})
        traces["trace"] = [json.dumps(dict(source_archive=paths["archive"], source_member=row.source_member,
            source_record=row.source_record, screenshot_members=row.screenshot_members), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses, "traces": traces}


if __name__ == "__main__":
    WeaveBench(__file__).main_from_args()
