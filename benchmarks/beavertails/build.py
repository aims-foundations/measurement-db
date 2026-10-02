#!/usr/bin/env python3
"""Curate the released BeaverTails model outputs and their three safety judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BeaverTails(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("evaluation")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the released model outputs, preserving each complete source record.
        paths = self.build_parameters["paths"]
        root = self.raw_dir / paths["release"]
        source = pd.read_json(root / paths["results"], dtype=False, convert_dates=False)
        source["record"] = source.to_dict("records")
        source = source.rename_axis("source_row").reset_index()
        if source.global_index.duplicated().any() or source.duplicated(["model", "index"]).any():
            raise ValueError("Released model/index and global_index keys must be unique")
        if not source.groupby("index")[["prompt", "category_id"]].nunique().eq(1).all().all():
            raise ValueError("An item index has different prompts or categories across models")

        # 2. Unpivot the three judgments; these are repeated grades of one generation.
        judges = list(self.grading["verifiers"])
        if not source.flagged.map(lambda value: set(value) == set(judges)).all():
            raise ValueError("Every released output must identify the declared judges")
        flags = pd.json_normalize(source.flagged)
        if not flags.map(lambda value: type(value) is bool).all().all():
            raise ValueError("Released safety flags must be explicit booleans")
        observations = source.join(flags).melt(id_vars=list(source.columns), value_vars=judges,
            var_name="judge_key", value_name="flagged_value").rename(columns={"model": "subject_key"})
        observations["response"] = observations.flagged_value.eq(False).astype(float)
        observations["response_key"] = observations.global_index.astype(str) + ":" + observations.judge_key
        observations["item_key"] = observations["index"].astype(str) + ":" + observations.judge_key

        # 3. Keep the generator as the subject and each grading protocol on the item.
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(harness="BeaverTails", released_model_label=model)
                                for model in subjects.subject_key]
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = items.prompt
        items["features"] = [dict(category_id=row.category_id, grading_judge=row.judge_key)
                             for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"])] * len(items)
        protocols = {key: dict(value) for key, value in self.grading["verifiers"].items()}
        protocols["gpt4"]["prompt_template"] = (root / paths["judge_prompt"]).read_text()
        items["verifier"] = [Judge(judge=protocols[key]["name"],
            judged_by=protocols[key]["judged_by"],
            spec=json.dumps(dict(judge_key=key, **protocols[key]), ensure_ascii=False, sort_keys=True))
            for key in items.judge_key]

        # 4. Link every judgment to the full output, all original flags and source row.
        source_file = str((root / paths["results"]).relative_to(self.raw_dir))
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=source_file, source_row=row.source_row,
            record=row.record, judge_key=row.judge_key), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BeaverTails(__file__).main_from_args()
