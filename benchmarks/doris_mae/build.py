#!/usr/bin/env python3
"""Curate DORIS-MAE's released Anno-GPT relevance judgments and explanations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DorisMae(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("dataset", "protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        source = parameters["paths"]["data"]

        # 1. Read the release as annotation, aspect, document and human-vote tables.
        data = json.loads((self.raw_dir / source).read_text())
        annotations = pd.json_normalize(data, "Annotation", max_level=0)
        annotations["record"] = annotations.to_dict("records")
        annotations["source_row"] = annotations.index
        aspects = pd.Series(data["aspect_id2aspect"], name="aspect").rename_axis("aspect_id").reset_index()
        documents = pd.json_normalize(data, "Corpus", max_level=0)[["abstract_id", "original_abstract"]]
        humans = pd.json_normalize(data, "Test_set", max_level=0)
        humans["aspect_id"] = humans.aspect_id.astype(str)

        # 2. Join the exact input text and any explicitly matching human votes.
        annotations["aspect_id"] = annotations.aspect_id.astype(str)
        annotations = annotations.merge(aspects, on="aspect_id", how="left", validate="many_to_one")
        annotations = annotations.merge(documents, on="abstract_id", how="left", validate="many_to_one")
        annotations = annotations.merge(humans, on=["aspect_id", "abstract_id"], how="left", validate="many_to_one")
        if annotations[["aspect", "original_abstract"]].isna().any().any():
            raise ValueError("A released annotation has an absent aspect or document")
        annotations["item_key"] = annotations.aspect_id + "__" + annotations.abstract_id.astype(str)
        items = annotations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = [parameters["prompt"]["template"].format(req=row.aspect, abstract=row.original_abstract)
            for row in items.itertuples()]
        votes = items.loc[items.human_annotation.notna(), ["item_key", "human_annotation"]].copy()
        votes["vote"] = votes.human_annotation.map(lambda value: list(value.values()))
        counts = votes.explode("vote").groupby(["item_key", "vote"], sort=False).size().rename("count").reset_index()
        majority = counts.loc[counts["count"].ge(2)].set_index("item_key").vote.astype(int).astype(str)
        items["reference"] = items.item_key.map(majority)
        items["grading_criterion"] = [dict(reference_answer=None if pd.isna(row.reference) else row.reference,
            rule=self.grading["rule"]) for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["released"], sort_keys=True))] * len(items)
        items["features"] = [dict(input_scope=parameters["input_scope"]["note"]) for _ in items.index]

        # 3. Preserve every published label and full explanation, including repeats.
        subject = parameters["subject"]
        subjects = pd.DataFrame([dict(subject_key=subject["raw_label"], raw_label=subject["raw_label"],
            features=dict(harness=subject["harness"], **parameters["subject_features"]))])
        annotations["subject_key"] = subject["raw_label"]
        annotations["response_key"] = annotations.source_row
        annotations["response"] = annotations.score.astype(float).mask(annotations.score.eq(int(parameters["parsing"]["ungraded_score"])))
        traces = annotations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=source, source_row=int(row.source_row), record=row.record,
            human_annotation=row.human_annotation if isinstance(row.human_annotation, dict) else None),
            ensure_ascii=False, allow_nan=False) for row in annotations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": annotations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DorisMae(__file__).main_from_args()
