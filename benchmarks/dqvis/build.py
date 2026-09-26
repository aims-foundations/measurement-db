#!/usr/bin/env python3
"""Curate DQVis's published individual expert reviews without regrading outputs."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class DQVis(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("dataset", "generation", "review")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        grading = self.grading["verifiers"]["human"]

        # 1. Read the complete review release; IDs are local to each reviewer.
        originals = pd.DataFrame({"record": json.loads((self.raw_dir / paths["reviews"]).read_text())})
        reviews = pd.json_normalize(originals.record, max_level=0).join(originals)
        reviews["source_row"] = reviews.index
        if reviews.duplicated(["reviewer", "id"]).any():
            raise ValueError("A reviewer has repeated native review IDs")
        if not reviews.review_status.isin(grading["labels"]).all():
            raise ValueError("Unexpected human review category")
        reviews["response"] = reviews.review_status.map(grading["labels"]).astype(float)

        # 2. Join the recorded generation input to its complete released schema.
        schemas = pd.DataFrame({"schema_record": json.loads((self.raw_dir / paths["schemas"]).read_text())})
        schemas["dataset_schema"] = schemas.schema_record.map(lambda value: value["udi:name"])
        reviews = reviews.merge(schemas, on="dataset_schema", how="left", validate="many_to_one")
        if reviews.schema_record.isna().any():
            raise ValueError("A reviewed generation references an absent dataset schema")
        reviews["content"] = [json.dumps(dict(query_base=row.query_base, dataset_schema=row.schema_record),
            ensure_ascii=False, sort_keys=True) for row in reviews.itertuples()]
        reviews["item_key"] = [hashlib.sha256(json.dumps([row.content, row.reviewer],
            ensure_ascii=False).encode()).hexdigest() for row in reviews.itertuples()]
        items = reviews.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(dataset_schema=row.dataset_schema,
            input_scope=parameters["options"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [Judge(judge="DQVis anonymous reviewer " + reviewer, judged_by="human",
            spec=json.dumps(dict(protocol=grading["protocol"], source=grading["source"],
                reviewer=reviewer), sort_keys=True)) for reviewer in items.reviewer]

        # 3. Preserve every individual judgment and its entire native triplet.
        # Multiple ratings of one output are not additional model executions.
        label = parameters["options"]["raw_label"]
        subjects = pd.DataFrame([dict(subject_key=label, raw_label=label,
            features=parameters["subject_features"])])
        reviews["subject_key"] = label
        reviews["response_key"] = reviews.source_row
        reviews["test_condition"] = [json.dumps(dict(data_id=row.data_id, reviewer=row.reviewer,
            review_id=row.id, observation_unit="human_rating"), sort_keys=True) for row in reviews.itertuples()]
        traces = reviews[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["reviews"], source_row=row.source_row,
            record=row.record), ensure_ascii=False, allow_nan=False) for row in reviews.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": reviews[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DQVis(__file__).main_from_args()
