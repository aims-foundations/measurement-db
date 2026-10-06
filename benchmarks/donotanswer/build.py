#!/usr/bin/env python3
"""Reshape Do-Not-Answer's released model outputs and two annotation families."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DoNotAnswer(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("annotations")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read prompts and expand the model/annotation columns into a long table.
        data = pd.read_csv(self.raw_dir / "data_en.csv")
        if data.id.isna().any() or data.id.duplicated().any():
            raise ValueError("Do-Not-Answer question IDs must be unique")
        models = self.build_parameters["models"]
        columns = [model + "_" + field for model in models for field in ["harmful", "action", "response"]]
        values = data.reindex(columns=["id", *columns]).melt(id_vars="id", var_name="field", value_name="value")
        labels = values.field.str.extract(r"^(?P<model>.+)_(?P<kind>harmful|action|response)$")
        observations = values.join(labels).pivot(index=["id", "model"], columns="kind", values="value").reset_index()
        observations = observations.rename(columns={"response": "trace"}).melt(
            id_vars=["id", "model", "trace"], value_vars=["harmful", "action"],
            var_name="label_kind", value_name="response"
        ).dropna(subset=["response"])
        observations["response"] = pd.to_numeric(observations.response, errors="raise")
        if observations.response.mod(1).ne(0).any():
            raise ValueError("Do-Not-Answer labels must be integer categories")

        # 2. Each annotation family defines a distinct grading protocol for a prompt.
        # This keeps mixed scales on their own items while retaining both observations.
        items = observations[["id", "label_kind"]].drop_duplicates().merge(
            data[["id", "question"]], on="id", how="left", sort=False, validate="many_to_one"
        )
        definitions = self.grading["verifiers"]
        items = items.assign(
            item_key=range(len(items)), raw_item_id=items.id.astype(str), content=items.question.astype(str),
            grading_criterion=items.label_kind.map(lambda kind: {
                "rule": definitions[kind]["spec"], "response_scale": definitions[kind]["response_scale"]
            }),
            verifier=items.label_kind.map(lambda kind: ExactMatcher(spec=json.dumps(
                {"kind": definitions[kind]["kind"], "spec": definitions[kind]["spec"]}, sort_keys=True
            ))),
        )

        # 3. Link the annotations and complete generated answers to their subjects/items.
        subjects = pd.DataFrame({"subject_key": list(models), "raw_label": list(models.values())})
        responses = observations.merge(
            items[["id", "label_kind", "item_key"]], on=["id", "label_kind"],
            how="left", sort=False, validate="many_to_one"
        ).rename(columns={"model": "subject_key"})
        responses = responses.assign(response_key=responses.index, test_condition="label=" + responses.label_kind)
        traces = responses.loc[responses.trace.map(lambda text: isinstance(text, str)), ["response_key", "trace"]]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DoNotAnswer(__file__).main_from_args()
