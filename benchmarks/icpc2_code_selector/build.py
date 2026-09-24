#!/usr/bin/env python3
"""Curate ICPC-2 code selections from the released predictions and question tables."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Icpc2CodeSelector(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("predictions", "questions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the provider tables, retaining the original prediction order.
        predictions = pd.read_csv(self.raw_dir / "predictions_data.csv", index_col=0)
        questions = pd.read_csv(self.raw_dir / "eval_dataset.csv", index_col=0)
        predictions = predictions.loc[predictions.model.notna() & predictions["query"].notna()].copy()
        predictions["response_key"] = range(len(predictions))
        predictions["top_k"] = pd.to_numeric(predictions.top_k).fillna(0).astype("int64")
        # Preserve the existing first-row choice for two repeated query definitions.
        # Later rows differ in candidate order; the response file contains no row ID.
        questions = questions[["query", "relevant_results", "search_engine_results_top_200"]].drop_duplicates("query")
        verdicts = predictions[["true_positive", "true_negative"]]
        if not verdicts.isin([True, False]).all().all():
            raise ValueError("ICPC-2 verdicts must be explicit booleans")

        # 2. A question and retrieval cutoff define the exact prompt shown to the model.
        items = predictions[["query", "top_k"]].drop_duplicates().merge(
            questions, on="query", how="left", validate="many_to_one", indicator=True
        )
        if items._merge.ne("both").any():
            raise ValueError("A predicted ICPC-2 question has no candidate list")
        candidates = items.search_engine_results_top_200.map(ast.literal_eval)
        # Parsing the provider's Python-literal list is the only non-tabular input step.
        candidates = pd.Series(
            [values[:limit] for values, limit in zip(candidates, items.top_k)], index=items.index
        ).astype(str)
        items = items.assign(
            item_key=range(len(items)),
            raw_item_id=items["query"] + "::top_k=" + items.top_k.astype(str),
            content=self.build_parameters["prompt"]["system"] + "\n\nQuery: "
                + items["query"] + "\n\nSearch engine results: " + candidates,
            features=items.top_k.astype(str).map(lambda value: {"top_k": value}),
            grading_criterion=items.relevant_results.map(
                lambda answer: {"rule": self.grading["rule"], **(
                    {"reference_answer": answer} if isinstance(answer, str) and answer else {}
                )}
            ),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["provider"], sort_keys=True)),
        )

        # 3. Preserve model labels and join each observation to its prompt.
        subjects = predictions[["model"]].drop_duplicates().rename(columns={"model": "raw_label"})
        subjects["subject_key"] = subjects.raw_label
        responses = predictions.merge(
            items[["query", "top_k", "item_key"]], on=["query", "top_k"],
            how="left", sort=False, validate="many_to_one"
        ).assign(subject_key=lambda frame: frame.model)
        responses["response"] = (responses.true_positive | responses.true_negative).astype(float)
        traces = responses.loc[responses.prediction.notna(), ["response_key", "prediction"]].rename(
            columns={"prediction": "trace"}
        )
        return {
            "subjects": subjects[["subject_key", "raw_label"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    Icpc2CodeSelector(__file__).main_from_args()
