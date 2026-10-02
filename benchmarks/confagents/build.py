#!/usr/bin/env python3
"""Curate ConfAgents' released medical question and agent-answer table."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ConfAgents(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the question table and reshape its agent columns into attempts.
        data = pd.read_json(self.raw_dir / "ConfAgents.json", dtype=False)
        data["item_key"] = data.index
        methods = self.build_parameters["methods"]
        attempts = data.reindex(columns=["item_key", *methods]).melt(
            id_vars="item_key", var_name="method", value_name="result"
        )
        attempts = attempts.loc[attempts.result.map(lambda value: isinstance(value, dict))].reset_index(drop=True)
        answers = pd.json_normalize(attempts.result.tolist(), max_level=0).reindex(
            columns=["final_answer", "reasoning"]
        )
        attempts = attempts.drop(columns="result").join(answers).dropna(subset=["final_answer"])
        attempts["response_key"] = attempts.index

        # 2. Expand answer choices, then group their text into the original prompt order.
        options = pd.json_normalize(data.options.tolist(), max_level=0).assign(item_key=data.item_key)
        options = options.melt(id_vars="item_key", var_name="letter", value_name="answer").dropna(subset=["answer"])
        options["line"] = options.letter + ". " + options.answer.astype(str)
        options = options.sort_values(["item_key", "letter"]).groupby("item_key").line.agg("\n".join)
        items = data.loc[data.item_key.isin(attempts.item_key)].copy()
        items["content"] = (items.question.astype(str).str.strip() + "\n\nOptions:\n"
                            + items.item_key.map(options).fillna("")).str.strip()
        items["reference_answer"] = items.reference_answer.astype(str).str.strip()
        items["dataset"] = items.dataset.astype(str).str.strip().str.lower()
        items = items.assign(
            raw_item_id=items.dataset + ":" + items.qid.astype(str),
            features=items.dataset.map(lambda value: {"source_dataset": value}),
            grading_criterion=items.reference_answer.map(
                lambda answer: {"rule": self.grading["rule"], **({"reference_answer": answer} if answer else {})}
            ),
            verifier=ExactMatcher(spec=self.grading["verifiers"]["exact_answer"]["spec"]),
        )

        # 3. Compare each released answer with its own question's reference option.
        responses = attempts.merge(
            items[["item_key", "reference_answer"]], on="item_key", how="left", sort=False, validate="many_to_one"
        )
        responses["subject_key"] = responses.method.map(methods)
        responses["response"] = (responses.reference_answer.ne("") & responses.final_answer.astype(str).str.strip().eq(
            responses.reference_answer
        )).astype(float)
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        traces = responses.loc[responses.reasoning.notna(), ["response_key", "reasoning"]].rename(columns={"reasoning": "trace"})
        traces["trace"] = traces.trace.map(
            lambda value: "\n".join(map(str, value)) if isinstance(value, list) else str(value)
        )
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ConfAgents(__file__).main_from_args()
