#!/usr/bin/env python3
"""Curate complete Legal RAG Bench observations and distinct grading protocols."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class LegalRagBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "tasks", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read released observations; keep every native field in the trace.
        parameters = self.build_parameters
        native = pd.read_json(self.raw_dir / parameters["paths"]["results"], lines=True,
                              dtype=False, convert_dates=False)
        native["source_row"] = native.index
        native["native_record"] = native.drop(columns="source_row").to_dict("records")
        judgments = pd.json_normalize(native.judge_verdict).rename(columns={
            "correct": "correctness", "grounded": "groundedness"})
        observations = native.join(judgments[["correctness", "groundedness"]])
        observations["retrieval"] = observations.gold_id_in_context

        # 2. Each generator/embedding combination defines one evaluated RAG pipeline.
        subjects = native[["model_name", "generative_model", "embedding_model"]].drop_duplicates()
        subjects = subjects.rename(columns={"model_name": "subject_key"})
        subjects["raw_label"] = subjects.generative_model.map(parameters["generators"])
        profiles = pd.DataFrame({
            "embedding_model": subjects.embedding_model.map(parameters["embedders"]),
            "generator_identifier": subjects.generative_model.map(parameters["generator_identifiers"]),
        })
        subjects["features"] = [{**parameters["subject_features"], **row}
                                for row in profiles.to_dict("records")]

        # 3. Melt the three grades; these describe the same answer, not new executions.
        responses = observations.melt(id_vars=["model_name", "question_id", "source_row", "native_record"],
                                      value_vars=list(parameters["criteria"]),
                                      var_name="criterion", value_name="response")
        responses = responses.rename(columns={"model_name": "subject_key"})
        responses["response_key"] = responses.index
        responses["response"] = responses.response.astype("Float64")

        # 4. Attach the complete corpus and an explicit grading protocol to each question.
        questions = pd.read_json(self.raw_dir / parameters["paths"]["questions"], lines=True,
                                 dtype=False, convert_dates=False).rename(columns={"id": "question_id"})
        questions["question_id"] = questions.question_id.astype(int)
        items = questions.merge(pd.DataFrame({"criterion": list(parameters["criteria"])}), how="cross")
        items["item_key"] = items.index
        items["raw_item_id"] = items.question_id.astype(str) + ":" + items.criterion
        items["content"] = items.question
        items["grading_criterion"] = [
            {"reference_answer": row.answer, "rule": json.dumps({
                "criterion": row.criterion, "rule": parameters["criteria"][row.criterion],
                "relevant_passage_id": row.relevant_passage_id}, ensure_ascii=False)}
            for row in items.itertuples()
        ]
        items["verifier"] = [
            ExactMatcher(spec=json.dumps(self.grading["verifiers"]["retrieval"], sort_keys=True))
            if criterion == "retrieval" else Judge(judged_by="llm", spec=json.dumps(
                {**self.grading["verifiers"]["llm"], "criterion": criterion}, sort_keys=True))
            for criterion in items.criterion
        ]
        corpus = {"path": "corpus.jsonl", "role": "retrieval_corpus", "media_type": "application/x-ndjson",
                  "data": (self.raw_dir / parameters["paths"]["corpus"]).read_bytes()}
        items["attachments"] = [[corpus] for _ in items.index]

        # 5. Join grades to their protocol and retain all retrieved passages and judgments.
        responses = responses.merge(items[["question_id", "criterion", "item_key"]],
                                    on=["question_id", "criterion"], how="left", validate="many_to_one")
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": parameters["paths"]["results"],
                                      "source_row": row.source_row, "record": row.native_record},
                                     ensure_ascii=False, allow_nan=False)
                           for row in responses.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    LegalRagBench(__file__).main_from_args()
