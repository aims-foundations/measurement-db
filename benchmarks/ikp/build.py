#!/usr/bin/env python3
"""Curate IKP's released per-probe verdicts against its authoritative v9 question bank."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class IKPBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results", "probes")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the authoritative probe bank and normalize the per-model result files.
        root = self.raw_dir / "ikp/data"
        probes = pd.read_json(root / "probes/final_probe_set_v9.json", dtype=False)
        if probes.id.isna().any() or probes.id.duplicated().any():
            raise ValueError("IKP probe IDs must be nonempty and unique")
        files = sorted((root / "results").glob("*.json"))
        records = [json.loads(path.read_text()) for path in files]
        # Summary files coexist with run files; only run files have both these fields.
        runs = pd.json_normalize(
            [record for record in records if isinstance(record, dict)], max_level=0
        ).reindex(columns=["model_name", "results"])
        runs = runs.loc[runs.model_name.notna() & runs.results.map(lambda value: isinstance(value, list))]
        attempts = pd.json_normalize(
            runs.to_dict("records"), record_path="results", meta="model_name", max_level=0
        ).reindex(columns=["model_name", "probe_id", "question", "model_response", "correct"])
        attempts["response_key"] = attempts.index

        # 2. Match the question actually asked, excluding superseded prompts and API failures.
        observations = attempts.merge(
            probes[["id", "question"]].rename(columns={"id": "probe_id", "question": "current_question"}),
            on="probe_id", how="inner", sort=False, validate="many_to_one"
        )
        observations = observations.loc[
            observations.question.eq(observations.current_question)
            & observations.model_response.fillna("").str.strip().ne("")
            & observations.correct.notna()
        ].copy()
        if not observations.correct.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("IKP correctness must be an explicit boolean")

        # 3. Keep the complete probe bank, including probes without eligible observations.
        items = probes.rename(columns={"id": "item_key", "question": "content"}).assign(
            raw_item_id=lambda frame: frame.item_key,
            grading_criterion=lambda frame: frame.answer.map(
                lambda answer: {"rule": self.grading["rule"], "reference_answer": answer}
            ),
            verifier=Judge(spec=json.dumps(self.grading["verifiers"]["judge"], sort_keys=True), judged_by="llm"),
        )
        subjects = runs[["model_name"]].drop_duplicates().rename(columns={"model_name": "raw_label"})
        subjects["subject_key"] = subjects.raw_label
        responses = observations.rename(columns={"model_name": "subject_key", "probe_id": "item_key"})
        responses["response"] = responses.correct.astype(float)
        traces = responses[["response_key", "model_response"]].rename(columns={"model_response": "trace"})
        return {
            "subjects": subjects[["subject_key", "raw_label"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    IKPBuild(__file__).main_from_args()
