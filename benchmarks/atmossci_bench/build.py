#!/usr/bin/env python3
"""Curate AtmosSci-Bench's recorded questions, model attempts and original judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class AtmosSciBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        release = self.raw_dir / self.build_parameters["layout"]["release"]

        # 1. Concatenate native exports, retaining their exact JSON text and row positions.
        frames = []
        for path in sorted((release / "output").glob("*/*/*/*.jsonl")):
            frame = pd.DataFrame({"native_json": path.read_text().split("\n")}).rename_axis("source_row").reset_index()
            frame = frame.loc[frame.native_json.str.strip().ne("")].reset_index(drop=True)
            if frame.empty:
                continue
            frame["record"] = frame.native_json.map(json.loads)
            frame["id"] = frame.record.map(lambda record: record["id"])
            frames.append(frame.assign(run=str(path.parent.relative_to(release)), kind=path.stem,
                source_file=str(path.relative_to(self.raw_dir)), subset=path.parent.parent.name))
        native = pd.concat(frames, ignore_index=True)
        attempts = native.loc[native.kind.eq("response")].drop(columns="kind")
        judgments = native.loc[native.kind.eq("evaluation")].drop(columns=["kind", "subset"])

        # 2. Collapse identical judgment copies, then retain every attempt in a one-to-one join.
        grouped = judgments.groupby(["run", "id"], sort=False)
        if grouped.native_json.nunique().gt(1).any():
            raise ValueError("Repeated AtmosSci judgment IDs contain different records")
        aliases = grouped.source_row.agg(list).rename("evaluation_rows")
        judgments = judgments.drop_duplicates(["run", "id"]).drop(columns="source_row").merge(
            aliases, on=["run", "id"], validate="one_to_one")
        observations = attempts.merge(judgments, on=["run", "id"], how="outer", validate="one_to_one",
            suffixes=("", "_evaluation"), indicator=True)
        if observations._merge.eq("right_only").any():
            raise ValueError("An AtmosSci judgment has no recorded model attempt")
        observations["response_key"] = observations.run + "/" + observations.id
        observations["response"] = observations.record_evaluation.map(
            lambda record: record["score"] if isinstance(record, dict) else None)

        # 3. Join recorded model configurations and preserve their native aliases and token limits.
        runs = pd.DataFrame({"path": sorted((release / "output").glob("*/*/*/metadata.json"))})
        runs["run"] = runs.path.map(lambda path: str(path.parent.relative_to(release)))
        runs["metadata_file"] = runs.path.map(lambda path: str(path.relative_to(self.raw_dir)))
        runs["metadata"] = runs.path.map(lambda path: json.loads(path.read_text()))
        runs["source_model"] = runs.path.map(lambda path: path.parent.name)
        runs["configuration"] = runs.metadata.map(lambda record: json.dumps({
            "model": {**record["model"], "details": {key: value for key, value in record["model"]["details"].items() if key != "gpu"}},
            "parameters": {key: record["parameters"][key] for key in ("max_tokens", "retries", "no_fallback")}}, sort_keys=True))
        runs["subject_key"] = runs.source_model + "/" + runs.configuration
        observations = observations.merge(runs.drop(columns="path"), on="run", how="left", validate="many_to_one")
        if observations.subject_key.isna().any():
            raise ValueError("An AtmosSci attempt has no recorded model configuration")
        subjects = observations.drop_duplicates("subject_key")[["subject_key", "source_model", "configuration", "metadata"]].copy()
        subjects["raw_label"] = subjects.metadata.map(lambda record: record["model"]["name"])
        subjects["features"] = [dict(self.build_parameters["subject"], model_identifier=row.source_model,
            provider=row.metadata["model"]["base"], api_model=row.metadata["model"]["details"]["model_name"],
            inference_configuration=row.configuration) for row in subjects.itertuples()]

        # 4. Use each recorded MCQ stimulus; the published bank supplies OEQ reference solutions.
        banks = []
        for path in sorted((release / "data/jsonl").glob("*.jsonl")):
            frame = pd.read_json(path, lines=True, dtype=False, precise_float=True)
            frame["bank_record"] = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
            banks.append(frame[["id", "bank_record"]].assign(subset=path.stem, question_file=str(path.relative_to(self.raw_dir))))
        observations = observations.merge(pd.concat(banks, ignore_index=True), on=["subset", "id"], how="left", validate="many_to_one")
        if observations.bank_record.isna().any():
            raise ValueError("An AtmosSci task ID has no published question-bank entry")
        observations["task"] = [row.record["question"] if isinstance(row.record["question"], dict)
                                else row.bank_record for row in observations.itertuples()]
        observations["question_type"] = observations.run.str.split("/").str[1]
        items = observations.copy()
        items["item_key"] = items.response_key
        items["raw_item_id"] = items.response_key
        items["content"] = items.task.map(lambda task: task["problem"].strip()
            + ("\n\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(task["options"])) if task.get("options") else "")
            + ("\n\nKnowledge:\n" + task["knowledge"] if task.get("knowledge") else ""))
        items["content"] = items.content.str.normalize("NFC").str.strip()
        items["features"] = items.question_type.map(lambda kind: {"question_type": kind})

        # 5. Describe the grading that was actually recorded, without recomputing any outcome.
        items["grading_criterion"] = [dict(rule=self.grading["rule"], reference_answer=json.dumps(
            row.record_evaluation["expected_answers"] if isinstance(row.record_evaluation, dict)
            else {"a": row.task["correct_option"]} if row.question_type == "MCQ" else row.task["answer"],
            ensure_ascii=False, sort_keys=True)) for row in items.itertuples()]
        items["verifier_spec"] = [json.dumps({**self.grading["verifiers"][row.question_type],
            "recorded_configuration": {key: (row.metadata.get("evaluation") or {}).get(key)
                for key in ("tolerance", "evaluators", "disabled_evaluators")},
            "reference_origin": "published_judgment" if isinstance(row.record_evaluation, dict) else "question_reference_without_judgment"},
            sort_keys=True) for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=row.verifier_spec) if row.question_type == "MCQ"
                             else Judge(spec=row.verifier_spec) for row in items.itertuples()]

        # 6. Keep complete native JSON, including upstream NaN tokens, inside a valid JSON trace.
        observations["item_key"] = observations.response_key
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            generation_json=row.native_json, evaluation_file=row.source_file_evaluation if isinstance(row.record_evaluation, dict) else None,
            evaluation_rows=row.evaluation_rows if isinstance(row.evaluation_rows, list) else [],
            evaluation_json=row.native_json_evaluation if isinstance(row.record_evaluation, dict) else None,
            metadata_file=row.metadata_file, metadata=row.metadata, question_file=row.question_file,
            question_bank_record=row.bank_record), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AtmosSciBench(__file__).main_from_args()
