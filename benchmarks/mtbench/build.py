#!/usr/bin/env python3
"""Tabulate released MT-Bench conversations and judgments without running a model."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MTBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths, labels = self.build_parameters["paths"], self.build_parameters["labels"]

        # 1. Load the original JSONL tables and retain every native record alongside its location.
        loaded = {}
        for name, pattern in paths.items():
            parts = []
            for path in sorted(self.raw_dir.glob(pattern)):
                frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
                frame["source_record"] = pd.Series(path.read_text().splitlines()).map(json.loads)
                parts.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=range(len(frame)), file_key=path.stem))
            loaded[name] = pd.concat(parts, ignore_index=True)
        questions, answers, references, prompts = (loaded[name] for name in ["questions", "answers", "references", "prompts"])
        records = loaded["judgments"].assign(judge_model=lambda x: x.judge.str[0], template=lambda x: x.judge.str[1])
        records = records.rename(columns={"source_record": "judgment_record", "source_row": "judgment_row"})

        # 2. Join source keys exactly; verify that the recorded judge saw these same answers and references.
        for name, table, keys, right_keys in [
            ("question", questions, ["question_id"], ["question_id"]),
            ("answer", answers, ["model", "question_id"], ["file_key", "question_id"]),
            ("prompt", prompts, ["template"], ["name"]),
            ("reference", references, ["question_id"], ["question_id"]),
        ]:
            selected = table[right_keys + ["source_record", "source_file", "source_row"]].rename(columns={
                "source_record": name + "_record", "source_file": name + "_file", "source_row": name + "_row"})
            records = records.merge(selected, left_on=keys, right_on=right_keys, how="left", validate="many_to_one", suffixes=("", "_joined"))
            if name != "reference" and records[name + "_record"].isna().any():
                raise ValueError("An MT-Bench judgment has no matching " + name)
        if records[["model", "question_id", "turn"]].duplicated().any() or not records.turn.isin([1, 2]).all():
            raise ValueError("MT-Bench requires one published judgment for each model/question/turn")
        for row in records.itertuples():
            if len(row.answer_record["choices"]) != 1 or row.answer_record["choices"][0]["index"] != 0:
                raise ValueError("MT-Bench single-answer judgments require the published choice zero")
            question, answer = row.question_record["turns"], row.answer_record["choices"][0]["turns"]
            reference = row.reference_record["choices"][0]["turns"] if isinstance(row.reference_record, dict) else [None, None]
            rendered = row.prompt_record["prompt_template"].format(question=question[0], question_1=question[0], question_2=question[1],
                answer=answer[0], answer_1=answer[0], answer_2=answer[1], ref_answer_1=reference[0], ref_answer_2=reference[1])
            if rendered != row.user_prompt or row.judge_model != self.grading["verifiers"]["published_judge"]["model_label"]:
                raise ValueError("MT-Bench native grading prompt disagrees with the joined conversation or judge")

        # 3. Preserve literal model labels and the actual stimulus at each turn, excluding the answer being scored.
        records["internal_model"] = records.answer_record.map(lambda x: x["model_id"])
        subjects = records[["model", "internal_model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(recorded_model_label=row.subject_key, internal_model_label=row.internal_model,
            settings_status=labels["settings_status"]) for row in subjects.itertuples()]
        records["item_key"] = records.question_id.astype(str) + "_turn" + records.turn.astype(str)
        records.loc[records.turn.eq(2), "item_key"] += "@" + records.model
        records["content"] = [row.question_record["turns"][0] if row.turn == 1 else json.dumps([
            dict(role="user", content=row.question_record["turns"][0]),
            dict(role="assistant", content=row.answer_record["choices"][0]["turns"][0]),
            dict(role="user", content=row.question_record["turns"][1])], ensure_ascii=False) for row in records.itertuples()]
        items = records.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(question_id=row.question_id, turn=row.turn, category=row.question_record["category"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.question_record.get("reference", [None, None])[row.turn - 1] or None,
            rule=json.dumps(dict(rule=self.grading["rule"], judge_reference=row.reference_record if "math" in row.template else None),
                            ensure_ascii=False, allow_nan=False)) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(dict(**self.grading["verifiers"]["published_judge"], prompt=row.prompt_record),
            ensure_ascii=False), judged_by="llm") for row in items.itertuples()]

        # 4. Preserve fractional ratings; the upstream unparseable sentinel becomes an explicit null grade.
        if not records.score.map(lambda score: type(score) in (int, float) and (score == -1 or 1 <= score <= 10)).all():
            raise ValueError("MT-Bench requires a finite rating in [1, 10], or the native -1 sentinel")
        responses = records[["model", "item_key"]].rename(columns={"model": "subject_key"})
        responses["response_key"] = records.model + "/" + records.question_id.astype(str) + "/" + records.turn.astype(str)
        responses["response"] = records.score.mask(records.score.eq(-1))
        responses["trial"], responses["test_condition"] = 1, labels["condition"]

        # 5. Link full answers, judge explanations and source records without clipping or regrading.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(judgment=dict(file=paths["judgments"], row=row.judgment_row, record=row.judgment_record),
            **{name: dict(file=getattr(row, name + "_file"), row=int(getattr(row, name + "_row")), record=getattr(row, name + "_record"))
               if isinstance(getattr(row, name + "_record"), dict) else None for name in ["question", "answer", "prompt", "reference"]}),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses, "traces": traces}


if __name__ == "__main__":
    MTBench(__file__).main_from_args()
