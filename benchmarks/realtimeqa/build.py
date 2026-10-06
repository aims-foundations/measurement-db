"""Tabulate RealTime QA's released weekly predictions and native grading rules."""

import itertools
import json
import re
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class RealTimeQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]
        raw = self.raw_dir / layout["release"]

        # 1. Load weekly prediction tables and their original question files.
        predictions, question_tables = [], {}
        for path in sorted((raw / "baseline_results").rglob("*.jsonl")):
            match = re.fullmatch(parameters["patterns"]["weekly_prediction"], path.name)
            if match is None:  # Cumulative exports remain in raw; they overlap weekly releases.
                continue
            date, variant, token = match.groups()
            if date in parameters["unavailable_questions"]:
                continue
            stem = date + "_qa" + (variant or "")
            alternatives = [raw / f"past/{date[:4]}/{stem}.jsonl",
                            raw / f"past/{date[:4]}/{stem}_public.jsonl",
                            raw / f"latest/{stem}_public.jsonl"]
            if stem in parameters["question_aliases"]:
                alternatives.append(raw / parameters["question_aliases"][stem])
            question_path = next((p for p in alternatives if p.is_file()), None)
            if question_path is None:
                raise ValueError(f"No original question file for {path.name}")
            question_file = str(question_path.relative_to(raw))
            if question_file not in question_tables:
                records = [json.loads(line) for line in question_path.read_text().splitlines() if line.strip()]
                questions = pd.json_normalize(records, max_level=0)
                questions["question_record"] = records
                questions["question_file"] = question_file
                questions["question_row"] = questions.index
                question_tables[question_file] = questions
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            frame = pd.json_normalize(records, max_level=0)
            frame["native_record"] = records
            frame["source_file"] = str(path.relative_to(raw))
            frame["source_row"] = frame.index
            frame["question_file"] = question_file
            frame["model_token"] = token
            frame["mode"] = "generation" if token.endswith("_gen") else "multiple_choice"
            questions = question_tables[question_file]
            selected = questions.loc[questions.question_source.eq("CNN")] if token.startswith("cnn_") else questions
            if frame.question_id.tolist() != selected.question_id.tolist():
                raise ValueError(f"Question IDs or their order disagree for {path.name}")
            predictions.append(frame)
        attempts = pd.concat(predictions, ignore_index=True)
        questions = pd.concat(question_tables.values(), ignore_index=True)

        # 2. Join on both the original file and task ID; keep every recorded attempt.
        attempts = attempts.merge(questions, on=["question_file", "question_id"], how="left",
                                  validate="many_to_one", indicator=True)
        if not attempts._merge.eq("both").all():
            raise ValueError("A prediction has no corresponding task")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["grade_status"] = "graded"
        generation = attempts["mode"].eq("generation")
        excluded = generation & attempts.question_sentence.str.lower().str.strip().str[-10:].str.contains("except")
        invalid = generation & ~attempts.prediction.map(lambda value: isinstance(value, str))
        attempts.loc[invalid, "grade_status"] = "invalid_generation_type"
        attempts.loc[excluded, "grade_status"] = "excluded_by_native_grader"
        attempts.loc[attempts.answer.isna(), "grade_status"] = "reference_unavailable"

        # 3. Apply native list equality for MC and normalized exact match for generation.
        attempts["response"] = attempts.prediction.eq(attempts.answer).astype(float)
        generated = attempts.loc[generation & attempts.grade_status.eq("graded")].copy()
        generated["reference"] = [[" ".join(order) for order in itertools.permutations(
            [choices[int(index)] for index in answer])]
            for choices, answer in zip(generated.choices, generated.answer)]
        generated = generated.explode("reference")
        for column in ("prediction", "reference"):
            generated[column] = (generated[column].str.lower()
                .str.translate(str.maketrans("", "", string.punctuation))
                .str.replace(parameters["patterns"]["counter"], "", regex=True)
                .str.split().str.join(" "))
        generated["exact_match"] = generated.prediction.eq(generated.reference)
        exact = generated.groupby("response_key").exact_match.max()
        attempts.loc[generation, "response"] = attempts.loc[generation, "response_key"].map(exact).astype(float)
        attempts.loc[~attempts.grade_status.eq("graded"), "response"] = None

        # 4. Keep literal model configurations and mode-specific task/grading definitions.
        attempts["subject_key"] = attempts.model_token
        subjects = attempts[["subject_key", "mode"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_token=row.subject_key,
            answer_mode=row.mode) for row in subjects.itertuples()]
        attempts["content"] = [json.dumps(dict(question_date=row.question_date,
            question_sentence=row.question_sentence,
            **({"choices": row.choices} if row.mode == "multiple_choice" else {})), ensure_ascii=False)
            for row in attempts.itertuples()]
        attempts["criterion"] = [json.dumps(dict(
            reference_answer=json.dumps(dict(indices=row.answer, choices=row.choices), ensure_ascii=False)
                if isinstance(row.answer, list) else None,
            rule=self.grading["verifiers"][row.mode]["rule"]), ensure_ascii=False, sort_keys=True)
            for row in attempts.itertuples()]
        keys = ["content", "criterion", "mode"]
        items = attempts.drop_duplicates(keys).copy()
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.question_id + "#" + items["mode"]
        items["features"] = [dict(question_source=row.question_source, question_url=row.question_url,
            answer_mode=row.mode) for row in items.itertuples()]
        items["grading_criterion"] = items.criterion.map(json.loads)
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"][mode], sort_keys=True))
                             for mode in items["mode"]]
        attempts = attempts.merge(items[keys + ["item_key"]], on=keys, how="left", validate="many_to_one")
        attempts["test_condition"] = parameters["labels"]["condition"]

        # 5. Preserve complete predictions and source questions, including ungraded attempts.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            question_file=row.question_file, question_row=row.question_row, model_token=row.model_token,
            mode=row.mode, grade_status=row.grade_status, native_record=row.native_record,
            question_record=row.question_record), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    RealTimeQA(__file__).main_from_args()
