#!/usr/bin/env python3
"""Curate BBQ's released predictions, complete inputs, and original answer matching."""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BBQ(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read JSONL and CSV tables, retaining native records and source positions.
        root = self.raw_dir / self.build_parameters["paths"]["release"]
        inputs, outputs = [], []
        for folder, frames in [(root / "data", inputs), (root / "results/UnifiedQA", outputs)]:
            for path in sorted(folder.glob("*.jsonl")):
                frame = pd.read_json(path, lines=True, dtype=False)
                frame["record"] = frame.to_dict("records")
                frames.append(frame.rename_axis("source_row").reset_index().assign(
                    source_file=str(path.relative_to(self.raw_dir))))
        definitions, generations = pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)
        keys = ["category", "example_id"]
        if definitions.duplicated(keys).any() or generations.duplicated(keys).any():
            raise ValueError("BBQ item keys must include category and be unique")
        matched = definitions.merge(generations, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not matched._merge.eq("both").all():
            raise ValueError("Prediction records and item definitions have different keys")
        # The result files retain older answer_info annotations. Keep both versions;
        # the original grader uses the result-file version, not the later annotations.
        for column in ["context", "question", "ans0", "ans1", "ans2", "label"]:
            if not matched[column + "_x"].eq(matched[column + "_y"]).all():
                raise ValueError("Source input or reference differs: " + column)
        generative_columns = [name for name in self.build_parameters["models"] if name in generations]
        text = generations.melt(id_vars=keys + ["source_file", "source_row", "record"],
            value_vars=generative_columns, var_name="subject_key", value_name="prediction")

        path = root / "results/RoBERTa_and_DeBERTaV3/df_bbq.csv"
        encoders = pd.read_csv(path, dtype=str, keep_default_na=False)
        encoders["record"] = encoders.to_dict("records")
        encoders = encoders.rename_axis("source_row").reset_index().assign(
            source_file=str(path.relative_to(self.raw_dir))).rename(
                columns={"index": "example_id", "cat": "category", "model": "subject_key"})
        encoders["example_id"] = encoders.example_id.astype(int)
        logits = encoders[["ans0", "ans1", "ans2"]].astype(float)
        if not logits.map(lambda value: float("-inf") < value < float("inf")).all().all():
            raise ValueError("Encoder logits must be finite")
        winning = logits.eq(logits.max(axis=1), axis=0)
        encoders["choice"] = winning.idxmax(axis=1).where(winning.sum(axis=1).eq(1))
        answers = generations.melt(id_vars=keys, value_vars=["ans0", "ans1", "ans2"],
            var_name="choice", value_name="prediction")
        encoders = encoders.drop(columns=["ans0", "ans1", "ans2"]).merge(
            answers, on=keys + ["choice"], how="left", validate="many_to_one")
        encoders["prediction"] = encoders.prediction.str.lower()
        observations = pd.concat([text, encoders[text.columns]], ignore_index=True)
        observations = observations.merge(generations.drop(columns=generative_columns).rename(columns={
            "record": "grading_record", "source_file": "grading_source_file", "source_row": "grading_source_row"}),
            on=keys, how="left", validate="many_to_one")
        observations = observations.merge(definitions[keys + ["record", "source_file", "source_row"]].rename(columns={
            "record": "definition_record", "source_file": "definition_source_file", "source_row": "definition_source_row"}),
            on=keys, how="left", validate="many_to_one")
        if observations.grading_record.isna().any():
            raise ValueError("An encoder result has no matching category/item definition")
        observations["format"] = observations.subject_key.map(self.build_parameters["formats"])
        if observations.format.isna().any():
            raise ValueError("A released model has no documented input format")

        # 2. Keep one question-only observation per context pair and retain its copy.
        paired = generations.assign(pair=generations.example_id // 2).groupby(["category", "pair"])
        same = ["question", "ans0", "ans1", "ans2", "question_index", "question_polarity",
                "unifiedqa-t5-11b_pred_qonly"]
        if not (paired.size().eq(2).all() and paired[same].nunique().eq(1).all().all()
                and paired.context_condition.nunique().eq(2).all()
                and generations.context_condition.eq("ambig").eq(generations.example_id.mod(2).eq(0)).all()):
            raise ValueError("Question-only copies no longer follow the released paired layout")
        copies = generations.loc[generations.context_condition.eq("disambig"),
            keys + ["record", "source_file", "source_row"]].copy()
        copies["example_id"] -= 1
        copies["copy"] = [dict(file=row.source_file, row=row.source_row, record=row.record) for row in copies.itertuples()]
        observations = observations.loc[~(observations.format.eq("question_only")
            & observations.context_condition.eq("disambig"))].merge(
                copies[keys + ["copy"]], on=keys, how="left", validate="many_to_one")

        # 3. Apply the author's ordered matching rules; unmatched outputs stay ungraded.
        prediction = observations.prediction.astype("string").str.replace("pantsu$", "pantsuit", regex=True)
        prediction = prediction.str.replace(r"\.$", "", regex=True).str.replace("o'brien", "obrien", regex=False).str.lower()
        choice = pd.Series(pd.NA, index=observations.index, dtype="Int64")
        for index in range(3):
            answer = observations[f"ans{index}"].str.replace("}", "", regex=False).str.replace(r"\.$", "", regex=True).str.lower()
            match = prediction.str.strip(" \t\r\n").eq(answer.str.strip(" \t\r\n"))
            choice.loc[choice.isna() & match.fillna(False)] = index
        for index in range(3):
            names = observations.answer_info.map(lambda value: value[f"ans{index}"][0]).str.lower()
            # stringr::word(..., 1, 2) is missing when fewer than two words exist.
            patterns = names.str.split(" ", n=2).str[:2].str.join(" ").where(names.str.contains(" ", regex=False))
            match = pd.Series([isinstance(text, str) and isinstance(pattern, str) and re.search(pattern, text) is not None
                for text, pattern in zip(prediction, patterns)], index=observations.index)
            choice.loc[choice.isna() & match] = index
        observations["parsed_choice"] = choice
        observations["response"] = choice.eq(observations.label).astype("Float64").where(choice.notna())

        # 4. Describe actual input components and keep grading annotations out of them.
        subjects = observations[["subject_key", "format"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.map(self.build_parameters["models"])
        subjects["features"] = [dict(harness="BBQ", source_model=row.subject_key, input_format=row.format)
            for row in subjects.itertuples()]
        observations["item_key"] = (observations.category + ":" + observations.example_id.astype(str)
                                     + ":" + observations.format)
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.category + ":" + items.example_id.astype(str)
        items["content"] = [json.dumps(dict(question=row.question, options=[row.ans0, row.ans1, row.ans2],
            input_format=row.format, **({} if row.format == "question_only" else {"context": row.context})),
            ensure_ascii=False, sort_keys=True) for row in items.itertuples()]
        items["features"] = [dict(category=row.category, question_polarity=row.question_polarity,
            context_condition="not_provided" if row.format == "question_only" else row.context_condition)
            for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=[row.ans0, row.ans1, row.ans2][row.label],
            rule=json.dumps(dict(description=self.grading["rule"], correct_option_index=row.label,
                matching_names=[row.answer_info[f"ans{index}"][0] for index in range(3)]), ensure_ascii=False, sort_keys=True))
            for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["answer"], sort_keys=True))] * len(items)

        # 5. Preserve full outputs/logits, both source annotation versions, and copied rows.
        observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str) + ":" + observations.subject_key
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(
            source=dict(file=row.source_file, row=row.source_row, record=row.record),
            definition=dict(file=row.definition_source_file, row=row.definition_source_row, record=row.definition_record),
            grading_input=dict(file=row.grading_source_file, row=row.grading_source_row, record=row.grading_record),
            question_only_copy=row.copy if row.format == "question_only" else None,
            parsed_choice=None if pd.isna(row.parsed_choice) else int(row.parsed_choice)),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BBQ(__file__).main_from_args()
