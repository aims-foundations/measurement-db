#!/usr/bin/env python3
"""Curate AI2D's released outputs, diagram assets, and exact-match grades."""

import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AI2DTest(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, parsing = parameters["layout"], parameters["parsing"]
        release = self.raw_dir / layout["release"]
        letters = list(parsing["letters"])
        padding = json.loads(parsing["padding_values"])

        # 1. Concatenate maintained primary exports, retaining every native record.
        frames = []
        for path in sorted(release.glob(layout["predictions"])):
            table = pd.read_excel(path, keep_default_na=False)
            table["native_record"] = table.to_dict("records")
            frames.append(table.assign(source_file=str(path.relative_to(self.raw_dir)),
                source_row=table.index, subject_key=path.name.removesuffix(parsing["filename_suffix"])))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda frame: frame.index)
        if responses.duplicated(["subject_key", "index"]).any():
            raise ValueError("Duplicate primary model/question records")

        # 2. Join questions to diagrams; normalize only known padding cells.
        items = pd.read_csv(self.raw_dir / layout["tasks"], sep="\t", keep_default_na=False)
        source_columns = items.columns.drop("image").to_list()
        for table in (responses, items):
            table[letters] = table[letters].replace(padding, "")
        responses = responses.merge(items[source_columns].assign(item_key=items['index']),
                                     on=source_columns, how="left", validate="many_to_one")
        if responses.item_key.isna().any():
            raise ValueError("A released question, option, reference, or image locator differs from the task bank")
        items["item_key"] = items['index']
        items["raw_item_id"] = parsing["item_prefix"] + items['index'].astype(str)
        choices = items.melt(id_vars="item_key", value_vars=letters, var_name="letter", value_name="option")
        choices = choices.loc[choices.option.ne("")].copy()
        choices["line"] = choices.letter + ". " + choices.option + "\n"
        option_text = choices.groupby("item_key", sort=False).line.sum()
        items["text"] = (parameters["prompt"]["question_prefix"] + items.question + "\n"
                         + parameters["prompt"]["options_prefix"] + items.item_key.map(option_text)
                         + parameters["prompt"]["suffix"])
        items["image_bytes"] = items.image.map(lambda value: base64.b64decode(value, validate=True))
        items["asset_path"] = items.image_bytes.map(lambda value: "images/" + hashlib.sha256(value).hexdigest() + ".jpg")
        items["content"] = [json.dumps({"multimedia_elements": [
            {"content_type": "image/jpeg", "location": row.asset_path},
            {"content_type": "text/plain", "text": row.text}]}, ensure_ascii=False) for row in items.itertuples()]
        items["attachments"] = [[{"data": row.image_bytes, "path": row.asset_path,
                                   "media_type": "image/jpeg", "role": "input"}] for row in items.itertuples()]

        # 3. Apply the frozen upstream matcher with token and option tables.
        punctuation = parsing["punctuation"]
        words = responses[["response_key", "item_key"]].assign(word=responses.prediction.astype(str)
            .str.translate(str.maketrans(punctuation, " " * len(punctuation))).str.split()).explode("word")
        words = words.drop_duplicates(["response_key", "word"])
        matches = words.merge(choices[["item_key", "letter"]], left_on=["item_key", "word"],
                              right_on=["item_key", "letter"], how="inner", validate="many_to_one")
        matches = matches.groupby("response_key").letter.agg(["size", "first"])
        option = responses.response_key.map(matches['first'].where(matches['size'].eq(1)))
        has_choice = responses.response_key.isin(matches.index)
        has_z = responses.response_key.isin(words.loc[words.word.eq("Z"), "response_key"])
        option = option.mask(~has_choice & has_z, "Z")
        refusal = responses.prediction.astype(str).str.contains(
            "|".join(re.escape(value) for value in parameters["refusals"].values()), regex=True)
        option = option.mask(refusal, "Z")
        unavailable = (responses.prediction.eq("") | responses.prediction.astype(str)
                       .str.contains(parsing["api_failure"], regex=False))
        remaining = responses.loc[option.isna() & ~unavailable, ["response_key", "item_key", "prediction"]]
        text_matches = remaining.merge(choices, on="item_key", how="inner", validate="many_to_many")
        text_matches = text_matches.loc[[str(choice).lower() in str(answer).lower()
                                         for choice, answer in zip(text_matches.option, text_matches.prediction)]]
        text_matches = text_matches.groupby("response_key").letter.agg(["size", "first"])
        fallback = responses.response_key.map(text_matches['first'].where(text_matches['size'].eq(1)))
        responses["extracted_answer"] = option.fillna(fallback).fillna("Z")
        responses["response"] = responses.extracted_answer.eq(responses.answer).astype(float).mask(unavailable)
        responses["grade_status"] = unavailable.map({True: "unavailable_output", False: "derived_exact_matching"})

        # 4. Retain the published GPT-4o grading record and require exact agreement.
        published = pd.read_excel(release / layout["published_grades"], keep_default_na=False)
        published["published_record"] = published.to_dict("records")
        published["subject_key"] = parameters["target"]["published_model"]
        responses = responses.merge(published[["subject_key", "index", "prediction", "hit", "published_record"]],
                                     on=["subject_key", "index", "prediction"], how="left", validate="one_to_one")
        selected = responses.subject_key.eq(parameters["target"]["published_model"])
        if responses.loc[selected, "hit"].isna().any() or not responses.loc[selected, "hit"].eq(responses.loc[selected, "response"]).all():
            raise ValueError("The frozen matcher must preserve every published GPT-4o grade")
        responses.loc[selected, "grade_status"] = "published_exact_matching"

        # 5. Project subjects, items, observations, and full source-linked traces.
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects["features"] = [{"model_identifier": quote(name, safe=" /-._"),
                                  "harness": parameters["target"]["harness"]} for name in subjects.subject_key]
        items["features"] = [{"category": row.category, "abc_label": str(row.abcLabel).lower(),
                               "source_image_path": row.image_path} for row in items.itertuples()]
        items["grading_criterion"] = [{"reference_answer": answer, "rule": self.grading["rule"]} for answer in items.answer]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_matching"], sort_keys=True))
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": int(row.source_row),
            "native_record": row.native_record, "grade_status": row.grade_status,
            "extracted_answer": None if row.grade_status == "unavailable_output" else row.extracted_answer,
            "published_record": row.published_record if isinstance(row.published_record, dict) else None},
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AI2DTest(__file__).main_from_args()
