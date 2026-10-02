#!/usr/bin/env python3
"""Curate ChartMuseum's released development attempts without inventing grades."""

import json
import mimetypes
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class ChartMuseum(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        protocol = self.grading["verifiers"]["equivalence"]

        # 1. Load the original question table and the two published output forms.
        items = pd.read_parquet(self.raw_dir / paths["questions"])
        items["native_record"] = items.to_dict("records")
        items["item_key"] = items.index
        outputs = pd.read_json(self.raw_dir / paths["full_output"], typ="series").to_frame("full_output")
        answers = pd.read_json(self.raw_dir / paths["short_output"], typ="series").to_frame("answer_only")
        if len(items) != len(outputs) or len(items) != len(answers):
            raise ValueError("ChartMuseum question and output arrays must have the same length")
        outputs = outputs.join(answers, validate="one_to_one").assign(item_key=lambda frame: frame.index)

        # 2. Reconcile the two formats using the released answer extractor. Row
        # order is the upstream join key; the hash is shared by several questions.
        extracted = (outputs.full_output + "</answer>").str.extract(
            protocol["answer_pattern"], flags=re.DOTALL, expand=False).str.strip().fillna("")
        short = (outputs.answer_only + "</answer>").str.extract(
            protocol["answer_pattern"], flags=re.DOTALL, expand=False).str.strip().fillna("")
        if not extracted.eq(short).all():
            raise ValueError("ChartMuseum full and answer-only exports disagree")
        outputs["extracted_answer"] = extracted
        attempts = outputs.merge(items[["item_key", "native_record"]], on="item_key", validate="one_to_one")

        # 3. Join each question to its original chart bytes and published prompt.
        images = items[["image"]].drop_duplicates()
        images["raw_path"] = images.image.str.replace(
            r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", regex=True)
        images["data"] = images.raw_path.map(lambda path: (self.raw_dir / paths["dataset"] / path).read_bytes())
        images["media_type"] = images.image.map(lambda path: mimetypes.guess_type(path)[0])
        if images.media_type.isna().any():
            raise ValueError("ChartMuseum image has an unknown media type")
        items = items.merge(images, on="image", how="left", validate="many_to_one")
        items["text"] = items.question.map(lambda question: parameters["prompts"]["question"].replace("[QUESTION]", question))
        items["content"] = [json.dumps({"multimedia_elements": [
            {"content_type": row.media_type, "location": row.image},
            {"content_type": "text/plain", "text": row.text}]}, ensure_ascii=False) for row in items.itertuples()]
        items["attachments"] = [[dict(data=row.data, path=row.image, media_type=row.media_type, role="input")]
                                 for row in items.itertuples()]

        # 4. Declare the original grading protocol; unavailable verdicts stay null.
        subjects = pd.DataFrame([dict(subject_key=0, raw_label=parameters["subject"]["label"],
                                      features=dict(harness=parameters["subject"]["harness"]))])
        items["raw_item_id"] = parameters["scope"]["split"] + "/" + items.item_key.astype(str)
        items["features"] = [dict(split=parameters["scope"]["split"], reasoning_type=row.reasoning_type,
            image_source=quote(row.source, safe="/:"), source_image_hash=row.hash) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=answer, rule=self.grading["rule"]) for answer in items.answer]
        items["verifier"] = Judge(spec=json.dumps(protocol, sort_keys=True), judge=protocol["model"], judged_by="llm")
        responses = attempts[["item_key"]].assign(response_key=attempts.index, subject_key=0,
                                                  response=None, test_condition="split=" + parameters["scope"]["split"])

        # 5. Retain both complete outputs and the exact source row for every attempt.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["full_output"], source_row=int(row.item_key),
            question_file=paths["questions"], question_record=row.native_record,
            full_output=row.full_output, answer_only_file=paths["short_output"], answer_only=row.answer_only,
            extracted_answer=row.extracted_answer, grade_status="upstream_judgment_unavailable"),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    ChartMuseum(__file__).main_from_args()
