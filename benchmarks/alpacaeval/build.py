#!/usr/bin/env python3
"""Curate AlpacaEval's released comparisons against its fixed GPT-4 reference."""

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class AlpacaEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "data_license")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read each primary annotation export, keeping original records and positions.
        parameters = self.build_parameters
        release = self.raw_dir / parameters["layout"]["release"]
        source = pd.concat([
            pd.DataFrame({"native_record": json.loads(path.read_text())}).assign(
                source_file=str(path.relative_to(self.raw_dir)), source_row=lambda frame: frame.index)
            for path in sorted(release.glob(parameters["layout"]["annotations"]))
        ], ignore_index=True)
        native = pd.json_normalize(source.native_record.tolist(), max_level=0)
        native = native.join(source).assign(response_key=lambda frame: frame.index)

        # 2. Join the fixed reference bank; its complete text is part of the grading rule.
        items = pd.read_json(release / parameters["layout"]["reference"], dtype=False, convert_dates=False)
        items = items.rename(columns={"output": "reference_output"}).assign(
            item_key=lambda frame: frame.index,
            raw_item_id=lambda frame: parameters["target"]["item_prefix"] + frame.index.astype(str))
        reference = parameters["target"]["reference"]
        if not (native.generator_1.eq(reference) | native.generator_2.eq(reference)).all():
            raise ValueError("Every annotation must compare against the declared reference model.")
        native["subject_key"] = native.generator_2.where(native.generator_1.eq(reference), native.generator_1)
        native["reference_output"] = native.output_1.where(native.generator_1.eq(reference), native.output_2)
        responses = native.merge(items[["instruction", "reference_output", "item_key"]],
                                 on=["instruction", "reference_output"], how="left", validate="many_to_one")
        if responses.item_key.isna().any():
            raise ValueError("An annotation does not match the released instruction and reference output.")

        # 3. Distinguish reported generators; retain templates in raw without assuming run settings.
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        # Escape separators in the compact feature encoding; raw_label keeps the original name.
        subjects["features"] = [
            {"model_identifier": quote(model, safe=" /-._"), "harness": parameters["target"]["harness"]}
            for model in subjects.subject_key
        ]
        items["content"] = items.instruction
        items["features"] = items[["dataset"]].to_dict("records")
        items["grading_criterion"] = [
            {"rule": json.dumps({"rule": self.grading["rule"], "reference_model": reference,
                                  "reference_output": output}, ensure_ascii=False)}
            for output in items.reference_output
        ]
        protocol = dict(self.grading["verifiers"]["weighted_preference"])
        protocol["prompt_template"] = (release / parameters["layout"]["judge_prompt"]).read_text()
        items["verifier"] = Judge(judged_by="llm", spec=json.dumps(protocol, ensure_ascii=False, sort_keys=True))

        # 4. Convert soft preferences to the evaluated generator's score; never clip invalid values.
        preference = pd.to_numeric(responses.preference, errors="raise")
        preference = preference.mask(preference.eq(float(parameters["parsing"]["unavailable_preference"])))
        if not (preference.isna() | preference.between(1, 2)).all():
            raise ValueError("Unexpected preference outside the declared native scale.")
        score = (preference - 1).where(responses.generator_1.eq(reference), 2 - preference)
        responses["response"] = score
        responses["interactors"] = parameters["target"]["interactors"]

        # 5. Retain every original field, including both outputs and the full judge completion.
        traces = source.assign(response_key=source.index)
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
                           for record in source.to_dict("records")]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "interactors"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    AlpacaEval(__file__).main_from_args()
