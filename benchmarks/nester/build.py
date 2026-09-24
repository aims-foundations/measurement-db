#!/usr/bin/env python3
"""Curate released NESTER human judgments, retaining every spreadsheet cell."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Nester(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "documentation", "paper")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load each native sheet, keeping original cells and Excel row positions.
        parameters = self.build_parameters
        sheets = []
        for source in parameters["models"]:
            sheet = pd.read_excel(self.raw_dir / source, keep_default_na=False)
            sheet["native_record"] = sheet.to_dict("records")
            sheet.columns = sheet.columns.str.strip()
            sheets.append(sheet.assign(subject_key=source, source_excel_row=sheet.index + 2))
        native = pd.concat(sheets, ignore_index=True)
        native["source_row"] = native.index

        # 2. Separate a visible annotation from the input using matching clean source IDs.
        annotated = native.prompt.str.contains(parameters["annotation"]["marker"], regex=False)
        clean = native.loc[~annotated, ["id", "prompt"]].drop_duplicates().rename(columns={"prompt": "content"})
        observations = native.merge(clean, on="id", how="left", validate="many_to_one")
        observations["input_reconstructed"] = observations.prompt.ne(observations.content)
        annotated = observations.input_reconstructed
        prefixes = observations.loc[annotated, "prompt"].str.split(parameters["annotation"]["marker"], n=1).str[0]
        if not prefixes.str.rstrip().eq(observations.loc[annotated, "content"].str.rstrip()).all():
            raise ValueError("Annotated source prompt does not match an independently released clean input")

        # 3. Each human criterion supplies its own grading protocol for the prompt.
        responses = observations.melt(
            id_vars=["subject_key", "source_row", "source_excel_row", "id", "content", "native_record", "input_reconstructed"],
            value_vars=list(parameters["criteria"]), var_name="grade_column", value_name="response")
        responses["criterion"] = responses.grade_column.map(parameters["criteria"])
        responses["response_key"] = responses.index
        items = responses.drop_duplicates(["content", "criterion"]).copy()
        items["item_key"] = range(len(items))
        items["raw_item_id"] = items.id + ":" + items.criterion
        rules = self.grading["verifiers"]["human"]["criteria"]
        items["grading_criterion"] = [{"rule": json.dumps({"criterion": criterion, "rule": rules[criterion]})}
                                      for criterion in items.criterion]
        items["verifier"] = [Judge(judged_by="human", spec=json.dumps(
            {**self.grading["verifiers"]["human"], "criterion": criterion}, sort_keys=True))
            for criterion in items.criterion]

        # 4. Link each source row to its model and protocol; keep blank output cells intact.
        subjects = pd.Series(parameters["models"], name="raw_label").rename_axis("subject_key").reset_index()
        responses = responses.merge(items[["content", "criterion", "item_key"]],
                                    on=["content", "criterion"], how="left", validate="many_to_one")
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.subject_key, "source_excel_row": row.source_excel_row,
                                      "record": row.native_record, "input_reconstructed": row.input_reconstructed},
                                     ensure_ascii=False, allow_nan=False)
                           for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    Nester(__file__).main_from_args()
