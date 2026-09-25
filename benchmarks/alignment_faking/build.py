#!/usr/bin/env python3
"""Import released alignment-faking judgments and the original encoded transcripts."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class AlignmentFaking(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("transcripts", "article")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Concatenate the native Parquet tables, retaining every original field.
        frames = []
        for path in sorted((self.raw_dir / layout["transcripts"]).glob("*.parquet")):
            frame = pd.read_parquet(path)
            records = frame.astype(object).where(frame.notna(), None).to_dict("records")
            frames.append(frame.assign(native_record=records, source_row=frame.index,
                                       source_file=str(path.relative_to(self.raw_dir)),
                                       source_run=path.stem))
        observations = pd.concat(frames, ignore_index=True)
        observations["response_key"] = observations.index
        labels = observations.alignment_faking
        if not labels.dropna().map(type).eq(bool).all():
            raise ValueError("Alignment-faking judgments must be boolean or missing")
        observations["response"] = labels.astype("Float64")

        # 2. Recover the published prompt variants from all three body rows of Appendix A.
        article = BeautifulSoup((self.raw_dir / layout["article"]).read_text(), "html.parser")
        rows = article.select_one(parameters["prompt_table"]["selector"]).find_all("tr")
        cells = pd.DataFrame([[cell.get_text("\n", strip=True) for cell in row.find_all("td", recursive=False)]
                              for row in rows[1:]])
        prompts = {name: "\n\n".join(text for text in cells[int(column)] if text)
                   for name, column in parameters["prompt_columns"].items()}
        if set(observations.system_prompt.dropna()) - prompts.keys():
            raise ValueError("An unknown released prompt label needs source review")

        # 3. Each released run and training checkpoint is a distinct system configuration.
        observations["subject_key"] = observations.source_run + "/step/" + observations.step.astype(str)
        subjects = observations.drop_duplicates("subject_key")[[
            "subject_key", "source_run", "initialised_from", "experiment_name", "intervention", "step"]].copy()
        subjects["raw_label"] = subjects.source_run
        subjects["features"] = [dict(parameters["subject"], model_identifier=row.source_run,
                                     initialised_from=row.initialised_from, experiment_name=row.experiment_name,
                                     intervention=row.intervention, training_step=int(row.step))
                                for row in subjects.itertuples()]

        # 4. Preserve the question, user tier and known prompt context; never guess a missing variant.
        context = ["user_input", "is_free_tier_user", "system_prompt"]
        items = observations[context].drop_duplicates().reset_index(drop=True)
        items["content"] = [json.dumps({"user_input": row.user_input,
                                       "is_free_tier_user": bool(row.is_free_tier_user),
                                       "system_prompt_name": None if pd.isna(row.system_prompt) else row.system_prompt,
                                       "published_system_prompt": prompts.get(row.system_prompt)},
                                      ensure_ascii=False, sort_keys=True, allow_nan=False)
                            for row in items.itertuples()]
        items["item_key"] = items.content.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        items["raw_item_id"] = items.item_key
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in items.index]
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["alignment_faking"], sort_keys=True),
                                  judge=self.grading["verifiers"]["alignment_faking"]["judge"], judged_by="llm")
        observations = observations.merge(items[context + ["item_key"]], on=context, how="left", validate="many_to_one")

        # 5. Link every original record to its grade without decoding or clipping the transcript.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "record": row.native_record}, ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AlignmentFaking(__file__).main_from_args()
