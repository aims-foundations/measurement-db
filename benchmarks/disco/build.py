#!/usr/bin/env python3
"""Curate published DIS-CO movie guesses and their original image/caption inputs."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DisCo(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "dataset")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, parsing = parameters["paths"], parameters["parsing"]

        # 1. Concatenate the frame bank and recover each scene's column order.
        frames = pd.concat([pd.read_parquet(path).assign(source_file=str(path.relative_to(self.raw_dir)),
            source_row=lambda frame: frame.index) for path in sorted(self.raw_dir.glob(paths["frames"]))],
            ignore_index=True).rename(columns={"Movie": "movie", "Frame_Type": "frame_type", "Scene_Number": "scene"})
        frames["frame_type"] = frames.frame_type.str.lower()
        frames = frames.sort_values(["movie", "frame_type", "scene", "Shot_Number"])
        frame_key = ["movie", "frame_type", "scene", "shot"]
        frames["shot"] = frames.groupby(frame_key[:-1]).cumcount() + 1
        frames["frame_key"] = frames.index
        if frames.duplicated(frame_key[:-1] + ["Shot_Number"]).any():
            raise ValueError("Repeated upstream frame coordinates")

        # 2. Unpivot every result sheet; retain literal 'null' outputs as attempts.
        sheets = []
        for source in sorted(name for name in self.source_files if name.startswith(paths["results"] + "/") and name.endswith(".xlsx")):
            path = self.raw_dir / source
            sheet = pd.read_excel(path, keep_default_na=False, dtype=object).rename(columns={"Scene": "scene"})
            sheet["source_row"] = sheet.index
            sheets.append(sheet.melt(id_vars=["scene", "source_row"], var_name="source_column",
                value_name="prediction").assign(source_file=str(path.relative_to(self.raw_dir)), filename=path.name))
        responses = pd.concat(sheets, ignore_index=True)
        responses = responses.join(responses.filename.str.extract(parsing["filename"]))
        responses["movie"] = responses.movie.replace(parameters["filename_escapes"], regex=True)
        responses["shot"] = responses.source_column.str.extract(parsing["column"])["shot"].astype(int)
        if responses[["movie", "frame_type", "model", "mode"]].isna().any().any():
            raise ValueError("Unexpected result filename")
        responses = responses.loc[responses.prediction.ne("")].copy()
        responses = responses.merge(frames.drop(columns="Image_File"), on=frame_key, how="left",
            validate="many_to_one", suffixes=("", "_frame"))
        if responses.frame_key.isna().any():
            raise ValueError("A saved prediction has no matching input frame")
        responses["item_key"] = responses.frame_key.astype(str) + "/" + responses["mode"]
        responses["response_key"] = responses.index
        responses["response"] = [float(prediction in answers) for prediction, answers
            in zip(responses.prediction, responses.Answer)]

        # 3. Keep image and caption questions distinct, with the actual image bytes.
        items = responses.drop_duplicates("item_key").copy()
        items["image_bytes"] = items.frame_key.map(frames.set_index("frame_key").Image_File).map(lambda value: value["bytes"])
        items["image_path"] = items.image_bytes.map(lambda value: "images/" + hashlib.sha256(value).hexdigest() + ".png")
        image_input = items["mode"].eq("single_image")
        items["content"] = parameters["prompts"]["caption_prompt"].replace("{caption}", "") + items.Caption
        items.loc[image_input, "content"] = [json.dumps({"multimedia_elements": [
            {"content_type": "text/plain", "text": parameters["prompts"]["image_prompt"]},
            {"content_type": "image/png", "location": path}]}, ensure_ascii=False)
            for path in items.loc[image_input, "image_path"]]
        items["attachments"] = [[dict(data=row.image_bytes, path=row.image_path, media_type="image/png", role="input")]
            if row.mode == "single_image" else [] for row in items.itertuples()]
        items["raw_item_id"] = items["movie"] + "/" + items.frame_type + "/" + items.scene.astype(str) + "/" + items.shot.astype(str) + "/" + items["mode"]
        items["features"] = [dict(query_mode=row.mode, frame_type=row.frame_type,
            input_scope=parameters["input_scope"]["note"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(list(answers), ensure_ascii=False),
            rule=self.grading["rule"]) for answers in items.Answer]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact"], sort_keys=True))] * len(items)

        # 4. Associate every saved title with its model, frame and source cell.
        subjects = responses[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [parameters["subject_features"] for _ in subjects.index]
        responses["subject_key"] = responses.model
        responses["test_condition"] = [json.dumps(dict(movie=row.movie, frame_type=row.frame_type,
            scene=int(row.scene), shot=int(row.shot), query_mode=row.mode), sort_keys=True) for row in responses.itertuples()]
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_column=row.source_column, scene=int(row.scene), prediction=row.prediction,
            frame_source_file=row.source_file_frame, frame_source_row=int(row.source_row_frame)),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DisCo(__file__).main_from_args()
