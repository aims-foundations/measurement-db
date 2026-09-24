#!/usr/bin/env python3
"""Import released CAF-Score decisions on BRACE without running audio models."""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Brace(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "annotations", "audio")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Put both annotation versions into one row per audio/caption pair.
        annotation_tables = {}
        for version in ("original", "processed"):
            frames = []
            for dataset, stem in parameters["datasets"].items():
                filename = stem + ".json" if version == "original" else "BRACE_" + stem + "_Processed.json"
                frame = pd.DataFrame(json.loads((self.raw_dir / layout[version] / filename).read_text()))
                frame = frame.drop(columns="references", errors="ignore").melt(
                    id_vars="file_name", var_name="pair_key", value_name=version + "_pair").dropna()
                frames.append(frame.assign(dataset=dataset))
            annotation_tables[version] = pd.concat(frames, ignore_index=True)
        keys = ["dataset", "file_name", "pair_key"]
        annotations = annotation_tables["processed"].merge(
            annotation_tables["original"], on=keys, how="outer", validate="one_to_one", indicator=True)
        if not annotations._merge.eq("both").all():
            raise ValueError("Original and processed annotation coverage differs")
        annotations = annotations.drop(columns="_merge")

        # 2. Read every published pair record, retaining its source position.
        frames = []
        result_root = self.raw_dir / layout["results"]
        for path in sorted(result_root.glob("*/*/weighted_8.json")):
            dataset, profile = path.relative_to(result_root).parts[:2]
            frame = pd.DataFrame(json.loads(path.read_text())[1]["Results"])
            frame = frame.assign(source_clip_row=frame.index).melt(
                id_vars=["file_name", "source_clip_row"], var_name="pair_key", value_name="native_pair").dropna()
            frames.append(frame.assign(dataset=dataset, profile=profile, source_file=str(path.relative_to(self.raw_dir))))
        observations = pd.concat(frames, ignore_index=True)
        observations = observations.join(pd.json_normalize(observations.native_pair, max_level=0))
        observations = observations.merge(annotations, on=keys, how="left", validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("A released result has no corresponding annotation")
        for column, position in (("caption0", 0), ("caption1", 1), ("answer", 2)):
            if not observations[column].eq(observations.processed_pair.str[position]).all():
                raise ValueError("A released result differs from its evaluated annotation: " + column)
        if not observations.answer.isin([0, 1]).all() or not observations.caf_prediction.isin([0, 1]).all() \
                or not observations.raw_caf_prediction.isin([-1, 0, 1]).all():
            raise ValueError("Unknown answer or prediction encoding")

        # 3. Keep smoothed and raw score variants as distinct system configurations.
        observations = observations.melt(
            id_vars=[column for column in observations if column not in parameters["variants"]],
            value_vars=list(parameters["variants"]), var_name="prediction_field", value_name="prediction")
        observations["variant"] = observations.prediction_field.map(parameters["variants"])
        observations["response"] = observations.prediction.eq(observations.answer).astype(float)
        observations["subject_key"] = observations.profile + ":" + observations.variant
        subjects = observations[["subject_key", "profile", "variant"]].drop_duplicates()
        subjects["raw_label"] = parameters["subject"]["name"]
        subjects["features"] = [{"harness": parameters["subject"]["harness"],
                                 "lalm": row.profile.split("_", 1)[0], "clap": row.profile.split("_", 1)[1],
                                 "score_variant": row.variant, "reported_alpha": parameters["subject"]["reported_alpha"]}
                                for row in subjects.itertuples()]

        # 4. Items contain the actual evaluated captions and the original audio bytes.
        observations["item_key"] = [json.dumps([row.dataset, row.file_name, row.pair_key])
                                    for row in observations.itertuples()]
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.dataset + "/" + items.file_name + "/" + items.pair_key
        items["content"] = [json.dumps(row, ensure_ascii=True) for row in items[["caption0", "caption1"]].to_dict("records")]
        items["features"] = [{"dataset": row.dataset, "pair_type": row.pair_key} for row in items.itertuples()]
        items["grading_criterion"] = items.answer.map(
            lambda answer: {"reference_answer": str(answer), "rule": self.grading["rule"]})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["native_prediction"], sort_keys=True))
        items["audio_path"] = items.dataset.map(parameters["audio_directories"]) + "/" + items.file_name
        items["attachments"] = [[{
            "source_path": re.sub(r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", layout["audio"] + "/" + name),
            "path": name, "media_type": "audio/wav", "role": "input"}]
            for name in items.audio_path]

        # 5. Preserve native scores, decisions and both annotations without truncation.
        observations["response_key"] = observations.index
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_clip_row": row.source_clip_row,
                                      "pair_key": row.pair_key, "prediction_field": row.prediction_field,
                                      "record": row.native_pair, "original_annotation": row.original_pair,
                                      "processed_annotation": row.processed_pair}, ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier", "attachments"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    Brace(__file__).main_from_args()
