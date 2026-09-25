#!/usr/bin/env python3
"""Curate original AutoResearchBench search attempts and their recorded judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class AutoResearchBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        release = self.raw_dir / self.build_parameters["layout"]["release"]

        # 1. Load the native input and inference lists, then expand one row per attempt.
        frames = []
        for family, stem in self.build_parameters["runs"].items():
            path = release / "output_data" / (stem + ".jsonl")
            frame = pd.read_json(path, lines=True, dtype=False, precise_float=True)
            frames.append(frame.rename_axis("source_row").reset_index().assign(
                family=family, source_file=str(path.relative_to(self.raw_dir)), model=stem.split("_academic_")[0]))
        observations = pd.concat(frames, ignore_index=True).explode("inference_results", ignore_index=True)
        observations = observations.rename(columns={"inference_results": "native_pass"})
        observations["question"] = observations.input_data.map(lambda value: value["question"])
        observations["pass_id"] = observations.native_pass.map(lambda value: value["pass_id"])
        observations["response_key"] = (observations.source_file + ":" + observations.source_row.astype(str)
                                        + ":" + observations.pass_id.astype(str))

        # 2. Flatten each released grader format without rerunning a model or judge.
        judgments = []
        for family, stem in self.build_parameters["runs"].items():
            path = release / "output_data" / (stem + self.build_parameters["evaluation_suffixes"][family])
            document = json.loads(path.read_text())
            rows = document["detailed_results" if family == "deep" else "per_record_results"]
            frame = pd.json_normalize(rows, max_level=0).rename_axis("evaluation_row").reset_index()
            frame["evaluation_record"] = rows
            if family == "deep":
                frame["question"] = frame.input_data.map(lambda value: value["question"])
                frame["reference"] = frame.input_data.map(lambda value: value["answer"])
                frame["response"] = frame.evaluation.map(lambda value: value["pass_scores"])
                frame = frame.explode(["inference_results", "response"], ignore_index=True)
                frame["pass_id"] = frame.inference_results.map(lambda value: value["pass_id"])
            else:
                frame = frame.explode("pass_results", ignore_index=True)
                frame["pass_id"] = frame.pass_results.map(lambda value: value["pass_id"])
                frame["reference"] = frame.pass_results.map(lambda value: value["gt_arxiv_ids"])
                frame["response"] = frame.pass_results.map(lambda value: value["iou"])
            frame["family"] = family
            frame["evaluation_file"] = str(path.relative_to(self.raw_dir))
            judgments.append(frame[["family", "question", "pass_id", "response", "reference",
                                    "evaluation_file", "evaluation_row", "evaluation_record"]])

        # 3. Join by task and native pass ID, requiring one judgment for every released attempt.
        observations = observations.merge(pd.concat(judgments, ignore_index=True),
            on=["family", "question", "pass_id"], how="outer", validate="one_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("A released inference or judgment has no unique matching task and pass")
        observations["item_key"] = observations.response_key
        observations["subject_key"] = observations.model
        subjects = observations[["subject_key", "model"]].drop_duplicates().rename(columns={"model": "raw_label"})
        subjects["features"] = subjects.raw_label.map(lambda model: dict(self.build_parameters["subject"], model_identifier=model))

        # 4. Preserve the recorded initial messages and the grading target actually used.
        items = observations.copy()
        items["raw_item_id"] = items.family + ":" + items.source_row.astype(str)
        items["content"] = items.native_pass.map(lambda value: json.dumps(value["messages"][:2], ensure_ascii=False, sort_keys=True))
        items["content"] = items.content.str.normalize("NFC").str.strip()
        items["features"] = items.family.map(lambda family: {"research_task": family})
        items["grading_criterion"] = [dict(reference_answer=json.dumps(row.reference, ensure_ascii=False),
            rule=self.grading["verifiers"][row.family]["logic"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][row.family], sort_keys=True))
            if row.family == "deep" else ExactMatcher(spec=json.dumps(self.grading["verifiers"][row.family], sort_keys=True))
            for row in items.itertuples()]

        # 5. Keep the complete native pass and evaluation context, including every search message.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            input_data=row.input_data, native_pass=row.native_pass, evaluation_file=row.evaluation_file,
            evaluation_row=row.evaluation_row, evaluation_record=row.evaluation_record),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AutoResearchBench(__file__).main_from_args()
