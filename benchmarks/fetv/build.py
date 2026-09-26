"""Preserve FETV's source prompts, original ratings and complete generated videos."""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class FETV(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("benchmark", "scores", "videos", "paper", "instructions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, protocols = self.build_parameters, self.grading["verifiers"]
        paths = parameters["paths"]

        # 1. Read both released prompt languages, retaining the original task records.
        task_records = pd.read_json(self.raw_dir / paths["prompts"], lines=True, typ="series", dtype=False)
        prompts = pd.json_normalize(task_records, max_level=0)
        prompts["prompt_index"] = prompts.index
        prompts["task_record"] = task_records
        prompts["chinese_prompt"] = (self.raw_dir / paths["chinese_prompts"]).read_text().splitlines()

        # 2. Unpivot individual human ratings and concatenate published automatic scores.
        frames = []
        for path in sorted(self.raw_dir.glob(paths["human_scores"])):
            model = path.stem.removeprefix("manual_eval_results_")
            model = parameters["model_aliases"].get(model, model)
            if model not in parameters["models"]:
                continue
            records = pd.read_json(path, lines=True, dtype=False, precise_float=True).stack(future_stack=True).dropna()
            records = records.rename("native_record").rename_axis(["source_row", "prompt_index"]).reset_index()
            wide = records.join(pd.json_normalize(records.native_record))
            ratings = wide.melt(id_vars=["source_row", "prompt_index", "native_record", "video_id"],
                value_vars=list(parameters["human_dimensions"]), var_name="field", value_name="response")
            ratings = ratings.dropna(subset=["response"])
            ratings["metric"] = ratings.field.map(parameters["human_dimensions"])
            frames.append(ratings.assign(subject_key=model, rater=path.parent.name,
                source_file=str(path.relative_to(self.raw_dir))))
        for path in sorted(self.raw_dir.glob(paths["automatic_scores"])):
            model = path.stem.removeprefix("auto_eval_results_")
            if model not in parameters["models"]:
                continue
            scores = pd.read_json(path, typ="series", dtype=False, precise_float=True)
            scores = scores.rename_axis("prompt_index").reset_index(name="response")
            scores["native_record"] = [{str(index): value} for index, value in zip(scores.prompt_index, scores.response)]
            frames.append(scores.assign(subject_key=model, rater="automatic", metric=path.parent.name,
                source_file=str(path.relative_to(self.raw_dir)), source_row=None))
        responses = pd.concat(frames, ignore_index=True)
        responses = responses.merge(prompts, on="prompt_index", how="left", validate="many_to_one", suffixes=("_rating", ""))
        human = responses.rater.ne("automatic")
        # The 78 unusual prompts have no real reference video: null in the task bank, "None" in ratings.
        if responses.prompt.isna().any() or not responses.loc[human, "video_id_rating"].eq(responses.loc[human, "video_id"].fillna("None")).all():
            raise ValueError("A rating does not match its original prompt/reference-video ID")

        # 3. Index the original output archives without extracting or rewriting them.
        outputs = []
        for model, source in parameters["archives"].items():
            with ZipFile(self.raw_dir / source) as archive:
                for member in sorted(archive.namelist()):
                    if member.startswith(parameters["video_prefixes"][model]) and Path(member).suffix in parameters["media_types"]:
                        with archive.open(member) as stream:
                            digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        outputs.append(dict(subject_key=model, prompt_index=int(Path(member).stem),
                            output=dict(source_file=source, member=member, sha256=digest,
                                bytes=archive.getinfo(member).file_size, media_type=parameters["media_types"][Path(member).suffix])))
        responses = responses.merge(pd.DataFrame(outputs), on=["subject_key", "prompt_index"], how="left", validate="many_to_one")
        if responses.output.isna().any():
            raise ValueError("A score has no uniquely associated generated video")
        responses["response_key"] = responses.source_file + "/" + responses.prompt_index.astype(str) + "/" + responses.metric
        responses["item_key"] = responses.subject_key + "/" + responses.prompt_index.astype(str) + "/" + responses.metric + "/" + responses.rater

        # 4. Keep language, rater and grading scale explicit; outputs stay out of items.
        items = responses.drop_duplicates("item_key").copy()
        items["content"] = items.prompt.where(items.subject_key.ne("cogvideo"), items.chinese_prompt)
        items["raw_item_id"] = items.prompt_index.astype(str)
        items["features"] = [dict(input_language="zh" if model == "cogvideo" else "en",
            input_scope=parameters["input_scope"]["description"]) for model in items.subject_key]
        items["grading_criterion"] = [dict(rule=protocols[metric]["rule"] + "\nEvaluation prompt: " + prompt,
            response_scale=protocols[metric]["response_scale"]) for metric, prompt in zip(items.metric, items.prompt)]
        items["verifier"] = [Judge(judged_by="human", spec=json.dumps(dict(protocols[metric]["implementation"], annotator=rater), sort_keys=True))
            if rater != "automatic" else ExactMatcher(spec=json.dumps(protocols[metric]["implementation"], sort_keys=True))
            for metric, rater in zip(items.metric, items.rater)]
        subjects = responses[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.map(parameters["models"])
        subjects["features"] = [dict(**parameters["subject_features"], source_model=model) for model in subjects.subject_key]

        # 5. Preserve full native ratings, task metadata and exact output references.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file,
            source_row=None if pd.isna(row.source_row) else int(row.source_row), prompt_index=int(row.prompt_index),
            metric=row.metric, rater=row.rater, native_record=row.native_record, task_record=row.task_record,
            generation_prompt=row.chinese_prompt if row.subject_key == "cogvideo" else row.prompt,
            output=row.output), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    FETV(__file__).main_from_args()
