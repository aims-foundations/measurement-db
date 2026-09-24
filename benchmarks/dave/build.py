#!/usr/bin/env python3
"""Join DAVE's native model results to unambiguous audio-visual stimuli."""

import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DAVE(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results", "stimuli")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the item bank. Option order differs between the bank and recorded prompts.
        frames = []
        for split, filename in self.build_parameters["splits"].items():
            frame = pd.read_json(self.raw_dir / filename, dtype=False, convert_dates=False)
            frames.append(frame.assign(split=split, source_item_key=split + "::" + frame.index.astype(str)))
        bank = pd.concat(frames, ignore_index=True)
        questions = pd.json_normalize(bank.choice_metadata.map(lambda value: value["audio_visual_alignment"]).tolist())
        bank = bank.assign(question_type=bank.type, sound=bank.audio_class.str.replace("_", " ", regex=False),
                           choices=questions.choices, correct_option=[choices[index] for choices, index in zip(questions.choices, questions.ground_truth)])
        bank["options_key"] = bank.choices.map(lambda choices: tuple(sorted(value.strip().lower() for value in choices)))
        bank["correct_option"] = bank.correct_option.str.strip().str.lower()
        match_columns = ["split", "question_type", "sound", "options_key", "correct_option"]
        candidates = bank.groupby(match_columns, dropna=False).size().rename("candidate_count").reset_index()
        unique_bank = bank.loc[~bank.duplicated(match_columns, keep=False)]

        # 2. Flatten released model -> condition -> attempt lists, retaining full response payloads.
        layout = self.build_parameters["layout"]
        models = self.build_parameters["models"]
        frames = []
        for filename in sorted(self.source_files):
            if Path(filename).parent.as_posix() != layout["results_directory"] or not filename.endswith(".json"):
                continue
            data = json.loads((self.raw_dir / filename).read_text())
            groups = pd.DataFrame.from_dict(data["predictions"], orient="index").rename_axis("model").reset_index()
            if layout["condition"] not in groups:
                continue
            groups = groups.loc[groups.model.isin(models), ["model", layout["condition"]]].dropna()
            if groups.empty:
                continue
            rows = groups.explode(layout["condition"], ignore_index=True)
            frame = pd.json_normalize(rows[layout["condition"]].tolist(), max_level=0)
            split = next(name for name in self.build_parameters["splits"] if f"_{name}_" in filename)
            frames.append(frame.assign(model=rows.model, split=split, source_file=filename))
        observations = pd.concat(frames, ignore_index=True)
        if not observations.is_correct.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("DAVE correctness must be a released boolean")

        # 3. Match question content, sound, task type and the reference option to the original media.
        patterns = self.build_parameters["patterns"]
        options = observations.prompt.str.findall(patterns["option"], flags=re.MULTILINE).map(dict)
        observations["options_key"] = options.map(lambda choices: tuple(sorted(value.strip().lower() for value in choices.values())))
        observations["sound"] = observations.prompt.str.extract(patterns["sound"], expand=False)
        if not observations.ground_truth.map(lambda value: isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)).all():
            raise ValueError("Expected one released correct option in DAVE's multimodal task")
        observations["reference"] = observations.ground_truth.str[0]
        observations["correct_option"] = [choices.get(label.strip("()")) for choices, label in zip(options, observations.reference)]
        observations["correct_option"] = observations.correct_option.str.strip().str.lower()
        observations = observations.merge(candidates, on=match_columns, how="left", sort=False, validate="many_to_one")
        if observations.candidate_count.isna().any():
            raise ValueError("A DAVE question/reference has no matching released stimulus")
        print(f"[dave] Excluding {observations.candidate_count.ne(1).sum()} ambiguous media matches", flush=True)
        observations = observations.loc[observations.candidate_count.eq(1)].merge(
            unique_bank[[*match_columns, "source_item_key", "video_with_overlayed_audio_path", "overlayed_audio_path"]],
            on=match_columns, how="left", sort=False, validate="many_to_one")

        # 4. Keep distinct complete prompts and references, even when they concern the same video.
        item_columns = ["source_item_key", "prompt", "reference"]
        observations["item_key"] = observations.groupby(item_columns, sort=False).ngroup()
        items = observations.drop_duplicates("item_key").copy()
        items = items.assign(raw_item_id=items.source_item_key, content=items.prompt,
                             grading_criterion=items.reference.map(lambda value: {"reference_answer": value, "rule": self.grading["rule"]}),
                             verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["recorded_choice"], sort_keys=True)))
        items["features"] = items[["split", "question_type"]].to_dict("records")

        # 5. Read each native media member once; ZIP archives remain unchanged in raw/.
        media_columns = ["video_with_overlayed_audio_path", "overlayed_audio_path"]
        media = items[["split", *media_columns]].melt(id_vars="split", value_name="member").drop_duplicates(["split", "member"])
        payloads = {}
        for split, members in media.groupby("split", sort=False):
            with ZipFile(self.raw_dir / layout["media_directory"] / f"{split}.zip") as archive:
                for member in members.member:
                    payloads[split, member] = archive.read(member)
        items["attachments"] = [
            [{"data": payloads[split, video], "path": "video.mp4", "media_type": "video/mp4", "role": "input_video"},
             {"data": payloads[split, audio], "path": "audio.mp3", "media_type": "audio/mpeg", "role": "input_audio"}]
            for split, video, audio in items[["split", *media_columns]].itertuples(index=False, name=None)
        ]

        # 6. Preserve native model labels, binary verdicts and the complete structured response.
        subjects = observations[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects = subjects.assign(raw_label=subjects.subject_key.map(models),
                                   features=subjects.subject_key.map(lambda name: {"source_model_name": name}))
        responses = observations.assign(response_key=range(len(observations)), subject_key=observations.model,
                                         response=observations.is_correct.astype(float),
                                         test_condition=observations.split + "/" + layout["condition"])
        traces = responses[["response_key"]].assign(trace=responses.response_dict.map(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)))
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DAVE(__file__).main_from_args()
