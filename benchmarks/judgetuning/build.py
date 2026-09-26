"""Tabulate the released JudgeTuning LMSys judge evaluations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class JudgeTuning(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Read each annotation export using the upstream CSV writer's escaping.
        exports = []
        for name in parameters["models"]:
            path = self.raw_dir / paths["annotations"] / (name + ".csv.zip")
            frame = pd.read_csv(path, escapechar="\\", keep_default_na=False, float_precision="round_trip")
            games = frame.to_dict("records")
            frame = frame.assign(subject_key=name, source_file=str(path.relative_to(self.raw_dir)),
                game=[dict(source_row=index, annotation=value) for index, value in enumerate(games)])
            exports.append(frame)
        annotations = pd.concat(exports, ignore_index=True)

        # 2. Join complete battle definitions; answer order and human labels must agree.
        instructions = pd.read_csv(self.raw_dir / paths["instructions"], keep_default_na=False)
        fields = ["model1", "model2", "output1", "output2", "human_preference"]
        if annotations.groupby("instruction_index")[fields].nunique(dropna=False).gt(1).any().any():
            raise ValueError("A source instruction ID refers to conflicting paired-answer comparisons")
        battles = annotations.drop_duplicates("instruction_index")[["instruction_index", *fields]].merge(
            instructions[["instruction_index", "instruction"]], on="instruction_index", how="left", validate="one_to_one")
        if battles.instruction.isna().any():
            raise ValueError("A recorded battle has no released instruction")
        items = battles[["instruction_index"]].rename(columns={"instruction_index": "item_key"})
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(instruction=row.instruction, output1=row.output1, output2=row.output2),
            ensure_ascii=False) for row in battles.itertuples()]
        items["features"] = [dict(candidate_model1=row.model1, candidate_model2=row.model2,
            **parameters["item_features"]) for row in battles.itertuples()]
        items["grading_criterion"] = [dict(reference_answer={0.0:"output1", 0.5:"tie", 1.0:"output2"}[value],
            rule=self.grading["rule"]) for value in battles.human_preference]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["human_agreement"], sort_keys=True))] * len(items)

        # 3. Aggregate actual games, preserving the original preference orientation.
        observations = annotations.groupby(["subject_key", "instruction_index"], sort=False).agg(
            preference=("preference", "mean"), human_preference=("human_preference", "first"),
            source_file=("source_file", "first"), games=("game", list)).reset_index()
        responses = observations[["subject_key", "instruction_index"]].rename(columns={"instruction_index": "item_key"})
        responses["response_key"] = observations.subject_key + ":" + observations.instruction_index
        responses["response"] = ((observations.preference.lt(.5) & observations.human_preference.lt(.5))
            | (observations.preference.eq(.5) & observations.human_preference.eq(.5))
            | (observations.preference.gt(.5) & observations.human_preference.gt(.5))).astype(float)
        responses["test_condition"] = "temperature=" + observations.subject_key.map(parameters["temperatures"])

        # 4. Keep named judge identities and every source game, including empty completions.
        subjects = pd.DataFrame({"subject_key": list(parameters["models"])})
        subjects["raw_label"] = subjects.subject_key.map(parameters["labels"])
        subjects["features"] = [dict(named_judge=name, model_identifier=parameters["models"][name],
            configuration_source=parameters["configuration_sources"][name], **parameters["subject_features"])
            for name in subjects.subject_key]
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, instruction_index=row.instruction_index,
            games=row.games), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {"subjects": subjects, "items": items, "responses": responses, "traces": traces}


if __name__ == "__main__":
    JudgeTuning(__file__).main_from_args()
