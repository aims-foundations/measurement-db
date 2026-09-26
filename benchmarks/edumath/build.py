"""Curate released EDUMATH generations and their original quality judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class EduMathBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the published CSV without changing text or missing-value spellings.
        source = pd.read_csv(self.raw_dir / parameters["input"]["file"], dtype=str, keep_default_na=False)
        source["native_record"] = source.to_dict("records")
        source["source_row"] = source.index
        conditions = list(parameters["conditions"].values())
        inputs = source[conditions].drop_duplicates().copy()
        inputs["input_key"] = inputs.index.astype(str)
        source = source.merge(inputs, on=conditions, how="left", validate="many_to_one")

        # 2. Unpivot the two judgments; retain their native, opposite label directions.
        observations = source.melt(id_vars=["source_row", "model", "input_key", "native_record"],
            value_vars=list(parameters["judges"]), var_name="label_column", value_name="response")
        observations["response"] = pd.to_numeric(observations.response, errors="raise")
        observations["judge"] = observations.label_column.map(parameters["judges"])
        observations["subject_key"] = observations.model
        observations["item_key"] = observations.input_key + "/" + observations.judge
        observations["response_key"] = observations.source_row.astype(str) + "/" + observations.judge

        # 3. Keep generation requests in items, with a separate grading protocol for each judge.
        items = inputs.merge(pd.DataFrame({"judge": list(parameters["judges"].values())}), how="cross")
        items["item_key"] = items.input_key + "/" + items.judge
        items["raw_item_id"] = items.item_key
        items["content"] = [parameters["prompt"]["template"].format(**row) for row in items.to_dict("records")]
        items["grading_criterion"] = items.judge.map(lambda name: self.grading["verifiers"][name]["criterion"])
        items["verifier"] = items.judge.map(lambda name: (Judge if name == "llm" else ExactMatcher)(
            spec=json.dumps(self.grading["verifiers"][name]["implementation"], sort_keys=True)))
        items["features"] = [dict(grading_protocol=name, input_scope=parameters["input_scope"]["description"])
            for name in items.judge]

        # 4. Preserve the reported generator identities without guessing historical configurations.
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [parameters["subject_features"] for _ in subjects.index]

        # 5. Keep every original generation, both labels and full reasoning linked to its source row.
        traces = observations[["response_key", "source_row", "label_column", "native_record"]].copy()
        traces["source_file"] = parameters["input"]["file"]
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EduMathBench(__file__).main_from_args()
