"""Curate published LLoVi answers and their available caption-based inputs."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class EgoSchema(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("initial", "llovi", "reference", "data", "output")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]
        configurations = pd.DataFrame.from_dict({name: parameters[group]
            for name, group in parameters["configurations"].items()}, orient="index")
        configurations.index.name = "source_file"

        # 1. Read complete result records, retaining their source-file and question keys.
        with ZipFile(self.raw_dir / layout["results"]) as results, ZipFile(self.raw_dir / layout["inputs"]) as inputs:
            sources = []
            for name in configurations.index:
                records = json.loads(results.read(layout["result_prefix"] + name))["data"]
                frame = pd.DataFrame.from_dict(records, orient="index")
                frame["native_record"] = frame.to_dict("records")
                sources.append(frame.assign(source_file=name))
            observations = pd.concat(sources, ignore_index=True)

            # 2. Join the six released demonstrations to the captions used by each few-shot mode.
            examples = pd.DataFrame.from_dict(json.loads(inputs.read(layout["examples"])), orient="index")
            demonstrations = {"none": []}
            for mode, archive, path in [("captions", inputs, layout["example_captions"]),
                    ("summary", results, layout["example_summaries"])]:
                captions = pd.Series(json.loads(archive.read(path)), name="narration")
                joined = examples.join(captions, how="left", validate="one_to_one")
                if joined.narration.isna().any():
                    raise ValueError("A released few-shot example lacks its input captions")
                joined["narration"] = joined.narration.map(lambda value: ". ".join(value) if isinstance(value, list) else value)
                demonstrations[mode] = joined.reset_index(names="uid").to_dict("records")

        # 3. Join configuration metadata and preserve the archive's original binary scoring rule.
        observations = observations.merge(configurations, on="source_file", how="left", validate="many_to_one")
        observations["subject_key"] = observations.source_file
        observations["response_key"] = observations.source_file + "/" + observations.uid
        observations["item_key"] = observations.response_key
        observations["response"] = observations.pred.eq(observations.truth).astype(float)
        observations["demonstration_records"] = observations.demonstrations.map(demonstrations)

        # 4. Keep input fields separate from the recorded answer and gold target.
        items = observations.copy()
        fields = list(parameters["target_fields"].values()) + ["demonstration_records"]
        items["content"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in items[fields].to_dict("records")]
        items["raw_item_id"] = items.uid
        items["grading_criterion"] = [dict(reference_answer=str(int(value)), rule=self.grading["rule"]) for value in items.truth]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["choice"], sort_keys=True)) for _ in items.index]
        items["features"] = [dict(input_scope=parameters["input_scope"]["description"]) for _ in items.index]
        subjects = configurations.reset_index().rename(columns={"source_file": "subject_key"})
        subjects["features"] = [dict(harness=row.harness, configuration=row.configuration, model_identifier=row.model,
            captioner=row.captioner, configuration_scope=parameters["subject_scope"]["description"])
            for row in subjects.itertuples()]

        # 5. Preserve every native answer, prompt template and source association without clipping.
        traces = observations[["response_key", "source_file", "uid", "native_record"]].copy()
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EgoSchema(__file__).main_from_args()
