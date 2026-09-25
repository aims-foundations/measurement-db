#!/usr/bin/env python3
"""Curate LLMDrift's released attempts with the authors' task-specific graders."""

import json
import re
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class ChatgptDrift(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        grading = self.grading["verifiers"]
        measures = grading["measures"]

        # 1. Read the native CSVs directly as tables, preserving every original field.
        frames = []
        for path in sorted((self.raw_dir / parameters["paths"]["generations"]).glob("*_EVAL.csv")):
            frame = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
            frame.columns = frame.iloc[0].tolist()  # Keep the original empty index-column heading.
            frame = frame.iloc[1:].reset_index(drop=True)
            frame["record"] = frame.to_dict("records")
            frame["source_row"] = frame.index
            frame["source_file"] = str(path.relative_to(self.raw_dir))
            frames.append(frame)
        observations = pd.concat(frames, ignore_index=True)
        observations["response_key"] = observations.index
        observations["protocol"] = observations.dataset.map(parameters["datasets"])
        if observations.protocol.isna().any() or observations["query"].eq("").any():
            raise ValueError("Review an unknown LLMDrift task or missing input")
        if not observations.trail.eq("0").all():
            raise ValueError("Review new native trial numbering before assigning canonical trials")

        # 2. Express the released graders as string/table operations, without
        # executing generated code, querying an API, or changing judgment labels.
        observations["response"] = pd.Series(index=observations.index, dtype=float)
        prime = observations.protocol.eq("prime")
        words = observations.loc[prime, "answer"].str.lower().str.replace(r"[,.]", " ", regex=True).str.split().str.join("] [")
        words = "[" + words + "]"
        parsed = pd.Series("undetermined", index=words.index)
        parsed.loc[words.str.contains("[no]", regex=False)] = "no"
        parsed.loc[words.str.contains("[yes]", regex=False)] = "yes"
        observations.loc[prime, "response"] = parsed.eq(observations.loc[prime, "ref_answer"].str.lower()).astype(float)

        happy = observations.protocol.eq("happy_count")
        reference = observations.loc[happy, "ref_answer"].str.lower().str.extract(r"boxed{([^}]*)}", expand=False)
        parsed = observations.loc[happy, "answer"].str.lower().str.extract(r"boxed{([^}]*)}", expand=False)
        if reference.isna().any():
            raise ValueError("A happy-number task has no reference count")
        observations.loc[happy, "response"] = parsed.eq(reference).astype(float)

        exact = observations.protocol.eq("exact_match")
        normalized = observations.loc[exact, ["answer", "ref_answer"]].copy()
        for column in normalized:
            normalized[column] = (normalized[column].str.lower()
                .str.replace("[" + re.escape(string.punctuation) + "]", "", regex=True)
                .str.replace(r"\b(a|an|the)\b", " ", regex=True).str.split().str.join(" "))
        observations.loc[exact, "response"] = normalized.answer.eq(normalized.ref_answer).astype(float)

        multiple_choice = observations.protocol.eq("multiple_choice")
        matches = observations.loc[multiple_choice, "answer"].str.lower().str.findall(r"the answer is (\([a-z]\))").explode().to_frame("parsed")
        matches["reference"] = observations.loc[multiple_choice, "ref_answer"].str.lower()
        observations.loc[multiple_choice, "response"] = matches.parsed.eq(matches.reference).groupby(level=0).any().astype(float)

        survey = observations.protocol.eq("survey")
        answers = observations.loc[survey, "answer"]
        observations.loc[survey, "response"] = (answers.str.contains(r"\([A-Za-z]\)") &
            ~answers.str.contains(r"\([A-Za-z]\)\. Refused")).astype(float)
        code = observations.protocol.eq("code")
        observations.loc[code, "response"] = pd.to_numeric(observations.loc[code, "Directly Usable"])
        if not observations.loc[code, "response"].eq(observations.loc[code, "Code_Submit"].str.contains("Accepted", regex=False).astype(float)).all():
            raise ValueError("The recorded code grade differs from the published grader")
        sensitive = observations.protocol.eq("sensitive")
        observations.loc[sensitive, "response"] = pd.to_numeric(observations.loc[sensitive, "Response Rate"])
        if not observations.response.isin([0.0, 1.0]).all():
            raise ValueError("Missing or invalid LLMDrift outcome")

        # 3. Distinguish API snapshots, serving paths and recorded generation limits.
        configuration = ["model", "max_tokens"]
        subjects = observations[configuration].drop_duplicates().reset_index(drop=True)
        subjects["subject_key"] = subjects.index
        subjects[["provider_path", "raw_label"]] = subjects.model.str.split("/", n=1, expand=True)
        subjects["harness"] = subjects.provider_path.map(parameters["harnesses"])
        if subjects.harness.isna().any():
            raise ValueError("Unknown LLMDrift provider path")
        subjects["features"] = [dict(harness=row.harness, source_model=row.model,
            recorded_max_tokens=row.max_tokens) for row in subjects.itertuples()]
        observations = observations.merge(subjects[configuration + ["subject_key"]],
            on=configuration, how="left", sort=False, validate="many_to_one")

        # 4. Preserve complete prompts and their grading protocol. OpinionQA's
        # placeholder reference is not a gold opinion or a predictor input.
        uses_reference = observations.protocol.isin(["prime", "happy_count", "exact_match", "multiple_choice"])
        observations["reference"] = observations.ref_answer.where(uses_reference, "")
        identity = ["dataset", "protocol", "query", "reference"]
        items = observations[identity + ["id"]].drop_duplicates(identity).reset_index(drop=True)
        items["item_key"] = items.index
        items["raw_item_id"] = items.dataset + "/" + items.id
        items["content"] = items["query"]
        items["features"] = items.dataset.map(lambda dataset: dict(dataset=dataset))
        items["grading_criterion"] = [dict(reference_answer=row.reference or None,
            rule=measures[row.protocol]["rule"], response_scale=measures[row.protocol]["scale"])
            for row in items.itertuples()]
        specifications = {name: json.dumps(dict(**grading["protocol"],
            **measure["verifier"]), sort_keys=True) for name, measure in measures.items()}
        items["verifier"] = [Judge(spec=specifications[name]) if name == "sensitive" else
            ExactMatcher(spec=specifications[name]) for name in items.protocol]

        # 5. Link each original attempt to its item, retaining configuration dates
        # as recorded. The shared registrar numbers repeats after IDs resolve.
        responses = observations.merge(items[identity + ["item_key"]], on=identity,
            how="left", sort=False, validate="many_to_one")
        responses["test_condition"] = "recorded_date=" + responses.date + ";temperature=" + responses.temperature
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            record=row.record), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ChatgptDrift(__file__).main_from_args()
