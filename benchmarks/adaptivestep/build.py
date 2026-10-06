#!/usr/bin/env python3
"""Curate AdaptiveStep's final candidate pools without running a model or sandbox."""

import json
import runpy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AdaptiveStep(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = {name: self.raw_dir / value for name, value in parameters["layout"].items()}
        patterns = parameters["math_patterns"]
        protocols = self.grading["verifiers"]
        # The pinned utility contains pure answer-normalization functions.
        # run_path reads it without modifying raw/ or importing the GPU evaluator.
        normalizer = runpy.run_path(str(paths["math_util"]))

        # 1. Load question banks and join references by exact task text.
        gsm = pd.read_parquet(paths["gsm_reference"])
        gsm["reference"] = gsm.answer.str.extract(patterns["reference_number"], expand=False)
        math = pd.read_json(paths["math_reference"], lines=True, dtype=False, convert_dates=False)
        boxed = math.solution.map(normalizer["last_boxed_only_string"])
        if not boxed.str.startswith("\\boxed{").all():
            raise ValueError("A MATH500 reference lacks the expected boxed answer")
        math["reference"] = boxed.str.slice(7, -1)
        references = pd.concat([
            gsm[["question", "reference"]].assign(dataset="gsm8k"),
            math.rename(columns={"problem": "question"})[["question", "reference"]].assign(dataset="math500"),
        ], ignore_index=True)

        # 2. Flatten aligned math candidate/step arrays, preserving positions.
        frames = []
        candidate_columns = ["pred", "math_random_list", "math_hard_list", "math_confidence_list"]
        for filename, dataset in parameters["datasets"].items():
            if dataset not in {"gsm8k", "math500"}:
                continue
            frame = pd.read_json(paths["math"] / filename, lines=True, dtype=False, convert_dates=False)
            frame["source_record"] = frame.drop(columns=candidate_columns).to_dict("records")
            frame = frame.assign(source_file=filename, source_row=frame.index, dataset=dataset,
                                 subject_key=parameters["models"][filename])
            frame["content"] = frame["input"] if "input" in frame else frame.question
            frame["prompt_origin"] = "recorded_input" if "input" in frame else "released_question"
            frame["raw_item_id"] = dataset + "_" + frame.idx.astype(str)
            frame["original_reference"] = (frame.gt_answer.str.extract(patterns["reference_number"], expand=False)
                if dataset == "gsm8k" else frame.gt_answer.map(normalizer["last_boxed_only_string"]).str.slice(7, -1))
            frames.append(frame)
        pools = pd.concat(frames, ignore_index=True).merge(
            references, on=["dataset", "question"], how="left", validate="many_to_one")
        if pools.reference.isna().any() or pools.original_reference.isna().any():
            raise ValueError("A released math task has no verified reference")
        pools["item_key"] = "math:" + pools.index.astype(str)
        observations = pools.explode(candidate_columns, ignore_index=True)
        observations["trial"] = observations.groupby("item_key", sort=False).cumcount() + 1

        # 3. Apply the native parser; retain the pre-correction verdict too.
        numeric = observations.dataset.eq("gsm8k")
        parsed = observations.loc[numeric, "pred"].str.extract(patterns["predicted_number"], expand=False)
        observations.loc[numeric, "response"] = parsed.eq(observations.loc[numeric, "reference"]).astype(float)
        symbolic = ~numeric
        marked = observations.loc[symbolic, "pred"].str.contains(patterns["answer_marker"], regex=False)
        parsed = (observations.loc[symbolic, "pred"].str.rsplit(patterns["answer_marker"], n=1).str[-1]
                  .str.split(".\n", n=1, regex=False).str[0].str.strip().str.removesuffix(".").str.strip())
        observations.loc[symbolic, "response"] = [
            float(present and normalizer["is_equiv"](answer, reference))
            for present, answer, reference in zip(marked, parsed, observations.loc[symbolic, "reference"], strict=True)]
        observations["original_response"] = observations.response
        observations.loc[symbolic, "original_response"] = [
            float(present and normalizer["is_equiv"](answer, reference))
            for present, answer, reference in zip(marked, parsed, observations.loc[symbolic, "original_reference"], strict=True)]
        observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str) + ":" + observations.trial.astype(str)
        math_traces = observations[["source_file", "source_row", "trial", "source_record", *candidate_columns,
                                    "original_reference", "reference", "original_response"]].to_dict("records")
        traces = [pd.DataFrame({"response_key": observations.response_key,
                               "trace": [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in math_traces]})]
        items = [pools[["item_key", "raw_item_id", "content", "dataset", "reference", "prompt_origin"]]]
        responses = [observations[["response_key", "subject_key", "item_key", "response", "trial", "dataset"]]]

        # 4. Join PRM/ORM annotations of the same final code candidates once.
        for filename, dataset in parameters["datasets"].items():
            if dataset in {"gsm8k", "math500"}:
                continue
            frame = pd.read_json(paths["code"] / filename, lines=True, dtype=False, convert_dates=False)
            orm_name = filename.replace(".jsonl", "_orm.jsonl")
            orm = pd.read_json(paths["code"] / orm_name, lines=True, dtype=False, convert_dates=False)
            if not frame.drop(columns="code_confidence_list").equals(orm.drop(columns="code_confidence_list")):
                raise ValueError("PRM and ORM files no longer describe the same candidate pool")
            source_columns = [name for name in frame if name not in {
                "pred", "code", "code_list", "code_confidence_list_pre", "code_confidence_list"}]
            frame["source_record"] = frame[source_columns].to_dict("records")
            frame["candidate"] = frame.code if dataset == "livecodebench" else frame.pred
            frame["orm_annotation"] = orm.code_confidence_list
            frame["content"] = frame.prompt_use if dataset == "livecodebench" else frame.question
            frame["prompt_origin"] = "recorded_input" if dataset == "livecodebench" else "released_question"
            frame["reference"] = None if dataset == "livecodebench" else frame.answer
            frame["raw_item_id"] = dataset + "_" + (frame.question_id if dataset == "livecodebench" else frame.task_id)
            frame["item_key"] = "code:" + dataset + ":" + frame.index.astype(str)
            frame = frame.assign(source_file=str((paths["code"] / filename).relative_to(self.raw_dir)),
                                 orm_source_file=str((paths["code"] / orm_name).relative_to(self.raw_dir)),
                                 source_row=frame.index, dataset=dataset, subject_key=parameters["models"][filename])
            items.append(frame[["item_key", "raw_item_id", "content", "dataset", "reference", "prompt_origin"]])
            expanded = frame.explode(["candidate", "code_confidence_list", "orm_annotation"], ignore_index=True)
            expanded["trial"] = expanded.groupby("item_key", sort=False).cumcount() + 1
            expanded["response_key"] = expanded.source_file + ":" + expanded.source_row.astype(str) + ":" + expanded.trial.astype(str)
            expanded["response"] = None
            records = expanded[["source_file", "orm_source_file", "source_row", "trial", "source_record", "candidate",
                                "code_confidence_list", "orm_annotation"]].to_dict("records")
            traces.append(pd.DataFrame({"response_key": expanded.response_key,
                                        "trace": [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in records]}))
            responses.append(expanded[["response_key", "subject_key", "item_key", "response", "trial", "dataset"]])

        # 5. Project linked measurement tables with explicit grading provenance.
        items = pd.concat(items, ignore_index=True)
        items["grading_criterion"] = [{"reference_answer": reference, "rule": protocols[dataset]["rule"]}
                                     for reference, dataset in zip(items.reference, items.dataset, strict=True)]
        items["verifier"] = items.dataset.map(lambda name: ExactMatcher(spec=json.dumps(protocols[name], sort_keys=True)))
        items["features"] = [{"dataset": dataset, "prompt_origin": origin}
                             for dataset, origin in zip(items.dataset, items.prompt_origin, strict=True)]
        responses = pd.concat(responses, ignore_index=True)
        responses["test_condition"] = "dataset=" + responses.dataset
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects["features"] = subjects.raw_label.map(lambda label: {"model_identifier": label, "harness": "ASPRM"})
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            # The shared builder numbers trials after identical items resolve.
            # Original candidate positions remain in the trace's trial field.
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": pd.concat(traces, ignore_index=True),
        }


if __name__ == "__main__":
    AdaptiveStep(__file__).main_from_args()
