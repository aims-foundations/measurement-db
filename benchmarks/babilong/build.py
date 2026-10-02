#!/usr/bin/env python3
"""Curate released BABILong attempts, complete input passages, and source configurations."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Babilong(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load native CSV tables and their recorded prompt/generation configurations.
        frames, runs = [], []
        for release, folder in self.build_parameters["results"].items():
            root = self.raw_dir / folder
            for path in sorted(root.rglob("*.csv")):
                source_path = re.sub(r"_x([0-9a-f]{2})_", lambda match: chr(int(match[1], 16)),
                                     path.relative_to(root).as_posix())
                task, length = path.stem.split("_")[:2]
                model = source_path.rsplit("/", 1)[0]
                canonical_model = self.build_parameters["copied_models"].get(model, model)
                canonical_file = canonical_model + "/" + path.name
                configuration = json.loads(path.with_suffix(".json").read_text())
                run_key = release + ":" + source_path
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame = frame.rename(columns={frame.columns[0]: "native_index"})
                frame["native_record"] = frame.to_dict("records")
                frames.append(frame.rename_axis("source_row").reset_index().assign(run_key=run_key))
                runs.append(dict(run_key=run_key, source_file=str(path.relative_to(self.raw_dir)),
                    source_path=source_path, canonical_file=canonical_file, model=canonical_model,
                    copied_model=model != canonical_model, task=task, length=length,
                    configuration=configuration,
                    csv_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                    config_digest=hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()))
        records = pd.concat(frames, ignore_index=True)
        runs = pd.DataFrame(runs)
        records = records.merge(runs, on="run_key", validate="many_to_one")

        # 2. Count original attempts once; keep every matching copy in its trace.
        overlap = runs.loc[~runs.copied_model].groupby("source_path")[["csv_digest", "config_digest"]].nunique()
        if overlap.gt(1).any().any():
            raise ValueError("Overlapping release paths contain different results or configurations")
        primary_runs = runs.loc[~runs.copied_model].drop_duplicates("source_path")
        observations = records.loc[records.run_key.isin(primary_runs.run_key)].copy()
        observations["response_key"] = observations.run_key + ":" + observations.source_row.astype(str)
        keys = ["canonical_file", "native_index", "target", "output", "question"]
        records["occurrence"] = records.groupby(["run_key", *keys], sort=False).cumcount()
        observations["occurrence"] = observations.groupby(["run_key", *keys], sort=False).cumcount()
        copies = records.merge(observations[keys + ["occurrence", "response_key", "config_digest"]],
            on=keys + ["occurrence"], how="left", validate="many_to_one", suffixes=("", "_primary"))
        if copies.response_key.isna().any() or copies.config_digest.ne(copies.config_digest_primary).any():
            raise ValueError("A copied export differs from its complete original run")
        copies["export"] = [dict(source_file=row.source_file, source_row=row.source_row,
            record=row.native_record) for row in copies.itertuples()]
        exports = copies.groupby("response_key", sort=False).export.agg(list)
        del records, copies, frames

        # 3. Match complete runs against both input banks, including all Parquet shards.
        banks = []
        for bank, folder in self.build_parameters["inputs"].items():
            root = self.raw_dir / folder
            for path in sorted(root.rglob("*")):
                if path.suffix == ".json":
                    frame = pd.read_json(path, dtype=False)
                    task, length = path.parent.name, path.stem
                elif path.suffix == ".parquet":
                    frame = pd.read_parquet(path)
                    task, length = path.stem.split("-")[0], path.parent.name
                else:
                    continue
                banks.append(frame.rename_axis("input_row").reset_index().assign(
                    bank=bank, task=task, length=length, input_file=str(path.relative_to(self.raw_dir))))
        inputs = pd.concat(banks, ignore_index=True)
        inputs["index"] = inputs.groupby(["bank", "task", "length"], sort=False).cumcount()
        observations["index"] = observations.native_index.astype(int)
        match_keys = ["task", "length", "index", "question", "target"]
        candidates = observations[["run_key", *match_keys]].merge(
            inputs[["bank", *match_keys]], on=match_keys, how="inner", validate="many_to_many")
        matches = candidates.groupby(["run_key", "bank"]).size().rename("matched").reset_index()
        totals = observations.groupby("run_key").size().rename("total")
        matches = matches.merge(totals, on="run_key", validate="many_to_one")
        matches = matches.loc[matches.matched.eq(matches.total), ["run_key", "bank"]]
        if matches.run_key.duplicated().any():
            raise ValueError("An observed run matches more than one released input bank")
        observations = observations.merge(matches, on="run_key", how="left", validate="many_to_one")
        unresolved = observations.loc[observations.bank.isna(), "source_path"].drop_duplicates()
        if not unresolved.str.fullmatch(self.build_parameters["limitations"]["unmapped_runs"]).all():
            raise ValueError("An unexpected run lacks a complete ordered question/reference match")
        observations = observations.merge(inputs[["bank", *match_keys, "input_file", "input_row"]],
            on=["bank", *match_keys], how="left", validate="many_to_one")
        observations["context_status"] = observations.bank.notna().map(
            {True: "released_context_matched", False: "released_context_unresolved"})
        # Assets hold exact UTF-8 source strings; neither the passage nor the output is clipped.
        inputs["attachment"] = inputs.input.map(lambda text: [
            dict(path="context.txt", data=text.encode("utf-8"), media_type="text/plain", role="input")])
        observations = observations.merge(
            inputs[["bank", "task", "length", "index", "attachment"]],
            on=["bank", "task", "length", "index"], how="left", validate="many_to_one")
        observations["prompt"] = observations.configuration.map(lambda value: value["prompt"])
        observations["prompt_key"] = observations.prompt.map(
            lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        observations["item_key"] = (
            observations.bank.fillna("") + ":" + observations.task + ":" + observations.length
            + ":" + observations["index"].astype(str) + ":" + observations.prompt_key)
        missing = observations.bank.isna()
        observations.loc[missing, "item_key"] = observations.loc[missing, "response_key"]
        del inputs, banks, candidates

        # 4. Apply the captured substring-label metric and separate generation configurations.
        normalized = observations.output.str.lower()
        for delimiter in [".", "<context>", "<example>", "Question"]:
            normalized = normalized.str.split(delimiter, n=1, regex=False).str[0]
        label_sets = self.grading["verifiers"]["answer"]["task_labels"]
        allowed = observations.task.map(lambda task: [label.lower() for label in label_sets[task]])
        detected = [set(label for label in labels if label in output and label not in question.lower())
                    for labels, output, question in zip(allowed, normalized, observations.question)]
        expected = observations.target.str.lower().map(
            lambda target: target.split(",") if "," in target and len(target) > 3 else [target])
        observations["response"] = [float(len(actual) == len(wanted) and all(label in actual for label in wanted))
                                   for actual, wanted in zip(detected, expected)]
        observations["subject_configuration"] = [
            dict(model_identifier=row.model, generation_parameters=row.configuration["generate_kwargs"],
                 chat_template=row.prompt.get("chat_template"), system_prompt=row.prompt.get("system_prompt"))
            for row in observations.itertuples()]
        observations["subject_key"] = observations.subject_configuration.map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True))
        subjects = observations.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = subjects.model
        subjects["features"] = subjects.subject_configuration.map(lambda value: dict(
            harness="BABILong", model_identifier=value["model_identifier"],
            released_configuration=json.dumps(value, ensure_ascii=True, sort_keys=True)))

        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(question=row.question, prompt_configuration=row.prompt,
            context={"asset_path": "context.txt"} if row.context_status == "released_context_matched" else None),
            ensure_ascii=False, sort_keys=True) for row in items.itertuples()]
        items["features"] = [dict(task=row.task, context_length=row.length, input_context_status=row.context_status,
            **({"unresolved_input_record": row.source_file + ":" + str(row.source_row)}
               if row.context_status == "released_context_unresolved" else {})) for row in items.itertuples()]
        items["attachments"] = [value if isinstance(value, list) else [] for value in items.attachment]
        items["grading_criterion"] = items.target.map(
            lambda target: dict(reference_answer=target, rule=self.grading["rule"]))
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["answer"], sort_keys=True))] * len(items)

        # 5. Retain every native output and config, exact source positions, and context availability.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(exports=exports[row.response_key], configuration=row.configuration,
            input_context_status=row.context_status,
            input_source=dict(file=row.input_file, row=int(row.input_row), bank=row.bank)
                if row.context_status == "released_context_matched" else None),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "attachments", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    Babilong(__file__).main_from_args()
