#!/usr/bin/env python3
"""Curate CARE's released rankings with its native EC-level-4 grading rule."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CARE(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "historical")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        release = self.raw_dir / parameters["paths"]["release"]
        historical = self.raw_dir / parameters["paths"]["historical"]

        # 1. Read the complete native rankings and the corresponding test banks.
        observations, banks = [], []
        for path in sorted((release / "splits").glob("*/*_test.csv")):
            task = path.parent.name
            split, unit = path.stem.removesuffix("_test").rsplit("_", 1)
            identity = parameters["identity_columns"][unit]
            bank = pd.read_csv(path, dtype=str, keep_default_na=False).rename(
                columns={identity: "raw_item_id", "EC number": "reference"})
            if identity == "Reaction":
                bank["Reaction"] = bank.raw_item_id
            banks.append(bank.assign(task=task, split=split, collection="paper",
                bank_file=str(path.relative_to(self.raw_dir)), bank_row=bank.index + 2))
        for path in sorted(release.glob("task*_baselines/results_summary/*/*_test_results_df.csv")):
            if str(path.relative_to(release)) == parameters["historical_sample"]["file"]:
                continue
            task = path.relative_to(release).parts[0].removesuffix("_baselines")
            method = path.parent.name
            split, unit = path.stem.removesuffix("_test_results_df").rsplit("_", 1)
            identity = parameters["identity_columns"][unit]
            config = f"{task}/{method}"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            record = frame.to_dict("records")
            frame = frame[[identity, "EC number", "0"]].rename(
                columns={identity: "raw_item_id", "EC number": "reference", "0": "rank_zero"})
            training_split = split if config in parameters["split_specific_models"] else "shared"
            observations.append(frame.assign(record=record, task=task, method=method, split=split,
                collection="paper", input_mode=parameters["input_modes"][config],
                delimiter=parameters["delimiters"][config] or None,
                training_split=training_split, subject_key=f"{config}/{training_split}",
                source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index + 2))

        # The older one-row export names its reaction column Entry. Its original
        # input bank resolves that identifier; keep its uncertain prompt mode explicit.
        sample = parameters["historical_sample"]
        path = historical / sample["bank"]
        bank = pd.read_csv(path, dtype=str, keep_default_na=False).rename(
            columns={"Reaction": "raw_item_id", "EC number": "reference"})
        bank["Reaction"] = bank.raw_item_id
        banks.append(bank.assign(task=sample["task"], split=sample["split"], collection=sample["collection"],
            bank_file=str(path.relative_to(self.raw_dir)), bank_row=bank.index + 2))
        path = release / sample["file"]
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        record = frame.to_dict("records")
        frame = frame[["Entry", "EC number", "0"]].rename(
            columns={"Entry": "raw_item_id", "EC number": "reference", "0": "rank_zero"})
        observations.append(frame.assign(record=record, task=sample["task"], method=sample["method"],
            split=sample["split"], collection=sample["collection"], input_mode=sample["input"], delimiter=None,
            training_split="unresolved", subject_key=sample["collection"] + "/ChatGPT",
            source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index + 2))
        observations = pd.concat(observations, ignore_index=True)
        observations["response_key"] = observations.index
        banks = pd.concat(banks, ignore_index=True)

        # 2. Match complete references and input identities; attach text only to
        # the upstream conditions that actually use the EC-description signal.
        keys = ["task", "split", "collection", "raw_item_id", "reference"]
        if banks.duplicated(keys).any() or observations.duplicated(["source_file", "source_row"]).any():
            raise ValueError("Duplicate CARE input or source observation")
        descriptions = pd.read_csv(release / "processed_data/text2EC.csv", dtype=str, keep_default_na=False)
        if descriptions["EC number"].duplicated().any():
            raise ValueError("Conflicting EC descriptions")
        banks["Text"] = banks.reference.map(descriptions.set_index("EC number").Text)
        observations = observations.merge(banks, on=keys, how="left", validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("A CARE result has no matching original input and reference")
        observations = observations.drop(columns="_merge")
        for field in ["Sequence", "Reaction Text", "Text"]:
            native = observations.record.map(lambda row: row.get(field))
            present = native.notna() & observations.collection.eq("paper")
            if not native.loc[present].eq(observations.loc[present, field]).all():
                raise ValueError("A CARE result disagrees with its input bank: " + field)

        # 3. Express the source's k=1 grade as a join of decoded candidates and
        # reference labels. Averaging over labels preserves fractional credit.
        missing = self.grading["verifiers"]["rank_zero_level_four"]["missing_prediction"]
        predictions = observations[["response_key", "rank_zero", "delimiter"]].copy()
        predictions["candidate"] = predictions.rank_zero.replace("", missing).map(lambda value: [value])
        for delimiter in predictions.delimiter.dropna().unique():
            selected = predictions.delimiter.eq(delimiter)
            predictions.loc[selected, "candidate"] = predictions.loc[selected, "rank_zero"].replace(
                "", missing).str.split(delimiter, regex=False)
        predictions = predictions.explode("candidate")
        predictions["candidate"] = predictions.candidate.where(predictions.candidate.str.count(r"\.").eq(3), missing)
        predictions = predictions[["response_key", "candidate"]].drop_duplicates()
        references = observations[["response_key", "reference"]].assign(
            reference=lambda frame: frame.reference.str.split(";", regex=False)).explode("reference")
        matched = references.merge(predictions, left_on=["response_key", "reference"],
            right_on=["response_key", "candidate"], how="left", validate="many_to_one", indicator=True)
        grades = matched._merge.eq("both").groupby(matched.response_key).mean()
        observations["response"] = observations.response_key.map(grades)

        # 4. Keep trained/reference-bank variants distinct and build items from
        # the available source inputs, with grading labels outside their content.
        subjects = observations[["subject_key", "task", "method", "training_split", "collection", "input_mode"]].drop_duplicates()
        if subjects.subject_key.duplicated().any():
            raise ValueError("Conflicting CARE subject configuration")
        subjects["raw_label"] = "CARE/" + subjects.subject_key
        subjects["features"] = [dict(harness=parameters["harness"]["name"],
            harness_version=release.name if row.collection == "paper" else historical.name, source_method=row.method,
            source_task=row.task, training_split=row.training_split, collection=row.collection,
            input_modality=row.input_mode) for row in subjects.itertuples()]
        item_keys = keys + ["input_mode"]
        items = observations[item_keys + ["Sequence", "Reaction", "Reaction Text", "Text"]].drop_duplicates(item_keys).copy()
        items["item_key"] = range(len(items))
        inputs = []
        for record in items.to_dict("records"):
            fields = [parameters["primary_input"][record["input_mode"]]]
            if additional := parameters["additional_input"].get(record["input_mode"]):
                fields.append(additional)
            if any(not isinstance(record[field], str) or not record[field] for field in fields):
                raise ValueError("A CARE task lacks a required source input")
            inputs.append(json.dumps({field: record[field] for field in fields}, ensure_ascii=False, sort_keys=True))
        items["content"] = inputs
        items["features"] = [dict(task=row.task, input_modality=row.input_mode) for row in items.itertuples()]
        items["grading_criterion"] = [{"reference_answer": reference, "rule": self.grading["rule"]} for reference in items.reference]
        verifier = json.dumps(self.grading["verifiers"]["rank_zero_level_four"], sort_keys=True)
        items["verifier"] = [ExactMatcher(spec=verifier) for _ in range(len(items))]
        observations = observations.merge(items[item_keys + ["item_key"]], on=item_keys, how="left", validate="many_to_one")

        # 5. Retain every source cell and rank in valid JSON, including blanks,
        # source positions, original input-bank links and the separate legacy sample.
        observations["test_condition"] = ("collection=" + observations.collection + ";task=" + observations.task +
            ";split=" + observations.split + ";metric=k1_ec_level4")
        # The shared registrar numbers repeated observations after identical
        # content and grading resolve to the same canonical item.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            bank_file=row.bank_file, bank_row=int(row.bank_row), record=row.record),
            ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    CARE(__file__).main_from_args()
