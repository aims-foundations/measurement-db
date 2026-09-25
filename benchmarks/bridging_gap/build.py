#!/usr/bin/env python3
"""Import the complete paper results and genuine native supplementary runs."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BridgingGap(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "large_files")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Read paper observations and their distinct model/training configurations.
        master = pd.read_csv(self.raw_dir / paths["responses"], dtype=str, keep_default_na=False)
        master["record"] = master.to_dict("records")
        master = master.rename(columns=parameters["response_columns"])
        master["source_row"] = master.index + 2
        master["subject_key"] = master.model + ":" + master.training_id
        if not master.correct.isin(["True", "False"]).all():
            raise ValueError("Unknown or absent published correctness value")
        master["response"] = master.correct.map({"True": 1.0, "False": 0.0})
        master["trial"] = pd.to_numeric(master.trial, errors="raise").astype("int64")
        if master.duplicated(["subject_key", "item_key", "trial"]).any():
            raise ValueError("Repeated consolidated observation")
        configuration_columns = ["subject_key"] + list(parameters["response_columns"].values())[:9]
        subjects = master[configuration_columns].drop_duplicates()
        if subjects.subject_key.duplicated().any():
            raise ValueError("Conflicting model/training configuration")
        training = pd.read_csv(self.raw_dir / paths["training"], dtype=str, keep_default_na=False)
        if training["Fine-Tuning.Dataset ID"].duplicated().any() or not set(master.training_id) - {""} <= set(training["Fine-Tuning.Dataset ID"]):
            raise ValueError("A fine-tuned model has no unique released training dataset")
        subjects["raw_label"] = subjects.model.where(subjects.training_id.eq(""), subjects.model + " (" + subjects.training_id + ")")
        subjects["features"] = [dict(harness=parameters["harness"]["name"],
            harness_version=parameters["harness"]["revision"], source_model_identifier=row.model,
            **{key: getattr(row, key) for key in configuration_columns[2:] if getattr(row, key)},
            training_dataset_source=paths["training"]) for row in subjects.itertuples()]

        # 2. Read observed items; repair missing Winogrande fields from original translated banks.
        bank = pd.read_csv(self.raw_dir / paths["items"], dtype=str, keep_default_na=False).rename(columns=parameters["item_columns"])
        bank = bank.loc[bank.item_key.isin(master.item_key)].copy()
        repairs = []
        for code, language in parameters["languages"].items():
            suffix = "" if code == "en" else "_" + code
            frame = pd.read_csv(self.raw_dir / paths["human_winogrande"] / f"winogrande{suffix}.csv", dtype=str, keep_default_na=False, lineterminator="\n")
            frame = frame.rename(columns={"qID": "winogrande_id", "Answer": "gold",
                f"{language} Sentence": "question", f"{language} Option 1": "option1", f"{language} Option 2": "option2"})
            repairs.append(frame[["winogrande_id", "question", "option1", "option2", "gold"]].assign(language=code))
        incomplete = bank.family.eq("winogrande") & bank[["option1", "option2", "gold"]].eq("").any(axis=1)
        replacements = bank.loc[incomplete, ["item_key", "winogrande_id", "language"]].merge(
            pd.concat(repairs), on=["winogrande_id", "language"], how="left", validate="one_to_one").set_index("item_key")
        if replacements.question.isna().any():
            raise ValueError("A missing Winogrande input has no original source record")
        for column in ["question", "option1", "option2", "gold"]:
            bank.loc[incomplete, column] = bank.loc[incomplete, "item_key"].map(replacements[column])

        # 3. Join full request templates; Belebele row order is not a stable key.
        frames = []
        for collection, source_file in parameters["templates"].items():
            frame = pd.read_json(self.raw_dir / source_file, lines=True)
            frame = frame.join(frame.custom_id.str.extract(parameters["patterns"]["custom_id"]))
            prefix = parameters["template_prefixes"][collection]
            frame["template_id"] = frame.family + "-" + prefix + frame.language + "-" + frame["index"]
            mmlu, wino = frame.family.str.startswith("mmlu-"), frame.family.eq("winogrande")
            frame.loc[mmlu, "template_id"] = "mmlu-" + prefix + frame.loc[mmlu, "language"] + "-" + frame.loc[mmlu, "family"].str.removeprefix("mmlu-") + "-test-" + frame.loc[mmlu, "index"]
            frame.loc[wino, "template_id"] = "winogrande-" + prefix + frame.loc[wino, "language"] + "-test-" + frame.loc[wino, "index"]
            frame["messages"] = frame.body.map(lambda body: body["messages"])
            frame["message"] = frame.messages.map(lambda messages: messages[0]["content"])
            frames.append(frame.assign(prefix=prefix, locale=frame.language, template_file=source_file))
        templates = pd.concat(frames, ignore_index=True)
        belebele = templates.loc[templates.family.eq("belebele")].copy()
        belebele = belebele.join(belebele.message.str.extract(parameters["patterns"]["belebele"]))
        inputs = bank.loc[bank.family.eq("belebele")].copy()
        inputs["prefix"] = inputs.item_key.str.extract(r"^belebele-(gt-|bt-)?")[0].fillna("")
        inputs["locale"] = inputs.item_key.str.extract(r"^belebele-(?:gt-|bt-)?([a-z]+)-")[0]
        fields = ["locale", "prefix", "question", "passage", "gold"] + [f"option{i}" for i in range(1, 5)]
        for column in [f"option{i}" for i in range(1, 5)]:
            missing = inputs[column].eq("")
            if not missing.any():
                continue
            keys = [key for key in fields if key != column]
            restored = inputs.loc[missing, keys + ["item_key"]].merge(
                belebele[fields], on=keys, how="left", validate="one_to_many")
            if restored.item_key.duplicated().any() or restored[column].isna().any():
                raise ValueError("An incomplete answer option has no unique matching template")
            values = restored.set_index("item_key")[column]
            inputs.loc[missing, column] = inputs.loc[missing, "item_key"].map(values)
            selected = bank.item_key.isin(values.index)
            bank.loc[selected, column] = bank.loc[selected, "item_key"].map(values)
        matched = belebele.merge(inputs, on=fields, how="left", suffixes=("_template", "_bank"), validate="one_to_one")
        if matched.item_key.isna().any():
            raise ValueError("A Belebele template does not match its complete original task")
        mapping = matched.set_index("template_id").item_key
        templates["item_key"] = templates.template_id.map(mapping).fillna(templates.template_id)
        bank = bank.merge(templates[["item_key", "template_id", "template_file", "messages", "message", "gold"]],
            on="item_key", how="left", suffixes=("", "_template"), validate="one_to_one")
        if bank.messages.isna().any() or not bank.gold.eq(bank.gold_template).all():
            bad = bank.loc[bank.messages.isna() | bank.gold.ne(bank.gold_template), ["item_key", "gold", "gold_template"]]
            raise ValueError(f"Items without matching complete templates: {bad.head().to_dict('records')}")
        if not master.gold.eq(master.item_key.map(bank.set_index("item_key").gold)).all():
            raise ValueError("A consolidated reference disagrees with its item")
        for row in bank.loc[~bank.family.eq("belebele")].to_dict("records"):
            family = "winogrande" if row["family"] == "winogrande" else "mmlu"
            if not row["message"].endswith(parameters["target_templates"][family].format(**row)):
                raise ValueError("Request target differs from item bank: " + row["item_key"])

        # 4. Read genuine native runs and preserve the legacy Winogrande verdicts.
        frames = []
        sources = [(path, "paper") for path in sorted((self.raw_dir / paths["paper_runs"]).glob("*.jsonl"))]
        sources += [(path, "auxiliary_example") for path in sorted((self.raw_dir / paths["examples"]).glob("*.json*"))]
        for path, collection in sources:
            if path.suffix == ".jsonl":
                frame = pd.read_json(path, lines=True)
                frame["record"] = frame.astype(object).where(frame.notna(), None).to_dict("records")
                if not frame.response.map(lambda value: value["status_code"] == 200).all() or not frame.error.isna().all():
                    raise ValueError("Unsuccessful native provider record needs review")
                frame["output"] = frame.response.map(lambda value: value["body"]["choices"][0]["message"]["content"])
            else:
                frame = pd.read_json(path, typ="series").rename_axis("custom_id").reset_index(name="output")
                frame["record"] = frame.to_dict("records")
            frame = frame[["custom_id", "output", "record"]].join(frame.custom_id.str.extract(parameters["patterns"]["custom_id"]))
            frame["model"] = frame.model.replace(parameters["model_aliases"])
            frame["item_key"] = frame.family + "-" + frame.language + "-" + frame["index"]
            mmlu, wino = frame.family.str.startswith("mmlu-"), frame.family.eq("winogrande")
            frame.loc[mmlu, "item_key"] = "mmlu-" + frame.loc[mmlu, "language"] + "-" + frame.loc[mmlu, "family"].str.removeprefix("mmlu-") + "-test-" + frame.loc[mmlu, "index"]
            frame.loc[wino, "item_key"] = "winogrande-" + frame.loc[wino, "language"] + "-test-" + frame.loc[wino, "index"]
            if collection == "auxiliary_example":
                frame["item_key"] = frame.item_key.map(templates.set_index("template_id").item_key)
            frame["trial"] = int(path.stem.rsplit("_", 1)[-1]) + 1 if collection == "paper" else 1
            frames.append(frame.assign(collection=collection, source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index + 1))
        native = pd.concat(frames, ignore_index=True)
        native["subject_key"] = native.model + ":"
        native = native.merge(bank[["item_key", "gold", "winogrande_id"]], on="item_key", how="left", suffixes=("", "_bank"), validate="many_to_one")
        if native.gold_bank.isna().any() or not native.gold.eq(native.gold_bank).all():
            raise ValueError("A native run does not match the reference of its assigned item")
        native["response"] = native.output.str.strip().str.replace("(", "", regex=False).str.replace(")", "", regex=False).str.upper().str[:1].eq(native.gold).astype(float)
        wino = native.family.eq("winogrande")
        native.loc[wino, "response"] = [float(gold in output and str(3 - int(gold)) not in output)
            for output, gold in native.loc[wino, ["output", "gold"]].itertuples(index=False, name=None)]
        matrices = []
        for path in sorted((self.raw_dir / paths["legacy_verdicts"]).glob("wino_evaluation_results_*.csv")):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns=parameters["matrix_columns"])
            frame["matrix_row"] = frame.index + 2
            frame = frame.melt(id_vars=["winogrande_id", "matrix_reference", "matrix_row"], var_name="matrix_column", value_name="matrix_grade")
            frame[["model", "language"]] = frame.matrix_column.str.rsplit("_", n=1, expand=True)
            frame["model"] = frame.model.map(parameters["legacy_model_aliases"])
            frame["matrix_grade"] = pd.to_numeric(frame.matrix_grade, errors="raise")
            matrices.append(frame.assign(trial=int(path.stem.rsplit("_", 1)[-1]) + 1, matrix_file=str(path.relative_to(self.raw_dir))))
        native = native.merge(pd.concat(matrices), on=["model", "language", "winogrande_id", "trial"], how="left", validate="many_to_one")
        legacy = native.collection.eq("paper") & native.family.eq("winogrande")
        if not native.loc[legacy, "matrix_grade"].isin([0, 1]).all() or not native.loc[legacy, "matrix_reference"].eq(native.loc[legacy, "gold"]).all():
            raise ValueError("A paper Winogrande run has no matching original matrix verdict")
        native.loc[legacy, "response"] = native.loc[legacy, "matrix_grade"]
        native["protocol"] = "multiple_choice"
        native.loc[wino, "protocol"] = "winogrande"
        native.loc[legacy, "protocol"] = "legacy_winogrande"

        # 5. Restore matched outputs once; retain all additional runs and complete source records.
        first = native.loc[native.collection.eq("paper") & native.trial.eq(1)].copy()
        restored = master.merge(first[["subject_key", "item_key", "trial", "gold", "output", "response", "source_file", "source_row", "record", "protocol"]],
            on=["subject_key", "item_key", "trial"], how="left", suffixes=("", "_native"), validate="one_to_one")
        present = restored.record_native.notna()
        if int(present.sum()) != len(first) or not (restored.loc[present, "output"].eq(restored.loc[present, "output_native"].str[:12]) &
                restored.loc[present, "gold"].eq(restored.loc[present, "gold_native"]) & restored.loc[present, "response"].eq(restored.loc[present, "response_native"])).all():
            raise ValueError("Full native output cannot be reconciled with the consolidated observation")
        restored["protocol"] = restored.protocol.fillna(restored.family.map(lambda value: "winogrande" if value == "winogrande" else "multiple_choice"))
        restored["test_condition"] = "collection=paper"
        restored["response_key"] = "consolidated:" + restored.source_row.astype(str)
        restored = restored.astype(object).where(restored.notna(), None)
        restored["trace"] = [json.dumps(dict(consolidated=dict(file=paths["responses"], row=row.source_row, record=row.record),
            native=None if row.record_native is None else dict(file=row.source_file, row=row.source_row_native, record=row.record_native)),
            ensure_ascii=False, allow_nan=False) for row in restored.itertuples()]
        extra = native.loc[~(native.collection.eq("paper") & native.trial.eq(1))].copy()
        extra["test_condition"] = "collection=" + extra.collection
        extra["response_key"] = extra.source_file + ":" + extra.source_row.astype(str)
        extra = extra.astype(object).where(extra.notna(), None)
        extra["trace"] = [json.dumps(dict(consolidated=None, native=dict(file=row.source_file, row=row.source_row, record=row.record),
            verdict=dict(kind="legacy_matrix", file=row.matrix_file, row=row.matrix_row, column=row.matrix_column) if row.protocol == "legacy_winogrande"
            else dict(kind="released_answer_parser", protocol=row.protocol)), ensure_ascii=False, allow_nan=False) for row in extra.itertuples()]
        columns = ["response_key", "subject_key", "item_key", "response", "trial", "test_condition", "protocol", "trace"]
        observations = pd.concat([restored[columns], extra[columns]], ignore_index=True)
        definitions = observations[["item_key", "protocol"]].drop_duplicates().merge(bank, on="item_key", validate="many_to_one")
        definitions["raw_item_id"] = definitions.item_key + ":" + definitions.protocol
        definitions["features"] = [dict(source_item_id=row.item_key, family=row.family, source_language=row.source_language,
            target_language=row.language, translation=row.translation, partition=row.partition,
            template_id=row.template_id, template_file=row.template_file) for row in definitions.itertuples()]
        definitions["content"] = definitions.messages.map(lambda messages: json.dumps(messages, ensure_ascii=False))
        definitions["grading_criterion"] = [dict(reference_answer=row.gold, rule=self.grading["verifiers"][row.protocol]["rule"]) for row in definitions.itertuples()]
        definitions["verifier"] = definitions.protocol.map(lambda name: ExactMatcher(spec=json.dumps(self.grading["verifiers"][name], sort_keys=True)))
        definitions["item_key"] = definitions.raw_item_id
        observations["item_key"] = observations.item_key + ":" + observations.protocol
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": definitions[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": observations[["response_key", "trace"]],
        }


if __name__ == "__main__":
    BridgingGap(__file__).main_from_args()
