"""Tabulate the released option logits and their documented input configurations."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class LLMUncertaintyBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, formatting = parameters["layout"], parameters["formatting"]
        repository = self.raw_dir / layout["repository"]

        # 1. Load native JSON and numeric-pickle records directly into tables.
        question_tables = [pd.json_normalize(json.loads(path.read_text()), max_level=0).assign(dataset=path.stem)
            for path in sorted(repository.glob(layout["questions"]))]
        questions = pd.concat(question_tables, ignore_index=True)
        paths = pd.DataFrame({"path": sorted(repository.glob(layout["results"]))})
        names = paths.path.map(lambda path: path.name).str.extract(layout["result_pattern"])
        if names.isna().any().any():
            raise ValueError("Unrecognized upstream result filename")
        paths = paths.join(names).assign(variant=paths.path.map(lambda path: path.parent.name))
        frames = []
        for row in paths.itertuples():
            frame = pd.DataFrame(read_native_pickle(row.path))
            frames.append(frame.assign(model=row.model, dataset=row.dataset, method=row.method,
                variant=row.variant, source_position=range(len(frame)), source_file=str(row.path.relative_to(repository))))
        attempts = pd.concat(frames, ignore_index=True)
        if questions.duplicated(["dataset", "id"]).any() or attempts.duplicated(["source_file", "id"]).any():
            raise ValueError("Duplicate original question or result ID")

        # 2. Reconstruct the documented pre-tokenization prompts, including demonstrations.
        contexts = questions.source.map(parameters["context_prefixes"])
        questions["example"] = (contexts + questions.context.fillna("") + "\n").where(contexts.ne(""), "")
        questions["example"] += formatting["question_prefix"] + questions.question + formatting["choices_prefix"]
        questions["example"] += questions.choices.map(lambda choices: "".join(key + ". " + str(value) + "\n" for key, value in choices.items()))
        questions["example"] += formatting["answer_suffix"]
        demo_keys = pd.Series(parameters["demonstrations"]).str.split(",").rename_axis("dataset").rename("id").explode().reset_index()
        demo_keys["id"] = demo_keys.id.astype(int)
        demos = demo_keys.merge(questions[["dataset", "id", "example", "answer"]], on=["dataset", "id"], validate="one_to_one")
        demos["text"] = demos.example + " " + demos.answer + "\n"
        examples = demos.groupby("dataset", sort=False).text.sum()
        settings = paths[["dataset", "method"]].drop_duplicates().sort_values(["dataset", "method"])
        settings["prefix"] = settings.dataset.map(examples)
        shared, task = settings.method.eq("shared"), settings.method.eq("task")
        settings.loc[shared, "prefix"] = formatting["shared_prefix"] + settings.loc[shared, "prefix"] + formatting["following_question"]
        task_headers = settings.loc[task, "dataset"].map(parameters["datasets"]).map(parameters["task_prefixes"])
        settings.loc[task, "prefix"] = task_headers + settings.loc[task, "prefix"] + formatting["following_question"]
        items = settings.merge(questions, on="dataset", validate="many_to_many")
        items["content"] = items.prefix + items.example
        items["item_key"] = items.dataset + ":" + items.id.astype(str) + ":" + items.method
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(dataset=row.dataset, prompt_method=row.method,
            demonstration_ids=list(map(int, parameters["demonstrations"][row.dataset].split(","))), input_scope=parameters["labels"]["input_scope"])
            for row in items.itertuples()]
        items["grading_criterion"] = items.answer.map(lambda answer: dict(reference_answer=answer, rule=self.grading["rule"]))
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["option_argmax"], sort_keys=True))

        # 3. Separate recorded model/prompt formats without guessing historical revisions.
        subjects = paths[["model", "variant", "method"]].drop_duplicates().sort_values(["model", "variant", "method"])
        subjects["subject_key"] = subjects.model + ":" + subjects.variant + ":" + subjects.method
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=row.model,
            input_variant=row.variant, prompt_method=row.method, input_format=parameters["variants"][row.variant],
            chat_template=json.dumps(formatting["falcon_chat"]) if row.variant == "outputs_chat_v1" and "falcon" in row.model else None,
            option_token_strings=json.dumps(list(parameters["options_chat_yi"].values()) if row.variant == "outputs_chat_v1" and "Yi" in row.model else list(parameters["options_default"].values())))
            for row in subjects.itertuples()]

        # 4. Join by native IDs and reproduce the original first-maximum grade.
        attempts = attempts.merge(items[["dataset", "id", "method", "item_key", "answer"]],
            on=["dataset", "id", "method"], how="left", validate="many_to_one")
        if attempts.item_key.isna().any():
            raise ValueError("A source logit record has no question definition")
        attempts = attempts.merge(subjects[["model", "variant", "method", "subject_key"]],
            on=["model", "variant", "method"], validate="many_to_one")
        logits = np.stack(attempts.logits_options)
        if logits.shape != (len(attempts), len(parameters["options_default"])) or not np.isfinite(logits).all():
            raise ValueError("Original logits must be finite six-option vectors")
        attempts["predicted_option"] = np.asarray(list(parameters["options_default"]))[logits.argmax(axis=1)]
        attempts["response"] = attempts.predicted_option.eq(attempts.answer).astype(float)
        attempts["response_key"] = attempts.source_file + "#" + attempts.id.astype(str)
        attempts["test_condition"] = "dataset=" + attempts.dataset + ";prompt=" + attempts.method + ";variant=" + attempts.variant.str.removeprefix("outputs_")

        # 5. Keep every source vector and ID; generated prose was not released.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_position=row.source_position,
            source_item_id=row.id, logits_options=row.logits_options.tolist(), logits_dtype=str(row.logits_options.dtype),
            predicted_option=row.predicted_option), allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LLMUncertaintyBench(__file__).main_from_args()
