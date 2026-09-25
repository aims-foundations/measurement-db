#!/usr/bin/env python3
"""Curate BenGER's recorded generations, original graders and complete legal task inputs."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BenGER(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "derivation", "decision_rule")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        protocols = self.grading["verifiers"]

        # 1. Read the three native task tables; retain only cleared ZJS case texts.
        with ZipFile(self.raw_dir / parameters["paths"]["archive"]) as archive:
            tasks = pd.concat([pd.DataFrame(json.loads(archive.read(member))["tasks"], dtype=object)
                .assign(corpus=corpus, source_member=member)
                for corpus, member in parameters["exports"].items()], ignore_index=True)
        tasks = tasks.rename(columns={"id": "task_id", "data": "task_data", "meta": "task_metadata"})
        cleared = tasks.task_data.str["ip_cleared"].eq(True)
        tasks = tasks.loc[tasks.corpus.ne("zjs") | cleared].reset_index(drop=True)
        if tasks.duplicated(["corpus", "task_id"]).any():
            raise ValueError("Duplicate native task identifiers")

        # 2. Expand generations and select the documented grading configuration.
        observations = tasks[["corpus", "source_member", "task_id", "task_data", "task_metadata", "generations"]].explode(
            "generations", ignore_index=True).rename(columns={"generations": "generation"})
        generations = pd.DataFrame(observations.generation.tolist(), dtype=object).rename(columns={"id": "generation_id"})
        observations = observations.join(generations)
        observations["response_key"] = observations.corpus + ":" + observations.generation_id
        if observations.response_key.duplicated().any():
            raise ValueError("Duplicate native generation identifiers")
        evaluations = observations[["response_key", "corpus", "evaluations"]].explode("evaluations", ignore_index=True)
        evaluations = evaluations.dropna(subset=["evaluations"]).reset_index(drop=True)
        evaluations = evaluations.join(pd.DataFrame(evaluations.evaluations.tolist(), dtype=object).add_prefix("grade_"))
        expected = evaluations.corpus.map(parameters["grading_fields"])
        evaluations = evaluations.loc[evaluations.grade_field_name.str.split("|", regex=False).str[0].eq(expected)]
        if evaluations.response_key.duplicated().any():
            raise ValueError("More than one recorded grade for the selected configuration")
        grade_fields = evaluations[["response_key", "grade_id", "grade_prediction", "grade_metrics", "grade_judge_model"]]
        observations = observations.merge(grade_fields, on="response_key", how="left", validate="one_to_one")
        metrics = pd.json_normalize(observations.grade_metrics.map(lambda value: value if isinstance(value, dict) else {}))
        passed = metrics["llm_judge_falloesung.details.passed"]
        if not passed.dropna().map(lambda value: isinstance(value, bool)).all():
            raise ValueError("Recorded rubric pass judgments must be Boolean or unavailable")
        observations["response"] = passed.map({True: 1.0, False: 0.0})

        # The original parsed decisions also reproduce the author's normalized Ja/Nein rule.
        doctrinal = observations.corpus.eq("grundprinzipien")
        answer = observations.grade_prediction.str["value"].astype("string").str.lower().str.extract(
            parameters["decision"]["pattern"], expand=False)
        gold = observations.task_data.str["binary_solution"].astype("string").str.lower().str.extract(
            parameters["decision"]["pattern"], expand=False)
        if gold.loc[doctrinal].isna().any() or observations.loc[doctrinal, "grade_id"].isna().any():
            raise ValueError("The doctrinal reference or original parsed decision is unavailable")
        observations.loc[doctrinal, "response"] = answer.loc[doctrinal].eq(gold.loc[doctrinal]).fillna(False).astype(float)

        # 3. Separate recorded inference settings from task inputs and result metadata.
        settings = pd.DataFrame(observations.response_metadata.map(json.loads).tolist(), dtype=object)
        columns = list(parameters["subject_settings"].values())
        configurations = settings.reindex(columns=columns).astype(object).where(lambda frame: frame.notna(), None)
        observations["features"] = [dict(harness="BenGER", model_identifier=model, **configuration)
            for model, configuration in zip(observations.model_id, configurations.to_dict("records"))]
        observations["subject_key"] = observations.features.map(lambda value: json.dumps(value, sort_keys=True, allow_nan=False))
        subjects = observations[["subject_key", "model_id", "features"]].drop_duplicates("subject_key").rename(
            columns={"model_id": "raw_label"})
        observations["test_condition"] = "corpus=" + observations.corpus
        temperature = pd.to_numeric(settings.temperature, errors="raise")
        recorded_temperature = temperature.notna()
        observations.loc[recorded_temperature, "test_condition"] += (
            ";temperature=" + temperature.loc[recorded_temperature].map("{:g}".format))
        recorded_seed = settings.seed.notna()
        observations.loc[recorded_seed, "test_condition"] += ";seed=" + settings.loc[recorded_seed, "seed"].astype(str)
        observations["system_prompt"] = settings.system_prompt.where(settings.system_prompt.notna(), None)
        observations["instruction_prompt"] = settings.instruction_prompt.where(settings.instruction_prompt.notna(), None)
        recorded = observations.instruction_prompt.notna() & observations.instruction_prompt.ne("")
        observations["input_scope"] = recorded.map({True: "recorded_prompt", False: "published_task_only"})
        observations["content"] = [json.dumps(dict(messages=[dict(role="system", content=system),
            dict(role="user", content=instruction)]), ensure_ascii=False, sort_keys=True)
            for system, instruction in zip(observations.system_prompt, observations.instruction_prompt)]
        fallback = observations.loc[~recorded]
        observations.loc[~recorded, "content"] = [json.dumps(dict(task=data[parameters["fallback_inputs"][corpus]],
            scope="Published task text; the historical prompt was not recorded."), ensure_ascii=False, sort_keys=True)
            for corpus, data in zip(fallback.corpus, fallback.task_data)]

        # 4. Define each input with its source reference and named grading protocol.
        observations["item_key"] = observations.corpus + ":" + observations.task_id + ":" + observations.content
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.corpus + ":" + items.task_id
        items["features"] = [dict(corpus=row.corpus, source_task_id=row.task_id, input_scope=row.input_scope)
            for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.task_data[parameters["reference_fields"][row.corpus]],
            rule=protocols[row.corpus]["rule"]) for row in items.itertuples()]
        items["verifier"] = [Judge(judge=protocols[row.corpus]["judge"], judged_by=protocols[row.corpus]["judged_by"],
            spec=json.dumps(protocols[row.corpus], ensure_ascii=False, sort_keys=True)) for row in items.itertuples()]

        # 5. Retain full outputs, original grading records, prompts and source identifiers.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_archive=parameters["paths"]["archive"], source_member=row.source_member,
            task_id=row.task_id, task_data=row.task_data, task_metadata=row.task_metadata, generation=row.generation,
            selected_evaluation_id=None if pd.isna(row.grade_id) else row.grade_id), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BenGER(__file__).main_from_args()
