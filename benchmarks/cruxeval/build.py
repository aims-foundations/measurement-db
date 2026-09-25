#!/usr/bin/env python3
"""Curate CRUXEval's released per-generation verdicts without executing code."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CRUXEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the task bank and the original scored exports as tables.
        bank = pd.read_json(self.raw_dir / parameters["paths"]["tasks"], lines=True, dtype=False).rename(columns={"id": "sample_id"})
        frames = []
        with ZipFile(self.raw_dir / parameters["paths"]["results"]) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json"):
                    continue
                source = json.loads(archive.read(member))
                table = pd.DataFrame({"response": pd.Series(source["raw_scored_generations"]),
                    "generation": pd.Series(source["raw_generations"])}).rename_axis("sample_id").reset_index()
                if not table.response.str.len().eq(table.generation.str.len()).all():
                    raise ValueError("CRUXEval verdicts do not align with their native generations")
                frames.append(table.assign(configuration=Path(member).stem, source_member=member,
                    pass_at_1=source["pass_at_1"], pass_at_5=source["pass_at_5"]))
        if not frames:
            raise ValueError("CRUXEval has no scored native exports")
        records = pd.concat(frames, ignore_index=True)

        # 2. Expand the aligned verdict/output lists and their recorded settings.
        records = records.explode(["response", "generation"], ignore_index=True)
        records["trial"] = records.groupby(["configuration", "sample_id"], sort=False).cumcount() + 1
        records = pd.concat([records, records.configuration.str.extract(parameters["patterns"]["configuration"])], axis=1)
        if records[["model", "temperature", "task"]].isna().any().any():
            raise ValueError("CRUXEval export has unknown configuration syntax")
        if not records.response.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("CRUXEval requires original boolean execution verdicts")
        records["response"] = records.response.astype(float)
        records["prompting"] = "direct"
        records.loc[records.model.str.endswith("+cot"), "prompting"] = "cot"
        records["prompt_variant"] = records.prompting + "_" + records.task
        overrides = (records.model + "_" + records.task).map(parameters["prompt_overrides"])
        records["prompt_variant"] = overrides.fillna(records.prompt_variant)
        records["subject_key"] = records.model
        records["item_key"] = records.prompt_variant + "/" + records.sample_id

        # 3. Reconstruct task prompts from the published templates, including
        # examples and the separate Phind output format. This is not a request log.
        items = records[["item_key", "sample_id", "task", "prompt_variant"]].drop_duplicates().merge(
            bank, on="sample_id", how="left", validate="many_to_one")
        if items[["code", "input", "output"]].isna().any().any():
            raise ValueError("CRUXEval result references a task absent from the captured bank")
        items["raw_item_id"] = items.item_key
        items["content"] = [parameters["prompt_templates"][row.prompt_variant].format(
            code=row.code, input=row.input, output=row.output) for row in items.itertuples()]
        items["features"] = [dict(task=row.task, source_task_id=row.sample_id,
            prompt_variant=row.prompt_variant, prompt_source="reconstructed_from_released_template") for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer="f(" + row.input + ")" if row.task == "input" else row.output,
            rule=json.dumps(dict(protocol=self.grading["verifiers"][row.task]["rule"], code=row.code,
                reference_input=row.input, reference_output=row.output), ensure_ascii=False, sort_keys=True))
            for row in items.itertuples()]
        items["verifier"] = items.task.map(lambda task: ExactMatcher(
            spec=json.dumps(self.grading["verifiers"][task], sort_keys=True)))

        # 4. Prompting distinguishes model configurations; temperature is an
        # observation condition, and prediction direction belongs to the item.
        subjects = records[["subject_key", "model", "prompting"]].drop_duplicates()
        subjects["raw_label"] = subjects.model.str.removesuffix("+cot")
        subjects["features"] = [dict(harness=self.name, reported_model=row.model,
            prompting=row.prompting,
            api_system_prompt=parameters["system_prompts"]["openai"] if row.model.startswith("gpt-") else "not_applicable")
            for row in subjects.itertuples()]

        # 5. Keep every recorded verdict and complete postprocessed generation.
        # Empty/invalid outputs retain the original failed verdict, not a null.
        records["response_key"] = records.configuration + "/" + records.sample_id + "/" + records.trial.astype(str)
        records["test_condition"] = "temperature=" + records.temperature + "; source_configuration=" + records.configuration
        responses = records[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]]
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_archive=parameters["paths"]["results"], source_member=row.source_member,
            sample_id=row.sample_id, generation_index=row.trial - 1, generation=row.generation,
            recorded_verdict=bool(row.response), recorded_pass_at_1=row.pass_at_1, recorded_pass_at_5=row.pass_at_5,
            trace_scope="released_postprocessed_generation"), ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CRUXEval(__file__).main_from_args()
