"""Curate the released review subset and its reported completion flags."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class EHRFlowBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("evaluation", "datasets", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]

        # 1. Normalize the public review payload into tasks, runs and attempts.
        snapshot = json.loads((self.raw_dir / paths["evaluation"]).read_text())["snapshot"]
        questions = pd.json_normalize(snapshot["questions"], max_level=0)
        runs = pd.json_normalize(snapshot["runs"], max_level=0)
        runs["run_record"] = runs.to_dict("records")
        attempts = pd.json_normalize(snapshot["questions"], record_path="candidates", meta="qid", max_level=0)
        attempts = attempts.astype(object).where(attempts.notna(), None)
        attempts["native_record"] = attempts.drop(columns="qid").to_dict("records")
        attempts = attempts.merge(runs[["id", "run_record"]].rename(columns={"id": "runId"}),
            on="runId", validate="many_to_one")

        # 2. Match every reviewed task to the fixed 100-task upstream release.
        with ZipFile(self.raw_dir / paths["datasets"]) as archive:
            tasks = pd.read_json(archive.open(paths["tasks"]), lines=True)
            tasks["qid"] = tasks.qid.astype(str)
            questions = questions.merge(tasks, on="qid", suffixes=("", "_released"), validate="one_to_one")
            if not questions.task.eq(questions.task_released).all():
                raise ValueError("Review prompts differ from the released benchmark tasks")

            # 3. Attach the original data and split files named by each task manifest.
            manifests = pd.json_normalize([json.loads(archive.read(paths["processed"] + path))
                for path in questions.reference_answer], max_level=0)
            manifests["qid"] = manifests.qid.astype(str)
            inputs = manifests[["qid", "required_inputs"]].explode("required_inputs")
            inputs["attachments"] = [dict(path=path.replace(paths["archive_prefix"], paths["task_prefix"], 1),
                data=archive.read(path), role="input", media_type=self.build_parameters["media_types"][Path(path).suffix])
                for path in inputs.required_inputs]
            inputs = inputs.groupby("qid", sort=False).attachments.agg(list).reset_index()
            questions = questions.merge(inputs, on="qid", validate="one_to_one")

        # 4. Preserve full reference reports and the published completion semantics.
        questions["item_key"] = questions.qid
        questions["raw_item_id"] = questions.datasetId + ":" + questions.qid
        questions["content"] = questions.task
        questions["grading_criterion"] = [dict(reference_answer=reference["text"], rule=self.grading["rule"])
            for reference in questions.reference]
        questions["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["completion"], sort_keys=True))
            for _ in questions.index]
        questions["features"] = questions[["dataset", "paper_id"]].to_dict("records")
        runs["subject_key"] = runs.id
        runs["raw_label"] = runs.label + " (" + runs.modelId + ")"
        variant = runs.modelId.str.startswith("variant=")
        runs["model_identifier"] = runs.modelId.mask(variant, None)
        runs["variant"] = runs.modelId.str.removeprefix("variant=").where(variant, None)
        settings = runs[["label", "model_identifier", "variant"]].rename(columns={"label": "harness"})
        settings = settings.astype(object).where(settings.notna(), None)
        runs["features"] = settings.assign(**self.build_parameters["subject_features"]).to_dict("records")

        # 5. Keep the exported Boolean flag; it is not the report-quality rubric score.
        if not attempts.success.map(type).eq(bool).all():
            raise ValueError("Every released attempt must have an explicit Boolean completion flag")
        attempts["response_key"] = attempts.id
        attempts["subject_key"] = attempts.runId
        attempts["item_key"] = attempts.qid
        attempts["response"] = attempts.success.astype(float)
        attempts["test_condition"] = self.build_parameters["test_condition"]["value"]
        attempts["trace"] = [json.dumps(dict(candidate=record, run=run), ensure_ascii=False, allow_nan=False)
            for record, run in zip(attempts.native_record, attempts.run_record)]
        return {
            "subjects": runs[["subject_key", "raw_label", "features"]],
            "items": questions[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": attempts[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EHRFlowBench(__file__).main_from_args()
