"""Curate original circuit-homework transcriptions and available recognition judgments."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class EduCircuitHW(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "archive")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        keys = list(parameters["keys"].values())

        # 1. Concatenate the original judge CSVs, preserving their complete records.
        sources = []
        for path in sorted(self.raw_dir.glob(parameters["layout"]["results"])):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["judge_record"] = frame.to_dict("records")
            frame["judge_row"] = frame.index
            frame["judge_file"] = str(path.relative_to(self.raw_dir))
            frame["model"] = pd.Series(path.name, index=frame.index).str.extract(parameters["patterns"]["result_model"], expand=False)
            sources.append(frame.rename(columns=parameters["columns"]))
        judgments = pd.concat(sources, ignore_index=True)

        with ZipFile(self.raw_dir / parameters["layout"]["archive"]) as archive:
            samples = pd.read_csv(archive.open(parameters["layout"]["split"]), dtype=str,
                keep_default_na=False).rename(columns=parameters["columns"])[keys]
            samples["item_key"] = samples[keys].agg("::".join, axis=1)

            # 2. Read the released Markdown and join only observation-set samples.
            files = pd.Series(sorted(archive.namelist()), name="path").to_frame()
            documents = files.join(files.path.str.extract(parameters["patterns"]["transcript"]))
            documents = documents.dropna(subset=keys).merge(samples, on=keys, how="inner", validate="many_to_one")
            documents["markdown"] = [archive.read(path).decode("utf-8") for path in documents.path]
            documents["comparison"] = documents.markdown.str.split("\n", n=1).str[1].fillna("")
            documents["document"] = documents[["path", "markdown"]].to_dict("records")
            documents = documents.groupby(keys + ["item_key", "folder", "model"], sort=False).agg(
                documents=("document", list), comparison=("comparison", "".join)).reset_index()

            # 3. Prefer expert corrections, then the author's reviewed Gemini references.
            references = documents.loc[documents.folder.isin(parameters["reference_priority"])].copy()
            references["priority"] = references.folder.map(parameters["reference_priority"]).astype(int)
            references = references.sort_values("priority", kind="stable").drop_duplicates(keys)
            references = references[keys + ["comparison", "documents"]].rename(columns={
                "comparison": "reference_answer", "documents": "reference_documents"})
            observations = documents.loc[documents.folder.isin(parameters["model_folders"])].copy()
            if not observations.folder.map(parameters["model_folders"]).eq(observations.model).all():
                raise ValueError("A transcription folder has an unexpected model identity")
            observations = observations.merge(references, on=keys, how="left", validate="many_to_one")
            observations = observations.merge(judgments, on=keys + ["model"], how="left", validate="one_to_one")

            # 4. Attach every original PNG page; no resizing or judgment-derived input text.
            images = files.join(files.path.str.extract(parameters["patterns"]["image"]))
            images["question"] = images.question.str.replace(".", "_", n=1, regex=False)
            images = images.dropna(subset=keys).merge(samples, on=keys, how="inner", validate="many_to_one")
            images["attachment"] = [dict(path=path, data=archive.read(path), media_type="image/png", role="input")
                for path in images.path]
            images["element"] = [dict(content_type="image/png", location=path) for path in images.path]
            images = images.groupby("item_key", sort=False).agg(attachments=("attachment", list), elements=("element", list))
            items = samples.merge(images, on="item_key", how="left", validate="one_to_one").merge(
                references, on=keys, how="left", validate="one_to_one")

        # 5. Preserve attempts without grades, including explicit judge API failures.
        unavailable = observations.judgment.isna() | observations.judgment.str.startswith(parameters["grading_text"]["exception"], na=False)
        observations["response"] = observations.judgment.str.strip().str.lower().str.startswith(
            parameters["grading_text"]["no_error"], na=False).astype(float).mask(unavailable)
        observations["grade_status"] = "released_judgment"
        observations.loc[observations.judgment.isna(), "grade_status"] = "judgment_not_released"
        observations.loc[observations.judgment.str.startswith(parameters["grading_text"]["exception"], na=False), "grade_status"] = "judge_api_exception"
        observations["subject_key"] = observations.model
        observations["response_key"] = observations.model + "/" + observations.item_key
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(multimedia_elements=[dict(content_type="text/plain", text=parameters["task"]["instruction"])]
            + elements), ensure_ascii=False) for elements in items.elements]
        items["grading_criterion"] = [dict(reference_answer=text, rule=self.grading["rule"]) for text in items.reference_answer]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["recognition"], sort_keys=True)) for _ in items.index]
        items["features"] = [dict(input_scope=parameters["input_scope"]["description"]) for _ in items.index]
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.map(parameters["model_labels"])
        subjects["features"] = [dict(**parameters["subject_features"], model_identifier=model) for model in subjects.subject_key]

        # 6. Keep full model outputs, reference provenance and original judge records together.
        traces = observations[["response_key", "model", "item_key", "documents", "reference_documents",
            "judge_file", "judge_row", "judge_record", "grade_status"]].copy()
        traces["judge_row"] = traces.judge_row.astype("Int64")
        traces = traces.astype(object).where(traces.notna(), None)
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EduCircuitHW(__file__).main_from_args()
