#!/usr/bin/env python3
"""Curate FinanceBench's released human reviews and source filing attachments."""

from io import BytesIO
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class FinanceBenchBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("provider_archive", "paper_v1", "documents")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read upstream JSONL members directly into tables; leave the archive intact.
        paths = self.build_parameters["paths"]
        with tarfile.open(self.raw_dir / paths["archive"], "r:gz") as archive:
            questions = pd.read_json(BytesIO(archive.extractfile(
                paths["root"] + "/" + paths["questions_member"]
            ).read()), lines=True, dtype=False)
            result_files = sorted(
                member.name for member in archive.getmembers()
                if member.isfile() and PurePosixPath(member.name).parent.as_posix()
                == paths["root"] + "/" + paths["results_prefix"] and member.name.endswith(".jsonl")
            )
            frames = [pd.read_json(BytesIO(archive.extractfile(name).read()), lines=True, dtype=False).assign(
                source_file=PurePosixPath(name).name
            ) for name in result_files]
        results = pd.concat(frames, ignore_index=True)
        if questions.financebench_id.duplicated().any():
            raise ValueError("FinanceBench question IDs must be unique")
        if results.duplicated(["source_file", "financebench_id"]).any():
            raise ValueError("FinanceBench repeats a question within one released result file")
        labels = results.label.isin(["Correct Answer", "Incorrect Answer", "Refusal"])
        if not labels.all() or results.model_answer.isna().any():
            raise ValueError("FinanceBench has an unknown label or a missing model answer")

        # 2. Attach each answer to its question and check the provider's repeated prompt text.
        observations = results.merge(
            questions[["financebench_id", "question"]].rename(columns={"question": "item_question"}),
            on="financebench_id", how="left", sort=False, validate="many_to_one", indicator=True
        )
        if observations._merge.ne("both").any() or observations.question.ne(observations.item_question).any():
            raise ValueError("FinanceBench result and question definitions disagree")
        if not results.temp.eq(0.01).all():
            raise ValueError("FinanceBench run temperature differs from the recorded release")

        # 3. Construct model/context configurations using their recorded run fields.
        settings = self.build_parameters["settings"]
        subjects = results[["source_file", "model_name", "eval_mode"]].drop_duplicates()
        if subjects.source_file.duplicated().any():
            raise ValueError("A FinanceBench result file contains multiple subject configurations")
        features = subjects.rename(columns={"eval_mode": "evaluation_mode", "model_name": "endpoint_label"})[
            ["evaluation_mode", "endpoint_label"]
        ].assign(harness=settings["harness"], max_output_tokens=int(settings["max_output_tokens"]),
                 system_prompt=settings["system_prompt"])
        features["replicate_target"] = subjects.model_name.map(
            lambda model: settings["replicate_target"] if model == "llama2" else None
        )
        subjects = subjects.assign(
            subject_key=subjects.source_file, raw_label=subjects.model_name,
            access_date=settings["access_date"],
            features=[{key: value for key, value in row.items() if pd.notna(value)}
                      for row in features.to_dict("records")],
        )

        # 4. Link question text to its released reference answer and exact filing bytes.
        items = questions.assign(
            item_key=questions.financebench_id,
            raw_item_id=questions.financebench_id,
            content=questions.question,
            grading_criterion=questions.answer.map(lambda answer: {
                "reference_answer": answer, "rule": self.grading["rule"]
            }),
            verifier=Judge(spec=self.grading["verifiers"]["human_review"]["spec"], judged_by="human"),
            attachments=questions.doc_name.map(lambda name: [{
                "source_path": "source_documents/" + name + ".pdf", "path": "pdfs/" + name + ".pdf",
                "media_type": "application/pdf", "role": "source_document"
            }]),
        )
        responses = observations.assign(
            response_key=observations.index, subject_key=observations.source_file,
            item_key=observations.financebench_id, response=observations.label.eq("Correct Answer").astype(float),
            trial=1, test_condition="temperature=0.01",
        )
        traces = responses[["response_key", "model_answer"]].rename(columns={"model_answer": "trace"})
        traces["trace"] = traces.trace.astype(str)
        return {
            "subjects": subjects[["subject_key", "raw_label", "features", "access_date"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    FinanceBenchBuild(__file__).main_from_args()
