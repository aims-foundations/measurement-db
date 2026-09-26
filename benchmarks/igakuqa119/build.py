"""Tabulate IgakuQA119's native answers, exact grading and original exam resources."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class IgakuQA119(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]
        protocol = self.grading["verifiers"]["answer_match"]

        # 1. Read native JSON tables, the answer key and unchanged image bytes.
        with tarfile.open(self.raw_dir / paths["archive"]) as archive:
            members = pd.DataFrame({"member": sorted(member.name for member in archive if member.isfile())})
            members["source_file"] = members.member.str.removeprefix(paths["root"] + "/")
            answers = members.loc[members.source_file.str.startswith(paths["answers"]) & members.source_file.str.endswith(".json")].copy()
            answers["document"] = [json.load(archive.extractfile(name)) for name in answers.member]
            questions = members.loc[members.source_file.str.startswith(paths["questions"]) & members.source_file.str.endswith(".json")].copy()
            questions["bank_record"] = [json.load(archive.extractfile(name)) for name in questions.member]
            references = pd.read_csv(archive.extractfile(paths["root"] + "/" + paths["references"]), dtype=str, keep_default_na=False)
            resources = members.loc[members.source_file.str.startswith(paths["images"])].copy()
            resources["data"] = [archive.extractfile(name).read() for name in resources.member]

        # 2. Expand saved question/answer records and join their source definitions.
        answers["experiment_id"] = answers.document.str["experiment_id"]
        answers["source_record"] = answers.document.str["results"]
        answers = answers.explode("source_record", ignore_index=True)
        answers["source_row"] = answers.groupby("source_file", sort=False).cumcount()
        answers = answers.join(pd.json_normalize(answers.source_record, max_level=0))
        answers = answers.explode("answers", ignore_index=True).rename(columns={"answers": "native_answer", "question_number": "raw_item_id"})
        answers["answer_index"] = answers.groupby(["source_file", "source_row"], sort=False).cumcount()
        answers = answers.join(pd.json_normalize(answers.native_answer, max_level=0))
        questions = questions.explode("bank_record", ignore_index=True)
        questions["raw_item_id"] = questions.bank_record.str["number"]
        answers = answers.merge(questions[["raw_item_id", "source_file", "bank_record"]].rename(columns={"source_file": "bank_file"}),
            on="raw_item_id", how="left", validate="many_to_one")
        references = references.rename(columns={"問題番号": "raw_item_id", "解答": "reference"})
        answers = answers.merge(references, on="raw_item_id", how="left", validate="many_to_one")
        if answers[["model", "answer", "reference", "bank_record"]].isna().any().any() or not answers.model.isin(parameters["models"]).all():
            raise ValueError("A released answer has no supported model, source definition or reference")
        answers["subject_key"] = answers.model
        answers["item_key"] = answers.source_file + "/" + answers.source_row.astype(str)
        answers["response_key"] = answers.item_key + "/" + answers.answer_index.astype(str)

        # 3. Match the upstream grader, including digits and accepted alternatives.
        translation = str.maketrans("", "", protocol["removed_characters"])
        normalized = answers[["answer", "reference"]].apply(lambda column:
            column.str.strip().str.lower().str.translate(translation).map(sorted).str.join(""))
        correct = normalized.answer.ne("") & normalized.answer.eq(normalized.reference)
        alternatives = pd.Series(protocol["accepted_alternatives"], name="answer").explode().rename_axis("raw_item_id").reset_index()
        special = answers.raw_item_id.isin(alternatives.raw_item_id)
        answer_keys = pd.MultiIndex.from_frame(pd.DataFrame({"raw_item_id": answers.raw_item_id, "answer": normalized.answer}))
        correct.loc[special] = answer_keys.isin(pd.MultiIndex.from_frame(alternatives))[special]
        answers["response"] = correct.astype(float)
        answers["test_condition"] = "experiment=" + answers.experiment_id

        # 4. Keep the saved text version; archive images as resources, not sent inputs.
        resources["raw_item_id"] = resources.source_file.str.extract(r"/(119[A-F]\d+)(?:-\d+)?\.(?:jpg|png)$", expand=False)
        resources["media_type"] = resources.source_file.map(lambda path: parameters["image_types"].get(Path(path).suffix))
        if resources[["raw_item_id", "media_type"]].isna().any().any() or not resources.raw_item_id.isin(answers.raw_item_id).all():
            raise ValueError("An original image has no matching question or media type")
        resources["attachment"] = [dict(data=row.data, path=row.source_file, media_type=row.media_type,
            role=parameters["presentation"]["resource_role"]) for row in resources.itertuples()]
        resources = resources.groupby("raw_item_id", sort=False).agg(attachments=("attachment", list)).reset_index()
        items = answers.drop_duplicates("item_key").merge(resources, on="raw_item_id", how="left", validate="many_to_one")
        items["attachments"] = items.attachments.map(lambda value: value if isinstance(value, list) else [])
        user_text = [parameters["prompts"]["user"].format(question=question, choices="\n".join(choices))
            for question, choices in zip(items.question_text, items.choices)]
        items["content"] = [json.dumps({"messages": [{"role": "system", "content": parameters["prompts"]["system"]},
            {"role": "user", "content": text}]}, ensure_ascii=False) for text in user_text]
        weights = parameters["point_weights"]
        points = pd.Series(int(weights["ordinary"]), index=items.index)
        required = items.raw_item_id.str[3].isin(list(weights["required_blocks"])) & items.raw_item_id.str[4:].astype(int).between(int(weights["three_point_start"]), int(weights["three_point_end"]))
        points.loc[required] = int(weights["required"])
        items["features"] = [dict(source_question_has_image=str(row.has_image).lower(), exam_points=str(point),
            saved_question_text_missing=str(not row.question_text).lower(), image_delivery=parameters["presentation"]["historical_vision"],
            input_scope=parameters["presentation"]["input_scope"]) for row, point in zip(items.itertuples(), points)]
        items["grading_criterion"] = [dict(reference_answer=row.reference, rule=self.grading["rule"] +
            "\nAccepted alternatives for this question: " + json.dumps(protocol["accepted_alternatives"].get(row.raw_item_id))) for row in items.itertuples()]
        items["verifier"] = ExactMatcher(spec=json.dumps(protocol, sort_keys=True))

        # 5. Preserve complete native parsed outputs, task-bank differences and provenance.
        subjects = answers[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.map(parameters["models"])
        subjects["features"] = [dict(**parameters["subject_features"], model_identifier=model) for model in subjects.subject_key]
        traces = answers[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row), answer_index=int(row.answer_index),
            experiment_id=row.experiment_id, source_record=row.source_record, bank_file=row.bank_file, bank_record=row.bank_record,
            saved_question_text_missing=not bool(row.question_text), image_delivery=parameters["presentation"]["historical_vision"]),
            ensure_ascii=False, allow_nan=False) for row in answers.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": answers[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    IgakuQA119(__file__).main_from_args()
