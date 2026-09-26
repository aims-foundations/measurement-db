"""Tabulate original tutoring-study grades, published worksheets and conversations."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class GenAILearning(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "paper")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        protocols = self.grading["verifiers"]

        # 1. Read the source tables directly, preserving every original CSV cell as text.
        with tarfile.open(self.raw_dir / parameters["paths"]["archive"]) as archive:
            root = parameters["paths"]["root"] + "/"
            grades = []
            for phase, path in parameters["student_files"].items():
                table = pd.read_csv(archive.extractfile(root + path), dtype=str, keep_default_na=False)
                table["record"] = table.to_dict("records")
                grades.append(table.assign(phase=phase, source_file=path, source_row=range(len(table))))
            students = pd.concat(grades, ignore_index=True)
            sources = {name: pd.read_csv(archive.extractfile(root + path), dtype=str, keep_default_na=False,
                encoding="latin1" if name == "questions" else "utf-8") for name, path in parameters["files"].items()}
        students = students.rename(columns={"Student ID": "participant", "Problem": "problem", "Treatment arm": "arm"})
        students["response"] = students.Score.map(float)
        students["subject_key"] = "student:" + students.participant
        students["item_key"] = students.phase + ":" + students.problem
        students["response_key"] = students.source_file + ":" + students.source_row.astype(str)
        students["test_condition"] = students.arm.map(parameters["conditions"])
        students.loc[students.phase.eq("exam"), "test_condition"] = parameters["conditions"]["exam"]

        # 2. Group original messages without changing characters or merging conversations.
        chats = sources["chats"].rename(columns={"Unnamed: 0": ""})  # Original blank CSV index header.
        chats["record"] = chats.to_dict("records")
        chats["problem"] = "s" + chats.session_id + "_" + chats.grade + "_" + chats.problem_id
        chats["phase"] = "practice"
        chats = chats.rename(columns={"username": "participant"})
        conversations = chats.groupby(["participant", "problem", "phase"], sort=False).agg(messages=("record", list)).reset_index()
        students = students.merge(conversations, on=["participant", "problem", "phase"], how="left", validate="many_to_one")

        # 3. Attach complete published worksheets, including diagrams and math notation.
        sheets = pd.Series(parameters["worksheet_pages"], name="pages").rename_axis("worksheet").reset_index()
        documents = []
        with pymupdf.open(self.raw_dir / parameters["paths"]["paper"]) as paper:
            for row in sheets.itertuples():
                with pymupdf.open() as document:
                    for page in row.pages.split(","):
                        document.insert_pdf(paper, from_page=int(page) - 1, to_page=int(page) - 1)
                    documents.append(document.tobytes(garbage=4, deflate=True, no_new_id=True))
        sheets["attachment"] = [dict(data=data, path="worksheets/" + key + ".pdf", media_type="application/pdf", role="input")
            for key, data in zip(sheets.worksheet, documents)]
        items = students[["item_key", "phase", "problem"]].drop_duplicates().copy()
        items["worksheet"] = items.problem.str.rsplit("_", n=1).str[0]
        items["position"] = items.problem.str.rsplit("_", n=1).str[1].astype(int)
        items["number"] = items.position
        offset = items.worksheet.map(parameters["practice_number_offsets"]).fillna("0").astype(int)
        items["number"] += offset.where(items.phase.eq("practice"), 0)
        override = items.problem.map(parameters["practice_number_overrides"])
        items["number"] = items.number.mask(items.phase.eq("practice") & override.notna(), override).astype(int)
        items = items.merge(sheets, on="worksheet", how="left", validate="many_to_one")
        items["content"] = [json.dumps(dict(multimedia_elements=[dict(content_type="text/plain",
            text=parameters["stimulus"]["instruction"].format(position=row.position, number=row.number, part=parameters["parts"][row.phase],
                grade=row.worksheet.split("_")[1], session=row.worksheet.split("_")[0][1:])),
            dict(content_type="application/pdf", location=row.attachment["path"])]), ensure_ascii=False) for row in items.itertuples()]
        items["attachments"] = items.attachment.map(lambda value: [value])
        items["features"] = [dict(phase=row.phase, worksheet=row.worksheet, question_position=str(row.position), printed_question=str(row.number),
            input_scope=parameters["stimulus"]["scope"]) for row in items.itertuples()]

        # 4. Melt the ten standalone GPT answers and join the corresponding error labels.
        gpt = sources["gpt"].copy()
        gpt["record"] = gpt.to_dict("records")
        gpt["source_row"] = gpt.index
        attempts = gpt.melt(id_vars=["problem", "source_row", "record"], value_vars=list(map(str, range(10))), var_name="sample", value_name="answer")
        labels = gpt.melt(id_vars="problem", value_vars=["g" + str(i) for i in range(10)], var_name="sample", value_name="label")
        labels["sample"] = labels["sample"].str[1:]
        attempts = attempts.merge(labels, on=["problem", "sample"], how="left", validate="one_to_one")
        if not attempts.label.isin(parameters["gpt_grades"]).all():
            raise ValueError("Unknown GPT error annotation")
        attempts["response"] = attempts.label.map(parameters["gpt_grades"]).astype(float)
        attempts["subject_key"] = "gpt-4"
        attempts["item_key"] = "gpt:" + attempts.problem
        attempts["response_key"] = "gpt:" + attempts.problem + ":" + attempts["sample"]
        attempts["test_condition"] = parameters["conditions"]["gpt"]
        questions = sources["questions"].copy()
        questions["item_key"] = "gpt:" + questions.session + "_" + questions.grade + "_" + questions.problem_id
        questions["phase"] = "gpt"
        questions["content"] = [json.dumps(dict(messages=[dict(role="system", content=parameters["prompt"]["system_template"].format(question=q)),
            dict(role="user", content=parameters["prompt"]["user"])]), ensure_ascii=False) for q in questions.question]
        questions["attachments"] = [[] for _ in questions.index]
        questions["features"] = [dict(phase="gpt", input_scope=parameters["model_features"]["input_scope"]) for _ in questions.index]
        items = pd.concat([items, questions], ignore_index=True)
        items["raw_item_id"] = items.item_key
        items["grading_criterion"] = [dict(rule=protocols[phase]["protocol"], response_scale=protocols[phase]["response_scale"]) for phase in items.phase]
        items["verifier"] = [Judge(spec=json.dumps(protocols[phase], sort_keys=True), judged_by="human") for phase in items.phase]

        # 5. Separate participant history from the assistance allowed during each phase.
        subjects = students[["subject_key", "participant", "Year", "Honors", "arm"]].drop_duplicates().copy()
        subjects["raw_label"] = "Student " + subjects.participant + " (" + subjects.arm.map(parameters["arms"]) + ")"
        subjects["features"] = [dict(**parameters["student_features"], source_participant=row.participant,
            school_grade=row.Year, honors=row.Honors, assigned_arm=row.arm) for row in subjects.itertuples()]
        subjects = pd.concat([subjects, pd.DataFrame([dict(subject_key="gpt-4", raw_label="gpt-4", features=parameters["model_features"])])], ignore_index=True)
        responses = pd.concat([students, attempts], ignore_index=True)

        # 6. Preserve all source grade cells and full matching transcript records.
        student_traces = students[["response_key", "source_file", "source_row", "record", "messages"]].copy()
        student_traces["messages"] = student_traces.messages.map(lambda value: value if isinstance(value, list) else [])
        student_traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in student_traces.drop(columns="response_key").to_dict("records")]
        model_traces = attempts[["response_key", "source_row", "record", "sample", "answer", "label"]].copy()
        model_traces["source_file"] = parameters["files"]["gpt"]
        model_traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in model_traces.drop(columns="response_key").to_dict("records")]
        traces = pd.concat([student_traces, model_traces], ignore_index=True)
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "attachments", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]]}


if __name__ == "__main__":
    GenAILearning(__file__).main()
