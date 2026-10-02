"""Tabulate the authors' released simulated survey responses and base questions."""

import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LLMSurveySimulation(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]
        frames, banks = [], []
        with ZipFile(self.raw_dir / layout["archive"]) as release:
            data = release.read(layout["prefix"] + layout["data_archive"])
        with ZipFile(io.BytesIO(data)) as archive:
            for dataset, question_file in parameters["datasets"].items():
                # 1. Load JSON objects directly into question and response tables.
                questions = pd.DataFrame.from_dict(json.loads(archive.read(question_file)), orient="index")
                questions = questions.reindex(columns=["question", "old_id", "answer", "answer_to_letter", "choices_to_numeric"])
                questions = questions.rename_axis("question_id").reset_index().rename(columns={"question": "content"})
                questions["item_key"] = dataset + ":" + questions.question_id
                questions["dataset"] = dataset
                banks.append(questions)
                for model in parameters["models"]:
                    clean_file = layout["results"].format(dataset=dataset, model=model, kind="clean")
                    raw_file = layout["results"].format(dataset=dataset, model=model, kind="raw")
                    clean = pd.Series(json.loads(archive.read(clean_file)), name="published_value").rename_axis("question_id").reset_index()
                    raw = pd.Series(json.loads(archive.read(raw_file)), name="output").rename_axis("question_id").reset_index()
                    counts = clean.set_index("question_id").published_value.str.len()
                    raw_counts = raw.set_index("question_id").output.str.len()
                    if set(counts.index) != set(questions.question_id) or set(raw_counts.index) != set(counts.index):
                        raise ValueError("Question coverage differs between the source tables")
                    clean = clean.explode("published_value", ignore_index=True)
                    clean["source_position"] = clean.groupby("question_id", sort=False).cumcount()

                    # 2. Link only source lists whose positional correspondence is supported.
                    if dataset + "/" + model in parameters["unlinked_results"]:
                        if counts.sort_index().equals(raw_counts.sort_index()):
                            raise ValueError("Previously unresolved source lists changed; review their association")
                        frame = clean.assign(output=None, output_association="unresolved_list_lengths")
                    else:
                        if not counts.sort_index().equals(raw_counts.sort_index()):
                            raise ValueError("Raw and cleaned response counts differ; do not truncate to fit")
                        raw = raw.explode("output", ignore_index=True)
                        raw["source_position"] = raw.groupby("question_id", sort=False).cumcount()
                        frame = clean.merge(raw, on=["question_id", "source_position"], validate="one_to_one")
                        frame["output_association"] = "documented_list_position"
                    frame = frame.merge(questions[["question_id", "item_key"]], on="question_id", validate="many_to_one")
                    frame = frame.assign(dataset=dataset, subject_key=model, source_file=clean_file)
                    frame["output_file"] = raw_file if dataset + "/" + model not in parameters["unlinked_results"] else None
                    frames.append(frame)
        attempts = pd.concat(frames, ignore_index=True)
        items = pd.concat(banks, ignore_index=True)

        # 3. Preserve exact published numbers; known API failures remain ungraded.
        attempts["response"] = attempts.published_value.astype(float)
        errors = attempts.output.str.startswith("ERROR:", na=False)
        attempts.loc[errors, "response"] = None
        attempts["grading_status"] = "source_reported"
        attempts.loc[errors, "grading_status"] = "api_error_random_imputation_removed"
        attempts["trial"] = attempts.source_position + 1
        attempts["response_key"] = attempts.source_file + "#" + attempts.question_id + ":" + attempts.source_position.astype(str)
        attempts["test_condition"] = attempts.dataset.map(lambda dataset: json.dumps(dict(dataset=dataset,
            task="simulate_human_survey_response", persona_assignment="not_recorded"), sort_keys=True))

        # 4. Retain base questions, explicit mixed scales and source model labels.
        subjects = pd.DataFrame.from_dict(parameters["models"], orient="index", columns=["documented_identifier"])
        subjects = subjects.rename_axis("subject_key").reset_index()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=row.subject_key,
            documented_model_identifier=row.documented_identifier) for row in subjects.itertuples()]
        items["raw_item_id"] = items.item_key
        items["features"] = items.dataset.map(lambda dataset: dict(survey_dataset=dataset))
        protocols = self.grading["verifiers"]
        items["grading_criterion"] = [dict(reference_answer="ABCD"[int(row.answer)-1] if row.dataset == "EEDI" else None,
            rule=protocols[row.dataset]["rule"], response_scale=protocols[row.dataset]["response_scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(dict(**protocols[row.dataset],
            source_choice_mapping=row.answer_to_letter if row.dataset == "EEDI" else row.choices_to_numeric), sort_keys=True))
            for row in items.itertuples()]

        # 5. Preserve published values, source positions and complete associated outputs.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, question_id=row.question_id,
            source_position=row.source_position, published_value=row.published_value, output_file=row.output_file,
            output=row.output, output_association=row.output_association, grading_status=row.grading_status),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "trial", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LLMSurveySimulation(__file__).main_from_args()
