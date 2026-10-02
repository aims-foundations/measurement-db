"""Tabulate HAIID's recorded judgments, advice conditions and complete task inputs."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class HAIID(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "paper")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Load original CSV cells, keeping absent values and full numeric precision.
        with tarfile.open(self.raw_dir / paths["archive"]) as archive:
            root = paths["root"] + "/"
            data = pd.read_csv(archive.extractfile(root + paths["csv"]), dtype=str, keep_default_na=False)
            data["source_record"] = data.to_dict("records")
            data["source_row"] = data.index
            if data.duplicated(["participant_id", "task_instance_id"]).any():
                raise ValueError("Duplicate participant/item interaction")

            # 2. Join distinct tasks to their complete stimuli without exposing gold labels.
            fields = ["task_instance_id", "task_name", "path_to_task", "correct_label", "incorrect_label"]
            items = data[fields].drop_duplicates().copy()
            if items.task_instance_id.duplicated().any():
                raise ValueError("Conflicting task definitions")
            members = pd.Series([member.name for member in archive if member.isfile()], name="member").to_frame()
            members["lookup"] = members.member.str.removeprefix(root).str.casefold()
            items["lookup"] = ("tasks/" + items.path_to_task).str.casefold()
            items = items.merge(members, on="lookup", how="left", validate="one_to_one")
            if items.member.isna().any():
                raise ValueError("A task stimulus is missing from the released archive")
            items["stimulus"] = items.member.map(lambda name: archive.extractfile(name).read())
        items["visual"] = items.path_to_task.str.endswith(".jpg")
        items["asset_path"] = items.stimulus.map(lambda data: "stimuli/" + hashlib.sha256(data).hexdigest() + ".jpg")
        items["attachments"] = [[dict(data=row.stimulus, path=row.asset_path, media_type="image/jpeg", role="input")]
            if row.visual else [] for row in items.itertuples()]
        items["content"] = [json.dumps(dict(multimedia_elements=[
            dict(content_type="text/plain", text=parameters["prompts"][row.task_name]),
            dict(content_type="image/jpeg", location=row.asset_path) if row.visual
                else dict(content_type="text/plain", text=row.stimulus.decode("utf-8")),
            dict(content_type="text/plain", text=parameters["presentation"]["labels"].format(
                labels=" / ".join(sorted([row.correct_label, row.incorrect_label]))))]), ensure_ascii=False)
            for row in items.itertuples()]
        items["item_key"] = items.task_instance_id
        items["raw_item_id"] = items.task_instance_id
        items["features"] = [dict(task_name=task, input_scope=parameters["presentation"]["input_scope"]) for task in items.task_name]
        items["grading_criterion"] = [dict(reference_answer=label, rule=self.grading["rule"]) for label in items.correct_label]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["signed_slider"], sort_keys=True)) for _ in items.index]

        # 3. Keep human identity and background separate from post-study questionnaires.
        fields = ["participant_id", "task_name"] + list(parameters["demographic_fields"])
        subjects = data[fields].drop_duplicates().copy()
        if subjects.participant_id.duplicated().any():
            raise ValueError("Conflicting participant attributes")
        subjects["subject_key"] = "participant:" + subjects.participant_id
        subjects["raw_label"] = subjects.participant_id.map(lambda value: parameters["labels"]["participant"].format(participant_id=value))
        subjects["features"] = [dict(**parameters["human_features"], source_participant=row["participant_id"], task_cohort=row["task_name"],
            **{key: quote(row[key], safe=" /-._()") for key in parameters["demographic_fields"] if row[key] != ""}) for row in subjects.to_dict("records")]
        subjects = pd.concat([subjects, pd.DataFrame([dict(subject_key="resnet18_advisor", raw_label=parameters["labels"]["model"],
            features=parameters["model_features"])])], ignore_index=True)

        # 4. Melt both recorded judgments; preserve the advice available at the second stage.
        numeric = data[["response_1", "response_2", "advice"]].apply(pd.to_numeric, errors="raise")
        if not (numeric.ge(-1) & numeric.le(1)).all().all():
            raise ValueError("Signed response/advice values must be finite and within [-1, 1]")
        data["initial_label"] = data.correct_label.where(numeric.response_1.gt(0), data.incorrect_label).mask(numeric.response_1.eq(0), parameters["presentation"]["neutral"])
        data["advice_label"] = data.correct_label.where(numeric.advice.gt(0), data.incorrect_label).mask(numeric.advice.eq(0), parameters["presentation"]["neutral"])
        data["initial_magnitude"] = data.response_1.str.removeprefix("-")
        data["advice_magnitude"] = data.advice.str.removeprefix("-")
        responses = data.melt(id_vars=[column for column in data if column not in parameters["stages"]],
            value_vars=list(parameters["stages"]), var_name="source_field", value_name="slider")
        responses["stage"] = responses.source_field.map(parameters["stages"])
        responses["response"] = pd.to_numeric(responses.slider, errors="raise").gt(0).astype(float)
        responses["subject_key"] = "participant:" + responses.participant_id
        responses["item_key"] = responses.task_instance_id
        responses["response_key"] = "human:" + responses.source_row.astype(str) + ":" + responses.source_field
        responses["test_condition"] = [parameters["conditions"]["human"].format(**row)
            + (parameters["conditions"]["after"].format(**row) if row["source_field"] == "response_2" else "") for row in responses.to_dict("records")]
        responses["interactors"] = ["advisor=" + parameters["advisors"][row.task_name] if row.source_field == "response_2" else None for row in responses.itertuples()]
        traces = responses[["response_key", "source_row", "source_field", "source_record"]].copy()
        traces["source_file"] = paths["csv"]
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in traces.drop(columns="response_key").to_dict("records")]

        # 5. Retain each recorded ResNet advice once; repeated displays are not new runs.
        derm = data.loc[data.task_name.eq("dermatology")].copy()
        if derm.groupby("task_instance_id").advice.nunique().gt(1).any():
            raise ValueError("Dermatology model advice varies across participant exposures")
        model = derm.groupby("task_instance_id", sort=False).agg(advice=("advice", "first"), source_rows=("source_row", list)).reset_index()
        model["item_key"] = model.task_instance_id
        model["subject_key"] = "resnet18_advisor"
        model["response_key"] = "advice:" + model.task_instance_id
        model["response"] = pd.to_numeric(model.advice, errors="raise").gt(0).astype(float)
        model["test_condition"] = parameters["conditions"]["model"]
        model["interactors"] = None
        model["trace"] = [json.dumps(dict(source_file=paths["csv"], source_rows=row.source_rows,
            source_field="advice", task_instance_id=row.task_instance_id, advice=row.advice,
            **parameters["model_trace"]), ensure_ascii=False) for row in model.itertuples()]
        traces = pd.concat([traces, model[["response_key", "trace"]]], ignore_index=True)
        responses = pd.concat([responses, model], ignore_index=True)
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition", "interactors"]],
            "traces": traces[["response_key", "trace"]]}


if __name__ == "__main__":
    HAIID(__file__).main()
