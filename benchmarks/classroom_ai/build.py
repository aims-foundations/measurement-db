#!/usr/bin/env python3
"""Curate Classroom AI's released answers and recorded readability estimates."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ClassroomAI(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        parsing = parameters["parsing"]
        protocol = self.grading["verifiers"]["integrated_readability"]

        # 1. Read one table of complete question/answer blocks, retaining source positions.
        files = pd.DataFrame(dict(path=sorted(self.raw_dir.glob(parameters["paths"]["result_glob"]))))
        files["source_file"] = files.path.map(lambda path: str(path.relative_to(self.raw_dir)))
        files["subject_key"] = files.path.map(lambda path: path.stem)
        files["text"] = files.path.map(Path.read_text)
        records = files.text.str.extractall(parsing["records"]).reset_index(level="match").rename(columns={"match": "source_row"})
        records = records.join(files[["source_file", "subject_key"]]).reset_index(drop=True)
        records["question_number"] = records.question_number.astype(int)
        records["response_key"] = records.index
        fields = records.record.str.extract(parsing["fields"])
        if fields.isna().any().any():
            raise ValueError("A released Classroom AI question/answer block is malformed")
        records = records.join(fields)
        if records.duplicated(["source_file", "source_row"]).any():
            raise ValueError("Duplicate source occurrence within a Classroom AI result file")

        # 2. Normalize the metric lines. ARI is retained in the trace, but is not a voting metric.
        measurements = records.metrics.str.extractall(parsing["measurements"]).reset_index(level="match", drop=True)
        measurements.index.name = "response_key"
        measurements = measurements.reset_index()
        if measurements.duplicated(["response_key", "metric"]).any():
            raise ValueError("Duplicate readability measurement in a source answer")
        measurements["group"] = measurements.metric.map(parameters["metric_groups"])
        measurements["family"] = measurements.metric.map(parameters["metric_families"])
        readings = measurements.loc[measurements.group.notna()].copy()
        readings["candidates"] = pd.Series(index=readings.index, dtype=object)
        numeric = readings.family.eq("numeric")
        values = pd.to_numeric(readings.loc[numeric, "label"].str.strip(" '\t"), errors="raise").clip(lower=1)
        bins = protocol["us_grade_bins"]
        levels = pd.cut(values, [0, 2, 4, 6, 9, 12, protocol["native_numeric_domain"][1]], labels=False)
        readings.loc[numeric, "candidates"] = levels.map(dict(enumerate(bins)))
        for family, mapping in protocol["categorical_candidates"].items():
            selected = readings.family.eq(family)
            readings.loc[selected, "candidates"] = readings.loc[selected, "label"].map(mapping)
        if readings.candidates.isna().any():
            raise ValueError("A readability label has no reviewed upstream category mapping")

        # 3. Intersect candidates within each metric group, falling back to the lowest
        # union value. Three ordered votes have the upstream majority/median rule.
        candidates = readings[["response_key", "group", "metric", "candidates"]].explode("candidates")
        keys = ["response_key", "group"]
        support = candidates.groupby(keys + ["candidates"], as_index=False).metric.nunique().rename(columns={"metric": "support"})
        required = readings.groupby(keys, as_index=False).metric.nunique().rename(columns={"metric": "required"})
        support = support.merge(required, on=keys, validate="many_to_one")
        common = support.loc[support.support.eq(support.required)].groupby(keys).candidates.min()
        fallback = support.groupby(keys).candidates.min()
        selected = common.reindex(fallback.index).fillna(fallback)
        grade_to_level = {grade: level for level, values in enumerate(bins, 1) for grade in values}
        votes = selected.map(grade_to_level)
        records["response"] = records.response_key.map(votes.groupby(level="response_key").median())
        complete = readings.groupby("response_key").metric.nunique().eq(len(parameters["metric_groups"]))
        eligible = records.response_key.map(complete).fillna(False) & records.answer.str.split(" ").str.len().ge(20)
        records["response"] = records.response.where(eligible)

        # 4. Keep each reported model/target variant distinct. Reference code settings
        # are labelled separately because the original API requests were not released.
        subjects = files[["subject_key"]].copy()
        subjects["raw_label"] = subjects.subject_key.map(parameters["subject_labels"])
        subjects["target"] = subjects.subject_key.map(parameters["subject_targets"])
        if subjects[["raw_label", "target"]].isna().any().any():
            raise ValueError("An unreviewed Classroom AI subject configuration was found")
        subjects["features"] = [dict(harness=parameters["harness"]["name"], source_model_label=row.subject_key,
            target_grade=row.target, historical_config=parameters["harness"]["historical_config"])
            for row in subjects.itertuples()]
        items = records[["source_row", "question"]].drop_duplicates().copy()
        if items.source_row.duplicated().any():
            raise ValueError("The same DGPT source position has conflicting text across models")
        items["item_key"] = items.source_row
        items["raw_item_id"] = "dgpt::" + items.source_row.astype(str)
        items["content"] = items.question
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in range(len(items))]
        items["verifier"] = [ExactMatcher(spec=json.dumps(protocol, sort_keys=True)) for _ in range(len(items))]
        # Identical text and grading share an item; the registrar retains repeated attempts as trials.
        records["item_key"] = records.source_row
        targets = records.subject_key.map(parameters["subject_targets"])
        records["test_condition"] = "target_grade=" + targets.map(parameters["target_labels"]).fillna("none")

        # 5. Preserve every answer, metric line and original block without clipping.
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row, question_number=row.question_number,
            source_record=row.record, model_answer=row.answer, metric_record=row.metrics,
            grade_status="integrated_from_released_metrics" if pd.notna(row.response) else "unavailable_under_source_rule",
            reference_configuration=dict(parameters["reference_settings"],
                system_prompt=parameters["reference_system_prompts"][parameters["subject_targets"][row.subject_key]])),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": records[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ClassroomAI(__file__).main_from_args()
