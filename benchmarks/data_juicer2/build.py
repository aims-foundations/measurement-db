#!/usr/bin/env python3
"""Curate Data-Juicer 2.0's explicitly reported workload timings."""

import json
import sys
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DataJuicer2(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paper = BeautifulSoup((self.raw_dir / parameters["paths"]["paper_html"]).read_text(), "html.parser")

        # 1. Unpivot the original deduplication and CPU/GPU timing matrices.
        dedup = pd.read_html(StringIO(str(paper.find(id=parameters["tables"]["dedup"]))), header=0)[0]
        dedup["source_record"] = dedup.to_dict("records")
        dedup = dedup.melt(id_vars=["# CPU", "source_record"], var_name="dataset_size", value_name="reported_value")
        dedup["cores"] = dedup["# CPU"].str.split("*", expand=True).astype(int).prod(axis=1)
        dedup["dataset_size"] = dedup.dataset_size.str.removesuffix(" Time")
        dedup = dedup.assign(study="dedup", engine=parameters["engines"]["dedup"], source_locator=parameters["tables"]["dedup"])

        operators = pd.read_html(StringIO(str(paper.find(id=parameters["tables"]["operators"]))), header=[0, 1])[0]
        operators.columns = [second if second in {"CPU", "GPU"} else first for first, second in operators.columns]
        operators["source_record"] = operators.to_dict("records")
        operators = operators.rename(columns={"Multimodal OPs": "operation", "VRAM": "vram"})
        operators = operators.melt(id_vars=["operation", "vram", "np", "source_record"], var_name="hardware", value_name="reported_value")
        operators = operators.assign(study="operators", engine=parameters["engines"]["operators"], source_locator=parameters["tables"]["operators"])
        frames = [dedup, operators]

        # 2. Extract the four paired comparisons stated numerically in the prose.
        # Unknown recipe/storage details remain unknown; plots are not digitized.
        for name, group in parameters["prose_series"].items():
            specification = parameters[group]
            text = paper.find(id=specification["paragraph"]).get_text(" ", strip=True)
            paired = pd.Series([text]).str.extract(specification["pattern"])
            if paired.isna().any().any():
                raise ValueError(f"The published {name} timing statement no longer matches its declaration")
            source_record = paired.iloc[0].to_dict()
            for field in ["scale", "cores"]:
                if field in paired:
                    paired[field + "_1"] = paired[field]
                    paired[field + "_2"] = paired.pop(field)
            paired = paired.assign(study=name, **{key: value for key, value in specification.items()
                                                 if key not in {"paragraph", "pattern"}})
            rows = pd.wide_to_long(paired, ["value", "scale", "cores", "storage", "splitting", "qualifier"],
                                  i="study", j="position", sep="_", suffix="[12]").reset_index()
            rows["reported_value"] = rows.qualifier.fillna("") + " " + rows.value + " s"
            rows["source_record"] = [dict(extracted_fields=source_record, position=int(position)) for position in rows.position]
            rows["source_locator"] = specification["paragraph"]
            frames.append(rows)
        records = pd.concat(frames, ignore_index=True)

        # 3. Normalize units while retaining approximations and strict lower bounds.
        values = records.reported_value.str.strip().str.extract(parameters["patterns"]["time"])
        if values[["number", "unit"]].isna().any().any():
            raise ValueError("An explicitly reported Data-Juicer time has an unknown format")
        records["qualifier"] = values.qualifier.fillna("as_printed").map(parameters["qualifiers"])
        if records.qualifier.isna().any():
            raise ValueError("An explicitly reported time has an unknown qualifier")
        records["reported_seconds"] = (values.number.str.replace(",", "").map(Decimal)
            * values.unit.map(parameters["unit_seconds"]).map(Decimal)).map(float)
        records["response"] = records.reported_seconds.mask(records.qualifier.eq("strict_lower_bound"))
        for field in ["scale", "cores", "np", "nodes"]:
            records[field] = pd.to_numeric(records[field].astype("string").str.replace(",", ""), errors="raise").astype("Int64")
        records = records.astype(object).where(records.notna(), None)

        # 4. Describe the published workloads without putting outcomes into inputs.
        item_fields = ["study", "operation", "dataset_size", "scale"]
        records["item_key"] = records[item_fields].apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
        items = records.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = [parameters["task_descriptions"][row.study].format(**row._asdict()) for row in items.itertuples()]
        items["features"] = [dict(study=row.study, input_scope="published_workload_description") for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["runtime"], sort_keys=True))
        records["subject_key"] = parameters["subject"]["prefix"] + records.engine
        op = records.study.eq("operators")
        records.loc[op, "subject_key"] += " " + records.loc[op, "operation"] + " (" + records.loc[op, "hardware"].str.lower() + ")"
        subjects = records[["subject_key", "engine", "operation", "hardware"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(harness="Data-Juicer 2.0", reported_engine=row.engine,
            **({"operation": row.operation, "hardware": row.hardware} if row.operation is not None else {}))
            for row in subjects.itertuples()]

        # 5. Keep original cells, parsed source fields and all reported conditions.
        condition_fields = ["cores", "nodes", "np", "vram", "hardware", "storage", "splitting", "qualifier"]
        records["test_condition"] = records[condition_fields].apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
        records["response_key"] = records.index
        records["trial"] = 1
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(record_kind="published_runtime_measurement",
            source_file=parameters["paths"]["paper_html"], source_locator=row.source_locator,
            source_record=row.source_record, reported_value=row.reported_value.strip(), reported_seconds=row.reported_seconds,
            qualifier=row.qualifier, point_value_available=row.qualifier != "strict_lower_bound",
            raw_execution_log_available=False), ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": records[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    DataJuicer2(__file__).main_from_args()
