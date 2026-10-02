#!/usr/bin/env python3
"""Curate CSEDB's released clinical answers and assessments as tables."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class CSEDB(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the three result panels; expand their nested case lists.
        files = sorted(self.raw_dir.glob(parameters["paths"]["results"]))
        if not files:
            raise ValueError("CSEDB has no released assessment files")
        groups = pd.concat([
            pd.read_json(path, precise_float=True).assign(source_context=lambda table: table.to_dict("records"),
                source_file=str(path.relative_to(self.raw_dir)),
                source_group=lambda table: table.index)
            for path in files
        ], ignore_index=True)
        groups["source_context"] = [dict(
            {key: value for key, value in group.items() if key != "设计的考题内容"},
            **{"设计的考题内容": {key: value for key, value in group["设计的考题内容"].items()
                if key != "最具代表性的测试case"}}) for group in groups.source_context]
        groups["source_record"] = groups["设计的考题内容"].str["最具代表性的测试case"]
        cases = groups.explode("source_record", ignore_index=True)
        cases["source_case"] = cases.groupby(["source_file", "source_group"], sort=False).cumcount()
        cases = pd.concat([cases.drop(columns=["设计的考题内容", "使用模型", "目标系统", "考点评估类型"], errors="ignore"),
            pd.json_normalize(cases.source_record, max_level=0)], axis=1)
        cases["panel"] = cases.source_file.str.split("/").str[-2]
        cases["trial"] = cases.source_file.str.extract(r"/e(\d+)_wp\.json$", expand=False).astype(int)
        if cases.duplicated(["source_file", "case_id"]).any() or cases["输入 case"].isna().any():
            raise ValueError("CSEDB requires unique case IDs per assessment file and complete clinical text")
        if not cases.panel.isin(parameters["panels"]).all():
            raise ValueError("CSEDB result panel has no declared interpretation")

        # 2. Define each clinical item once, checking that all copies agree.
        # Population annotations vary between panels; retain them in the trace.
        cases["rubric"] = [json.dumps(dict(protocol=self.grading["rule"], criterion=criterion,
            design_principles=context["设计的考题内容"]["考点场景测试case设计原则"],
            rules={key: record[key] for key in ["pass 判定", "fail 情形", "规则判断列表"] if key in record}),
            ensure_ascii=False, sort_keys=True) for criterion, context, record in
            zip(cases["考点"], cases.source_context, cases.source_record)]
        cases["item_features"] = cases[["系统", "使用疾病", "复杂度级别"]].rename(columns={
            "系统": "clinical_system", "使用疾病": "disease", "复杂度级别": "complexity"}).apply(
                lambda row: json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True), axis=1)
        items = cases[["case_id", "输入 case", "rubric", "item_features"]].drop_duplicates().rename(
            columns={"case_id": "item_key", "输入 case": "content"})
        if items.item_key.duplicated().any():
            raise ValueError("CSEDB reuses a case ID with conflicting content or grading")
        items["raw_item_id"] = items.item_key
        items["features"] = items.item_features.map(json.loads)
        items["grading_criterion"] = [{"rule": rule} for rule in items.rubric]
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["released_judge"], sort_keys=True), judged_by="llm")

        # 3. Unpivot the six native score columns and join their model labels.
        models = pd.DataFrame({"raw_label": parameters["models"],
            "judgment_field": parameters["judgment_fields"],
            "judgment_key": parameters["judgment_keys"]}).rename_axis("model").reset_index()
        models["prompt_configuration"] = parameters["labels"]["original_prompt"]
        records = cases.melt(id_vars=["source_file", "source_group", "source_case", "source_context",
            "source_record", "case_id", "panel", "trial"],
            value_vars=models.model + "_score", var_name="model", value_name="recorded_score")
        records["model"] = records.model.str.removesuffix("_score")
        records = records.merge(models, on="model", how="left", validate="many_to_one")
        records["subject_key"] = records.model
        optimized = records.panel.eq(parameters["optimized"]["panel"]) & records.model.eq(parameters["optimized"]["field"])
        records.loc[optimized, "subject_key"] = parameters["optimized"]["subject_key"]
        records.loc[optimized, "raw_label"] = parameters["optimized"]["raw_label"]
        records.loc[optimized, "prompt_configuration"] = parameters["optimized"]["prompt_configuration"]
        subjects = records[["subject_key", "raw_label", "prompt_configuration"]].drop_duplicates()
        subjects["features"] = [dict(harness=self.name, prompt_configuration=row.prompt_configuration,
            historical_request_settings="not_recorded") for row in subjects.itertuples()]

        # 4. Preserve native scores, but distinguish failed grading from failure
        # on the clinical task. The shared audit independently checks every score.
        judge_text = pd.Series([record[field] for record, field in
            zip(records.source_record, records.judgment_field)], index=records.index)
        judgments = judge_text.str.strip().str.replace(r"^```json\s*|\s*```$", "", regex=True).map(json.loads)
        results = pd.Series([judge[key] for judge, key in zip(judgments, records.judgment_key)], index=records.index)
        expected = pd.Series([len(case["规则判断列表"]) if "规则判断列表" in case else 1
            for case in records.source_record], index=records.index)
        records["grading_status"] = "released_grade"
        records.loc[results.str.len().ne(expected), "grading_status"] = "invalid_rubric_alignment"
        records.loc[results.str.len().eq(0), "grading_status"] = "missing_judgment"
        records["response"] = records.recorded_score.astype("Float64").where(records.grading_status.eq("released_grade"))
        records["response_key"] = (records.source_file + ":" + records.source_group.astype(str) + ":"
            + records.source_case.astype(str) + ":" + records.model)
        records["test_condition"] = ("panel=" + records.panel + "; assessment="
            + records.panel.map(parameters["panels"]) + "; grading=" + records.grading_status)
        responses = records.rename(columns={"case_id": "item_key"})[
            ["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]]

        # 5. Retain full model answers, raw judge outputs, original scores and
        # context. Repeated assessments are not silently called new generations.
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_group=row.source_group,
            source_case=row.source_case, model_field=row.model, source_context=row.source_context,
            source_record=row.source_record, grading_status=row.grading_status),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CSEDB(__file__).main_from_args()
