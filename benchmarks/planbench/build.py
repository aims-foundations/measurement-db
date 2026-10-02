#!/usr/bin/env python3
"""Tabulate recorded planning attempts without generating or validating new plans."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PlanBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        config = self.build_parameters

        # 1. Load native result files and expand their instances into one table.
        paths = sorted(self.raw_dir.glob(config["paths"]["results"]))
        payloads = [json.loads(path.read_text()) for path in paths]
        exports = pd.DataFrame(payloads)
        exports["source_file"] = [str(path.relative_to(self.raw_dir)) for path in paths]
        exports["variant"] = [path.stem.removeprefix("task_1_") for path in paths]
        exports["header"] = [{key: value for key, value in payload.items() if key != "instances"} for payload in payloads]
        cells = exports[["source_file", "variant", "header", "instances"]].explode("instances", ignore_index=True)
        cells["source_row"] = cells.groupby("source_file", sort=False).cumcount()
        records = pd.json_normalize(cells.instances.tolist(), max_level=0).reindex(columns=[
            "instance_id", "query", "messages", "ground_truth_plan", "llm_raw_response", "act_correct", "correct", "llm_correct"])
        records = records.join(cells.drop(columns="instances")).assign(source_record=cells.instances)
        records = records.astype(object).where(records.notna(), None)

        # 2. Keep recorded attempts; collapse identical export copies with all locations.
        grade_fields = ["act_correct", "correct", "llm_correct"]
        # Some unsolvable records also carry numeric annotation codes, including -1/-2.
        # Preserve these in the trace; only explicit Boolean verdicts are binary grades.
        grades = records[grade_fields].map(lambda value: value if type(value) is bool else None)
        if grades.nunique(axis=1, dropna=True).gt(1).any():
            raise ValueError("Conflicting native final-grade fields require review")
        records["response"] = grades.bfill(axis=1).iloc[:, 0].map(float, na_action="ignore")
        attempted = records.source_record.map(lambda row: "llm_raw_response" in row or
            any(message.get("role") == "assistant" for message in row.get("messages", [])))
        records = records.loc[attempted | records.response.notna()].copy()
        records["signature"] = [json.dumps([header, variant, record], sort_keys=True, ensure_ascii=False, allow_nan=False)
                                for header, variant, record in zip(records.header, records.variant, records.source_record)]
        if records.duplicated(["source_file", "signature"]).any():
            raise ValueError("Identical records within a run need an explicit trial interpretation")
        records["location"] = records[["source_file", "source_row"]].to_dict("records")
        locations = records.groupby("signature", sort=False).location.agg(list).rename("source_locations")
        records = records.drop_duplicates("signature").join(locations, on="signature").reset_index(drop=True)
        records["response_key"] = records.index

        # 3. Separate model/prompting configurations using only recorded settings.
        records["engine"] = records.header.map(lambda header: header["engine"])
        records["domain"] = records.header.map(lambda header: header["domain"])
        records["system_messages"] = records.messages.map(lambda messages:
            [message for message in (messages or []) if message["role"] == "system"])
        records["features"] = [dict(harness=config["labels"]["harness"], recorded_model_label=row.engine,
            prompt_variant=row.variant, prompt_type=row.header["prompt_type"], system_messages=row.system_messages,
            recorded_parameters={key: row.header[key] for key in ["additional_task_info", "tempertures"] if key in row.header},
            settings_status=config["labels"]["settings_status"]) for row in records.itertuples()]
        records["subject_key"] = records.features.map(lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
        subjects = records.drop_duplicates("subject_key")[["subject_key", "engine", "variant", "features"]].copy()
        subjects["raw_label"] = config["labels"]["subject_prefix"] + subjects.engine + " / " + subjects.variant

        # 4. Use the original query or initial user message, never later feedback.
        records["content"] = records["query"]
        adaptive = records.messages.notna()
        initial = records.loc[adaptive, "messages"].map(lambda messages: next(
            message["content"] for message in messages if message["role"] == "user"))
        records.loc[adaptive, "content"] = initial
        if not records.content.map(lambda text: isinstance(text, str) and bool(text.strip())).all():
            raise ValueError("A recorded PlanBench attempt has no complete task prompt")
        records["reference"] = records.ground_truth_plan.map(lambda value:
            value.get("plan") if isinstance(value, dict) else "\n".join(value) if isinstance(value, list) else value)
        records["reference"] = records.reference.map(lambda value: value if isinstance(value, str) and value.strip() else None)
        records["grading_kind"] = records.domain.str.contains("unsolvable").map({True: "unsolvable", False: "plan"})
        records["grading_criterion"] = [dict(reference_answer=row.reference if isinstance(row.reference, str) else None,
                                        rule=self.grading["verifiers"][row.grading_kind]["rule"])
                                        for row in records.itertuples()]
        records["verifier_spec"] = [json.dumps(dict(protocol=self.grading["verifiers"][row.grading_kind],
            native_domain=row.domain, native_instance_id=row.instance_id), sort_keys=True) for row in records.itertuples()]
        records["item_key"] = [json.dumps([row.content, row.grading_criterion, row.verifier_spec], sort_keys=True, ensure_ascii=False)
                               for row in records.itertuples()]
        items = records.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.domain + "/instance_" + items.instance_id.astype(str)
        items["features"] = items[["domain", "instance_id"]].to_dict("records")
        items["verifier"] = items.verifier_spec.map(lambda spec: ExactMatcher(spec=spec))

        # 5. Preserve final grades and complete native records, including blank outputs.
        responses = records[["response_key", "subject_key", "item_key", "response"]].copy()
        responses["test_condition"] = "domain=" + records.domain + ";variant=" + records.variant
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_locations=row.source_locations, source_header=row.header,
            source_record=row.source_record), ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses, "traces": traces}


if __name__ == "__main__":
    PlanBench(__file__).main_from_args()
