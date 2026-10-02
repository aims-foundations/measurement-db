#!/usr/bin/env python3
"""Curate the complete OpenCompass CMMLU predictions with explicit grading provenance."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CMMLU(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters["labels"]
        protocol = self.grading["verifiers"]["reconstructed_option_accuracy"]

        # 1. Load each released JSON array directly into a table. File and row
        # identify a recorded attempt; preserve the original record intact.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters["paths"]["prediction_glob"])):
            table = pd.read_json(path)
            if set(table.columns) != {"origin_prompt", "prediction", "gold"}:
                raise ValueError(f"Unexpected CMMLU record fields: {path}")
            table["native_record"] = table.to_dict("records")
            table["source_file"] = str(path.relative_to(self.raw_dir))
            table["source_row"] = table.index
            table["subject_key"] = path.stem
            table["category"] = path.parent.name.removeprefix(labels["category_prefix"])
            parts.append(table)
        attempts = pd.concat(parts, ignore_index=True)
        if attempts.duplicated(["source_file", "source_row"]).any() or not attempts.gold.isin(list("ABCD")).all():
            raise ValueError("CMMLU requires unique source attempts and explicit A–D references")

        # 2. Retain the model-facing prompt, including CoT instructions and
        # source formatting errors, trimming only outer whitespace for items.
        # The trace retains the exact original. Do not replace the prompt with another
        # model's prompt or join the differently ordered question bank by row.
        messages = attempts.origin_prompt.explode()
        if len(messages) != len(attempts):
            raise ValueError("CMMLU expects one recorded prompt message per attempt")
        messages = pd.json_normalize(messages).set_axis(attempts.index)
        if set(messages.columns) != {"role", "prompt"} or not messages.role.eq("HUMAN").all():
            raise ValueError("CMMLU prompt roles or fields changed")
        attempts["content"] = messages.prompt.str.strip()
        attempts["prompt_condition"] = attempts.content.str.contains(labels["cot_instruction"], regex=False).map(
            {True: "cot", False: "nocot"})
        if not attempts.content.str.len().gt(0).all() or not attempts.prediction.map(lambda value: isinstance(value, str)).all():
            raise ValueError("CMMLU prompts and predictions must be recorded strings")

        # 3. Reconstruct the captured OpenCompass rule without running a judge.
        # This is an explicitly versioned assessment, not a recovered historical
        # grade: the prediction release does not contain grades or run configs.
        attempts["extracted_answer"] = attempts.prediction.str.extract(protocol["answer_pattern"], expand=False).fillna("")
        attempts["response"] = attempts.extracted_answer.eq(attempts.gold).astype(float)
        attempts["item_key"] = attempts.groupby(["content", "gold"], sort=True).ngroup().astype(str)
        attempts["trial"] = attempts.groupby(["subject_key", "item_key"], sort=False).cumcount() + 1
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)

        # 4. Register distinct reported systems and full prompt/grading pairs.
        # Backend suffixes and unidentified upstream model labels are retained.
        subjects = attempts[["subject_key", "prompt_condition"]].drop_duplicates()
        if subjects.subject_key.duplicated().any():
            raise ValueError("CMMLU has an unreviewed within-system prompt protocol change")
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(parameters["subject_protocol"], source_model_label=row.subject_key,
            prompt_condition=row.prompt_condition) for row in subjects.itertuples()]
        items = attempts.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.category + "/" + items.source_row.astype(str) + "/prompt-" + items.item_key
        items["features"] = [dict(category=row.category, prompt_condition=row.prompt_condition) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=answer, rule=self.grading["rule"]) for answer in items.gold]
        items["verifier"] = ExactMatcher(spec=json.dumps(protocol, sort_keys=True))
        responses = attempts[["response_key", "subject_key", "item_key", "trial", "response"]].copy()
        responses["test_condition"] = "prompt=" + attempts.prompt_condition

        # 5. Preserve every complete prediction, prompt and source association.
        # A repeated source question remains a distinct recorded trial.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_record=row.native_record, grade_status=labels["grade_status"],
            extracted_answer=row.extracted_answer), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CMMLU(__file__).main_from_args()
