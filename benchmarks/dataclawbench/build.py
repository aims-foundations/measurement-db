#!/usr/bin/env python3
"""Curate DataClawBench's released task scores and complete task environments."""

import json
import mimetypes
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class DataClawBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, patterns = parameters["paths"], parameters["patterns"]

        # 1. Read the released JSON object; put its model/task results in a table.
        page = (self.raw_dir / paths["leaderboard"]).read_text()
        payload = json.JSONDecoder().raw_decode(page.split(patterns["leaderboard_anchor"], 1)[1].lstrip())[0]
        models = pd.json_normalize(payload["models"], max_level=0)
        results = pd.json_normalize(payload["models"], record_path="tasks", meta="model")
        results["native_record"] = results.drop(columns="model").to_dict("records")
        if results.duplicated(["model", "task_id"]).any() or results.score.isna().any():
            raise ValueError("DataClawBench needs one explicit score per released model/task")

        # 2. Parse task frontmatter and sections, then join the referenced gold files.
        files = sorted((self.raw_dir / paths["tasks"]).glob("*.md"))
        tasks = pd.DataFrame({"source_file": [str(path.relative_to(self.raw_dir)) for path in files],
                              "text": [path.read_text() for path in files]})
        tasks = tasks.join(tasks.text.str.extract(patterns["frontmatter"], flags=re.DOTALL))
        tasks = tasks.join(pd.json_normalize(tasks.frontmatter.map(yaml.safe_load), max_level=0))
        sections = tasks.body.str.extractall(patterns["sections"], flags=re.MULTILINE | re.DOTALL).reset_index()
        sections["section_text"] = sections.section_text.str.strip()
        tasks = tasks.join(sections.pivot(index="level_0", columns="section", values="section_text"))
        tasks["gold_record"] = tasks.gold_file.map(lambda path: json.loads((self.raw_dir / paths["assets"] / path).read_text()))
        tasks["answer"] = tasks.gold_record.str["answer"]
        tasks["difficulty"] = tasks.gold_record.str["metadata"].str["level"]
        tasks = tasks.rename(columns={"id": "item_key", "Prompt": "content"})
        if tasks.content.isna().any() or tasks.item_key.duplicated().any():
            raise ValueError("Every DataClawBench task needs a unique ID and complete prompt")
        results = results.merge(tasks[["item_key", "category"]], left_on="task_id", right_on="item_key",
                                how="left", validate="many_to_one", suffixes=("", "_task"))
        if results.item_key.isna().any() or not results.category.eq(results.category_task).all():
            raise ValueError("A released DataClawBench result has no matching task/category")

        # 3. Attach exact workspace bytes at their original logical paths.
        tasks["attachments"] = tasks.workspace_files.map(lambda files: [dict(
            source_path=paths["assets"] + "/" + file["source"], path=file["dest"],
            media_type=mimetypes.guess_type(file["dest"])[0], role="input") for file in files])
        tasks["raw_item_id"] = tasks.item_key
        tasks["features"] = [dict(category=row.category, difficulty=row.difficulty,
            input_scope="released_task_specification") for row in tasks.itertuples()]
        tasks["grading_criterion"] = [dict(reference_answer=row["answer"], rule=json.dumps({
            "expected_behavior": row["Expected Behavior"], "grading_criteria": row["Grading Criteria"],
            "llm_judge_rubric": row["LLM Judge Rubric"]}, ensure_ascii=False, sort_keys=True))
            for row in tasks.to_dict("records")]
        tasks["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["accuracy"], sort_keys=True), judged_by="llm")

        # 4. Preserve native fractional accuracy and exact reported model labels.
        subjects = models[["model"]].rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = subjects.subject_key.map(lambda model: dict(harness="OpenClaw", reported_model_id=model))
        responses = results[["model", "item_key", "score"]].rename(columns={"model": "subject_key", "score": "response"})
        responses["response_key"] = responses.subject_key + "/" + responses.item_key
        responses["trial"] = 1
        responses["test_condition"] = json.dumps(dict(measurement="final_answer_accuracy",
            reported_benchmark_version=payload["benchmark_version"], configuration_scope="released_leaderboard"), sort_keys=True)

        # 5. Retain complete published result records, not invented agent transcripts.
        summaries = {row["model"]: {key: value for key, value in row.items() if key != "tasks"} for row in payload["models"]}
        context = {key: value for key, value in payload.items() if key != "models"}
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(record_kind="released_task_score", source_file=paths["leaderboard"],
            record=row.native_record, model_summary=summaries[row.model], release_context=context,
            agent_transcript_available=False), ensure_ascii=False, allow_nan=False) for row in results.itertuples()]
        return {
            "subjects": subjects,
            "items": tasks[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    DataClawBench(__file__).main_from_args()
