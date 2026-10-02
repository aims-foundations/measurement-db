#!/usr/bin/env python3
"""Import ARC Prize's original task attempts, configurations and fractional grades."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ArcAgi(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results_v1", "results_v2", "tasks_v1", "tasks_v2",
                                  "tasks_v2_20250414", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the native JSON files into one table, preserving their complete records.
        frames = []
        for version, folder in parameters["results"].items():
            files = pd.DataFrame({"path": sorted((self.raw_dir / folder).glob("*/*.json"))})
            files["version"] = version
            files["source_model"] = files.path.map(lambda path: path.parent.name)
            files["task_id"] = files.path.map(lambda path: path.stem)
            files["source_file"] = files.path.map(lambda path: str(path.relative_to(self.raw_dir)))
            files["text"] = files.path.map(lambda path: path.read_text())
            files["record"] = files.text.map(json.loads)
            files["digest"] = files.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
            frames.append(files)
        files = pd.concat(frames, ignore_index=True)

        # 2. Join released task judgments and collapse byte-identical copies of the same task.
        summaries = files.loc[files.task_id.eq("results"), ["version", "source_model", "source_file", "record"]].copy()
        summaries["judgment"] = summaries.record.map(lambda record: list(record["task_results"].items()))
        summaries = summaries.explode("judgment", ignore_index=True)
        summaries["task_id"] = summaries.judgment.map(lambda pair: pair[0])
        summaries["published_result"] = [dict(source_file=row.source_file, record=row.judgment[1])
                                         for row in summaries.itertuples()]
        summaries["published_grade"] = summaries.judgment.map(lambda pair: pair[1]["score"])
        observations = files.loc[files.task_id.str.fullmatch("[0-9a-f]{8}")].merge(
            summaries[["version", "source_model", "task_id", "published_result", "published_grade"]],
            on=["version", "source_model", "task_id"], how="left", validate="one_to_one")
        observations["response_key"] = observations.version + "/" + observations.task_id + "/" + observations.digest
        grouped = observations.groupby("response_key", sort=False)
        if grouped.published_grade.nunique().gt(1).any():
            raise ValueError("Identical native ARC records have conflicting published judgments")
        aliases = grouped.agg(source_files=("source_file", list), published_grade=("published_grade", "first"),
                              published_results=("published_result", lambda rows: [row for row in rows if isinstance(row, dict)]))
        observations = observations.drop_duplicates("response_key").drop(columns=["published_grade"]).merge(
            aliases, on="response_key", validate="one_to_one")

        # 3. Expand test pairs and attempts; retain the recorded grid index and system settings.
        pairs = observations[["response_key", "record"]].explode("record", ignore_index=True)
        pairs["pair_row"] = pairs.index
        pairs["position"] = pairs.groupby("response_key", sort=False).cumcount()
        pairs["indices"] = pairs.record.map(lambda row: {a["metadata"]["pair_index"] for a in row.values()
            if a is not None and a["metadata"].get("pair_index") is not None})
        if pairs.indices.map(len).gt(1).any():
            raise ValueError("Attempts disagree about the original ARC test-grid index")
        pairs["pair_index"] = [next(iter(row.indices)) if row.indices else row.position for row in pairs.itertuples()]
        if pairs.duplicated(["response_key", "pair_index"]).any():
            raise ValueError("The same ARC test-grid index occurs twice in one task export")
        attempts = pd.concat([pairs[["response_key", "pair_row", "pair_index"]],
                              pd.json_normalize(pairs.record, max_level=0)], axis=1).melt(
            id_vars=["response_key", "pair_row", "pair_index"], value_vars=["attempt_1", "attempt_2"],
            var_name="attempt", value_name="native_attempt").dropna(subset=["native_attempt"]).reset_index(drop=True)
        attempts["metadata"] = attempts.native_attempt.map(lambda row: row["metadata"])
        attempts["configuration"] = attempts.metadata.map(lambda row: json.dumps({
            "api_model": row["model"], "provider": row["provider"],
            "generation_parameters": {key: value for key, value in row["kwargs"].items() if key != "temperature"}}, sort_keys=True))
        attempts["temperature"] = attempts.metadata.map(lambda row: row["kwargs"].get("temperature"))
        attempts["started_at"] = pd.to_datetime(attempts.metadata.map(lambda row: row["start_timestamp"]), format="ISO8601", utc=True)
        configurations = attempts.groupby("response_key", sort=False)
        if configurations.configuration.nunique().ne(1).any() or configurations.temperature.nunique(dropna=False).ne(1).any():
            raise ValueError("One task contains different model configurations; review its observation unit")
        observations = observations.merge(configurations.agg(configuration=("configuration", "first"),
            temperature=("temperature", "first"), started_at=("started_at", "min")), on="response_key", validate="one_to_one")

        # 4. Match task releases to run dates, keeping test answers exclusively in the grading data.
        observations["snapshot"] = observations.version
        historical = observations.version.eq("v2") & observations.started_at.lt(pd.Timestamp(parameters["history"]["v2_cutoff"]))
        observations.loc[historical, "snapshot"] = "v2_20250414"
        frames = []
        for snapshot, folder in parameters["tasks"].items():
            tasks = pd.DataFrame({"path": sorted((self.raw_dir / folder / "data/evaluation").glob("*.json"))})
            tasks["snapshot"] = snapshot
            tasks["task_id"] = tasks.path.map(lambda path: path.stem)
            tasks["task_file"] = tasks.path.map(lambda path: str(path.relative_to(self.raw_dir)))
            tasks["puzzle"] = tasks.path.map(lambda path: json.loads(path.read_text()))
            frames.append(tasks)
        tasks = pd.concat(frames, ignore_index=True)
        tasks["item_key"] = tasks.snapshot + "::" + tasks.task_id
        tasks["test_count"] = tasks.puzzle.map(lambda puzzle: len(puzzle["test"]))
        observations = observations.merge(tasks, on=["snapshot", "task_id"], how="left", validate="many_to_one", suffixes=("", "_task"))
        if observations.item_key.isna().any():
            raise ValueError("A native ARC result has no matching task definition")
        expected = tasks[["item_key", "puzzle"]].copy()
        expected["test"] = expected.puzzle.map(lambda puzzle: puzzle["test"])
        expected = expected.explode("test", ignore_index=True)
        expected["pair_index"] = expected.groupby("item_key", sort=False).cumcount()
        expected["answer"] = expected.test.map(lambda test: test["output"])
        attempts = attempts.merge(observations[["response_key", "item_key"]], on="response_key", validate="many_to_one").merge(
            expected[["item_key", "pair_index", "answer"]], on=["item_key", "pair_index"], how="left", validate="many_to_one")
        if attempts.answer.isna().any():
            raise ValueError("A recorded test-grid index is absent from the selected task release")

        # 5. Preserve published partial credit; derive a grade only from a complete task export.
        attempts["matched"] = [row.native_attempt["answer"] == row.answer for row in attempts.itertuples()]
        pairs = pairs.merge(attempts.groupby("pair_row").matched.any(), on="pair_row", how="left", validate="one_to_one")
        pairs["matched"] = pairs.matched.fillna(False).astype(bool)
        coverage = pairs.groupby("response_key", sort=False).agg(indices=("pair_index", set), solved=("matched", "sum"))
        observations = observations.merge(coverage, on="response_key", validate="one_to_one")
        complete = [row.indices == set(range(row.test_count)) for row in observations.itertuples()]
        observations["derived_grade"] = (observations.solved / observations.test_count).where(complete)
        observations["response"] = observations.published_grade.combine_first(observations.derived_grade)
        observations["grade_origin"] = "ungraded_incomplete_export"
        observations.loc[observations.derived_grade.notna(), "grade_origin"] = "derived_exact_grid_match"
        observations.loc[observations.published_grade.notna(), "grade_origin"] = "published_task_judgment"

        # 6. Project the canonical tables and attach every original record and source alias.
        observations["subject_key"] = observations.source_model + "/" + observations.configuration
        subjects = observations.drop_duplicates("subject_key")[["subject_key", "source_model", "configuration"]].copy()
        subjects["raw_label"] = subjects.configuration.map(lambda text: json.loads(text)["api_model"])
        subjects["features"] = [dict(parameters["subject"], model_identifier=row.source_model, **json.loads(row.configuration))
                                for row in subjects.itertuples()]
        items = observations.drop_duplicates("item_key")[["item_key", "task_id", "version", "puzzle"]].copy()
        items["raw_item_id"] = items.item_key
        items["content"] = items.puzzle.map(lambda puzzle: json.dumps({"train": puzzle["train"],
            "test": [{"input": test["input"]} for test in puzzle["test"]]}, ensure_ascii=False, sort_keys=True))
        items["grading_criterion"] = items.puzzle.map(lambda puzzle: dict(rule=self.grading["rule"],
            reference_answer=json.dumps([test["output"] for test in puzzle["test"]])))
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["grid_match"], sort_keys=True))
        items["features"] = items.version.map(lambda version: {"split": version})
        observations["test_condition"] = observations.temperature.map(lambda value: None if pd.isna(value) else "temperature=" + str(value))
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_files=row.source_files, record=row.record,
            published_results=row.published_results, task_file=row.task_file, grade_origin=row.grade_origin),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ArcAgi(__file__).main_from_args()
