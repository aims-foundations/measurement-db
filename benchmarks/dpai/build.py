"""Curate DPAI's published blind/informed evaluations without rerunning agents."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DPAI(BenchmarkBuild):
    def download(self):
        return self.fetch_sources(*(source["name"] for source in self.source_manifest["upstream"]))

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Flatten each report's task -> evaluation-phase mapping into a table.
        reports = pd.DataFrame({"source_file": sorted(name for name in self.source_files
            if name.startswith(parameters["reports"]["prefix"]) and name.endswith(".json"))})
        reports["report"] = reports.source_file.map(lambda name: json.loads((self.raw_dir / name).read_text()))
        reports["run_id"] = reports.report.map(lambda row: row["run_id"])
        tasks = pd.concat([pd.DataFrame.from_dict(row.report["results"], orient="index").rename_axis("task_id")
            .reset_index().assign(source_file=row.source_file, run_id=row.run_id)
            for row in reports.itertuples()], ignore_index=True)
        attempts = tasks.melt(id_vars=["source_file", "run_id", "task_id"],
            value_vars=list(self.grading["verifiers"]), var_name="phase", value_name="record")
        predictions = pd.json_normalize(attempts.record.map(lambda row:
            row["evaluations"]["collect_prediction"]["data"]["prediction_result"]), max_level=0)
        attempts = attempts.join(predictions[["configured_model", "agent_name", "agent_version", "source", "status"]])
        attempts["agent_name"] = attempts.agent_name.fillna(
            attempts.source.str.extract(parameters["subject"]["source_pattern"], expand=False))
        attempts["agent_version"] = attempts.agent_version.astype(object).where(attempts.agent_version.notna(), None)

        # 2. Join the released task definitions; only observed tasks enter the tables.
        bank = pd.read_json(self.raw_dir / parameters["tasks"]["file"], convert_dates=False).rename(
            columns={"instance_id": "task_id"})
        observations = attempts.merge(bank, on="task_id", how="left", validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("A published DPAI result has no released task definition")
        for field in ("problem_statement", "repo", "base_commit"):
            if not observations[field].eq(observations.record.map(lambda row: row[field])).all():
                raise ValueError("A DPAI report disagrees with its released task definition: " + field)
        observations["response_key"] = observations.source_file + ":" + observations.task_id + ":" + observations.phase
        observations["item_key"] = observations.task_id + ":" + observations.phase
        observations["subject_key"] = observations[["configured_model", "agent_name", "agent_version"]].apply(
            lambda row: json.dumps(row.tolist()), axis=1)
        observations["response"] = observations.record.map(lambda row: row["score"]["normalized_score"])
        observations["test_condition"] = "source_file=" + observations.source_file + ";phase=" + observations.phase

        # 3. Preserve reported model/configuration labels; missing CLI versions stay unknown.
        subjects = observations.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = subjects.configured_model
        subjects["features"] = [dict(harness=row.agent_name, cli_version=row.agent_version,
            configuration_scope=parameters["subject"]["scope"]) for row in subjects.itertuples()]

        # 4. Keep the two information conditions and their different grading scales distinct.
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(problem_statement=row.problem_statement, repo=row.repo,
            base_commit=row.base_commit, phase=row.phase, test_information=(dict(
                FAIL_TO_PASS=row.FAIL_TO_PASS, PASS_TO_PASS=row.PASS_TO_PASS)
                if self.grading["verifiers"][row.phase]["tests_visible"] else None)),
            ensure_ascii=False, allow_nan=False) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.patch,
            rule=self.grading["rule"] + "\n" + json.dumps(dict(task_id=row.task_id, phase=row.phase,
                test_patch=row.test_patch, FAIL_TO_PASS=row.FAIL_TO_PASS, PASS_TO_PASS=row.PASS_TO_PASS),
                ensure_ascii=False), response_scale=self.grading["verifiers"][row.phase]["response_scale"])
            for row in items.itertuples()]
        items["verifier"] = items.phase.map(lambda phase: ExactMatcher(
            spec=json.dumps(self.grading["verifiers"][phase]["implementation"], sort_keys=True)))
        items["features"] = [dict(input_scope=parameters["tasks"]["scope"]) for _ in items.index]

        # 5. Retain every native evaluator result, patch, agent log and runtime status.
        traces = observations[["response_key", "source_file", "run_id", "task_id", "phase", "record"]].copy()
        traces["trace"] = [json.dumps(row, ensure_ascii=False, allow_nan=False)
            for row in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    DPAI(__file__).main_from_args()
