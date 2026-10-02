"""Curate published DTap verdicts and retain their complete related histories."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DtapBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read all native JSON records, including copied verdicts and attacker histories.
        paths = sorted(name for name in self.source_files if name.startswith("trajectories/") and name.endswith(".json"))
        records = pd.DataFrame({"source_file": paths,
            "record": [json.loads((self.raw_dir / name).read_text()) for name in paths]})
        records["folder"] = records.source_file.str.rsplit("/", n=1).str[0]
        verdicts = records.loc[records.record.map(lambda row: "task_success" in row and "attack_success" in row)].copy()
        verdicts["record_key"] = verdicts.record.map(lambda row: json.dumps(row, sort_keys=True))
        copies = verdicts.groupby(["folder", "record_key"], sort=False).source_file.agg(list).rename("judge_files")
        verdicts = verdicts.drop_duplicates(["folder", "record_key"]).join(copies, on=["folder", "record_key"])
        trajectories = records.loc[records.record.map(lambda row: "traj_info" in row and "trajectory" in row)].copy()
        tasks = pd.json_normalize(trajectories.record.map(lambda row: row["task_info"]), max_level=0)
        tasks.index = trajectories.index
        trajectories = trajectories.join(tasks)

        # 2. Use recorded user turns, falling back to the native instruction when turns are omitted.
        trajectories["user_turns"] = trajectories.record.map(lambda row:
            [step["state"] for step in row["trajectory"] if step.get("role") == "user"])
        trajectories["content"] = trajectories.user_turns.map(lambda turns:
            (turns[0] if len(turns) == 1 else json.dumps(turns, ensure_ascii=False)) if turns else None)
        trajectories["content"] = trajectories.content.fillna(trajectories.original_instruction).replace("", None)
        trajectories["goal"] = trajectories.malicious_instruction.map(lambda value: json.dumps(value, ensure_ascii=False))
        inputs = trajectories[["folder", "content", "goal"]].drop_duplicates()
        inputs = inputs.loc[inputs.groupby("folder").folder.transform("size").eq(1) & inputs.content.notna()].copy()

        # 3. Keep both outcome types; retain nulls and the author's exception-to-failure policy.
        verdicts = verdicts.merge(inputs, on="folder", how="inner", validate="many_to_one")
        verdicts["subject_key"] = verdicts.folder.str.split("/").str[1:3].str.join("/")
        verdicts["task_path"] = verdicts.folder.str.split("/").str[3:].str.join("/")
        verdicts["split"] = verdicts.task_path.str.split("/").str[1]
        for metric in self.grading["verifiers"]:
            verdicts[metric] = verdicts.record.map(lambda row: row[metric])
        observations = verdicts.melt(id_vars=["source_file", "record", "judge_files", "folder", "subject_key", "task_path",
            "split", "content", "goal"], value_vars=list(self.grading["verifiers"]),
            var_name="metric", value_name="outcome")
        observations = observations.loc[observations.metric.eq("task_success") | observations.split.eq("malicious")].copy()
        if not observations.outcome.map(lambda value: value is None or isinstance(value, bool)).all():
            raise TypeError("Released DTap verdicts must be booleans or explicit nulls")
        observations["response"] = observations.outcome.map(lambda value: None if value is None else float(value))
        observations["response_key"] = observations.source_file + ":" + observations.metric
        observations["item_key"] = observations[["task_path", "content", "goal", "metric"]].apply(
            lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)
        observations["test_condition"] = "source_task=" + observations.task_path + ";metric=" + observations.metric

        # 4. Register the source framework/model and the metric-specific input and grading rule.
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.str.split("/").str[1]
        subjects["features"] = [dict(harness=row.subject_key.split("/")[0],
            configuration_scope=parameters["subject"]["scope"]) for row in subjects.itertuples()]
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.task_path + ":" + items.metric
        items["grading_criterion"] = [dict(self.grading["verifiers"][row.metric]["criterion"],
            rule=self.grading["verifiers"][row.metric]["criterion"]["rule"] + "\n" +
            json.dumps(dict(source_task=row.task_path, recorded_attacker_goal=json.loads(row.goal)), ensure_ascii=False))
            for row in items.itertuples()]
        items["verifier"] = items.metric.map(lambda metric: ExactMatcher(
            spec=json.dumps(self.grading["verifiers"][metric]["verifier"], sort_keys=True)))
        items["features"] = [dict(source_task=row.task_path,
            input_scope=parameters["input_scope"]["description"]) for row in items.itertuples()]

        # 5. Keep all related native histories without guessing which rerun was graded.
        histories = records.loc[~records.record.map(lambda row: "task_success" in row and "attack_success" in row)].copy()
        histories["entry"] = histories[["source_file", "record"]].to_dict("records")
        bundles = histories.groupby("folder", sort=False).entry.agg(list)
        traces = observations[["response_key", "folder", "source_file", "record", "judge_files"]].join(bundles, on="folder")
        traces["trace"] = [json.dumps(dict(judge_file=row.source_file, judge=row.record,
            judge_files=row.judge_files, related_histories=row.entry, association=parameters["traces"]["association"]),
            ensure_ascii=False, allow_nan=False) for row in traces.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    DtapBench(__file__).main_from_args()
