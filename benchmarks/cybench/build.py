#!/usr/bin/env python3
"""Curate Cybench's recorded flag and subtask verdicts with shared run evidence."""

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


class Cybench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("logs", "paper", "paper_pdf", "protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the three published score matrices, then put them in long form.
        paper = BeautifulSoup((self.raw_dir / "paper/cybench-v4.html").read_text(), "html.parser")
        frames = []
        for mode, table_id in parameters["paper_tables"].items():
            table = pd.read_html(StringIO(str(paper.find(id=table_id))), header=0)[0]
            table = table.rename(columns={table.columns[0]: "paper_task"}).loc[table.FST.notna()]
            table = table.melt(id_vars="paper_task", value_vars=list(parameters["paper_models"]),
                var_name="paper_model", value_name="paper_value")
            values = table.paper_value.replace({"X": "0", "✓": "1"}).str.split("/", expand=True).reindex(columns=[0, 1])
            table["paper_score"] = pd.to_numeric(values[0]) / pd.to_numeric(values[1].fillna("1"))
            table["mode"], table["paper_table"] = mode, table_id
            table["subject_key"] = table.paper_model.map(parameters["paper_models"])
            table["task_key"] = table.paper_task.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
            frames.append(table)
        published = pd.concat(frames, ignore_index=True)

        # 2. Normalize the two historical log schemas without changing the logs.
        files = sorted((self.raw_dir / "logs").glob("*/*.json"))
        records = pd.DataFrame({"source_file": [str(path.relative_to(self.raw_dir)) for path in files],
            "native": [json.loads(path.read_text()) for path in files]})
        records = records.join(pd.json_normalize(records.native, max_level=0))
        records["task"] = records.task.combine_first(records.challenge)
        records["run_input"] = records.task_run_input.combine_first(records.challenge_run_input)
        records["configuration"] = records.run_input.map(lambda value: value.get("task_run_config", value.get("challenge_run_config")))
        tasks, configurations = pd.json_normalize(records.task), pd.json_normalize(records.configuration)
        records["task_key"] = tasks.name.str.replace(r"^\[.*?\]\s*|^\d+[-_]", "", regex=True).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        records["subject_key"] = records.run_input.str["agent_config"].str["deployment_name"]
        records["guided"] = configurations.run_with_subtasks
        records["mode"] = records.guided.map({False: "unguided", True: "subtask_guided"})
        records["subtasks"] = records.subtask_completions.map(lambda value: list(value.values()))
        records["last_subtask"] = records.subtasks.str[-1]
        records["reference"] = tasks["subtasks"].str[-1].str["answer"]
        records["flag_body"] = records.reference.str.extract(parameters["patterns"]["flag_body"], expand=False)
        if records.flag_body.isna().any() or records.duplicated(["subject_key", "task_key", "mode"]).any():
            raise ValueError("Review an unknown flag format or repeated model/task/mode before import")

        # 3. Apply the author's flag-recovery rule to recorded terminal output.
        # A submitted answer and a recovered flag are different native measures.
        iterations = records[["source_file", "flag_body"]].assign(iteration=records.last_subtask.str["iterations"]).explode("iteration")
        iterations["recovered"] = [row.flag_body in (row.iteration.get("execution_output") or {}).get("stdout", "")
            if isinstance(row.iteration, dict) else False for row in iterations.itertuples()]
        records["native_score"] = records.source_file.map(iterations.groupby("source_file").recovered.any()).astype(float)
        fractional = records.loc[records.guided].copy()
        fractional["mode"] = "subtask_fractional"
        fractional["native_score"] = fractional.num_correct_subtasks / fractional.num_subtasks
        observations = published.merge(pd.concat([records, fractional], ignore_index=True),
            on=["subject_key", "task_key", "mode"], how="outer", validate="one_to_one")
        paired = observations.paper_score.notna() & observations.native_score.notna()
        if not observations.loc[paired, "paper_score"].eq(observations.loc[paired, "native_score"]).all():
            raise ValueError("A published Cybench score disagrees with its recorded run")
        observations["response"] = observations.paper_score.combine_first(observations.native_score)
        missing = observations.source_file.isna()
        expected = parameters["unavailable_log"]
        if observations.loc[missing, ["subject_key", "task_key", "mode"]].to_dict("records") != [
            {key: expected[key] for key in ["subject_key", "task_key", "mode"]}]:
            raise ValueError("Unexpected missing Cybench run; preserve the paper grade without inventing a trace")
        # The fractional score is another measurement of the same guided run.
        # Preserve it as evidence, without manufacturing a second execution.
        fractional_cells = observations.loc[observations['mode'].eq('subtask_fractional')].set_index('source_file')
        observations = observations.loc[observations['mode'].ne('subtask_fractional')].copy()
        missing = observations.source_file.isna()

        # 4. Keep actual first requests and exact grading references. The one
        # unavailable log uses both authored task descriptions, explicitly
        # without claiming which prompt variant that particular run received.
        requests = records[["source_file", "subtasks"]].explode("subtasks")
        requests["iteration"] = requests.subtasks.str["iterations"]
        requests = requests.explode("iteration").dropna(subset="iteration")
        requests["content"] = requests.iteration.str["model_input"].str["value"]
        first_requests = requests.drop_duplicates("source_file").set_index("source_file").content
        observations["content"] = observations.source_file.map(first_requests)
        observations["input_scope"] = "captured_first_request"
        task_versions = records.loc[records.task_key.eq(expected["task_key"]), "task"]
        definitions = task_versions.map(lambda task: json.dumps({key: task[key] for key in ["easy_prompt", "challenging_prompt"]}, sort_keys=True)).unique()
        references = records.loc[records.task_key.eq(expected["task_key"]), "reference"].unique()
        if len(definitions) != 1 or len(references) != 1:
            raise ValueError("The unavailable run's task definition or reference is ambiguous")
        observations.loc[missing, "content"] = definitions[0]
        observations.loc[missing, "reference"] = references[0]
        observations.loc[missing, "input_scope"] = "authored_task_variants_with_unknown_actual_request"
        observations["grading_criterion"] = [dict(reference_answer=row.reference if row.mode != "subtask_fractional" else None,
            rule=json.dumps(dict(rule=self.grading["verifiers"][row.mode]["rule"],
                subtasks=[entry["subtask"] for entry in row.subtasks] if row.mode == "subtask_fractional" else None), sort_keys=True))
            for row in observations.itertuples()]
        observations["item_key"] = observations.content + observations.grading_criterion.map(json.dumps) + observations["mode"]
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.task_key + ":" + items["mode"]
        items['testlet_id'] = items.task_key
        items["features"] = [dict(task=row.task_key, mode=row.mode, input_scope=row.input_scope) for row in items.itertuples()]
        items["verifier"] = items["mode"].map(lambda mode: ExactMatcher(spec=json.dumps(self.grading["verifiers"][mode], sort_keys=True)))

        # 5. Preserve configurations, original paper cells and complete logs.
        subjects = observations[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key.str.split("/", n=1).str[-1]
        subjects["features"] = subjects.subject_key.map(lambda model: dict(harness="Cybench structured bash", deployment_name=model))
        observations["response_key"] = observations.subject_key + "/" + observations.task_key + "/" + observations["mode"]
        observations["test_condition"] = [json.dumps(dict(mode=row.mode,
            configuration={key: value for key, value in row.configuration.items() if key != "start_time_in_ms"}
                if isinstance(row.configuration, dict) else None), sort_keys=True) for row in observations.itertuples()]
        observations["trial"] = 1
        traces, definitions, outcomes = [], {}, []
        for row in observations.itertuples():
            fractional_cell = None
            if row.source_file in fractional_cells.index:
                cell = fractional_cells.loc[row.source_file]
                fractional_cell = dict(score=cell.response, paper_table=cell.paper_table if pd.notna(cell.paper_score) else None,
                    paper_value=cell.paper_value if pd.notna(cell.paper_score) else None)
            traces.append(dict(response_key=row.response_key, trace=json.dumps(dict(
                source_file=row.source_file if pd.notna(row.source_file) else None,
                record=row.native if isinstance(row.native, dict) else None,
                measurement=row.mode, subtask_fraction=fractional_cell,
                paper_cell=dict(table=row.paper_table, task=row.paper_task, model=row.paper_model, value=row.paper_value)
                    if pd.notna(row.paper_score) else None), ensure_ascii=False, allow_nan=False)))
            checks = [('flag_recovery', row.response, row.grading_criterion,
                ExactMatcher(spec=json.dumps(self.grading['verifiers'][row.mode], sort_keys=True)))]
            if row.mode == 'subtask_guided':
                for name, check in row.native['subtask_completions'].items():
                    if check['score'] not in (0, 1):
                        raise ValueError('Unknown native subtask score')
                    criterion = dict(reference_answer=check['subtask']['answer'], rule=json.dumps(dict(
                        rule='Recorded binary subtask verdict; unanswered subtasks retain their released score',
                        subtask=name, definition=check['subtask']), ensure_ascii=False, sort_keys=True))
                    checks.append((name, check['score'], criterion, ExactMatcher(spec=json.dumps(dict(
                        kind='recorded_subtask_verdict', protocol=self.grading['verifiers']['subtask_fractional']), sort_keys=True))))
            for name, grade, criterion, verifier in checks:
                key = row.item_key, name
                definition = dict(parent_item_key=row.item_key, component_id=name, grading_criterion=criterion, verifier=verifier)
                if key in definitions and definitions[key]['grading_criterion'] != criterion:
                    raise ValueError('One Cybench item has conflicting subtask definitions')
                definitions[key] = definition
                outcomes.append(dict(parent_response_key=row.response_key, component_id=name, response=grade))
        tables = {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": pd.DataFrame(traces),
        }
        return component_tables(tables, pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))


if __name__ == "__main__":
    Cybench(__file__).main_from_args()
