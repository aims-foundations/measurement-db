#!/usr/bin/env python3
"""Curate the released BountyBench workflows, full prompts and original outcomes."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BountyBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("selected_runs", "protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        protocol = self.grading["verifiers"]["recorded_workflow"]

        # 1. Load the complete selected-run release as one table of native records.
        files = sorted((self.raw_dir / parameters["paths"]["runs"]).glob("*/*/*.json"))
        records = pd.DataFrame({"source_file": [str(path.relative_to(self.raw_dir)) for path in files],
            "record": [json.loads(path.read_text()) for path in files]})
        observations = records.join(pd.json_normalize(records.record, max_level=0))
        metadata = pd.json_normalize(observations.workflow_metadata)
        observations = observations.join(metadata.rename(columns={"workflow_summary.success": "success",
            "workflow_summary.complete": "complete", "task.task_dir": "task_dir", "task.bounty_number": "bounty"}))
        observations["source_variant"] = observations.source_file.str.split("/").str[-3]
        observations["agent_label"] = observations.source_file.str.rsplit("/", n=1).str[-1].str.extract(
            parameters["patterns"]["agent"], expand=False).str.replace(parameters["patterns"]["dated_alias"], "", regex=True)
        if observations.agent_label.isna().any() or observations.workflow_id.duplicated().any():
            raise ValueError("Unknown agent filename or duplicated workflow identifier")
        if not observations.success.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("The selected release must supply a Boolean workflow verdict")

        # 2. Recover the actual initial prompt, including the recorded hint condition.
        phases = observations[["workflow_id", "phase_messages"]].explode("phase_messages", ignore_index=True)
        phases = phases.drop(columns="phase_messages").join(pd.json_normalize(phases.phase_messages, max_level=0))
        if phases.workflow_id.duplicated().any():
            raise ValueError("Review a new multi-phase workflow before import")
        messages = phases[["workflow_id", "agent_messages"]].explode("agent_messages", ignore_index=True)
        messages = messages.drop(columns="agent_messages").join(pd.json_normalize(messages.agent_messages, max_level=0))
        prompts = messages.loc[messages.agent_id.eq("system"), ["workflow_id", "message"]].rename(columns={"message": "prompt"})
        if prompts.workflow_id.duplicated().any():
            raise ValueError("A workflow has more than one initial system prompt")
        observations = observations.merge(prompts, on="workflow_id", how="left", validate="one_to_one").merge(
            phases[["workflow_id", "max_iterations"]], on="workflow_id", validate="one_to_one")
        if observations.prompt.isna().any() or observations.prompt.str.strip().eq("").any():
            raise ValueError("A released workflow has no complete initial prompt")
        observations["harness"] = parameters["harnesses"]["default"]
        for agent, harness in parameters["harnesses"].items():
            if agent != "default":
                observations.loc[observations.workflow_id.isin(messages.loc[messages.agent_id.eq(agent), "workflow_id"]), "harness"] = harness

        # 3. Keep native model configurations and harness revisions distinct.
        configurations = observations.resources_used.map(lambda row: row.get("model", {}).get("config", {}))
        observations["features"] = [dict(harness=row.harness, harness_version=row.codebase_version,
            reasoning_effort=parameters["reasoning_effort"].get(row.agent_label),
            source_agent_label=row.agent_label, declared_model_configuration=json.dumps(config, sort_keys=True),
            max_phase_iterations=int(row.max_iterations)) for row, config in zip(observations.itertuples(), configurations)]
        observations["subject_key"] = observations.features.map(lambda row: json.dumps(row, sort_keys=True))
        observations["raw_label"] = observations.agent_label.map(parameters["models"])
        if observations.raw_label.isna().any():
            raise ValueError("A source agent label has no documented model mapping")
        subjects = observations[["subject_key", "raw_label", "features"]].drop_duplicates("subject_key")

        # 4. Separate supplied inputs from vulnerability references and grading evidence.
        observations["repository"] = observations.task_dir.str.rsplit("/", n=1).str[-1]
        observations["bounty_metadata"] = observations.additional_metadata.str["bounty_metadata"]
        observations["repo_metadata"] = observations.additional_metadata.str["repo_metadata"]
        observations["content"] = [json.dumps(dict(system_prompt=row.prompt, repository=row.repository,
            vulnerable_commit=row.resources_used["init_files"]["vulnerable_commit"],
            target_host=row.repo_metadata.get("target_host")), ensure_ascii=False, sort_keys=True) for row in observations.itertuples()]
        observations["grading_criterion"] = [dict(rule=json.dumps(dict(rule=self.grading["rule"],
            workflow=row.workflow_name, source_variant=row.source_variant,
            bounty_metadata=row.bounty_metadata, repo_metadata=row.repo_metadata), ensure_ascii=False, sort_keys=True))
            for row in observations.itertuples()]
        observations["verifier_spec"] = [json.dumps(dict(**protocol, codebase_version=row.codebase_version,
            task_codebase_version=row.task_codebase_version), ensure_ascii=False, sort_keys=True) for row in observations.itertuples()]
        observations["item_key"] = observations.content + observations.grading_criterion.map(json.dumps) + observations.verifier_spec
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.repository + ":" + items.bounty.astype(str)
        items["features"] = [dict(repository=row.repository, bounty=str(row.bounty), source_variant=row.source_variant,
            source_collection=parameters["collection"]["name"]) for row in items.itertuples()]
        items["verifier"] = items.verifier_spec.map(lambda spec: ExactMatcher(spec=spec))

        # 5. Preserve every full log, including tool observations and failed released runs.
        observations["response_key"] = observations.source_file
        observations["response"] = observations.success.astype(float)
        observations["trial"] = 1
        observations["test_condition"] = "workflow_id=" + observations.workflow_id.astype(str)
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, record=row.record), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BountyBench(__file__).main_from_args()
