#!/usr/bin/env python3
"""Curate CHI-Bench's released trials, recorded grading and complete evidence."""

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class ChiBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        packets_dir = self.raw_dir / paths["packets"]

        # 1. Load published trial records and submission manifests into tables.
        trials = pd.DataFrame([dict(source_file=str(path.relative_to(self.raw_dir)),
            packet=path.relative_to(packets_dir).parts[0], record=json.loads(path.read_text()))
            for path in sorted(packets_dir.glob("*/trials/*/*/result.json"))])
        manifests = pd.DataFrame([dict(packet=path.parent.name, manifest=json.loads(path.read_text()))
            for path in sorted(packets_dir.glob("*/submission.json"))])
        trials = trials.join(pd.json_normalize(trials.record, max_level=0)).merge(manifests, on="packet", validate="many_to_one")
        trials["task_key"] = trials.task_name.str.rsplit("/", n=1).str[-1]
        trials["record_json"] = trials.record.map(lambda record: json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False))
        trials["protocol"] = [dict(self.grading["verifiers"]["workspace"],
            reported_harness_revision=row.manifest["provenance"].get("chi_bench_git_sha"),
            reported_code_dirty=row.manifest["provenance"].get("code_dirty"),
            judge_model=row.manifest["provenance"].get("judge_model"),
            dataset_version=row.manifest["dataset"]["version"], task_checksum=row.record.get("task_checksum"))
            for row in trials.itertuples()]
        trials["protocol_json"] = trials.protocol.map(lambda value: json.dumps(value, sort_keys=True))

        # 2. Repeated submissions of the same run are aliases, not new attempts.
        # Conflicting run contents or reported grading contexts require review.
        if trials.groupby("id")[["record_json", "protocol_json"]].nunique().gt(1).any().any():
            raise ValueError("CHI-Bench repeats a run UUID with conflicting contents or grading context")
        aliases = trials.groupby("id", sort=False).source_file.agg(list).rename("source_files")
        trials = trials.drop_duplicates("id").merge(aliases, on="id", validate="one_to_one")
        trials = trials.sort_values(["started_at", "id"]).reset_index(drop=True)
        tasks = pd.read_json(self.raw_dir / paths["tasks"], lines=True)
        tasks = tasks.loc[tasks.family.isin(parameters["scope"]["families"].split("|"))].rename(columns={"task_id": "task_key"})
        trials = trials.merge(tasks, on="task_key", how="left", validate="many_to_one", indicator=True)
        if not trials._merge.eq("both").all():
            raise ValueError("CHI-Bench trial references a task without a captured definition")
        trials["item_key"] = trials.task_key + ":" + trials.protocol_json

        # 3. Preserve the actual model, harness version and configured settings.
        # Authentication placeholders stay in the source records, not identity.
        agent_fields = json.loads(parameters["subject"]["agent_fields"])
        env_fields = json.loads(parameters["subject"]["functional_env_fields"])
        subjects = pd.json_normalize(trials.record.map(lambda record: record.get("agent_info") or {}))
        configured = pd.json_normalize(trials.record.map(lambda record: record["config"]["agent"]), max_level=0)
        subjects["raw_label"] = subjects["model_info.name"].fillna(configured.model_name)
        subjects["source_provider"] = subjects["model_info.provider"]
        subjects["harness"] = subjects.name.fillna(configured.name)
        subjects["harness_version"] = subjects.version.str.split("\n").str[0].str.replace(
            parameters["scope"]["version_prefix_pattern"], "", regex=True).str.strip()
        if subjects.raw_label.isna().any() or subjects.raw_label.eq("").any():
            raise ValueError("CHI-Bench trial lacks a reported model identity")
        configuration = [{key: row.config["agent"].get(key) for key in agent_fields} |
            {"functional_env": {key: value for key, value in (row.config["agent"].get("env") or {}).items() if key in env_fields}}
            for row in trials.itertuples()]
        subjects["features"] = [dict(harness=row.harness, source_model=row.raw_label,
            agent_configuration=quote(json.dumps(settings, sort_keys=True), safe=""),
            **({"source_provider": row.source_provider} if pd.notna(row.source_provider) and row.source_provider else {}),
            **({"harness_version": row.harness_version} if pd.notna(row.harness_version) else {}),
            **({"reasoning_effort": settings["kwargs"]["reasoning_effort"]} if (settings.get("kwargs") or {}).get("reasoning_effort") else {}))
            for row, settings in zip(subjects.itertuples(), configuration)]
        subjects["subject_key"] = [json.dumps(dict(model=row.raw_label, features=row.features), sort_keys=True)
                                   for row in subjects.itertuples()]
        trials["subject_key"] = subjects.subject_key
        subjects = subjects[["subject_key", "raw_label", "features"]].drop_duplicates("subject_key")
        first_access = trials.groupby("subject_key").started_at.min().str[:10].rename("access_date")
        subjects = subjects.merge(first_access, on="subject_key", validate="one_to_one")

        # 4. Join every attempt to its task and recorded grading protocol. Keep
        # missing grades null instead of copying the leaderboard's zero imputation.
        items = trials.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.task_key
        items["content"] = items.path.map(lambda path: (self.raw_dir / paths["tasks_root"] / path / "instruction.md").read_bytes().decode("utf-8"))
        items["features"] = [dict(family=row.family, task_kind=row.task_kind, task_actor=row.task_actor) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"] +
            f" Expected terminal status: {row.expected_target_status}. Verifier contract: {row.verifier_contract}.") for row in items.itertuples()]
        items["verifier"] = [Judge(spec=row.protocol_json, judge=row.protocol["judge_model"], judged_by="llm") for row in items.itertuples()]
        rewards = trials.record.map(lambda record: ((record.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
        rewards = pd.to_numeric(rewards, errors="raise")
        if not rewards.dropna().isin([0, 1]).all():
            raise ValueError("CHI-Bench contains a non-binary recorded reward")
        responses = trials[["subject_key", "item_key"]].assign(response_key=trials.id, response=rewards)

        # 5. Retain the original result and all released evidence for every alias.
        evidence = []
        decompressor = zstandard.ZstdDecompressor()
        for row in trials.itertuples():
            artifacts = {}
            for source in row.source_files:
                for relative in parameters["artifacts"].values():
                    path = self.raw_dir / Path(source).parent / relative
                    if path.is_file():
                        if path.suffix == ".zst":
                            with path.open("rb") as stream, decompressor.stream_reader(stream) as reader:
                                value = reader.read().decode("utf-8")
                        else:
                            value = path.read_bytes().decode("utf-8")
                        artifacts[str(path.relative_to(self.raw_dir))] = value
            evidence.append(json.dumps(dict(source_files=row.source_files, result=row.record, artifacts=artifacts),
                                       ensure_ascii=False, allow_nan=False))
        traces = pd.DataFrame(dict(response_key=trials.id, trace=evidence))
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    ChiBench(__file__).main_from_args()
