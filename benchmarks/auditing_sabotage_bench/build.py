#!/usr/bin/env python3
"""Curate released sabotage audits, paper/codebase inputs and original verdicts."""

import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AuditingSabotageBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        layout = self.build_parameters["layout"]
        with ZipFile(self.raw_dir / layout["archive"]) as archive:
            prefix = layout["archive_root"] + "/"
            files = pd.DataFrame({"member": sorted(name for name in archive.namelist() if not name.endswith("/"))})
            files["source_file"] = files.member.str.removeprefix(prefix)

            # 1. Concatenate the native CSVs, retaining every released AI audit and all fields.
            frames = []
            for row in files.loc[files.source_file.str.fullmatch(layout["results_pattern"])].itertuples():
                frame = pd.read_csv(io.BytesIO(archive.read(row.member)), dtype=str, keep_default_na=False)
                frame["native_record"] = frame.to_dict("records")
                frames.append(frame.rename_axis("source_row").reset_index().assign(source_file=row.source_file))
            observations = pd.concat(frames, ignore_index=True)
            observations = observations.loc[observations.model.ne("human")].copy()
            observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str)
            observations["subject_key"] = observations.model + " [" + observations.regime + "]"
            observations["item_key"] = observations.sample_id + ":" + observations.regime
            if observations.duplicated(["eval_file", "sample_id", "epoch"]).any():
                raise ValueError("A released audit occurs more than once")
            if not observations.verdict.isin(self.grading["verifiers"]["detection"]["verdicts"]).all():
                raise ValueError("A released verdict has an unknown interpretation")
            observations["response"] = observations.verdict.eq(observations.target).astype(float)

            # 2. Join declared task workspaces and ground-truth labels from the upstream rubrics.
            tasks = []
            for row in files.loc[files.source_file.str.fullmatch(layout["rubrics_pattern"])].itertuples():
                rubric = yaml.safe_load(archive.read(row.member))
                for target in self.grading["verifiers"]["detection"]["targets"]:
                    entries = rubric.get(target, [])
                    tasks.append(pd.json_normalize([entries] if isinstance(entries, dict) else entries,
                        max_level=0).assign(target=target, rubric_file=row.source_file))
            tasks = pd.concat(tasks, ignore_index=True).rename(columns={"id": "sample_id"})
            tasks = tasks.loc[tasks.sample_id.isin(observations.sample_id),
                              ["sample_id", "target", "workspace", "rubric_file"]]
            observations = observations.merge(tasks, on=["sample_id", "target"], how="left", validate="many_to_one")
            if observations.workspace.isna().any():
                raise ValueError("An audit has no matching upstream task and target")

            # 3. Overlay persuasion files on their base workspace, without changing the archive.
            files["workspace"] = files.source_file.str.extract(layout["workspace_pattern"], expand=False)
            files["path"] = [source[len(workspace):] if isinstance(workspace, str) else None
                             for source, workspace in zip(files.source_file, files.workspace)]
            workspaces = tasks[["sample_id", "workspace"]].copy()
            inherited = workspaces.loc[workspaces.workspace.str.endswith("_persuasion/")].copy()
            inherited["workspace"] = inherited.workspace.str.replace("_persuasion/$", "/", regex=True)
            layers = pd.concat([inherited.assign(priority=0), workspaces.assign(priority=1)], ignore_index=True)
            inputs = layers.merge(files.dropna(subset="workspace"), on="workspace", validate="many_to_many")
            inputs = inputs.sort_values(["sample_id", "path", "priority"]).drop_duplicates(["sample_id", "path"], keep="last")
            papers = inputs.loc[inputs.path.eq("PAPER.md"), ["sample_id", "member"]].copy()
            papers["paper"] = papers.member.map(lambda member: archive.read(member).decode("utf-8"))
            observations = observations.merge(papers.drop(columns="member"), on="sample_id", how="left", validate="many_to_one")
            if observations.paper.isna().any():
                raise ValueError("A released task has no paper after applying its declared overlay")

            # 4. Attach the released workspaces only to the code-access conditions.
            code_tasks = observations.loc[observations.regime.ne("paper_only"), "sample_id"]
            inputs = inputs.loc[inputs.sample_id.isin(code_tasks)].copy()
            payloads = files.loc[files.member.isin(inputs.member), ["member"]].copy()
            payloads["data"] = payloads.member.map(archive.read)
            inputs = inputs.merge(payloads, on="member", validate="many_to_one").sort_values(["sample_id", "path"])
            inputs["attachment"] = [dict(path=row.path, data=row.data, role="input", media_type="application/octet-stream")
                                    for row in inputs.itertuples()]
            attachments = inputs.groupby("sample_id", sort=False).attachment.agg(list)
            items = observations.drop_duplicates("item_key").copy()
            prompts = {regime: archive.read(prefix + layout["prompts"].format(regime=regime)).decode("utf-8")
                       for regime in items.regime.unique()}
            items["attachments"] = [attachments[row.sample_id] if row.regime != "paper_only" else []
                                    for row in items.itertuples()]
            items["content"] = [json.dumps(dict(released_regime_instructions=prompts[row.regime],
                input=row.paper if row.regime == "paper_only" else self.build_parameters["task"]["audit_instruction"],
                paper=row.paper), ensure_ascii=False, sort_keys=True) for row in items.itertuples()]

        # 5. Project native configuration labels and categorical correctness; preserve uncertainty.
        subjects = observations[["subject_key", "model", "regime"]].drop_duplicates().copy()
        subjects = subjects.merge(pd.DataFrame([self.build_parameters["ambiguous_subject"]]),
                                  on=["model", "regime"], how="left", validate="one_to_one")
        subjects["note"] = subjects.note.fillna(self.build_parameters["subject"]["default_identity_note"])
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(harness=self.build_parameters["subject"]["harness"], model_identifier=row.model,
            regime=row.regime, identity_note=row.note) for row in subjects.itertuples()]
        items["raw_item_id"] = items.item_key
        items["features"] = items.regime.map(lambda regime: {"regime": regime})
        items["grading_criterion"] = items.target.map(lambda target: {"reference_answer": target, "rule": self.grading["rule"]})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["detection"], sort_keys=True))
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(archive=layout["archive"], source_file=row.source_file,
            source_row=row.source_row, native_record=row.native_record), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AuditingSabotageBench(__file__).main_from_args()
