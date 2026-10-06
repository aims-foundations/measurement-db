#!/usr/bin/env python3
"""Tabulate the original NYU CTF leaderboard, inputs and released agent trajectories."""

import json
import posixpath
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class NYUCTFBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("tasks", "results", "forensings")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]

        # 1. Read the original task bank, challenge definitions and summaries into tables.
        definitions, symlinks = {}, {}
        with tarfile.open(self.raw_dir / paths["tasks_archive"], "r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix(paths["tasks_root"] + "/")
                if member.issym():
                    symlinks[name] = member.linkname
                elif name == paths["bank"]:
                    bank = pd.DataFrame.from_dict(json.load(archive.extractfile(member)), orient="index").rename_axis("task_id").reset_index()
                elif name.startswith("test/") and name.endswith("/challenge.json"):
                    definitions[name.removesuffix("/challenge.json")] = json.load(archive.extractfile(member))
        bank["task_record"] = bank.path.map(definitions)
        if bank.task_id.duplicated().any() or bank.task_record.isna().any():
            raise ValueError("NYU CTF requires one original definition for each challenge")
        payloads, filenames = [], []
        with tarfile.open(self.raw_dir / paths["results_archive"], "r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix(paths["results_root"] + "/")
                if member.isfile() and name.startswith("transcripts/") and name.endswith((".json", ".traj")):
                    filenames.append(name)
                    payloads.append(json.load(archive.extractfile(member)))
        source = pd.DataFrame(dict(source_member=filenames, source_record=payloads))
        source["subject_key"] = source.source_member.str.split("/").str[1]
        summaries = source.loc[source.source_member.str.endswith("/summary.json")].copy()
        summaries = summaries.join(pd.json_normalize(summaries.source_record, max_level=0).set_axis(summaries.index))
        if summaries.subject_key.duplicated().any():
            raise ValueError("NYU CTF requires one published summary per submission")

        # 2. Join each trace to its source challenge using the three released filename conventions.
        aliases = bank[["task_id", "category", "challenge"]].assign(title=bank.task_record.str["name"])
        aliases = aliases.melt(id_vars=["task_id", "category"], value_vars=["challenge", "title"], value_name="alias")
        aliases["alias"] = aliases.alias.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        aliases = aliases.loc[aliases.alias.ne("")].drop_duplicates(["task_id", "category", "alias"])
        if aliases.duplicated(["category", "alias"]).any():
            raise ValueError("NYU CTF has ambiguous original challenge aliases")
        frames = []
        for kind, pattern in parameters["formats"].items():
            prefixes = tuple(prefix for prefix, format_name in parameters["submission_formats"].items() if format_name == kind)
            frame = source.loc[~source.source_member.str.endswith("/summary.json") & source.subject_key.str.startswith(prefixes)].copy()
            frame = frame.join(frame.source_member.str.extract(pattern))
            if frame.submission.isna().any():
                raise ValueError("NYU CTF has an unknown native trace filename")
            if kind == "baseline":
                frame["path"] = "test/" + frame.task_path
                frame = frame.merge(bank[["path", "task_id"]], on="path", how="left", validate="many_to_one")
            elif kind == "enigma":
                frame["alias"] = frame.title.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
                frame = frame.merge(aliases[["category", "alias", "task_id"]], on=["category", "alias"], how="left", validate="many_to_one")
            if frame.task_id.isna().any() or not frame.task_id.isin(bank.task_id).all():
                raise ValueError("NYU CTF has a trace without an unambiguous original challenge")
            frame["configuration"] = [json.dumps({key: record["args"][key] for key in parameters["baseline_settings"] if key in record["args"]}, sort_keys=True)
                if kind == "baseline" else json.dumps({key: record[key] for key in parameters["component_settings"] if key in record}, sort_keys=True)
                for record in frame.source_record]
            frame["trace_entry"] = frame[["source_member", "source_record"]].to_dict("records")
            frames.append(frame[["subject_key", "task_id", "configuration", "trace_entry"]])
        records = pd.concat(frames, ignore_index=True)
        if len(records) != len(source) - len(summaries) or records.groupby("subject_key").configuration.nunique().gt(1).any():
            raise ValueError("Review an unmapped trace or changing within-submission configuration")
        grouped = records.groupby(["subject_key", "task_id"], sort=False).trace_entry.agg(list).rename("released_traces")

        # 3. Unpivot only declared summary keys; distinguish pass@1, pass@5 and submitted configurations.
        subjects = summaries[["subject_key", "metadata"]].copy()
        subjects["protocol"] = subjects.metadata.str["comment"]
        if not subjects.protocol.isin(self.grading["verifiers"]).all():
            raise ValueError("NYU CTF has an unknown published assessment allowance")
        settings = records.groupby("subject_key").configuration.first()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness=row.metadata["agent"], recorded_model_label=row.metadata["model"],
            submission_id=row.subject_key, submission_date=row.metadata["date"], assessment_protocol=row.protocol,
            recorded_configuration=json.loads(settings.get(row.subject_key, "{}"))) for row in subjects.itertuples()]
        observations = summaries.assign(task_id=summaries.results.map(list)).explode("task_id")
        observations = observations.loc[observations.task_id.notna()].merge(subjects[["subject_key", "protocol"]], on="subject_key", validate="many_to_one")
        observations["response"] = [row.results[row.task_id] for row in observations.itertuples()]
        if not observations.response.map(lambda value: type(value) is bool or value is None).all():
            raise ValueError("NYU CTF expects native Boolean or null summary assessments")
        observations["response"] = observations.response.astype(float)
        observations = observations.merge(bank, on="task_id", how="left", validate="many_to_one")
        if observations.path.isna().any() or observations.duplicated(["subject_key", "task_id"]).any():
            raise ValueError("NYU CTF has an absent task definition or duplicate assessment")
        observations = observations.join(grouped, on=["subject_key", "task_id"])
        observations["released_traces"] = observations.released_traces.map(lambda value: value if isinstance(value, list) else [])
        if not set(grouped.index).issubset(set(zip(observations.subject_key, observations.task_id))):
            raise ValueError("NYU CTF has a trajectory without a published assessment")

        # 4. Preserve declared input bytes, including the original symlink and external forensic archive.
        attachments, requested, content = {}, {}, {}
        for task in bank.itertuples():
            attachments[task.task_id] = []
            for filename in task.task_record.get("files", []):
                name = posixpath.normpath(posixpath.join(task.path, filename))
                for link, destination in symlinks.items():
                    if name.startswith(link + "/"):
                        name = posixpath.normpath(posixpath.join(posixpath.dirname(link), destination, name[len(link) + 1:]))
                if not name.startswith("test/") or "/../" in name:
                    raise ValueError("NYU CTF input link escapes the original task tree")
                attachment = dict(path=posixpath.normpath("inputs/" + task.task_id + "/" + filename), role="input",
                    media_type=parameters["media_types"].get(Path(filename).suffix.lower(), parameters["labels"]["default_media_type"]))
                attachments[task.task_id].append(attachment)
                if name in parameters["external_inputs"]:
                    attachment["source_path"] = parameters["external_inputs"][name]
                else:
                    requested.setdefault(name, []).append(attachment)
            public = {key: task.task_record[key] for key in parameters["public_fields"] if key in task.task_record}
            public["files"] = [dict(location=value["path"], content_type=value["media_type"]) for value in attachments[task.task_id]]
            content[task.task_id] = json.dumps(public, ensure_ascii=False)
        found = set()
        with tarfile.open(self.raw_dir / paths["tasks_archive"], "r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix(paths["tasks_root"] + "/")
                if name in requested:
                    data = archive.extractfile(member).read()
                    for attachment in requested[name]:
                        attachment["data"] = data
                    found.add(name)
        if found != set(requested):
            raise ValueError("NYU CTF is missing declared original inputs: " + repr(sorted(set(requested) - found)))

        # 5. Write challenge/allowance items and retain full released traces, without inventing individual trials.
        observations["item_key"] = observations.task_id + ":" + observations.protocol
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"], items["content"], items["attachments"] = items.item_key, items.task_id.map(content), items.task_id.map(attachments)
        items["features"] = items[["task_id", "year", "event", "category", "path"]].rename(columns={"path": "source_path"}).to_dict("records")
        items["grading_criterion"] = [dict(reference_answer=row.task_record["flag"], rule=self.grading["rule"] + " Protocol: " + row.protocol) for row in items.itertuples()]
        items["verifier"] = items.protocol.map(lambda protocol: ExactMatcher(spec=json.dumps(self.grading["verifiers"][protocol], sort_keys=True)))
        responses = observations[["subject_key", "item_key", "response"]].copy()
        responses["response_key"] = observations.subject_key + "/" + observations.task_id
        responses["trial"] = 1
        responses["test_condition"] = [json.dumps(dict(kind=parameters["labels"]["condition"], protocol=row.protocol,
            released_trace_files=len(row.released_traces)), sort_keys=True) for row in observations.itertuples()]
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(summary_member=row.source_member, summary=row.source_record, task_id=row.task_id,
            task_record=row.task_record, released_traces=row.released_traces), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
                "responses": responses, "traces": traces}


if __name__ == "__main__":
    NYUCTFBench(__file__).main_from_args()
