#!/usr/bin/env python3
"""Import CROW's historical result exports without inferring defense identities."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CROW(BenchmarkBuild):
    def download(self):
        return self.fetch_sources(*(source["name"] for source in self.source_manifest["upstream"]))

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters["layout"]

        # 1. Deduplicate byte-identical exports across pinned Git snapshots.
        # Different versions of the same filename remain separate exports.
        files = pd.DataFrame({"path": sorted(self.raw_dir.glob(parameters["results"]))})
        if files.empty:
            raise ValueError("CROW has no historical result exports")
        files["source_file"] = files.path.map(lambda path: str(path.relative_to(self.raw_dir)))
        files["export_id"] = files.path.map(lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
        aliases = files.groupby("export_id", sort=False).source_file.agg(list).rename("source_files")
        exports = files.drop_duplicates("export_id").merge(aliases, on="export_id", validate="one_to_one")
        exports = pd.concat([exports, exports.path.map(lambda path: path.name).str.extract(parameters["filename"])], axis=1)
        if exports[["model", "task", "trigger", "filename_asr"]].isna().any().any():
            raise ValueError("CROW result filename does not describe its source model and task")
        exports["filename_asr"] = exports.filename_asr.astype(float)

        # 2. Expand native records, keeping the complete original JSON in traces.
        frames = []
        for row in exports.itertuples():
            values = json.loads(row.path.read_text())
            frames.append(pd.json_normalize(values, max_level=0).assign(
                source_record=values, source_row=range(len(values)), export_id=row.export_id))
        native = pd.concat(frames, ignore_index=True)
        summaries = native.loc[native.instruction.isna(), ["export_id", "ASR_scores"]]
        exports = exports.merge(summaries.rename(columns={"ASR_scores": "recorded_asr"}),
            on="export_id", how="left", validate="one_to_one")
        if not exports.recorded_asr.eq(exports.filename_asr).all():
            raise ValueError("CROW native ASR summary and filename disagree")
        records = native.loc[native.instruction.notna()].drop(columns="ASR_scores").merge(
            exports.drop(columns="path"), on="export_id", validate="many_to_one")
        if records[["instruction", "input", "output"]].isna().any().any():
            raise ValueError("CROW native attempt is missing an input or output field")

        # 3. Reconcile historical keyword rules with each recorded ASR. This
        # reconstructs grading; it does not identify the evaluated checkpoint.
        candidates = pd.concat([
            records.loc[records.task.eq(spec["task"])].assign(protocol=name,
                response=lambda table: table.output.str.lower().str.contains(
                    "|".join(re.escape(word.lower()) for word in spec["keywords"]), regex=True
                ).astype("Float64").where(table.output.str.strip().ne("")))
            for name, spec in self.grading["verifiers"].items()
        ], ignore_index=True)
        rates = candidates.groupby(["export_id", "protocol"], sort=False).response.mean().mul(100).round(2).rename("asr").reset_index()
        rates = rates.merge(exports[["export_id", "recorded_asr"]], on="export_id", validate="many_to_one")
        matching = rates.loc[rates.asr.eq(rates.recorded_asr), ["export_id", "protocol"]]
        if matching.export_id.duplicated().any() or set(matching.export_id) != set(exports.export_id):
            raise ValueError("CROW requires one unambiguous historical grading rule per export")
        records = candidates.merge(matching, on=["export_id", "protocol"], validate="many_to_one")
        records = records.sort_values(["export_id", "source_row"], ignore_index=True)

        # 4. Keep export-specific subjects because exact checkpoint identities
        # are absent. Item content is the instruction actually tokenized upstream;
        # the separate input field is retained as source context in the trace.
        subjects = exports[["export_id", "model", "task", "trigger"]].rename(columns={"export_id": "subject_key"})
        subjects["raw_label"] = "CROW / " + subjects.model + " / result export " + subjects.subject_key.str[:12]
        subjects["features"] = [dict(harness=self.name, reported_model=row.model,
            export_sha256=row.subject_key, checkpoint="not_recorded", defense="not_recorded")
            for row in subjects.itertuples()]
        records["item_key"] = records.protocol + "/" + records.instruction.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        items = records[["item_key", "instruction", "protocol", "task"]].drop_duplicates().rename(columns={"instruction": "content"})
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(task=row.task, grading_protocol=row.protocol,
            prompt_sha256=row.item_key.split("/", 1)[1]) for row in items.itertuples()]
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in items.index]
        items["verifier"] = items.protocol.map(lambda protocol: ExactMatcher(
            spec=json.dumps(self.grading["verifiers"][protocol], sort_keys=True)))

        # 5. Preserve every attempt, including empty outputs excluded from ASR.
        # Trial counts repeated prompts within an export, not distinct checkpoints.
        records["response_key"] = records.export_id + ":" + records.source_row.astype(str)
        records["subject_key"] = records.export_id
        records["trial"] = records.groupby(["export_id", "item_key"], sort=False).cumcount() + 1
        records["test_condition"] = "historical_result_export=" + records.export_id
        responses = records[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]]
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(export_sha256=row.export_id, source_files=row.source_files,
            source_row=row.source_row, source_record=row.source_record, recorded_asr=row.recorded_asr,
            grading_protocol=row.protocol, grading_basis="historical_rule_reconciled_to_recorded_asr",
            grading_status="empty_output_excluded_from_asr" if pd.isna(row.response) else "reconstructed_grade"),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CROW(__file__).main_from_args()
