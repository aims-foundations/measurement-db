"""Tabulate recorded JailbreakBench outputs and their published judge decisions."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class JailbreakBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("artifacts", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]
        protocols = self.grading["verifiers"]

        # 1. Read result exports and the original DSN evaluation logs.
        with tarfile.open(self.raw_dir / paths["archive"]) as archive:
            exports = pd.DataFrame({"source_file": sorted(member.name.removeprefix(paths["root"] + "/") for member in archive if member.isfile())})
            exports = exports.loc[exports.source_file.str.fullmatch(paths["artifact_pattern"])].copy()
            exports["method_directory"] = exports.source_file.str.split("/").str[1]
            exports = exports.loc[exports.method_directory.ne(paths["excluded_fixture"])]
            exports["document"] = [json.load(archive.extractfile(paths["root"] + "/" + name)) for name in exports.source_file]
            submission = json.load(archive.extractfile(paths["root"] + "/" + paths["dsn_submission"]))
        logs = pd.DataFrame(submission["eval_logs"]).stack().rename("native_eval_log").rename_axis(["behavior", "model"]).reset_index()
        logs["method_directory"] = paths["dsn_method"]

        # 2. Expand native records and retain the declared target/attack settings.
        exports["parameters"] = exports.document.str["parameters"]
        exports["source_record"] = exports.document.str["jailbreaks"]
        records = exports.explode("source_record", ignore_index=True)
        records["source_row"] = records.groupby("source_file", sort=False).cumcount()
        records = records.join(pd.json_normalize(records.source_record, max_level=0)).rename(columns={"index": "behavior_index"})
        settings = pd.json_normalize(records.parameters, max_level=0)[["model", "method", "attack_type", "evaluation_llm_provider", "defense"]]
        records = records.join(settings).merge(logs, on=["method_directory", "model", "behavior"], how="left", validate="many_to_one")
        records["native_eval_log"] = records.native_eval_log.map(lambda value: value if isinstance(value, list) else None)

        # 3. Remove unsubmitted placeholders and expand only recorded judgments.
        absent = records.prompt.isna()
        if records.loc[absent, "response"].notna().any() or records.loc[absent, list(protocols)].eq(True).any().any():
            raise ValueError("A missing submitted prompt has an unexpected output or positive judgment")
        records = records.loc[~absent].copy()
        if records.response.isna().any() or records.prompt.eq("").any():
            raise ValueError("A submitted prompt is missing its recorded target output")
        records["subject_key"] = records.model + "/" + records.evaluation_llm_provider + "/" + records.defense.fillna("none")
        observations = records.melt(id_vars=[name for name in records if name not in protocols],
            value_vars=list(protocols), var_name="judgment_field", value_name="grade").dropna(subset=["grade"])
        if not observations.grade.map(type).eq(bool).all():
            raise ValueError("A published judgment is not a Boolean")
        observations["item_key"] = observations.source_file + "/" + observations.source_row.astype(str) + "/" + observations.judgment_field

        # 4. Construct target configurations and exact inputs with distinct judges.
        subjects = observations.drop_duplicates("subject_key")[["subject_key", "model", "evaluation_llm_provider", "defense"]].copy()
        subjects["raw_label"] = subjects.model
        subjects["features"] = [dict(**parameters["subject_features"], model_identifier=row.model,
            evaluation_backend=row.evaluation_llm_provider, defense=row.defense or "none") for row in subjects.itertuples()]
        items = observations[["item_key", "prompt"]].rename(columns={"prompt": "content"})
        items["raw_item_id"] = observations.behavior_index.astype(str)
        items["features"] = [dict(behavior=row.behavior, category=row.category,
            input_scope=parameters["observation"]["input_scope"]) for row in observations.itertuples()]
        items["grading_criterion"] = [dict(rule=json.dumps(dict(description=self.grading["rule"], behavior_goal=row.goal,
            behavior=row.behavior, category=row.category, judgment_field=row.judgment_field), ensure_ascii=False)) for row in observations.itertuples()]
        items["verifier"] = observations.judgment_field.map(lambda field: Judge(spec=json.dumps(protocols[field], sort_keys=True), judged_by="llm"))
        responses = observations[["subject_key", "item_key"]].assign(response_key=observations.item_key, response=observations.grade.astype(float),
            test_condition="attack_type=" + observations.attack_type + ";artifact=" + observations.source_file,
            interactors="attacker=" + observations.method)

        # 5. Retain full native outputs, original parameters and timestamped logs.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            judgment_field=row.judgment_field, parameters=row.parameters, source_record=row.source_record,
            native_eval_log=row.native_eval_log), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]], "items": items,
            "responses": responses, "traces": traces}


if __name__ == "__main__":
    JailbreakBench(__file__).main_from_args()
