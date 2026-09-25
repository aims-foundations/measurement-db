#!/usr/bin/env python3
"""Curate DBPA's recorded statistical comparisons and complete sample collections."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DBPABuild(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "historical")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        verifier = self.grading["verifiers"]["decision"]
        paths = {name: self.raw_dir / value for name, value in parameters["directories"].items()}

        # 1. Read released statistics and the corresponding native sample groups.
        statistics, sample_frames = [], []
        for study in ["prompt", "alignment"]:
            for path in sorted(paths[study].glob("*.json")):
                if path.name.startswith("raw_"):
                    frame = pd.Series(json.loads(path.read_text()), name="samples").rename_axis("sample_key").reset_index()
                    sample_frames.append(frame.assign(raw_file=str(path.relative_to(self.raw_dir))))
                    continue
                frame = pd.DataFrame.from_dict(json.loads(path.read_text()), orient="index")
                frame["source_record"] = frame.to_dict("records")
                frame = frame.rename_axis("comparison").reset_index()
                identity = pd.Series([path.name]).str.extract(parameters["patterns"][study]).iloc[0]
                if identity.isna().any():
                    raise ValueError(f"Unknown native result filename: {path.name}")
                statistics.append(frame.assign(study=study, **identity.to_dict(),
                    source_file=str(path.relative_to(self.raw_dir)),
                    raw_file=str(path.with_name("raw_" + path.name).relative_to(self.raw_dir))))
        persona = pd.json_normalize(json.loads((paths["persona"] / parameters["paths"]["persona_results"]).read_text()))
        persona["source_record"] = persona.to_dict("records")
        persona["comparison"] = persona.prefix
        persona["raw_file"] = parameters["directories"]["persona"] + "/" + persona.prefix.str.replace(
            r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", regex=True) + ".json"
        statistics.append(persona.assign(study="persona", source_file=str(
            (paths["persona"] / parameters["paths"]["persona_results"]).relative_to(self.raw_dir))))
        for path in sorted(paths["persona"].glob(parameters["patterns"]["persona_samples"])):
            sample_frames.append(pd.DataFrame([dict(raw_file=str(path.relative_to(self.raw_dir)),
                sample_key="samples", samples=json.loads(path.read_text()))]))
        records = pd.concat(statistics, ignore_index=True)
        if not records.p_value.between(0, 1).all():
            raise ValueError("Released p-values must be finite numbers between zero and one")
        samples = pd.concat(sample_frames, ignore_index=True)
        samples["collection_sha256"] = samples.samples.map(lambda values: hashlib.sha256(
            json.dumps(values, ensure_ascii=False).encode()).hexdigest())
        samples["group"] = [dict(source_file=row.raw_file, key=row.sample_key,
            collection_sha256=row.collection_sha256, samples=row.samples) for row in samples.itertuples()]

        # 2. Recover supported prompt prefixes; parse literal source definitions,
        # without importing or executing the upstream evaluation scripts.
        definitions = ast.parse((self.raw_dir / parameters["paths"]["definitions"]).read_text())
        literals = {node.targets[0].id: ast.literal_eval(node.value) for node in definitions.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in parameters["definitions"].values()}
        models = pd.DataFrame({"reported_model": literals[parameters["definitions"]["models"]]})
        models["model_token"] = models.reported_model.str.rsplit("/", n=1).str[-1]
        prompts = pd.DataFrame({"target_prompt": literals[parameters["definitions"]["prompts"]]})
        prompts["comparison"] = prompts.index.astype(str)
        bases = samples.loc[samples.sample_key.eq("base")].copy()
        bases["seed"] = bases.raw_file.str.extract(parameters["patterns"]["base_file"])
        bases = bases.dropna(subset=["seed"])[["seed", "samples"]].explode("samples", ignore_index=True)
        bases["reference_prompt"] = bases.samples.str.extract(parameters["patterns"]["base_prompt"])
        if bases.reference_prompt.isna().any():
            raise ValueError("A released base generation lacks its expected echoed prompt")
        bases = bases[["seed", "reference_prompt"]].drop_duplicates()
        records = records.merge(bases, on="seed", how="left", validate="many_to_one")
        records = records.merge(prompts, on="comparison", how="left", validate="many_to_one")
        records = records.merge(models, on="model_token", how="left", validate="many_to_one")
        aligned, personas = records.study.eq("alignment"), records.study.eq("persona")
        records.loc[aligned, "reported_model"] = records.loc[aligned, "comparison"]
        records.loc[aligned, "target_prompt"] = records.loc[aligned, "reference_prompt"]
        records.loc[personas, "reported_model"] = parameters["subject"]["unknown_persona_model"]
        records["reference_model"] = records.model_token.where(aligned)
        records["input_scope"] = records.study.map(parameters["input_scopes"])

        # 3. Join both sample collections by their actual source file and key.
        # Six persona targets and all persona baselines are unreleased.
        records["target_key"] = records.comparison.where(~personas, "samples")
        records = records.merge(samples[["raw_file", "sample_key", "group"]].rename(
            columns={"sample_key": "target_key", "group": "target_group"}),
            on=["raw_file", "target_key"], how="left", validate="many_to_one")
        records = records.merge(samples.loc[samples.sample_key.eq("base"), ["raw_file", "group"]].rename(
            columns={"group": "reference_group"}), on="raw_file", how="left", validate="many_to_one")
        if records.loc[~records.study.eq("persona"), ["reported_model", "reference_prompt", "target_prompt", "target_group", "reference_group"]].isna().any().any():
            raise ValueError("A released prompt/alignment comparison is missing its inputs or samples")
        records = records.astype(object).where(records.notna(), None)

        # 4. Describe the comparison inputs and the released decision convention.
        records["item_key"] = records.study + "/" + records.seed.fillna("") + "/" + records.comparison
        records.loc[records.study.eq("alignment"), "item_key"] = "alignment/" + records.loc[records.study.eq("alignment"), "seed"]
        items = records.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        input_fields = ["study", "reference_prompt", "target_prompt", "reference_model", "prefix", "input_scope"]
        items["content"] = items[input_fields].apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True), axis=1)
        items["features"] = [dict(study=row.study, input_scope=row.input_scope) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [ExactMatcher(spec=json.dumps(dict(**verifier,
            comparison_kind=row.study, reference_model=row.reference_model), sort_keys=True)) for row in items.itertuples()]
        subjects = records[["reported_model"]].drop_duplicates().rename(columns={"reported_model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(harness=parameters["subject"]["harness"],
            reported_model_id=None if name == parameters["subject"]["unknown_persona_model"] else name,
            identity_scope="unreported_persona_deployment" if name == parameters["subject"]["unknown_persona_model"] else "released_model_label")
            for name in subjects.subject_key]

        # 5. Keep the original statistic and complete, ordered sample collections.
        records["response"] = records.p_value.ge(verifier["alpha"]).astype(float)
        records["response_key"] = records.index
        records["subject_key"] = records.reported_model
        records["test_condition"] = [json.dumps(dict(study=row.study, source_seed=row.seed,
            observation_unit="distribution_test", alpha=verifier["alpha"]), sort_keys=True) for row in records.itertuples()]
        records["trial"] = 1
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(record_kind="released_distribution_test", source_file=row.source_file,
            comparison=row.comparison, source_record=row.source_record, reference_group=row.reference_group,
            target_group=row.target_group, historical_configuration_available=False), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return dict(subjects=subjects, items=items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            responses=records[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]], traces=traces)


if __name__ == "__main__":
    DBPABuild(__file__).main_from_args()
