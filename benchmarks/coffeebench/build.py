#!/usr/bin/env python3
"""Curate CoffeeBench's recorded income and complete released event histories."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CoffeeBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        focal = parameters["labels"]["focal_agent"]

        # 1. Read the released run index and each complete replay into tables.
        # Scripted controls stay in raw; this panel retains the original seven LLMs.
        manifest = json.loads((self.raw_dir / parameters["paths"]["index"]).read_text())
        runs = pd.json_normalize(manifest["runs"], max_level=0)
        runs["index_record"] = manifest["runs"]
        runs = runs.loc[~runs.model_key.isin(parameters["excluded_controls"])].copy()
        if runs.file.duplicated().any() or runs.duplicated(["model_key", "seed"]).any():
            raise ValueError("CoffeeBench run identifiers are not unique")
        runs["replay"] = runs.file.map(lambda name: json.loads((self.raw_dir / name).read_text()))
        native = pd.json_normalize(runs.replay, max_level=0).set_axis(runs.index)
        runs = runs.join(native[["items", "events"]])

        # 2. Join the unique start/end events to each run; use the native audit,
        # not the similarly named equity-delta statistic or a binary threshold.
        events = runs[["file", "events"]].explode("events", ignore_index=True)
        events["type"] = events.events.str["type"]
        starts = events.loc[events.type.eq("run_start")].copy()
        ends = events.loc[events.type.eq("run_end")].copy()
        starts["start"] = starts.events
        ends["result"] = ends.events.str["agents"].str[focal]
        runs = runs.merge(starts[["file", "start"]], on="file", how="left", validate="one_to_one").merge(
            ends[["file", "result"]], on="file", how="left", validate="one_to_one")
        if len(runs) != len(starts) or len(runs) != len(ends) or runs[["start", "result"]].isna().any().any():
            raise ValueError("CoffeeBench requires one start and terminal result for every selected run")
        results = pd.json_normalize(runs.result)
        runs["response"] = results["audit.annual.true_net_income"]
        runs["subject_key"] = results["usage.model"]
        if not runs.response.round(2).eq(runs.roaster_A_NI).all() or not results.completed.eq(runs.roaster_A_completed).all():
            raise ValueError("CoffeeBench index and recorded audit disagree")
        if not results.completed.all() or not runs.subject_key.eq(runs.focal_model_id).all():
            raise ValueError("Review incomplete runs or inconsistent reported model identities")
        if not runs.start.str["max_days"].eq(runs.max_days).all():
            raise ValueError("CoffeeBench run horizons disagree")

        # 3. Read literal reference instructions without importing/executing the
        # harness. These reconstruct the reference task, not historical API messages.
        code = ast.parse((self.raw_dir / parameters["paths"]["reference"]).read_text())
        assignments = {node.targets[0].id: node.value for node in ast.walk(code)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        instructions = {name: ast.literal_eval(assignments[symbol])
            for name, symbol in parameters["source_symbols"].items()}
        mechanics = next(node for node in code.body if isinstance(node, ast.FunctionDef)
                         and node.name == parameters["labels"]["mechanics_function"])
        instructions["operational_mechanics"] = ast.literal_eval(next(node.value for node in mechanics.body if isinstance(node, ast.Return)))
        endowments = pd.DataFrame([{keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords
            if keyword.arg in {"agent_id", "display_name", "role"}} for node in ast.walk(code)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AgentEndowment"])
        profile = endowments.set_index("agent_id").loc[focal].to_dict()
        participants = ("  - " + endowments.agent_id + " (" + endowments.role + ", " + endowments.display_name + ")").str.cat(sep="\n")
        catalogs = runs[["file", "items"]].explode("items", ignore_index=True)
        catalog = pd.json_normalize(catalogs["items"]).assign(file=catalogs.file)
        catalog["line"] = "  - " + catalog.id + ": " + catalog.name + " — " + catalog.description + catalog.retail_reservation_price.notna().map(
            {True: "  [consumer-facing]", False: ""})
        runs = runs.merge(catalog.groupby("file", sort=False).line.agg("\n".join).rename("catalog"), on="file", validate="one_to_one")
        runs["content"] = [instructions["prompt"].format(**profile, agent_id=focal, max_days=row.max_days,
            score_framing=instructions["score"], persona=instructions["persona"], participants=participants,
            catalog=row.catalog, operational_mechanics=instructions["operational_mechanics"]).strip()
            + "\n\n" + instructions["initial"] for row in runs.itertuples()]

        # 4. Keep the common business task separate from stochastic repetitions.
        # Seeds identify trials; other market participants belong to responses.
        runs["background_models"] = runs.start.str["models"].map(lambda models: {key: value for key, value in models.items() if key != focal})
        if not runs.start.str["models"].str[focal].eq(runs.subject_key).all():
            raise ValueError("CoffeeBench start and usage model labels disagree")
        runs["features"] = [dict(horizon=int(row.max_days), focal_agent=focal,
            prompt_status=parameters["labels"]["prompt_status"]) for row in runs.itertuples()]
        runs["environment"] = runs.features.map(lambda value: json.dumps(value, sort_keys=True))
        runs["item_key"] = runs.groupby(["content", "environment"], sort=True).ngroup()
        subjects = runs[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness=self.name, recorded_model_id=model, historical_inference_settings="not_recorded")
                                for model in subjects.subject_key]
        items = runs.drop_duplicates("item_key").copy()
        items["raw_item_id"] = focal + "-" + items.max_days.astype(str) + "-days"
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in range(len(items))]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["recorded_audit"], sort_keys=True))
        responses = runs.rename(columns={"file": "response_key"})[["response_key", "subject_key", "item_key", "response"]]
        responses["trial"] = runs.groupby(["subject_key", "item_key"], sort=False).seed.rank(method="dense").astype(int)
        responses["test_condition"] = "seed=" + runs.seed.astype(str)
        responses["interactors"] = runs.background_models.map(lambda models: json.dumps(models, sort_keys=True))

        # 5. Preserve every released field, including all agents' events and the
        # distinction between audited income, equity changes and published rounding.
        traces = runs[["file"]].rename(columns={"file": "response_key"})
        traces["trace"] = [json.dumps(dict(source_file=row.file, index_record=row.index_record,
            source_record=row.replay), ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in runs.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CoffeeBench(__file__).main_from_args()
