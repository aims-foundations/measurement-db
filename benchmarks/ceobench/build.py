#!/usr/bin/env python3
"""Curate CEO-Bench's published terminal outcomes and full released histories."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CEOBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        grading = self.grading["verifiers"]

        # 1. Load each model's manifest rows and the complete released run JSON.
        models = pd.DataFrame(json.loads((self.raw_dir / paths["manifest"]).read_text())["models"])
        manifest = models[["model", "model_display", "runs"]].explode("runs", ignore_index=True).rename(
            columns={"model": "panel_model", "model_display": "panel_display", "runs": "manifest_record"})
        manifest = manifest.join(pd.json_normalize(manifest.manifest_record, max_level=0))
        manifest["source_position"] = manifest.index
        native = pd.DataFrame({"source_file": sorted((self.raw_dir / paths["trajectories"]).glob("*.json"))})
        native["run_record"] = native.source_file.map(lambda path: json.loads(path.read_text()))
        native["run_id"] = native.run_record.map(lambda record: record["run_id"])
        native["source_file"] = native.source_file.map(lambda path: str(path.relative_to(self.raw_dir)))

        # 2. Reconcile the complete panel. Endpoint days remain original trace
        # fields: the viewer's nominal-horizon normalization is not a task input.
        runs = manifest.merge(native, on="run_id", how="outer", validate="one_to_one", indicator=True)
        if not runs._merge.eq("both").all():
            raise ValueError("A CEO-Bench manifest row or trajectory lacks its partner")
        runs = runs.drop(columns="_merge").sort_values("source_position").reset_index(drop=True)
        for field in ["model", "model_display", "cash", "bankrupt", "status", "dnf", "action_count"]:
            expected = runs.panel_model if field == "model" else runs[field]
            actual = runs.run_record.map(lambda record: record[field])
            if not actual.eq(expected).all():
                raise ValueError("CEO-Bench manifest and native run disagree: " + field)
        if not runs.status.isin(["complete", "bankrupt"]).all() or runs.dnf.any():
            raise ValueError("Review nonterminal CEO-Bench runs before assigning final outcomes")
        if not runs.bankrupt.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("CEO-Bench bankruptcy must be an explicit Boolean")
        if not runs.status.eq("bankrupt").eq(runs.bankrupt).all():
            raise ValueError("CEO-Bench bankruptcy status is inconsistent")
        if not runs.final_cash.eq(runs.cash).all():
            raise ValueError("CEO-Bench final_cash differs from its published cash")
        if runs.run_record.map(lambda row: bool(row.get("weeks_index")) or row.get("hidden", False)).any():
            raise ValueError("Review newly linked weekly artifacts or hidden run markers")

        # 3. Keep recorded API/provider settings; unspecified historical settings
        # stay unspecified, including the exact harness source revision.
        settings = {}
        for feature, field in parameters["settings"].items():
            runs[field] = pd.Series([row.get(field) for row in runs.run_record], dtype=object)
            settings[feature] = field
        subject_columns = ["panel_model", "panel_display", *settings.values()]
        subjects = runs[subject_columns].drop_duplicates().reset_index(drop=True)
        subjects["subject_key"] = subjects.index
        subjects["raw_label"] = subjects.panel_display
        subjects["features"] = [dict(harness=self.name, source_model=row["panel_model"],
            **{feature: row[field] for feature, field in settings.items() if pd.notna(row[field])})
            for row in subjects.to_dict("records")]
        runs = runs.merge(subjects[subject_columns + ["subject_key"]], on=subject_columns, how="left", validate="many_to_one")

        # 4. Each reported measure has its own grading protocol and scale. The
        # stimulus is the published task instruction, never the observed outcome.
        items = pd.DataFrame.from_dict(grading["measures"], orient="index").rename_axis("metric").reset_index()
        items["item_key"] = items.metric
        items["raw_item_id"] = parameters["scenario"]["raw_item_id"] + "/" + items.metric
        items["content"] = (self.raw_dir / paths["instructions"]).read_text().strip()
        items["grading_criterion"] = [dict(rule=row.rule, response_scale=row.scale) for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=json.dumps(dict(**grading["protocol"], metric=metric), sort_keys=True))
                             for metric in items.metric]
        runs["reported_final_cash"] = runs.final_cash
        runs["survived"] = (~runs.bankrupt).astype(float)
        runs["above_starting_cash"] = runs.final_cash.gt(float(parameters["scenario"]["starting_cash"])).astype(float)
        responses = runs[["run_id", "subject_key", *items.metric]].melt(
            id_vars=["run_id", "subject_key"], value_vars=list(items.metric), var_name="item_key", value_name="response")
        responses["response_key"] = responses.index
        responses["test_condition"] = "outcome=" + responses.item_key

        # 5. Link every metric to its complete manifest and trajectory. The
        # shared registrar numbers distinct runs after canonical IDs resolve.
        runs["trace"] = [json.dumps(dict(manifest=row.manifest_record, source_file=row.source_file,
            trajectory=row.run_record), ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in runs.itertuples()]
        traces = responses[["response_key", "run_id"]].merge(runs[["run_id", "trace"]], on="run_id", validate="many_to_one")
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    CEOBench(__file__).main_from_args()
