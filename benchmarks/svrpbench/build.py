#!/usr/bin/env python3
"""Join SVRPBench's released individual routes to their captured instance definitions."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SVRPBench(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("instances", "results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read original instance parameters and individual solver exports.
        bank = pd.read_parquet(self.raw_dir / "instances.parquet")
        paths = sorted((self.raw_dir / "results").glob("*.json"))
        text = [path.read_text() for path in paths]
        runs = pd.json_normalize([json.loads(value) for value in text], max_level=0).assign(trace=text)
        metrics = pd.json_normalize(runs.metrics.tolist()).add_prefix("metric.")
        runs = runs.drop(columns="metrics").join(metrics)

        # 2. Check both the upstream instance key and its coordinates before joining.
        instances = bank.rename(columns={"subset_name": "problem", "instance_id": "instance"})
        observations = runs.merge(instances, on=["problem", "instance"], suffixes=("_result", "_bank"),
                                  how="left", sort=False, validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("A released SVRPBench result has no matching instance")
        coordinates = observations[["locations_result", "locations_bank"]].to_json(orient="records")
        if any(row["locations_result"] != row["locations_bank"] for row in json.loads(coordinates)):
            raise ValueError("SVRPBench instance coordinates disagree with the result release")

        # 3. Describe each task with every released parameter, excluding solver output.
        keys = observations[["problem", "instance"]].drop_duplicates()
        definitions = keys.merge(instances, on=["problem", "instance"], validate="one_to_one")
        descriptions = definitions.to_json(orient="records", force_ascii=False)
        definitions = definitions[["problem", "instance"]].assign(
            content=[json.dumps(row, sort_keys=True, ensure_ascii=False) for row in json.loads(descriptions)]
        )

        # 4. Each metric supplies a separate grading protocol for the same instance.
        specs = self.grading["verifiers"]
        values = ["metric." + name for name in specs]
        responses = observations.melt(id_vars=["problem", "instance", "solver", "trace"],
                                       value_vars=values, var_name="metric", value_name="response")
        responses["metric"] = responses.metric.str.removeprefix("metric.")
        items = responses[["problem", "instance", "metric"]].drop_duplicates().merge(
            definitions, on=["problem", "instance"], how="left", sort=False, validate="many_to_one"
        )
        items = items.assign(
            item_key=range(len(items)),
            raw_item_id=items.problem + ":" + items.instance.astype(str) + ":" + items.metric,
            grading_criterion=items.metric.map(lambda metric: {
                "rule": specs[metric]["rule"], "response_scale": specs[metric]["response_scale"]
            }),
            verifier=items.metric.map(lambda metric: ExactMatcher(spec=json.dumps(
                {"source": specs[metric]["source"], "metric": metric, "kind": "released_solver_metric"}, sort_keys=True
            ))),
        )

        # 5. Link each recorded metric and preserve its complete native route export.
        subjects = runs[["solver"]].drop_duplicates().rename(columns={"solver": "subject_key"})
        subjects = subjects.assign(raw_label=subjects.subject_key,
                                   features=subjects.subject_key.map(lambda solver: {"harness": "SVRPBench", "solver": solver}))
        responses = responses.merge(items[["problem", "instance", "metric", "item_key"]],
                                     on=["problem", "instance", "metric"], how="left", sort=False, validate="many_to_one")
        responses = responses.assign(response_key=responses.index, subject_key=responses.solver,
                                     test_condition="released individual route;metric=" + responses.metric)
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses[["response_key", "trace"]],
        }


if __name__ == "__main__":
    SVRPBench(__file__).main_from_args()
