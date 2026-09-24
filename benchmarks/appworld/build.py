"""Convert AppWorld's official task and leaderboard bundles into linked tables."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.appworld_bundle import open_bundle


class AppWorldBuild(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "tasks")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load each experiment's native result table and available log files.
        experiments = []
        for path in sorted((self.raw_dir / "bundles").glob("*.bundle")):
            name = path.stem
            with open_bundle(path) as bundle:
                metadata = json.loads(bundle.read(f"{name}/metadata.json"))
                split, method = metadata["dataset"], metadata["method"]["name"]
                model = metadata["llm"].get("tooltip") or metadata["llm"]["name"]
                if split not in {"test_normal", "test_challenge"}:
                    raise ValueError(f"Unknown AppWorld split in {name}: {split}")
                evaluation = json.loads(bundle.read(f"{name}/evaluations/{split}.json"))
                results = pd.DataFrame.from_dict(evaluation["individual"], orient="index").rename_axis(
                    "item_key"
                ).reset_index()
                log_paths = sorted(p for p in bundle.namelist() if p.endswith("/logs/environment_io.md"))
                logs = pd.DataFrame({
                    "item_key": [p.split("/")[-3] for p in log_paths],
                    "trace": [bundle.read(p).decode("utf-8", "replace") for p in log_paths],
                })
                experiments.append(results.merge(logs, on="item_key", how="left", validate="one_to_one").assign(
                    source_file=path.name, raw_label=model, harness=method, split=split,
                    subject_key=json.dumps([model, method]),
                ))
        observations = pd.concat(experiments, ignore_index=True)
        if not observations.success.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("AppWorld results must contain native success booleans")

        # 2. Join evaluated tasks to the released instructions and grading checks.
        with open_bundle(self.raw_dir / "data-0.1.0.bundle") as bank:
            spec_paths = sorted(p for p in bank.namelist() if p.startswith("data/tasks/") and p.endswith("/specs.json"))
            tasks = pd.json_normalize([json.loads(bank.read(p)) for p in spec_paths], max_level=0).assign(
                item_key=[p.split("/")[-2] for p in spec_paths]
            )
            items = observations[["item_key", "split"]].drop_duplicates().merge(
                tasks, on="item_key", how="left", validate="one_to_one", indicator=True
            )
            if not items._merge.eq("both").all() or not items.instruction.fillna("").str.strip().astype(bool).all():
                raise ValueError("An evaluated AppWorld task lacks its released instruction")
            prefixes = "data/tasks/" + items.item_key + "/ground_truth/"
            items = items.assign(
                verifier=prefixes.map(lambda p: ExactMatcher(spec=bank.read(p + "evaluation.py").decode("utf-8"))),
                grading_criterion=prefixes.map(lambda p: {"rule": json.dumps({
                    "criterion": self.grading["rule"],
                    "test_data": json.loads(bank.read(p + "test_data.json")),
                }, sort_keys=True)}),
                features=items.rename(columns={"item_key": "appworld_task_id"})[
                    ["appworld_task_id", "split", "db_version", "datetime"]
                ].to_dict("records"),
                raw_item_id=items.item_key, content=items.instruction,
            )

        # 3. Each model/scaffold combination is a subject; split belongs to items.
        subjects = observations[["subject_key", "raw_label", "harness"]].drop_duplicates()
        subjects = subjects.assign(features=subjects[["harness"]].to_dict("records"))
        responses = observations.assign(
            response_key=observations.source_file + "/" + observations.item_key,
            response=observations.success.astype(float), trial=1,
        )

        # 4. Preserve complete log text; missing or empty logs are not traces.
        traces = responses.loc[responses.trace.fillna("").str.strip().astype(bool), ["response_key", "trace"]]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "trial", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AppWorldBuild(__file__).main_from_args()
