"""Join WorkBench's released verdicts, task definitions and recorded transcripts."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class WorkBenchRevisited(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "verdicts")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the authors' model manifest and its selected result files.
        layout = self.build_parameters["layout"]
        repo = self.raw_dir / layout["repository"]
        report = json.loads((repo / layout["model_summary"]).read_text())
        models = pd.DataFrame.from_dict(report["models"], orient="index").rename_axis("label").reset_index()
        selected = pd.DataFrame(models.sources.tolist(), index=models.model_name).rename_axis(
            index="model", columns="domain",
        ).stack().rename("results_file").reset_index()
        runs = pd.concat([
            pd.read_csv(repo / path, dtype=str, keep_default_na=False).assign(results_file=path)
            for path in selected.results_file
        ], ignore_index=True)

        # 2. Select the published Revisited verdicts and join the original, unmodified traces.
        verdicts = pd.read_csv(self.raw_dir / layout["verdicts"], keep_default_na=False)
        verdicts = verdicts.loc[verdicts.run_group.eq(layout["run_group"])].copy()
        if not pd.api.types.is_bool_dtype(verdicts.correct) or not verdicts.ground_truth_version.eq("v2").all():
            raise ValueError("Expected boolean WorkBench verdicts against the v2 task definitions")
        observations = verdicts.merge(
            selected, on=["model", "domain", "results_file"], how="left", validate="many_to_one", indicator=True,
        )
        if not observations._merge.eq("both").all() or set(verdicts.results_file) != set(selected.results_file):
            raise ValueError("The WorkBench verdict export and model manifest select different runs")
        observations = observations.drop(columns="_merge").merge(
            runs[["results_file", "task", "full_response"]], on=["results_file", "task"],
            how="outer", validate="one_to_one", indicator=True,
        )
        if not observations._merge.eq("both").all():
            raise ValueError("A WorkBench verdict or recorded attempt has no counterpart")
        totals = observations.groupby("model").correct.agg(["sum", "size"])
        expected = models.set_index("model_name")[["correct", "total"]].rename(columns={"correct": "sum", "total": "size"})
        if not totals.eq(expected).all().all():
            raise ValueError("WorkBench per-task grades disagree with the released model totals")

        # 3. Retain model names and each task's reference actions and grading protocol.
        subjects = models.rename(columns={"model_name": "subject_key"}).assign(
            raw_label=models.label.replace(self.build_parameters["display_names"]),
        )
        observations = observations.assign(item_key=observations.domain + "::" + observations.task)
        items = observations[["item_key", "task", "ground_truth_actions"]].drop_duplicates()
        if items.item_key.duplicated().any():
            raise ValueError("A WorkBench task has conflicting reference actions")
        items = items.assign(
            raw_item_id=items.item_key, content=items.task,
            grading_criterion=items.ground_truth_actions.map(
                lambda actions: {"rule": self.grading["rule"], "reference_answer": actions}),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["sandbox"], sort_keys=True)),
        )

        # 4. Preserve every released verdict and every available full-response string.
        responses = observations.rename(columns={"model": "subject_key"}).assign(
            response_key=observations.results_file + "::" + observations.task,
            response=observations.correct.astype(float), trial=1,
        )
        traces = responses.loc[responses.full_response.ne(""), ["response_key", "full_response"]].rename(
            columns={"full_response": "trace"},
        )
        return {
            "subjects": subjects[["subject_key", "raw_label"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": traces,
        }


if __name__ == "__main__":
    WorkBenchRevisited(__file__).main_from_args()
