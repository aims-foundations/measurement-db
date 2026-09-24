"""Join AgentDojo's native runs to task definitions and preserve both outcome types."""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.agentdojo_tasks import (
    identify_model_and_defense, load_task_prompts,
)


class AgentDojo(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load run records; parse the released Python task literals without executing them.
        layout = self.build_parameters["layout"]
        with tarfile.open(self.raw_dir / layout["archive"]) as archive, tempfile.TemporaryDirectory(
            prefix="agentdojo-tasks-", dir=self.dir,
        ) as scratch:
            files = {m.name.split("/", 1)[1]: m for m in archive.getmembers() if m.isfile()}
            task_sources = [m for name, m in files.items() if name.startswith(layout["tasks"]) and name.endswith(".py")]
            archive.extractall(scratch, members=task_sources, filter="data")
            root = task_sources[0].name.split("/", 1)[0]
            prompts, goals, references = load_task_prompts(Path(scratch) / root / layout["tasks"])
            paths = sorted(name for name in files if name.startswith("runs/") and name.endswith(".json"))
            runs = pd.json_normalize([json.load(archive.extractfile(files[name])) for name in paths], max_level=0)
        runs["source_file"] = paths
        location = runs.source_file.str.extract(
            r"^runs/(?P<subject_key>[^/]+)/(?P<suite>[^/]+)/(?P<task>[^/]+)/(?P<attack>[^/]+)/[^/]+\.json$"
        )
        if location.isna().any().any():
            raise ValueError("Unexpected AgentDojo run path")
        runs = runs.join(location).assign(
            suite=runs.suite_name.fillna(location.suite), task=runs.user_task_id.fillna(location.task),
            attack=runs.attack_type.replace("", None).fillna(location.attack.replace("none", None)),
        )

        # 2. Keep model/defense identity; turn utility and attacker success into separate observations.
        subjects = runs[["subject_key"]].drop_duplicates().copy()
        subjects[["raw_label", "defense"]] = pd.DataFrame(
            subjects.subject_key.map(identify_model_and_defense).tolist(), index=subjects.index,
        )
        subjects["features"] = subjects.apply(lambda row: {
            "harness": "AgentDojo", "source_model_variant": row.raw_label,
            **({"defense": row.defense} if pd.notna(row.defense) and row.defense else {}),
        }, axis=1)
        observations = runs.melt(
            id_vars=["source_file", "subject_key", "suite", "task", "attack", "injection_task_id", "messages"],
            value_vars=["utility", "security"], var_name="metric", value_name="outcome",
        )
        observations = observations.loc[observations.metric.eq("utility") | observations.attack.notna()].copy()
        if not observations.outcome.map(lambda value: pd.isna(value) or isinstance(value, bool)).all():
            raise TypeError("AgentDojo outcomes must be released booleans or null")
        observations = observations.assign(
            injection=observations.injection_task_id.fillna("none").where(observations.metric.eq("security"), ""),
            response=observations.outcome.map(lambda value: None if pd.isna(value) else float(value)),
            response_key=observations.source_file + ":" + observations.metric,
            interactors=("attacker=" + observations.attack).where(observations.attack.notna(), None),
        )
        observations["item_key"] = observations[["suite", "task", "metric", "injection"]].apply(
            lambda row: json.dumps(row.tolist()), axis=1,
        )

        # 3. Join user prompts and injection goals; attach the metric's grading protocol.
        tasks = pd.concat({"prompt": pd.Series(prompts).combine_first(pd.Series(goals)),
                           "reference_answer": pd.Series(references)}, axis=1)
        tasks.index.names = ["suite", "task"]
        injections = pd.Series(goals, name="goal").rename_axis(["suite", "injection"]).reset_index()
        items = observations[["item_key", "suite", "task", "metric", "injection"]].drop_duplicates().merge(
            tasks.reset_index(), on=["suite", "task"], how="left", validate="many_to_one",
        ).merge(injections, on=["suite", "injection"], how="left", validate="many_to_one")
        security = items.metric.eq("security")
        if items.prompt.isna().any() or (security & items.injection.ne("none") & items.goal.isna()).any():
            raise ValueError("An AgentDojo run lacks its user prompt or injection goal")
        items = items.assign(
            raw_item_id=items.suite + "::" + items.task + ("::" + items.injection).where(security, ""),
            content=items.prompt.where(~security, "[user_task] " + items.prompt
                + ("\n[injection_task] " + items.goal).fillna("")),
            grading_criterion=items.apply(lambda row: {
                **self.grading["verifiers"][row.metric]["criterion"],
                **({"reference_answer": row.reference_answer}
                   if row.metric == "utility" and pd.notna(row.reference_answer) and row.reference_answer else {}),
            }, axis=1),
            verifier=items.metric.map(lambda metric: ExactMatcher(
                spec=json.dumps(self.grading["verifiers"][metric]["verifier"], sort_keys=True))),
            features=items[["suite"]].to_dict("records"),
        )

        # 4. Preserve complete message histories, including recorded attempts without a grade.
        traces = observations.loc[observations.messages.map(lambda value: isinstance(value, list) and bool(value))].copy()
        traces["trace"] = traces.messages.map(lambda value: json.dumps(value, ensure_ascii=False))
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "interactors"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    AgentDojo(__file__).main_from_args()
