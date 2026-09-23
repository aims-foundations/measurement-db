#!/usr/bin/env python3
"""Curate REAL's pinned leaderboard and task definitions without filling source gaps."""

import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class REALWebAgents(BenchmarkBuild):

    def download(self):
        return self.fetch_sources('results', 'tasks')
    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform captured REAL results into linked tables in six stages."""
        # 1. Flatten model -> website -> task results, retaining source positions.
        data = json.loads((self.raw_dir / "model_data.json").read_text())
        models = pd.json_normalize(data["aiModels"], max_level=0).assign(
            source_model_row=lambda frame: frame.index, source_file="model_data.json")

        # Select the needed fields; absent optional fields become missing values.
        models = models.reindex(columns=["source_model_row", "id", "name", "run_id", "websites", "source_file"])
        models = models.loc[models["name"].fillna("").astype(bool)]

        websites = models[["source_model_row", "websites"]].explode("websites").dropna(subset=["websites"])
        sites = pd.json_normalize(websites.websites.tolist(), max_level=0).reindex(columns=["tasks"]).assign(
            source_model_row=websites.source_model_row.to_numpy(),
            website_id=[site.get("websiteId") or site.get("name") for site in websites.websites])
        task_rows = sites.explode("tasks", ignore_index=True).dropna(subset=["tasks"])
        attempts = pd.json_normalize(task_rows.tasks.tolist(), max_level=0).reindex(
            columns=["prompt", "accuracy", "evalsFailed", "retrievedAnswer"])
        # Convert IDs before pandas can coerce integer IDs with missing values to floats.
        attempts = attempts.assign(
            raw_item_id=[str(task["id"]) if task.get("id") is not None else None for task in task_rows.tasks],
            source_model_row=task_rows.source_model_row.to_numpy(),
            website_id=task_rows.website_id.to_numpy(),
            source_attempt_row=task_rows.index.to_numpy(),
            source_file="model_data.json")
        models = models.drop(columns="websites").rename(
            columns={"id": "upstream_model_id", "name": "model_name"}
        )

        # 2. Load task definitions; filenames provide their upstream item IDs.
        task_paths = sorted((self.raw_dir / "tasks").glob("*.json"))
        task_records = [json.loads(path.read_text()) for path in task_paths]
        tasks = pd.json_normalize(task_records, max_level=0).assign(
            raw_item_id=[path.stem for path in task_paths],
            source_file=[str(path.relative_to(self.raw_dir)) for path in task_paths])
        if "evals" not in tasks:
            tasks["evals"] = None
        models, tasks, attempts = (
            frame.astype(object).where(frame.notna(), None) for frame in (models, tasks, attempts)
        )

        # 3. Select scored attempts and attach available task grading rules.
        # Ungraded attempts remain available in the input table for inspection.
        scored = attempts.loc[attempts.raw_item_id.notna() & attempts.accuracy.notna()]
        observations = scored.merge(
            tasks[["raw_item_id", "evals"]],
            on="raw_item_id",
            how="left",
            sort=False,
            validate="many_to_one"
        )
        # Only this joined column introduces new missing values (including v2 tasks).
        observations["evals"] = observations.evals.where(observations.evals.notna(), None)

        # 4. Construct subjects and unique items with their grading descriptions.
        # Retain named subjects even when they have no scored attempts.
        subjects = models.rename(columns={
            "source_model_row": "subject_key",
            "model_name": "raw_label"
        })[["subject_key", "raw_label"]]

        # Keep the first captured definition of each task/website combination.
        items = observations.drop_duplicates(["raw_item_id", "website_id"])
        verifier_spec = json.dumps(self.grading["verifiers"]["task_checks"], sort_keys=True)
        provider_mapping = json.dumps(self.grading["verifiers"]["provider_result"], sort_keys=True)
        rules = items.evals.map(
            lambda evals: json.dumps(evals, sort_keys=True) if evals else self.grading["fallback_rule"]
        )
        items = items.assign(
            item_key=range(len(items)),
            content=items.prompt.where(items.prompt.fillna("").astype(bool), items.raw_item_id),
            features=items.website_id.map(lambda site: {"website": site} if site else None),
            grading_criterion=rules.map(lambda rule: {"rule": rule}),
            verifier=items.evals.map(lambda evals: ExactMatcher(spec=verifier_spec if evals else provider_mapping))
        )
        uses_llm = items.evals.map(lambda evals: any(e.get("type") == "llm_boolean" for e in (evals or [])))
        items.loc[uses_llm, "verifier"] = Judge(spec=verifier_spec, judged_by="llm")

        # 5. Link responses to subjects/items and translate provider verdicts.
        responses = observations.merge(
            items[["raw_item_id", "website_id", "item_key"]], on=["raw_item_id", "website_id"],
            how="left", sort=False, validate="many_to_one"
            ).rename(
                columns={"source_attempt_row": "response_key", "source_model_row": "subject_key"}
            )
        has_checks = responses.evalsFailed.notna()
        grades = responses.evalsFailed.map(len, na_action="ignore").eq(0).astype(float)
        grades.loc[~has_checks] = pd.to_numeric(responses.loc[~has_checks, "accuracy"]).ge(100).astype(float)
        responses = responses.assign(response=grades)

        # 6. Extract usable traces, preserving the original, unstripped text.
        has_text = responses.retrievedAnswer.map(lambda answer: isinstance(answer, str))
        normalized = responses.retrievedAnswer.where(has_text, "").astype("string").str.strip()
        traces = responses.loc[
            has_text & ~normalized.isin(["", "Done", "No response"]), ["response_key", "retrievedAnswer"]
        ].rename(columns={"retrievedAnswer": "trace"})

        # Local keys link the frames; the shared builder resolves canonical IDs
        # and numbers trials after aliases have resolved to the same subject/item.
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    REALWebAgents(__file__).main_from_args()
