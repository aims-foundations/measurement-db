#!/usr/bin/env python3
"""Curate original LLMail-Inject endpoint outcomes without executing attacks."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class LLMailInject(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "context", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["root"]

        # 1. Read both native exports and preserve every original field before joining.
        frames = []
        for phase, filename in parameters["results"].items():
            frame = pd.read_json(root / filename, lines=True, dtype=False, convert_dates=False)
            frame = frame.astype(object).where(frame.notna(), None)
            frame["native_record"] = frame.to_dict("records")
            frames.append(frame.assign(phase=phase, source_row=frame.index,
                                       source_file=str((root / filename).relative_to(self.raw_dir))))
        native = pd.concat(frames, ignore_index=True)
        flags = pd.json_normalize(native.objectives.map(json.loads))
        objectives = json.loads((root / parameters["paths"]["objectives"]).read_text())
        if set(flags.columns) != set(objectives) or not flags.stack().dropna().map(type).eq(bool).all():
            raise ValueError("Unexpected native objective names or nonboolean verdicts")
        native["response"] = flags.eq(True).all(axis=1).astype("Float64").where(flags.notna().all(axis=1))

        # 2. Remove requests that never reached a configured target; failed jobs stay null.
        observations = native.loc[native.started_time.notna()
                                  & native.output.ne(parameters["request_status"]["missing_scenario"])].copy()
        observations = observations.join(observations.scenario.str.extract(parameters["patterns"]["level"]))
        if observations[["scenario_number", "level_letter"]].isna().any().any():
            raise ValueError("Unknown scenario identifier")
        observations["scenario_key"] = "scenario_" + observations.scenario_number
        observations["subject_key"] = observations.phase + ":" + observations.level_letter
        observations = observations.rename(columns={"subject": "email_subject"})

        # 3. Targets are AI systems, with separate phase/defense configurations.
        levels = pd.DataFrame(json.loads((root / parameters["paths"]["levels"]).read_text()))
        levels = levels.rename_axis("level_letter").reset_index().melt(
            id_vars="level_letter", var_name="phase", value_name="reported_configuration").dropna()
        levels = levels.join(levels.reported_configuration.str.extract(parameters["patterns"]["configuration"]))
        subjects = observations[["subject_key", "phase", "level_letter"]].drop_duplicates().merge(
            levels, on=["phase", "level_letter"], how="left", validate="one_to_one")
        subjects["raw_label"] = subjects.model_family.map(parameters["models"])
        subjects["raw_label"] = subjects.raw_label.fillna(subjects.subject_key.map(parameters["legacy_targets"]))
        if subjects.raw_label.isna().any():
            raise ValueError("An observed target has no documented or explicitly unresolved identity")
        subjects["features"] = [{**parameters["subject_features"], "challenge_phase": row.phase,
                                 "level_letter": row.level_letter,
                                 **({"defense": row.defense, "model_identifier": parameters["model_identifiers"][row.model_family]}
                                    if pd.notna(row.model_family) else {"configuration_status": "unreported"})}
                                for row in subjects.itertuples()]

        # 4. Deduplicate email/scenario inputs, attaching the released background pools.
        # JSON escapes embedded NULs, which pandas multi-column deduplication conflates.
        observations["item_key"] = [json.dumps([row.scenario_key, row.email_subject, row.body])
                                    for row in observations.itertuples()]
        scenarios = pd.DataFrame.from_dict(json.loads((root / parameters["paths"]["scenarios"]).read_text()),
                                           orient="index").rename_axis("scenario_key").reset_index()
        scenarios["attachments"] = [[{"path": row.scenario_key + "_background_emails.json",
                                       "role": "released_scenario_corpus", "media_type": "application/json",
                                       "data": json.dumps(row.emails, ensure_ascii=False).encode("utf-8")}]
                                    for row in scenarios.itertuples()]
        items = observations.drop_duplicates("item_key").merge(
            scenarios, on="scenario_key", how="left", validate="many_to_one")
        if items.user_query.isna().any():
            raise ValueError("A source scenario has no released context")
        items["raw_item_id"] = items.job_id
        inputs = items[["scenario_key", "user_query", "position", "email_subject", "body"]].rename(
            columns={"body": "email_body", "position": "released_insertion_position"})
        # Escapes also keep distinct Unicode sequences intact through shared NFC hashing.
        items["content"] = [json.dumps(row, ensure_ascii=True, sort_keys=True) for row in inputs.to_dict("records")]
        items["grading_criterion"] = [{"rule": json.dumps({"attack_objective": task, "success_rule": self.grading["rule"]})}
                                     for task in items.task]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["objectives"], sort_keys=True))

        # 5. Keep repeated jobs and full returned records; output echoes are not generations.
        responses = observations.copy()
        responses["response_key"] = responses.job_id
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "record": row.native_record}, ensure_ascii=False, allow_nan=False)
                           for row in responses.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    LLMailInject(__file__).main_from_args()
