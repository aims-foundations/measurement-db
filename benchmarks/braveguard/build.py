#!/usr/bin/env python3
"""Import historical BraveGuard decisions and their complete source trajectories."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BraveGuard(BenchmarkBuild):
    def download(self):
        return self.fetch_sources(*[source["name"] for source in self.source_manifest["upstream"]])

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read each native guard export and join its documented configuration.
        configurations = pd.DataFrame({"subject_key": list(parameters["results"]),
            "source_file": list(parameters["results"].values())})
        configurations["raw_label"] = configurations.subject_key.map(parameters["models"])
        configurations["collection"] = configurations.subject_key.map(parameters["result_collections"])
        configurations["features"] = configurations.subject_key.map(lambda key: parameters["subject_" + key])
        frames = []
        for source in configurations.itertuples():
            frame = pd.read_csv(self.raw_dir / source.source_file, dtype=str, keep_default_na=False)
            frame["record"] = frame.to_dict("records")
            frames.append(frame.assign(subject_key=source.subject_key, source_row=frame.index + 2))
        observations = pd.concat(frames, ignore_index=True).merge(
            configurations, on="subject_key", how="left", validate="many_to_one")
        if observations.duplicated(["subject_key", "session_id"]).any():
            raise ValueError("Repeated native guard observation")
        for column in ["harmful", "expected_harmful", "correct"]:
            if not observations[column].isin(["True", "False"]).all():
                raise ValueError(f"Missing or invalid native {column} value")
        if not observations.correct.eq(observations.harmful.eq(observations.expected_harmful).map(str)).all():
            raise ValueError("Published correctness disagrees with the published decision and reference")

        # 2. Read the actual trajectory inputs, keeping the two source collections distinct.
        inputs = pd.DataFrame([dict(collection=collection,
            source_trajectory=str(path.relative_to(self.raw_dir)), session_id=path.stem.split("-")[1])
            for collection, directory in parameters["collections"].items()
            for path in sorted((self.raw_dir / directory).glob("session_item-*.jsonl"))])
        inputs["trajectory"] = inputs.source_trajectory.map(
            lambda path: [json.loads(line) for line in (self.raw_dir / path).read_text().splitlines() if line.strip()])
        inputs["item_key"] = inputs.collection + ":" + inputs.session_id
        observations = observations.merge(inputs, on=["collection", "session_id"],
            how="left", validate="many_to_one")
        if observations.item_key.isna().any():
            raise ValueError("A guard decision has no source trajectory")

        # 3. Reconstruct the documented message formatter with ordered table operations.
        records = inputs[["item_key", "trajectory"]].explode("trajectory", ignore_index=True)
        records["type"] = records.trajectory.map(lambda row: row.get("type"))
        if not records.type.isin(["session", "model_change", "thinking_level_change", "custom", "message"]).all():
            raise ValueError("An unfamiliar trajectory record needs formatter review")
        backend = records.loc[records.type.eq("model_change")].copy()
        backend["backend"] = backend.trajectory.map(lambda row: dict(provider=row["provider"], model=row["modelId"]))
        messages = records.loc[records.type.eq("message")].copy()
        messages["message_number"] = messages.index
        messages["role"] = messages.trajectory.map(lambda row: row.get("message", {}).get("role", "unknown"))
        messages["content"] = messages.trajectory.map(lambda row: row.get("message", {}).get("content", ""))
        parts = messages.explode("content").copy()
        parts["part_type"] = parts.content.map(lambda part: part.get("type") if isinstance(part, dict) else "literal")
        parts["text"] = parts.content.map(lambda part: part if isinstance(part, str) else "")
        text = parts.part_type.eq("text")
        parts.loc[text, "text"] = parts.loc[text, "content"].map(lambda part: part.get("text", ""))
        tool = parts.part_type.eq("toolCall")
        parts.loc[tool, "text"] = parts.loc[tool, "content"].map(lambda part:
            f"[ToolCall: {part.get('name', '')}({json.dumps(part.get('arguments', {}), ensure_ascii=False)[:int(parameters['formatter']['tool_argument_characters'])]})]")
        retained = parts.part_type.isin(["literal", "text", "toolCall"])
        message_text = parts.loc[retained].groupby("message_number", sort=False).text.agg("\n".join)
        messages["text"] = "[" + messages.role + "] " + messages.message_number.map(message_text).fillna("")
        trajectory_text = messages.groupby("item_key", sort=False).text.agg("\n".join)

        # 4. Build guard inputs and keep released safety references outside those inputs.
        observations["trajectory_key"] = observations.item_key
        observations["item_key"] = observations.subject_key + ":" + observations.trajectory_key
        items = observations[["item_key", "trajectory_key", "subject_key", "collection", "session_id", "source_trajectory", "expected_harmful"]].drop_duplicates()
        if items.item_key.duplicated().any():
            raise ValueError("Conflicting reference labels for one trajectory")
        items["raw_item_id"] = items.item_key
        items["trajectory_text"] = items.trajectory_key.map(trajectory_text)
        items["content"] = [json.dumps([
            dict(role="system", content=parameters["system_prompts"][row.subject_key]),
            dict(role="user", content=parameters["prompt_templates"][row.subject_key].format(
                instruction=parameters["instructions"][row.subject_key], trajectory=row.trajectory_text))],
            ensure_ascii=False) for row in items.itertuples()]
        items = items.merge(backend[["item_key", "backend"]].rename(columns={"item_key": "trajectory_key"}),
            on="trajectory_key", how="left", validate="many_to_one")
        items["features"] = [dict(collection=row.collection, session_id=row.session_id,
            source_trajectory=row.source_trajectory, prompt_revision=row.subject_key,
            trajectory_provider=row.backend["provider"], trajectory_model=row.backend["model"]) for row in items.itertuples()]
        items["grading_criterion"] = items.expected_harmful.map(lambda value: dict(
            reference_answer="harmful" if value == "True" else "non-harmful", rule=self.grading["rule"]))
        verifier = json.dumps(self.grading["verifiers"]["recorded_decision"], sort_keys=True)
        items["verifier"] = [ExactMatcher(spec=verifier) for _ in range(len(items))]

        # 5. Preserve every original decision, output field and unshortened trajectory.
        observations["response_key"] = observations.subject_key + ":" + observations.session_id
        observations["response"] = observations.harmful.map({"False": 0.0, "True": 1.0})
        observations["trial"] = 1
        observations["test_condition"] = "temperature=" + parameters["inference"]["temperature"] + ";mode=3;collection=" + observations.collection
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            record=row.record, source_trajectory=row.source_trajectory, trajectory=row.trajectory),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": configurations[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BraveGuard(__file__).main_from_args()
