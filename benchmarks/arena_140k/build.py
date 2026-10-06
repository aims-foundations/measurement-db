#!/usr/bin/env python3
"""Curate Arena's released human votes, conversational context and complete records."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Arena140k(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Concatenate the native shards, retaining every field and source position.
        frames = []
        for path in sorted((self.raw_dir / self.build_parameters["layout"]["release"] / "data").glob("*.parquet")):
            frame = pd.read_parquet(path)
            frame["record"] = json.loads(frame.to_json(
                orient="records", date_format="iso", date_unit="ns", force_ascii=False))
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        battles = pd.concat(frames, ignore_index=True)
        if battles.id.isna().any() or battles.id.duplicated().any():
            raise ValueError("Arena feedback IDs must be present and unique")

        # 2. Retain the shared user turns and earlier context, excluding current model replies.
        turns = battles[["id", "conversation_a"]].explode("conversation_a", ignore_index=True)
        messages = pd.json_normalize(turns.conversation_a, max_level=0).assign(id=turns.id)
        prompts = messages.loc[messages.role.eq("user")].groupby("id", sort=False).agg(
            user_turns=("content", list), turn_count=("content", "size"))
        items = battles.merge(prompts, on="id", how="left", validate="one_to_one")
        if items.turn_count.isna().any():
            raise ValueError("An Arena vote has no user turn")
        items["prior_context"] = [row.full_conversation[:-row.turn_count] for row in items.itertuples()]
        # Split only JSONL record delimiters; user text can contain Unicode line separators.
        items["content"] = items[["prior_context", "user_turns"]].to_json(
            orient="records", lines=True, force_ascii=False).split("\n")[:-1]
        items = items.assign(item_key=items.id, raw_item_id=items.id,
            features=items.language.map(lambda language: {"lang": language}),
            grading_criterion=[{"rule": self.grading["rule"]}] * len(items),
            verifier=Judge(spec=json.dumps(self.grading["verifiers"]["human_vote"], sort_keys=True), judged_by="human"))

        # 3. Unpivot each vote into its two participating models and join the outcome mapping.
        responses = battles.melt(id_vars=["id", "winner"], value_vars=["model_a", "model_b"],
            var_name="side", value_name="subject_key")
        scores = pd.DataFrame(self.grading["verifiers"]["human_vote"]["scores"]).rename_axis("winner").reset_index().melt(
            id_vars="winner", var_name="side", value_name="response")
        responses = responses.merge(scores, on=["winner", "side"], how="left", validate="many_to_one")
        if responses.subject_key.isna().any() or responses.response.isna().any():
            raise ValueError("An Arena vote has an unknown model or outcome")
        opponents = responses[["id", "side", "subject_key"]].rename(columns={"subject_key": "opponent"})
        opponents["side"] = opponents.side.map({"model_a": "model_b", "model_b": "model_a"})
        responses = responses.merge(opponents, on=["id", "side"], validate="one_to_one")
        responses = responses.assign(response_key=responses.id + "/" + responses.side,
            item_key=responses.id, interactors="opponent=" + responses.opponent, test_condition="side=" + responses.side)

        # 4. Preserve source model identities and attach the entire battle record to each side.
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects["features"] = subjects.subject_key.map(
            lambda model: {"model_identifier": model, **self.build_parameters["subject"]})
        traces = responses[["response_key", "id", "side"]].merge(
            battles[["id", "source_file", "source_row", "record"]], on="id", validate="many_to_one")
        traces["trace"] = traces[["source_file", "source_row", "side", "record"]].to_json(
            orient="records", lines=True, force_ascii=False).split("\n")[:-1]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition", "interactors"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    Arena140k(__file__).main_from_args()
