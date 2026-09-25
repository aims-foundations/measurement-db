#!/usr/bin/env python3
"""Curate BEDD's recorded human judgments and their original gameplay-video links."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BEDD(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("judgments", "videos", "readme", "instructions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the source annotations and apply the author's worker exclusion.
        parameters = self.build_parameters
        paths = parameters["paths"]
        grading = self.grading["verifiers"]["human"]
        with ZipFile(self.raw_dir / paths["judgments"]) as archive:
            originals = pd.DataFrame({"record": json.loads(archive.read(paths["answers_member"]))})
            banned = pd.DataFrame(json.loads(archive.read(paths["banned_member"])))
        matches = pd.json_normalize(originals.record, max_level=0).join(originals)
        matches = matches.loc[~matches.worker_id.isin(banned.workerId)].rename_axis("source_row").reset_index()
        detail = pd.json_normalize(matches.result).rename(columns={
            "eval_metadata.win_player": "winner", "eval_metadata.responses.direct_question": "direct",
            "eval_metadata.responses.comparisons": "comparisons"})
        matches = matches.join(detail[["winner", "direct", "comparisons"]])
        if matches.hash.duplicated().any() or not matches.episodes.str.len().eq(2).all():
            raise ValueError("Each source annotation must identify a unique two-player comparison")

        episodes = matches[["source_row", "episodes"]].explode("episodes", ignore_index=True)
        episodes["position"] = episodes.groupby("source_row", sort=False).cumcount()
        episodes = episodes.drop(columns="episodes").join(pd.json_normalize(episodes.episodes).rename(
            columns={"agent_name": "subject_key", "hash": "episode_hash"}))
        opponents = episodes.rename(columns={"subject_key": "opponent",
            "episode_hash": "opponent_episode", "task": "opponent_task", "seed": "opponent_seed"})
        opponents["position"] = 1 - opponents.position
        episodes = episodes.merge(opponents, on=["source_row", "position"], validate="one_to_one")
        if not (episodes.task.eq(episodes.opponent_task) & episodes.seed.eq(episodes.opponent_seed)).all():
            raise ValueError("Paired agents must share the same task and world seed")

        # 2. Unpivot direct questions, and expand comparative judgments by player.
        direct = matches[["source_row", "direct"]].explode("direct", ignore_index=True)
        direct["question_index"] = direct.groupby("source_row", sort=False).cumcount()
        direct = direct.drop(columns="direct").join(pd.json_normalize(direct.direct))
        direct = direct.melt(id_vars=["source_row", "question_index", "question"],
            value_vars=["player_1", "player_2"], var_name="player", value_name="answer")
        if not direct.answer.isin(["true", "false"]).all():
            raise ValueError("Unexpected direct-question answer")
        direct = direct.assign(position=direct.player.map({"player_1": 0, "player_2": 1}),
            metric="direct", response=direct.answer.eq("true").astype(float)).drop(columns="player")
        direct = direct.merge(episodes, on=["source_row", "position"], validate="many_to_one")

        comparative = matches[["source_row", "comparisons"]].explode("comparisons", ignore_index=True)
        comparative["question_index"] = comparative.groupby("source_row", sort=False).cumcount()
        comparative = comparative.drop(columns="comparisons").join(pd.json_normalize(comparative.comparisons))
        comparative["metric"] = "comparison"
        overall = matches[["source_row", "winner"]].rename(columns={"winner": "answer"}).assign(
            metric="overall", question_index=0, question=grading["overall_question"])
        comparative = pd.concat([overall, comparative], ignore_index=True).merge(
            episodes, on="source_row", validate="many_to_many")
        if not comparative.answer.isin(["p1", "p2", "draw", "na"]).all():
            raise ValueError("Unexpected comparative answer")
        comparative["response"] = comparative.answer.eq("p" + (comparative.position + 1).astype(str)).astype(float)
        comparative.loc[comparative.answer.eq("draw"), "response"] = 0.5
        comparative.loc[comparative.answer.eq("na"), "response"] = float("nan")
        observations = pd.concat([direct, comparative], ignore_index=True)
        observations = observations.loc[~observations.subject_key.isin(parameters["human_references"].values())].merge(
            matches[["source_row", "hash", "worker_id", "record"]], on="source_row", validate="many_to_one")

        # 3. Match each recorded episode to its video; videos are outputs, not inputs.
        with ZipFile(self.raw_dir / paths["videos"]) as archive:
            videos = pd.DataFrame([dict(member=entry.filename, byte_size=entry.file_size, crc32=entry.CRC)
                for entry in archive.infolist() if entry.filename.endswith(".mp4")])
        videos = videos.join(videos.member.str.extract(parameters["options"]["video_path_pattern"]))
        if videos[["subject_key", "task", "seed"]].isna().any().any() or videos.duplicated(["subject_key", "task", "seed"]).any():
            raise ValueError("Video paths do not identify unique agent/task/seed records")
        observations = observations.merge(videos, on=["subject_key", "task", "seed"], how="left", validate="many_to_one")
        reference = videos.rename(columns={"subject_key": "opponent", "member": "reference_member",
            "byte_size": "reference_byte_size", "crc32": "reference_crc32"})
        observations = observations.merge(reference, on=["opponent", "task", "seed"], how="left", validate="many_to_one")
        if observations.member.isna().any() or observations.reference_member.isna().any():
            raise ValueError("A recorded generation or reference has no released video")

        # 4. Keep world/task inputs separate from human grading and comparison context.
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["options"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness="MineRL BASALT", source_agent=name) for name in subjects.subject_key]
        key_columns = ["task", "seed", "metric", "question_index", "worker_id", "opponent_episode", "position"]
        observations["item_key"] = observations[key_columns].astype(str).agg(":".join, axis=1)
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(task=row.task, world_seed=row.seed,
            environment=parameters["environments"][row.task],
            task_goals=parameters["task_goals"][row.task]), ensure_ascii=False, sort_keys=True)
            for row in items.itertuples()]
        items["features"] = [dict(task=row.task, world_seed=row.seed) for row in items.itertuples()]
        items["scale"] = items.metric.where(items.metric.ne("direct"), "direct_positive")
        items.loc[items.metric.eq("direct") & items.question.isin(grading["negative_questions"]), "scale"] = "direct_negative"
        items["grading_criterion"] = [dict(rule=json.dumps(dict(metric=row.metric, question=row.question,
            interpretation=grading["rules"][row.metric]), ensure_ascii=False, sort_keys=True),
            response_scale=grading["scales"][row.scale]) for row in items.itertuples()]
        items["verifier"] = [Judge(judge="BEDD anonymous worker " + row.worker_id, judged_by="human",
            spec=json.dumps(dict(protocol=grading["protocol"], worker_id=row.worker_id,
                player_position=row.position, reference_agent=row.opponent, reference_episode=row.opponent_episode,
                reference_video=dict(archive=paths["videos"], member=row.reference_member,
                    byte_size=row.reference_byte_size, crc32=row.reference_crc32)), sort_keys=True))
            for row in items.itertuples()]

        # 5. Preserve complete original ratings, justifications and video associations.
        observations["response_key"] = observations[["source_row", "position", "metric", "question_index"]].astype(str).agg(":".join, axis=1)
        observations["test_condition"] = "episode=" + observations.episode_hash + ";annotation=" + observations.hash
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_archive=paths["judgments"], source_member=paths["answers_member"],
            source_row=row.source_row, record=row.record, player_position=row.position,
            metric=row.metric, question_index=row.question_index,
            model_output_video=dict(archive=paths["videos"], member=row.member,
                byte_size=row.byte_size, crc32=row.crc32)), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BEDD(__file__).main_from_args()
