"""Tabulate the released Mind2Web candidate rankings and their complete inputs."""

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class Mind2Web(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, layout = self.build_parameters, self.build_parameters["layout"]

        # 1. Flatten every source task into action steps, preserving its prior actions.
        files = [(str(path.relative_to(self.raw_dir)), path)
                 for path in sorted(self.raw_dir.glob(layout["training_glob"]))]
        with ZipFile(self.raw_dir / layout["test_archive"]) as archive:
            files += [(layout["test_archive"] + "!" + name, name)
                      for name in sorted(archive.namelist()) if name.endswith(".json")]
        frames = []
        for source_file, source in files:
            data = source.read_bytes() if isinstance(source, Path) else subprocess.check_output(
                ["unzip", "-p", "-P", layout["published_test_password"],
                 str(self.raw_dir / layout["test_archive"]), source])
            tasks = pd.json_normalize(json.loads(data), max_level=0).reset_index(names="source_task_position")
            if not tasks.actions.str.len().eq(tasks.action_reprs.str.len()).all():
                raise ValueError("Source action and readable-history lengths disagree")
            actions = tasks.explode("actions", ignore_index=True)
            actions["source_action_position"] = actions.groupby("annotation_id", sort=False).cumcount()
            actions = actions.drop(columns="actions").join(pd.json_normalize(actions.actions, max_level=0))
            actions["source_file"] = source_file
            actions["source_split"] = layout["training_split"] if isinstance(source, Path) else source.split("/")[0]
            actions["query"] = [layout["query_template"].format(task=row.confirmed_task,
                previous_actions="; ".join(row.action_reprs[max(0, row.source_action_position - 3):row.source_action_position]))
                for row in actions.itertuples()]
            frames.append(actions.drop(columns=["raw_html", "action_reprs"]))
            del data, tasks, actions
        actions = pd.concat(frames, ignore_index=True)
        actions["sample_id"] = actions.annotation_id + "_" + actions.action_uid
        if actions.sample_id.duplicated().any():
            raise ValueError("Source action identifiers are not unique")

        # 2. Join cached model scores and ranks to their source actions.
        native = read_native_pickle(self.raw_dir / layout["rankings"])
        if set(native) != {"scores", "ranks"} or set(native["scores"]) != set(native["ranks"]):
            raise ValueError("Cached score and rank action coverage differs")
        rankings = pd.DataFrame({key: pd.Series(value) for key, value in native.items()}).rename_axis("sample_id").reset_index()
        attempts = actions.merge(rankings, on="sample_id", how="outer", validate="one_to_one", indicator=True)
        unmatched = sorted(attempts.loc[attempts._merge.eq("right_only"), "sample_id"])
        if unmatched != sorted(parameters["unmatched_outputs"]):
            raise ValueError("Unreviewed cached rankings have no source action")
        attempts = attempts.loc[attempts._merge.eq("both")].copy()
        attempts["positive_ids"] = attempts.pos_candidates.map(lambda rows: [row["backend_node_id"] for row in rows])
        attempts["candidate_ids"] = [sorted({candidate["backend_node_id"] for candidate in positive + negative})
                                    for positive, negative in zip(attempts.pos_candidates, attempts.neg_candidates)]
        if any(set(row.scores) != set(row.ranks) or set(row.ranks) != set(row.candidate_ids)
               for row in attempts.itertuples()):
            raise ValueError("A cached ranking differs from the task's candidate set")
        attempts["response"] = [int(any(row.ranks[key] < int(layout["recall_k"]) for key in row.positive_ids))
                                for row in attempts.itertuples()]
        attempts["response_key"] = attempts.sample_id
        attempts["subject_key"] = parameters["subject"]["key"]

        # 3. Keep input material separate from the gold targets used for grading.
        subjects = pd.DataFrame([dict(subject_key=parameters["subject"]["key"],
            raw_label=parameters["subject"]["label"], features=parameters["subject_features"])])
        items = attempts[["sample_id", "query", "cleaned_html", "candidate_ids", "positive_ids", "source_split"]].copy()
        items["item_key"] = items["raw_item_id"] = items.sample_id
        items["content"] = [json.dumps(dict(query=row.query, cleaned_html=row.cleaned_html,
            candidate_node_ids=row.candidate_ids), ensure_ascii=False, allow_nan=False) for row in items.itertuples()]
        items["features"] = [dict(**parameters["item_features"], source_split=split) for split in items.source_split]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(sorted(ids)), rule=self.grading["rule"])
                                      for ids in items.positive_ids]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["rank_recall"], sort_keys=True))
                             for _ in range(len(items))]
        attempts["item_key"] = attempts.sample_id

        # 4. Preserve every native score and rank, with exact task/action coordinates.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_split=row.source_split,
            source_task_position=int(row.source_task_position), source_action_position=int(row.source_action_position),
            annotation_id=row.annotation_id, action_uid=row.action_uid, source_query=row.query,
            candidate_scores=row.scores, candidate_ranks=row.ranks, gold_candidate_ids=row.positive_ids,
            source_operation=row.operation, source_positive_candidates=row.pos_candidates,
            source_negative_candidates=row.neg_candidates), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects, "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    Mind2Web(__file__).main_from_args()
