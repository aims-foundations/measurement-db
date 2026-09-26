"""Tabulate the released Katakomba final-checkpoint episode outcomes."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Katakomba(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("experiments", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read each policy configuration beside its last recorded score array.
        policies, episodes = [], []
        for path in sorted((self.raw_dir / parameters["paths"]["experiments"]).glob("*/*_normalized_scores.npy")):
            run = path.parent.name
            step = int(path.name.split("_")[0])
            configuration = yaml.safe_load((path.parent / "config.yaml").read_text())
            configuration = {key:value["value"] for key,value in configuration.items()
                if isinstance(value, dict) and "value" in value}
            code = path.parent / configuration["_wandb"]["code_path"]
            policies.append(dict(subject_key=run, checkpoint_step=step, configuration=configuration,
                code_sha256=hashlib.sha256(code.read_bytes()).hexdigest()))
            frame = pd.DataFrame({metric:np.load(path.parent / f"{step}_{metric}.npy", allow_pickle=False)
                for metric in ["normalized_scores", "returns", "depths"]})
            episodes.append(frame.assign(subject_key=run, source_file=str(path.relative_to(self.raw_dir)),
                source_position=frame.index, trial=frame.index + 1))
        policies = pd.DataFrame(policies)
        if policies.subject_key.duplicated().any():
            raise ValueError("Expected one selected checkpoint per trained policy")
        configurations = pd.json_normalize(policies.configuration, max_level=0)
        policies = pd.concat([policies, configurations[["name", "character", "train_seed", "eval_seed", "eval_processes"]]], axis=1)
        observations = pd.concat(episodes, ignore_index=True).merge(policies, on="subject_key", validate="many_to_one")
        references = observations.character.map(parameters["normalizers"]).astype(float)
        if not np.array_equal(observations.normalized_scores.to_numpy(), (observations.returns / references).to_numpy()):
            raise ValueError("Published episode ratios disagree with the source scoring rule")

        # 2. Separate independently trained policies, retaining their actual seeds.
        subjects = policies[["subject_key"]].copy()
        subjects["raw_label"] = "Katakomba " + policies.name + " step " + policies.checkpoint_step.astype(str)
        subjects["features"] = [dict(harness="Katakomba", training_run=row.subject_key,
            algorithm=parameters["algorithms"][row.name.split("-")[0]], training_character=row.character,
            training_seed=int(row.train_seed), checkpoint_step=int(row.checkpoint_step),
            code_sha256=row.code_sha256, training_configuration=json.dumps(
                {key:value for key,value in row.configuration.items() if key != "_wandb"}, sort_keys=True))
            for row in policies.itertuples()]

        # 3. Character configurations define tasks; episode slots are repeated trials.
        items = policies[["character"]].drop_duplicates().rename(columns={"character":"item_key"})
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(character=character, **parameters["task"]), sort_keys=True)
            for character in items.item_key]
        items["features"] = [parameters["item_features"]] * len(items)
        items["grading_criterion"] = [dict(rule=json.dumps(dict(rule=self.grading["rule"],
            autoascend_mean=float(parameters["normalizers"][character])), sort_keys=True)) for character in items.item_key]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["normalized_return"], sort_keys=True))] * len(items)

        # 4. Preserve every episode's score and auxiliary outcomes without clipping.
        responses = observations[["subject_key", "character", "trial", "normalized_scores"]].rename(
            columns={"character":"item_key", "normalized_scores":"response"})
        responses["response_key"] = observations.subject_key + ":" + observations.source_position.astype(str)
        responses["test_condition"] = "eval_seed=" + observations.eval_seed.astype(str) + ";eval_processes=" + observations.eval_processes.astype(str)
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(kind="episode_outcome_record", source_file=row.source_file,
            source_position=int(row.source_position), checkpoint_step=int(row.checkpoint_step),
            normalized_score=row.normalized_scores, episode_return=row.returns, depth=row.depths), allow_nan=False)
            for row in observations.itertuples()]
        return {"subjects":subjects, "items":items, "responses":responses, "traces":traces}


if __name__ == "__main__":
    Katakomba(__file__).main_from_args()
