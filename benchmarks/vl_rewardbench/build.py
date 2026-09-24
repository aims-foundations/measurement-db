#!/usr/bin/env python3
"""Join released VL-RewardBench judgments to the matching public image/question bank."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class VLRewardBenchBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("items", "predictions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the image/question bank and normalize both released JSONL layouts.
        bank = pd.read_parquet(self.raw_dir / "items.parquet")
        bank = bank.assign(bank_key=bank.index, candidates=bank.response.map(tuple),
                           reference_ranking=bank.human_ranking.map(tuple))
        model_names = self.build_parameters["models"]
        frames = []
        with ZipFile(self.raw_dir / "inference_results.zip") as archive:
            for filename in model_names:
                with archive.open("infer_results/" + filename) as handle:
                    records = pd.read_json(handle, lines=True)
                # Some releases wrap each record in {"0": record}; nulls are failed requests.
                if len(records.columns) == 1 and str(records.columns[0]) == "0":
                    records = pd.json_normalize(records.iloc[:, 0].dropna().tolist(), max_level=0)
                records = records.dropna(subset=["id"])
                details = pd.json_normalize(records.meta.tolist()).reindex(columns=[
                    "flag_status", "filter_prompt", "filter_choice", "random_number"
                ]).set_axis(records.index)
                frames.append(records.drop(columns="meta").join(details).assign(source_file=filename))
        attempts = pd.concat(frames, ignore_index=True)
        attempts = attempts.assign(candidates=attempts.response.map(tuple),
                                   reference_ranking=attempts.ranking.map(tuple))

        # 2. Match the complete question, candidates, and reference ranking, not just ID.
        # The inference archive predates edits to the published bank. Other versions
        # remain in the captured archive rather than being attached to different items.
        join = ["id", "query", "candidates", "reference_ranking"]
        observations = attempts.merge(bank[join + ["bank_key"]], on=join, how="inner",
                                      sort=False, validate="many_to_one")
        allowed = observations.flag_status.isin(["agree", "reject", "doesntMatch"])
        if not allowed.all() or observations.filter_prompt.isna().any():
            raise ValueError("VL-RewardBench judgments need a known verdict and captured prompt")
        if not observations.random_number.isin([0, 1]).all():
            raise ValueError("Unknown candidate presentation order")

        # 3. Keep exact presented prompts and image bytes, including both candidate orders.
        items = observations[["bank_key", "filter_prompt", "random_number"]].drop_duplicates()
        items = items.merge(bank[["bank_key", "id", "image", "candidates", "reference_ranking"]],
                            on="bank_key", how="left", validate="many_to_one")
        preferred = [answers[ranks.index(0)] for answers, ranks in
                     zip(items.candidates, items.reference_ranking)]
        verifier_spec = json.dumps(self.grading["verifiers"]["released_preference"], sort_keys=True)
        items = items.assign(
            item_key=range(len(items)), raw_item_id=items.id.astype(str), content=items.filter_prompt,
            grading_criterion=[{"reference_answer": answer, "rule": self.grading["rule"]}
                               for answer in preferred],
            verifier=ExactMatcher(spec=verifier_spec),
            attachments=[[
                {"data": picture["bytes"], "path": f"images/{index}", "role": "image",
                 "media_type": "image/png" if picture["bytes"].startswith(b"\x89PNG") else "image/jpeg"}
            ] for index, picture in zip(items.bank_key, items.image)],
        )

        # 4. Link model judgments to items; an unparsed choice retains its trace and null grade.
        subjects = pd.DataFrame({"subject_key": list(model_names), "raw_label": list(model_names.values())})
        responses = observations.merge(
            items[["bank_key", "filter_prompt", "random_number", "item_key"]],
            on=["bank_key", "filter_prompt", "random_number"], how="left", validate="many_to_one"
        ).rename(columns={"source_file": "subject_key", "filter_choice": "trace"})
        responses = responses.assign(response_key=responses.index,
                                     response=responses.flag_status.map({"agree": 1., "reject": 0.}))
        traces = responses.loc[responses.trace.map(lambda value: isinstance(value, str)), ["response_key", "trace"]]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    VLRewardBenchBuild(__file__).main_from_args()
