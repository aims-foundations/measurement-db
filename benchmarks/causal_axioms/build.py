#!/usr/bin/env python3
"""Reshape the released d-separation predictions into model/item observations."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CausalAxioms(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("predictions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the JSONL prediction files; the directory records graph length.
        models = self.build_parameters["models"]
        frames = [pd.read_json(path, lines=True, dtype=False).assign(
            subject_key=path.name, test_condition="length=" + path.parent.name.removeprefix("length_")
        ) for path in sorted(self.raw_dir.glob("length_*/*.jsonl"))]
        data = pd.concat(frames, ignore_index=True)
        data = data.loc[data.prompt.map(lambda value: isinstance(value, str) and bool(value.strip()))].copy()
        data["content"] = data.prompt.str.strip()
        data["gold"] = data.completion.fillna("").astype(str).str.strip()
        data = data.loc[data.gold.isin(["Yes", "No"]) & data.prediction.notna()].copy()

        # 2. Form unique prompt/reference definitions with deterministic source aliases.
        items = data[["content", "gold"]].drop_duplicates().reset_index(drop=True)
        spec = json.dumps(self.grading["verifiers"]["exact_match"], sort_keys=True)
        items = items.assign(
            item_key=items.index,
            raw_item_id=items.content.map(lambda prompt: "dsep_" + hashlib.sha256(prompt.encode()).hexdigest()[:24]),
            grading_criterion=items.gold.map(lambda answer: {"reference_answer": answer, "rule": self.grading["rule"]}),
            verifier=ExactMatcher(spec=spec),
        )

        # 3. Join item keys and apply the provider's case-insensitive exact match.
        responses = data.merge(items[["content", "gold", "item_key"]], on=["content", "gold"],
                               how="left", sort=False, validate="many_to_one")
        responses = responses.assign(
            response_key=responses.index,
            response=responses.prediction.astype(str).str.strip().str.lower().eq(responses.gold.str.lower()).astype(float),
            trace=responses.prediction.astype(str),
        )
        return {
            "subjects": pd.DataFrame({"subject_key": list(models), "raw_label": list(models.values())}),
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses[["response_key", "trace"]],
        }


if __name__ == "__main__":
    CausalAxioms(__file__).main_from_args()
