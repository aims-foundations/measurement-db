#!/usr/bin/env python3
"""Link ProofAug's complete cumulative success set to the curated miniF2F-test bank."""

import json
import sys
from pathlib import Path
import tarfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ProofAug(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("benchmark")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the full item bank and only the problem identifiers from the success archive.
        source = self.raw_dir / "ProofAug/isabelle_src"
        bank = pd.read_json(source / "datasets/minif2f-test-curated.jsonl", lines=True)
        with tarfile.open(source / "cum.tar.gz") as archive:
            with archive.extractfile("output/cum_result.jsonl") as handle:
                solved = pd.read_json(handle, lines=True)[["problem_name"]].drop_duplicates()
        if not solved.problem_name.isin(bank.problem_name).all():
            raise ValueError("ProofAug success records refer to unknown curated problems")

        # 2. The release documents a complete cumulative campaign, not isolated attempts.
        outcomes = bank.merge(solved.assign(response=1.), on="problem_name", how="left", validate="one_to_one")
        outcomes["response"] = outcomes.response.fillna(0.)
        subject = self.build_parameters["subject"]
        subjects = pd.DataFrame({"subject_key": [0], "raw_label": [subject["model"]],
                                 "features": [self.build_parameters["subject_features"]]})

        # 3. Describe the proof-checking criterion without copying withheld successful proofs.
        spec = json.dumps(self.grading["verifiers"]["isabelle"], sort_keys=True)
        text = self.build_parameters["item_text"]
        items = bank.assign(
            item_key=bank.problem_name, raw_item_id=bank.problem_name,
            content=text["instruction"] + "\n\n" + text["informal"] + bank.xi.str.strip()
                    + "\n\n" + text["formal"] + bank.xf.str.strip(),
            grading_criterion=[{"rule": self.grading["rule"]}] * len(bank),
            verifier=ExactMatcher(spec=spec),
        )
        responses = outcomes.assign(response_key=outcomes.index, subject_key=0, item_key=outcomes.problem_name,
                                    test_condition=self.build_parameters["evaluation"]["condition"])
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
        }


if __name__ == "__main__":
    ProofAug(__file__).main_from_args()
