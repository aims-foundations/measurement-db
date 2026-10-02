"""Join released detection verdicts to perturbed papers and injected-error rubrics."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class AgenticReviewPerturbation(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read native scores, retained perturbations and perturbed paper text.
        root = self.raw_dir / self.build_parameters["paths"]["root"]
        score_paths = sorted(root.glob("results_*/**/score/llm/*_score.json"))
        scores = pd.json_normalize([json.loads(path.read_text()) for path in score_paths], max_level=0).assign(
            source_file=[str(path.relative_to(root)) for path in score_paths]
        )
        coordinates = scores.source_file.str.extract(
            r"^(?P<tree>results_[^/]+)/(?P<method>[^/]+)/(?P<raw_label>[^/]+)/experimental/"
            r"(?P<run_method>[^/]+)/(?P<paper_id>paper_[^/]+)/score/llm/(?P<stem>[^/]+)_score\.json$"
        )
        if coordinates.isna().any().any() or not coordinates.method.eq(coordinates.run_method).all():
            raise ValueError("An agentic review score has an unrecognized configuration path")
        scores = scores.join(coordinates).assign(
            kept_file=lambda frame: frame.tree + "/" + frame.method + "/perturb/experimental/"
                + frame.paper_id + "/" + frame.stem + "_kept_perturbations.json",
            review_prompt=lambda frame: frame.tree.str.removeprefix("results_"),
        )
        perturbations = []
        for filename in scores.kept_file.drop_duplicates():
            records = json.loads((root / filename).read_text())["perturbations"]
            perturbations.append(pd.DataFrame({"injected_error": records}).assign(kept_file=filename))
        rubrics = pd.concat(perturbations, ignore_index=True).assign(
            perturbation_id=lambda frame: frame.injected_error.map(lambda record: record["perturbation_id"])
        )
        paper_paths = sorted((root / "perturbation_results").glob("*/all/*/experimental/*_recorrupted.md"))
        papers = pd.DataFrame({
            "paper_id": ["paper_" + path.parts[-3] for path in paper_paths],
            "stem": [path.name.removesuffix("_recorrupted.md") for path in paper_paths],
            "content": [path.read_text().strip() for path in paper_paths],
        })

        # 2. Expand each detected/missed list into one binary observation per error.
        observations = scores.melt(
            id_vars=[column for column in scores if column not in {"detected", "missed"}],
            value_vars=["detected", "missed"], var_name="verdict", value_name="perturbation_id",
        ).explode("perturbation_id", ignore_index=True).dropna(subset=["perturbation_id"])
        observations["response"] = observations.verdict.eq("detected").astype(float)
        counts = observations.groupby("source_file").agg(n=("response", "size"), detected=("response", "sum"))
        counts = scores.set_index("source_file")[["n_injected", "n_detected"]].join(counts)
        if (observations.duplicated(["source_file", "perturbation_id"]).any()
                or not counts.n.eq(counts.n_injected).all() or not counts.detected.eq(counts.n_detected).all()):
            raise ValueError("Conflicting, duplicate or incomplete native detection verdicts")

        # 3. Attach the exact rubric and paper; missing or ambiguous matches fail.
        observations = observations.merge(
            rubrics, on=["kept_file", "perturbation_id"], how="left", validate="many_to_one", indicator=True,
        )
        if not observations._merge.eq("both").all():
            raise ValueError("A scored perturbation has no released grading rubric")
        observations = observations.drop(columns="_merge").merge(
            papers, on=["paper_id", "stem"], how="left", validate="many_to_one", indicator=True,
        )
        if not observations._merge.eq("both").all() or not observations.content.fillna("").astype(bool).all():
            raise ValueError("A scored perturbation has no unique released paper text")
        observations = observations.sort_values(["source_file", "perturbation_id"]).assign(
            item_key=lambda frame: frame.paper_id + "/" + frame.stem + "/" + frame.perturbation_id,
            rule=lambda frame: frame.injected_error.map(lambda error: json.dumps({
                "rule": self.grading["rule"], "injected_error": error,
            }, sort_keys=True)),
        )

        # 4. Separate shared paper/error items from model, method and prompt settings.
        items = observations[["item_key", "content", "rule", "paper_id", "perturbation_id"]].drop_duplicates()
        if items.item_key.duplicated().any():
            raise ValueError("A review item has conflicting paper or rubric definitions")
        items = items.assign(
            raw_item_id=items.item_key, grading_criterion=items.rule.map(lambda rule: {"rule": rule}),
            verifier=Judge(spec=self.grading["verifiers"]["detection"]["spec"], judged_by="llm"),
            features=items[["paper_id", "perturbation_id"]].to_dict("records"),
        )
        observations["subject_key"] = observations[["raw_label", "method", "review_prompt"]].apply(
            lambda row: json.dumps(row.tolist()), axis=1
        )
        subjects = observations[["subject_key", "raw_label", "method", "review_prompt"]].drop_duplicates()
        subjects = subjects.assign(features=subjects.rename(columns={"method": "harness"})[
            ["harness", "review_prompt"]
        ].to_dict("records"))
        responses = observations.assign(response_key=observations.source_file + "/" + observations.perturbation_id)
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
        }


if __name__ == "__main__":
    AgenticReviewPerturbation(__file__).main_from_args()
