#!/usr/bin/env python3
"""Build Example Benchmark from its provider-owned per-item release.

See ``metadata.yaml`` for pinned sources and ``record_curation.md`` for the
evidence behind the ingestion and grading choices.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from build_base import BenchmarkBuild, ExactMatcher  # noqa: E402


class ExampleBenchmarkBuild(BenchmarkBuild):
    """Translate the provider release into measurement-db response rows."""

    def build_subject_item_response_rows(self) -> None:
        item_records = json.loads(
            (self.raw_dir / "per_item_results.json").read_text(encoding="utf-8")
        )

        for item_record in item_records:
            item_id = self.add_item(
                raw_item_id=str(item_record["question_id"]),
                content=item_record["question"],
                grading_criterion={"reference_answer": item_record["gold_answer"]},
                verifier=ExactMatcher(spec="response equals gold answer verbatim"),
                # Item identity is complete when it is registered.
                features={"shot": item_record["n_shots"]},
            )
            for model_run in item_record["model_runs"]:
                subject_id = self.add_subject(model_run["model_name"])
                self.add_response(
                    subject_id=subject_id,
                    item_id=item_id,
                    trial=int(model_run.get("trial", 1)),
                    test_condition=None,
                    interactors=None,
                    response=1.0 if model_run["correct"] else 0.0,
                    trace=model_run.get("model_output"),
                )


if __name__ == "__main__":
    ExampleBenchmarkBuild(__file__).main()
