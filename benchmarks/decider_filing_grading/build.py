#!/usr/bin/env python3
"""Build Decider Filing Grading from the AITrailblazer provider release.

See ``metadata.yaml`` for pinned sources and ``record_curation.md`` for the
evidence behind the ingestion and grading choices.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from build_base import BenchmarkBuild, Judge  # noqa: E402

VERDICT_CODES = {"SIGNAL": 1.0, "NO SIGNAL": 2.0, "ABSTAIN": 3.0}

# The four grading configurations that produced the released grades. Access
# dates are at month precision, as recorded for the October 2026 round.
SUBJECTS = {
    "pass_c": {
        "raw_label": "Muse Spark 1.3 (Decider Pass C)",
        "features": {
            "harness": "Muse",
            "reasoning_effort": "Standard",
            "grading_pass": "C",
        },
        "access_date": "2026-10",
    },
    "pass_r": {
        "raw_label": "Muse Spark 1.3 (Decider Pass R)",
        "features": {
            "harness": "Muse",
            "reasoning_effort": "Standard",
            "grading_pass": "R",
        },
        "access_date": "2026-10",
    },
    "iris_remediation": {
        "raw_label": "Muse Spark 1.3 (Decider Iris Remediation)",
        "features": {
            "harness": "Muse",
            "reasoning_effort": "Standard",
            "grading_pass": "remediation",
        },
        "access_date": "2026-10",
    },
    "human_lock": {
        "raw_label": "Human Lock (Decider corpus owner)",
        "features": None,
        "access_date": "2026-10",
    },
}

VERIFIER = Judge(
    judge="Decider locked-label protocol",
    judged_by="human",
    spec=(
        "The item's reference verdict is the locked final label of the "
        "AITrailblazer Decider corpus: two independent blind grading passes "
        "under the rubric (Pass C and Pass R), splits and abstentions "
        "resolved by a fresh adjudicator session shown the two verdicts "
        "anonymized and shuffled, blind remediation re-grades for items "
        "whose packets were rebuilt at complete-submission scope, and a "
        "final human lock by the corpus owner (v1 locked 2026-10-16; v4 "
        "amendment lock 2026-10-24)."
    ),
)


def _filing_descriptor(basis: str) -> str:
    """First sentence of the locked grade's basis: the form/filer facts."""
    first = basis.split(". ")[0].strip()
    if first and not first.endswith("."):
        first += "."
    return first


class DeciderFilingGradingBuild(BenchmarkBuild):
    """Translate the provider release into measurement-db response rows."""

    def download(self):
        override = os.environ.get("DECIDER_RELEASE_PATH")
        if not override:
            return self.fetch_sources("release")
        # Local proof route: accept a local copy of the pinned release only
        # if its bytes match the size and SHA-256 pinned in metadata.yaml.
        pin = next(
            source
            for source in self.source_manifest["upstream"]
            if source["name"] == "release"
        )
        payload = Path(override).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != pin["size"] or digest != pin["sha256"]:
            raise ValueError(
                f"DECIDER_RELEASE_PATH file does not match the pinned release: "
                f"size {len(payload)} != {pin['size']} or sha256 {digest} != "
                f"{pin['sha256']}"
            )
        target = self.raw_dir / pin["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self._source_artifacts = [
            {
                "name": pin["name"],
                "url": pin["url"],
                "file": pin["file"],
                "size": pin["size"],
                "sha256": pin["sha256"],
            }
        ]
        self.source_files = (pin["file"],)
        return [pin["url"]]

    def build_subject_item_response_rows(self) -> None:
        release = json.loads(
            (self.raw_dir / "decider-filing-grading-release.json").read_text(
                encoding="utf-8"
            )
        )
        rubric = release["rubric"]["text"]

        subject_ids = {
            key: self.add_subject(
                spec["raw_label"],
                features=spec["features"],
                access_date=spec["access_date"],
            )
            for key, spec in SUBJECTS.items()
        }

        for item_record in release["items"]:
            accession = item_record["accession"]
            grades = item_record["grades"]
            descriptor = _filing_descriptor(grades["human_lock"]["basis"])
            item_id = self.add_item(
                raw_item_id=accession,
                content=f"SEC filing {accession}: {descriptor}",
                grading_criterion={"rule": rubric},
                verifier=VERIFIER,
            )
            for key, subject_id in subject_ids.items():
                grade = grades.get(key)
                if grade is None:
                    continue
                self.add_response(
                    subject_id=subject_id,
                    item_id=item_id,
                    trial=1,
                    test_condition=None,
                    interactors=None,
                    response=VERDICT_CODES[grade["verdict"]],
                    trace=grade["basis"],
                )


if __name__ == "__main__":
    DeciderFilingGradingBuild(__file__).main_from_args()
