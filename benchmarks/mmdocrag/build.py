"""Build MMDocRAG from the provider's pinned question, judge and answer files.

Source facts live in metadata.yaml; curation_record.md documents the legacy
measurement policy, current-schema identity migration and known coverage gaps.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild

# The five LLM-judge quality dimensions. Each is graded 0-5 by the judge
# (prompt_bank/evaluation_answer.txt), 0 being "completely fails to meet the
# requirement" — an observed grade, not a missing value.
JUDGE_DIMS = (
    "Fluency",
    "Citation Quality",
    "Text-Image Coherence",
    "Reasoning Logic",
    "Factuality",
)
JUDGE_MAX = 5.0

# The judge returns free-form JSON keys: the example format in its own system
# prompt is written `{' Fluency': score, '  Citation Quality': score, ...}`, so
# a minority of rows come back space-padded, quote-wrapped or de-spaced. Match
# on alphanumerics only so those rows are read as the fully judged rows they
# are, instead of being silently zero-filled.
_KEY_NOISE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_dim_key(key: str) -> str:
    return _KEY_NOISE_RE.sub("", str(key).lower())


_DIM_BY_KEY = {_normalize_dim_key(d): d for d in JUDGE_DIMS}


def answer_quality(judge_output: dict[str, object]) -> float | None:
    """Normalized answer-quality score for one judged answer, or None.

    Returns the mean of the five judge dimensions divided by `JUDGE_MAX`, i.e.
    a value in [0, 1]. Returns None — unobserved — when the judge output does
    not carry all five dimensions as in-range numbers, which happens when the
    payload came back empty (`{}`) or a key is unrecoverable ("Aspect",
    "Text-Image Cohesion"). Two alternatives are deliberately not taken:
    treating a missing dimension as 0 (upstream's `eval_all.py` does this;
    it invents a grade the judge never assigned and pushes the cell below the
    worst genuinely-bad answer) and averaging over only the dimensions present
    (the denominator, and so the meaning of the score, would then vary per
    cell, and the dimensions are not interchangeable — Text-Image Coherence in
    particular is not distributed like Fluency).
    """
    scores: dict[str, float] = {}
    for key, value in judge_output.items():
        dim = _DIM_BY_KEY.get(_normalize_dim_key(key))
        if dim is None or dim in scores:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if not 0.0 <= value <= JUDGE_MAX:
            continue
        scores[dim] = value
    if len(scores) < len(JUDGE_DIMS):
        return None
    return sum(scores.values()) / len(JUDGE_DIMS) / JUDGE_MAX


def parse_eval_filename(name: str) -> tuple[str, str, str] | None:
    """Parse a `response/evaluation/` filename into (model, mode, quotes).

    Handles both the common `{model}_{mode}_quotes{N}_llm-judge.jsonl` form and
    the legacy `{model}_{mode}_response_quotes{N}.jsonl[_evaluation.jsonl]` form
    — the legacy names carry two suffixes, so they are stripped repeatedly.
    Returns None for files that don't match.
    """
    stem = name
    while True:
        for suffix in ("_evaluation.jsonl", "_llm-judge.jsonl", ".jsonl"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            break
    # strip a trailing _response between mode and quotes in legacy names
    m = re.match(
        r"^(?P<model>.+?)_(?P<mode>pure-text|multimodal)(?:_response)?_quotes(?P<n>\d+)$",
        stem,
    )
    if not m:
        return None
    return m.group("model"), m.group("mode"), m.group("n")


class MMDocRAG(BenchmarkBuild):
    """Preserve the released attempt cells, including missing judge grades."""

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _load_gold(self) -> dict[int, tuple[str, str | None]]:
        """Prefer the 20-quote gold record, as in the accepted legacy build."""
        gold: dict[int, tuple[str, str | None]] = {}
        downloads = self.source_manifest["downloads"]
        for source_name in self.archive_layout["gold_sources"]:
            for record in self._read_jsonl(
                self.raw_dir / downloads[source_name]["file"]
            ):
                question_id = record.get("q_id")
                if question_id is None or question_id in gold:
                    continue
                answer = record.get("answer_short")
                if isinstance(answer, list):
                    answer = ", ".join(str(part) for part in answer)
                gold[question_id] = (record.get("question") or "", answer)

        expected_ids = self.expectations["question_ids"]
        if set(gold) != set(range(expected_ids["first"], expected_ids["last"] + 1)):
            raise ValueError("Pinned MMDocRAG gold question IDs do not match metadata")
        return gold

    def _load_traces(
        self,
        model: str,
        mode: str,
        quotes: str,
    ) -> dict[int, object]:
        """Use the first exact-case filename; last duplicate q_id wins."""
        trace_directory = self.archive_layout["trace_directory"]
        candidates = (
            f"{trace_directory}/{model}_{mode}_quotes{quotes}_response.jsonl",
            f"{trace_directory}/{model}_{mode}_response_quotes{quotes}.jsonl",
        )
        for relative_file in candidates:
            if relative_file in self._manifest_files:
                break
        else:
            return {}
        traces: dict[int, object] = {}
        for record in self._read_jsonl(self.raw_dir / relative_file):
            question_id = record.get("q_id")
            if question_id is not None:
                traces[question_id] = record.get("response")
        return traces

    def build_subject_item_response_rows(self) -> None:
        gold = self._load_gold()
        self._manifest_files = {
            source["file"] for source in self.source_manifest["downloads"].values()
        }
        evaluation_directory = self.archive_layout["evaluation_directory"]
        evaluation_paths = sorted(
            self.raw_dir / relative_file
            for relative_file in self._manifest_files
            if Path(relative_file).parent.as_posix() == evaluation_directory
        )
        item_ids: dict[int, str] = {}
        subject_ids: dict[str, str] = {}
        for evaluation_path in evaluation_paths:
            parsed = parse_eval_filename(evaluation_path.name)
            if parsed is None:
                raise ValueError(
                    f"Unrecognized pinned evaluation file: {evaluation_path}"
                )
            model, mode, quotes = parsed
            # Legacy registration coalesced five case-only InternVL aliases.
            # Resolve the two first-seen unmapped spellings before registration:
            # the current contract freezes metadata when the ID is derived.
            subject_key = unicodedata.normalize("NFC", model).strip().lower()
            if subject_key not in subject_ids:
                registry_label = self.archive_layout["subject_aliases"].get(
                    model, model
                )
                # Preserve the released variant without guessing an effort
                # level or treating a no-think run as a different base model.
                features = self.archive_layout["subject_features"].get(model)
                subject_ids[subject_key] = self.add_subject(
                    registry_label, features=features
                )
            traces = self._load_traces(model, mode, quotes)

            for record in self._read_jsonl(evaluation_path):
                question_id = record.get("q_id")
                judge_output = record.get("response")
                if question_id is None or not isinstance(judge_output, dict):
                    continue
                if question_id not in item_ids:
                    question, answer = gold[question_id]
                    item_ids[question_id] = self.add_item(
                        raw_item_id=f"q_id::{question_id}",
                        content=question,
                        reference_answer=answer,
                    )
                trace_text = traces.get(question_id)
                self.add_response(
                    subject_id=subject_ids[subject_key],
                    item_id=item_ids[question_id],
                    trial=1,
                    test_condition=f"{mode}/quotes{quotes}",
                    interactors=None,
                    response=answer_quality(judge_output),
                    reference_answer=None,
                    trace=(
                        trace_text
                        if isinstance(trace_text, str) and trace_text
                        else None
                    ),
                )


if __name__ == "__main__":
    MMDocRAG(__file__).main()
