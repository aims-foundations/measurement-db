#!/usr/bin/env python3
"""Translate pinned MathArena releases; see metadata.yaml and curation_record.md."""

import base64
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


def nonempty_text(value: object) -> str | None:
    """Keep released text, including whitespace, without legacy truncation."""
    return value if isinstance(value, str) and value.strip() else None


def grading_details(value: object) -> list[dict]:
    """Decode both released rubric encodings; malformed rubrics fail loudly."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(c, dict) for c in value):
        raise ValueError("grading_details must be a JSON or native list of criteria")
    return value


def criterion_fraction(criterion: Mapping) -> float | None:
    """Retain the legacy points/max_points definition, skips, and clipping."""
    try:
        points = float(criterion.get("points"))
        maximum = float(criterion.get("max_points"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(points) or not math.isfinite(maximum) or maximum == 0:
        return None
    return max(0.0, min(1.0, points / maximum))


def prompt_components(record: Mapping) -> tuple[str, list[dict], str]:
    """Separate released prompt text and inline image bytes, never answer text."""
    prompt = nonempty_text(record.get("user_message"))
    prompt_source = "user_message"
    if prompt is None:
        prompt = nonempty_text(record.get("problem"))
        prompt_source = "problem"
    if prompt is None:
        raise ValueError("source record has no released prompt")
    parts = None
    if prompt.startswith("["):
        try:
            parts = json.loads(prompt)
        except json.JSONDecodeError:
            pass  # A mathematical statement may itself start with '['.
    images = []
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if part["type"] in ("text", "input_text"):
                if images:
                    raise ValueError("unexpected interleaved image/text prompt")
                texts.append(part["text"])
            elif part["type"] in ("image_url", "input_image"):
                image_url = part["image_url"]
                detail = part.get("detail")
                if isinstance(image_url, dict):
                    detail = image_url.get("detail")
                    image_url = image_url["url"]
                match = re.fullmatch(
                    r"data:(image/[a-z0-9.+-]+);base64,(.+)", image_url
                )
                if match is None:
                    raise ValueError("image prompt must contain released inline bytes")
                images.append(
                    {
                        "data": base64.b64decode(match[2], validate=True),
                        "media_type": match[1],
                        "detail": detail,
                    }
                )
            elif part["type"] == "image":
                source = part["source"]
                if source["type"] != "base64":
                    raise ValueError("image source must contain released base64 bytes")
                images.append(
                    {
                        "data": base64.b64decode(source["data"], validate=True),
                        "media_type": source["media_type"],
                        "detail": None,
                    }
                )
            else:
                raise ValueError(f"unsupported prompt part: {part['type']!r}")
        if len(texts) != 1:
            raise ValueError("expected one released textual prompt component")
        prompt = texts[0]
    if record.get("image") is not None and not images:
        raise ValueError("released image was not recovered from the delivered prompt")
    return prompt, images, prompt_source


def subject_features(model_name: str, model_config: str) -> dict[str, str]:
    """Retain configuration identity and only explicitly labelled effort levels."""
    features = {"source_model_name": model_name, "model_config": model_config}
    effort = re.search(r"\((low|medium|high|xhigh|max)\)", model_name, re.IGNORECASE)
    if effort:
        features["reasoning_effort"] = effort[1].lower()
    # A config path is evidence of a configuration, not its unreleased contents.
    # In particular, do not infer a harness version, token budget, or access date.
    return features


class MathArenaBuild(BenchmarkBuild):
    """One observation per released final verdict or usable rubric criterion."""

    def source_records(self, competition: Mapping) -> Iterator[tuple[str, int, dict]]:
        """Read bounded Arrow batches without materializing unused message logs."""
        for shard in competition["shards"]:
            source = self.source_manifest["downloads"][shard]
            parquet = pq.ParquetFile(self.raw_dir / source["file"])
            columns = [
                column
                for column in parquet.schema_arrow.names
                if column
                in {
                    "problem_idx",
                    "problem",
                    "user_message",
                    "image",
                    "model_name",
                    "model_config",
                    "idx_answer",
                    "correct",
                    "gold_answer",
                    "answer",
                    "parsed_answer",
                }
                or column.startswith("grading_details_judge_")
            ]
            row_number = 0
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                for record in batch.to_pylist():
                    yield shard, row_number, record
                    row_number += 1

    def _attachments(self, images: list[dict]) -> list[dict]:
        attachments = []
        for ordinal, image in enumerate(images, 1):
            payload = image["data"]
            digest = hashlib.sha256(payload).hexdigest()
            source_path = self.raw_dir / "decoded_images" / digest
            if not source_path.exists() or source_path.read_bytes() != payload:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(payload)
            attachments.append(
                {
                    "source_path": source_path,
                    "path": f"image-{ordinal}.{image['media_type'].split('/')[-1]}",
                    "media_type": image["media_type"],
                    "role": "input_image",
                }
            )
        return attachments

    def build_subject_item_response_rows(self) -> None:
        subjects: dict[tuple[str, str], str] = {}
        items: dict[tuple, str] = {}
        for competition_name, competition in self.source_manifest[
            "competitions"
        ].items():
            print(f"[matharena] translating {competition_name}", flush=True)
            for shard, row_number, record in self.source_records(competition):
                model_name = record["model_name"]
                model_config = record["model_config"]
                subject_key = (model_name, model_config)
                if subject_key not in subjects:
                    subjects[subject_key] = self.add_subject(
                        model_name,
                        features=subject_features(model_name, model_config),
                    )
                problem_id = str(record["problem_idx"])
                trial = int(record["idx_answer"]) + 1
                content, images, prompt_source = prompt_components(record)
                reference = record.get("gold_answer")
                if reference is not None:
                    reference = str(reference)
                trace = nonempty_text(record.get("answer")) or nonempty_text(
                    record.get("parsed_answer")
                )
                grades = []
                if competition["kind"] == "final_answer":
                    if record["correct"] is None:
                        continue
                    if not isinstance(record["correct"], bool):
                        raise ValueError(
                            "final-answer correctness must be a released boolean"
                        )
                    grades.append((None, None, None, float(record["correct"])))
                else:
                    judge_columns = sorted(
                        key
                        for key in record
                        if key.startswith("grading_details_judge_")
                    )
                    for column in judge_columns:
                        judge_slot = int(column.rsplit("_", 1)[1])
                        for criterion_index, criterion in enumerate(
                            grading_details(record[column])
                        ):
                            fraction = criterion_fraction(criterion)
                            if fraction is not None:
                                grades.append(
                                    (judge_slot, criterion_index, criterion, fraction)
                                )
                for grade_index, (
                    judge_slot,
                    criterion_index,
                    criterion,
                    value,
                ) in enumerate(grades):
                    if criterion is None:
                        grading_criterion = {
                            "reference_answer": reference,
                            "rule": "The provider's parsed final answer matches gold_answer.",
                            "response_scale": {"kind": "discrete", "values": [0, 1]},
                        }
                        verifier = ExactMatcher(
                            spec=(
                                "Import the provider's released correct boolean as 0 or 1; "
                                "the provider compares its parsed final answer with gold_answer. "
                                "Do not reparse or regrade the solution."
                            )
                        )
                        # Some releases spell the same problem's gold answer
                        # differently across records. Preserve each instrument
                        # rather than letting first-registration wins erase it.
                        verifier_features = {
                            "reference_answer_sha256": hashlib.sha256(
                                json.dumps(reference, ensure_ascii=False).encode()
                            ).hexdigest(),
                        }
                    else:
                        rubric = {
                            key: criterion.get(key)
                            for key in ("title", "grading_scheme_desc", "max_points")
                        }
                        rubric_json = json.dumps(
                            rubric, sort_keys=True, ensure_ascii=False
                        )
                        grading_criterion = {
                            "reference_answer": reference, "rule": rubric_json,
                            "response_scale": {"kind": "interval", "min": 0, "max": 1},
                        }
                        verifier = Judge(
                            spec="Provider's human criterion grading; normalize awarded points by max_points.",
                            judge="human",
                            judged_by="human",
                        )
                        # Preserve the recorded judge slot and criterion attribution.
                        verifier_features = {
                            "judge_slot": judge_slot,
                            "criterion_index": criterion_index,
                            "rubric_sha256": hashlib.sha256(
                                rubric_json.encode()
                            ).hexdigest(),
                        }
                    image_keys = tuple(
                        (hashlib.sha256(i["data"]).hexdigest(), i["detail"])
                        for i in images
                    )
                    item_key = (
                        competition_name,
                        problem_id,
                        content,
                        reference,
                        image_keys,
                        verifier,
                        judge_slot,
                        criterion_index,
                    )
                    if item_key not in items:
                        features = {
                            "competition": competition_name,
                            "problem_idx": problem_id,
                            "prompt_source": prompt_source,
                        }
                        if images:
                            features["image_detail"] = json.dumps(
                                [i["detail"] for i in images]
                            )
                        items[item_key] = self.add_item(
                            raw_item_id=f"{competition_name}::{problem_id}",
                            content=content,
                            attachments=self._attachments(images),
                            grading_criterion=grading_criterion,
                            verifier=verifier,
                            verifier_features=verifier_features,
                            features=features,
                        )
                    self.add_response(
                        subject_id=subjects[subject_key],
                        item_id=items[item_key],
                        trial=trial,
                        test_condition=None,
                        interactors=None,
                        response=value,
                        trace=trace if grade_index == 0 else None,
                    )


if __name__ == "__main__":
    MathArenaBuild(__file__).main()
