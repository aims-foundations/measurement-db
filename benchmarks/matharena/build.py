#!/usr/bin/env python3
"""Curate MathArena's released final verdicts and proof criteria as linked tables."""

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class MathArenaBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Concatenate the native result shards, retaining their competition and attempt order.
        layout = self.build_parameters["layout"]
        source_columns = ['problem_idx', 'problem', 'user_message', 'image', 'model_name', 'model_config', 'idx_answer', 'correct', 'gold_answer', 'answer', 'parsed_answer']
        frames = []
        for name in sorted(self.source_files):
            path = Path(name)
            if len(path.parts) != 4 or path.parts[0] != "sources" or path.parts[2] != "data" or path.suffix != ".parquet":
                continue
            columns = [column for column in pq.read_schema(self.raw_dir / path).names
                       if column in source_columns or column.startswith(layout["judge_prefix"])]
            frame = pd.read_parquet(self.raw_dir / path, columns=columns, dtype_backend="pyarrow").astype(object)
            frames.append(frame.assign(competition=path.parts[1]))
        attempts = pd.concat(frames, ignore_index=True).astype(object)
        attempts = attempts.reindex(columns=list(dict.fromkeys([*attempts.columns, *source_columns])))
        attempts = attempts.where(attempts.notna(), None).assign(attempt_key=lambda frame: frame.index)
        attempts["problem_idx"] = attempts.problem_idx.astype(str)
        attempts["reference"] = attempts.gold_answer.map(lambda value: str(value) if value is not None else None).astype(object)
        attempts["reference"] = attempts.reference.where(attempts.reference.notna(), None)
        attempts["trial"] = attempts.idx_answer.astype("int64") + 1
        attempts["kind"] = np.where(attempts.competition.isin(self.grading["verifiers"]["proof"]["competitions"]), "proof", "final_answer")

        # 2. Decode each distinct prompt once; image bytes stay in memory, leaving raw/ unchanged.
        text = attempts.user_message.map(lambda value: isinstance(value, str) and bool(value.strip()))
        attempts["prompt_source"] = np.where(text, "user_message", "problem")
        attempts["prompt"] = attempts.user_message.where(text, attempts.problem)
        if not attempts.prompt.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("source record has no released prompt")
        attempts["has_image"] = attempts.image.notna()
        prompt_keys = ["prompt", "prompt_source", "has_image"]
        prompts = attempts[prompt_keys].drop_duplicates().copy()
        decoded = [decode_prompt(prompt, has_image) for prompt, has_image in zip(prompts.prompt, prompts.has_image)]
        prompts = prompts.join(pd.DataFrame(decoded, index=prompts.index))
        attempts = attempts.drop(columns=["image", "user_message", "problem"]).merge(
            prompts, on=prompt_keys, how="left", sort=False, validate="many_to_one")

        # 3. Keep distinct model configurations and only explicitly recorded reasoning effort.
        subjects = attempts[["model_name", "model_config"]].drop_duplicates().copy()
        subjects["subject_key"] = range(len(subjects))
        subjects["raw_label"] = subjects.model_name
        features = subjects.rename(columns={"model_name": "source_model_name"})[["source_model_name", "model_config"]].copy()
        features["reasoning_effort"] = subjects.model_name.str.extract(layout["effort_pattern"], expand=False).str.lower()
        subjects["features"] = features.apply(lambda row: row.dropna().to_dict(), axis=1)
        attempts = attempts.merge(subjects[["model_name", "model_config", "subject_key"]],
                                  on=["model_name", "model_config"], how="left", sort=False, validate="many_to_one")

        # 4. Expand proof rubrics to one row per judge/criterion; retain final-answer booleans directly.
        final = attempts.loc[attempts.kind.eq("final_answer") & attempts.correct.notna(), ["attempt_key", "correct"]].copy()
        if not final.correct.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("final-answer correctness must be a released boolean")
        final = final.assign(response=final.correct.astype(float), judge_slot=None, criterion_index=None, rubric_json=None)
        judge_columns = sorted(column for column in attempts if column.startswith(layout["judge_prefix"]))
        proof = attempts.loc[attempts.kind.eq("proof"), ["attempt_key", *judge_columns]].melt(
            id_vars="attempt_key", value_vars=judge_columns, var_name="judge", value_name="criteria").dropna(subset="criteria")
        proof["criteria"] = proof.criteria.map(lambda value: json.loads(value) if isinstance(value, str)
                                                else value.tolist() if isinstance(value, np.ndarray) else value)
        if not proof.criteria.map(lambda value: isinstance(value, list) and all(isinstance(part, dict) for part in value)).all():
            raise ValueError("grading_details must be a JSON or native list of criteria")
        proof = proof.explode("criteria", ignore_index=True).dropna(subset="criteria")
        proof["judge_slot"] = proof.judge.str.rsplit("_", n=1).str[-1].astype("int64")
        proof["criterion_index"] = proof.groupby(["attempt_key", "judge"], sort=False).cumcount()
        points = pd.json_normalize(proof.criteria.tolist()).reindex(columns=["points", "max_points"]).set_axis(proof.index)
        points = points.apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(points.points) & np.isfinite(points.max_points) & points.max_points.ne(0)
        proof["response"] = (points.points / points.max_points).clip(0, 1).where(valid)
        proof = proof.loc[valid].copy()
        proof["rubric_json"] = proof.criteria.map(lambda value: json.dumps(
            {field: value.get(field) for field in ['title', 'grading_scheme_desc', 'max_points']}, sort_keys=True, ensure_ascii=False))
        grade_columns = ["attempt_key", "response", "judge_slot", "criterion_index", "rubric_json"]
        grades = pd.concat([final[grade_columns], proof[grade_columns]], ignore_index=True)
        grades[["judge_slot", "criterion_index"]] = grades[["judge_slot", "criterion_index"]].astype("Int64")
        grades = grades.sort_values(["attempt_key", "judge_slot", "criterion_index"], kind="stable", na_position="first")
        observations = grades.merge(attempts, on="attempt_key", how="left", sort=False, validate="many_to_one")

        # 5. Distinguish item content, images and grading protocols before assigning local item keys.
        item_columns = ["competition", "problem_idx", "content", "reference", "image_key", "kind", "judge_slot", "criterion_index", "rubric_json"]
        observations["item_key"] = observations.groupby(item_columns, sort=False, dropna=False).ngroup()
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.competition + "::" + items.problem_idx
        item_features = items[["competition", "problem_idx", "prompt_source", "image_detail"]]
        items["features"] = item_features.apply(lambda row: row.dropna().to_dict(), axis=1)
        final_spec, proof_spec = (self.grading["verifiers"][kind] for kind in ["final_answer", "proof"])
        items["verifier"] = ExactMatcher(spec=final_spec["spec"])
        is_proof = items.kind.eq("proof")
        items.loc[is_proof, "verifier"] = Judge(spec=proof_spec["spec"], judge=proof_spec["judge"], judged_by=proof_spec["judged_by"])
        rules = items.rubric_json.where(is_proof, final_spec["rule"])
        scales = items.kind.map({"final_answer": final_spec["response_scale"], "proof": proof_spec["response_scale"]})
        items["grading_criterion"] = [dict(reference_answer=reference, rule=rule, response_scale=scale)
                                     for reference, rule, scale in zip(items.reference, rules, scales)]
        items["verifier_features"] = items.reference.map(lambda value: {
            "reference_answer_sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()})
        items.loc[is_proof, "verifier_features"] = pd.Series([
            {"judge_slot": int(judge), "criterion_index": int(index), "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest()}
            for judge, index, rubric in items.loc[is_proof, ["judge_slot", "criterion_index", "rubric_json"]].itertuples(index=False, name=None)
        ], index=items.index[is_proof], dtype=object)

        # 6. Link each complete attempt trace to its first usable grade, without copying it per criterion.
        responses = observations.assign(response_key=range(len(observations)))
        traces = responses.drop_duplicates("attempt_key").copy()
        has_answer = traces.answer.map(lambda value: isinstance(value, str) and bool(value.strip()))
        traces["trace"] = traces.answer.where(has_answer, traces.parsed_answer)
        traces = traces.loc[traces.trace.map(lambda value: isinstance(value, str) and bool(value.strip()))]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "grading_criterion", "verifier", "verifier_features", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "trial", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


def decode_prompt(prompt: str, has_image: bool) -> dict:
    """Decode native multimodal payloads; this is input parsing, not grading."""
    parts = None
    if prompt.startswith("["):
        try:
            parts = json.loads(prompt)
        except json.JSONDecodeError:
            pass  # A mathematical statement may itself begin with '['.
    images = []
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if part["type"] in ("text", "input_text"):
                if images:
                    raise ValueError("unexpected interleaved image/text prompt")
                texts.append(part["text"])
            elif part["type"] in ("image_url", "input_image"):
                url, detail = part["image_url"], part.get("detail")
                if isinstance(url, dict):
                    url, detail = url["url"], url.get("detail")
                match = re.fullmatch(r"data:(image/[a-z0-9.+-]+);base64,(.+)", url)
                if match is None:
                    raise ValueError("image prompt must contain released inline bytes")
                images.append({"data": base64.b64decode(match[2], validate=True), "media_type": match[1], "detail": detail})
            elif part["type"] == "image":
                source = part["source"]
                if source["type"] != "base64":
                    raise ValueError("image source must contain released base64 bytes")
                images.append({"data": base64.b64decode(source["data"], validate=True), "media_type": source["media_type"], "detail": None})
            else:
                raise ValueError(f"unsupported prompt part: {part['type']!r}")
        if len(texts) != 1:
            raise ValueError("expected one released textual prompt component")
        prompt = texts[0]
    if has_image and not images:
        raise ValueError("released image was not recovered from the delivered prompt")
    return {
        "content": prompt,
        "image_key": json.dumps([(hashlib.sha256(image["data"]).hexdigest(), image["detail"]) for image in images]),
        "image_detail": json.dumps([image["detail"] for image in images]) if images else None,
        "attachments": [{"data": image["data"], "path": f"image-{index}.{image['media_type'].split('/')[-1]}",
                         "media_type": image["media_type"], "role": "input_image"} for index, image in enumerate(images, 1)],
    }


if __name__ == "__main__":
    MathArenaBuild(__file__).main_from_args()
