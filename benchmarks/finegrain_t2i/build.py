"""Join FineGRAIN's released images to its original human annotations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class FineGrainT2I(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("dataset", "original", "code", "paper")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]

        # 1. Read current references and the original, genuinely human-labeled release.
        current = pd.read_csv(self.raw_dir / paths["current"], keep_default_na=False)
        current["source_row"] = current.index
        current["native_record"] = current.drop(columns="source_row").to_dict("records")
        original = pd.read_csv(self.raw_dir / paths["original"], keep_default_na=False)
        original["original_record"] = original.to_dict("records")
        original = original.rename(columns={"prompt_id": "original_prompt_id", "human_labels": "human_grade"})
        rows = current.merge(original, on=["model", "prompt_text", "failure_mode"],
            how="left", validate="many_to_one")
        matched = rows.original_record.notna()
        if not rows.loc[matched, "human_labels"].eq(rows.loc[matched, "human_grade"]).all():
            raise ValueError("Current and original human annotations disagree")
        rows["image_file"] = rows.image_filename.mask(rows.image_filename.eq(""), rows.file_name)
        if rows.image_file.isna().any() or rows.image_file.eq("").any():
            raise ValueError("An attempt has no released image reference")
        # Only the original human-label release establishes grades. Later zeros are placeholders.
        rows["response"] = 1 - pd.to_numeric(rows.human_grade.replace("", pd.NA))
        rows["annotation_status"] = "not_in_human_annotation_release"
        rows.loc[matched & rows.response.isna(), "annotation_status"] = "human_grade_missing"
        rows.loc[rows.response.notna(), "annotation_status"] = "human_graded"

        # 2. Collapse repeated references to an identical model/image, retaining all source aliases.
        identity = ["model", "image_file"]
        if rows.groupby(identity)[["prompt_text", "failure_mode", "human_labels"]].nunique().gt(1).any().any():
            raise ValueError("A repeated image reference has conflicting content or labels")
        aliases = rows.groupby(identity, sort=False).agg(source_rows=("source_row", list), native_records=("native_record", list)).reset_index()
        responses = rows.drop_duplicates(identity).drop(columns=["source_row", "native_record"])
        responses = responses.merge(aliases, on=identity, validate="one_to_one")
        responses["response_key"] = responses.image_file
        responses["subject_key"] = responses.model
        responses["item_key"] = responses.prompt_text

        # 3. Join captured image hashes and the authors' failure-mode descriptions.
        outputs = pd.DataFrame(self._source_artifacts)
        outputs = outputs.loc[outputs.file.str.startswith("dataset/images/")].copy()
        if not outputs.hash_kind.eq("sha256").all():
            raise ValueError("Generated images must have verified SHA-256 hashes")
        outputs["image_file"] = outputs.hf_path
        responses = responses.merge(outputs[["image_file", "file", "size", "digest"]], on="image_file", how="left", validate="one_to_one")
        if responses.digest.isna().any():
            raise ValueError("An attempt has no complete captured output image")
        descriptions = pd.read_json(self.raw_dir / paths["descriptions"], orient="index")
        items = responses[["item_key", "prompt_text", "failure_mode"]].drop_duplicates().copy()
        items = items.join(descriptions[["description"]], on="failure_mode", validate="many_to_one")
        if items.description.isna().any() or items.item_key.duplicated().any():
            raise ValueError("A prompt does not have exactly one declared failure mode")

        # 4. Keep the grading protocol in item identity and historical subject settings explicit.
        items["content"] = items.prompt_text
        items["raw_item_id"] = items.prompt_text.map(original.drop_duplicates("prompt_text").set_index("prompt_text").original_prompt_id).astype(str)
        items["features"] = [{"failure_mode": mode} for mode in items.failure_mode]
        items["grading_criterion"] = [dict(rule=self.grading["rule"] + "\nFailure mode: " + mode + ". " + description)
            for mode, description in zip(items.failure_mode, items.description)]
        items["verifier"] = [Judge(judged_by="human", spec=json.dumps(self.grading["verifiers"]["human"], sort_keys=True)) for _ in items.index]
        subjects = responses[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(**self.build_parameters["subject_features"], source_model=model) for model in subjects.subject_key]

        # 5. Preserve native source rows, annotation availability and full generated-image references.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["current"], source_rows=row.source_rows,
            native_records=row.native_records, original_source_file=paths["original"],
            original_record=row.original_record if isinstance(row.original_record, dict) else None,
            annotation_status=row.annotation_status, output=dict(file=row.file, upstream_file=row.image_file, bytes=int(row.size), sha256=row.digest)),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    FineGrainT2I(__file__).main_from_args()
