"""Preserve human segment grades without putting generated answers in the input."""

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class FELM(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("data", "reference")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the released answer table and preserve each complete original line.
        text = (self.raw_dir / parameters["paths"]["records"]).read_text()
        records = pd.read_json(io.StringIO(text), lines=True, dtype=False, convert_dates=False)
        records["native_line"] = [line for line in text.split("\n") if line.strip()]
        records["source_line"] = records.index
        records["parent_response_missing"] = records.response.isna()
        records["index"] = records["index"].astype(str)
        lengths = records.segmented_response.str.len()
        if not records.labels.str.len().eq(lengths).all() or not records.comment.str.len().eq(lengths).all():
            raise ValueError("Segments, factuality labels and comments must align exactly")
        if not records.domain.isin(parameters["domains"]).all() or records["index"].duplicated().any():
            raise ValueError("Unrecognized domain or repeated native record identifier")

        # 2. Expand aligned segment annotations without guessing reference-list alignment.
        attempts = records.explode(["segmented_response", "labels", "comment"], ignore_index=True)
        attempts["segment_index"] = attempts.groupby("source_line", sort=False).cumcount()
        if not attempts.labels.map(type).eq(bool).all():
            raise ValueError("Factuality labels must be explicit original Booleans")
        attempts["item_key"] = attempts["index"] + ":" + attempts.segment_index.astype(str)

        # 3. Keep the prompt as input; select the assessed output segment in the verifier.
        items = attempts[["item_key", "index", "segment_index", "prompt", "domain"]].copy()
        items["raw_item_id"] = items["index"] + "_" + items.segment_index.astype(str)
        items["content"] = items.prompt
        items["features"] = items.domain.map(parameters["domains"]).to_frame("domain").to_dict("records")
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [Judge(judged_by="human", spec=json.dumps(
            dict(segment_index=int(index), **self.grading["verifiers"]["annotation"]), sort_keys=True))
            for index in items.segment_index]
        subjects = pd.DataFrame([dict(subject_key="generator", raw_label=parameters["subject"]["label"],
            features=parameters["subject_features"])])

        # 4. Preserve labels and full parent context, including source-format anomalies.
        attempts["response_key"] = attempts.item_key
        attempts["subject_key"] = "generator"
        attempts["response"] = attempts.labels.astype(float)
        attempts["trace"] = [json.dumps(dict(source_line=int(line), segment_index=int(index),
            segment=segment, parent_record_json=record,
            source_issues=(["parent_response_is_upstream_NaN"] if missing else []) +
                (["empty_segment_has_original_label"] if not segment else [])), ensure_ascii=False, allow_nan=False)
            for line, index, segment, record, missing in zip(attempts.source_line, attempts.segment_index,
                attempts.segmented_response, attempts.native_line, attempts.parent_response_missing)]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response"]],
            "traces": attempts[["response_key", "trace"]],
        }


if __name__ == "__main__":
    FELM(__file__).main_from_args()
