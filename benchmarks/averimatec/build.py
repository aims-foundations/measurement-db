#!/usr/bin/env python3
"""Curate released AIC CTU development predictions and the original verdict component."""

import ast
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AVerImaTeC(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        layout = self.build_parameters["layout"]

        # 1. Read the released claim table and both saved notebook displays as tables.
        claims = pd.read_json(self.raw_dir / layout["claims"], dtype=False, convert_dates=False)
        claims = claims.rename_axis("id").reset_index()
        records = []
        for variant, source_file in self.build_parameters["exports"].items():
            notebook = json.loads((self.raw_dir / source_file).read_text())
            output = notebook["cells"][int(layout["cell"])]["outputs"][int(layout["output"])]
            # text/plain is a Python literal representation; no notebook code is executed.
            frame = pd.DataFrame(ast.literal_eval("".join(output["data"]["text/plain"])))
            frame["native_record"] = frame.to_dict("records")
            records.append(frame.rename_axis("source_row").reset_index().assign(variant=variant, source_file=source_file))
        exports = pd.concat(records, ignore_index=True)
        copies = exports.pivot(index="id", columns="variant", values="native_record")
        original = pd.DataFrame(copies.original.tolist(), index=copies.index)
        reformatted = pd.DataFrame(copies.reformatted.tolist(), index=copies.index)
        if not original.drop(columns="evidence").equals(reformatted.drop(columns="evidence")):
            raise ValueError("The notebook copies no longer contain identical predictions")

        # 2. Count each prediction once and require exact claim correspondence by native ID.
        observations = exports.loc[exports.variant.eq("original")].merge(
            claims, on="id", how="outer", validate="one_to_one", suffixes=("", "_reference"), indicator=True)
        if not observations._merge.eq("both").all() or not observations.claim.eq(observations.claim_text).all():
            raise ValueError("A prediction has no uniquely matching released claim")
        observations["response_key"] = observations.id.astype(str)
        observations["item_key"] = observations.response_key
        observations["subject_key"] = self.build_parameters["subject"]["label"]
        observations["response"] = observations.verdict.str.lower().str.strip().eq(observations.label.str.lower()).astype(float)
        subjects = observations[["subject_key"]].drop_duplicates().assign(
            raw_label=self.build_parameters["subject"]["label"],
            features=[dict(self.build_parameters["subject"])])

        # 3. Attach only claim images; gold evidence and fact-check annotations are not inputs.
        items = observations.copy()
        images = items[["item_key", "claim_images"]].explode("claim_images").dropna(subset="claim_images")
        images["path"] = "images/" + images.claim_images
        with ZipFile(self.raw_dir / layout["images"]) as archive:
            images["data"] = images.path.map(archive.read)
        images["attachment"] = [dict(path=row.path, data=row.data, role="input", media_type="image/jpeg")
                                for row in images.itertuples()]
        attachments = images.groupby("item_key", sort=False).attachment.agg(list)
        items["attachments"] = items.item_key.map(attachments)
        items["content"] = [json.dumps(dict(claim_text=row.claim_text, claim_date=row.date,
            claim_images=["images/" + name for name in row.claim_images], speaker=row.metadata["speaker"],
            original_claim_url=row.metadata["original_claim_url"]), ensure_ascii=False, sort_keys=True)
            for row in items.itertuples()]
        items["content"] = items.content.str.normalize("NFC").str.strip()
        items["raw_item_id"] = "val:" + items.id.astype(str)
        items["features"] = [{"split": "val"} for _ in items.index]
        items["grading_criterion"] = items.label.map(lambda label: dict(reference_answer=label, rule=self.grading["rule"]))
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["verdict"], sort_keys=True))

        # 4. Keep both complete output variants and their source locations in one trace.
        exports["source_record"] = [dict(source_file=row.source_file, source_row=row.source_row,
            notebook_cell=int(layout["cell"]), notebook_output=int(layout["output"]),
            variant=row.variant, prediction=row.native_record) for row in exports.itertuples()]
        contexts = exports.groupby("id", sort=False).source_record.agg(list)
        traces = observations[["response_key", "id"]].copy()
        traces["trace"] = [json.dumps(dict(claim_file=layout["claims"], claim_row=key,
            exports=contexts[key]), ensure_ascii=False, allow_nan=False) for key in traces.id]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    AVerImaTeC(__file__).main_from_args()
