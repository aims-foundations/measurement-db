"""Tabulate released GHOSTS prompts, original human ratings and full annotations."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class GhostsMath(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read native JSON arrays and retain both coordinates of nested records.
        with ZipFile(self.raw_dir / parameters["paths"]["archive"]) as archive:
            root = parameters["paths"]["root"] + "/"
            paths = sorted(name for name in archive.namelist() if name.endswith(".json"))
            files = pd.DataFrame({"source_file": [name.removeprefix(root) for name in paths],
                                  "records": [json.loads(archive.read(name)) for name in paths]})
        rows = files.explode("records", ignore_index=True)
        rows["source_row"] = rows.groupby("source_file").cumcount()
        rows["records"] = rows.records.map(lambda value: value if isinstance(value, list) else [value])
        rows = rows.explode("records", ignore_index=True)
        rows["source_subrow"] = rows.groupby(["source_file", "source_row"]).cumcount()
        rows = rows.join(pd.json_normalize(rows.records.tolist(), max_level=0))
        rows["subject_key"] = rows.source_file.str.split("/").str[0]
        rows["category"] = rows.source_file.str.split("/").str[-1].str.removesuffix(".json")

        # 2. Keep the released prompts. Withheld questions remain in the raw archive only.
        rows = rows.loc[rows.prompt.str.strip().ne("")].copy()
        rows["response"] = pd.to_numeric(rows.rating, errors="raise")
        rows["response_key"] = rows.source_file + ":" + rows.source_row.astype(str) + ":" + rows.source_subrow.astype(str)
        rows["item_key"] = rows.response_key
        if not rows.subject_key.isin(parameters["models"]).all():
            raise ValueError("Unknown GHOSTS model/version folder")

        # 3. Register the original stimulus with the common human-grading protocol.
        items = rows[["item_key", "prompt", "category", "source_row", "source_subrow"]].rename(columns={"prompt": "content"})
        items["raw_item_id"] = items.category + ":" + items.source_row.astype(str) + ":" + items.source_subrow.astype(str)
        items["features"] = items[["category"]].to_dict("records")
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["rating"], sort_keys=True), judged_by="human") for _ in items.index]
        subjects = rows[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key.map(parameters["models"])
        subjects["features"] = [dict(**parameters["subject_features"], source_run=key) for key in subjects.subject_key]

        # 4. Preserve every original field; annotation comments are not extracted as gold answers.
        traces = rows[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_subrow=row.source_subrow, record=row.records), ensure_ascii=False, allow_nan=False) for row in rows.itertuples()]
        return {"subjects": subjects, "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": rows[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    GhostsMath(__file__).main()
