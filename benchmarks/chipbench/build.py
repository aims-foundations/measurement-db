#!/usr/bin/env python3
"""Curate published ChiPBench measurements and their released circuit designs."""

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ChiPBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, parsing = parameters["paths"], parameters["parsing"]
        metrics = parsing["metrics"].split(",")
        frames, resource_frames = [], []

        # 1. Read the printed circuit groups as tables, then explode their method rows.
        with pymupdf.open(self.raw_dir / paths["paper"]) as paper:
            for page_number in map(int, parsing["placement_pages"].split(",")):
                groups = pd.DataFrame(paper[page_number].find_tables().tables[0].extract(), columns=["design", "rows"])
                groups["design"] = groups.design.str.replace(r"\s+", "", regex=True).replace(parameters["design_aliases"])
                groups["rows"] = groups.rows.str.split("\n")
                rows = groups.explode("rows", ignore_index=True)
                values = rows.rows.str.split(expand=True)
                values.columns = ["method", *metrics]
                frames.append(rows[["design"]].join(values).assign(source_page=page_number + 1, source_line=rows.rows))
            main = pd.concat(frames, ignore_index=True)

            # 2. Retain resource-only attempts and the separate commercial/synthesis studies.
            resources = paper[int(parsing["resource_page"])].get_text(sort=True)
            for name, marker in parameters["resource_markers"].items():
                section = resources.split(marker, 1)[1].split("Table ", 1)[0]
                lines = pd.Series(section.splitlines())
                lines = lines.loc[lines.str.fullmatch(parsing["design_row_pattern"])].reset_index(drop=True)
                values = lines.str.split(expand=True)
                values.columns = ["design", *parsing["resource_methods"].split(",")]
                values["resource_line"] = lines
                resource_frames.append(values.melt(id_vars=["design", "resource_line"], var_name="method", value_name=name))
            resource_rows = resource_frames[0].merge(resource_frames[1], on=["design", "method"], validate="one_to_one", suffixes=("_time", "_memory"))
            main = main.merge(resource_rows, on=["design", "method"], how="outer", validate="one_to_one")
            main["stage"] = "macro_placement"
            main["native_record"] = main.astype(object).where(main.notna(), None).to_dict("records")
            placement = main.melt(id_vars=["design", "method", "stage", "native_record"], value_vars=metrics, var_name="metric", value_name="response")

            section = paper[int(parsing["commercial_page"])].get_text(sort=True).split(parsing["commercial_end"], 1)[0]
            lines = pd.Series(section.splitlines())
            lines = lines.loc[lines.str.fullmatch(parsing["design_row_pattern"])].reset_index(drop=True)
            commercial = lines.str.split(expand=True)
            commercial.columns = ["design", *metrics]
            commercial["native_record"] = commercial.assign(source_line=lines, source_page=int(parsing["commercial_page"]) + 1).to_dict("records")
            commercial = commercial.melt(id_vars=["design", "native_record"], value_vars=metrics, var_name="metric", value_name="response")
            commercial = commercial.assign(method="Synopsys commercial placer", stage="macro_placement")

            section = paper[int(parsing["synthesis_page"])].get_text(sort=True).split(parsing["synthesis_end"], 1)[0]
            lines = pd.Series(section.splitlines())
            lines = lines.loc[lines.str.fullmatch(parsing["design_row_pattern"])].reset_index(drop=True)
            synthesis = lines.str.split(expand=True)
            synthesis.columns = parsing["synthesis_columns"].split(",")
            synthesis["native_record"] = synthesis.assign(source_line=lines, source_page=int(parsing["synthesis_page"]) + 1).to_dict("records")
            synthesis = synthesis.melt(id_vars=["design", "native_record"], var_name="column", value_name="response")
            synthesis[["metric", "method"]] = synthesis.column.str.split(":", expand=True)
            synthesis["stage"] = "logic_synthesis"
        responses = pd.concat([placement, commercial, synthesis.drop(columns="column")], ignore_index=True)
        responses["response"] = pd.to_numeric(responses.response, errors="raise")
        responses["response_key"] = responses.index
        responses["subject_key"] = responses.stage + ":" + responses.method
        responses["item_key"] = responses.stage + ":" + responses.design + ":" + responses.metric
        if responses.duplicated(["subject_key", "item_key"]).any():
            raise ValueError("A printed design/algorithm/metric cell occurs more than once")

        # 3. Keep all released design kits; distinguish the reference RTL and baseline layouts.
        design_root = self.raw_dir / paths["designs"]
        designs = pd.DataFrame(dict(design=sorted(path.name for path in design_root.iterdir() if path.is_dir())))
        items = designs.merge(pd.DataFrame(dict(metric=metrics)), how="cross").assign(stage="macro_placement")
        items = pd.concat([items, synthesis[["design", "metric", "stage"]].drop_duplicates()], ignore_index=True)
        items["item_key"] = items.stage + ":" + items.design + ":" + items.metric
        if not responses.item_key.isin(items.item_key).all():
            raise ValueError("A published result does not match the released design bank")
        attachments = {}
        for stage, design in items[["stage", "design"]].drop_duplicates().itertuples(index=False, name=None):
            files = sorted((design_root / design).rglob("*"))
            if stage == "logic_synthesis":
                for relative in parameters["synthesis_rtl"][design].split("|"):
                    root = self.raw_dir / paths["reference"] / relative
                    if not root.is_dir():
                        raise ValueError(f"Missing released reference RTL/configuration directory: {relative}")
                    files.extend(sorted(root.rglob("*")))
            attachments[stage, design] = [dict(source_path=path, path=str(path.relative_to(self.raw_dir)), media_type="text/plain",
                role="baseline_placement" if path.name == "macro_placed.def" else
                     "reference_rtl" if path.is_relative_to(self.raw_dir / paths["reference"]) else "published_design_kit")
                for path in files if path.is_file()]
        items["attachments"] = [attachments[row.stage, row.design] for row in items.itertuples()]
        items["content"] = [json.dumps(dict(design=row.design, stage=row.stage,
            released_files=[dict(path=value["path"], role=value["role"]) for value in row.attachments]), ensure_ascii=False)
            for row in items.itertuples()]
        items["raw_item_id"] = items.item_key

        # 4. Each physical metric retains its native direction, unit and grading protocol.
        items["features"] = [dict(design=row.design, stage=row.stage, grading_channel=row.metric,
            reported_unit=parameters["reported_units"][row.metric]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"] + " " + parameters["metric_rules"][row.metric],
            response_scale=dict(kind="interval", min=None if row.metric in {"wns", "tns"} else 0, max=None,
                direction="higher_is_better" if row.metric in {"wns", "tns"} else "lower_is_better")) for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=json.dumps({**self.grading["verifiers"]["native_metric"],
            "metric": row.metric, "stage": row.stage, "reported_unit": parameters["reported_units"][row.metric]}, sort_keys=True)) for row in items.itertuples()]
        subjects = responses[["subject_key", "method", "stage"]].drop_duplicates().rename(columns={"method": "raw_label"})
        subjects["features"] = [dict(parameters["subject"], source_component=row.stage,
            **({"reported_settings": quote(parameters["reported_settings"][row.raw_label], safe="")} if row.raw_label in parameters["reported_settings"] else {}))
            for row in subjects.itertuples()]

        # 5. Preserve printed cells and complete source rows, including absent PPA grades.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["paper"], record_kind="published_measurement",
            design=row.design, method=row.method, stage=row.stage, metric=row.metric,
            native_record=row.native_record, grade_status="not_reported" if pd.isna(row.response) else "published_value"),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ChiPBench(__file__).main_from_args()
