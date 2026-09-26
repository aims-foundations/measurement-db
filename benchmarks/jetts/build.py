"""Tabulate the complete released JETTS generator response pool."""

import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class JETTS(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("response_pool", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, protocols = self.build_parameters, self.grading["verifiers"]

        # 1. Read each original JSONL export directly into a table.
        exports = []
        with tarfile.open(self.raw_dir / parameters["paths"]["pool"], mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".jsonl") or Path(member.name).name.startswith("._"):
                    continue
                component, generator = Path(member.name).stem.split("_", 1)
                frame = pd.read_json(io.BytesIO(archive.extractfile(member).read()), lines=True, dtype=False, convert_dates=False, precise_float=True)
                exports.append(frame.assign(source_file=member.name, source_row=frame.index, component=component, generator=generator))
        records = pd.concat(exports, ignore_index=True).sort_values(["source_file", "source_row"], ignore_index=True)
        records["query_metadata"] = records["query"].map(lambda value: value["metadata"])
        records["raw_item_id"] = records.component + ":" + pd.Series([
            str(next((value[key] for key in ["task_id", "problem_id", "key"] if value.get(key) is not None), position))
            for value, position in zip(records.query_metadata, records.source_row)], index=records.index)
        records["query_json"] = records["query"].map(lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
        if records.groupby("raw_item_id").query_json.nunique().gt(1).any():
            raise ValueError("A native query ID refers to conflicting task definitions")

        # 2. Keep one item per original task, with its component grading protocol.
        queries = records.drop_duplicates("raw_item_id").reset_index(drop=True)
        items = queries[["raw_item_id"]].assign(item_key=queries.raw_item_id, content=queries["query"].map(lambda value: value["content"]))
        items["features"] = [dict(component=row.component, component_name=protocols[row.component]["component"],
                                 **parameters["observation"]) for row in queries.itertuples()]
        references = [{key: row.query_metadata[key] for key in protocols[row.component]["reference_fields"]} for row in queries.itertuples()]
        constraints = [{key: row.query_metadata[key] for key in protocols[row.component]["constraint_fields"]} for row in queries.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(reference, sort_keys=True, ensure_ascii=False) if reference else None,
            rule=protocols[component]["rule"] + ("\n" + json.dumps(constraint, sort_keys=True, ensure_ascii=False) if constraint else ""),
            response_scale=protocols[component]["response_scale"]) for component, reference, constraint in zip(queries.component, references, constraints)]
        items["verifier"] = [Judge(spec=json.dumps(protocols[component], sort_keys=True), judged_by="llm")
            if protocols[component]["judged_by"] == "llm" else ExactMatcher(spec=json.dumps(protocols[component], sort_keys=True))
            for component in queries.component]

        # 3. Expand actual responses, preserving order and separate decoding settings.
        attempts = records.explode("responses", ignore_index=True)
        attempts["position"] = attempts.groupby(["source_file", "source_row"], sort=False).cumcount()
        attempts["decoding"] = attempts.position.eq(0).map({True: "greedy", False: "sampled"})
        attempts["subject_key"] = attempts.generator
        subjects = attempts[["subject_key", "generator"]].drop_duplicates().reset_index(drop=True)
        subjects["features"] = [dict(model_identifier=row.generator, **parameters["subject_features"]) for row in subjects.itertuples()]
        subjects = subjects.rename(columns={"generator": "raw_label"})
        responses = attempts[["subject_key", "raw_item_id"]].rename(columns={"raw_item_id": "item_key"})
        responses["response_key"] = attempts.source_file + ":" + attempts.source_row.astype(str) + ":" + attempts.position.astype(str)
        responses["response"] = attempts.responses.map(lambda value: value["metadata"].get("score"))
        responses["test_condition"] = ("decoding=" + attempts.decoding + ";temperature=" + attempts.decoding.map(parameters["temperatures"])
            + ";top_p=" + attempts.decoding.map(parameters["top_p"]))

        # 4. Preserve complete source queries, outputs and auxiliary grading fields.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            response_position=int(row.position), query=row.query, response=row.responses), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects, "items": items, "responses": responses, "traces": traces}


if __name__ == "__main__":
    JETTS(__file__).main_from_args()
