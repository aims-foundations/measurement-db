"""Tabulate every published Lo-Hi prediction without inventing individual grades."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class LoHiDrugDiscovery(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Load all original CSV cells as strings, preserving unnamed index columns.
        frames = []
        with ZipFile(self.raw_dir / layout["archive"]) as archive:
            paths = pd.DataFrame({"path": sorted(name.removeprefix(layout["prefix"]) for name in archive.namelist()
                if name.startswith(layout["prefix"] + "predictions/") and name.endswith(".csv"))})
            tags = paths.path.str.extract(layout["prediction_pattern"])
            if tags.isna().any().any():
                raise ValueError("Unrecognized source prediction path")
            for row in paths.join(tags).itertuples():
                frame = pd.read_csv(archive.open(layout["prefix"] + row.path), header=None, dtype=str, keep_default_na=False)
                frame.columns = frame.iloc[0]
                frame = frame.iloc[1:].reset_index(drop=True)
                if frame.columns.duplicated().any():
                    raise ValueError("Duplicate source CSV header")
                frame["source_record"] = frame.to_dict("records")
                frames.append(frame.assign(task=row.task, dataset=row.dataset, model=row.model,
                    partition=row.partition, fold=row.fold, source_file=row.path, source_row=range(len(frame))))
        attempts = pd.concat(frames, ignore_index=True)
        attempts["reference_value"] = attempts.value.replace({"True": "1", "False": "0"}).map(float)
        prediction_values = attempts.preds.replace({"True": "1", "False": "0"}).map(float)
        if not np.isfinite(attempts.reference_value).all() or not np.isfinite(prediction_values).all():
            raise ValueError("Nonfinite source reference or prediction")
        if not attempts.loc[attempts.task.eq("hi"), "reference_value"].isin([0, 1]).all():
            raise ValueError("Hi references must be binary as declared upstream")
        attempts["reference"] = attempts.reference_value.map(lambda value: str(float(value)))

        # 2. Define molecules by their task, dataset and released reference measurement.
        identity = ["task", "dataset", "smiles", "reference"]
        items = attempts[identity + ["source_file", "source_row"]].drop_duplicates(identity).copy()
        items["item_key"] = items.source_file + "#" + items.source_row.astype(str)
        items["raw_item_id"] = items.item_key
        items["content"] = items.smiles
        items["features"] = [dict(task=row.task, dataset=row.dataset, input_scope=parameters["labels"]["input_scope"])
            for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.reference,
            rule=self.grading["verifiers"][row.task]["rule"], response_scale=self.grading["verifiers"][row.task]["response_scale"])
            for row in items.itertuples()]
        items["verifier"] = items.task.map(lambda task: ExactMatcher(spec=json.dumps(self.grading["verifiers"][task], sort_keys=True)))

        # 3. Separate trained method/task/fold configurations and retain both partitions.
        configuration = ["model", "task", "dataset", "fold"]
        subjects = attempts[configuration].drop_duplicates().copy()
        subjects["subject_key"] = subjects[configuration].agg(":".join, axis=1)
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=row.model,
            training_task=row.task, training_dataset=row.dataset, training_fold=row.fold) for row in subjects.itertuples()]
        attempts = attempts.merge(subjects[configuration + ["subject_key"]], on=configuration, validate="many_to_one")
        attempts = attempts.merge(items[identity + ["item_key"]], on=identity, validate="many_to_one")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["test_condition"] = ("task=" + attempts.task + ";dataset=" + attempts.dataset
            + ";fold=" + attempts.fold + ";partition=" + attempts.partition)
        attempts["response"] = None

        # 4. Retain every original prediction and reference; aggregate metrics do not grade one molecule.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            record=row.source_record, grade_status=parameters["labels"]["grade_status"]), allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LoHiDrugDiscovery(__file__).main_from_args()
