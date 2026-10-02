"""Tabulate the complete published KorMedMCQA prediction records."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class KorMedMCQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters["labels"]

        # 1. Read the original test banks and retain their subset-relative IDs.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters["paths"]["tasks"])):
            table = pd.read_parquet(path)
            table["subset"] = path.parent.name
            table["bank_file"] = str(path.relative_to(self.raw_dir))
            table["bank_row"] = range(len(table))
            table["item_key"] = table.subset + "_" + table.bank_row.astype(str)
            parts.append(table)
        bank = pd.concat(parts, ignore_index=True)
        bank["gold"] = bank.answer.astype(str).map(parameters["answer_labels"])
        if bank.gold.isna().any() or bank[["question", "A", "B", "C", "D", "E"]].isna().any().any():
            raise ValueError("KorMedMCQA has an unknown answer or missing question/choice")

        # 2. Read every saved CSV row; repeated IDs are not silently discarded.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters["paths"]["results"])):
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if list(table.columns) != ["id", "category", "trial", "answer", "pred", "response"]:
                raise ValueError(f"Unexpected result columns in {path.name}")
            table["source_record"] = table.to_dict("records")
            table["source_file"] = str(path.relative_to(self.raw_dir))
            table["source_row"] = range(len(table))
            table["subject_key"] = path.stem
            table = table.rename(columns={"id": "item_key", "trial": "source_trial"})
            parts.append(table)
        attempts = pd.concat(parts, ignore_index=True)

        # 3. Join explicit source IDs and verify every category and reference.
        attempts = attempts.merge(bank[["item_key", "subset", "gold", "bank_file", "bank_row"]],
            on="item_key", how="left", validate="many_to_one")
        if attempts.gold.isna().any() or not attempts.answer.eq(attempts.gold).all() or not attempts.category.eq(attempts.subset).all():
            raise ValueError("A KorMedMCQA record disagrees with its original question bank")
        if not attempts.source_trial.str.fullmatch(r"\d+").all():
            raise ValueError("A source trial is not a nonnegative integer")
        attempts["response"] = attempts.pred.eq(attempts.gold).astype(float)
        attempts.loc[attempts.pred.eq(labels["ungraded_marker"]), "response"] = None
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["test_condition"] = "subset=" + attempts.subset + ";source_trial=" + attempts.source_trial

        # 4. Retain source model labels; do not infer unrecorded run settings.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=label)
            for label in subjects.subject_key]

        # 5. Format questions and all choices, with references kept separate.
        items = bank.loc[bank.item_key.isin(attempts.item_key)].copy()
        items["raw_item_id"] = items.item_key
        items["content"] = (items.question + "\n\nA: " + items.A + "\nB: " + items.B
            + "\nC: " + items.C + "\nD: " + items.D + "\nE: " + items.E).str.strip()
        items["features"] = [dict(input_scope=labels["input_scope"]) for _ in items.index]
        items["grading_criterion"] = [dict(reference_answer=gold, rule=self.grading["rule"]) for gold in items.gold]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["recorded_option_match"], sort_keys=True))

        # 6. Preserve full records and positions. Shared registration numbers
        # occurrences after canonical IDs resolve; source trials remain in traces.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            bank_file=row.bank_file, bank_row=int(row.bank_row), source_record=row.source_record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    KorMedMCQA(__file__).main_from_args()
