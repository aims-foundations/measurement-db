"""Tabulate published KMMLU predictions against their original question bank."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class KMMLU(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters["labels"]

        # 1. Read original question and result CSVs, retaining their source positions.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters["paths"]["tasks"])):
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            table["bank_file"] = str(path.relative_to(self.raw_dir))
            table["bank_row"] = range(len(table))
            table["category_key"] = path.stem.removesuffix("-test").lower().replace("-", "_")
            parts.append(table)
        bank = pd.concat(parts, ignore_index=True)
        bank["gold"] = bank.answer.map(parameters["answer_labels"])
        if bank.gold.isna().any():
            raise ValueError("KMMLU has an unknown reference-answer encoding")
        bank["item_key"] = bank.category_key + "/" + bank.bank_row.astype(str)
        parts = []
        for path in sorted(self.raw_dir.glob(parameters["paths"]["results"])):
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if list(table.columns) != ["category", "answer", "pred", "response"]:
                raise ValueError(f"Unexpected result columns in {path.name}")
            table["source_record"] = table.to_dict("records")
            table["source_file"] = str(path.relative_to(self.raw_dir))
            table["source_row"] = range(len(table))
            table["subject_key"] = path.stem
            table["category_key"] = table.category.str.lower().str.replace("-", "_")
            table["bank_row"] = table.groupby("category_key", sort=False).cumcount()
            parts.append(table)
        attempts = pd.concat(parts, ignore_index=True)

        # 2. Join only complete category blocks whose entire reference order agrees.
        # Partial or reordered blocks remain captured in raw; never align by a grade.
        attempts = attempts.merge(bank[["category_key", "bank_row", "bank_file", "item_key", "gold"]],
            on=["category_key", "bank_row"], how="left", validate="many_to_one")
        if attempts.item_key.isna().any():
            raise ValueError("A KMMLU record has no matching category or exceeds its task bank")
        attempts["reference_matches"] = attempts.answer.eq(attempts.gold)
        groups = attempts.groupby(["subject_key", "category_key"], sort=False).agg(
            rows=("source_row", "size"), references_match=("reference_matches", "all"))
        groups = groups.join(bank.groupby("category_key").size().rename("expected_rows"), on="category_key")
        eligible = groups.loc[groups.rows.eq(groups.expected_rows) & groups.references_match].reset_index()
        attempts = attempts.merge(eligible[["subject_key", "category_key"]],
            on=["subject_key", "category_key"], how="inner", validate="many_to_one")

        # 3. Apply the released exact matcher to the recorded parsed prediction.
        # FAILED was excluded by that scorer; retain the attempt with a null grade.
        attempts["response"] = attempts.pred.eq(attempts.gold).astype(float)
        attempts.loc[attempts.pred.eq(labels["ungraded_marker"]), "response"] = None
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["test_condition"] = "category=" + attempts.category_key

        # 4. Keep the published model/configuration labels, without guessed settings.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=label,
            prompt_condition=label.rsplit("-", 1)[-1] if label.endswith(("-0shot", "-5shot")) else "not_recorded")
            for label in subjects.subject_key]

        # 5. Format the original question and all choices; references stay separate.
        items = bank.loc[bank.item_key.isin(attempts.item_key)].copy()
        items["content"] = items.question + "\n\n" + (
            "A: " + items.A + "\nB: " + items.B + "\nC: " + items.C + "\nD: " + items.D)
        items["content"] = items.content.str.strip()
        items["raw_item_id"] = items.item_key
        # Shared registration merges identical stimuli and grading. Each trace
        # retains its original bank row, including repeated source questions.
        items["features"] = [dict(input_scope=labels["input_scope"]) for _ in items.index]
        items["grading_criterion"] = [dict(reference_answer=gold, rule=self.grading["rule"]) for gold in items.gold]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["recorded_option_match"], sort_keys=True))

        # 6. Preserve each full CSV record and its exact question-bank association.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            bank_file=row.bank_file, bank_row=int(row.bank_row), source_record=row.source_record,
            grade_status="ungraded_source_failure" if row.pred == labels["ungraded_marker"] else "derived_from_recorded_prediction"),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    KMMLU(__file__).main_from_args()
