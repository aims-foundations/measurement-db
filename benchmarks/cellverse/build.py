"""Join CellVerse's released DeepSeek-R1 outputs to the cell-type task bank."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CellVerse(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the two native record tables, preserving literal text and source order.
        layout = self.build_parameters["layout"]
        bank = pd.read_json(self.raw_dir / layout["tasks"], dtype=False, convert_dates=False)
        results = pd.read_json(self.raw_dir / layout["results"], dtype=False, convert_dates=False)
        bank = bank.assign(item_key=bank.index.astype(str))
        for table in (bank, results):
            if not table.messages.map(lambda messages: [m["role"] for m in messages] == ["system", "user", "assistant"]).all():
                raise ValueError("CellVerse records must distinguish input messages from the reference answer")
            table["content"] = table.messages.map(lambda messages: json.dumps(messages[:-1], ensure_ascii=False))
            table["reference"] = table.messages.str[-1].str["content"]
        if not results.ground_truth.eq(results.reference).all():
            raise ValueError("A CellVerse result disagrees with its embedded reference answer")
        if not results[["ground_truth", "prediction", "model_response"]].map(lambda value: isinstance(value, str) and bool(value)).all().all():
            raise ValueError("CellVerse results need released predictions, references and complete outputs")

        # 2. Match complete inputs and references, never a positional guess or aggregate score.
        responses = results.merge(bank[["item_key", "content", "reference"]],
            on=["content", "reference"], how="outer", validate="one_to_one", indicator=True)
        if not responses._merge.eq("both").all():
            raise ValueError("The CellVerse result file does not exactly cover the pinned task bank")
        responses = responses.assign(
            response_key=responses.item_key, subject_key=self.build_parameters["model"]["raw_label"],
            response=responses.prediction.eq(responses.reference).astype(float),
        )

        # 3. Keep the reference outside the input, and preserve the full generated answer.
        items = bank.assign(
            raw_item_id=Path(layout["tasks"]).stem + ":" + bank.item_key,
            grading_criterion=bank.reference.map(lambda answer: {"reference_answer": answer, "rule": self.grading["rule"]}),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_match"], sort_keys=True)),
            features=[self.build_parameters["item_features"] for _ in range(len(bank))],
        )
        subjects = pd.DataFrame([{
            "subject_key": self.build_parameters["model"]["raw_label"],
            "raw_label": self.build_parameters["model"]["raw_label"],
            "features": self.build_parameters["subject_features"],
        }])
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": responses[["response_key", "model_response"]].rename(columns={"model_response": "trace"}),
        }


if __name__ == "__main__":
    CellVerse(__file__).main_from_args()
