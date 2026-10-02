"""Join released answers only to uniquely identified questions and sensor inputs."""

import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class EngineMTQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "data", "input_audit")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Flatten alternating question/answer turns while retaining source positions.
        questions = pd.read_json(self.raw_dir / paths["questions"], lines=True, dtype=False)
        questions["source_line"] = questions.index
        turns = questions[["source_line", "conversations"]].explode("conversations", ignore_index=True)
        turns["turn"] = turns.groupby("source_line", sort=False).cumcount()
        turns["pair"] = turns.turn // 2
        turns = pd.concat([turns.drop(columns="conversations"), pd.json_normalize(turns.conversations)], axis=1)
        if not turns["from"].eq(np.where(turns.turn.mod(2).eq(0), "human", "gpt")).all():
            raise ValueError("Expected alternating human and reference-answer turns")
        prompts = turns.loc[turns["from"].eq("human"), ["source_line", "pair", "stage", "value"]].rename(columns={"value": "question"})
        answers = turns.loc[turns["from"].eq("gpt"), ["source_line", "pair", "value"]].rename(columns={"value": "gold"})
        prompts = prompts.merge(answers, on=["source_line", "pair"], validate="one_to_one")
        prompts["stage"] = prompts.stage.astype(int)
        prompts["gold"] = prompts.gold.str.strip()

        # 2. Keep the defined single-choice subset only when the question is unique.
        keys = ["source_line", "stage", "gold"]
        unique = prompts.groupby(keys).question.transform("nunique").eq(1)
        lookup = prompts.loc[unique].drop_duplicates(keys)
        dump = json.loads((self.raw_dir / paths["results"]).read_text())
        attempts = pd.DataFrame({key: dump[key] for key in ["predictions", "labels", "stages", "index"]})
        attempts["source_position"] = attempts.index
        attempts["native_record"] = attempts.to_dict("records")
        attempts = attempts.rename(columns={"index": "source_line", "stages": "stage"})
        attempts["gold"] = attempts.labels.str.strip()
        selected = attempts.stage.astype(str).isin(parameters["stage_names"]) & attempts.gold.str.fullmatch("[a-f]")
        attempts = attempts.loc[selected].merge(lookup, on=keys, validate="many_to_one")
        attempts["item_key"] = attempts.source_line.astype(str) + ":" + attempts["pair"].astype(str)
        items = attempts.drop_duplicates("item_key").merge(
            questions[["source_line", "id", "name"]], on="source_line", validate="many_to_one")

        # 3. Read the original sensor values; package each selected sequence once.
        inputs = items[["source_line", "id"]].drop_duplicates("source_line").copy()
        attachments = []
        with h5py.File(self.raw_dir / paths["sensors"], "r") as sensors:
            for line, identifiers in inputs.itertuples(index=False, name=None):
                ids = [int(value) for value in identifiers] if isinstance(identifiers, list) else [int(identifiers)]
                if len(ids) not in (1, 10) or not all(sensors["data_ID"][value - 1] == value for value in ids):
                    raise ValueError("Sensor identifiers do not match the released array")
                arrays = [sensors["seq_data"][value - 1] for value in ids]
                values = arrays[0] if len(ids) == 1 else np.concatenate([array[:60] for array in arrays])
                if values.shape != (600, 33) or values.dtype != np.dtype("float64"):
                    raise ValueError("Unexpected released sensor shape or type")
                buffer = io.BytesIO()
                np.save(buffer, values, allow_pickle=False)
                attachments.append([dict(path=f"sensors/line_{line}.npy", data=buffer.getvalue(),
                    media_type="application/x-npy", role="input")])
        inputs["attachments"] = attachments
        items = items.merge(inputs[["source_line", "attachments"]], on="source_line", validate="many_to_one")

        # 4. Preserve complete questions, grading criteria and source task identifiers.
        items["raw_item_id"] = "test_qa.jsonl:" + items.item_key
        items["content"] = items.question
        items["grading_criterion"] = [dict(reference_answer=gold, rule=self.grading["rule"]) for gold in items.gold]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_match"], sort_keys=True)) for _ in items.index]
        features = items[["source_line", "pair", "id", "name", "stage"]].rename(columns={"id": "sensor_ids", "name": "sensor_files"})
        features["stage"] = features.stage.astype(str).map(parameters["stage_names"])
        items["features"] = features.assign(**parameters["input_features"]).to_dict("records")
        subjects = pd.DataFrame([dict(subject_key="released_run", raw_label=parameters["subject"]["label"],
            features=parameters["subject_features"])])

        # 5. Apply the historical option-set rule to the existing answers, then link traces.
        predicted = attempts.predictions.str.lower().str.split().map(lambda tokens: set(tokens) & set("abcdef"))
        references = attempts.labels.str.split().map(set)
        attempts["response"] = predicted.eq(references).astype(float)
        attempts["response_key"] = attempts.source_position.astype(str)
        attempts["subject_key"] = "released_run"
        attempts["test_condition"] = "task=" + attempts.stage.astype(str).map(parameters["stage_names"])
        attempts["trace"] = attempts.native_record.map(lambda record: json.dumps(record, ensure_ascii=False, allow_nan=False))
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": attempts[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EngineMTQA(__file__).main_from_args()
