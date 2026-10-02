#!/usr/bin/env python3
"""Join FGBench's native model outputs to its test questions and deterministic grader."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class FGBench(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("questions", "predictions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the test bank. Preserve row order: batch IDs refer to these positions.
        bank = pd.read_json(self.raw_dir / "test.jsonl", lines=True, dtype=False, convert_dates=False)
        bank = bank.assign(item_key=bank.index, task=bank.type.str.rsplit("_", n=1).str[0])
        bank["question_key"] = bank.question.str.extract(r"(For a [\s\S]*)", expand=False).str.strip().str.replace(
            " at postion ", " at position ", regex=False
        )
        if bank.question_key.isna().any() or bank.question_key.duplicated().any():
            raise ValueError("FGBench questions must have distinct, nonempty source keys")
        bank["gold"] = pd.to_numeric(bank.answer.replace({"True": "1", "False": "0"}), errors="raise")

        # 2. Normalize the eight readable exports and retain full generated answers.
        models = {name: fields for name, fields in self.build_parameters.items() if name != "parser_patterns"}
        frames = []
        for name, config in models.items():
            frame = pd.read_json(self.raw_dir / config["file"], lines=True, dtype=False, convert_dates=False)
            if config["format"] == "openai_batch":
                body = pd.json_normalize(frame.response.tolist())
                if frame.error.notna().any() or not body.status_code.eq(200).all():
                    raise ValueError("A FGBench API batch record is not a successful response")
                frame = frame.assign(
                    item_key=frame.custom_id.str.rsplit("_", n=1).str[-1].astype(int),
                    batch_dataset=frame.custom_id.str.rsplit("_", n=1).str[0],
                    trace=body["body.choices"].str[0].map(lambda choice: choice["message"]["content"]),
                    recorded_model=body["body.model"], label=None,
                )
                if not frame.recorded_model.eq(config["raw_label"]).all():
                    raise ValueError("FGBench API model version differs from metadata")
                frame = frame.merge(bank[["item_key", "dataset"]], on="item_key", how="left", validate="one_to_one")
                if not frame.batch_dataset.eq(frame.dataset).all():
                    raise ValueError("A FGBench batch ID does not match the pinned question bank")
            else:
                frame["question_key"] = frame.question.str.extract(r"(For a [\s\S]*)", expand=False).str.split(
                    r"<\|(?:eot_id|im_end)\|>", regex=True
                ).str[0].str.strip().str.replace(" at postion ", " at position ", regex=False)
                frame = frame.merge(bank[["question_key", "item_key"]], on="question_key", how="left", validate="one_to_one")
                frame = frame.assign(trace=frame[config["prediction_field"]],
                                     label=frame.get(config.get("label_field", "")))
            if frame.item_key.isna().any() or len(frame) != len(bank) or frame.item_key.duplicated().any():
                raise ValueError("Each FGBench model export must cover the pinned test bank exactly once")
            frames.append(frame[["item_key", "trace", "label"]].assign(subject_key=name, parser=config["parser"]))
        responses = pd.concat(frames, ignore_index=True).merge(
            bank[["item_key", "task", "gold"]], on="item_key", how="left", sort=False, validate="many_to_one"
        )
        labels = pd.to_numeric(responses.label.replace({"True": "1", "False": "0"}), errors="raise")
        if not np.isclose(labels[labels.notna()], responses.loc[labels.notna(), "gold"], rtol=0, atol=1e-12).all():
            raise ValueError("An embedded FGBench reference answer disagrees with the question bank")
        if not responses.trace.map(lambda value: isinstance(value, str)).all():
            raise ValueError("FGBench predictions must be strings")

        # 3. Apply the released parser precedence with vectorized string operations.
        patterns = self.build_parameters["parser_patterns"]
        boxed = responses.trace.str.extract(patterns["boxed"], expand=False).str.strip().str.replace(r"[\[\]]", "", regex=True)
        sentence = responses.trace.str.extract(patterns["sentence"], expand=False)
        parsed = responses.trace.copy()
        molinst = responses.parser.eq("molinst")
        parsed.loc[molinst] = parsed.loc[molinst].str.replace(r"[\[\]]", "", regex=True)
        llasmol = responses.parser.eq("llasmol")
        tagged = parsed.str.extract(patterns["boolean_tag"], expand=False).fillna(
            parsed.str.extract(patterns["number_tag"], expand=False)
        ).fillna(parsed.str.extract(patterns["brackets"], expand=False))
        parsed.loc[llasmol] = tagged.loc[llasmol].fillna(parsed.loc[llasmol])
        parsed = boxed.fillna(sentence).fillna(parsed)
        prediction = pd.to_numeric(parsed, errors="coerce")
        missing = prediction.isna()
        prediction.loc[missing & parsed.str.contains("rue", regex=False)] = 1.0
        prediction.loc[prediction.isna() & parsed.str.contains("alse", regex=False)] = 0.0
        nach0 = responses.parser.eq("nach0") & boxed.isna() & sentence.isna()
        prediction.loc[prediction.isna() & nach0 & parsed.str.contains("yes|Yes")] = 1.0
        prediction.loc[prediction.isna() & nach0 & parsed.str.contains("no|No") & ~parsed.str.contains("one")] = 0.0
        if np.isinf(prediction).any():
            raise ValueError("FGBench predictions must be finite")

        # 4. Boolean tasks use exact correctness; numeric tasks retain squared error.
        # Averaging the latter and taking its square root reproduces upstream RMSE.
        boolean = responses.task.str.endswith("bool")
        grade = (responses.gold - prediction).pow(2)
        grade.loc[prediction.isna() | prediction.eq(0) | responses.gold.eq(0)] = np.nan
        grade.loc[boolean] = prediction.loc[boolean].eq(responses.loc[boolean, "gold"]).astype(float)
        responses = responses.assign(response_key=responses.index, response=grade,
                                     test_condition="task=" + responses.task)

        # 5. Attach grading protocols and link the local table keys.
        specs = self.grading["verifiers"]
        items = bank.assign(raw_item_id=bank.dataset + ":" + bank.index.astype(str), content=bank.question)
        kinds = bank.task.str.endswith("bool").map({True: "boolean", False: "numeric"})
        items["grading_criterion"] = pd.Series([
            {"reference_answer": answer, "rule": specs[kind]["rule"], "response_scale": specs[kind]["response_scale"]}
            for answer, kind in zip(bank.answer, kinds)
        ])
        items["verifier"] = kinds.map(lambda kind: ExactMatcher(spec=json.dumps(specs[kind], sort_keys=True)))
        subjects = pd.DataFrame.from_dict(models, orient="index").reset_index(names="subject_key")
        subjects["features"] = subjects.parser.map(lambda parser: {"harness": "FGBench", "answer_parser": parser})
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses[["response_key", "trace"]],
        }


if __name__ == "__main__":
    FGBench(__file__).main_from_args()
