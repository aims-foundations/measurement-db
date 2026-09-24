#!/usr/bin/env python3
"""Join MMDocRAG's question, judgment and answer tables without filling missing grades."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MMDocRAG(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load questions in the declared precedence order and parse evaluation filenames.
        layout = self.build_parameters["layout"]
        bank = pd.concat([pd.read_json(self.raw_dir / layout[key], lines=True, dtype=False, convert_dates=False)
                          for key in ["preferred_gold", "fallback_gold"]], ignore_index=True)
        bank = bank.dropna(subset="q_id").drop_duplicates("q_id", keep="first")
        bank["answer_short"] = bank.answer_short.map(lambda value: ", ".join(map(str, value)) if isinstance(value, list) else value)
        names = sorted(name for name in self.source_files if Path(name).parent.as_posix() == layout["evaluation_directory"])
        files = pd.DataFrame({"source_file": names, "filename": [Path(name).name for name in names]})
        files = files.join(files.filename.str.extract(self.build_parameters["patterns"]["evaluation"]))
        if files[["model", "mode", "quotes"]].isna().any().any():
            raise ValueError("Unrecognized MMDocRAG evaluation filename")
        frames = []
        for filename in files.source_file:
            frame = pd.read_json(self.raw_dir / filename, lines=True, dtype=False, convert_dates=False)
            frames.append(frame.rename(columns={"model": "judge_model", "response": "dimensions"}).assign(source_file=filename))
        observations = pd.concat(frames, ignore_index=True)
        observations = observations.loc[observations.q_id.notna() & observations.dimensions.map(lambda value: isinstance(value, dict))]
        observations = observations.assign(response_key=observations.index)

        # 2. Normalize dimension keys; keep the first valid numeric value for each dimension.
        dimensions = observations[["response_key", "dimensions"]].assign(
            pair=lambda frame: frame.dimensions.map(lambda value: list(value.items()))
        ).explode("pair", ignore_index=True).dropna(subset="pair")
        fields = pd.DataFrame(dimensions.pair.tolist(), columns=["dimension", "value"], dtype=object).set_axis(dimensions.index)
        dimensions = dimensions[["response_key"]].join(fields)
        rubric = self.grading["verifiers"]["release"]
        noise = self.build_parameters["patterns"]["dimension_noise"]
        dimensions["dimension"] = dimensions.dimension.astype(str).str.lower().str.replace(noise, "", regex=True)
        expected = pd.Series(rubric["dimensions"]).str.lower().str.replace(noise, "", regex=True)
        numeric = dimensions.value.map(lambda value: type(value) in (int, float))
        dimensions = dimensions.loc[numeric & dimensions.dimension.isin(expected)].copy()
        dimensions["value"] = pd.to_numeric(dimensions.value)
        dimensions = dimensions.loc[dimensions.value.between(0, rubric["maximum"])]
        dimensions = dimensions.drop_duplicates(["response_key", "dimension"], keep="first")
        quality = dimensions.groupby("response_key", sort=False).value.agg(["sum", "count"])
        quality["response"] = (quality["sum"] / len(expected) / rubric["maximum"]).where(quality["count"].eq(len(expected)))
        observations = observations.join(quality.response, on="response_key")

        # 3. Resolve subjects and select trace files by exact case, preferring the current filename.
        files["subject_key"] = files.model.str.normalize("NFC").str.strip().str.lower()
        subjects = files.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = subjects.model.replace(self.build_parameters["subject_aliases"])
        variants = subjects.model.map(self.build_parameters["released_variants"])
        subjects["features"] = variants.map(lambda value: {"released_variant": value} if pd.notna(value) else None)
        prefix = layout["trace_directory"] + "/" + files.model + "_" + files["mode"]
        preferred = prefix + "_quotes" + files.quotes + "_response.jsonl"
        fallback = prefix + "_response_quotes" + files.quotes + ".jsonl"
        files["trace_file"] = preferred.where(preferred.isin(self.source_files), fallback)
        trace_frames = []
        for filename in files.loc[files.trace_file.isin(self.source_files), "trace_file"].drop_duplicates():
            frame = pd.read_json(self.raw_dir / filename, lines=True, dtype=False, convert_dates=False)
            frame = frame.reindex(columns=["q_id", "response"]).dropna(subset="q_id").drop_duplicates("q_id", keep="last")
            trace_frames.append(frame.rename(columns={"response": "trace"}).assign(trace_file=filename))
        answers = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame(columns=["q_id", "trace", "trace_file"])
        observations = observations.merge(files[["source_file", "subject_key", "mode", "quotes", "trace_file"]],
                                           on="source_file", how="left", sort=False, validate="many_to_one")
        observations = observations.merge(answers, on=["q_id", "trace_file"], how="left", sort=False, validate="many_to_one")

        # 4. Link the evaluated questions to their reference answers and recorded grading rule.
        items = observations[["q_id"]].drop_duplicates().merge(bank[["q_id", "question", "answer_short"]],
                    on="q_id", how="left", sort=False, validate="one_to_one", indicator=True)
        if not items._merge.eq("both").all():
            raise ValueError("An MMDocRAG judgment has no matching question")
        verifier = self.grading["verifiers"]["release"]
        items = items.assign(item_key=items.q_id, raw_item_id="q_id::" + items.q_id.astype(str),
            content=items.question.fillna(""),
            grading_criterion=items.answer_short.map(lambda value: {"reference_answer": value, "rule": self.grading["rule"]}),
            verifier=Judge(spec=verifier["spec"], judged_by=verifier["judged_by"]))
        responses = observations.assign(item_key=observations.q_id, trial=1,
                                         test_condition=observations["mode"] + "/quotes" + observations.quotes)
        # Compatibility input only: the old response-ID payload had a null reference slot.
        # The shared writer keeps the answer exclusively in items.parquet.
        responses["reference_answer"] = None
        has_trace = responses.trace.map(lambda value: isinstance(value, str) and bool(value))
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "trial", "test_condition", "response", "reference_answer"]],
            "traces": responses.loc[has_trace, ["response_key", "trace"]],
        }


if __name__ == "__main__":
    MMDocRAG(__file__).main_from_args()
