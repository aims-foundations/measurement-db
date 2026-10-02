"""Tabulate LLM4IR's original program-level assessment tables."""

import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LLM4IR(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        with ZipFile(self.raw_dir / paths["archive"]) as archive:
            # 1. Load original CSVs directly into tables, preserving native rows.
            specifications = [spec for spec in parameters.values() if spec.get("role")]
            filenames = {spec["file"] for spec in specifications}
            filenames.update(parameters["decompile_files"].values())
            filenames.update(paths[key] for key in ["cfg_gold", "summary_outputs", "summary_scores", "summary_cosine"])
            tables = {}
            for filename in sorted(filenames):
                frame = pd.read_csv(io.BytesIO(archive.read(paths["prefix"] + filename)), dtype=str, keep_default_na=False)
                frame["source_record"] = [dict(file=filename, row=index, record=record)
                    for index, record in enumerate(frame.to_dict("records"))]
                frame["source_key"] = [f"{filename}#{index}" for index in range(len(frame))]
                tables[filename] = frame
            frames = []

            # 2. Reshape each task's native assessment layout to the same columns.
            for spec in specifications:
                frame = tables[spec["file"]].copy()
                task = spec["role"]
                frame = frame.rename(columns={"file_name": "program", "CPP_number": "program",
                    "category" if task == "cfg" else spec["column"]: "native_grade"})
                frame["program"] = frame.program.str.removesuffix(".dot")
                frame = frame.assign(task=task, protocol=task, model=spec["label"], form=spec.get("form", "IR"),
                    level="O3", trial=int(spec["trial"]), reference=None, output=None, output_path=None,
                    assertion_count=None)
                output_template = parameters["cfg_outputs"][spec["label"]] if task == "cfg" else spec["output"]
                if output_template:
                    frame["output_path"] = frame.program.map(lambda program: output_template.format(program=program))
                frame["source_records"] = frame.source_record.map(lambda record: [record])
                frames.append(frame)

            decompiled = tables[parameters["decompile_files"]["o3"]].rename(columns={"Number": "program"})
            decompiled = decompiled.melt(id_vars=["program", "source_record", "source_key"],
                value_vars=list(parameters["decompile_models"]), var_name="model", value_name="native_grade")
            decompiled["level"] = "O3"
            optimized = tables[parameters["decompile_files"]["optimizations"]].rename(columns={"file": "program"})
            optimized = optimized.melt(id_vars=["program", "source_record", "source_key"],
                value_vars=["GPT4o-O0", "GPT4o-O1", "GPT4o-O2", "GPT4o-O3"], var_name="column", value_name="native_grade")
            optimized["level"] = optimized.column.str.rsplit("-", n=1).str[-1]
            optimized["model"] = "GPT4o"
            alias = optimized[optimized.level.eq("O3")].set_index("program")
            primary = decompiled[decompiled.model.eq("GPT4o")].set_index("program")
            if not primary.native_grade.sort_index().equals(alias.native_grade.sort_index()):
                raise ValueError("The two GPT4o O3 assessment exports disagree")
            decompiled["source_records"] = [([record, alias.loc[program, "source_record"]] if model == "GPT4o" else [record])
                for program, model, record in decompiled[["program", "model", "source_record"]].itertuples(index=False, name=None)]
            optimized["source_records"] = optimized.source_record.map(lambda record: [record])
            frame = pd.concat([decompiled, optimized[~optimized.level.eq("O3")]], ignore_index=True)
            frame["source_key"] = frame.source_key + ":" + frame.model + ":" + frame.level
            frame["program"] = "CPP_" + frame.program
            frame = frame.assign(task="decompile", protocol="decompile", form="IR", trial=1,
                reference=None, output=None, assertion_count=None)
            frame["output_path"] = [parameters["decompile_outputs"][model + "_" + level].format(program=program)
                for model, level, program in frame[["model", "level", "program"]].itertuples(index=False, name=None)]
            frames.append(frame)

            summaries = tables[paths["summary_outputs"]].drop(columns="source_key").melt(
                id_vars=["Index", "Golden", "source_record"], var_name="Model", value_name="output")
            scores = tables[paths["summary_scores"]].merge(tables[paths["summary_cosine"]],
                on=["Index", "Model"], validate="one_to_one", suffixes=("", "_cosine"))
            scores = scores.merge(summaries.rename(columns={"source_record": "output_record"}),
                on=["Index", "Model"], validate="one_to_one")
            scores["source_records"] = [[row.source_record, row.source_record_cosine, row.output_record]
                for row in scores.itertuples()]
            frame = scores.melt(id_vars=["Index", "Model", "Golden", "output", "source_records", "source_key"],
                value_vars=["BLEU_Score", "ROUGE-L_Score", "METEOR_Score", "Similarity"],
                var_name="protocol", value_name="native_grade").rename(columns={"Index": "program", "Model": "model", "Golden": "reference"})
            frame["source_key"] = frame.source_key + ":" + frame.protocol
            frame = frame.assign(task="summary", form="IR", level="O3", trial=1, output_path=None, assertion_count=None)
            frames.append(frame)
            attempts = pd.concat(frames, ignore_index=True)

            # 3. Join full outputs and actual inputs; never substitute source code for IR.
            members = set(archive.namelist())
            output_files = attempts.output_path.dropna().drop_duplicates()
            outputs = {filename: archive.read(paths["prefix"] + filename).decode("utf-8")
                for filename in output_files if paths["prefix"] + filename in members}
            linked = attempts.output_path.notna()
            attempts.loc[linked, "output"] = attempts.loc[linked, "output_path"].map(outputs)
            attempts["output"] = attempts.output.where(attempts.output.notna(), None)
            execution = attempts.task.eq("execution") & attempts.output.notna()
            attempts.loc[execution, "assertion_count"] = attempts.loc[execution, "output"].map(
                lambda text: len(pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)))
            attempts = attempts[~(execution & attempts.assertion_count.eq(0))].copy()
            attempts["item_key"] = attempts.task + "/" + attempts.protocol + "/" + attempts.form + "/" + attempts.level + "/" + attempts.program
            items = attempts[["item_key", "task", "protocol", "form", "level", "program", "reference"]].drop_duplicates("item_key")
            items["input_file"] = [(paths["source"] if row.form == "SC" else paths["ir"]).format(
                level=row.level, program=row.program) for row in items.itertuples()]
            items["content"] = items.input_file.map(lambda filename: archive.read(paths["prefix"] + filename).decode("utf-8"))
            for index, row in items[items.task.eq("execution")].iterrows():
                filename = paths["assertions"].format(program=row.program)
                assertions = pd.read_csv(io.BytesIO(archive.read(paths["prefix"] + filename)), dtype=str, keep_default_na=False)
                items.loc[index, "content"] = json.dumps(dict(program=row.content,
                    released_assertions=assertions.assert_statement.tolist()), ensure_ascii=False)
            gold = tables[paths["cfg_gold"]].set_index("file_name")
            for index, row in items[items.task.isin(["cfg", "decompile"])].iterrows():
                items.loc[index, "reference"] = (json.dumps(gold.loc[row.program + ".dot"].drop(["source_record", "source_key"]).to_dict(), sort_keys=True)
                    if row.task == "cfg" else archive.read(paths["prefix"] + paths["source"].format(program=row.program)).decode("utf-8"))

        # 4. Keep task-specific literal labels and explicit mixed grading scales.
        attempts["subject_key"] = attempts.task + "/" + attempts.model
        subjects = attempts[["subject_key", "task", "model"]].drop_duplicates()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_task=row.task, source_model_label=row.model)
            for row in subjects.itertuples()]
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(task=row.task, input_form=row.form, optimization_level=row.level,
            source_program=row.program, source_input=row.input_file) for row in items.itertuples()]
        protocols = self.grading["verifiers"]
        items["grading_criterion"] = [dict(reference_answer=row.reference, rule=protocols[row.protocol]["rule"],
            response_scale=protocols[row.protocol]["response_scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(protocols[row.protocol], sort_keys=True)) for row in items.itertuples()]
        categorical = attempts.task.ne("summary")
        attempts["response"] = attempts.native_grade.where(~categorical,
            attempts.native_grade.map(parameters["category_codes"])).map(float)
        if attempts.response.isna().any():
            raise ValueError("An assessment has an unknown category or missing numeric grade")
        attempts["response_key"] = attempts.source_key
        attempts["test_condition"] = [json.dumps(dict(task=row.task, input_form=row.form,
            optimization_level=row.level, assessment="source_reported"), sort_keys=True) for row in attempts.itertuples()]

        # 5. Preserve native rows and complete outputs, including source failure text.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_records=row.source_records, source_program=row.program,
            source_metric=row.protocol, source_grade=row.native_grade, output_file=row.output_path,
            output=row.output, processed_assertions=row.assertion_count), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LLM4IR(__file__).main_from_args()
