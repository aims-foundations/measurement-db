#!/usr/bin/env python3
"""Curate released EngDesign trials and refinement outcomes without rerunning models."""

import ast
import io
import json
import mimetypes
from pathlib import Path
import sys
import tokenize
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class EngDesign(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        with zipfile.ZipFile(self.raw_dir / parameters["layout"]["archive"]) as archive:
            # 1. Index native files. The open subset is a newline-only copy of tasks/.
            files = pd.DataFrame({"member": [name for name in archive.namelist() if not name.endswith("/")]})
            files["source_file"] = files.member.str.split("/", n=1).str[1]
            logs = files.loc[files.source_file.str.endswith(".jsonl")].copy()
            single = logs.join(logs.source_file.str.extract(parameters["patterns"]["single"])).dropna(subset=["task"])
            single = single.assign(protocol="single_answer", run="", raw_label=single.model.map(parameters["models"]))
            iterative = logs.join(logs.source_file.str.extract(parameters["patterns"]["iterative"])).dropna(subset=["task"])
            iterative = iterative.assign(protocol="iterative", raw_label=iterative.run.map(parameters["iterative_models"]))
            observations = pd.concat([single, iterative], ignore_index=True)
            native_paths = logs.loc[logs.source_file.str.match(r"(?:tasks/[^/]+/logs/|iterative_result/)")].source_file
            if set(native_paths) != set(observations.source_file):
                raise ValueError("A native log filename was not recognized")
            if observations.raw_label.isna().any():
                raise ValueError("An observed native model label has no documented mapping")
            observations["native_record"] = observations.member.map(lambda member: archive.read(member).decode("utf-8"))
            observations = observations.loc[observations.native_record.str.strip().ne("")].copy()

            # 2. Some .jsonl files are Python representations containing MATLAB/NumPy
            # objects. Read only top-level scalar fields with the standard tokenizer;
            # never execute response objects or discard the original record text.
            scalars = []
            for row in observations.itertuples():
                fields, depth = {}, 0
                tokens = tokenize.generate_tokens(io.StringIO(row.native_record).readline)
                for token in tokens:
                    if token.string in ("{", "(", "["):
                        depth += 1
                    elif token.string in ("}", ")", "]"):
                        depth -= 1
                    if token.type != tokenize.STRING or depth != 1:
                        continue
                    key = ast.literal_eval(token.string)
                    if key not in {"passed", "iteration"} or next(tokens).string != ":":
                        continue
                    scalar = next(tokens).string
                    if scalar == "np":
                        if next(tokens).string != ".":
                            raise ValueError("Unknown native scalar representation")
                        scalar = next(tokens).string.removesuffix("_")
                    fields[key] = ast.literal_eval(scalar.title() if scalar in {"true", "false"} else scalar)
                    if "passed" in fields and (row.protocol == "single_answer" or "iteration" in fields):
                        break
                flag = fields.get("passed")
                if type(flag) is not bool and flag != parameters["null_pass_status"]["evaluation_failure"]:
                    raise ValueError(f"Unknown or missing pass flag in {row.source_file}")
                scalars.append({"response": float(flag) if type(flag) is bool else None,
                                "iteration": fields.get("iteration", 0)})
            observations[["response", "iteration"]] = pd.DataFrame(scalars, index=observations.index)

            # Four iterative files repeat an earlier record under a different filename.
            # Group by the recorded iteration, requiring the complete records to agree.
            single = observations.loc[observations.protocol.eq("single_answer")].copy()
            single["source_files"] = single.source_file.map(lambda path: [path])
            iterative = observations.loc[observations.protocol.eq("iterative")].copy()
            identity = ["run", "task", "iteration"]
            if iterative.groupby(identity).native_record.nunique().gt(1).any():
                raise ValueError("Conflicting records for a refinement iteration")
            aliases = iterative.groupby(identity, sort=False).source_file.agg(list).rename("source_files").reset_index()
            iterative = iterative.drop_duplicates(identity).merge(aliases, on=identity, validate="one_to_one")
            observations = pd.concat([single, iterative], ignore_index=True)
            observations["response_key"] = observations.index
            observations["test_condition"] = observations.iteration.map(lambda n: f"iteration={int(n)}")
            observations.loc[observations.protocol.eq("single_answer"), "test_condition"] = "single_answer"
            observations["trial"] = pd.to_numeric(observations.trial, errors="coerce").fillna(0).astype(int) + 1

            # 3. Read each task's prompt, response schema, evaluator and input images.
            items = observations[["task"]].drop_duplicates().rename(columns={"task": "item_key"})
            members = files.set_index("source_file").member
            for column, filename in [("prompt", "LLM_prompt.txt"), ("output_structure", "output_structure.py"),
                                     ("evaluator", "evaluate.py")]:
                paths = "tasks/" + items.item_key + "/" + filename
                items[column] = paths.map(members).map(lambda member: archive.read(member).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
            items["prompt"] = items.prompt.str.strip()
            items["content"] = [json.dumps(row, ensure_ascii=False) for row in items[["prompt", "output_structure"]].to_dict("records")]
            images = files.join(files.source_file.str.extract(r"^tasks/(?P<item_key>[^/]+)/images/[^/]+\.(?i:png|jpg|jpeg|svg|gif|webp)$"))
            images = images.dropna(subset=["item_key"])
            images["attachment"] = [{"path": path, "role": "input", "media_type": mimetypes.guess_type(path)[0],
                                      "data": archive.read(member)} for path, member in zip(images.source_file, images.member)]
            attachments = images.groupby("item_key").attachment.agg(list)
            items["attachments"] = items.item_key.map(attachments).map(lambda value: value if isinstance(value, list) else [])
            items["raw_item_id"] = items.item_key
            items["grading_criterion"] = [{"rule": json.dumps({"interpretation": self.grading["rule"], "released_evaluator": code})}
                                           for code in items.evaluator]
            specification = self.grading["verifiers"]["native_pass"]
            items["verifier"] = [ExactMatcher(spec=json.dumps({
                **{k: v for k, v in specification.items() if k != "source_template"},
                "source": specification["source_template"].format(task=task)}, sort_keys=True)) for task in items.item_key]

            # 4. Separate single-answer and iterative configurations of each model.
            subjects = observations[["raw_label", "protocol"]].drop_duplicates().reset_index(drop=True)
            subjects["subject_key"] = subjects.index
            subjects["features"] = [
                {"harness": "EngDesign", "protocol": row.protocol,
                 **({"reasoning_effort": parameters["reasoning_effort"][row.raw_label]}
                    if row.raw_label in parameters["reasoning_effort"] else {})}
                for row in subjects.itertuples()
            ]
            responses = observations.merge(subjects[["raw_label", "protocol", "subject_key"]],
                                           on=["raw_label", "protocol"], validate="many_to_one").rename(columns={"task": "item_key"})

            # 5. Join full recorded refinement outputs and feedback by native iteration.
            sections = files.loc[files.source_file.str.startswith("iterative_result/") &
                                 files.source_file.str.endswith(("_responses.txt", "_evaluations.txt"))].copy()
            sections["text"] = sections.member.map(lambda member: archive.read(member).decode("utf-8"))
            sections["group"] = sections.source_file.str.rsplit("/", n=1).str[0]
            sections["kind"] = sections.source_file.str.extract(r"_(responses|evaluations)\.txt$")[0]
            chunks = sections.text.str.extractall(parameters["patterns"]["sections"]).reset_index(level="match", drop=True)
            chunks = chunks.join(sections[["group", "kind", "source_file"]])
            chunks["iteration"] = chunks.iteration.astype(int)
            if chunks.duplicated(["group", "kind", "iteration"]).any():
                raise ValueError("Ambiguous refinement trace sections")
            responses["group"] = responses.source_file.str.rsplit("/", n=1).str[0]
            for kind in ["responses", "evaluations"]:
                part = chunks.loc[chunks.kind.eq(kind), ["group", "iteration", "section", "source_file"]]
                part = part.rename(columns={"section": kind, "source_file": kind + "_source"})
                responses = responses.merge(part, on=["group", "iteration"], how="left", validate="many_to_one")
            responses = responses.astype(object).where(responses.notna(), None)
            trace_columns = ["source_files", "native_record", "responses", "responses_source", "evaluations", "evaluations_source"]
            traces = responses[["response_key"]].copy()
            traces["trace"] = [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in responses[trace_columns].to_dict("records")]

        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    EngDesign(__file__).main_from_args()
