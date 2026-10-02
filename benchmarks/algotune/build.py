#!/usr/bin/env python3
"""Curate AlgoTune's published final evaluations and complete conversation logs."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AlgoTune(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "site_index", "conversations")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = {name: self.raw_dir / value for name, value in parameters["layout"].items()}

        # 1. Unpivot the provider's task-by-model grid into one row per result.
        summary = json.loads(paths["summary"].read_text())
        observations = (pd.DataFrame.from_dict(summary, orient="index").rename_axis("item_key")
                        .reset_index().melt(id_vars="item_key", var_name="subject_key", value_name="result"))
        observations["speedup"] = observations.result.str["final_speedup"]
        unavailable = observations.speedup.eq("N/A")
        numeric = pd.to_numeric(observations.speedup.mask(unavailable), errors="raise")
        if observations.speedup.isna().any() or not numeric.loc[~unavailable].map(math.isfinite).all():
            raise ValueError("AlgoTune results must contain finite speedups or explicit N/A failures")
        observations["response"] = numeric.ge(float(parameters["grading"]["speedup_threshold"])).astype(float)
        observations["response_key"] = observations.item_key + ":" + observations.subject_key

        # 2. Load task text and its grading implementation without stripping text.
        descriptions = sorted(paths["tasks"].glob("*/description.txt"))
        items = pd.DataFrame({"item_key": [path.parent.name for path in descriptions],
                              "content": [path.read_text() for path in descriptions]})
        items["raw_item_id"] = items.item_key
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in items.index]
        task_code = items.item_key.map(lambda task: (paths["tasks"] / task / f"{task}.py").read_text())
        items["verifier"] = task_code.map(lambda code: ExactMatcher(
            spec=json.dumps({**self.grading["verifiers"]["task"], "task_code": code}, sort_keys=True)))
        if set(items.item_key) != set(observations.item_key):
            raise ValueError("AlgoTune task definitions and published results do not match")

        # 3. Read native HTML logs; retain message markup and final solver text.
        logs = []
        for path in sorted(paths["site"].glob("*.html")):
            if path.name == "index.html":
                continue
            document = BeautifulSoup(path.read_text(), "html.parser")
            messages = [{"role": " ".join(node.get("class", [])[1:]), "html": str(node)}
                        for node in document.select("div.message")]
            best_files = [{"name": block.select_one(".file-name").get_text(),
                           "content": block.select_one("pre.best-code").get_text()}
                          for block in document.select("div.best-file")]
            logs.append({"page": path.name, "source_file": str(path.relative_to(self.raw_dir)),
                         "messages": messages, "best_files": best_files})
        pages = pd.DataFrame(logs)
        observations["page"] = (observations.item_key + "_" +
                                observations.subject_key.map(parameters["model_page_suffix"]) + ".html")
        observations = observations.merge(pages, on="page", how="left", validate="one_to_one")
        if observations.source_file.isna().any() or set(observations.page) != set(pages.page):
            raise ValueError("AlgoTune log pages and task/model results do not match")
        with_logs = observations.loc[observations.messages.map(bool)]
        trace_records = with_logs[["source_file", "speedup", "messages", "best_files"]].to_dict("records")
        traces = pd.DataFrame({"response_key": with_logs.response_key,
                               "trace": [json.dumps(record, ensure_ascii=False, allow_nan=False)
                                         for record in trace_records]})

        # 4. Project linked tables, preserving upstream model labels and scores.
        subjects = observations[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects["features"] = subjects.raw_label.map(lambda model: {"model_identifier": model, "harness": "AlgoTuner"})
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AlgoTune(__file__).main_from_args()
