"""Tabulate KernelBench's original task definitions and released assessments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class KernelBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("samples", "tasks", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        samples = self.raw_dir / parameters["paths"]["samples"]

        # 1. Read standalone kernel evaluations as a table, preserving each JSON record.
        paths = sorted(samples.rglob("kernel.json"))
        records = [json.loads(path.read_text()) for path in paths]
        singles = pd.json_normalize(records, max_level=0).assign(record=records,
            source_file=[str(path.relative_to(self.raw_dir)) for path in paths],
            report_key="kernel", source_aliases=[[] for _ in paths],
            final_exact_matches=[[] for _ in paths], log_metadata=None)
        singles = singles.rename(columns={"correct":"response", "model_name":"api_model_id"})

        # 2. Expand refinement logs. Empty slots are not attempts; copied finals are aliases.
        logs = []
        for path in sorted(samples.rglob("log.json")):
            record = json.loads(path.read_text())
            frame = pd.DataFrame({"report_key":list(record), "record":list(record.values())})
            frame = frame.loc[frame.report_key.str.fullmatch(r"[0-9]+|result") & frame.record.map(bool)].copy()
            exact = [key for key, value in record.items() if key.isdigit() and value and value == record.get("result")]
            representative = max(exact, key=int) if exact else None
            if exact:
                frame = frame.loc[frame.report_key.ne("result")].copy()
            frame["source_aliases"] = [["result"] if key == representative else [] for key in frame.report_key]
            frame["final_exact_matches"] = [sorted(exact, key=int) if key == representative else [] for key in frame.report_key]
            frame["source_file"] = str(path.relative_to(self.raw_dir))
            frame["log_metadata"] = [record["metadata"]] * len(frame)
            frame["response"] = frame.record.map(lambda value: (value.get("eval_result") or {}).get("correctness"))
            frame["hardware"] = frame.record.map(lambda value: ((value.get("eval_result") or {}).get("metadata") or {}).get("hardware"))
            frame["run_name"] = record["metadata"]["run_name"]
            frame["api_model_id"] = None
            logs.append(frame)
        columns = ["source_file", "report_key", "record", "source_aliases", "final_exact_matches", "log_metadata",
            "response", "hardware", "run_name", "api_model_id"]
        observations = pd.concat([singles[columns], *logs], ignore_index=True)

        # 3. Decode source keys and join the historical tasks, never the updated task release.
        keys = observations.source_file.str.extract(
            r"samples/(?P<method>[^/]+)/level(?P<level>[0-9]+)/(?:(?P<feedback>[^/]+_last_only)/)?"
            r"(?P<model_label>[^/]+)/problem_(?P<problem_id>[0-9]+)/sample_(?P<sample_id>[0-9]+)/[^/]+\.json$")
        if keys[["method", "level", "model_label", "problem_id", "sample_id"]].isna().any().any():
            raise ValueError("Unrecognized KernelBench source path")
        keys[["level", "problem_id", "sample_id"]] = keys[["level", "problem_id", "sample_id"]].astype(int)
        keys["model_label"] = keys.model_label.str.lower()
        keys["feedback"] = keys.feedback.map(parameters["feedback_labels"]).fillna("")
        observations = pd.concat([observations, keys], axis=1)
        definitions = pd.concat([pd.read_parquet(path) for path in sorted(
            (self.raw_dir / parameters["paths"]["tasks"]).glob("level_*.parquet"))], ignore_index=True)
        definitions["item_key"] = "level" + definitions.level.astype(str) + "_problem" + definitions.problem_id.astype(str)
        observations = observations.merge(definitions[["level", "problem_id", "item_key"]],
            on=["level", "problem_id"], how="left", validate="many_to_one")
        if observations.item_key.isna().any():
            raise ValueError("A recorded assessment has no original task definition")
        observations = observations.astype(object).where(observations.notna(), None)
        observations["api_model_id"] = observations.api_model_id.replace("", None)

        # 4. Separate method, feedback and recorded model configurations, including unknown endpoints.
        identity = ["method", "feedback", "model_label", "api_model_id"]
        observations["subject_key"] = observations[identity].apply(lambda row: json.dumps(row.tolist()), axis=1)
        subjects = observations[["subject_key", *identity]].drop_duplicates("subject_key")
        subjects["raw_label"] = subjects.model_label.map(parameters["model_labels"])
        if subjects.raw_label.isna().any():
            raise ValueError("A source model label has no declared interpretation")
        subjects["features"] = [dict(**parameters["subject_features"], method=row.method, feedback=row.feedback,
            source_model_label=row.model_label, api_model_id=row.api_model_id,
            api_identifier_status="recorded" if row.api_model_id else "not_recorded") for row in subjects.itertuples()]

        # 5. Keep complete reference code and the source's functional-correctness grading rule.
        definitions = definitions.loc[definitions.item_key.isin(observations.item_key)].copy()
        items = definitions[["item_key", "code"]].rename(columns={"code":"content"})
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(level=int(row.level), problem_id=int(row.problem_id), name=row.name,
            **parameters["item_features"]) for row in definitions.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"])] * len(items)
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["functional_correctness"], sort_keys=True))] * len(items)

        # 6. Link every assessment and full trace to its sample, preserving final-report aliases.
        responses = observations[["subject_key", "item_key", "response"]].copy()
        responses["response_key"] = observations.source_file + "#" + observations.report_key
        responses["trial"] = observations.sample_id.astype(int) + 1
        responses["response"] = responses.response.map(lambda value: float(value) if value is not None else None)
        responses["test_condition"] = [json.dumps(dict(method=row.method, feedback=row.feedback,
            report_key=row.report_key, source_aliases=row.source_aliases, hardware=row.hardware,
            run_name=row.run_name), sort_keys=True) for row in observations.itertuples()]
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, report_key=row.report_key,
            source_aliases=row.source_aliases, final_exact_matches=row.final_exact_matches,
            log_metadata=row.log_metadata, record=row.record), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {"subjects":subjects[["subject_key", "raw_label", "features"]], "items":items,
            "responses":responses, "traces":traces}


if __name__ == "__main__":
    KernelBench(__file__).main_from_args()
