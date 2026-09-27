"""Tabulate published Lean grading rounds with their original proof diagnostics."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class LLMsLeanFormalization(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]
        repository = self.raw_dir / layout["repository"]

        # 1. Load the published JSONL files and decode their recorded configurations.
        paths = pd.DataFrame({"path": sorted(repository.glob(layout["results"]))})
        tags = paths.path.map(lambda path: path.name).str.extract(layout["result_pattern"])
        if tags.isna().any().any():
            raise ValueError("Unrecognized historical result filename")
        paths = paths.join(tags)
        frames = []
        for row in paths.itertuples():
            frame = pd.read_json(row.path, lines=True, convert_dates=False, dtype=False, precise_float=True)
            frames.append(frame.assign(dataset=row.dataset, model=row.model, policy=row.policy,
                round_limit=int(row.round_limit), source_row=range(len(frame)),
                source_file=str(row.path.relative_to(repository)).replace("_x40_", "@")))
        records = pd.concat(frames, ignore_index=True)
        records["item_key"] = records.source_file + "#" + records.source_row.astype(str)
        lengths = records[["responses", "verification", "verify_time"]].map(len)
        if not lengths.eq(records.round_limit, axis=0).all().all():
            raise ValueError("Released proof, verdict and verification-time arrays must align")

        # 2. Keep the actual theorem and context, with the historical grading scope.
        items = records[["item_key", "dataset", "header", "formal_statement"]].copy()
        items["raw_item_id"] = items.item_key
        items["content"] = items.header + "\n" + items.formal_statement
        items["features"] = [dict(dataset=row.dataset, input_scope=parameters["labels"]["input_scope"])
            for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in range(len(items))]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["lean"], sort_keys=True))

        # 3. Expand aligned round arrays; round is a pipeline configuration, not an independent trial.
        records["round"] = records.round_limit.map(lambda count: list(range(1, count + 1)))
        rounds = records.explode(["responses", "verification", "verify_time", "round"], ignore_index=True)
        rounds["subject_key"] = rounds.model + ":" + rounds.policy + ":" + rounds["round"].astype(str)
        subjects = rounds[["subject_key", "model", "policy", "round"]].drop_duplicates()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=row.model,
            policy=row.policy, recorded_round=row.round) for row in subjects.itertuples()]
        rounds["response"] = rounds.verification.eq("Pass").astype(float)
        unknown = ~rounds.verification.eq("Pass") & ~rounds.verification.str.startswith("Fail:")
        rounds.loc[unknown | rounds.responses.str.startswith("ERROR: Generation failed"), "response"] = None
        rounds["response_key"] = rounds.item_key + "#round=" + rounds["round"].astype(str)
        rounds["test_condition"] = "dataset=" + rounds.dataset + ";policy=" + rounds.policy + ";round=" + rounds["round"].astype(str)

        # 4. Preserve full proof text and diagnostics, including dependencies on earlier rounds.
        grouped = rounds.groupby("item_key", sort=False)
        rounds["previous_output"] = grouped.responses.shift()
        rounds["previous_verification"] = grouped.verification.shift()
        rounds["prior_pass"] = grouped.verification.transform(lambda values: values.eq("Pass").cummax().shift(fill_value=False))
        context_columns = ["model_time", "input_tokens", "output_tokens", "name", "goal", "informal_prefix", "split"]
        context = records.reindex(columns=context_columns).astype(object).where(pd.notna(records.reindex(columns=context_columns)), None)
        context.index = records.item_key
        record_metadata = context.to_dict("index")
        traces = rounds[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            round=row.round, round_limit=row.round_limit, output=row.responses, verification=row.verification,
            verify_time=row.verify_time, prior_pass=bool(row.prior_pass),
            previous_output=None if pd.isna(row.previous_output) else row.previous_output,
            previous_verification=None if pd.isna(row.previous_verification) else row.previous_verification,
            record_metadata=record_metadata[row.item_key]), allow_nan=False) for row in rounds.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": rounds[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LLMsLeanFormalization(__file__).main_from_args()
