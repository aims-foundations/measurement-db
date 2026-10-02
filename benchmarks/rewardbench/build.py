"""Tabulate recorded RewardBench preferences against their original task versions."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class RewardBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Read the original task-bank versions and the column-oriented result files.
        banks = {}
        for path in sorted(self.raw_dir.glob(layout["task_banks"])):
            name = str(path.relative_to(self.raw_dir))
            tasks = pd.read_parquet(path)
            tasks["task_record"] = tasks.to_dict("records")
            tasks["task_file"] = name
            tasks["task_row"] = tasks.index
            banks[name] = tasks
        attempts, configurations = [], []
        results_root = self.raw_dir / layout["results"]
        for path in sorted(results_root.rglob("*.json")):
            native = json.loads(path.read_text())
            frame = pd.DataFrame(native)
            source_file = str(path.relative_to(self.raw_dir))
            frame["native_record"] = [json.loads(json.dumps(record, ensure_ascii=False),
                parse_constant=lambda value: {"native_nonfinite_number": value})
                for record in frame.to_dict("records")]
            frame["source_file"] = source_file
            frame["source_row"] = frame.index

            # 2. Require one complete bank to match every ID, subset and full input pair.
            matching = []
            for name, tasks in banks.items():
                if len(frame) != len(tasks) or frame.subset.tolist() != tasks.subset.tolist():
                    continue
                if "id" in frame and frame.id.tolist() != tasks.id.tolist():
                    continue
                matches = pd.Series(True, index=frame.index)
                for native_field, task_field in (("text_chosen", "chosen"), ("text_rejected", "rejected")):
                    matches &= pd.Series([
                        (len(actual) == 2 and actual[0].get("role") == "user"
                         and actual[1].get("role") == "assistant"
                         and actual[0]["content"].strip() == prompt.strip()
                         and actual[1]["content"].strip() == answer.strip())
                        if isinstance(actual, list) else (prompt.strip() in actual and answer.strip() in actual)
                        for actual, prompt, answer in zip(frame[native_field], tasks.prompt, tasks[task_field])],
                        index=frame.index)
                if matches.all():
                    matching.append(name)
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one complete historical task bank for {source_file}: {matching}")
            frame["task_file"] = matching[0]
            frame["task_row"] = frame.source_row

            # 3. Preserve native model settings, including the DPO reference model.
            configuration = {key: value for key, value in native.items() if not isinstance(value, list)}
            scoring = "direct"
            if native["model_type"] == "DPO":
                relative = path.relative_to(results_root)
                aggregate = json.loads((self.raw_dir / layout["aggregates"] / relative).read_text())
                free = path.name.endswith("_ref_free.json")
                scoring = "dpo_implicit_ref_free" if free else "dpo_implicit_ref"
                if (aggregate["model"] != native["model"]
                        or aggregate["model_type"] != ("DPO Ref. Free" if free else "DPO")
                        or (aggregate["ref_model"] is None) != free):
                    raise ValueError(f"Conflicting DPO configuration for {source_file}")
                means = frame.groupby("subset").results.mean()
                if any(abs(value - aggregate[subset]) > 1e-12 for subset, value in means.items()):
                    raise ValueError(f"Published DPO summary disagrees with {source_file}")
                configuration["reference_model"] = aggregate["ref_model"]
            frame["subject_key"] = source_file
            frame["scoring"] = scoring
            configurations.append(dict(subject_key=source_file,
                raw_label=parameters["labels"]["subject_prefix"] + native["model"] + " / " + scoring,
                features=dict(**parameters["subject_features"], native_configuration=configuration, scoring=scoring)))
            attempts.append(frame)
        attempts = pd.concat(attempts, ignore_index=True)
        subjects = pd.DataFrame(configurations)

        # 4. Join the selected task versions and retain distinct complete preference pairs.
        tasks = pd.concat(banks.values(), ignore_index=True)
        tasks["content"] = [json.dumps(dict(prompt=row.prompt, candidate_a=row.chosen, candidate_b=row.rejected),
            ensure_ascii=False) for row in tasks.itertuples()]
        items = tasks.drop_duplicates("content").copy()
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.subset + ":" + items.id.astype(str)
        subsets = tasks.groupby("content").subset.agg(lambda values: sorted(set(values)))
        items["features"] = [dict(source_subsets=values) for values in items.content.map(subsets)]
        items["grading_criterion"] = [dict(reference_answer="A", rule=self.grading["rule"])
                                     for _ in range(len(items))]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["reported"], sort_keys=True))
                             for _ in range(len(items))]
        tasks = tasks.merge(items[["content", "item_key"]], on="content", how="left", validate="many_to_one")
        attempts = attempts.merge(tasks[["task_file", "task_row", "task_record", "item_key"]],
            on=["task_file", "task_row"], how="left", validate="many_to_one", indicator=True)
        if not attempts._merge.eq("both").all():
            raise ValueError("A native result lost its task association")
        attempts["response"] = attempts.results.astype(float)
        if not attempts.response.isin([0, 0.5, 1]).all():
            raise ValueError("Unexpected native preference grade")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["test_condition"] = "subset=" + attempts.subset + ";scoring=" + attempts.scoring

        # 5. Preserve original inputs, rewards and verdicts; no reference answer becomes a trace.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            task_file=row.task_file, task_row=row.task_row,
            association=parameters["labels"]["association"], native_record=row.native_record,
            task_record=row.task_record), ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    RewardBench(__file__).main_from_args()
