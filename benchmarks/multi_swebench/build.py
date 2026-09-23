#!/usr/bin/env python3
"""Curate pinned Multi-SWE-bench releases using table transformations."""

import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MultiSWEBench(BenchmarkBuild):

    def download(self):
        return self.fetch_sources('tasks', 'runs', 'python_tasks')
    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform captured tasks, outcomes, and patches into linked tables."""
        # 1. Load task definitions, then prepare their text and grading information.
        task_columns = [
            "instance_id", "org", "repo", "number", "resolved_issues", "title", "body",
            "fix_patch", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "f2p_tests", "p2p_tests"
        ]
        task_files = sorted(
            name for name in self.source_files if "/" not in name and name.endswith(".jsonl")
        )
        task_frames = []
        for name in task_files:
            frame = _read_jsonl(self.raw_dir / name).reindex(columns=task_columns)
            task_frames.append(frame.assign(source_file=name))
        tasks = pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame(
            columns=[*task_columns, "source_file"]
        )
        tasks = tasks.astype(object).where(tasks.notna(), None)

        # Some task files identify the pull request using separate org/repo/number fields.
        pr_numbers = tasks.number.astype("Int64").astype("string")
        pr_ids = tasks.org.astype("string") + "__" + tasks.repo.astype("string") + "-" + pr_numbers
        pr_ids = pr_ids.where(tasks.org.map(bool) & tasks.repo.map(bool) & tasks.number.notna())
        tasks["instance_id"] = tasks.instance_id.where(tasks.instance_id.map(bool), pr_ids)
        tasks = tasks.dropna(subset=["instance_id"]).set_index("instance_id")
        duplicate_ids = tasks.index[tasks.index.duplicated()].tolist()
        if duplicate_ids:
            raise ValueError(f"Duplicate task definitions: {duplicate_ids[:5]}")

        # Expand issues into rows, assemble their text, and group it back by task ID.
        issues = tasks.resolved_issues.explode()
        issues = issues.loc[issues.map(lambda value: isinstance(value, dict))]
        issues = pd.json_normalize(issues.tolist(), max_level=0).reindex(
            columns=["title", "body"]
        ).fillna("").astype("string").set_axis(issues.index)
        issues["content"] = (issues.title.str.strip() + "\n" + issues.body.str.strip()).str.strip()
        issue_text = issues.loc[issues.content.ne("")].groupby("instance_id").content.agg("\n\n".join)
        pr_text = tasks[["title", "body"]].fillna("").astype("string")
        pr_text = (pr_text.title.str.strip() + "\n" + pr_text.body.str.strip()).str.strip()
        content = issue_text.reindex(tasks.index).fillna(pr_text)

        # A long table of test-map keys provides the fallback for missing test-name arrays.
        test_names = tasks[["f2p_tests", "p2p_tests"]].reset_index().melt(
            id_vars="instance_id", var_name="kind", value_name="test_name"
        )
        is_mapping = test_names.test_name.map(lambda value: isinstance(value, dict))
        test_names = test_names.loc[is_mapping].explode("test_name").dropna(subset=["test_name"])
        test_names = test_names.sort_values("test_name").groupby(
            ["instance_id", "kind"]
        ).test_name.agg(list).unstack("kind").reindex(index=tasks.index, columns=["f2p_tests", "p2p_tests"])
        failed_tests = tasks.FAIL_TO_PASS.map(_test_list)
        passed_tests = tasks.PASS_TO_PASS.map(_test_list)
        failed_tests = failed_tests.mask(failed_tests.str.len().eq(0) & test_names.f2p_tests.notna(), test_names.f2p_tests)
        passed_tests = passed_tests.mask(passed_tests.str.len().eq(0) & test_names.p2p_tests.notna(), test_names.p2p_tests)
        test_specs = pd.DataFrame({
            "test_patch": tasks.test_patch.replace("", None),
            "fail_to_pass": failed_tests,
            "pass_to_pass": passed_tests,
            "n_pass_to_pass": passed_tests.str.len()
        })
        # Store complete grading artifacts as JSON, with no length or test-count limits.
        verifiers = pd.Series(test_specs.to_dict("records"), index=tasks.index).map(json.dumps)
        has_tests = test_specs.test_patch.notna() | failed_tests.str.len().gt(0) | passed_tests.str.len().gt(0)
        reference_patches = tasks.fix_patch.where(tasks.fix_patch.map(bool), tasks.patch)
        tasks = tasks.assign(
            content=content.where(content.ne(""), None),
            reference_answer=reference_patches.replace("", None),
            verifier=verifiers.where(has_tests, None)
        )[["content", "reference_answer", "verifier", "source_file"]]

        # SWE-bench Verified supplies missing Python prompts without replacing released text.
        path = self.raw_dir / "swebench_verified.parquet"
        if path.is_file() and path.stat().st_size:
            fallback = pd.read_parquet(path, columns=["instance_id", "problem_statement"])
            fallback = fallback.dropna(subset=["instance_id"]).set_index("instance_id")
            duplicate_ids = fallback.index[fallback.index.duplicated()].tolist()
            if duplicate_ids:
                raise ValueError(f"Duplicate supplementary prompts: {duplicate_ids[:5]}")
            prompts = fallback.problem_statement.map(
                lambda text: text.strip() if isinstance(text, str) and text.strip() else None
            )
            tasks["content"] = tasks.content.fillna(prompts)

        # 2. Load run metadata and reshape outcome lists into response rows.
        files = pd.DataFrame({"source_file": sorted(
            name for name in self.source_files if name.startswith("results/")
        )})
        files["original_path"] = files.source_file.astype("string").str.replace(
            r"_x([0-9a-f]{2})_", lambda match: chr(int(match[1], 16)), regex=True
        )
        files = files.sort_values(["original_path", "source_file"]).reset_index(drop=True)
        files["run_id"] = files.original_path.str.removeprefix("results/").str.removesuffix(".json")
        files["prediction_file"] = "preds/" + files.run_id.str.replace(
            r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", regex=True
        ) + ".jsonl"
        records = [json.loads((self.raw_dir / name).read_text()) for name in files.source_file]
        runs = pd.json_normalize(records, max_level=0).reindex(
            columns=["resolved", "resolved_ids", "unresolved_ids", "unresolved"]
        ).join(files[["run_id", "source_file"]]).assign(run_order=files.index)
        runs = runs.astype(object).where(runs.notna(), None)
        duplicate_runs = runs.loc[runs.run_id.duplicated(), "run_id"].tolist()
        if duplicate_runs:
            raise ValueError(f"Duplicate result runs: {duplicate_runs[:5]}")

        # Run names encode <language>__<optional date>_<optional agent>_<model>.
        run_names = runs.run_id.str.split("__", n=1, expand=True).reindex(columns=[0, 1])
        run_names.columns = ["lang", "label"]
        dated_names = run_names.label.astype("string").str.extract(r"^(?P<date>\d{8})_(?P<label>.+)$")
        labels = dated_names.label.fillna(run_names.label)
        configurations = labels.str.extract(r"^(?:(?P<agent>[^_]*)_)?(?P<model>.*)$")
        runs = runs.assign(
            lang=run_names.lang.astype("string"),
            model=configurations.model,
            agent=configurations.agent,
            access_date=dated_names.date.str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True),
            passed=runs.resolved.where(runs.resolved.str.len().gt(0), runs.resolved_ids),
            failed=runs.unresolved_ids.where(runs.unresolved_ids.str.len().gt(0), runs.unresolved)
        )
        responses = runs.melt(
            id_vars=["run_order", "run_id", "source_file", "lang"], value_vars=["passed", "failed"],
            var_name="outcome", value_name="instance_id"
        ).explode("instance_id").dropna(subset=["instance_id"])
        responses["response"] = responses.outcome.map({"passed": 1.0, "failed": 0.0})
        # Match org/repo:pr-123 result IDs to org__repo-123 task IDs.
        responses["instance_id"] = responses.instance_id.str.replace(
            r"^([^/]+)/([^:]+):pr-(\d+)$", r"\1__\2-\3", regex=True
        )
        duplicates = responses.loc[
            responses.duplicated(["run_id", "instance_id"], keep=False),
            ["source_file", "instance_id", "response"]
        ]
        if not duplicates.empty:
            raise ValueError(f"Duplicate or conflicting outcomes:\n{duplicates.head().to_string(index=False)}")
        responses = responses.sort_values(["run_order", "instance_id"]).reset_index(drop=True)
        responses["response_key"] = responses.index

        # 3. Construct subjects from runs that have observations, retaining each configuration's earliest date.
        subjects = runs.loc[runs.run_id.isin(responses.run_id)].copy()
        subjects["access_date"] = subjects.groupby(["model", "agent"], dropna=False).access_date.transform("min")
        subjects = subjects.astype(object).where(subjects.notna(), None)
        subjects = subjects.assign(
            subject_key=subjects.run_id,
            raw_label=subjects.model,
            features=subjects.agent.map(lambda agent: {"harness": agent} if agent else None)
        )[["subject_key", "raw_label", "features", "access_date"]]

        # 4. Join distinct task/language pairs to their definitions, then link response rows to items.
        items = responses[["instance_id", "lang"]].drop_duplicates().merge(
            tasks[["content", "reference_answer", "verifier"]],
            on="instance_id", how="left", sort=False, validate="many_to_one"
        )
        missing = items.content.isna() | items.verifier.isna()
        if missing.any():
            ids = items.loc[missing, "instance_id"].head(5).tolist()
            raise ValueError(f"Attempted tasks lack content or grading information: {ids}")
        items = items.astype(object).where(items.notna(), None)
        criteria = items[["reference_answer"]].assign(rule=self.grading["rule"])
        items = items.assign(
            item_key=range(len(items)),
            raw_item_id=items.instance_id,
            features=items[["lang"]].to_dict("records"),
            grading_criterion=criteria.to_dict("records"),
            verifier=items.verifier.map(ExactMatcher)
        )
        responses = responses.merge(
            items[["instance_id", "lang", "item_key"]],
            on=["instance_id", "lang"], how="left", sort=False, validate="many_to_one"
        ).rename(columns={"run_id": "subject_key"})

        # 5. Load patches for evaluated runs, retaining complete nonempty text.
        prediction_frames = []
        selected_files = files.loc[files.run_id.isin(responses.subject_key), ["run_id", "prediction_file"]]
        for run_id, name in selected_files.itertuples(index=False, name=None):
            frame = _read_jsonl(self.raw_dir / name).reindex(columns=["instance_id", "model_patch"])
            prediction_frames.append(frame.assign(run_id=run_id))
        patches = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(
            columns=["instance_id", "model_patch", "run_id"]
        )
        has_text = patches.model_patch.map(lambda value: isinstance(value, str))
        nonempty = patches.model_patch.where(has_text, "").astype("string").str.strip().ne("")
        patches = patches.loc[patches.instance_id.notna() & nonempty].set_index(["run_id", "instance_id"])
        duplicate_keys = patches.index[patches.index.duplicated()].tolist()
        if duplicate_keys:
            raise ValueError(f"Duplicate nonempty prediction patches: {duplicate_keys[:5]}")

        # 6. Attach available patches to responses; absent patches leave the recorded grade intact.
        traces = responses.merge(
            patches[["model_patch"]], left_on=["subject_key", "instance_id"],
            right_index=True, how="left", sort=False, validate="one_to_one"
        )
        traces = traces.loc[traces.model_patch.notna(), ["response_key", "model_patch"]].rename(
            columns={"model_patch": "trace"}
        )
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


def _test_list(value) -> list[str]:
    """Read test-name arrays, including the JSON-encoded arrays in Python tasks."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return list(map(str, value)) if isinstance(value, list) else []


def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read one source file; an absent patch file or LFS pointer has no records."""
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    with path.open("rb") as stream:
        if stream.readline().startswith(b"version https://git-lfs.github.com/spec/v1"):
            # One archived prediction file contains only this pointer, not data.
            return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    except ValueError as exc:
        raise ValueError(f"Cannot read JSONL source {path}: {exc}") from exc


if __name__ == "__main__":
    MultiSWEBench(__file__).main_from_args()
