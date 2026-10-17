"""Reanalyze pinned released experiments without invoking any agent or model.

This is a study-format prototype, not a canonical Measurement DB importer.
Run from any directory: python scripts/analyze_measurements/intervention_pilot.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analyze_measurements import intervention_pilot_tables


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/intervention-pilot"
ACTORS = {
    "kimi-k2.5": "Kimi-K2.5",
    "qwen3-coder-30b": "Qwen3Coder-30B",
    "sonnet4.5": "Sonnet-4.5",
}
SETUPS = {
    "bashonly": "BashOnly", "atomic-common": "Atomic", "bash-search": "NLSearch",
    "bash-hypo": "HypoTrack", "bash-scratchpad": "ScratchPad", "py-interface": "Python",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch(lock, raw, offline):
    paths = {}
    for source in lock["files"]:
        path = raw / source["local_path"]
        if not path.exists():
            require(not offline, f"Missing cached input: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            with requests.get(source["url"], stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            require(digest(temporary) == source["sha256"], f"Source hash changed: {path}")
            temporary.replace(path)
        require(path.stat().st_size == source["bytes"], f"Source size changed: {path}")
        require(digest(path) == source["sha256"], f"Source hash changed: {path}")
        paths[source["id"]] = path
    return paths


def onlycodes(path):
    source = pd.read_csv(path)
    keys = ["benchmark", "agent", "arm", "instance_id", "seed", "run"]
    require(len(source) == 3474 and not source.duplicated(keys).any(), "Onlycodes row identity changed")
    require(set(source.verdict) == {"PASS", "FAIL"}, "Unexpected Onlycodes verdict")
    require(set(source.seed) == {1, 2, 3} and set(source.run) == {1}, "Repeat design changed")
    versions = source.groupby(["benchmark", "agent", "instance_id", "seed"]).agent_version.nunique()
    require(versions.eq(1).all(), "Versions differ across arms within a task/repeat")
    data = source.rename(columns={"instance_id": "task_id", "seed": "source_repeat",
                                  "agent_version": "harness_version", "arm": "source_arm"}).copy()
    data["study_id"] = "onlycodes"
    data["experiment_id"] = "onlycodes:" + data.benchmark
    data["model"] = data.agent.map({"claude": "claude-sonnet-4-6", "codex": "gpt-5.5"})
    data["harness"] = data.agent_surface
    data["arm_id"] = data.source_arm.replace({"tool_rich": "baseline", "onlycode": "code_only"})
    data["grade_observed"] = data.verdict.map({"PASS": 1.0, "FAIL": 0.0})
    data["success_all_attempts"] = data.grade_observed
    data["source_locator"] = [f"paper/data/raw/all_results.csv#data-row={i + 1}" for i in range(len(data))]
    for (benchmark, agent, arm), group in data.groupby(["benchmark", "agent", "arm_id"]):
        n = 100 if benchmark == "swebench" else 93
        require(len(group) == n * 3 and group.task_id.nunique() == n, "Onlycodes coverage changed")
        require(group.groupby("task_id").source_repeat.apply(set).eq({1, 2, 3}).all(), "Missing repeat")
    return data


def tool_architecture(path):
    with zipfile.ZipFile(path) as archive:
        table = json.loads(archive.read("data/processed/table2/inputs.json"))["per_instance"]
        runs = json.loads(archive.read("data/processed/figure4/inputs.json"))["trajectories"]
        members = set(archive.namelist())
        outcomes = {}
        for row in table:
            require(row["n_repeats"] == len(row["resolved_by_repeat"]) == 10, "Repeat design changed")
            require(row["n_pass"] == sum(row["resolved_by_repeat"]), "Outcome summary inconsistent")
            for repeat, value in enumerate(row["resolved_by_repeat"]):
                key = (row["actor"], row["setup"], row["instance"], repeat)
                require(key not in outcomes and isinstance(value, bool), "Duplicate or invalid outcome")
                outcomes[key] = value
        require(len(outcomes) == len(runs) == 11700, "Main cohort size changed")
        for index, row in enumerate(runs):
            key = (row["actor"], row["setup"], row["instance"], row["repeat"])
            require(key in outcomes, "Unknown trajectory")
            require(row["resolved"] is None or isinstance(row["resolved"], bool), "Invalid resolved status")
            require(outcomes[key] == (row["resolved"] is True), "Outcome policies no longer reconcile")
            suffix = f"__{row['repeat']}" if row["repeat"] else ""
            trace = (f"data/raw_traj/main_live/{ACTORS[row['actor']]}/{SETUPS[row['setup']]}/"
                     f"{row['instance']}{suffix}.json")
            # The upstream archive spells the notes setup Scratchpad.
            trace = trace.replace("/ScratchPad/", "/Scratchpad/")
            require(trace in members, f"Missing main-cohort trace: {trace}")
            row["success_all_attempts"] = float(outcomes[key])
            row["source_locator"] = f"data/processed/figure4/inputs.json#/trajectories/{index}"
            row["trace_member"] = trace
    data = pd.DataFrame(runs).rename(columns={"actor": "model", "setup": "arm_id",
                                            "instance": "task_id", "repeat": "source_repeat"})
    data["study_id"] = "tool_architecture"
    data["experiment_id"] = "tool_architecture:main_live"
    data["grade_observed"] = data.resolved.map({True: 1.0, False: 0.0})
    data["harness"] = np.where(data.arm_id.eq("py-interface"), "adapted_simple_agent", "trae_derived")
    data["harness_version"] = None  # Repository release revision is not the historical run version.
    keys = ["model", "arm_id", "task_id", "source_repeat"]
    require(not data.duplicated(keys).any(), "Duplicate trajectory")
    require(set(data.model) == set(ACTORS) and set(data.arm_id) == set(SETUPS), "Unexpected design")
    for _, group in data.groupby(["model", "arm_id"]):
        require(group.task_id.nunique() == 65 and len(group) == 650, "Unbalanced main cohort")
        require(group.groupby("task_id").source_repeat.apply(set).eq(set(range(10))).all(), "Missing repeat")
    return data


def add_configuration_ids(data):
    columns = ["study_id", "model", "harness", "harness_version", "arm_id"]
    def identity(row):
        value = {key: None if pd.isna(row[key]) else row[key] for key in columns}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    data["configuration_id"] = data.apply(identity, axis=1)
    configurations = data[columns + ["configuration_id"]].drop_duplicates()
    require(not configurations.configuration_id.duplicated().any(), "Configuration identity collision")
    # IDs are study-local handles for the available description, not canonical subject identities.
    return configurations


def analyze(data, bootstrap_samples):
    comparisons, effects, task_effects = [], [], []
    metrics = ["success_all_attempts", "input_tokens", "output_tokens"]
    metrics += ["cost_usd_adjusted"] if data.study_id.iloc[0] == "onlycodes" else ["steps"]
    for (experiment, model), panel in data.groupby(["experiment_id", "model"], sort=True):
        baseline = "baseline" if panel.study_id.iloc[0] == "onlycodes" else "bashonly"
        pairs = [(baseline, arm) for arm in sorted(set(panel.arm_id) - {baseline})]
        if baseline == "baseline":
            pairs.append(("bash_only", "code_only"))
        for reference, target in pairs:
            cid = f"{experiment}:{model}:{reference}->{target}"
            left = panel[panel.arm_id.eq(reference)]
            right = panel[panel.arm_id.eq(target)]
            require(set(left.task_id) == set(right.task_id), f"Task mismatch in {cid}")
            comparisons.append({
                "comparison_id": cid, "experiment_id": experiment, "model": model,
                "reference_arm_id": reference, "target_arm_id": target,
                "reference_configuration_ids": sorted(left.configuration_id.unique()),
                "target_configuration_ids": sorted(right.configuration_id.unique()),
                "matching_unit": "task_id", "repeat_pairing_assumed": False,
                "origin": "pilot_analyst_defined", "n_tasks": left.task_id.nunique(),
            })
            for metric in metrics:
                # A metric may be absent even when the success grade is known.
                # Require every released repeat in both arms for a task's metric contrast.
                left_complete = left.groupby("task_id")[metric].apply(lambda x: x.notna().all())
                right_complete = right.groupby("task_id")[metric].apply(lambda x: x.notna().all())
                eligible = left_complete & right_complete
                joined = pd.concat([left.groupby("task_id")[metric].mean().rename("reference"),
                                    right.groupby("task_id")[metric].mean().rename("target")], axis=1)
                joined = joined.loc[eligible]
                require(len(joined) > 1, f"Insufficient complete tasks for {cid}/{metric}")
                require(np.isfinite(joined.to_numpy()).all(), "Nonfinite metric")
                delta = (joined.target - joined.reference).to_numpy()
                # Resample tasks as blocks, preserving both arms and all their repetitions.
                rng = np.random.default_rng(int(hashlib.sha256((cid + metric).encode()).hexdigest()[:8], 16))
                samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
                effects.append({"comparison_id": cid, "metric": metric, "n_tasks": len(delta),
                                "tasks_excluded_missing_metric": int((~eligible).sum()),
                                "reference_mean": joined.reference.mean(), "target_mean": joined.target.mean(),
                                "delta_target_minus_reference": delta.mean(), "ci95_low": low, "ci95_high": high,
                                "tasks_increased": int((delta > 0).sum()),
                                "tasks_decreased": int((delta < 0).sum()), "tasks_tied": int((delta == 0).sum())})
                for task, row in joined.iterrows():
                    task_effects.append({"comparison_id": cid, "task_id": task, "metric": metric,
                                         "reference_mean": row.reference, "target_mean": row.target,
                                         "delta": row.target - row.reference})
    return pd.DataFrame(comparisons), pd.DataFrame(effects), pd.DataFrame(task_effects)


def main(package_study=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/intervention_pilot")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--package-root", type=Path, default=ROOT / "benchmarks",
                        help="Write study packages here; shared catalogs remain under --output-dir/release_v2/catalog")
    args = parser.parse_args()
    require(args.bootstrap_samples >= 1000, "Use at least 1000 bootstrap samples")
    lock = json.loads((DOCS / "sources.lock.json").read_text())
    paths = fetch(lock, args.output_dir / "raw", args.offline)
    summary = {"format_status": "pilot_only_not_canonical", "bootstrap_samples": args.bootstrap_samples,
               "uncertainty": "Pointwise paired-task percentile bootstrap; no multiplicity adjustment; descriptive pilot.",
               "studies": {}}
    for study, loader, source_id in [("onlycodes", onlycodes, "onlycodes_results"),
                                      ("tool_architecture", tool_architecture, "tool_archive")]:
        data = loader(paths[source_id])
        configs = add_configuration_ids(data)
        comparisons, effects, per_task = analyze(data, args.bootstrap_samples)
        out = args.output_dir / "studies" / study
        out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((ROOT / "benchmarks" / study / "study.json").read_text())
        write_json(out / "study.json", manifest)
        experiments = pd.DataFrame(manifest["experiments"])
        for name, table in [("observations", data), ("configurations", configs),
                            ("comparisons", comparisons), ("task_effects", per_task), ("experiments", experiments)]:
            table.to_parquet(out / f"{name}.parquet", index=False)
            require(len(pd.read_parquet(out / f"{name}.parquet")) == len(table), f"Write failed: {name}")
        effects.to_csv(out / "effects.csv", index=False)
        coverage = data.groupby(["experiment_id", "model", "arm_id"]).agg(
            observations=("task_id", "size"), tasks=("task_id", "nunique"),
            known_grades=("grade_observed", "count"), configurations=("configuration_id", "nunique"))
        coverage.to_csv(out / "coverage.csv")
        summary["studies"][study] = {"status": "released_results_analyzed", "observations": len(data),
            "analysis_scopes": data.experiment_id.nunique(), "configurations": len(configs),
            "comparisons": len(comparisons), "unknown_observed_grades": int(data.grade_observed.isna().sum()),
            "missing_metric_counts": {column: int(data[column].isna().sum())
                for column in ["input_tokens", "output_tokens", "cost_usd_adjusted", "steps"] if column in data}}
        if study == "tool_architecture":
            consistency = []
            for key, group in data.groupby(["model", "arm_id", "task_id"]):
                passed = int(group.success_all_attempts.sum())
                consistency.append(dict(zip(["model", "arm_id", "task_id"], key)) | {
                    f"consistency_at_{k}": math.comb(passed, k) / math.comb(10, k) if passed >= k else 0.0
                    for k in [5, 7, 9]})
            pd.DataFrame(consistency).groupby(["model", "arm_id"]).mean(numeric_only=True).to_csv(out / "consistency.csv")
    manifest = json.loads((ROOT / "benchmarks/agentarch/study.json").read_text())
    write_json(args.output_dir / "studies/agentarch/study.json", manifest)
    summary["studies"]["agentarch"] = {"status": "design_mapped_original_results_not_located", "observations": 0}
    verification = intervention_pilot_tables.export(args.output_dir, DOCS, paths,
        package_root=args.package_root, package_study=package_study)
    summary["release_v2"] = {"schema_version": verification["schema_version"],
        "validation": verification["status"], "table_rows": verification["table_rows"],
        "unknown_grades": verification["unknown_grades"], "study_packages": verification["study_packages"]}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
