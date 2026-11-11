#!/usr/bin/env python3
"""Before-vs-after intervention comparisons over the production builders' tables.

Reads the comparison list (intervention_comparisons.yaml, next to this file) and each listed
benchmark's formatted_tables/ (subjects, items, responses), and writes to
artifacts/analyze_measurements/intervention_comparisons/:

    comparisons.parquet   one row per comparison: its definition, the subjects each side resolved
                          to, the harness versions pooled on each side, and what else differs
    effects.parquet       one row per comparison x outcome policy: task coverage, the task-averaged
                          change and its 95% task-block bootstrap range
    task_effects.parquet  one row per comparison x outcome policy x task
    manifest.json         input file hashes and run settings

Method:
  * Subjects are selected by features (model, harness, harness_version, reasoning_effort, and the
    keys of subject_features_extra such as tool_mode), never by subject ID. A side pools every
    subject it matches, e.g. several harness versions. Both sides are the same model and share no
    subject; every subject comes from the comparison's own benchmark.
  * also_differs lists features whose values differ between the sides but that neither selector
    names. harness_version is left out when the harness itself differs (versions of different
    programs are not comparable).
  * Outcome policies: observed (ungraded runs skipped) and null_as_fail (ungraded runs count 0).
  * Within each task and side, all graded runs are averaged; seeds are not paired across sides.
    A task enters only when both sides have a graded run; tasks only one side ran and tasks
    dropped for lack of a graded run are counted.
  * The change is the mean over tasks of (target mean - reference mean). Its 95% range is a
    percentile bootstrap that resamples whole tasks, with a seed derived from the comparison ID
    and policy. Ranges are pointwise: there is no multiple-comparison correction, and
    n_comparisons_total is reported for context.
  * Comparisons never mix benchmarks; tasks are matched by item_id.

Usage:
    python scripts/analyze_measurements/intervention_comparisons.py \
        [--config PATH] [--benchmarks-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix(".yaml")
OUT = REPO / "artifacts" / "analyze_measurements" / "intervention_comparisons"

POLICIES = ("observed", "null_as_fail")
ORIGINS = {"paper", "analyst"}
CONFIG_KEYS = {"expected_comparisons", "bootstrap_draws", "outcome_policies", "benchmarks"}
BLOCK_REQUIRED = {"name", "family", "models", "reference", "target", "changes", "origin"}
BLOCK_OPTIONAL = {"harnesses", "item_scope", "source"}
# Subject features read from fixed columns; subject_features_extra keys are added beside them.
SUBJECT_COLUMNS = {"model": "display_name", "harness": "harness",
                   "harness_version": "harness_version", "reasoning_effort": "reasoning_effort"}

S, I, F = pa.string(), pa.int64(), pa.float64()
SCHEMAS = {
    "comparisons": pa.schema([
        pa.field("comparison_id", S, False), pa.field("benchmark_id", S, False),
        pa.field("name", S, False), pa.field("family", S, False), pa.field("model", S, False),
        pa.field("reference_selector", S, False), pa.field("target_selector", S, False),
        pa.field("reference_subject_ids", pa.list_(S), False),
        pa.field("target_subject_ids", pa.list_(S), False),
        pa.field("reference_harness_versions", pa.list_(S), False),
        pa.field("target_harness_versions", pa.list_(S), False),
        pa.field("also_differs", S, False), pa.field("item_scope", S, False),
        pa.field("n_scope_items", I, False), pa.field("changes", S, False),
        pa.field("origin", S, False), pa.field("source", S, True),
    ]),
    "effects": pa.schema([
        pa.field("comparison_id", S, False), pa.field("outcome_policy", S, False),
        pa.field("n_items", I, False), pa.field("n_items_reference", I, False),
        pa.field("n_items_target", I, False), pa.field("n_items_reference_only", I, False),
        pa.field("n_items_target_only", I, False), pa.field("n_items_dropped_ungraded", I, False),
        pa.field("reference_runs", I, False), pa.field("reference_passes", I, False),
        pa.field("target_runs", I, False), pa.field("target_passes", I, False),
        pa.field("reference_mean", F, False), pa.field("target_mean", F, False),
        pa.field("delta", F, False), pa.field("ci95_low", F, False), pa.field("ci95_high", F, False),
        pa.field("n_up", I, False), pa.field("n_down", I, False), pa.field("n_tie", I, False),
        pa.field("bootstrap_draws", I, False), pa.field("bootstrap_seed", S, False),
        pa.field("n_comparisons_total", I, False),
    ]),
    "task_effects": pa.schema([
        pa.field("comparison_id", S, False), pa.field("outcome_policy", S, False),
        pa.field("item_id", S, False), pa.field("raw_item_id", S, False),
        pa.field("reference_runs", I, False), pa.field("reference_mean", F, False),
        pa.field("target_runs", I, False), pa.field("target_mean", F, False),
        pa.field("delta", F, False),
    ]),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    """A feature value as a string, or None when missing (checked with pd.isna, never truthiness)."""
    return None if pd.isna(value) else str(value)


def parse_features(value):
    """'k=v;k=v' -> {k: v}; a missing value gives {}."""
    if pd.isna(value) or value == "":
        return {}
    out = {}
    for part in value.split(";"):
        key, sep, val = part.partition("=")
        require(sep == "=" and key and key not in out, f"Malformed feature string {value!r}")
        out[key] = val
    return out


def null_last(values):
    return sorted(values, key=lambda v: (v is None, v or ""))


# ---------------------------------------------------------------- inputs

def prepare(slug, subjects, items, responses):
    """Flatten subject and item features and check what the comparisons rely on."""
    require(subjects.subject_id.is_unique and items.item_id.is_unique, f"{slug}: duplicate subject or item IDs")
    rows = []
    for row in subjects.itertuples(index=False):
        fixed = {key: text(getattr(row, column)) for key, column in SUBJECT_COLUMNS.items()}
        extra = parse_features(row.subject_features_extra)
        require(not set(fixed) & set(extra), f"{slug}: subject_features_extra repeats a fixed feature")
        rows.append({"subject_id": row.subject_id, **fixed, **extra})
    features = pd.DataFrame(rows).set_index("subject_id")
    item_features = {row.item_id: parse_features(row.item_features) for row in items.itertuples(index=False)}
    raw_ids = items.set_index("item_id").raw_item_id.astype(str)
    require(responses.benchmark_id.eq(slug).all(), f"{slug}: responses from another benchmark")
    require(responses.subject_id.isin(features.index).all() and responses.item_id.isin(raw_ids.index).all(),
            f"{slug}: responses reference unknown subjects or items")
    require(responses.test_condition.isna().all() and responses.interactors.isna().all(),
            f"{slug}: comparisons assume no test conditions or interactors")
    require(responses.response.dropna().isin([0.0, 1.0]).all(), f"{slug}: comparisons need binary responses")
    return {"subjects": features, "items": item_features, "raw_item_ids": raw_ids,
            "responses": responses[["subject_id", "item_id", "trial", "response"]].reset_index(drop=True)}


def load_benchmark(slug, benchmarks_dir):
    table_dir = Path(benchmarks_dir) / slug / "formatted_tables"
    tables, hashes = {}, {}
    for name in ("subjects", "items", "responses"):
        path = table_dir / f"{name}.parquet"
        require(path.exists(), f"{path} is missing; run benchmarks/{slug}/build.py first")
        tables[name] = pd.read_parquet(path)
        hashes[f"{slug}/{name}.parquet"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return prepare(slug, **tables), hashes


def expand(config):
    """One spec per (block, model, harness); checks the block layout and the expected count."""
    require(set(config) == CONFIG_KEYS, f"Config keys must be {sorted(CONFIG_KEYS)}")
    require(set(config["outcome_policies"]) <= set(POLICIES) and config["outcome_policies"],
            f"Outcome policies must come from {POLICIES}")
    require(int(config["bootstrap_draws"]) >= 1000, "Use at least 1000 bootstrap draws")
    specs = []
    for slug, blocks in config["benchmarks"].items():
        for block in blocks:
            where = f"{slug}/{block.get('name')}"
            require(BLOCK_REQUIRED <= set(block) <= BLOCK_REQUIRED | BLOCK_OPTIONAL, f"{where}: unexpected keys")
            require(block["origin"] in ORIGINS, f"{where}: origin must be one of {sorted(ORIGINS)}")
            require(block["models"] and block["changes"], f"{where}: needs models and changes")
            require(all(set(change) == {"target", "operation", "value"} for change in block["changes"]),
                    f"{where}: each change needs exactly target, operation and value")
            for side in ("reference", "target"):
                require(isinstance(block[side], dict) and "model" not in block[side],
                        f"{where}: {side} must be a feature mapping without model")
                require(not (block.get("harnesses") and "harness" in block[side]),
                        f"{where}: harness is given both per side and in harnesses")
            require(block["reference"] != block["target"], f"{where}: the two sides are identical")
            for model in block["models"]:
                for harness in block.get("harnesses") or [None]:
                    pinned = {"model": model, **({"harness": harness} if harness else {})}
                    specs.append({
                        "comparison_id": "/".join([slug, block["name"], model] + ([harness] if harness else [])),
                        "benchmark_id": slug, "name": block["name"], "family": block["family"], "model": model,
                        "reference": {**pinned, **block["reference"]}, "target": {**pinned, **block["target"]},
                        "item_scope": block.get("item_scope") or {}, "changes": block["changes"],
                        "origin": block["origin"], "source": block.get("source"),
                    })
    ids = [spec["comparison_id"] for spec in specs]
    require(len(ids) == len(set(ids)), "Duplicate comparison IDs")
    require(len(specs) == config["expected_comparisons"],
            f"Config expands to {len(specs)} comparisons, expected {config['expected_comparisons']}")
    return specs


# ---------------------------------------------------------------- one comparison

def select(subjects, selector, where):
    unknown = set(selector) - set(subjects.columns)
    require(not unknown, f"{where}: unknown subject features {sorted(unknown)}")
    mask = pd.Series(True, index=subjects.index)
    for key, value in selector.items():
        mask &= subjects[key].isna() if value is None else subjects[key].eq(str(value))
    ids = sorted(subjects.index[mask])
    require(ids, f"{where}: {selector} matches no subject")
    return ids


def scope_items(items, scope, where):
    if scope:
        known = set().union(*(features.keys() for features in items.values()))
        require(set(scope) <= known, f"{where}: unknown item features {sorted(set(scope) - known)}")
    kept = {item for item, features in items.items()
            if all(features.get(key) == str(value) for key, value in scope.items())}
    require(kept, f"{where}: item scope {scope} matches no item")
    return kept


def also_differs(subjects, reference, target, named):
    harness_differs = set(subjects.loc[reference, "harness"]) != set(subjects.loc[target, "harness"])
    out = []
    for key in subjects.columns:
        if key in named or (key == "harness_version" and harness_differs):
            continue
        left = set(subjects.loc[reference, key].map(text))
        right = set(subjects.loc[target, key].map(text))
        if left != right:
            out.append({"feature": key, "reference": null_last(left), "target": null_last(right)})
    return out


def task_means(responses, subject_ids, items, policy):
    """Items the side ran at all, and per-item mean / runs / passes over its graded runs."""
    panel = responses[responses.subject_id.isin(subject_ids) & responses.item_id.isin(items)]
    grade = panel.response.fillna(0.0) if policy == "null_as_fail" else panel.response
    graded = panel.assign(grade=grade).dropna(subset=["grade"]).groupby("item_id").grade
    return set(panel.item_id), pd.DataFrame({"mean": graded.mean(), "runs": graded.size(), "passes": graded.sum()})


def bootstrap_seed(comparison_id, policy):
    return int(hashlib.sha256(f"{comparison_id}|{policy}".encode()).hexdigest()[:16], 16)


def bootstrap_range(delta, draws, seed):
    """Percentile 95% range of the mean change, resampling whole tasks."""
    rng = np.random.default_rng(seed)
    means = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compare(spec, data, policies, draws):
    cid, subjects = spec["comparison_id"], data["subjects"]
    reference = select(subjects, spec["reference"], f"{cid} reference")
    target = select(subjects, spec["target"], f"{cid} target")
    require(not set(reference) & set(target), f"{cid}: the two sides share subjects")
    require(subjects.loc[reference + target, "model"].eq(spec["model"]).all(), f"{cid}: model differs across sides")
    items = scope_items(data["items"], spec["item_scope"], cid)
    versions = lambda ids: sorted({v for v in subjects.loc[ids, "harness_version"].map(text) if v is not None})
    definition = {
        "comparison_id": cid, "benchmark_id": spec["benchmark_id"], "name": spec["name"],
        "family": spec["family"], "model": spec["model"],
        "reference_selector": json.dumps(spec["reference"], sort_keys=True),
        "target_selector": json.dumps(spec["target"], sort_keys=True),
        "reference_subject_ids": reference, "target_subject_ids": target,
        "reference_harness_versions": versions(reference), "target_harness_versions": versions(target),
        "also_differs": json.dumps(also_differs(subjects, reference, target, set(spec["reference"]) | set(spec["target"]))),
        "item_scope": json.dumps(spec["item_scope"], sort_keys=True), "n_scope_items": len(items),
        "changes": json.dumps(spec["changes"]), "origin": spec["origin"], "source": spec["source"],
    }
    effects, tasks = [], []
    for policy in policies:
        ref_seen, ref = task_means(data["responses"], reference, items, policy)
        tgt_seen, tgt = task_means(data["responses"], target, items, policy)
        usable = sorted(set(ref.index) & set(tgt.index))
        require(len(usable) >= 2, f"{cid}/{policy}: fewer than two tasks graded on both sides")
        ref, tgt = ref.loc[usable], tgt.loc[usable]
        delta = tgt["mean"].to_numpy() - ref["mean"].to_numpy()
        seed = bootstrap_seed(cid, policy)
        low, high = bootstrap_range(delta, draws, seed)
        effects.append({
            "comparison_id": cid, "outcome_policy": policy, "n_items": len(usable),
            "n_items_reference": len(ref_seen), "n_items_target": len(tgt_seen),
            "n_items_reference_only": len(ref_seen - tgt_seen), "n_items_target_only": len(tgt_seen - ref_seen),
            "n_items_dropped_ungraded": len(ref_seen & tgt_seen) - len(usable),
            "reference_runs": int(ref["runs"].sum()), "reference_passes": int(ref["passes"].sum()),
            "target_runs": int(tgt["runs"].sum()), "target_passes": int(tgt["passes"].sum()),
            "reference_mean": float(ref["mean"].mean()), "target_mean": float(tgt["mean"].mean()),
            "delta": float(delta.mean()), "ci95_low": low, "ci95_high": high,
            "n_up": int((delta > 0).sum()), "n_down": int((delta < 0).sum()), "n_tie": int((delta == 0).sum()),
            "bootstrap_draws": draws, "bootstrap_seed": str(seed),
        })
        tasks += [{"comparison_id": cid, "outcome_policy": policy, "item_id": item,
                   "raw_item_id": data["raw_item_ids"][item],
                   "reference_runs": int(ref.at[item, "runs"]), "reference_mean": float(ref.at[item, "mean"]),
                   "target_runs": int(tgt.at[item, "runs"]), "target_mean": float(tgt.at[item, "mean"]),
                   "delta": float(tgt.at[item, "mean"] - ref.at[item, "mean"])} for item in usable]
    return definition, effects, tasks


# ---------------------------------------------------------------- run

def run(config, datasets):
    """config: the parsed YAML; datasets: {benchmark_id: prepare(...) output}."""
    specs = expand(config)
    policies, draws = list(config["outcome_policies"]), int(config["bootstrap_draws"])
    rows = {name: [] for name in SCHEMAS}
    for spec in specs:
        require(spec["benchmark_id"] in datasets, f"No tables loaded for {spec['benchmark_id']}")
        definition, effects, tasks = compare(spec, datasets[spec["benchmark_id"]], policies, draws)
        rows["comparisons"].append(definition)
        rows["effects"] += effects
        rows["task_effects"] += tasks
    for effect in rows["effects"]:
        effect["n_comparisons_total"] = len(specs)
    frames = {name: pd.DataFrame(rows[name], columns=SCHEMAS[name].names) for name in SCHEMAS}
    for name, frame in frames.items():
        required = [field.name for field in SCHEMAS[name] if not field.nullable]
        require(not frame[required].isna().any().any(), f"{name}: missing values in required columns")
    return frames


def write(frames, out_dir, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        table = pa.Table.from_pandas(frame, schema=SCHEMAS[name], preserve_index=False)
        pq.write_table(table, out_dir / f"{name}.parquet")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--benchmarks-dir", type=Path, default=REPO / "benchmarks")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    datasets, hashes = {}, {}
    for slug in config["benchmarks"]:
        datasets[slug], table_hashes = load_benchmark(slug, args.benchmarks_dir)
        hashes.update(table_hashes)
    frames = run(config, datasets)
    write(frames, args.out, {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "input_sha256": hashes, "bootstrap_draws": int(config["bootstrap_draws"]),
        "outcome_policies": list(config["outcome_policies"]), "n_comparisons": len(frames["comparisons"]),
        "uncertainty": "pointwise 95% percentile task-block bootstrap; no multiple-comparison correction",
    })
    effects = frames["effects"]
    print(f"{len(frames['comparisons'])} comparisons -> {args.out}")
    for policy, panel in effects.groupby("outcome_policy"):
        clear = int(((panel.ci95_low > 0) | (panel.ci95_high < 0)).sum())
        print(f"  {policy}: {clear} of {len(panel)} 95% ranges exclude 0")


if __name__ == "__main__":
    main()
