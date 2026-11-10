#!/usr/bin/env python3
"""Read-only checks of the reviewed release against the cached released trial table."""
import csv
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    check_source_claims,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "harness_buy_swe"
MODELS = {"opus5": "claude-opus-5", "ds-v4-flash": "deepseek-v4-flash", "ds-v4-pro": "deepseek-v4-pro",
          "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B",
          "glm-5.3-flash": "GLM-5.3-Flash", "hy4-preview": "HY4-Preview"}
HARNESSES = {"claude-code": "claude_code", "mini-swe-agent": "mini_swe_agent", "opencode": "opencode",
             "dsh": "deepseek_harness"}
INFRA = {"NetworkConnectionError", "AgentSetupTimeoutError", "RuntimeError", "CancelledError",
         "OpenCodeStartupHang", "OpenCodeStartupError", "AgentVerifierOverlap", "VerifierTimeoutError",
         "verifier_incomplete", "VerifierUvMissing", "RewardFileNotFoundError", "ApiQuotaOrAuthError",
         "AgentIncomplete", "cc_api_error"}


def features(value):
    return dict(part.split("=", 1) for part in value.split(";")) if value else {}


def no_web(arm):
    return arm.endswith("-nw") or "-nw-" in arm or arm in {"B-stock", "B-ac"}


def released_slots():
    """Re-derive one released trial per (arm, task, seed) from the raw CSV, independently of build.py."""
    pool = set((DIRECTORY / "raw/pools/benchmarks/pool492.txt").read_text().split())
    with open(DIRECTORY / "raw/trials.csv", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["exclude"] == "0" and r["task"] in pool and no_web(r["arm"])]
    usable = lambda r: r["scored"] == "1" and r["exception_type"] not in INFRA
    best = lambda lst: max(lst, key=lambda r: (usable(r), r["finished_at"]))
    buckets = defaultdict(lambda: defaultdict(list))
    packed = defaultdict(set)
    for r in rows:
        key = (r["harness"], r["model"], r["arm"], r["subset"], int(r["repeat"]))
        buckets[key][r["task"]].append(r)
        if int(r["attempt"]) >= 2:
            packed[key].add(r["job"])
    slots = []
    for key, tasks in buckets.items():
        for task, lst in tasks.items():
            base = {int(r["attempt"]): r for r in lst if r["job"] in packed[key]}
            if not base:
                slots.append((key[4], best(lst)))
                continue
            repairs = [r for r in lst if r["job"] not in packed[key] and usable(r)]
            repair = max(repairs, key=lambda r: r["finished_at"]) if repairs else None
            if 1 not in base and repair:
                base[1] = repair
            slots += [(seed, r if usable(r) or repair is None else repair) for seed, r in sorted(base.items())]
    return rows, slots, usable


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY

    def table_responses(self):
        subjects = self.tables["subjects"].set_index("subject_id")
        items = self.tables["items"].set_index("item_id").raw_item_id
        merged = self.tables["responses"].join(subjects, on="subject_id").assign(task=lambda t: t.item_id.map(items))
        merged["tool_mode"] = merged.subject_features_extra.map(lambda v: features(v)["tool_mode"])
        return merged

    def test_released_outcomes(self):
        rows, slots, usable = released_slots()
        versions = defaultdict(set)
        for r in rows:
            if r["agent_version"] not in ("", "unknown"):
                versions[r["job"]].add(r["agent_version"])
        expected, unplaced = Counter(), 0
        for seed, r in slots:
            version = r["agent_version"] if r["agent_version"] not in ("", "unknown") else (
                next(iter(versions[r["job"]])) if len(versions[r["job"]]) == 1 else None)
            if version is None:
                self.assertFalse(usable(r))
                unplaced += 1
                continue
            mode = "control-nw" if r["arm"] == "cc-v1.0.100-nw" else r["arm"]
            response = float(r["reward"]) if usable(r) else None
            expected[(MODELS[r["model"]], HARNESSES[r["harness"]], version, mode, r["task"], seed, response)] += 1
        table = self.table_responses()
        observed = Counter(
            (t.display_name, t.harness, t.harness_version, t.tool_mode, t.task, int(t.trial),
             None if pd.isna(t.response) else float(t.response))
            for t in table.itertuples()
        )
        self.assertEqual(observed, expected)
        check_source_claims(self.expected, {
            "released_no_web_trials": len(rows),
            "canonical_slots": len(slots),
            "unplaced_ungraded_slots": unplaced,
            "released_passes": sum(k[-1] == 1.0 for k in expected.elements()),
            "ungraded_responses": sum(k[-1] is None for k in expected.elements()),
            **self.reported_observations(table),
        })

    def reported_observations(self, table):
        """Recompute the paper's tables from the built responses."""
        def cell(model, harness, mode, seed, pool=None, version=None):
            # The paper's control arms exclude the Claude Code 1.0.100 drift run, which shares control's tool mode.
            t = table.loc[table.display_name.eq(MODELS[model]) & table.harness.eq(HARNESSES[harness])
                          & table.tool_mode.eq(mode) & table.response.notna()
                          & (table.harness_version.ne("1.0.100") | (version == "1.0.100"))]
            if seed is not None:
                t = t.loc[t.trial.eq(seed)]
            if pool:
                pools = self.tables["items"].set_index("raw_item_id").item_features.map(lambda v: features(v)["pool"])
                t = t.loc[t.task.map(pools).eq(pool)]
            if version:
                t = t.loc[t.harness_version.eq(version)]
            return t
        table1 = {}
        for model in ("qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview", "opus5"):
            for harness, mode in (("claude-code", "control-nw"), ("mini-swe-agent", "control-nw"),
                                  ("opencode", "control-nw-rg")):
                t = cell(model, harness, mode, 1, "hard45")
                if len(t):
                    table1[f"{model}:{harness}:solved"] = int(t.response.sum())
                    table1[f"{model}:{harness}:usable"] = len(t)
        table2 = {}
        for model in ("qwen3.6-35b-a3b", "qwen3.8-27b"):
            arms = {"CC": cell(model, "claude-code", "control-nw", 1, "pool447"),
                    "mini": cell(model, "mini-swe-agent", "control-nw", 1, "pool447"),
                    "OC": cell(model, "opencode", "control-nw-rg", 1, "pool447")}
            for a, b in (("CC", "mini"), ("CC", "OC"), ("mini", "OC")):
                pa, pb = arms[a].set_index("task").response, arms[b].set_index("task").response
                shared_tasks = pa.index.intersection(pb.index)
                table2[f"{model}:{a}-{b}:n"] = len(shared_tasks)
                table2[f"{model}:{a}-{b}:A"] = int(pa[shared_tasks].sum())
                table2[f"{model}:{a}-{b}:B"] = int(pb[shared_tasks].sum())
        opus = {}
        for label, mode, seed, version in (("control-1", "control-nw", 1, "2.1.263"),
                                           ("control-2", "control-nw", 2, "2.1.263"),
                                           ("control-3", "control-nw", 3, "2.1.263"),
                                           ("bash-only-1", "c2-bashonly-nw", 1, None),
                                           ("bash-only-2", "c2-bashonly-nw", 2, None),
                                           ("no-shell", "c2-noshell-nw", 1, None),
                                           ("claude-code-1.0.100", "control-nw", 1, "1.0.100")):
            opus[label] = int(cell("opus5", "claude-code", mode, seed, "hard45", version).response.sum())
        thinking = {}
        for harness in ("claude-code", "mini-swe-agent", "opencode"):
            for side, mode in (("on", "control-nw"), ("off", "control-nw-nothink")):
                t = cell("qwen3.6-35b-a3b", harness, mode, None, "hard45")
                thinking[f"{harness}:{side}:solved"] = int(t.response.sum())
                thinking[f"{harness}:{side}:usable"] = len(t)
        return {"reported_hard45_table1": table1, "reported_pool447_table2": table2,
                "reported_opus_settings": opus, "reported_thinking_off": thinking}

    def test_items_match_swebench_verified(self):
        verified = pd.read_parquet(DIRECTORY / "raw/swebench_verified.parquet").set_index("instance_id")
        pool = set((DIRECTORY / "raw/pools/benchmarks/pool492.txt").read_text().split())
        self.assertEqual(set(self.tables["items"].raw_item_id), pool)
        for row in self.tables["items"].itertuples():
            source = verified.loc[row.raw_item_id]
            spec = json.loads(json.loads(row.verifier)["spec"])
            self.assertEqual(row.content, source.problem_statement)
            self.assertEqual(spec["test_patch"], source.test_patch)
            self.assertEqual(spec["FAIL_TO_PASS"], json.loads(source.FAIL_TO_PASS))
            self.assertEqual(spec["PASS_TO_PASS"], json.loads(source.PASS_TO_PASS))
            self.assertIsNone(json.loads(row.grading_criterion)["reference_answer"])


if __name__ == "__main__":
    unittest.main()
