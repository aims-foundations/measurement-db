#!/usr/bin/env python3
"""Curate Harness Buy's released per-trial SWE-bench Verified table into measurement tables."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher

# Released model labels -> registry keys.
MODELS = {"opus5": "claude-opus-5", "ds-v4-flash": "deepseek-v4-flash", "ds-v4-pro": "deepseek-v4-pro",
          "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B",
          "glm-5.3-flash": "GLM-5.3-Flash", "hy4-preview": "HY4-Preview"}
HARNESSES = {"claude-code": "claude_code", "mini-swe-agent": "mini_swe_agent", "opencode": "opencode",
             "dsh": "deepseek_harness"}
# Arms with agent-phase web access blocked. The web-enabled arms are out of scope: the authors found agents
# fetching the upstream fix, replaced those arms with these reruns, and ran them on unpatched registry tasks.
NO_WEB_ARMS = {"control-nw", "control-nw-rg", "control-nw-nothink", "control-nw-midsys", "c2-bashonly-nw",
               "c2-noshell-nw", "cc-v1.0.100-nw", "B-stock", "B-ac"}
WEB_ARMS = {"control", "c1off", "c2-bashonly", "c3-notask", "c7-notodo", "ctx40k", "ctx50k", "smoke"}
# The version-drift arm differs from control only in the Claude Code release, which harness_version records.
TOOL_MODES = {"cc-v1.0.100-nw": "control-nw"}
STATUSES = {"complete", "running"}
ROWS, KEPT_ROWS, POOL = 8032, 6879, 492
RULE = ("After the agent exits, Harbor's SWE-bench verifier resets the files the held-out test patch touches, "
        "applies it, runs the repository's test command and parses the log with swebench. Reward 1 requires "
        "every FAIL_TO_PASS and PASS_TO_PASS test to pass (FAIL_TO_PASS only for swebench's FAIL_ONLY_REPOS); "
        "otherwise reward 0.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def literal(source: str, name: str):
    """Read one top-level literal from a pinned analysis script without executing it."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"main_table.py no longer defines {name}")


def job_efforts(config_dir: Path) -> dict:
    """Reasoning effort each released job configuration states (None where it states none)."""
    efforts = {}
    for path in sorted(config_dir.glob("*.json")):
        job = json.loads(path.read_text())
        agent = job["agents"][0]
        kwargs = agent.get("kwargs", {})
        found = {agent.get("env", {}).get("CLAUDE_CODE_EFFORT_LEVEL"), kwargs.get("reasoning_effort"),
                 kwargs.get("config", {}).get("model", {}).get("model_kwargs", {}).get("extra_body", {})
                 .get("reasoning_effort")}
        for provider in kwargs.get("opencode_config", {}).get("provider", {}).values():
            found |= {model.get("options", {}).get("reasoningEffort") for model in provider.get("models", {}).values()}
        found.discard(None)
        require(len(found) <= 1, f"{path.name} states conflicting reasoning efforts")
        efforts.setdefault(job["job_name"], set()).add(next(iter(found), None))
    return efforts


class HarnessBuySWE(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("trials", "configs", "reported", "grader", "swebench_verified")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Transform released no-web SWE-bench Verified trials into subjects, items, and responses."""
        # 1. Load the released trial table; any change to its identity stops the build.
        trials = pd.read_csv(self.raw_dir / "trials.csv", dtype={"agent_version": str}, low_memory=False)
        require(len(trials) == ROWS and trials.trial_id.is_unique, "Released trial identity changed")
        require(set(trials.arm) == NO_WEB_ARMS | WEB_ARMS, "Unexpected arm")
        require(set(trials.status) == STATUSES, "Unexpected trial status")
        require(trials.scored.eq(1).eq(trials.reward.notna()).all(), "A scored trial lacks its reward, or vice versa")
        require(trials.reward.dropna().isin([0.0, 1.0]).all(), "Unexpected reward")

        # 2. Keep non-excluded SWE-bench Verified trials of the no-web arms.
        pool = set((self.raw_dir / "pools/benchmarks/pool492.txt").read_text().split())
        require(len(pool) == POOL, "Task pool changed")
        # Only excluded smoke trials ran psf/requests, the repository the authors dropped from the pool.
        live = trials.loc[trials.exclude.eq(0)]
        swe = live.task.isin(pool)
        require(live.loc[~swe].subset.eq("tb21").all() and live.loc[swe].subset.ne("tb21").all(),
                "Terminal-Bench rows and SWE-bench rows are not separated by subset")
        kept = live.loc[swe & live.arm.isin(NO_WEB_ARMS)].copy()
        require(len(kept) == KEPT_ROWS and set(kept.model) == set(MODELS) and set(kept.harness) == set(HARNESSES),
                "No-web cohort changed")

        # 3. A trial is usable when it was scored and not voided by one of the authors' infrastructure exceptions.
        main_table = (self.raw_dir / "configs/analysis/main_table.py").read_text()
        infra = literal(main_table, "INFRA")
        require(len(infra) == 14, "The authors' infrastructure exception set changed")
        kept["usable"] = kept.scored.eq(1) & ~kept.exception_type.isin(infra)

        # 4. Pick one trial per (arm, task, seed) with the authors' canonical rule (main_table.canonical):
        #    seeds are the job-level repeat, except in jobs that pack k=2 attempts, where the attempt is the seed
        #    and the latest usable repair-job trial refills unusable attempts; elsewhere a usable trial beats an
        #    unusable one and ties go to the latest finished trial.
        kept["finished"] = kept.finished_at.fillna("")
        slots = []
        for (*_, repeat), bucket in kept.groupby(["harness", "model", "arm", "subset", "repeat"]):
            packed = set(bucket.loc[bucket.attempt.ge(2), "job"])
            for task, rows in bucket.groupby("task"):
                rows = rows.sort_values(["usable", "finished"])
                base = rows.loc[rows.job.isin(packed)]
                if base.empty:
                    slots.append((repeat, rows.iloc[-1]))
                    continue
                require(base.attempt.is_unique, f"{task}: a packed job repeats an attempt")
                repairs = rows.loc[~rows.job.isin(packed) & rows.usable].sort_values("finished")
                repair = repairs.iloc[-1] if len(repairs) else None
                seeds = {int(row.attempt): row for _, row in base.iterrows()}
                if 1 not in seeds and repair is not None:
                    seeds[1] = repair  # The task was only ever run by a repair job.
                for seed, row in sorted(seeds.items()):
                    slots.append((seed, row if row.usable or repair is None else repair))
        picked = pd.DataFrame([row for _, row in slots]).reset_index(drop=True)
        picked["trial"] = [seed for seed, _ in slots]
        require(picked.trial_id.is_unique, "One released trial fills two seeds")
        require(not picked.duplicated(["harness", "model", "arm", "task", "trial"]).any(),
                "Two subsets fill the same (arm, task, seed)")

        # 5. Each model, harness, harness version and arm is one subject; the version is the picked trial's,
        #    or, for an ungraded trial that never recorded one, its job's single recorded version.
        recorded = kept.agent_version.notna() & kept.agent_version.ne("unknown")
        job_versions = kept.loc[recorded].groupby("job").agent_version.agg(lambda s: sorted(set(s)))
        known = picked.agent_version.notna() & picked.agent_version.ne("unknown")
        require(picked.loc[~known].usable.eq(False).all(), "A graded trial lacks its harness version")
        fallback = picked.job.map(lambda job: job_versions[job][0] if len(job_versions.get(job, [])) == 1 else None)
        picked["harness_version"] = picked.agent_version.where(known, fallback)
        # An ungraded trial whose job ran several versions cannot be placed in a subject.
        picked = picked.loc[picked.harness_version.notna()].reset_index(drop=True)

        efforts = job_efforts(self.raw_dir / "configs/jobs")
        cell_effort = {}
        for cell, jobs in kept.groupby(["harness", "model", "arm"]).job.agg(lambda s: set(s)).items():
            stated = set().union(*(efforts.get(job, {"no config"}) for job in jobs))
            cell_effort[cell] = next(iter(stated)) if len(stated) == 1 and stated <= {"high", "xhigh"} else None
        picked["reasoning_effort"] = [cell_effort[(r.harness, r.model, r.arm)] for r in picked.itertuples()]
        picked["tool_mode"] = picked.arm.map(lambda arm: TOOL_MODES.get(arm, arm))
        picked["subject_key"] = picked.harness + "|" + picked.model + "|" + picked.tool_mode + "|" + picked.harness_version
        subjects = picked.drop_duplicates("subject_key").reset_index(drop=True)
        subjects["raw_label"] = subjects.model.map(MODELS)
        subjects["features"] = [
            {"harness": HARNESSES[row.harness], "harness_version": row.harness_version, "tool_mode": row.tool_mode,
             **({"reasoning_effort": row.reasoning_effort} if row.reasoning_effort else {})}
            for row in subjects.itertuples()
        ]

        # 6. Items are the SWE-bench Verified instances the harness was given; Harbor's instruction is the
        #    problem statement verbatim.
        verified = pd.read_parquet(self.raw_dir / "swebench_verified.parquet")
        require(verified.instance_id.is_unique and pool <= set(verified.instance_id), "A pool task is not in SWE-bench Verified")
        tasks = verified.loc[verified.instance_id.isin(set(picked.task))].sort_values("instance_id").reset_index(drop=True)
        require(set(tasks.instance_id) == set(picked.task), "An attempted task has no SWE-bench row")
        template = self.raw_dir / "grader/adapters/swebench/src/swebench_adapter/task-template"
        require((template / "instruction.md").read_text() == "{problem_statement}", "Harbor's instruction template changed")
        test_sh = (template / "tests/test.sh").read_text()
        require('dependencies = ["swebench==4.0.3"' in test_sh, "Harbor's parser no longer pins swebench 4.0.3")
        hard45 = set((self.raw_dir / "pools/benchmarks/hard45.txt").read_text().split())

        hb = next(s["revision"] for s in self.source_manifest["upstream"] if s.get("name") == "configs")
        harbor = next(s["revision"] for s in self.source_manifest["upstream"] if s.get("name") == "grader")
        adapter = f"https://github.com/harbor-framework/harbor/blob/{harbor}/adapters/swebench/src/swebench_adapter"
        patches = {name: f"https://github.com/YangzeLiu/what-does-a-harness-buy/blob/{hb}/scripts/{name}.py"
                   for name in ("patch_testsh_newfiles", "patch_testsh_uv", "patch_testsh_ghfix")}
        for name in patches:
            require((self.raw_dir / f"configs/scripts/{name}.py").is_file(), f"{name}.py is missing")
        verifiers = [
            json.dumps({"kind": "harbor_swebench", "protocol": f"{adapter}/task-template/tests/test.sh",
                        "test_commands": f"{adapter}/utils.py", "parser": "swebench==4.0.3",
                        "instance_id": row.instance_id, "repo": row.repo, "version": row.version,
                        "base_commit": row.base_commit, "test_patch": row.test_patch,
                        "FAIL_TO_PASS": json.loads(row.FAIL_TO_PASS), "PASS_TO_PASS": json.loads(row.PASS_TO_PASS),
                        "test_sh_patches": [patches["patch_testsh_uv"], patches["patch_testsh_ghfix"]]
                        + ([patches["patch_testsh_newfiles"]] if "\nnew file mode" in row.test_patch else [])},
                       sort_keys=True)
            for row in tasks.itertuples()
        ]
        items = tasks.assign(
            item_key=tasks.instance_id, raw_item_id=tasks.instance_id, content=tasks.problem_statement,
            grading_criterion=[{"rule": RULE}] * len(tasks), verifier=[ExactMatcher(spec) for spec in verifiers],
            features=[{"repo": row.repo, "base_commit": row.base_commit, "swebench_version": row.version,
                       "difficulty": row.difficulty, "pool": "hard45" if row.instance_id in hard45 else "pool447"}
                      for row in tasks.itertuples()],
        )

        # 7. Map picked trials to responses: reward when usable, null when only voided attempts exist.
        responses = picked.assign(
            response_key=picked.trial_id, item_key=picked.task,
            response=picked.reward.where(picked.usable),
        )
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
        }


if __name__ == "__main__":
    HarnessBuySWE(__file__).main_from_args()
