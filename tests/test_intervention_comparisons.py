"""Intervention comparisons: hand-computed toy cases, guard rails, and the paper's published counts."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.analyze_measurements import intervention_comparisons as ic

REPO = Path(__file__).resolve().parents[1]
BUILDERS = ("onlycodes_swe", "onlycodes_artifact", "tool_architecture_live", "harness_buy_swe")


def toy_tables(**overrides):
    """One benchmark, model m. Reference arm a pools two harness versions; target arm b has one.

    Runs per item (reference | target):
      i1  1, 1, 0  | 1, null      i2  0, 0 | 1, 1      i3  null | 0
      i5  1        | (not run)    i4  outside the pool=x scope
    """
    subjects = pd.DataFrame([
        dict(subject_id="ref1", display_name="m", harness="h", harness_version="1", reasoning_effort=None,
             subject_features_extra="tool_mode=a"),
        dict(subject_id="ref2", display_name="m", harness="h", harness_version="2", reasoning_effort=None,
             subject_features_extra="tool_mode=a"),
        dict(subject_id="tgt", display_name="m", harness="h", harness_version="3", reasoning_effort=None,
             subject_features_extra="tool_mode=b"),
        dict(subject_id="other", display_name="n", harness="h", harness_version="1", reasoning_effort=None,
             subject_features_extra="tool_mode=b"),
    ])
    items = pd.DataFrame([dict(item_id=f"i{k}", raw_item_id=f"task-{k}", item_features=f"pool={'y' if k == 4 else 'x'}")
                          for k in range(1, 6)])
    runs = [("ref1", "i1", 1, 1.0), ("ref1", "i1", 2, 1.0), ("ref2", "i1", 1, 0.0), ("tgt", "i1", 1, 1.0),
            ("tgt", "i1", 2, None), ("ref1", "i2", 1, 0.0), ("ref2", "i2", 1, 0.0), ("tgt", "i2", 1, 1.0),
            ("tgt", "i2", 2, 1.0), ("ref1", "i3", 1, None), ("tgt", "i3", 1, 0.0), ("ref1", "i5", 1, 1.0),
            ("ref1", "i4", 1, 1.0), ("tgt", "i4", 1, 0.0), ("other", "i1", 1, 1.0)]
    responses = pd.DataFrame([dict(subject_id=s, item_id=i, trial=t, response=r, benchmark_id="toy",
                                   test_condition=None, interactors=None) for s, i, t, r in runs])
    tables = dict(subjects=subjects, items=items, responses=responses)
    tables.update(overrides)
    return tables


def toy_config(**block):
    base = dict(name="a->b", family="tool_surface", models=["m"], item_scope={"pool": "x"},
                reference={"tool_mode": "a"}, target={"tool_mode": "b"},
                changes=[{"target": "tool_surface", "operation": "replace", "value": "a -> b"}], origin="analyst")
    base.update(block)
    return dict(expected_comparisons=1, bootstrap_draws=2000, outcome_policies=list(ic.POLICIES),
                benchmarks={"toy": [base]})


def run_toy(config=None, **tables):
    return ic.run(config or toy_config(), {"toy": ic.prepare("toy", **toy_tables(**tables))})


class ToyComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = run_toy()
        cls.effects = cls.frames["effects"].set_index("outcome_policy")

    def test_observed_skips_ungraded_runs(self):
        e = self.effects.loc["observed"]
        # i1: 2/3 -> 1 (its null run skipped); i2: 0 -> 1; i3 has no graded reference run; i5 is reference-only.
        self.assertEqual((e.n_items, e.n_items_dropped_ungraded, e.n_items_reference_only, e.n_items_target_only),
                         (2, 1, 1, 0))
        self.assertAlmostEqual(e.delta, ((1 - 2 / 3) + (1 - 0)) / 2)
        self.assertAlmostEqual(e.reference_mean, (2 / 3 + 0) / 2)
        self.assertEqual((e.reference_runs, e.reference_passes, e.target_runs, e.target_passes), (5, 2, 3, 3))
        self.assertEqual((e.n_up, e.n_down, e.n_tie), (2, 0, 0))

    def test_null_as_fail_counts_ungraded_runs_as_zero(self):
        e = self.effects.loc["null_as_fail"]
        # i1: 2/3 -> 1/2; i2: 0 -> 1; i3: 0 -> 0.
        self.assertEqual((e.n_items, e.n_items_dropped_ungraded), (3, 0))
        self.assertAlmostEqual(e.delta, ((0.5 - 2 / 3) + 1 + 0) / 3)
        self.assertEqual((e.n_up, e.n_down, e.n_tie), (1, 1, 1))

    def test_bootstrap_resamples_tasks_reproducibly(self):
        e = self.effects.loc["null_as_fail"]
        delta = np.array([0.5 - 2 / 3, 1.0, 0.0])
        means = np.random.default_rng(int(e.bootstrap_seed)).choice(delta, size=(2000, 3)).mean(axis=1)
        self.assertEqual((e.ci95_low, e.ci95_high), tuple(np.quantile(means, [0.025, 0.975])))
        again = run_toy()["effects"].set_index("outcome_policy").loc["null_as_fail"]
        self.assertEqual((e.ci95_low, e.ci95_high), (again.ci95_low, again.ci95_high))
        self.assertLessEqual(e.ci95_low, e.delta)
        self.assertGreaterEqual(e.ci95_high, e.delta)

    def test_definition_lists_pooled_subjects_and_other_differences(self):
        c = self.frames["comparisons"].iloc[0]
        self.assertEqual((list(c.reference_subject_ids), list(c.target_subject_ids)), (["ref1", "ref2"], ["tgt"]))
        self.assertEqual((list(c.reference_harness_versions), list(c.target_harness_versions)), (["1", "2"], ["3"]))
        self.assertEqual(json.loads(c.also_differs),
                         [{"feature": "harness_version", "reference": ["1", "2"], "target": ["3"]}])
        self.assertEqual((c.comparison_id, c.n_scope_items), ("toy/a->b/m", 4))
        tasks = self.frames["task_effects"]
        self.assertEqual(set(tasks.loc[tasks.outcome_policy.eq("observed"), "raw_item_id"]), {"task-1", "task-2"})

    def test_named_features_are_not_reported_as_also_differing(self):
        config = toy_config(reference={"tool_mode": "a", "harness_version": "1"},
                            target={"tool_mode": "b", "harness_version": "3"})
        self.assertEqual(run_toy(config)["comparisons"].iloc[0].also_differs, "[]")

    def test_harness_version_is_not_compared_across_different_harnesses(self):
        subjects = toy_tables()["subjects"].assign(harness=["h", "h", "g", "h"])
        frames = run_toy(subjects=subjects)
        self.assertEqual(json.loads(frames["comparisons"].iloc[0].also_differs),
                         [{"feature": "harness", "reference": ["h"], "target": ["g"]}])


class GuardRails(unittest.TestCase):
    def assertRejected(self, message, config=None, **tables):
        with self.assertRaisesRegex(ValueError, message):
            run_toy(config, **tables)

    def test_selector_errors(self):
        self.assertRejected("unknown subject features", toy_config(target={"arm": "b"}))
        self.assertRejected("matches no subject", toy_config(target={"tool_mode": "c"}))
        self.assertRejected("share subjects", toy_config(reference={"harness": "h"}))
        self.assertRejected("unknown item features", toy_config(item_scope={"split": "x"}))
        self.assertRejected("identical", toy_config(target={"tool_mode": "a"}))
        self.assertRejected("without model", toy_config(target={"model": "n", "tool_mode": "b"}))
        self.assertRejected("harnesses", toy_config(harnesses=["h"], target={"harness": "h", "tool_mode": "b"}))

    def test_config_errors(self):
        config = toy_config()
        config["expected_comparisons"] = 2
        self.assertRejected("expands to 1", config)
        config = toy_config()
        config["benchmarks"]["toy"].append(copy.deepcopy(config["benchmarks"]["toy"][0]))
        config["expected_comparisons"] = 2
        self.assertRejected("Duplicate comparison IDs", config)
        self.assertRejected("origin", toy_config(origin="guess"))
        self.assertRejected("exactly target, operation and value", toy_config(changes=[{"target": "x"}]))
        self.assertRejected("fewer than two tasks", toy_config(item_scope={"pool": "y"}))

    def test_table_errors(self):
        responses = toy_tables()["responses"]
        self.assertRejected("binary", responses=responses.assign(response=responses.response.replace(1.0, 0.5)))
        self.assertRejected("another benchmark", responses=responses.assign(benchmark_id="other"))
        self.assertRejected("test conditions", responses=responses.assign(test_condition="t=0.7"))
        subjects = toy_tables()["subjects"].assign(subject_features_extra="harness=x")
        self.assertRejected("repeats a fixed feature", subjects=subjects)


class ConfigFile(unittest.TestCase):
    def test_reviewed_comparison_list(self):
        config = yaml.safe_load(ic.CONFIG.read_text())
        specs = ic.expand(config)
        self.assertEqual(len(specs), 58)
        self.assertEqual(set(config["outcome_policies"]), {"observed", "null_as_fail"})
        counts = pd.Series([s["benchmark_id"] for s in specs]).value_counts().to_dict()
        self.assertEqual(counts, {"harness_buy_swe": 31, "tool_architecture_live": 15,
                                  "onlycodes_swe": 6, "onlycodes_artifact": 6})
        modes = {s[side].get("tool_mode") for s in specs for side in ("reference", "target")}
        self.assertFalse(modes & {"control-nw-midsys", "B-stock", "B-ac"})
        self.assertEqual(sum(s["family"] == "harness" for s in specs), 21)


@unittest.skipUnless(all((REPO / "benchmarks" / b / "formatted_tables" / "responses.parquet").exists() for b in BUILDERS),
                     "builder tables not built")
class PublishedNumbers(unittest.TestCase):
    """Check the real run against counts derived by hand and the authors' published tables."""

    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load(ic.CONFIG.read_text())
        datasets = {slug: ic.load_benchmark(slug, REPO / "benchmarks")[0] for slug in config["benchmarks"]}
        frames = ic.run(config, datasets)
        cls.effects = frames["effects"].set_index(["comparison_id", "outcome_policy"])
        claims = yaml.safe_load((REPO / "benchmarks/harness_buy_swe/characterization.yaml").read_text())["source_claims"]
        cls.paper = {name: claims[name]["expected"] for name in
                     ("reported_hard45_table1", "reported_pool447_table2", "reported_opus_settings", "reported_thinking_off")}

    def effect(self, comparison_id, policy="observed"):
        return self.effects.loc[(comparison_id, policy)]

    def test_onlycodes_swe(self):
        # Three seeds on 100 tasks per side: Claude 155 -> 150 passes of 300, Codex 139 -> 140.
        claude = self.effect("onlycodes_swe/baseline->code_only/claude-sonnet-4-6")
        self.assertEqual((claude.reference_passes, claude.target_passes, claude.reference_runs), (155, 150, 300))
        self.assertAlmostEqual(claude.delta, -5 / 300)
        self.assertAlmostEqual(self.effect("onlycodes_swe/baseline->code_only/gpt-5.5").delta, 1 / 300)

    def test_tool_architecture_matches_the_pilot(self):
        # The pilot's all-attempt success rule (null = fail): Qwen Atomic - BashOnly = +4.92 pp.
        qwen = "tool_architecture_live/bashonly->atomic/Qwen3-Coder-30B-A3B-Instruct"
        self.assertEqual(round(self.effect(qwen, "null_as_fail").delta * 100, 2), 4.92)
        self.assertEqual(self.effect(qwen, "null_as_fail").reference_runs, 650)

    def test_opus_settings_table(self):
        paper = self.paper["reported_opus_settings"]
        control = paper["control-1"] + paper["control-2"] + paper["control-3"]
        for name, passes, runs in [("control->bash_only", paper["bash-only-1"] + paper["bash-only-2"], 90),
                                   ("control->no_shell", paper["no-shell"], 45),
                                   ("v2.1.263->v1.0.100", paper["claude-code-1.0.100"], 45)]:
            e = self.effect(f"harness_buy_swe/{name}/claude-opus-5/claude_code")
            self.assertEqual((e.reference_passes, e.reference_runs, e.target_passes, e.target_runs),
                             (control, 135, passes, runs))

    def test_thinking_off_table(self):
        paper = self.paper["reported_thinking_off"]
        for harness, label in [("claude_code", "claude-code"), ("mini_swe_agent", "mini-swe-agent")]:
            e = self.effect(f"harness_buy_swe/thinking_on->off/Qwen3.6-35B-A3B/{harness}")
            self.assertEqual((e.reference_passes, e.reference_runs, e.target_passes, e.target_runs),
                             tuple(paper[f"{label}:{side}:{kind}"] for side in ("on", "off") for kind in ("solved", "usable")))

    def test_harness_swap_tables(self):
        # Sides with one run per task reproduce the paper's seed-1 counts exactly.
        table1 = self.paper["reported_hard45_table1"]
        for model, key in [("GLM-5.3-Flash", "glm-5.3-flash"), ("HY4-Preview", "hy4-preview")]:
            for pair, a, b in [("claude_code->mini_swe_agent", "claude-code", "mini-swe-agent"),
                               ("claude_code->opencode", "claude-code", "opencode"),
                               ("mini_swe_agent->opencode", "mini-swe-agent", "opencode")]:
                e = self.effect(f"harness_buy_swe/{pair}@hard45/{model}")
                self.assertEqual((e.reference_passes, e.target_passes, e.n_items),
                                 (table1[f"{key}:{a}:solved"], table1[f"{key}:{b}:solved"], 45))
        table2 = self.paper["reported_pool447_table2"]
        for model, key, pair, label in [("Qwen3.8-27B", "qwen3.8-27b", "claude_code->mini_swe_agent", "CC-mini"),
                                        ("Qwen3.8-27B", "qwen3.8-27b", "claude_code->opencode", "CC-OC"),
                                        ("Qwen3.8-27B", "qwen3.8-27b", "mini_swe_agent->opencode", "mini-OC"),
                                        ("Qwen3.6-35B-A3B", "qwen3.6-35b-a3b", "claude_code->opencode", "CC-OC")]:
            e = self.effect(f"harness_buy_swe/{pair}@pool447/{model}")
            self.assertEqual((e.n_items, e.reference_passes, e.target_passes),
                             tuple(table2[f"{key}:{label}:{kind}"] for kind in ("n", "A", "B")))


if __name__ == "__main__":
    unittest.main()
