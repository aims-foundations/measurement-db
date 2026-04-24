#!/usr/bin/env python3
"""Build AgentDojo long-form responses from per-task JSON result files.

Data source: https://github.com/ethz-spylab/agentdojo (cloned to raw/agentdojo/runs)

Directory structure:
  runs/{model}/{suite}/{user_task}/{attack}/{injection_task}.json
  runs/{model}/{suite}/{user_task}/none/none.json   (no-attack baseline)

Each JSON has:
  - suite_name, pipeline_name (model), user_task_id, injection_task_id
  - attack_type (None for no-attack, else attack name)
  - utility (bool), security (bool)

Mapping to long form:
  - item_id:
      no-attack or per-user-task utility -> "{suite}::{user_task_id}"
      security rows (per injection) -> "{suite}::{user_task_id}::{injection_task_id}"
  - subject_id: resolve_subject(model_label)  (model_label includes defense suffix)
  - test_condition encodes metric + attack:
      "metric=utility|attack=none"
      "metric=utility|attack={attack_type}"
      "metric=security|attack={attack_type}"

Outputs:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'AgentDojo: prompt-injection evaluation across many tool-using agents and environments.',
    'testing_condition': 'test_condition encodes metric (utility|security) and attack type.',
    'paper_url': 'https://arxiv.org/abs/2406.13352',
    'data_source_url': 'https://github.com/ethz-spylab/agentdojo',
    'subject_type': 'agent',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{debenedetti2024agentdojodynamicenvironmentevaluate,
      title={AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents},
      author={Edoardo Debenedetti and Jie Zhang and Mislav Balunović and Luca Beurer-Kellner and Marc Fischer and Florian Tramèr},
      year={2024},
      eprint={2406.13352},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2406.13352},
}""",
    'tags': ['agent'],
    'modality': ['text'],
    'domain': ['tool_use', 'safety'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2024-06',
}


import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
RUNS_DIR = RAW_DIR / "agentdojo/runs"
SUITES_DIR = RAW_DIR / "agentdojo/src/agentdojo/default_suites"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)

DEFENSE_SUFFIXES = [
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
    "tool_filter",
    "transformers_pi_detector",
]


def download():
    """Clone the agentdojo repo (contains per-task run results)."""
    clone_dir = RAW_DIR / "agentdojo"
    if not clone_dir.exists():
        print("Cloning AgentDojo repo...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ethz-spylab/agentdojo.git",
                 str(clone_dir)],
                check=True,
            )
        except Exception as e:
            print(f"WARNING: Failed to clone agentdojo: {e}")
    else:
        print(f"  Repo already exists at {clone_dir}, pulling...")
        subprocess.run(["git", "-C", str(clone_dir), "pull", "--ff-only"],
                       capture_output=True)


_USER_TASK_RE = re.compile(r"^UserTask(\d+)$")
_INJ_TASK_RE = re.compile(r"^InjectionTask(\d+)$")


def _string_from_ast_node(node, module_consts: dict[str, str]) -> str | None:
    """Extract a literal string (incl. simple f-strings, concats, parens) from an AST node.

    For f-strings we stitch together constant parts and any formatted values
    whose reference points to a string constant in ``module_consts`` (e.g.
    module-level ``_ATTACKER_IBAN = "..."``). Unknown references are kept as
    ``{name}`` placeholders so the content is still informative. Adjacent
    string concatenation via ``+`` and implicit juxtaposition is supported.
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in module_consts:
        return module_consts[node.id]
    if isinstance(node, ast.Attribute):
        attr = getattr(node, "attr", None)
        if attr and attr in module_consts:
            return module_consts[attr]
        return None
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                pieces.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = v.value
                resolved = _string_from_ast_node(inner, module_consts)
                if resolved is not None:
                    pieces.append(resolved)
                else:
                    # Unknown expression: drop an opaque placeholder in.
                    name = getattr(inner, "id", None) or getattr(inner, "attr", "?")
                    pieces.append("{" + str(name) + "}")
        return "".join(pieces)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_from_ast_node(node.left, module_consts)
        right = _string_from_ast_node(node.right, module_consts)
        if left is not None and right is not None:
            return left + right
        return None
    return None


def _parse_tasks_file(
    path: Path,
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Parse a user_tasks.py / injection_tasks.py file.

    Returns three maps: ``{index: PROMPT_text}``, ``{index: GOAL_text}`` and
    ``{index: GROUND_TRUTH_OUTPUT_text}``. The first two depend on the class
    hierarchy naming convention (UserTask vs InjectionTask). The third is
    populated only for UserTask classes that declare a
    ``GROUND_TRUTH_OUTPUT`` literal (the expected final-answer string).
    Classes whose ground truth is only an imperative ``ground_truth()``
    method (returning tool calls) are not captured here — that's the
    "not extractable via the AST loader" case noted in the report.
    """
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return {}, {}, {}

    # First pass: collect module-level string constants for f-string interp.
    module_consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    module_consts[t.id] = node.value.value

    user_tasks: dict[int, str] = {}
    injection_tasks: dict[int, str] = {}
    user_task_gt: dict[int, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        # Also pick up class-level string consts for f-string resolution.
        class_consts = dict(module_consts)
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                t = item.targets[0]
                if isinstance(t, ast.Name) and isinstance(item.value, ast.Constant):
                    if isinstance(item.value.value, str):
                        class_consts[t.id] = item.value.value

        um = _USER_TASK_RE.match(node.name)
        im = _INJ_TASK_RE.match(node.name)
        if not (um or im):
            continue

        prompt = None
        goal = None
        ground_truth_output = None
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                t = item.targets[0]
                if isinstance(t, ast.Name):
                    if t.id == "PROMPT":
                        prompt = _string_from_ast_node(item.value, class_consts)
                    elif t.id == "GOAL":
                        goal = _string_from_ast_node(item.value, class_consts)
                    elif t.id == "GROUND_TRUTH_OUTPUT":
                        ground_truth_output = _string_from_ast_node(
                            item.value, class_consts
                        )
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == "PROMPT" and item.value is not None:
                    prompt = _string_from_ast_node(item.value, class_consts)
                elif item.target.id == "GOAL" and item.value is not None:
                    goal = _string_from_ast_node(item.value, class_consts)
                elif (item.target.id == "GROUND_TRUTH_OUTPUT"
                      and item.value is not None):
                    ground_truth_output = _string_from_ast_node(
                        item.value, class_consts
                    )

        if um and prompt:
            user_tasks[int(um.group(1))] = prompt
        if um and ground_truth_output:
            user_task_gt[int(um.group(1))] = ground_truth_output
        if im and goal:
            injection_tasks[int(im.group(1))] = goal

    # Second pass: TaskCombinator.create_combined_task calls that synthesize
    # UserTask17, UserTask18, ... from existing tasks. Handles both the
    # explicit ``prompt="..."`` kwarg form and the implicit "a + b"
    # combination (we stitch the two source prompts together).
    def _resolve_task_ref(arg) -> str | None:
        # Shape: task_suite.user_tasks["user_task_N"]
        if isinstance(arg, ast.Subscript):
            slc = arg.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                m = re.match(r"user_task_(\d+)", slc.value)
                if m:
                    return user_tasks.get(int(m.group(1)))
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_combine = (
            isinstance(fn, ast.Attribute) and fn.attr == "create_combined_task"
        )
        if not is_combine or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        name = first.value
        um = _USER_TASK_RE.match(name)
        if not um:
            continue
        idx = int(um.group(1))
        if idx in user_tasks:
            continue  # explicit class already set it

        # Explicit prompt kwarg?
        explicit = None
        for kw in node.keywords:
            if kw.arg == "prompt":
                explicit = _string_from_ast_node(kw.value, module_consts)
        if explicit:
            user_tasks[idx] = explicit
            continue

        # Compose from referenced tasks.
        parts: list[str] = []
        for arg in node.args[1:]:
            ref = _resolve_task_ref(arg)
            if ref:
                parts.append(ref)
        if parts:
            user_tasks[idx] = " ".join(parts)

    return user_tasks, injection_tasks, user_task_gt


def load_task_prompts(
    suites_dir: Path,
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
]:
    """Return user prompts, injection goals, and user-task ground truths.

    Keys are ``(suite, user_task_id)`` / ``(suite, injection_task_id)``.
    Walks every versioned default_suites/v*/<suite>/{user,injection}_tasks.py
    file. Later versions override earlier ones (v1 -> v1_2_2), so the latest
    definition wins — consistent with how AgentDojo itself loads suites.
    """
    user_prompts: dict[tuple[str, str], str] = {}
    injection_goals: dict[tuple[str, str], str] = {}
    user_ground_truths: dict[tuple[str, str], str] = {}

    if not suites_dir.exists():
        print(f"  WARNING: suites dir not found: {suites_dir}")
        return user_prompts, injection_goals, user_ground_truths

    versions = sorted(
        p for p in suites_dir.iterdir() if p.is_dir() and p.name.startswith("v")
    )
    for vdir in versions:
        for suite_dir in sorted(vdir.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name
            ut_path = suite_dir / "user_tasks.py"
            it_path = suite_dir / "injection_tasks.py"
            if ut_path.exists():
                ut_map, _, ut_gt = _parse_tasks_file(ut_path)
                for idx, text in ut_map.items():
                    user_prompts[(suite, f"user_task_{idx}")] = text
                for idx, gt in ut_gt.items():
                    user_ground_truths[(suite, f"user_task_{idx}")] = gt
            if it_path.exists():
                _, it_map, _ = _parse_tasks_file(it_path)
                for idx, text in it_map.items():
                    injection_goals[(suite, f"injection_task_{idx}")] = text

    return user_prompts, injection_goals, user_ground_truths


def identify_model_and_defense(model_dir_name: str) -> tuple:
    """Split model directory name into (base_model, defense)."""
    for suffix in DEFENSE_SUFFIXES:
        if model_dir_name.endswith(f"-{suffix}"):
            base = model_dir_name[: -(len(suffix) + 1)]
            return base, suffix
    return model_dir_name, None


def _model_label(model_dir_name: str) -> str:
    base, defense = identify_model_and_defense(model_dir_name)
    return f"{base} ({defense})" if defense else base


def parse_all_runs(runs_dir: Path):
    """Walk runs/ and yield one record per result JSON file."""
    if not runs_dir.exists():
        print(f"  WARNING: {runs_dir} does not exist")
        return

    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for suite_dir in sorted(model_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite_name = suite_dir.name
            for task_dir in sorted(suite_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name
                for attack_dir in sorted(task_dir.iterdir()):
                    if not attack_dir.is_dir():
                        continue
                    attack_name = attack_dir.name
                    for json_file in sorted(attack_dir.glob("*.json")):
                        try:
                            with open(json_file, "r") as f:
                                data = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"  could not parse {json_file}: {e}",
                                  file=sys.stderr)
                            continue
                        # Agent message trajectory is the natural trace.
                        messages = data.get("messages")
                        trace = None
                        if messages:
                            try:
                                trace = json.dumps(messages, ensure_ascii=False)[:16000]
                            except (TypeError, ValueError):
                                trace = None
                        yield {
                            "model": model_name,
                            "suite": data.get("suite_name", suite_name),
                            "user_task_id": data.get("user_task_id", task_id),
                            "injection_task_id": data.get("injection_task_id"),
                            "attack_type": data.get("attack_type")
                                or (None if attack_name == "none" else attack_name),
                            "utility": data.get("utility"),
                            "security": data.get("security"),
                            "trace": trace,
                        }


def build_long_form(runs_dir: Path) -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "agentdojo",
        name="AgentDojo",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
        modality=INFO.get("modality"),
        domain=INFO.get("domain"),
        response_type=INFO.get("response_type"),
        response_scale=INFO.get("response_scale"),
        categorical=INFO.get("categorical"),
        paper_url=INFO.get("paper_url"),
        release_date=INFO.get("release_date"),
    )

    user_prompts, injection_goals, user_ground_truths = load_task_prompts(SUITES_DIR)
    print(f"  loaded {len(user_prompts)} user-task prompts, "
          f"{len(injection_goals)} injection goals, "
          f"{len(user_ground_truths)} user-task ground truths")

    def _truncate_gt(s: str | None, limit: int = 4000) -> str | None:
        if s is None:
            return None
        if not s:
            return None
        if len(s) <= limit:
            return s
        suffix = "\n...[truncated]"
        return s[: max(0, limit - len(suffix))] + suffix

    rows = []
    n_records = 0
    for rec in parse_all_runs(runs_dir):
        n_records += 1
        model_label = _model_label(rec["model"])
        subj = resolve_subject(model_label)

        attack = rec["attack_type"]
        attack_str = attack if attack else "none"
        # Primary prompt lookup: user_task_id may name a UserTask OR an
        # InjectionTask (the AgentDojo "injection-as-primary-task" ablation).
        key = (rec["suite"], rec["user_task_id"])
        user_prompt = user_prompts.get(key) or injection_goals.get(key)

        trace = rec.get("trace")

        # Utility metric: one observation per (model, suite, user_task, attack)
        if isinstance(rec["utility"], bool):
            item_raw = f"{rec['suite']}::{rec['user_task_id']}"
            ut_correct = _truncate_gt(user_ground_truths.get(key))
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=item_raw,
                content=user_prompt,
                correct_answer=ut_correct,
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": f"metric=utility|attack={attack_str}",
                "response": float(rec["utility"]),
                "correct_answer": ut_correct,
                "trace": trace,
            })

        # Security metric: only meaningful under attack; per (user_task, injection) pair
        if attack and isinstance(rec["security"], bool):
            inj = rec.get("injection_task_id") or "none"
            item_raw = f"{rec['suite']}::{rec['user_task_id']}::{inj}"
            injection_goal = injection_goals.get((rec["suite"], inj))
            # Pair content: user prompt + injection goal, both labelled.
            if user_prompt and injection_goal:
                pair_content = (
                    f"[user_task] {user_prompt}\n"
                    f"[injection_task] {injection_goal}"
                )
            elif user_prompt:
                pair_content = f"[user_task] {user_prompt}"
            elif injection_goal:
                pair_content = f"[injection_task] {injection_goal}"
            else:
                pair_content = None
            # Security items don't have a single-string ground truth upstream
            # (the judge runs post-environment checks on the injection task);
            # leave correct_answer=None for these rows.
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=item_raw,
                content=pair_content,
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": f"metric=security|attack={attack_str}",
                "response": float(rec["security"]),
                "correct_answer": None,
                "trace": trace,
            })

    print(f"  parsed {n_records:,} result records")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)

    # Split: responses.parquet keeps trace=None; traces.parquet holds the trace column.
    resp_cols = ["subject_id", "item_id", "benchmark_id", "trial",
                 "test_condition", "response", "correct_answer", "trace"]
    resp = df[resp_cols].copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)

    traces_path = _BENCHMARK_DIR / "traces.parquet"
    traces = df.loc[df["trace"].notna(), [
        "subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace",
    ]]
    if len(traces) > 0:
        traces.to_parquet(traces_path, index=False)

    registry_save(CONTRIB_DIR)
    print(f"  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    if len(traces) > 0:
        print(f"  wrote {traces_path.name} ({len(traces):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    conds = sorted(df['test_condition'].dropna().unique())
    print(f"  test_conditions: {len(conds)}")
    for cond in conds[:20]:
        g = df[df['test_condition'] == cond]
        print(f"    {cond}: {len(g):,} rows, mean={g['response'].mean():.3f}")
    if len(conds) > 20:
        print(f"    ... and {len(conds) - 20} more")


def main():
    print(f"[agentdojo] building from {_BENCHMARK_DIR}")
    download()
    df = build_long_form(RUNS_DIR)
    print_stats(df)


if __name__ == "__main__":
    main()
