"""Parse captured AgentDojo task declarations without running agents."""

import ast
import re
from pathlib import Path

_USER_TASK_RE = re.compile(r"^UserTask(\d+)$")
_INJ_TASK_RE = re.compile(r"^InjectionTask(\d+)$")
DEFENSE_SUFFIXES = [
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
    "tool_filter",
    "transformers_pi_detector",
]


def _numeric_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = _numeric_literal(node.left), _numeric_literal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("Not numeric literal arithmetic")


def _string_from_ast_node(node, module_consts: dict[str, str]) -> str | None:
    """Extract a literal string (incl. simple f-strings, concats, parens) from an AST node.

    For f-strings we stitch together constant parts and any formatted values
    whose reference points to a string constant in ``module_consts`` (e.g.
    module-level ``_ATTACKER_IBAN = "..."``). Unknown interpolations raise an
    error; literal braces in source instructions remain unchanged. Adjacent
    string concatenation via ``+`` and implicit juxtaposition is supported.
    """
    if node is None:
        return None
    if isinstance(node, ast.BinOp):
        try:
            return str(_numeric_literal(node))
        except ValueError:
            pass
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return str(node.value)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.right, ast.Constant)
    ):
        return str(node.left.value * node.right.value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        return node.func.value.value.join(module_consts[node.args[0].id])
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
                    raise ValueError(
                        f"Unresolved task definition: {ast.unparse(inner)}"
                    )
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
        return ({}, {}, {})
    module_consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                try:
                    module_consts[t.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    user_tasks: dict[int, str] = {}
    injection_tasks: dict[int, str] = {}
    user_task_gt: dict[int, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_consts = dict(module_consts)
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                t = item.targets[0]
                if isinstance(t, ast.Name):
                    try:
                        class_consts[t.id] = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        pass
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                try:
                    class_consts[item.target.id] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
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
                elif item.target.id == "GROUND_TRUTH_OUTPUT" and item.value is not None:
                    ground_truth_output = _string_from_ast_node(
                        item.value, class_consts
                    )
        if um and prompt:
            user_tasks[int(um.group(1))] = prompt
        if um and ground_truth_output:
            user_task_gt[int(um.group(1))] = ground_truth_output
        if im and goal:
            injection_tasks[int(im.group(1))] = goal

    def _resolve_task_ref(arg) -> str | None:
        if isinstance(arg, ast.Subscript):
            slc = arg.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                m = re.match("user_task_(\\d+)", slc.value)
                if m:
                    return user_tasks.get(int(m.group(1)))
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_combine = isinstance(fn, ast.Attribute) and fn.attr == "create_combined_task"
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
            continue
        explicit = None
        for kw in node.keywords:
            if kw.arg == "prompt":
                explicit = _string_from_ast_node(kw.value, module_consts)
        if explicit:
            user_tasks[idx] = explicit
            continue
        parts: list[str] = []
        for arg in node.args[1:]:
            ref = _resolve_task_ref(arg)
            if ref:
                parts.append(ref)
        if parts:
            user_tasks[idx] = " ".join(parts)
    return (user_tasks, injection_tasks, user_task_gt)


def load_task_prompts(
    suites_dir: Path,
) -> tuple[
    dict[tuple[str, str], str], dict[tuple[str, str], str], dict[tuple[str, str], str]
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
        return (user_prompts, injection_goals, user_ground_truths)
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
                    user_prompts[suite, f"user_task_{idx}"] = text
                for idx, gt in ut_gt.items():
                    user_ground_truths[suite, f"user_task_{idx}"] = gt
            if it_path.exists():
                _, it_map, _ = _parse_tasks_file(it_path)
                for idx, text in it_map.items():
                    injection_goals[suite, f"injection_task_{idx}"] = text
    return (user_prompts, injection_goals, user_ground_truths)


def identify_model_and_defense(model_dir_name: str) -> tuple:
    """Split model directory name into (base_model, defense)."""
    for suffix in DEFENSE_SUFFIXES:
        if model_dir_name.endswith(f"-{suffix}"):
            base = model_dir_name[: -(len(suffix) + 1)]
            return (base, suffix)
    return (model_dir_name, None)


def _model_label(model_dir_name: str) -> str:
    base, defense = identify_model_and_defense(model_dir_name)
    return f"{base} ({defense})" if defense else base
