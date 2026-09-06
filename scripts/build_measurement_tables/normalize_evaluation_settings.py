"""Canonical vocabulary for evaluation settings.

Single source of truth for (1) canonical setting-key spellings and (2) each
key's class — what the value is a property of:

  * ``subject``   — configures the test-taker (agent scaffold, reasoning
                    effort, ...). Folds into subject identity: enters the
                    ``subject_id`` hash and materializes as columns on
                    ``subjects.parquet``.
  * ``item``      — alters the stimulus (task, split, difficulty tier,
                    few-shot count, ...). Enters the ``item_id`` hash and
                    materializes as ``item_features`` on ``items.parquet``.
  * ``verifier``  — identifies what applied the grading criterion (judge
                    model, evaluator, refusal detector, annotator). Not a
                    condition: the grader is part of the measurement
                    instrument, so the key is merged into the item's
                    ``verifier`` JSON payload on ``items.parquet`` and
                    enters the ``item_id`` hash — the same prompt graded
                    by two judges is two items.
  * ``interactor``— another party in the interaction that produced the
                    response: the attacker probing the subject, the
                    (simulated or human) user the agent converses with, the
                    opponent in a pairwise comparison. A participant, not a
                    knob of the occasion — so not a condition. Passed to
                    ``add_response(interactors=...)``,
                    which is part of the primary key: an episode against a
                    different interactor is a different observation.
  * ``condition`` — a property of the measurement occasion, not of either
                    side or of any participant (temperature). Passed to
                    ``add_response(test_condition=...)``.
  * ``trial``     — an interchangeable repeat index passed directly to
                    ``add_response(trial=...)``.

``add_subject`` and ``add_item`` consult this module to canonicalize feature
names and reject globally known keys passed to the wrong identity. Response
conditions, interactors, and trials are already final when passed to
``add_response``. Benchmark-specific feature names may remain absent from this
vocabulary because the add method receiving them makes their placement
explicit.

Extend ``KEY_CLASS``/``KEY_ALIASES`` only for keys whose meaning is the same
in every benchmark that could plausibly use them. A key that is subject-side
in one benchmark and item-side in another must stay out and be placed
explicitly by that benchmark's add call.
"""
from __future__ import annotations

import re
import unicodedata

SUBJECT = "subject"
ITEM = "item"
VERIFIER = "verifier"
INTERACTOR = "interactor"
CONDITION = "condition"
TRIAL = "trial"
CLASSES = {SUBJECT, ITEM, VERIFIER, INTERACTOR, CONDITION, TRIAL}

# canonical key -> class. Conservative by design: only keys whose class is
# unambiguous across benchmarks.
KEY_CLASS = {
    # subject features — configure the test-taker
    "harness": SUBJECT,
    "harness_version": SUBJECT,
    "reasoning_effort": SUBJECT,
    "decoding": SUBJECT,
    "thinking": SUBJECT,
    "step_budget": SUBJECT,
    "tool_mode": SUBJECT,
    "sandbox": SUBJECT,
    # item features — alter the stimulus presented
    "task": ITEM,
    "dataset": ITEM,
    "scenario": ITEM,
    "split": ITEM,
    "subset": ITEM,
    "subtask": ITEM,
    "tier": ITEM,
    "difficulty": ITEM,
    "level": ITEM,
    "category": ITEM,
    "domain": ITEM,
    "shot": ITEM,
    "lang": ITEM,
    "qtype": ITEM,
    # verifier — identity of what applied the grading criterion. Per the
    # 2026-08-09 decision these are NOT conditions: the grader belongs to
    # the item's `verifier` payload, and grader variation splits items.
    "judge": VERIFIER,
    "annotator": VERIFIER,
    "evaluator": VERIFIER,
    "refusal_detector": VERIFIER,
    # interactors — other parties in the interaction (2026-08-09 decision:
    # these are participants, not conditions, and never touch test_condition)
    "attacker": INTERACTOR,
    "user_sim": INTERACTOR,
    "opponent": INTERACTOR,
    # conditions — observation-level knobs. Per the 2026-08-08 decision,
    # test_condition holds only trial-identity keys and temperature (or
    # null): temperature is a stochastic sampling knob of the observation,
    # not a configuration of the test-taker.
    # "release" is deliberately NOT listed: it is not one class everywhere.
    # DeepSWE's v1/v1.1 "release" is grader identity (verifier-class,
    # routed as `evaluator`); elsewhere a release can version the item bank
    # (item-class). Builds must name the thing that actually varied.
    "temperature": CONDITION,
    # trial — interchangeable repeats
    "episode": TRIAL,
    "replicate": TRIAL,
    "seed": TRIAL,
    "attempt": TRIAL,
    "repeat": TRIAL,
}

# alias (after _canonical_form) -> canonical key. This is the drift control:
# the four spellings of "the scaffold the subject ran inside" observed across
# built benchmarks all land in one column.
KEY_ALIASES = {
    "scaffold": "harness",
    "agent": "harness",
    "framework": "harness",
    "agent_scaffold": "harness",
    "agent_framework": "harness",
    "harness_setting": "harness",
    "agent_version": "harness_version",
    "scaffold_version": "harness_version",
    "framework_version": "harness_version",
    "agent_reasoning_effort": "reasoning_effort",
    "effort": "reasoning_effort",
    "thinking_effort": "reasoning_effort",
    "agent_temperature": "temperature",
    "agent_decoding": "decoding",
    "language": "lang",
    "shots": "shot",
    "few_shot": "shot",
    "fewshot": "shot",
    "n_shot": "shot",
    "user_simulator": "user_sim",
    "judge_model": "judge",
    "adversary": "attacker",
    "attacker_model": "attacker",
    "opponent_model": "opponent",
}

# Keys that are never a setting of any class; passing one is a hard error.
FORBIDDEN_KEYS = {
    "metric": (
        "a metric is a second measurement of the same response, not a "
        "setting; encode each metric as its own benchmark or response column"
    ),
    "model": (
        "the model is the subject itself; put it in the resolve_subject "
        "label, never in a feature dict"
    ),
}

# Subject-class canonical keys that get a dedicated column on
# subjects.parquet. Every other subject-class key is packed into the
# subject_features_extra string column.
SUBJECT_FEATURE_COLUMNS = ("harness", "reasoning_effort", "harness_version")

_KEY_CLEAN_RE = re.compile(r"[^a-z0-9_]+")


def canonical_key(key: str) -> str:
    """Canonical spelling of a setting key: lowercased, non-alphanumerics
    folded to ``_``, then alias-mapped (``scaffold`` -> ``harness``)."""
    k = _KEY_CLEAN_RE.sub("_", str(key).strip().lower()).strip("_")
    return KEY_ALIASES.get(k, k)


def key_class(key: str) -> str | None:
    """The vocabulary class of ``key`` (any spelling), or None if unlisted."""
    return KEY_CLASS.get(canonical_key(key))


def canonicalize_features(features: dict | None) -> dict[str, str]:
    """Normalize a feature mapping to {canonical_key: clean_value}.

    Keys are alias-folded; values become NFC-normalized stripped strings.
    ``None`` values mean "feature absent" and are dropped. Raises on
    forbidden keys, on ``;``/``=``/newline inside a value (they would corrupt
    the canonical ``k=v;...`` encoding), and on two aliases of the same key
    carrying different values.
    """
    out: dict[str, str] = {}
    for k, v in (features or {}).items():
        if v is None:
            continue
        ck = canonical_key(k)
        if not ck:
            raise ValueError(f"setting key {k!r} normalizes to nothing")
        if ck in FORBIDDEN_KEYS:
            raise ValueError(f"setting key {k!r} is forbidden: {FORBIDDEN_KEYS[ck]}")
        sv = unicodedata.normalize("NFC", str(v)).strip()
        if any(c in sv for c in ";=\n"):
            raise ValueError(
                f"setting {k!r}={sv!r}: values may not contain ';', '=' or "
                "newlines (they would corrupt the canonical k=v;... encoding)"
            )
        if ck in out and out[ck] != sv:
            raise ValueError(
                f"features carry two spellings of {ck!r} with different "
                f"values: {out[ck]!r} vs {sv!r}"
            )
        out[ck] = sv
    return out


def features_string(features: dict[str, str] | None) -> str | None:
    """The canonical ``k=v;...`` encoding: sorted by key, ``;``-joined.
    None for an empty dict, so "no features" is a null, not an empty string."""
    if not features:
        return None
    return ";".join(f"{k}={v}" for k, v in sorted(features.items()))
