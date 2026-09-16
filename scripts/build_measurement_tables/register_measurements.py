"""Register subjects, items, and benchmarks while building measurement tables.

Every modern ``benchmarks/{dataset}/build.py`` calls
``BenchmarkBuild.add_subject`` and ``BenchmarkBuild.add_item`` with final row
values. Those methods use ``resolve_subject`` and ``register_item`` below so
each response references stable IDs. See ``parquet_schemas.yaml`` at the repo
root for the enforced schema.

**Concurrency model.** Builds run in parallel processes. Each build accumulates
its registrations **locally in memory** and flushes them to its own dataset
folder at the end as ``{dataset}/{subjects,items,benchmarks}.parquet``. There is
no shared registry and no merge step — every benchmark folder is self-contained
and ships its own registry tables to HuggingFace.

IDs are deterministic from row content: a ``subject_id`` hashes
``normalized_name`` (the registry's canonical model name; unmapped labels fall
back to the cleaned raw label) plus every metadata column at its observed
value, nulls included. Two builds that assemble the same row produce the same
``subject_id`` — including through different aliases of the same mapped model.
Builds that observe different values — a different access_date, or a registry
entry that appeared between runs — produce different subjects. Because tables
are per-dataset, a shared subject's row is duplicated across folders (it is
not unioned into one canonical row).

Lower-level usage (normal benchmark authors subclass ``BenchmarkBuild``
instead, starting from the complete ``benchmarks/_template/`` folder)::

    from scripts.build_measurement_tables import (
        resolve_subject, register_item, get_benchmark_id, save,
    )

    bench_id = get_benchmark_id("mtbench", name="MT-Bench", ...)
    for raw_label, raw_item in iter_registry_records():
        subj = resolve_subject(raw_label)
        item = register_item(bench_id, raw_item_id=..., content=...)
    save(Path(__file__).resolve().parent)
"""
from __future__ import annotations

import json
import sys
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import normalize_evaluation_settings as _vocab
from .hash_measurement_ids import (
    canonical_asset_manifest,
    canonical_grading_criterion,
    content_hash,
    item_id_from_content,
    subject_id_from_row,
)

# Turn structure of a benchmark, no default: whether the subject receives any
# further input after its first output. A benchmark whose items genuinely span
# both structures (e.g. bfcl: single-call categories plus agent-loop
# multi_turn_* categories) declares the combined value.
_MULTI_SINGLE_TURN = {"single_turn", "multi_turn", "multi_turn, single_turn"}

# In-process accumulators are insertion-ordered dicts keyed by id
# ({subject_id|item_id|benchmark_id} -> row dict). This gives O(1) membership
# and append; the DataFrames are materialized once in `save()`. (Previously
# these were DataFrames grown by a per-row `pd.concat`, which made each build
# O(n^2) in its number of unique items/subjects.)
_lock = threading.Lock()
_subjects: dict | None = None
_items: dict | None = None
_benchmarks: dict | None = None


# --------------------------------------------------------------------------- #
# Model registry (raw subject label -> canonical model + company)
# --------------------------------------------------------------------------- #

# The durable crosswalk maintained in the curation guide's model-mapping phase.
# Keys are raw subject
# strings byte-for-byte; values are {"model": <canonical name>, "company": ...}
# plus, when dated, {"release_date": <first public release>,
# "release_date_url": <citation>} — identical on every alias of a model.
_MODEL_REGISTRY_PATH = Path(__file__).with_name("map_model_registry.json")
_model_registry: dict | None = None


def _load_model_registry() -> dict:
    """Load ``map_model_registry.json`` once per process.

    If the file is absent or malformed, return an empty mapping after warning.
    """
    global _model_registry
    if _model_registry is None:
        try:
            with open(_MODEL_REGISTRY_PATH, encoding="utf-8") as f:
                _model_registry = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(
                f"⚠ register_measurements: could not load "
                f"{_MODEL_REGISTRY_PATH} ({exc}); "
                "normalized_name/provider/release_date will be null this run\n"
            )
            _model_registry = {}
    return _model_registry


# --------------------------------------------------------------------------- #
# In-process state (per build.py run)
# --------------------------------------------------------------------------- #

def _ensure_init():
    global _subjects, _items, _benchmarks
    if _subjects is None:
        _subjects = {}
    if _items is None:
        _items = {}
    if _benchmarks is None:
        _benchmarks = {}


@dataclass(frozen=True)
class _RegistrationRows:
    """Locked view of the current in-process registration dictionaries."""

    subjects: dict[str, dict]
    items: dict[str, dict]
    benchmarks: dict[str, dict]


@contextmanager
def _locked_registration_rows() -> Iterator[_RegistrationRows]:
    """Yield current registration rows while holding the shared state lock.

    Validation and writing use this seam instead of importing the mutable
    dictionaries directly. ``reload()`` rebinds those dictionaries, so direct
    imported aliases could otherwise become stale.
    """
    with _lock:
        _ensure_init()
        assert _subjects is not None and _items is not None and _benchmarks is not None
        yield _RegistrationRows(_subjects, _items, _benchmarks)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class UnknownSubject(KeyError):
    """Raised when a raw label doesn't match any registered subject."""


def resolve_subject(
    raw_label: str,
    *,
    features: dict | None = None,
    access_date: str | None = None,
    auto_register: bool = True,
) -> str:
    """Return the ``subject_id`` for ``raw_label`` (+ optional subject features).

    ``features`` holds the subject features of this configuration
    (``{"harness": "codex", "reasoning_effort": "high"}``). Keys are
    alias-folded by ``normalize_evaluation_settings`` and must not be a known
    non-subject class there. The canonical feature string enters the
    ``subject_id`` hash —
    each feature combination is a distinct subject — and materializes as the
    ``harness`` / ``reasoning_effort`` / ``harness_version`` columns (other
    keys land in ``subject_features_extra``). Without ``features`` the id and
    row are byte-identical to the historical behavior. ``BenchmarkBuild``
    subclasses pass final features through ``add_subject()``.

    The ``subject_id`` hashes row content — ``normalized_name`` as the
    identity base (unmapped labels fall back to the cleaned raw label), plus
    ``provider``, ``release_date``, ``access_date``, and the feature columns,
    each at its observed value with null hashed as null (see
    ``subject_id_from_row``). The row is assembled before the id is minted
    and never mutated afterwards: there is no backfill path, and any
    difference in any hashed column is a distinct subject.

    Within a single build.py run, the in-memory subjects table deduplicates
    on ``subject_id``: two calls whose rows agree on the hashed columns
    resolve to one subject row — including different raw aliases of the same
    mapped model, which keep the first-seen alias as ``display_name``.

    There is no cross-build deduplication: two builds that assemble the same
    row emit the same ``subject_id`` in their own
    ``{dataset}/subjects.parquet``, so that subject is duplicated across
    folders (each with the aliases that benchmark saw).

    ``normalized_name`` (canonical model), ``provider`` (company), and
    ``release_date`` (the model's first public release) are auto-filled from
    the entry in ``scripts/build_measurement_tables/map_model_registry.json``
    when
    ``raw_label`` matches a registry key byte-for-byte. All stay null for
    unmapped labels (``save`` reports those as an advisory; the nullable
    subject schema preserves the released label without guessing a mapping); a
    label mapped after a build ran derives a different ``subject_id`` on the
    next run, since the registry values enter the hash.

    ``access_date`` (``YYYY-MM-DD`` or ``YYYY-MM``) records when THIS
    benchmark queried the model. Unlike ``release_date`` it is per-benchmark,
    so it is caller-supplied, never registry-filled. It enters the hash like
    every other column: the same model observed at two access dates is two
    subjects.
    """
    feats = _vocab.canonicalize_features(features)
    for k in feats:
        cls = _vocab.KEY_CLASS.get(k)
        if cls not in (None, _vocab.SUBJECT):
            raise ValueError(
                f"resolve_subject: {k!r} is a {cls}-class setting per "
                "normalize_evaluation_settings.py, not a subject feature"
            )
    extra = {
        k: v for k, v in feats.items()
        if k not in _vocab.SUBJECT_FEATURE_COLUMNS
    }

    global _subjects
    with _lock:
        _ensure_init()
        assert _subjects is not None
        entry = _load_model_registry().get(raw_label)

        # The full row is assembled first: every column below (except
        # subject_id/display_name, whose normalized form is the hash base)
        # enters the id, so the id is minted from final content and rows are
        # never mutated after registration. There is no backfill: a label
        # whose registry entry appears later, or a different access_date, is
        # a different subject.
        row = {
            "subject_id": None,
            "display_name": raw_label,
            "normalized_name": entry["model"] if entry else None,
            "provider": entry["company"] if entry else None,
            "release_date": entry.get("release_date") if entry else None,
            "access_date": access_date,
            "harness": feats.get("harness"),
            "reasoning_effort": feats.get("reasoning_effort"),
            "harness_version": feats.get("harness_version"),
            "subject_features_extra": _vocab.features_string(extra),
        }
        sid = subject_id_from_row(raw_label, row)

        if sid in _subjects:
            return sid
        if not auto_register:
            raise UnknownSubject(raw_label)

        row["subject_id"] = sid
        _subjects[sid] = row
        return sid


VERIFIER_CLASSES = ("judge", "exact_matcher")


@dataclass(frozen=True)
class Judge:
    """A verifier applied by judgment: an LLM or a human reads the response
    and decides whether it meets the criterion.  The verdict is an opinion,
    not a computation — running the judge again may disagree with itself.

    ``spec`` describes the execution instructions, judge prompt, or human
    annotation procedure. The item's required grading_criterion supplies the
    reference answer and/or rule that this procedure applies.

    ``judge`` names the judge itself (``"gpt-4o"``, ``"LlamaGuard3"``,
    ``"human"``): the grader is part of the measurement instrument, not of
    the measurement occasion, so its identity lives here — never in
    ``test_condition``. Builds that vary the judge construct the final
    ``Judge`` per item or pass the identity through ``verifier_features``;
    either form splits the item per judge.

    ``judged_by`` distinguishes an LLM judge from human annotators when
    known (``"llm"`` or ``"human"``).

    At least one of ``spec`` / ``judge`` / ``judged_by`` is required.
    """
    spec: str | None = None
    judge: str | None = None
    judged_by: str | None = None

    def __post_init__(self):
        for f in ("spec", "judge"):
            v = getattr(self, f)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(f"Judge.{f} must be a non-empty string or None")
        if self.judged_by not in (None, "llm", "human"):
            raise ValueError(
                f"Judge.judged_by must be 'llm', 'human' or None, "
                f"got {self.judged_by!r}"
            )
        if self.spec is None and self.judge is None and self.judged_by is None:
            raise ValueError(
                "Judge requires at least one of spec / judge / judged_by"
            )


@dataclass(frozen=True)
class ExactMatcher:
    """A verifier applied by a deterministic rule: the verdict is a
    mechanical computation over the response — string or grid equality,
    numeric tolerance, a test suite's pass/fail, a grader script's return
    value.  Running it twice on the same response gives the same verdict.

    ``spec`` describes the implementation that applies grading_criterion:
    the parser/comparator, test harness, JSON execution descriptor, or
    a ``grade()`` function's source. It can consume the criterion's reference answer.
    """
    spec: str

    def __post_init__(self):
        if not isinstance(self.spec, str) or not self.spec.strip():
            raise ValueError("ExactMatcher.spec must be a non-empty string")


def _serialize_verifier(
    verifier: "Judge | ExactMatcher | str | None",
) -> str | None:
    """Canonical JSON for the items.parquet ``verifier`` column:
    ``{"class": "judge"|"exact_matcher", "spec": ..., ...}``.

    Canonicalizes already-serialized JSON for internal registry round-trips.
    Any other string is free text and rejected.
    """
    if verifier is None:
        raise ValueError("verifier is required; provide a Judge or ExactMatcher")
    if isinstance(verifier, Judge):
        payload = {"class": "judge"}
        for f in ("spec", "judge", "judged_by"):
            v = getattr(verifier, f)
            if v is not None:
                payload[f] = v
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if isinstance(verifier, ExactMatcher):
        return json.dumps(
            {"class": "exact_matcher", "spec": verifier.spec},
            sort_keys=True, ensure_ascii=False,
        )
    if isinstance(verifier, str):
        try:
            obj = json.loads(verifier)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("class") in VERIFIER_CLASSES:
            if obj["class"] == "judge":
                Judge(**{key: obj.get(key) for key in ("spec", "judge", "judged_by")})
            else:
                ExactMatcher(spec=obj.get("spec"))
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        raise ValueError(
            "register_item: `verifier` must be a Judge or ExactMatcher "
            "instance, not free text. Wrap the artifact: "
            "Judge(spec=<judge prompt / rubric / annotation protocol>) when "
            "an LLM or human decides, ExactMatcher(spec=<rule or grader "
            "artifact>) when a deterministic rule decides. "
            f"Got: {verifier[:120]!r}"
        )
    raise TypeError(
        "register_item: `verifier` must be Judge or ExactMatcher, "
        f"got {type(verifier).__name__}"
    )


def _merge_verifier_settings(
    serialized: str | None, settings: dict[str, str],
) -> str | None:
    """Merge verifier-class setting keys (``judge``, ``evaluator``, ...)
    into the canonical verifier payload.  With no declared verifier the
    payload defaults to ``{"class": "judge"}`` — a grader-identity key
    implies judged grading; a build whose grader is deterministic should
    declare an ``ExactMatcher`` itself.  A key already present with a
    different value is a hard error, not an overwrite.
    """
    if not settings:
        return serialized
    payload = {"class": "judge"} if serialized is None else json.loads(serialized)
    for k, v in sorted(settings.items()):
        if k in ("class", "spec"):
            raise ValueError(
                f"verifier-class setting {k!r} would shadow the payload's "
                f"{k!r} field"
            )
        if k in payload and payload[k] != v:
            raise ValueError(
                f"item verifier already carries {k}={payload[k]!r} but the "
                f"supplied verifier features say {k}={v!r} — reconcile the build"
            )
        payload[k] = v
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def register_item(
    benchmark_id: str,
    raw_item_id: str,
    content: str | None,
    *,
    grading_criterion: dict | str,
    verifier: "Judge | ExactMatcher | str",
    response_scale: dict | str,
    asset_manifest: str | None = None,
    features: dict | None = None,
    verifier_features: dict | None = None,
) -> str:
    """Register (or look up) an item under a benchmark and return its ``item_id``.

    ``grading_criterion`` is a mapping or JSON object with nullable string
    fields ``reference_answer`` and ``rule``. At least one must be nonempty.
    It defines the answer or condition being assessed; ``verifier`` describes
    the required Judge or ExactMatcher that applies it, including parsing,
    normalization, judge instructions, or implementation details.
    Mixed benchmarks additionally require a concrete ``response_scale`` in
    the criterion; uniform scales are inherited from benchmark metadata.

    Identity includes benchmark, normalized stimulus, item features, assets,
    the complete criterion, verifier, and effective response scale. Supply the
    benchmark's structured response_scale; mixed scales resolve through the
    criterion. Changing the answer, rule, grader, or scale defines a distinct
    item. JSON key order and equivalent numeric scale spellings do not.
    All identity-bearing data must be supplied before responses are added.

    Verifier features are merged into the verifier payload. Supplying a judge
    directly or through verifier features gives the same identity. Item
    features describe presentation settings, such as few-shot examples.
    Re-registering an identity must reproduce the same item data; raw upstream
    aliases and normalization-equivalent content may differ.

    Note: ``test_condition`` is NOT an argument here.  A *condition* — a
    property of the measurement occasion rather than of the stimulus
    (temperature) — lives on the ``responses.parquet``
    row, not on the item; so does an *interactor* — another party in the
    interaction (attacker, user simulator, pairwise opponent) — on the
    ``interactors`` column.  Records of the same prompt under different
    conditions or interactors share this ``item_id`` and are distinguished
    by ``test_condition`` / ``interactors`` on their responses.
    """
    if asset_manifest is not None:
        if not isinstance(asset_manifest, str):
            raise TypeError("register_item: asset_manifest must be a string or None")
        try:
            decoded_manifest = json.loads(asset_manifest)
            canonical_manifest = canonical_asset_manifest(decoded_manifest)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"register_item: invalid asset_manifest — {exc}"
            ) from None
        if canonical_manifest != asset_manifest:
            raise ValueError(
                "register_item: asset_manifest must use canonical compact JSON"
            )
    grading_criterion = canonical_grading_criterion(grading_criterion)
    if response_scale is None:
        raise ValueError("register_item: response_scale is required")
    verifier = _serialize_verifier(verifier)
    feats = _vocab.canonicalize_features(features)
    # Verifier-class keys arrive two ways: recognized by the global vocabulary,
    # or passed explicitly via `verifier_features` when their meaning is
    # benchmark-specific and therefore invisible to this module.
    ver_feats = _vocab.canonicalize_features(verifier_features)
    item_feats = {}
    for k, v in feats.items():
        if k in ver_feats:
            if ver_feats[k] != v:
                raise ValueError(
                    f"register_item: {k!r} passed in both features and "
                    f"verifier_features with different values"
                )
            continue
        if _vocab.KEY_CLASS.get(k) == _vocab.VERIFIER:
            ver_feats[k] = v
        else:
            item_feats[k] = v
    for k in item_feats:
        cls = _vocab.KEY_CLASS.get(k)
        if cls not in (None, _vocab.ITEM):
            raise ValueError(
                f"register_item: {k!r} is a {cls}-class setting per "
                "normalize_evaluation_settings.py, not an item feature"
            )
    verifier = _merge_verifier_settings(verifier, ver_feats)
    feats_str = _vocab.features_string(item_feats)

    global _items
    with _lock:
        _ensure_init()
        assert _items is not None

        hash_input = (
            content
            if content is not None
            else ("" if asset_manifest else f"raw:{raw_item_id}")
        )
        iid = item_id_from_content(
            benchmark_id,
            hash_input,
            feats_str,
            asset_manifest=asset_manifest,
            verifier=verifier,
            grading_criterion=grading_criterion,
            response_scale=response_scale,
        )

        row = {
            "item_id": iid,
            "benchmark_id": benchmark_id,
            "raw_item_id": str(raw_item_id),
            "content": content,
            "asset_manifest": asset_manifest,
            "grading_criterion": grading_criterion,
            "verifier": verifier,
            "content_hash": (
                content_hash(hash_input)
                if content is not None or asset_manifest is None
                else None
            ),
            "item_features": feats_str,
        }
        if iid in _items:
            existing = _items[iid]
            conflicts = []
            for column, value in row.items():
                if column in ("item_id", "raw_item_id"):
                    continue
                prior = existing[column]
                if column == "content":
                    if value is not None:
                        value = unicodedata.normalize("NFC", value).strip()
                    if prior is not None:
                        prior = unicodedata.normalize("NFC", prior).strip()
                if prior != value:
                    conflicts.append(column)
            if conflicts:
                raise ValueError(
                    f"register_item: conflicting registration for item {iid}: "
                    f"{', '.join(conflicts)}; supply one consistent item definition"
                )
            return iid

        _items[iid] = row
        return iid


def get_subject_label(subject_id: str) -> str:
    """Return the first raw label associated with a registered subject.

    KeyError if the id was never registered in this process.
    """
    with _lock:
        _ensure_init()
        assert _subjects is not None
        return _subjects[subject_id]["display_name"]


def set_subject_access_date(subject_id: str, value: str | None) -> str:
    """Replace a registered subject's ``access_date`` and return its new ID.

    Every column enters the subject hash at its observed value, so callers must
    replace references to the old ID. Prefer supplying the final date when the
    subject is first registered. No-op when the date already agrees; KeyError
    if the ID was never registered in this process.
    """
    with _lock:
        _ensure_init()
        assert _subjects is not None
        row = _subjects[subject_id]
        if row["access_date"] == value:
            return subject_id
        row["access_date"] = value
        new_sid = subject_id_from_row(row["display_name"], row)
        del _subjects[subject_id]
        row["subject_id"] = new_sid
        # setdefault: if a subject with this exact row (date included) was
        # already minted directly, fold into it rather than duplicating.
        _subjects.setdefault(new_sid, row)
        return new_sid


def get_subject_registration(subject_id: str) -> dict:
    """Return a copy of a registered subject row.

    KeyError if the ID was never registered in this process.
    """
    with _lock:
        _ensure_init()
        assert _subjects is not None
        return dict(_subjects[subject_id])


def get_item_registration(item_id: str) -> dict:
    """Return a copy of a registered item row.

    KeyError if the ID was never registered in this process.
    """
    with _lock:
        _ensure_init()
        assert _items is not None
        return dict(_items[item_id])


def drop_registrations(
    subject_ids: tuple | list = (), item_ids: tuple | list = ()
) -> None:
    """Remove selected in-process subject and item registrations.

    Unknown IDs are ignored.
    """
    with _lock:
        _ensure_init()
        assert _subjects is not None and _items is not None
        for sid in subject_ids:
            _subjects.pop(sid, None)
        for iid in item_ids:
            _items.pop(iid, None)


def get_benchmark_id(
    benchmark_id: str,
    *,
    name: str | None = None,
    version: str | None = None,
    license: str | None = None,
    source_url: str | None = None,
    description: str | None = None,
    one_line_description: str | None = None,
    modality: list[str] | None = None,
    domain: list[str] | None = None,
    multi_single_turn: str | None = None,
    response_type: str | None = None,
    response_scale: dict | str | None = None,
    categorical: bool | None = None,
    paper_url: str | None = None,
    release_date: str | None = None,
    granularity: str | None = None,
    release: str | None = None,
    benchmark_features: dict | str | None = None,
    n_response_values: int | None = None,
    n_subjects: int | None = None,
    n_items: int | None = None,
    n_responses: int | None = None,
    max_trial: int | None = None,
    coverage: float | None = None,
    has_reference_answer: bool | None = None,
) -> str:
    """Register a benchmark once, or return its id if already registered.

    ``benchmark_id`` is the canonical short key (typically the folder name).
    Kwargs populate the row on first registration. A subsequent call returns
    the same id and may add the complete set of derived build statistics.

    ``modality`` is the list of input modalities required to solve items in
    this benchmark: ``"text"``, ``"image"``, ``"grid"``, ``"gui_screenshot"``,
    ``"audio"``, etc.  Defaults to ``["text"]``.  Use a list so multimodal
    benchmarks (e.g. vision-language QA) can declare multiple. This lower-level
    helper has defaults; ``BenchmarkBuild`` authors still declare the field.

    ``domain`` is the list of subject areas. The canonical vocabulary (19)
    lives in ``define_benchmark_vocabulary.py`` (``DOMAINS``):
    ``"software_engineering"``, ``"ml_engineering"``, ``"mathematics"``,
    ``"science"``, ``"medicine"``, ``"law"``, ``"finance"``,
    ``"cybersecurity"``, ``"cultural"``, ``"education"``, ``"knowledge"``,
    ``"reasoning"``, ``"safety"``, ``"agents_and_tool_use"``, ``"preference"``,
    ``"reward_modeling"``, ``"nlp_task"``, ``"multilingual"``, ``"general"``.
    Defaults to ``["general"]`` at this lower-level API. ``BenchmarkBuild``
    requires a non-empty list but deliberately permits a genuinely new domain
    to be proposed rather than forcing a false category.

    ``description`` is the fuller benchmark overview.
    ``one_line_description`` is its optional, display-ready TL;DR; it remains
    nullable while legacy benchmark definitions are migrated.

    ``response_type`` names how the grader emits the response:
    ``"binary"``, ``"likert_5"``, ``"likert_10"``, ``"win_rate"``,
    ``"ordinal"``, ``"fraction"``, ``"continuous_bounded"``,
    ``"continuous_unbounded"``, ``"error_presence"``, ``"mixed"``.  Defaults
    to ``"binary"``. ``response_scale`` is a structured mapping or JSON object
    declaring discrete values, inclusive interval bounds, or mixed item scales.
    It is stored as canonical JSON, including optional category meanings and
    score direction. These annotations never reverse or rescale grades.
    ``categorical`` is derived for binary, error-presence, Likert, ordinal,
    and continuous types; contradictory declarations fail. Fractions, rates,
    and mixed scales require an explicit boolean for downstream modeling.

    ``multi_single_turn`` is the benchmark's turn structure, decided by what
    inputs the subject receives *after* its first output: none →
    ``"single_turn"``; anything at all → ``"multi_turn"``, whether it comes
    from a counterpart that read the previous output (simulated user, another
    model, a judge asking follow-ups) or from an environment the subject acted
    on (tool results, bash output, a re-rendered page) — so an agent loop is
    ``"multi_turn"``. Few-shot exemplars, long chain-of-thought and repeated
    sampling are all still ``"single_turn"``: none is a new input. A benchmark
    whose items genuinely span both structures declares
    ``"multi_turn, single_turn"`` (e.g. bfcl mixes single-call categories with
    agent-loop ``multi_turn_*`` categories). There is no author-facing default:
    new builds must declare it. Historical omissions print a targeted warning
    and remain null, which the current non-null parquet schema rejects at save.

    ``granularity`` declares what response data upstream released, so consumers
    can distinguish a true response matrix from an item-bank-only folder:

      * ``"item"`` (default) — each item is a single example/question; a
        response is one (subject, item) cell.
      * ``"aggregate"`` — upstream released only aggregate scores. The build
        still registers the real item bank and reported subjects, but emits no
        ``responses.parquet``; aggregates are never expanded into item rows.
      * ``"not_released"`` — upstream released no per-(subject, item) model
        data at all; items may be registered as an item bank but no response
        matrix exists.

    ``release`` is the public/private publish gate — ``"public"`` makes the
    benchmark eligible for the public repo/HuggingFace mirror, ``"private"``
    keeps it private. It is a deliberate decision, not inferable from the data,
    declared per-benchmark in build.py's ``INFO`` dict. It is the single source
    of truth for the two-repo split (this replaced the old ``manifest.yaml``).
    Omitting it defaults to ``"private"``; mark publishable datasets ``"public"``.

    Modern ``BenchmarkBuild`` callers register the benchmark only after all
    subject, item, and response rows are known. They pass the seven completed
    table statistics above so ``benchmarks.parquet`` can be validated and
    written once. Lower-level legacy callers may omit them and retain the
    historical non-derived benchmark schema.

    ``n_response_values`` counts distinct non-null grades; ``max_trial`` is
    the maximum repetition index, not a trial count. ``coverage`` is the
    fraction of registered subject-item pairs with at least one non-null
    grade; repeated observations of a pair count only once.
    ``has_reference_answer`` indicates that at least one item has a reference
    answer; a grading rule alone does not make this flag true.
    """
    from .response_scales import canonical_response_scale, resolve_categorical, validate_scale_type
    if response_scale is None and response_type not in (None, "binary", "error_presence"):
        raise ValueError("an explicit response_scale is required for non-binary benchmarks")
    response_scale = canonical_response_scale(
        response_scale if response_scale is not None
        else {"kind": "discrete", "values": [0, 1]}
    )
    validate_scale_type(response_type or "binary", response_scale)
    categorical = resolve_categorical(response_type or "binary", categorical)
    if isinstance(benchmark_features, dict):
        benchmark_features = _vocab.features_string(
            _vocab.canonicalize_features(benchmark_features))
    derived_statistics = {
        "n_subjects": n_subjects,
        "n_items": n_items,
        "n_responses": n_responses,
        "max_trial": max_trial,
        "coverage": coverage,
        "has_reference_answer": has_reference_answer,
    }
    has_derived_statistics = any(
        value is not None for value in derived_statistics.values()
    )
    if has_derived_statistics and not all(
        value is not None for value in derived_statistics.values()
    ):
        raise ValueError(
            f"{benchmark_id}: derived benchmark statistics must be supplied "
            "together"
        )
    if has_derived_statistics and n_response_values is None:
        raise ValueError(
            f"{benchmark_id}: n_response_values is required with derived "
            "benchmark statistics"
        )
    global _benchmarks
    with _lock:
        _ensure_init()
        assert _benchmarks is not None

        if benchmark_id in _benchmarks:
            if n_response_values is not None:
                _benchmarks[benchmark_id]["n_response_values"] = (
                    n_response_values
                )
            if has_derived_statistics:
                _benchmarks[benchmark_id].update(derived_statistics)
            return benchmark_id

        if multi_single_turn not in _MULTI_SINGLE_TURN:
            sys.stderr.write(
                f" {benchmark_id}: multi_single_turn is "
                f"{multi_single_turn!r}; declare INFO['multi_single_turn'] as "
                f"one of {sorted(_MULTI_SINGLE_TURN)} (no default — an agent "
                f"loop counts as 'multi_turn')\n"
            )

        benchmark_row = {
            "benchmark_id": benchmark_id,
            "name": name or benchmark_id,
            "version": version,
            "license": license,
            "source_url": source_url,
            "description": description,
            "one_line_description": one_line_description,
            "modality": list(modality) if modality else ["text"],
            "domain": list(domain) if domain else ["general"],
            # Turn structure. Deliberately has NO default: "single_turn" is not a
            # safe fallback (an agent loop feeding observations back is
            # "multi_turn"), so an undeclared value stays null and warns rather
            # than mislabelling the benchmark.
            "multi_single_turn": multi_single_turn,
            "response_type": response_type or "binary",
            "response_scale": response_scale,
            "categorical": categorical,
            "paper_url": paper_url,
            "release_date": release_date,
            "granularity": granularity or "item",
            # Publish gate. ``release`` is the publish-to-public decision,
            # declared in build.py's INFO dict. It is NOT inferable from the
            # data, so a build that omits it defaults to "private" (nothing is
            # published without an explicit decision); mark publishable
            # datasets "public" explicitly.
            "release": release or "private",
            "n_response_values": n_response_values,
            "benchmark_features": benchmark_features,
        }
        if has_derived_statistics:
            benchmark_row.update(derived_statistics)
        _benchmarks[benchmark_id] = benchmark_row
        return benchmark_id


def set_response_stats(benchmark_id: str, n_response_values: int | None) -> None:
    """Record post-build statistics on a benchmark's registry row.

    Retained for lower-level legacy callers that register a benchmark before
    its responses are known. Modern ``BenchmarkBuild`` passes this statistic
    during its single late benchmark registration. No-op if the benchmark was
    never registered this process.
    """
    with _lock:
        _ensure_init()
        assert _benchmarks is not None
        if benchmark_id in _benchmarks:
            _benchmarks[benchmark_id]["n_response_values"] = n_response_values


def set_benchmark_granularity(benchmark_id: str, granularity: str) -> None:
    """Update a pending benchmark row before it is validated and written."""
    with _lock:
        _ensure_init()
        assert _benchmarks is not None
        if benchmark_id in _benchmarks:
            _benchmarks[benchmark_id]["granularity"] = granularity


def registration_counts() -> dict[str, int]:
    """Return row and item-property counts from the in-memory registry."""
    with _lock:
        _ensure_init()
        assert _subjects is not None and _items is not None and _benchmarks is not None
        return {
            "subjects": len(_subjects),
            "items": len(_items),
            "items_with_content": sum(
                isinstance(row.get("content"), str)
                and bool(row["content"].strip())
                for row in _items.values()
            ),
            "items_with_stimulus": sum(
                (
                    isinstance(row.get("content"), str)
                    and bool(row["content"].strip())
                )
                or (
                    isinstance(row.get("asset_manifest"), str)
                    and bool(row["asset_manifest"])
                )
                for row in _items.values()
            ),
            "items_with_reference_answer": sum(
                json.loads(row["grading_criterion"])["reference_answer"] is not None
                for row in _items.values()
            ),
            "benchmarks": len(_benchmarks),
        }


def reload() -> None:
    """Reset the in-process state — mainly for tests."""
    global _subjects, _items, _benchmarks
    with _lock:
        _subjects = _items = _benchmarks = None
