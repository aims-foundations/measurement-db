#!/usr/bin/env python3
"""Curate pinned Multi-SWE-bench releases using the shared registration contract."""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge

import re


def cache_path(path: str) -> str:
    """Escape legacy filenames to the portable raw-manifest path alphabet."""
    return re.sub(r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", path)


def _instance_id(value: str) -> str:
    """Accept the native grader's org/repo:pr-number and released instance IDs."""
    return re.sub(r"^([^/]+)/([^:]+):pr-(\d+)$", r"\1__\2-\3", value)


CONTENT_CAP = 12000

# Cap for the reference patch (reference_answer) and the test-patch diff carried
# in `verifier`. Matches the sibling swebench / swebench_multilingual builds,
# which truncate the same upstream `fix_patch` field the same way.
PATCH_CAP = 4000

# Cap on how many PASS_TO_PASS test names travel in `verifier`. Some repos'
# full regression suites run to hundreds of thousands of characters as a
# JSON list; the true count is kept alongside the capped sample so the size
# of the held-constant suite is not lost even when the list is.
TEST_LIST_CAP = 50


def _truncate_patch(p: str | None, limit: int = PATCH_CAP) -> str | None:
    """Truncate a reference/test patch string to at most ``limit`` chars."""
    if p is None:
        return None
    s = str(p)
    if not s:
        return None
    if len(s) <= limit:
        return s
    suffix = "\n...[truncated]"
    return s[: max(0, limit - len(suffix))] + suffix


def _extract_test_list(rec: dict, std_key: str, dict_key: str) -> list[str]:
    """Test names for FAIL_TO_PASS / PASS_TO_PASS. The python item bank carries
    the standard SWE-bench list field (``std_key``, sometimes JSON-encoded as a
    string); every language carries Multi-SWE-bench's own per-test status dict
    (``dict_key``, e.g. ``f2p_tests``) whose keys are the test names -- used as
    a fallback when the standard field is absent."""
    raw = rec.get(std_key)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    d = rec.get(dict_key)
    if isinstance(d, dict) and d:
        return sorted(d.keys())
    return []


def _build_verifier(rec: dict) -> str | None:
    """JSON descriptor of the grading artifact for one instance: the test-suite
    diff to apply (``test_patch``) plus the FAIL_TO_PASS / PASS_TO_PASS test
    names that must flip to (resp. remain) passing -- the harness inputs
    Multi-SWE-bench (and SWE-bench before it) actually grades against. The
    reference fix itself (``fix_patch``) is stored separately as
    ``reference_answer``."""
    test_patch = _truncate_patch(rec.get("test_patch"))
    f2p = _extract_test_list(rec, "FAIL_TO_PASS", "f2p_tests")
    p2p = _extract_test_list(rec, "PASS_TO_PASS", "p2p_tests")
    if not (test_patch or f2p or p2p):
        return None
    return json.dumps({
        "test_patch": test_patch,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p[:TEST_LIST_CAP],
        "n_pass_to_pass": len(p2p),
    })


def _build_problem_statement(rec: dict) -> str | None:
    """Construct the problem statement an issue-resolving model sees from a
    Multi-SWE-bench item-bank record: the linked resolved issues (title + body),
    falling back to the PR title/body."""
    parts: list[str] = []
    for issue in rec.get("resolved_issues") or []:
        if not isinstance(issue, dict):
            continue
        t = (issue.get("title") or "").strip()
        b = (issue.get("body") or "").strip()
        seg = "\n".join(x for x in (t, b) if x)
        if seg:
            parts.append(seg)
    text = "\n\n".join(parts).strip()
    if not text:
        t = (rec.get("title") or "").strip()
        b = (rec.get("body") or "").strip()
        text = "\n".join(x for x in (t, b) if x).strip()
    if not text:
        return None
    return text[:CONTENT_CAP]


def _parse_folder(folder: str) -> tuple[str | None, str | None, str]:
    """``<date>_<agent>_<model>`` -> (access_date, scaffold, model).

    The 8-digit prefix is the submission's run date (``20250329`` ->
    ``2025-03-29``). Scaffold and model are split at the source rather than
    joined into one label: the model part goes to `resolve_subject` verbatim
    (a dated snapshot marker like ``Claude-3.5-Sonnet(Oct)`` is part of the
    model name), the scaffold to ``settings["agent"]``. A body with no
    ``_`` separator (upstream's ``20251027_iSWE-OpenModels``) is a whole
    subject label with no scaffold -> (date, None, body).
    """
    head, _, rest = folder.partition("_")
    if head.isdigit() and len(head) == 8 and rest:
        date = f"{head[:4]}-{head[4:6]}-{head[6:]}"
        body = rest
    else:
        date = None
        body = folder
    agent, sep, model = body.partition("_")
    if not sep:
        return date, None, body
    return date, agent, model



class MultiSWEBench(BenchmarkBuild):
    def _load_item_data(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Map instance_id -> (problem statement, reference_answer, verifier) from
        every cached item-bank JSONL, plus the SWE-bench Verified parquet for
        python problem statements.

        Every item-bank record (all 8 languages) carries the standard
        SWE-bench grading schema alongside the issue text: ``fix_patch`` (the
        reference patch that resolves the issue -- the criterion the
        recorded resolved/unresolved response was scored against) and
        ``test_patch`` + ``FAIL_TO_PASS``/``PASS_TO_PASS`` (or their
        Multi-SWE-bench equivalents ``f2p_tests``/``p2p_tests``) -- the test
        suite that applies that criterion. Both are read here alongside the
        text `_build_problem_statement` already extracts.
        """
        content: dict[str, str] = {}
        correct: dict[str, str] = {}
        verifier: dict[str, str] = {}
        for path in sorted(self.raw_dir / name for name in self.source_files
                           if "/" not in name and name.endswith(".jsonl")):
            if path.stat().st_size == 0:
                continue
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    iid = rec.get("instance_id")
                    if not iid:
                        org, repo, num = rec.get("org"), rec.get("repo"), rec.get("number")
                        if org and repo and num is not None:
                            iid = f"{org}__{repo}-{num}"
                    if not iid:
                        continue
                    text = _build_problem_statement(rec)
                    if text:
                        content[iid] = text
                    fix_patch = _truncate_patch(rec.get("fix_patch") or rec.get("patch"))
                    if fix_patch and iid not in correct:
                        correct[iid] = fix_patch
                    v = _build_verifier(rec)
                    if v and iid not in verifier:
                        verifier[iid] = v

        # Python problem statements from SWE-bench Verified.
        pq = self.raw_dir / "swebench_verified.parquet"
        if pq.exists() and pq.stat().st_size > 0:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["instance_id", "problem_statement"])
            for iid, ps in zip(df["instance_id"], df["problem_statement"]):
                if iid and isinstance(ps, str) and ps.strip() and iid not in content:
                    content[str(iid)] = ps.strip()[:CONTENT_CAP]
        return content, correct, verifier

    def _load_preds(self, stem: str) -> dict[str, str]:
        """instance_id -> model output patch text for one submission folder,
        read from its cached all_preds.jsonl (matched by the results.json stem)."""
        preds: dict[str, str] = {}
        path = self.raw_dir / cache_path(f"preds/{stem}.jsonl")
        if not (path.exists() and path.stat().st_size > 2):
            return preds
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                iid = rec.get("instance_id")
                patch = rec.get("model_patch")
                if iid and isinstance(patch, str) and patch.strip():
                    preds[iid] = patch
        return preds


    def build_subject_item_response_rows(self) -> None:
        content, answers, verifiers = self._load_item_data()
        sources = []
        earliest = {}
        for local in self.source_files:
            original = re.sub(r"_x([0-9a-f]{2})_", lambda m: chr(int(m[1], 16)), local)
            if not original.startswith("results/"):
                continue
            stem = Path(original).stem
            lang, folder = stem.split("__", 1)
            record = json.loads((self.raw_dir / local).read_text())
            resolved = {_instance_id(value) for value in (record.get("resolved") or record.get("resolved_ids") or [])}
            unresolved = {_instance_id(value) for value in (record.get("unresolved_ids") or record.get("unresolved") or [])}
            if not resolved and not unresolved:
                continue
            date, agent, model = _parse_folder(folder)
            sources.append((original, stem, lang, model, agent, resolved, unresolved))
            if date:
                earliest[model, agent] = min(date, earliest.get((model, agent), date))
        items = {}
        trials = Counter()
        for _, stem, lang, model, agent, resolved, unresolved in sorted(sources):
            subject = self.add_subject(model, features={"harness": agent} if agent else None,
                                       access_date=earliest.get((model, agent)))
            preds = self._load_preds(stem)
            for raw_id in sorted(resolved | unresolved):
                key = (raw_id, lang)
                if key not in items:
                    items[key] = self.add_item(
                        raw_item_id=raw_id, content=content[raw_id], features={"lang": lang},
                        grading_criterion={"reference_answer": answers.get(raw_id),
                                           "rule": "The patch must pass the released FAIL_TO_PASS tests and preserve PASS_TO_PASS tests."},
                        verifier=ExactMatcher(spec=verifiers[raw_id]),
                    )
                item = items[key]
                trials[subject, item] += 1
                self.add_response(subject_id=subject, item_id=item, trial=trials[subject, item],
                                  response=float(raw_id in resolved), trace=preds.get(raw_id))


if __name__ == "__main__":
    MultiSWEBench(__file__).main_from_args()
