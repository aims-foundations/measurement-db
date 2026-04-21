"""
AEGIS audit (processed). Categories from 02_build_response_matrix.AEGIS_CATEGORIES; no PII checks.

aegis_full.csv: id unique; prompt_label and response_label safe or unsafe; each comma-separated token in violated_categories is an allowed category name.
aegis_labels.csv: columns id, prompt_label, response_label, violated_categories, _split in that order; same id and label rules as aegis_full; _split is train, test, or validation.
response_matrix.csv: first column sample ids unique nonempty; remaining columns match AEGIS_CATEGORIES; cells 0 or 1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import bad_pct_suffix, path_errors_if_missing, processed_dir_from_script  # noqa: E402

EXPECTED_LABEL_COLUMNS = [
    "id",
    "prompt_label",
    "response_label",
    "violated_categories",
    "_split",
]

REQUIRED_FULL_COLUMNS = [
    "id",
    "violated_categories",
    "prompt_label",
    "response_label",
]

SPLITS_OK = frozenset({"train", "test", "validation"})
PROMPT_LABEL_OK = frozenset({"safe", "unsafe"})
RESPONSE_LABEL_OK = frozenset({"safe", "unsafe"})


def _bad_required_safe_unsafe(series: pd.Series, ok: frozenset[str]) -> pd.Series:
    """True where value is missing, empty after strip, or not exactly one of ``ok`` (no strip on match)."""
    return series.isna() | (series.astype(str).str.strip() == "") | ~series.astype(str).isin(ok)


def _load_aegis_categories() -> tuple[str, ...]:
    """Load ``AEGIS_CATEGORIES`` from the sibling build script (single source of truth)."""
    build_path = Path(__file__).resolve().parent / "02_build_response_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "aegis_build_response_matrix_audit", build_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load build script: {build_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cats = getattr(mod, "AEGIS_CATEGORIES", None)
    if not isinstance(cats, list) or not cats:
        raise RuntimeError("02_build_response_matrix.py must define non-empty AEGIS_CATEGORIES")
    return tuple(str(c) for c in cats)


def _parse_violated_categories(val: object) -> list[str]:
    """Same parsing rules as ``02_build_response_matrix.parse_categories``."""
    if pd.isna(val):
        return []
    s = str(val)
    st = s.strip()
    if st == "" or st.lower() in ("none", "nan", "safe"):
        return []
    return [c for c in s.split(",") if c != ""]


def _audit_violated_categories_column(
    *,
    label: str,
    series: pd.Series,
    allowed: frozenset[str],
    n: int,
    errors: list[str],
) -> None:
    """Ensure every comma-separated token is in ``allowed``."""
    unknown_per_row: list[frozenset[str]] = []
    for val in series:
        tokens = _parse_violated_categories(val)
        bad = frozenset(t for t in tokens if t not in allowed)
        if bad:
            unknown_per_row.append(bad)

    if not unknown_per_row:
        return

    all_unknown = set()
    for b in unknown_per_row:
        all_unknown.update(b)
    n_bad_rows = len(unknown_per_row)
    preview = ", ".join(repr(x) for x in sorted(all_unknown)[:8])
    extra = " …" if len(all_unknown) > 8 else ""
    errors.append(
        f"{label}: violated_categories contains tokens not in AEGIS_CATEGORIES "
        f"({preview}{extra})"
        + bad_pct_suffix(n_bad_rows, n)
    )


def _audit_aegis_full(df: pd.DataFrame, allowed: frozenset[str], errors: list[str]) -> None:
    n = len(df)
    label = "aegis_full.csv"
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    missing = [c for c in REQUIRED_FULL_COLUMNS if c not in df.columns]
    if missing:
        errors.append(
            f"{label}: missing required columns {missing}; have {list(df.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    bad_id = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
    if bad_id.any():
        errors.append(
            f"{label}: id must be non-empty" + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df["id"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate id values" + bad_pct_suffix(int(dup.sum()), n)
        )

    bad_pl = _bad_required_safe_unsafe(df["prompt_label"], PROMPT_LABEL_OK)
    if bad_pl.any():
        errors.append(
            f"{label}: prompt_label must be exactly 'safe' or 'unsafe' (not empty)"
            + bad_pct_suffix(int(bad_pl.sum()), n)
        )

    bad_rl = _bad_required_safe_unsafe(df["response_label"], RESPONSE_LABEL_OK)
    if bad_rl.any():
        errors.append(
            f"{label}: response_label must be exactly 'safe' or 'unsafe' (not empty)"
            + bad_pct_suffix(int(bad_rl.sum()), n)
        )

    _audit_violated_categories_column(
        label=label,
        series=df["violated_categories"],
        allowed=allowed,
        n=n,
        errors=errors,
    )


def _audit_aegis_labels(df: pd.DataFrame, allowed: frozenset[str], errors: list[str]) -> None:
    n = len(df)
    label = "aegis_labels.csv"
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    if list(df.columns) != EXPECTED_LABEL_COLUMNS:
        errors.append(
            f"{label}: columns mismatch; expected {EXPECTED_LABEL_COLUMNS}, got {list(df.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    bad_id = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
    if bad_id.any():
        errors.append(
            f"{label}: id must be non-empty" + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df["id"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate id values" + bad_pct_suffix(int(dup.sum()), n)
        )

    bad_split = (
        df["_split"].isna()
        | (df["_split"].astype(str).str.strip() == "")
        | ~df["_split"].astype(str).isin(SPLITS_OK)
    )
    if bad_split.any():
        errors.append(
            f"{label}: _split must be 'train', 'test', or 'validation'"
            + bad_pct_suffix(int(bad_split.sum()), n)
        )

    bad_pl = _bad_required_safe_unsafe(df["prompt_label"], PROMPT_LABEL_OK)
    if bad_pl.any():
        errors.append(
            f"{label}: prompt_label must be exactly 'safe' or 'unsafe' (not empty)"
            + bad_pct_suffix(int(bad_pl.sum()), n)
        )

    bad_rl = _bad_required_safe_unsafe(df["response_label"], RESPONSE_LABEL_OK)
    if bad_rl.any():
        errors.append(
            f"{label}: response_label must be exactly 'safe' or 'unsafe' (not empty)"
            + bad_pct_suffix(int(bad_rl.sum()), n)
        )

    _audit_violated_categories_column(
        label=label,
        series=df["violated_categories"],
        allowed=allowed,
        n=n,
        errors=errors,
    )


def _audit_response_matrix(path: Path, expected_cats: tuple[str, ...], errors: list[str]) -> None:
    label = path.name
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        errors.append(f"{label}: cannot read CSV ({exc!r})")
        return

    n = len(df)
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    id_col = df.columns[0]
    cat_cols = list(df.columns[1:])
    allowed_set = frozenset(expected_cats)

    if set(cat_cols) != allowed_set:
        missing = sorted(allowed_set - set(cat_cols))
        extra = sorted(set(cat_cols) - allowed_set)
        errors.append(
            f"{label}: category columns must match AEGIS_CATEGORIES exactly "
            f"(missing {missing!r}, extra {extra!r})"
            + bad_pct_suffix(n, n)
        )
        return

    bad_id = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    if bad_id.any():
        errors.append(
            f"{label}: sample id column {id_col!r} must be non-empty"
            + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate sample id values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    n_cells = n * len(cat_cols)
    bad_empty = 0
    bad_not_exact_01 = 0
    for c in cat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(arr)
        bad_empty += int((~finite).sum())
        bad_not_exact_01 += int((finite & (arr != 0.0) & (arr != 1.0)).sum())

    if bad_empty:
        pct = 100.0 * bad_empty / n_cells
        errors.append(
            f"{label}: matrix cells must be non-empty and finite"
            f" — malformed cells: {bad_empty:,} / {n_cells:,} ({pct:.2f}%)"
        )
    if bad_not_exact_01:
        pct = 100.0 * bad_not_exact_01 / n_cells
        errors.append(
            f"{label}: matrix cells must be exactly 0 or 1"
            f" — malformed cells: {bad_not_exact_01:,} / {n_cells:,} ({pct:.2f}%)"
        )


def main() -> int:
    proc = processed_dir_from_script(__file__)
    paths = {
        "aegis_full.csv": proc / "aegis_full.csv",
        "aegis_labels.csv": proc / "aegis_labels.csv",
        "response_matrix.csv": proc / "response_matrix.csv",
    }
    missing_errs = path_errors_if_missing(paths.values())
    if missing_errs:
        for e in missing_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        expected_cats = _load_aegis_categories()
    except Exception as exc:
        print(f"ERROR: cannot load AEGIS_CATEGORIES from build script: {exc}", file=sys.stderr)
        return 1

    allowed = frozenset(expected_cats)
    errors: list[str] = []

    for name in ("aegis_full.csv", "aegis_labels.csv"):
        p = paths[name]
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{name}: cannot read CSV ({exc!r})")
            continue
        if name == "aegis_full.csv":
            _audit_aegis_full(df, allowed, errors)
        else:
            _audit_aegis_labels(df, allowed, errors)

    _audit_response_matrix(paths["response_matrix.csv"], expected_cats, errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(
        "OK: AEGIS — aegis_full.csv, aegis_labels.csv, response_matrix.csv passed checks "
        f"({len(expected_cats)} canonical categories)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
