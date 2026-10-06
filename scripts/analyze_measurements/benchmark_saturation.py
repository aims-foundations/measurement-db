#!/usr/bin/env python3
"""Analyze whether locally built benchmarks are saturated.

Saturation is an interpretation of a response matrix, not a property of the
provider data.  This script therefore writes a regenerable analysis artifact
instead of modifying any benchmark's collection tables.

The current rule marks a benchmark as saturated when its best subject's mean
response, normalized to ``[0, 1]`` using the declared response scale, is at
least 0.9.  Scales without one meaningful increasing bounded direction do not
receive a verdict.

Usage::

    python scripts/analyze_measurements/benchmark_saturation.py
    python scripts/analyze_measurements/benchmark_saturation.py editbench mmlu
    python scripts/analyze_measurements/benchmark_saturation.py --out results.csv
    python scripts/analyze_measurements/benchmark_saturation.py --emit-web

With no benchmark slugs, every local folder containing ``benchmarks.parquet``
is analyzed.  The default output is
``artifacts/analyze_measurements/benchmark_saturation/summary.csv``.
``--emit-web`` additionally writes
``website/content/generated/benchmark-saturation.json`` as an object mapping
benchmark folder slugs to ``true`` (saturated), ``false`` (not saturated), or
``null`` (inconclusive).
"""

from __future__ import annotations

from pathlib import Path

import argparse
import json
import math
import re

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO / "benchmarks"
DEFAULT_OUTPUT = (
    REPO
    / "artifacts"
    / "analyze_measurements"
    / "benchmark_saturation"
    / "summary.csv"
)
WEB_OUTPUT = REPO / "website" / "content" / "generated" / "benchmark-saturation.json"

_SATURATION_THRESHOLD = 0.9
_SCALE_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE]-?\d+)?")


def _saturation_scale_bounds(
    response_type: str, response_scale: str
) -> tuple[float, float] | None:
    """Return bounds for normalizing responses, or ``None`` if incoherent."""
    if response_type == "continuous_unbounded":
        return None
    if response_type == "error_presence":
        # One means an error is present, so higher is worse.
        return None
    if response_type == "mixed":
        # Different metrics need not share bounds or direction.
        return None

    nums = [float(value) for value in _SCALE_NUM_RE.findall(response_scale or "")]
    nums = [value for value in nums if math.isfinite(value)]
    if len(nums) >= 2 and max(nums) > min(nums):
        return min(nums), max(nums)
    if response_type in ("binary", "fraction", "win_rate"):
        return 0.0, 1.0
    return None


def compute_saturation(
    responses: pd.DataFrame | None,
    granularity: object,
    response_type: str,
    response_scale: str,
) -> float:
    """Return a benchmark's saturation verdict.

    Returns ``1.0`` when the best normalized subject mean reaches 0.9, ``0.0``
    when it does not, and ``NaN`` when the available measurements do not
    support this analysis.  ``responses`` needs only ``subject_id`` and
    ``response`` columns.
    """
    nan = float("nan")
    if granularity != "item" or responses is None or responses.empty:
        return nan

    observations = pd.DataFrame(
        {
            "subject_id": responses["subject_id"],
            "response": pd.to_numeric(responses["response"], errors="coerce"),
        }
    ).dropna(subset=["response"])
    if observations.empty:
        return nan

    bounds = _saturation_scale_bounds(response_type, response_scale)
    if bounds is None:
        return nan

    lower, upper = bounds
    subject_means = observations.groupby("subject_id")["response"].mean()
    best = ((subject_means - lower) / (upper - lower)).clip(0.0, 1.0).max()
    return 1.0 if best >= _SATURATION_THRESHOLD else 0.0


def analyze_benchmark(folder: Path) -> dict[str, object]:
    """Compute one analysis row from a locally built benchmark folder."""
    tables_dir = folder / "formatted_tables"
    if not tables_dir.is_dir():
        tables_dir = folder
    benchmark_path = tables_dir / "benchmarks.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"no benchmarks.parquet in {folder}")

    benchmarks = pd.read_parquet(benchmark_path)
    if len(benchmarks) != 1:
        raise ValueError(
            f"expected one benchmark row in {benchmark_path}, found {len(benchmarks)}"
        )
    benchmark = benchmarks.iloc[0]

    responses = None
    response_error = None
    if benchmark.get("granularity") == "item":
        try:
            path = tables_dir / "responses.parquet"
            responses = pd.read_parquet(
                path if path.exists() else tables_dir / "response.parquet",
                columns=["subject_id", "response"],
            )
        except Exception as exc:  # Preserve a missing-evidence row in the rollup.
            response_error = str(exc)

    response_type = str(benchmark.get("response_type") or "")
    response_scale = str(benchmark.get("response_scale") or "")
    saturation = compute_saturation(
        responses,
        granularity=benchmark.get("granularity"),
        response_type=response_type,
        response_scale=response_scale,
    )
    return {
        "slug": folder.name,
        "benchmark_id": str(benchmark.get("benchmark_id") or folder.name),
        "name": benchmark.get("name"),
        "granularity": benchmark.get("granularity"),
        "response_type": response_type,
        "response_scale": response_scale,
        "saturation": saturation,
        "response_error": response_error,
    }


def analyze_folders(folders: list[Path]) -> pd.DataFrame:
    """Analyze folders in deterministic benchmark-id order."""
    rows = [analyze_benchmark(folder) for folder in folders]
    columns = [
        "slug",
        "benchmark_id",
        "name",
        "granularity",
        "response_type",
        "response_scale",
        "saturation",
        "response_error",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("slug", ignore_index=True)


def web_payload(results: pd.DataFrame) -> dict[str, bool | None]:
    """Convert analysis rows to the website's strict tri-state JSON payload."""
    if results["slug"].duplicated().any():
        duplicates = sorted(results.loc[results["slug"].duplicated(), "slug"].unique())
        raise ValueError(f"duplicate benchmark slugs: {', '.join(duplicates)}")

    payload: dict[str, bool | None] = {}
    for row in results.itertuples(index=False):
        if pd.isna(row.saturation):
            value = None
        elif row.saturation == 1.0:
            value = True
        elif row.saturation == 0.0:
            value = False
        else:
            raise ValueError(
                f"invalid saturation verdict for {row.slug!r}: "
                f"{row.saturation!r}"
            )
        payload[str(row.slug)] = value
    return payload


def write_web_output(results: pd.DataFrame, path: Path = WEB_OUTPUT, replace: bool = False) -> None:
    """Merge analyzed slugs into the website artifact.

    Retaining entries for slugs outside ``results`` makes a targeted analysis
    run safe: analyzing one benchmark cannot erase every other benchmark's
    previously generated verdict.
    """
    if path.exists() and not replace:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            not isinstance(slug, str)
            or (value is not None and not isinstance(value, bool))
            for slug, value in payload.items()
        ):
            raise ValueError(f"expected a slug-to-boolean/null object in {path}")
    else:
        payload = {}
    payload.update(web_payload(results))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        metavar="SLUG",
        help="benchmark folders to analyze (default: every locally built benchmark)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"analysis CSV path (default: {DEFAULT_OUTPUT.relative_to(REPO)})",
    )
    parser.add_argument(
        "--emit-web",
        action="store_true",
        help=f"also write {WEB_OUTPUT.relative_to(REPO)}",
    )
    parser.add_argument("--benchmarks-dir", type=Path, default=BENCHMARKS_DIR)
    parser.add_argument("--web-output", type=Path, default=WEB_OUTPUT)
    parser.add_argument("--replace", action="store_true", help="replace the website artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.benchmarks:
        folders = [args.benchmarks_dir / slug for slug in args.benchmarks]
    else:
        folders = []
        for folder in sorted(args.benchmarks_dir.glob("*")):
            tables_dir = folder / "formatted_tables"
            if not tables_dir.is_dir():
                tables_dir = folder
            if (tables_dir / "benchmarks.parquet").is_file():
                folders.append(folder)

    results = analyze_folders(folders)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    if args.emit_web:
        write_web_output(results, args.web_output, replace=args.replace)

    saturated = int(results["saturation"].eq(1.0).sum())
    unsaturated = int(results["saturation"].eq(0.0).sum())
    inconclusive = int(results["saturation"].isna().sum())
    print(
        f"Wrote {len(results)} benchmarks to {args.out}: "
        f"{saturated} saturated, {unsaturated} not saturated, "
        f"{inconclusive} inconclusive"
    )
    if args.emit_web:
        print(f"Wrote website saturation data to {args.web_output}")


if __name__ == "__main__":
    main()
