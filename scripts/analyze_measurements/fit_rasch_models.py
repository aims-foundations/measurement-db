#!/usr/bin/env python3
"""
Rasch (1PL) IRT fit for the benchmarks published on the website.

    p(correct) = sigmoid( theta[subject] - z[item] + c[test_condition] )

theta is the subject's ability, z the item's difficulty, and c a free effect per
distinct condition label — the `test_condition` string plus, on benchmarks
built since the interactor split, the `interactors` string (attacker,
user simulator, opponent). The
condition term is DROPPED when a benchmark has a single condition, so those
benchmarks are fitted with the plain sigmoid(theta - z).

theta and c are mean-centered after every step for identifiability; z is left
free so it absorbs the global intercept.

Responses use an 80/20 random split over observations. Sparse subjects or
items can occur only in the held-out split.

Only benchmarks whose responses are natively 0/1 are fitted. Graded benchmarks
(writingbench, mtbench, arena_*, ...) are skipped with a reason rather than
thresholded into a dichotomy the benchmark never defined.

Usage
-----
    # one benchmark
    python scripts/analyze_measurements/fit_rasch_models.py helm_harmbench

    # several
    python scripts/analyze_measurements/fit_rasch_models.py helm_harmbench afrimedqa mmlu

    # every benchmark currently on the website (published-benchmarks.json)
    python scripts/analyze_measurements/fit_rasch_models.py --all

    # ... and refresh the numbers the website shows
    python scripts/analyze_measurements/fit_rasch_models.py --all --emit-web

Outputs (per benchmark, default benchmarks/<slug>/model_fits/rasch1pl/)
------------------------------------------------------------------------
    responses.csv   every fitted observation: the data as read, its fitted
                    probability, and `match` — agreement between the IRT
                    prediction and the actual response, "-" on train rows
    subjects.csv    subject_id, display name, theta, n, observed accuracy
    items.csv       item_id, z (difficulty), n, observed solve rate
    conditions.csv  test_condition, effect, n          (multi-condition only)
    summary.json    AUC / log-likelihood on train and test, sizes, params
    history.csv     the same four metrics per iteration
    curves.png      log-likelihood and AUC per iteration (--no-curves to skip)

and, across benchmarks:
    artifacts/analyze_measurements/model_fits/rasch1pl/summary.csv
                                      one row per locally present fit
    website/content/generated/benchmark-irt.json  (--emit-web) what the site renders

Pass --out DIR to use the common-root layout DIR/<slug>/ instead, with the
cross-benchmark rollup at DIR/summary.csv.

Requires torch, scikit-learn, pandas, pyarrow, and huggingface_hub.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
RENDER_WEBSITE_DIR = REPO / "scripts" / "render_website"
# Reuse the gallery's HF source configuration and SDK downloads.
sys.path.insert(0, str(RENDER_WEBSITE_DIR))
from generate_benchmark_gallery import fetch  # noqa: E402
import generate_benchmark_gallery as gallery  # noqa: E402

FILTERED = REPO / "website" / "content" / "generated" / "published-benchmarks.json"
WEB_IRT = REPO / "website" / "content" / "generated" / "benchmark-irt.json"
RASCH1PL_SUBDIR = Path("model_fits") / "rasch1pl"

# Columns we need out of responses.parquet. Reading a subset keeps the big
# benchmarks (rewardbench2 is 271 MB) off the heap — `trace` alone dwarfs
# everything else.
RESP_COLS = ["subject_id", "item_id", "test_condition", "trial", "response"]


def num(v: float, digits: int) -> float | None:
    """A JSON-safe number: NaN/inf become null rather than invalid JSON.

    Everything downstream (benchmark-irt.json, the website) parses strict JSON,
    where NaN is a syntax error — so a non-finite metric has to surface as a
    missing value, not as a token no parser accepts.
    """
    return None if not np.isfinite(v) else round(float(v), digits)


def rel(path: Path) -> str:
    """Repo-relative when it can be — --out may point anywhere."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def fit_output_dir(slug: str, out_root: Path | None) -> Path:
    """Directory for one benchmark's fit.

    The default is colocated with the benchmark. An explicit --out preserves
    the former common-root layout for scratch runs and external destinations.
    """
    if out_root is not None:
        return out_root / slug
    return REPO / "benchmarks" / slug / RASCH1PL_SUBDIR


def summary_output_path(out_root: Path | None) -> Path:
    """Cross-benchmark rollup path for the selected output layout."""
    if out_root is not None:
        return out_root / "summary.csv"
    return (
        REPO
        / "artifacts"
        / "analyze_measurements"
        / "model_fits"
        / "rasch1pl"
        / "summary.csv"
    )


def fit_summary_paths(out_root: Path | None) -> list[Path]:
    """Existing per-benchmark summaries in the selected output layout."""
    if out_root is not None:
        return sorted(out_root.glob("*/summary.json"))
    pattern = f"*/{RASCH1PL_SUBDIR.as_posix()}/summary.json"
    return sorted((REPO / "benchmarks").glob(pattern))


def write_summary_rollup(out_root: Path | None) -> tuple[Path, int]:
    """Rebuild the rollup from the summaries that actually exist on disk."""
    rows: list[dict] = []
    seen: dict[str, Path] = {}
    for path in fit_summary_paths(out_root):
        row = json.loads(path.read_text())
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            raise ValueError(f"fit summary has no slug: {path}")
        expected_dir = fit_output_dir(slug, out_root)
        if path.parent != expected_dir:
            raise ValueError(
                f"fit summary slug {slug!r} does not match its directory: {path}"
            )
        if previous := seen.get(slug):
            raise ValueError(
                f"duplicate fit summaries for {slug!r}: {previous} and {path}"
            )
        seen[slug] = path
        rows.append(row)

    if not rows:
        raise ValueError("no per-benchmark fit summaries found")

    rollup = summary_output_path(out_root)
    rollup.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("slug").to_csv(rollup, index=False)
    return rollup, len(rows)


# ---------------------------------------------------------------------------
#  data
# ---------------------------------------------------------------------------

def website_slugs() -> list[str]:
    """The benchmarks the site actually shows — the curated card list."""
    return sorted(json.loads(FILTERED.read_text()))


def load_responses(slug: str, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    path = fetch(f"{slug}/responses.parquet", cache_dir, refresh)
    cols = list(RESP_COLS)
    # Benchmarks built since the interactor split keep the other parties of
    # the interaction (attacker, user simulator, opponent) in their own
    # column; for the condition effect they are part of the same measurement
    # context, so fold them back into the condition label.
    has_inter = "interactors" in pq.read_schema(path).names
    if has_inter:
        cols.append("interactors")
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")
    df["test_condition"] = df["test_condition"].fillna("").astype(str)
    if has_inter:
        inter = df.pop("interactors").fillna("").astype(str)
        both = df["test_condition"].str.cat(inter, sep=";")
        df["test_condition"] = both.str.strip(";")
    return df


def subject_names(slug: str, cache_dir: Path, refresh: bool) -> dict[str, str]:
    """subject_id -> display name; empty if the file is unavailable."""
    try:
        path = fetch(f"{slug}/subjects.parquet", cache_dir, refresh)
        s = pd.read_parquet(path, columns=["subject_id", "display_name"])
        return dict(zip(s["subject_id"], s["display_name"]))
    except SystemExit:
        return {}


class NotBinary(Exception):
    """Raised when a benchmark's responses can't be fitted as-is."""


def binary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The rows a Rasch model can be fitted on, or NotBinary with the reason.

    We do NOT threshold. A benchmark qualifies only if every observed response
    is already 0 or 1 — anything else means the benchmark defines a graded
    scale, and picking a cut point here would invent a dichotomy it never had.
    """
    obs = df[df["response"].notna()]
    if obs.empty:
        raise NotBinary("no observed responses")
    vals = pd.unique(obs["response"].astype(float))
    extra = sorted(v for v in vals if v not in (0.0, 1.0))
    if extra:
        shown = ", ".join(f"{v:g}" for v in extra[:4])
        more = f" (+{len(extra) - 4} more)" if len(extra) > 4 else ""
        raise NotBinary(f"graded responses — {len(vals)} distinct values incl. {shown}{more}")
    if len(vals) < 2:
        raise NotBinary(f"degenerate — every response is {vals[0]:g}")
    return obs


# ---------------------------------------------------------------------------
#  model
# ---------------------------------------------------------------------------

def fit(df: pd.DataFrame, *, test_size: float, seed: int, lr: float,
        max_iter: int, tol: float, eval_every: int, device: str,
        verbose: bool = True) -> dict:
    """Fit sigmoid(theta[s] - z[i] + c[cond]) and return params + diagnostics."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    theta_ids = pd.unique(df["subject_id"])
    z_ids = pd.unique(df["item_id"])
    c_ids = pd.unique(df["test_condition"])
    M, N, K = len(theta_ids), len(z_ids), len(c_ids)
    # One condition means the term is a constant, already carried by z.
    use_cond = K > 1

    dev = torch.device(device)
    idx = lambda col, ids: torch.tensor(  # noqa: E731
        pd.Series(df[col]).map({v: i for i, v in enumerate(ids)}).to_numpy(dtype=np.int64),
        device=dev,
    )
    theta_idx, z_idx = idx("subject_id", theta_ids), idx("item_id", z_ids)
    c_idx = idx("test_condition", c_ids) if use_cond else None
    y = torch.tensor(df["response"].to_numpy(dtype=np.float32), device=dev)

    n_obs = len(y)
    train_pos, test_pos = train_test_split(
        np.arange(n_obs), test_size=test_size, random_state=seed, shuffle=True
    )
    tr = torch.tensor(np.sort(train_pos), device=dev)
    te = torch.tensor(np.sort(test_pos), device=dev)

    theta = torch.zeros(M, device=dev, requires_grad=True)
    z = torch.zeros(N, device=dev, requires_grad=True)
    c = torch.zeros(K, device=dev, requires_grad=True)
    params = [theta, z, c] if use_cond else [theta, z]

    def logit(pos):
        s = theta[theta_idx[pos]] - z[z_idx[pos]]
        return s + c[c_idx[pos]] if use_cond else s

    def report(pos):
        """(log-likelihood, AUC) on a subset — no grad, one pass.

        The likelihood is taken from the LOGITS, not from clamped probabilities.
        A subject who answered every item correctly has no finite MLE, so its
        theta walks off until sigmoid saturates to exactly 1.0 in float32 —
        past that point `p.clamp(1e-9, 1-1e-9)` is a no-op (1 - 1e-9 rounds to
        1.0), log(1 - p) is -inf, and the y=1 term computes 0 * -inf = NaN.
        binary_cross_entropy_with_logits is the log-sum-exp form and stays exact.
        """
        lg = logit(pos)
        yy = y[pos]
        ll = -torch.nn.functional.binary_cross_entropy_with_logits(
            lg, yy, reduction="sum").item()
        p = torch.sigmoid(lg)
        yn, pn = yy.detach().cpu().numpy(), p.detach().cpu().numpy()
        auc = float(roc_auc_score(yn, pn)) if len(np.unique(yn)) > 1 else float("nan")
        return ll, auc

    history: list[dict] = []
    opt = torch.optim.Adam(params, lr=lr)
    prev_loss, iters, t0 = float("inf"), 0, time.time()

    for i in range(max_iter):
        if i % eval_every == 0:
            with torch.no_grad():
                ll_tr, auc_tr = report(tr)
                ll_te, auc_te = report(te)
            history.append({"iter": i, "train_ll": ll_tr, "test_ll": ll_te,
                            "train_auc": auc_tr, "test_auc": auc_te})
            if verbose and i % (eval_every * 20) == 0:
                sys.stderr.write(
                    f"    iter {i:>5}  train AUC {auc_tr:.4f}  test AUC {auc_te:.4f}\n"
                )

        # Same objective as binary_cross_entropy(sigmoid(.)), computed in the
        # stable log-sum-exp form so a separated subject or item can't poison
        # the gradient once its logit saturates.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit(tr), y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():  # identifiability centering
            theta -= theta.mean()
            if use_cond:
                c -= c.mean()

        iters = i + 1
        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    with torch.no_grad():
        ll_tr, auc_tr = report(tr)
        ll_te, auc_te = report(te)
        p_all = torch.sigmoid(logit(torch.arange(n_obs, device=dev))).cpu().numpy()

    split = np.array(["train"] * n_obs, dtype=object)
    split[test_pos] = "test"

    return {
        "theta_ids": theta_ids, "theta": theta.detach().cpu().numpy(),
        "z_ids": z_ids, "z": z.detach().cpu().numpy(),
        "c_ids": c_ids if use_cond else np.array([], dtype=object),
        "c": c.detach().cpu().numpy() if use_cond else np.array([]),
        "use_cond": use_cond,
        "p": p_all, "split": split,
        "train_ll": ll_tr, "test_ll": ll_te,
        "train_auc": auc_tr, "test_auc": auc_te,
        "n_train": len(train_pos), "n_test": len(test_pos),
        "n_subjects": M, "n_items": N, "n_conditions": K,
        "iterations": iters, "converged": iters < max_iter,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }


# ---------------------------------------------------------------------------
#  outputs
# ---------------------------------------------------------------------------

def write_outputs(slug: str, df: pd.DataFrame, res: dict, names: dict[str, str],
                  out_dir: Path, curves: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- responses.csv: the data as read + the fit + the train/test comparison
    obs = df.copy()
    obs.insert(1, "subject", obs["subject_id"].map(names).fillna(obs["subject_id"]))
    obs["split"] = res["split"]
    obs["irt_prob"] = np.round(res["p"], 6)
    obs["irt_pred"] = (res["p"] >= 0.5).astype(int)
    # Held-out agreement between the prediction and the actual response. Train
    # rows are "-": the model was fitted on them, so agreement there is fit, not
    # prediction, and averaging the two would flatter the model.
    agree = np.where(obs["irt_pred"].to_numpy() == obs["response"].to_numpy(),
                     "match", "miss")
    obs["match"] = np.where(res["split"] == "test", agree, "-")
    obs.to_csv(out_dir / "responses.csv", index=False)

    # --- per-subject: theta beside the observed accuracy the matrix shows
    acc = df.groupby("subject_id")["response"].agg(["mean", "size"])
    subj = pd.DataFrame({"subject_id": res["theta_ids"], "theta": res["theta"]})
    subj["subject"] = subj["subject_id"].map(names).fillna(subj["subject_id"])
    subj["n"] = subj["subject_id"].map(acc["size"]).astype(int)
    subj["accuracy"] = subj["subject_id"].map(acc["mean"]).round(6)
    subj = subj[["subject_id", "subject", "theta", "n", "accuracy"]]
    subj.sort_values("theta", ascending=False).to_csv(out_dir / "subjects.csv", index=False)

    # --- per-item: z is difficulty, so it runs opposite the solve rate
    rate = df.groupby("item_id")["response"].agg(["mean", "size"])
    item = pd.DataFrame({"item_id": res["z_ids"], "z": res["z"]})
    item["n"] = item["item_id"].map(rate["size"]).astype(int)
    item["solve_rate"] = item["item_id"].map(rate["mean"]).round(6)
    item.sort_values("z", ascending=False).to_csv(out_dir / "items.csv", index=False)

    if res["use_cond"]:
        n_by_cond = df.groupby("test_condition")["response"].size()
        cond = pd.DataFrame({"test_condition": res["c_ids"], "effect": res["c"]})
        cond["n"] = cond["test_condition"].map(n_by_cond).astype(int)
        cond.sort_values("effect", ascending=False).to_csv(
            out_dir / "conditions.csv", index=False)

    # A subject (or item) whose responses are all 0 or all 1 is perfectly
    # separated: its maximum-likelihood theta is ±infinity, so the fit only
    # walks it out until the logit saturates. Those rows are real data, not a
    # failure — but their theta is "off the scale", not a measured value, and a
    # reader who sees θ = 10.0 deserves to know how many such rows there are.
    separated_subjects = int((acc["mean"].isin([0.0, 1.0])).sum())
    separated_items = int((rate["mean"].isin([0.0, 1.0])).sum())

    test_rows = obs[obs["split"] == "test"]
    summary = {
        "slug": slug,
        "model": "sigmoid(theta - z + c)" if res["use_cond"] else "sigmoid(theta - z)",
        "aucTrain": num(res["train_auc"], 4),
        "aucTest": num(res["test_auc"], 4),
        "llTrain": num(res["train_ll"], 2),
        "llTest": num(res["test_ll"], 2),
        "separatedSubjects": separated_subjects,
        "separatedItems": separated_items,
        "accTest": round(float((test_rows["match"] == "match").mean()), 4),
        "nTrain": res["n_train"], "nTest": res["n_test"],
        "nSubjects": res["n_subjects"], "nItems": res["n_items"],
        "nConditions": res["n_conditions"],
        "meanResponse": round(float(df["response"].mean()), 4),
        "iterations": res["iterations"], "converged": res["converged"],
        "seconds": res["seconds"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(res["history"]).to_csv(out_dir / "history.csv", index=False)

    if curves:
        write_curves(slug, res, out_dir)
    return summary


def write_curves(slug: str, res: dict, out_dir: Path) -> None:
    """Log-likelihood and AUC per iteration, one figure, two panels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("  (matplotlib missing — skipping curves)\n")
        return
    h = pd.DataFrame(res["history"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(h["iter"], h["train_ll"], label="Train")
    ax1.plot(h["iter"], h["test_ll"], label="Test", linestyle="--")
    ax1.set(xlabel="Iteration", ylabel="Log-likelihood", title=f"{slug} — log-likelihood")
    ax1.legend()
    ax2.plot(h["iter"], h["train_auc"], label="Train", linewidth=2)
    ax2.plot(h["iter"], h["test_auc"], label="Test", linestyle="--")
    ax2.set(xlabel="Iteration", ylabel="AUC", title=f"{slug} — AUC")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)


def item_difficulties(fit_dir: Path) -> dict[str, float]:
    items = pd.read_csv(fit_dir / "items.csv", dtype={"item_id": str})
    return {r.item_id: round(float(r.z), 4) for r in items.itertuples()}


def emit_web(summaries: dict[str, dict], skipped: dict[str, str],
             out_root: Path | None) -> None:
    """Merge fitted abilities and item difficulties into the website analysis data."""
    payload = json.loads(WEB_IRT.read_text()) if WEB_IRT.exists() else {}
    for slug, summary in summaries.items():
        fit_dir = fit_output_dir(slug, out_root)
        theta = pd.read_csv(fit_dir / "subjects.csv", dtype={"subject_id": str})
        payload[slug] = {
            **{k: v for k, v in summary.items() if k != "slug"},
            "theta": {r.subject_id: round(float(r.theta), 4)
                      for r in theta.itertuples()},
            "zByItem": item_difficulties(fit_dir),
        }
    for slug, reason in skipped.items():
        payload.pop(slug, None)  # a benchmark that stops qualifying loses its panel
    WEB_IRT.parent.mkdir(parents=True, exist_ok=True)
    WEB_IRT.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")
    sys.stderr.write(f"\nwrote {rel(WEB_IRT)} ({len(payload)} benchmarks)\n")


# ---------------------------------------------------------------------------
#  cli
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rasch IRT fit per benchmark: sigmoid(theta - z + condition).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: fit_rasch_models.py helm_harmbench afrimedqa --emit-web",
    )
    ap.add_argument("slugs", nargs="*", help="benchmark slugs; one or many")
    ap.add_argument("--all", action="store_true",
                    help="every benchmark on the website (published-benchmarks.json)")
    ap.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="common output root override (writes DIR/<slug>/ and "
             "DIR/summary.csv instead of benchmark-local fits)",
    )
    ap.add_argument("--cache-dir", type=Path,
                    default=RENDER_WEBSITE_DIR / ".hf-cache")
    ap.add_argument("--refresh", action="store_true", help="re-download the parquet")
    ap.add_argument("--hf-repo", action="append", metavar="REPO[@COMMIT]",
                    help="same source banks and revisions as the gallery")
    ap.add_argument("--emit-web", action="store_true",
                    help="write website/content/generated/benchmark-irt.json")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=4000)
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="stop when the training loss moves less than this")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="record LL/AUC every N iterations (AUC is the slow part)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-curves", action="store_true")
    args = ap.parse_args()

    gallery.HF_REPOS = {repo: revision or None
                       for source in args.hf_repo or [gallery.HF_REPO]
                       for repo, _, revision in [source.partition("@")]}
    gallery.HF_REPO = next(iter(gallery.HF_REPOS))
    gallery.source_revision.cache_clear()
    gallery.list_slugs()

    slugs = website_slugs() if args.all else args.slugs
    if not slugs:
        ap.error("name at least one benchmark, or pass --all")
    known = set(website_slugs())
    if unknown := [s for s in slugs if s not in known]:
        sys.stderr.write(
            f"note: not on the website — {', '.join(unknown)} "
            f"(fitting anyway; see published-benchmarks.json for the shown set)\n")

    if args.out is None:
        missing = [s for s in slugs if not (REPO / "benchmarks" / s).is_dir()]
        if missing:
            ap.error(
                "benchmark-local output requires an existing benchmarks/<slug> "
                f"directory ({', '.join(missing)}); pass --out for scratch fits"
            )

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    summaries: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for n, slug in enumerate(slugs, 1):
        sys.stderr.write(f"\n[{n}/{len(slugs)}] {slug}\n")
        try:
            df = load_responses(slug, args.cache_dir, args.refresh)
            df = binary_frame(df)
        except NotBinary as e:
            sys.stderr.write(f"  skipped: {e}\n")
            skipped[slug] = str(e)
            continue
        except SystemExit as e:  # fetch() exits on a missing/gated file
            sys.stderr.write(f"  skipped: {e}\n")
            skipped[slug] = "responses.parquet unavailable"
            continue

        sys.stderr.write(
            f"  {len(df):,} responses · {df['subject_id'].nunique()} subjects · "
            f"{df['item_id'].nunique():,} items · "
            f"{df['test_condition'].nunique()} condition(s)\n")
        res = fit(df, test_size=args.test_size, seed=args.seed, lr=args.lr,
                  max_iter=args.max_iter, tol=args.tol,
                  eval_every=args.eval_every, device=args.device)
        names = subject_names(slug, args.cache_dir, args.refresh)
        summary = write_outputs(slug, df, res, names,
                                fit_output_dir(slug, args.out),
                                curves=not args.no_curves)
        repo = gallery.BENCHMARK_REPOS.get(slug, gallery.HF_REPO)
        summary["source"] = {"repo": repo, "revision": gallery.source_revision(repo)}
        summary["fit"] = {k: getattr(args, k) for k in
                          ("test_size", "seed", "lr", "max_iter", "tol", "eval_every")}
        (fit_output_dir(slug, args.out) / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n")
        summaries[slug] = summary
        sys.stderr.write(
            f"  train AUC {summary['aucTrain']}  test AUC {summary['aucTest']}  "
            f"({res['iterations']} iters, {res['seconds']}s"
            f"{'' if res['converged'] else ', HIT MAX-ITER'})\n")

    if summaries:
        path, rollup_count = write_summary_rollup(args.out)
        sys.stderr.write(
            f"\n{len(summaries)} fitted, {len(skipped)} skipped; "
            f"{rollup_count} in rollup → {rel(path)}\n")

    if args.emit_web and (summaries or skipped):
        emit_web(summaries, skipped, args.out)

    if skipped:
        sys.stderr.write("\nskipped:\n")
        for slug, reason in sorted(skipped.items()):
            sys.stderr.write(f"  {slug}: {reason}\n")


if __name__ == "__main__":
    main()
