#!/usr/bin/env python3
"""Regenerate artifacts/render_website/model_statistics.csv from the cached
parquets + scripts/build_measurement_tables/map_model_registry.json.
Deterministic: no network, no LLM.

Item counts are DEDUPED per (model, benchmark). When several subject strings in
one benchmark map to the same canonical model (e.g. `gpt-4o` and
`gpt-4o-2024-08-06` -> "OpenAI GPT-4o"), that benchmark's items are counted ONCE
for the model, not once per alias. So `total_items_asked` stays a true count of
distinct items the model was asked, and the leaderboard isn't biased toward
models that happen to appear under many alias strings. (`total_responses` is
still summed across aliases — it counts runs, which are legitimately additive.)

Subjects missing from the registry are kept as their own row (company=Unknown)
and printed as a report to fix by hand — see docs/update_model_registry.md.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / ".hf-cache"
REGISTRY = (
    REPO / "scripts" / "build_measurement_tables" / "map_model_registry.json"
)
CARDS = REPO / "website" / "content" / "generated" / "benchmark-cards.json"
OUT = REPO / "artifacts" / "render_website" / "model_statistics.csv"

registry = json.loads(REGISTRY.read_text())


def published_slugs_or_die(cards_path, cache):
    """Every benchmark the website publishes, or abort naming what is missing.

    The bank is defined by benchmark-cards.json — the same generated file that
    renders the gallery cards in sections 03 and 04 — NOT by what happens to be
    on this disk. Those used to be silently intersected: a slug whose
    .hf-cache/<slug>/ held only benchmarks.parquet was dropped without a word,
    and the run still exited 0. On 2026-08-02 that hid 25 of 66 benchmarks, so
    every derived statistic covered 62% of the bank while reporting success.

    A partial cache is easy to produce innocently: `generate_benchmark_gallery.py cards`
    fetches only benchmarks.parquet (read_meta needs nothing else), so a cards
    run leaves behind directories that look populated and are not.
    """
    published = [c["slug"] for c in json.loads(cards_path.read_text())]
    missing = [s for s in published
               if not any((cache / s / name).exists()
                          for name in ("responses.parquet", "response.parquet"))
               or not (cache / s / "subjects.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(published)} published benchmarks have no "
            f"cached response/subjects parquet:\n  " + "\n  ".join(sorted(missing))
            + "\n\nFetch them before regenerating — the output would otherwise "
            f"cover only {len(published) - len(missing)} benchmarks and still "
            "look successful. Neither this script nor the domain heatmap reads "
            "traces.parquet, so response+subjects alone is enough."
        )
    return published


slugs = published_slugs_or_die(CARDS, CACHE)

items = defaultdict(int)                      # model -> distinct items (deduped per benchmark)
resps = defaultdict(int)                      # model -> total response rows
bench = defaultdict(set)                      # model -> {benchmark slugs}
company = {}                                  # model -> company
unmapped = defaultdict(lambda: [0, set()])    # raw subject -> [items, {slugs}]

def resolve(name):
    """display_name -> (canonical model, company, is_mapped)."""
    e = registry.get(name)
    if e is None:
        return name, "Unknown", False          # keep as its own row, flag it
    return e["model"], e.get("company", "Unknown"), True

for slug in slugs:
    path = CACHE / slug / "responses.parquet"
    resp = pq.read_table(path if path.exists() else CACHE / slug / "response.parquet",
                         columns=["subject_id", "item_id"]).to_pandas()
    subj = pq.read_table(CACHE / slug / "subjects.parquet",
                         columns=["subject_id", "display_name"]).to_pandas()
    id2name = dict(zip(subj["subject_id"], subj["display_name"]))

    # map every subject_id in this benchmark to its canonical model
    sid2model = {}
    for sid in resp["subject_id"].unique():
        model, comp, _ = resolve(id2name.get(sid, sid))
        sid2model[sid] = model
        company[model] = comp
    resp["model"] = resp["subject_id"].map(sid2model)

    # DEDUPE: distinct item_ids per model (collapses aliases within this benchmark)
    for model, ni in resp.groupby("model")["item_id"].nunique().items():
        items[model] += int(ni)
        bench[model].add(slug)
    for model, nr in resp.groupby("model").size().items():
        resps[model] += int(nr)

    # report: distinct items each still-unmapped subject accounts for
    for sid, ni in resp.groupby("subject_id")["item_id"].nunique().items():
        name = id2name.get(sid, sid)
        if name not in registry:
            unmapped[name][0] += int(ni)
            unmapped[name][1].add(slug)

if unmapped:
    sys.stderr.write(f"\n⚠ {len(unmapped)} subject string(s) not in the registry "
                     f"(kept as-is, company=Unknown):\n")
    for name, (ni, sl) in sorted(unmapped.items(), key=lambda x: -x[1][0]):
        sys.stderr.write(f"   {ni:>8,} items  {name}   [{', '.join(sorted(sl))}]\n")
    sys.stderr.write(
        "  -> map these in scripts/build_measurement_tables/"
        "map_model_registry.json, then re-run.\n\n"
    )
else:
    sys.stderr.write("✓ every subject resolved via the registry.\n")

rows = sorted(items, key=lambda m: -items[m])
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "company", "model",
                "total_items_asked", "total_responses", "n_benchmarks"])
    for rank, m in enumerate(rows, 1):
        w.writerow([rank, company[m], m, items[m], resps[m], len(bench[m])])
sys.stderr.write(f"✓ wrote {len(rows)} models across {len(slugs)} benchmarks -> {OUT}\n")
