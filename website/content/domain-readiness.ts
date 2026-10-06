// ---------------------------------------------------------------------------
//  Domain Coverage — each benchmark domain ranked by the total number of
//  distinct evaluation items across the given catalog scope. FULLY
//  AUTO-COMPUTED from the benchmarks + benchmarkCategories in
//  measurement-db.ts (themselves generated from benchmark-cards.json by
//  generate_benchmark_gallery.py), so each section's chart covers exactly that section's
//  benchmarks and stays in sync whenever a benchmark is added, removed, or
//  its item count changes — no script to run, no JSON to hand-edit.
//
//  A benchmark can map to several domains (e.g. AlgoTune is math + coding);
//  its items count toward EACH of its domains, so domain totals intentionally
//  overlap and do not sum to the data-bank-wide item count.
// ---------------------------------------------------------------------------

import {
  benchmarks,
  benchmarkCategories,
  type BenchmarkCategoryId,
} from "@/content/measurement-db";

export type DomainReadinessEntry = {
  rank: number;
  id: BenchmarkCategoryId;
  label: string;
  /** Brand accent hex for this domain (from benchmarkCategories). */
  color: string;
  /** Sum of distinct items across every catalog benchmark tagged with this
   *  domain. */
  totalItems: number;
  /** How many catalog benchmarks contribute to this domain. */
  benchmarkCount: number;
  /** Thousands-separated item count. */
  itemsLabel: string;
  /** "60 benchmarks" — pluralized secondary label. */
  benchmarksLabel: string;
  /** Bar fill as a percentage of the most-measured domain (0–100). */
  fillPct: number;
};

/** "262,393" — thousands-separated, matching the model leaderboard. */
function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

// Preference outweighs the smallest domains by ~1700:1, so a strictly
// proportional fill renders the bottom rows at sub-pixel width — they read as
// a broken chart rather than as "very small". Floor the fill at a visible
// stub; the exact count sits beside every bar, so the stub never stands in
// for the value, it only keeps the row's encoding present.
const MIN_FILL_PCT = 1.2;

/** The domain-coverage ranking for one catalog scope. */
export function makeDomainReadiness(
  slugs: readonly string[],
): ReadonlyArray<DomainReadinessEntry> {
  const scope = new Set(slugs);
  const totals = new Map<
    BenchmarkCategoryId,
    { items: number; count: number }
  >();
  for (const b of benchmarks) {
    if (!scope.has(b.slug)) continue;
    for (const category of b.categories) {
      const agg = totals.get(category) ?? { items: 0, count: 0 };
      agg.items += b.items;
      agg.count += 1;
      totals.set(category, agg);
    }
  }

  const rows = benchmarkCategories
    .map((c) => {
      const agg = totals.get(c.id) ?? { items: 0, count: 0 };
      return {
        id: c.id,
        label: c.label,
        color: c.color,
        totalItems: agg.items,
        benchmarkCount: agg.count,
      };
    })
    .filter((r) => r.benchmarkCount > 0)
    .sort((a, b) => b.totalItems - a.totalItems);

  const maxItems = rows.reduce((m, r) => Math.max(m, r.totalItems), 0);
  return rows.map((r, i) => ({
    rank: i + 1,
    ...r,
    itemsLabel: formatCount(r.totalItems),
    benchmarksLabel: `${r.benchmarkCount} benchmark${r.benchmarkCount === 1 ? "" : "s"}`,
    fillPct: maxItems
      ? Math.max(MIN_FILL_PCT, (r.totalItems / maxItems) * 100)
      : 0,
  }));
}

export const domainReadinessMeta = {
  title: "Domain Coverage",
  description:
    "Domains ranked by evaluated items. Multi-domain benchmarks count toward every applicable domain, so totals overlap.",
} as const;
