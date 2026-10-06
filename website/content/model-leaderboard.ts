// ---------------------------------------------------------------------------
//  Provider coverage leaderboard — each company represented by its MOST-MEASURED
//  model on the given shelf (the one asked the most distinct benchmark items),
//  ranked by that count. Composed at build time from chart-marginals.json (see
//  chart-marginals.ts), so each section's leaderboard covers exactly that
//  section's benchmarks and follows any shelf change automatically.
//
//  This used to show each provider's LATEST model. That inverted the chart's
//  meaning: the newest model is the least-measured one, so a Coverage chart
//  reported Google at 502 items while the bank held 67,614 for Gemini 2.0 Flash.
// ---------------------------------------------------------------------------

import { marginalCompany, marginalsFor } from "@/content/chart-marginals";

// How many providers the chart shows. Enough to cover the majors and the
// significant open-weight labs without a scrolling list.
const TOP_N = 12;

// Not providers: the registry's placeholder buckets for subject strings that
// could not be resolved to a real model, and for agent scaffolds whose row
// would name a harness rather than a lab.
const NON_PROVIDERS = new Set(["Unknown", "(agent scaffold)"]);

export type LeaderboardEntry = {
  rank: number;
  company: string;
  model: string;
  totalItemsAsked: number;
  /** The flagship model name with the company prefix stripped. */
  flagship: string;
  /** Thousands-separated item count. */
  itemsLabel: string;
  /** Bar fill as a percentage of the top provider (0–100). */
  fillPct: number;
};

/** Drop the redundant company prefix from a flagship's canonical name so it
 *  reads cleanly under the company name ("OpenAI GPT-4o" → "GPT-4o"). Kept
 *  intact when the remainder would be a bare size ("Mistral 7B" stays). */
function flagshipLabel(company: string, model: string): string {
  const prefix = `${company} `;
  if (model.startsWith(prefix)) {
    const rest = model.slice(prefix.length);
    if (rest && !/^\d/.test(rest)) return rest;
  }
  return model;
}

/** "561,792" — thousands-separated. */
function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

/** The provider leaderboard for one shelf's benchmarks. */
export function makeModelLeaderboard(
  slugs: readonly string[],
): ReadonlyArray<LeaderboardEntry> {
  // model -> distinct items asked across the shelf (summable: items never
  // overlap across benchmarks, and each marginal is already alias-deduped).
  const items = new Map<string, number>();
  for (const m of marginalsFor(slugs)) {
    items.set(m.model, (items.get(m.model) ?? 0) + m.items);
  }

  // company -> its best-covered (model, items) on this shelf.
  const best = new Map<string, { model: string; items: number }>();
  for (const [model, n] of items) {
    const company = marginalCompany[model] ?? "Unknown";
    if (NON_PROVIDERS.has(company)) continue;
    const cur = best.get(company);
    if (!cur || n > cur.items) best.set(company, { model, items: n });
  }

  const rows = [...best.entries()]
    .map(([company, b]) => ({ company, ...b }))
    .sort((a, b) => b.items - a.items || a.company.localeCompare(b.company))
    .slice(0, TOP_N);

  const maxItems = rows.reduce((m, r) => Math.max(m, r.items), 0);
  return rows.map((r, i) => ({
    rank: i + 1,
    company: r.company,
    model: r.model,
    totalItemsAsked: r.items,
    flagship: flagshipLabel(r.company, r.model),
    itemsLabel: formatCount(r.items),
    fillPct: maxItems ? (r.items / maxItems) * 100 : 0,
  }));
}

export const leaderboardMeta = {
  title: "Provider Coverage",
  description:
    "Providers ranked by the distinct benchmark items evaluated for their most-measured AI subject.",
} as const;
