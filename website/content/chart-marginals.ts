// ---------------------------------------------------------------------------
//  Per-benchmark chart marginals — the composable building blocks behind the
//  provider leaderboard and the model × domain heatmap. Source of truth:
//  scripts/render_website/generate_chart_marginals.py, which emits one row per
//  (benchmark, model) instead of bank-wide totals, precisely so the website
//  can compose any slug list (a section's shelf) at build time — do not
//  hand-edit the JSON.
//
//  `items` (distinct items asked) is summable because items never overlap
//  across benchmarks; `sum`/`n` (binary-response sum and count, present only
//  for binary-scored benchmarks) compose into a domain cell's micro-average
//  as Σsum/Σn.
// ---------------------------------------------------------------------------

import marginalsJson from "@/content/generated/chart-marginals.json";

export type ChartMarginal = {
  slug: string;
  model: string;
  /** Distinct items this model was asked in this benchmark. */
  items: number;
  /** Binary-response sum — only on binary-scored benchmarks. */
  sum?: number;
  /** Binary-response count — only on binary-scored benchmarks. */
  n?: number;
};

const raw = marginalsJson as unknown as {
  slugs: Record<string, { binary: boolean }>;
  models: Record<string, string>;
  marginals: ChartMarginal[];
};

/** model -> company (registry-canonical). */
export const marginalCompany: Readonly<Record<string, string>> = raw.models;

/** marginal rows grouped by slug, for fast per-section composition. */
const bySlug = new Map<string, ChartMarginal[]>();
for (const m of raw.marginals) {
  const list = bySlug.get(m.slug);
  if (list) list.push(m);
  else bySlug.set(m.slug, [m]);
}

/** Every marginal row for the given shelf, after verifying the shelf is fully
 *  covered by the generator's manifest. A slug missing from the manifest means
 *  generate_chart_marginals.py was not re-run after the bank changed — fail the
 *  site build loudly instead of rendering charts missing a benchmark (the
 *  silent-partial failure mode that has bitten every chart generator here). */
export function marginalsFor(slugs: readonly string[]): ChartMarginal[] {
  const missing = slugs.filter((s) => !(s in raw.slugs));
  if (missing.length > 0) {
    throw new Error(
      `chart-marginals.json does not cover ${missing.length} shelf benchmark(s): ` +
        `${missing.join(", ")} — re-run ` +
        `scripts/render_website/generate_chart_marginals.py after changing the bank.`,
    );
  }
  return slugs.flatMap((s) => bySlug.get(s) ?? []);
}
