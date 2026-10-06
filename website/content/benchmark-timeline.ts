// Release timeline of a catalog scope, derived at build time from
// benchmark-cards.json (via measurement-db.ts) — the same generated file that
// feeds the gallery. Adding a benchmark to the data bank therefore updates this
// chart with no extra step.
//
// Each card's `releaseDate` is "YYYY-MM", "YYYY", or null. The chart is a dot
// histogram matching the model release timeline: every year gets the same
// width, split into 12 month columns, and each month-dated benchmark is one
// small dot stacked in its release month, colored by primary domain.
// Year-only and undated benchmarks cannot be placed in a month column, so
// they are listed in a footnote under the chart instead of shown — never
// dropped silently — as are any released outside the axis. The axis is
// the supplied AI-subject timeline year range, one equal-width block per
// year, so the two charts' year boundaries align vertically on the page.

import { benchmarkCategories, benchmarks } from "@/content/measurement-db";

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

const FALLBACK_COLOR = "#175e54";
const category = new Map(benchmarkCategories.map((c) => [c.id, c]));

export type TimelineItem = {
  slug: string;
  name: string;
  /** Primary-category display label — "Mathematics". */
  domain: string;
  /** "Apr 2025". */
  released: string;
  /** Primary-category accent color (same hue as the gallery card). */
  color: string;
};

export type TimelineYear = {
  /** Axis label — "2024". */
  label: string;
  /** Total month-dated benchmarks released this year. */
  count: number;
  /** Jan..Dec — each month's benchmarks grouped by primary domain
   *  (alphabetical from the bottom of the stack, names A→Z within a domain). */
  months: TimelineItem[][];
};

export type BenchmarkTimelineData = {
  years: TimelineYear[];
  /** Tallest month stack — sizes the chart so every column fits unclipped. */
  maxMonth: number;
  /** The primary domains that appear in the chart, in the site's canonical
   *  category order. */
  legend: ReadonlyArray<{ label: string; color: string }>;
  /** Catalog benchmarks the chart CANNOT place — no month-dated release, or a
   *  date outside the axis range — surfaced as a footnote under the chart so
   *  the exclusion is never silent. */
  excluded: ReadonlyArray<string>;
};

type Parsed = TimelineItem & { year: number; month: number };

/** The benchmark release timeline for one catalog scope. `range` is the
 *  AI-subject timeline year range, so the two charts' axes align. */
export function makeBenchmarkTimeline(
  slugs: readonly string[],
  range: { from: number; to: number },
): BenchmarkTimelineData {
  const scope = new Set(slugs);
  const parsed: Parsed[] = [];
  const excluded: string[] = [];
  for (const b of benchmarks) {
    if (!scope.has(b.slug)) continue;
    const m = /^(\d{4})-(\d{2})/.exec(b.releaseDate ?? "");
    if (!m) {
      excluded.push(b.name);
      continue;
    }
    const cat = category.get(b.categories?.[0]);
    parsed.push({
      slug: b.slug,
      name: b.name,
      domain: cat?.label ?? "Uncategorized",
      year: Number(m[1]),
      month: Number(m[2]),
      released: `${MONTHS[Number(m[2]) - 1]} ${m[1]}`,
      color: cat?.color ?? FALLBACK_COLOR,
    });
  }

  const item = ({
    slug,
    name,
    domain,
    released,
    color,
  }: Parsed): TimelineItem => ({
    slug,
    name,
    domain,
    released,
    color,
  });

  // A dot outside the axis range has no column — excluded, and said so.
  for (const p of parsed) {
    if (p.year < range.from || p.year > range.to) excluded.push(p.name);
  }

  const years: TimelineYear[] = [];
  if (parsed.length > 0 && range.to >= range.from) {
    for (let year = range.from; year <= range.to; year++) {
      const months: TimelineItem[][] = Array.from({ length: 12 }, () => []);
      for (const p of parsed)
        if (p.year === year) months[p.month - 1].push(item(p));
      // Group by domain (every domain has its own hue, so no pooled tail) —
      // each column shows contiguous color blocks in alphabetical order.
      for (const m of months)
        m.sort(
          (a, b) =>
            a.domain.localeCompare(b.domain, "en", { sensitivity: "base" }) ||
            a.name.localeCompare(b.name),
        );
      years.push({
        label: String(year),
        count: months.reduce((n, m) => n + m.length, 0),
        months,
      });
    }
  }

  return {
    years,
    maxMonth: Math.max(
      1,
      ...years.flatMap((y) => y.months.map((m) => m.length)),
    ),
    legend: benchmarkCategories
      .filter((c) => parsed.some((p) => p.domain === c.label))
      .map((c) => ({ label: c.label, color: c.color })),
    excluded: excluded.sort((a, b) => a.localeCompare(b)),
  };
}

export const timelineMeta = {
  title: "Benchmark Release Timeline",
  description:
    "Month-dated benchmarks, stacked by release month and colored by primary domain.",
} as const;
