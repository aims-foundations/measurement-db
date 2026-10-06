// ---------------------------------------------------------------------------
//  Hybrid catalog chart datasets.
//
//  Coverage and release timelines are descriptive metadata, so they are
//  recomputed once over the complete public catalog. Score heatmaps remain
//  separate because programmatic-verifier outcomes and human/model judgments
//  do not share one response meaning.
// ---------------------------------------------------------------------------

import { makeBenchmarkTimeline } from "@/content/benchmark-timeline";
import { makeDomainReadiness } from "@/content/domain-readiness";
import { makeModelDomainHeatmap } from "@/content/model-domain-heatmap";
import { makeModelLeaderboard } from "@/content/model-leaderboard";
import {
  makeModelTimeline,
  type ModelTimelineData,
} from "@/content/model-timeline";
import {
  automaticBenchmarkSlugs,
  benchmarks,
  publishedJudgedBenchmarkSlugs,
} from "@/content/measurement-db";

export type SharedCatalogCharts = ReturnType<typeof makeSharedCatalogCharts>;

function monthDatedBenchmarkYears(slugs: readonly string[]): number[] {
  const scope = new Set(slugs);
  return benchmarks
    .filter((benchmark) => scope.has(benchmark.slug))
    .map((benchmark) =>
      Number(/^(\d{4})-\d{2}/.exec(benchmark.releaseDate ?? "")?.[1]),
    )
    .filter((year) => Number.isInteger(year));
}

/** Pad the AI-subject timeline to the full shared release range so its axis
 * stays aligned with the benchmark timeline without excluding older, validly
 * dated benchmarks. */
function extendTimelineRange(
  timeline: ModelTimelineData,
  benchmarkYears: readonly number[],
): ModelTimelineData {
  const datedYears = timeline.years.map((year) => Number(year.label));
  const allYears = [...datedYears, ...benchmarkYears];
  if (allYears.length === 0) return timeline;

  const range = {
    from: Math.min(...allYears),
    to: Math.max(...allYears),
  };
  const byYear = new Map(timeline.years.map((year) => [year.label, year]));
  const years: ModelTimelineData["years"] = [];
  for (let year = range.from; year <= range.to; year += 1) {
    years.push(
      byYear.get(String(year)) ?? {
        label: String(year),
        count: 0,
        months: Array.from({ length: 12 }, () => []),
      },
    );
  }

  return { ...timeline, years, range };
}

function makeSharedCatalogCharts(slugs: readonly string[]) {
  const leaderboard = makeModelLeaderboard(slugs);
  // Timeline hues follow the all-catalog provider ranking. Both timelines use
  // the union of benchmark and subject release years so their axes align and
  // every month-dated benchmark remains visible.
  const subjectTimeline = extendTimelineRange(
    makeModelTimeline(
      slugs,
      leaderboard.map((entry) => entry.company),
    ),
    monthDatedBenchmarkYears(slugs),
  );

  return {
    leaderboard,
    domainReadiness: makeDomainReadiness(slugs),
    benchmarkTimeline: makeBenchmarkTimeline(slugs, subjectTimeline.range),
    modelTimeline: subjectTimeline,
  };
}

const publicBenchmarkSlugs = benchmarks.map((benchmark) => benchmark.slug);

/** Coverage and release metadata recomputed over all public benchmarks. */
export const sharedCatalogCharts =
  makeSharedCatalogCharts(publicBenchmarkSlugs);

/** Score summaries remain scoped to response-production type. */
export const automatedScoreHeatmap = makeModelDomainHeatmap(
  automaticBenchmarkSlugs,
);
export const humanCenteredScoreHeatmap = makeModelDomainHeatmap(
  publishedJudgedBenchmarkSlugs,
);
