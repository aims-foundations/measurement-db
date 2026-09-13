import type { Metadata } from "next";
import { BenchmarkCatalogClient } from "@/components/benchmark-catalog-client";
import { buildBenchmarkGalleryData } from "@/components/benchmark-gallery";
import { BenchmarkTimeline } from "@/components/benchmark-timeline";
import { CitationBlock } from "@/components/citation-block";
import { DomainReadinessLeaderboard } from "@/components/domain-readiness-leaderboard";
import { MeasurementDbHero } from "@/components/measurement-db-hero";
import { ModelDomainHeatmap } from "@/components/model-domain-heatmap";
import { ModelLeaderboard } from "@/components/model-leaderboard";
import { ModelTimeline } from "@/components/model-timeline";
import { MeasurementSchemaSection } from "@/components/measurement-schema-section";
import { TorchMeasureSection } from "@/components/torch-measure-section";
import { domainReadinessMeta } from "@/content/domain-readiness";
import { heatmapMeta } from "@/content/model-domain-heatmap";
import { timelineMeta } from "@/content/benchmark-timeline";
import { modelTimelineMeta } from "@/content/model-timeline";
import { leaderboardMeta } from "@/content/model-leaderboard";
import {
  automatedScoreHeatmap,
  humanCenteredScoreHeatmap,
  sharedCatalogCharts,
  type SharedCatalogCharts,
} from "@/content/section-charts";
import {
  automaticBenchmarkSlugs,
  benchmarkCategories,
  benchmarks,
  measurementDb,
  publishedJudgedBenchmarkSlugs,
} from "@/content/measurement-db";

export const metadata: Metadata = {
  title: "Data & Software | AIMS",
  description: measurementDb.description,
  alternates: { canonical: "/measurement-db" },
  openGraph: {
    title: "measurement-db",
    description: measurementDb.description,
  },
  twitter: {
    card: "summary_large_image",
    title: "measurement-db",
    description: measurementDb.description,
  },
};

// Folded row for one meta-analysis chart. Native <details>, so the fold
// needs no client JS; the row styling (.rd-acc, redesign.css) borrows the
// hairline-row + circled-icon language of the entry cards, and the plus
// rotates into a close mark when open.
function ChartAccordion({
  title,
  description,
  scope,
  children,
}: {
  title: string;
  description: string;
  scope: string;
  children: React.ReactNode;
}) {
  return (
    <details className="rd-acc-item">
      <summary>
        <span className="flex min-w-0 flex-col items-start gap-x-3 gap-y-1 sm:flex-row sm:items-center">
          <span
            role="heading"
            aria-level={4}
            className="text-lg font-medium leading-snug tracking-tight sm:text-xl"
          >
            {title}
          </span>
          <span className="rounded-full bg-black/[0.065] px-2 py-1 font-mono text-[0.62rem] uppercase tracking-wider opacity-60">
            {scope}
          </span>
        </span>
        <span className="rd-acc-icon" aria-hidden>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              d="M6 1v10M1 6h10"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </span>
      </summary>
      <div className="rd-acc-body">
        {/* Wide measure (same as the section standfirsts) — at caption size
            a 48ch cap breaks these two-sentence descriptions into ragged
            four-line stacks. */}
        <p className="rd-prose-wide text-sm font-light leading-relaxed opacity-65">
          {description}
        </p>
        <div className="mt-5">{children}</div>
      </div>
    </details>
  );
}

// Coverage and release metadata are safe to summarize over the complete bank.
// These four rows are recomputed from the union, not stitched together from
// the former shelf-specific rankings.
function SharedMetaAnalysisRows({
  charts,
  scope,
}: {
  charts: SharedCatalogCharts;
  scope: string;
}) {
  return (
    <>
      {charts.leaderboard.length > 0 ? (
        <ChartAccordion
          title={leaderboardMeta.title}
          description={leaderboardMeta.description}
          scope={scope}
        >
          <ModelLeaderboard entries={charts.leaderboard} />
        </ChartAccordion>
      ) : null}

      {charts.domainReadiness.length > 0 ? (
        <ChartAccordion
          title={domainReadinessMeta.title}
          description={domainReadinessMeta.description}
          scope={scope}
        >
          <DomainReadinessLeaderboard entries={charts.domainReadiness} />
        </ChartAccordion>
      ) : null}

      {charts.benchmarkTimeline.years.length > 0 ? (
        <ChartAccordion
          title={timelineMeta.title}
          description={timelineMeta.description}
          scope={scope}
        >
          <BenchmarkTimeline data={charts.benchmarkTimeline} />
        </ChartAccordion>
      ) : null}

      {charts.modelTimeline.years.length > 0 ? (
        <ChartAccordion
          title={modelTimelineMeta.title}
          description={modelTimelineMeta.description}
          scope={scope}
        >
          <ModelTimeline data={charts.modelTimeline} />
        </ChartAccordion>
      ) : null}
    </>
  );
}

function ScoreMetaAnalysisRow({
  heatmap,
  scope,
}: {
  heatmap: typeof automatedScoreHeatmap;
  scope: string;
}) {
  return (
    <ChartAccordion
      title={heatmapMeta.title}
      description={heatmapMeta.description}
      scope={scope}
    >
      {heatmap.rows.length > 0 ? (
        <ModelDomainHeatmap data={heatmap} metricLabel="Mean binary score" />
      ) : (
        <p className="text-sm font-light opacity-65">
          No binary-scored measurements are available for this benchmark type.
        </p>
      )}
    </ChartAccordion>
  );
}

export default function MeasurementDbPage() {
  const automatedCatalog = buildBenchmarkGalleryData(automaticBenchmarkSlugs);
  const humanCenteredCatalog = buildBenchmarkGalleryData(
    publishedJudgedBenchmarkSlugs,
  );
  const automatedSlugs = new Set(
    automatedCatalog.items.map((benchmark) => benchmark.slug),
  );
  const humanCenteredSlugs = new Set(
    humanCenteredCatalog.items.map((benchmark) => benchmark.slug),
  );
  const catalogIsPartitioned =
    automatedSlugs.size + humanCenteredSlugs.size === benchmarks.length &&
    [...automatedSlugs].every((slug) => !humanCenteredSlugs.has(slug));
  if (!catalogIsPartitioned) {
    throw new Error(
      "Automated and human-centered benchmark variants must partition the public catalog",
    );
  }

  return (
    <main id="main-content">
      {/* ------------------------------------------------------------ hero */}
      <MeasurementDbHero />

      {/* --------------------------------------------- benchmark catalog

          One visual shelf; benchmark type remains an explicit data filter. */}
      <section
        id="benchmark-catalog"
        className="rd-white-band rd-band scroll-mt-16"
        aria-labelledby="benchmark-catalog-title"
      >
        <div className="rd-container">
          <h2 id="benchmark-catalog-title" className="sr-only">
            Benchmark catalog
          </h2>

          <BenchmarkCatalogClient
            automated={automatedCatalog}
            humanCentered={humanCenteredCatalog}
            categories={benchmarkCategories}
            sharedAnalysisRows={
              <SharedMetaAnalysisRows
                charts={sharedCatalogCharts}
                scope={`All ${benchmarks.length}`}
              />
            }
            automatedScoreAnalysis={
              <ScoreMetaAnalysisRow
                heatmap={automatedScoreHeatmap}
                scope="Automated"
              />
            }
            humanCenteredScoreAnalysis={
              <ScoreMetaAnalysisRow
                heatmap={humanCenteredScoreHeatmap}
                scope="Human-centered"
              />
            }
          />
        </div>
      </section>

      {/* --------------------------------------- analysis and modeling layer */}
      <TorchMeasureSection />

      {/* --------------------------------------------- measurement data model */}
      <MeasurementSchemaSection />

      {/* --------------------------------------------- citation (quiet band) */}
      <section id="citation" className="rd-dark-band rd-band scroll-mt-16">
        <div className="rd-container">
          <div className="grid gap-8 sm:gap-12 lg:grid-cols-[auto_1fr] lg:items-start lg:gap-20">
            <div className="lg:whitespace-nowrap">
              <p className="rd-mono opacity-60">Citation</p>
              <h2 className="rd-h2 mt-4">Cite this work</h2>
            </div>
            <div className="min-w-0 max-w-3xl space-y-6 lg:justify-self-end">
              <p className="rd-prose text-base font-light leading-relaxed opacity-80">
                If you use the data we curated, please cite the following
                reference. The curation is released under{" "}
                <a
                  href={measurementDb.license.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline underline-offset-4 hover:opacity-100"
                >
                  {measurementDb.license.name}
                </a>
                ; individual benchmarks retain their upstream licenses.
              </p>
              <CitationBlock bibtex={measurementDb.citation.bibtex} />
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
