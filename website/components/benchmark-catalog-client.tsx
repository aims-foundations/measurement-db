"use client";

import { Fragment, type ReactNode, useId, useState } from "react";
import { BenchmarkGalleryClient } from "@/components/benchmark-gallery-client";
import type { BenchmarkGalleryData } from "@/components/benchmark-gallery";
import { ScrollReveal } from "@/components/scroll-reveal";
import type { BenchmarkCategory } from "@/content/measurement-db";

type BenchmarkType = "automated" | "human-centered";

type CatalogVariant = BenchmarkGalleryData & {
  label: string;
  scopeLabel: string;
  description: string;
};

export function BenchmarkCatalogClient({
  automated,
  humanCentered,
  categories,
  sharedAnalysisRows,
  automatedScoreAnalysis,
  humanCenteredScoreAnalysis,
}: {
  automated: BenchmarkGalleryData;
  humanCentered: BenchmarkGalleryData;
  categories: readonly BenchmarkCategory[];
  sharedAnalysisRows: ReactNode;
  automatedScoreAnalysis: ReactNode;
  humanCenteredScoreAnalysis: ReactNode;
}) {
  const [benchmarkType, setBenchmarkType] =
    useState<BenchmarkType>("automated");
  // Search is the one text-discovery state shared by the catalog. It survives
  // a type change and immediately reruns within the newly selected slug scope;
  // the keyed gallery resets only type-specific facets and sort state.
  const [searchQuery, setSearchQuery] = useState("");
  const descriptionId = useId();
  const groupName = useId();

  const variants: Record<BenchmarkType, CatalogVariant> = {
    automated: {
      ...automated,
      label: "Automated",
      scopeLabel: "automated benchmarks",
      description:
        "Responses are scored by deterministic or programmatic verifiers and test harnesses.",
    },
    "human-centered": {
      ...humanCentered,
      label: "Human-centered",
      scopeLabel: "human-centered benchmarks",
      description:
        "Responses are scored through human, model, or reward-model judgments, so the scores also reflect judge disagreement. Pairwise comparisons are included here.",
    },
  };
  const active = variants[benchmarkType];
  const activeScoreAnalysis =
    benchmarkType === "automated"
      ? automatedScoreAnalysis
      : humanCenteredScoreAnalysis;

  return (
    <>
      <div className="grid gap-5 lg:grid-cols-[auto_minmax(0,1fr)] lg:items-end lg:gap-12">
        <fieldset aria-describedby={descriptionId}>
          <legend className="rd-mono opacity-60">Benchmark type</legend>
          <div className="mt-2 grid grid-cols-2 rounded-lg bg-black/[0.055] p-1 sm:inline-grid">
            {(Object.keys(variants) as BenchmarkType[]).map((type) => {
              const variant = variants[type];
              const selected = type === benchmarkType;
              return (
                <label key={type} className="cursor-pointer">
                  <input
                    className="peer sr-only"
                    type="radio"
                    name={groupName}
                    value={type}
                    checked={selected}
                    onChange={() => setBenchmarkType(type)}
                  />
                  <span
                    className={`flex min-h-11 items-center justify-center gap-2 rounded-md border px-3 text-sm transition-[background-color,border-color,box-shadow] peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-[var(--digital-blue)] sm:min-w-44 ${
                      selected
                        ? "border-black/15 bg-white font-medium text-[var(--ink)] shadow-sm"
                        : "border-transparent text-[var(--muted)] hover:bg-white/55 hover:text-[var(--ink)]"
                    }`}
                  >
                    <span>{variant.label}</span>
                    <span className="rounded-full bg-black/[0.065] px-2 py-0.5 font-mono text-[0.68rem] font-normal">
                      {variant.items.length}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <p
          id={descriptionId}
          className="rd-prose-wide text-base font-light leading-relaxed opacity-70 lg:pb-1"
        >
          {active.description}
        </p>
      </div>

      <p className="sr-only" role="status" aria-live="polite" aria-atomic>
        {active.label} selected. {active.items.length} benchmarks available.
      </p>

      <div className="pt-8 sm:pt-10">
        <BenchmarkGalleryClient
          key={benchmarkType}
          items={active.items}
          categories={categories}
          searchDocuments={active.searchDocuments}
          searchCategoryLabels={active.searchCategoryLabels}
          searchScopeLabel={active.scopeLabel}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          metaAnalysisId="benchmark-meta-analyses"
        />
      </div>

      <section
        id="benchmark-meta-analyses"
        className="mt-16 scroll-mt-20 border-t border-black/20 pt-10 sm:mt-20 sm:pt-12"
        aria-labelledby="benchmark-meta-analyses-title"
      >
        <ScrollReveal className="rd-figures">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(28rem,1.2fr)] lg:items-end lg:gap-12">
            <h3 id="benchmark-meta-analyses-title" className="rd-h2">
              Meta-analyses
            </h3>
            <p className="rd-prose-wide text-sm font-light leading-relaxed opacity-65 lg:justify-self-end">
              Coverage and timelines use the full data bank. The score heatmap
              follows benchmark type, not the current search or filters.
            </p>
          </div>

          <div className="rd-acc mt-7 sm:mt-9">
            {sharedAnalysisRows}
            <Fragment key={benchmarkType}>{activeScoreAnalysis}</Fragment>
          </div>
        </ScrollReveal>
      </section>
    </>
  );
}
