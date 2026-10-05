"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useId, useMemo, useState } from "react";
import {
  ArrowUpRight,
  CircleNotch,
  MagnifyingGlass,
  X,
} from "@phosphor-icons/react";
import {
  BENCHMARKS_INDEX_UID,
  RESULTS_INDEX_UID,
  getPublicSearchClient,
} from "@/lib/meilisearch";
import type {
  BenchmarkSearchHit,
  LocalBenchmarkSearchDocument,
  ModelResultSearchHit,
} from "@/lib/search-types";

const MIN_QUERY_LENGTH = 2;
const MAX_QUERY_LENGTH = 120;
const BENCHMARK_LIMIT = 5;
const RESULT_LIMIT = 6;
// The public benchmark corpus is intentionally small. Fetch enough ranked
// benchmark hits to scope them client-side without requiring a new live-index
// setting; subject-result rows are filtered by benchmarkSlug before their
// result limit is applied.
const REMOTE_BENCHMARK_SCAN_LIMIT = 250;

type SearchState = {
  status: "idle" | "loading" | "ready";
  benchmarks: BenchmarkSearchHit[];
  results: ModelResultSearchHit[];
  fallback: boolean;
};

const EMPTY_STATE: SearchState = {
  status: "idle",
  benchmarks: [],
  results: [],
  fallback: false,
};

function localMatches(
  documents: LocalBenchmarkSearchDocument[],
  query: string,
): BenchmarkSearchHit[] {
  const normalizedQuery = query.toLocaleLowerCase();
  const terms = normalizedQuery.split(/\s+/).filter(Boolean);
  return documents
    .map((document) => ({
      document,
      score:
        (document.searchText.includes(normalizedQuery) ? 10 : 0) +
        terms.reduce(
          (total, term) =>
            total +
            (document.searchText.includes(term) ? 1 : 0) +
            (document.name.toLocaleLowerCase().includes(term) ? 2 : 0),
          0,
        ),
    }))
    .filter(({ score }) => score > 0)
    .sort(
      (left, right) =>
        right.score - left.score ||
        left.document.name.localeCompare(right.document.name),
    )
    .slice(0, BENCHMARK_LIMIT)
    .map(({ document }) => ({
      slug: document.slug,
      name: document.name,
      description: document.description,
      categories: document.categories,
      license: document.license,
      releaseYear: document.releaseYear,
      items: document.items,
      models: document.models,
      resultCount: document.resultCount,
      itemResponses: document.itemResponses,
      scaleType: document.scaleType,
    }));
}

function formatScore(hit: ModelResultSearchHit): string {
  if (hit.score === null) return "Not released";
  if (hit.isBinary) {
    return hit.score.toLocaleString("en-US", {
      style: "percent",
      minimumFractionDigits: 0,
      maximumFractionDigits: 1,
    });
  }
  return hit.score.toLocaleString("en-US", {
    maximumFractionDigits: 3,
  });
}

function normalizedIdentifier(value: string): string {
  return value.toLocaleLowerCase().replace(/[^a-z0-9]+/g, "");
}

function CategoryTags({
  categories,
  labels,
}: {
  categories: string[];
  labels: Record<string, string>;
}) {
  return (
    <>
      {categories.slice(0, 2).map((category) => (
        <span
          key={category}
          className="rounded-full border border-black/15 px-2 py-0.5 text-[0.65rem] uppercase tracking-wide opacity-65"
        >
          {labels[category] ?? category.replaceAll("_", " ")}
        </span>
      ))}
    </>
  );
}

function BenchmarkHit({
  hit,
  categoryLabels,
}: {
  hit: BenchmarkSearchHit;
  categoryLabels: Record<string, string>;
}) {
  return (
    <li>
      <Link
        href={`/${hit.slug}`}
        className="group block border-t border-black/15 py-4 transition-colors hover:bg-black/[0.025] focus-visible:outline-offset-4"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h4 className="text-base font-medium leading-snug tracking-tight">
              {hit.name}
            </h4>
            <p className="mt-1 line-clamp-2 text-sm font-light leading-relaxed opacity-65">
              {hit.description}
            </p>
          </div>
          <ArrowUpRight
            size={16}
            aria-hidden="true"
            className="mt-1 shrink-0 opacity-45 transition-opacity group-hover:opacity-100"
          />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <CategoryTags categories={hit.categories} labels={categoryLabels} />
          <span className="font-mono text-[0.68rem] opacity-55">
            {hit.items.toLocaleString("en-US")} items
          </span>
          <span aria-hidden="true" className="text-xs opacity-30">
            ·
          </span>
          <span className="font-mono text-[0.68rem] opacity-55">
            {hit.resultCount.toLocaleString("en-US")} aggregate results
          </span>
        </div>
      </Link>
    </li>
  );
}

function ModelResultHit({
  hit,
  categoryLabels,
}: {
  hit: ModelResultSearchHit;
  categoryLabels: Record<string, string>;
}) {
  return (
    <li>
      <Link
        href={`/${hit.benchmarkSlug}`}
        className="group block border-t border-black/15 py-4 transition-colors hover:bg-black/[0.025] focus-visible:outline-offset-4"
      >
        <div className="flex items-start justify-between gap-5">
          <div className="min-w-0">
            <h4 className="truncate text-sm font-medium leading-snug">
              {hit.modelName}
            </h4>
            <p className="mt-1 text-sm font-light opacity-65">
              {hit.benchmarkName}
            </p>
          </div>
          <div className="shrink-0 text-right">
            <p className="font-mono text-sm font-medium">{formatScore(hit)}</p>
            <p className="mt-0.5 text-[0.62rem] uppercase tracking-wide opacity-50">
              mean response
            </p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <CategoryTags categories={hit.categories} labels={categoryLabels} />
          <span className="line-clamp-1 text-[0.68rem] opacity-55">
            {hit.scaleLabel}
          </span>
          <ArrowUpRight
            size={13}
            aria-hidden="true"
            className="ml-auto shrink-0 opacity-0 transition-opacity group-hover:opacity-60"
          />
        </div>
      </Link>
    </li>
  );
}

export function BenchmarkSearchClient({
  localDocuments,
  categoryLabels,
  resultLayout = "responsive",
  presentation = "default",
  scopeLabel = "the Measurement Data Bank",
  trailingControls,
  query: controlledQuery,
  onQueryChange,
}: {
  localDocuments: LocalBenchmarkSearchDocument[];
  categoryLabels: Record<string, string>;
  resultLayout?: "responsive" | "single";
  presentation?: "default" | "catalog";
  scopeLabel?: string;
  trailingControls?: ReactNode;
  query?: string;
  onQueryChange?: (query: string) => void;
}) {
  const generatedId = useId();
  const inputId = `benchmark-result-search-${generatedId}`;
  const benchmarkHeadingId = `benchmark-search-heading-${generatedId}`;
  const subjectHeadingId = `subject-result-search-heading-${generatedId}`;
  const [internalQuery, setInternalQuery] = useState("");
  const query = controlledQuery ?? internalQuery;
  function updateQuery(nextQuery: string) {
    if (controlledQuery === undefined) setInternalQuery(nextQuery);
    onQueryChange?.(nextQuery);
  }
  const [state, setState] = useState<SearchState>(EMPTY_STATE);
  const normalizedQuery = useMemo(
    () => query.trim().slice(0, MAX_QUERY_LENGTH),
    [query],
  );
  const allowedSlugs = useMemo(
    () => new Set(localDocuments.map((document) => document.slug)),
    [localDocuments],
  );
  const resultScopeFilter = useMemo(
    () =>
      [...allowedSlugs]
        .filter((slug) => /^[a-z0-9_-]+$/.test(slug))
        .map((slug) => `benchmarkSlug = ${JSON.stringify(slug)}`)
        .join(" OR "),
    [allowedSlugs],
  );

  useEffect(() => {
    if (normalizedQuery.length < MIN_QUERY_LENGTH) return;

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setState({ ...EMPTY_STATE, status: "loading" });
      const client = getPublicSearchClient();
      if (!client || !resultScopeFilter) {
        setState({
          status: "ready",
          benchmarks: localMatches(localDocuments, normalizedQuery),
          results: [],
          fallback: true,
        });
        return;
      }

      try {
        const response = await client.multiSearch(
          {
            queries: [
              {
                indexUid: BENCHMARKS_INDEX_UID,
                q: normalizedQuery,
                limit: REMOTE_BENCHMARK_SCAN_LIMIT,
              },
              {
                indexUid: RESULTS_INDEX_UID,
                q: normalizedQuery,
                limit: RESULT_LIMIT,
                filter: resultScopeFilter,
              },
            ],
          },
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;

        const benchmarkHits = (
          (response.results[0]?.hits ?? []) as unknown as BenchmarkSearchHit[]
        )
          .filter((hit) => allowedSlugs.has(hit.slug))
          .slice(0, BENCHMARK_LIMIT);
        let resultHits = (
          (response.results[1]?.hits ?? []) as unknown as ModelResultSearchHit[]
        ).filter((hit) => allowedSlugs.has(hit.benchmarkSlug));

        // When the query names one benchmark exactly, raw scores are safely
        // comparable within that benchmark. Fetch its highest means instead
        // of accepting arbitrary insertion order among its aggregate rows.
        const queryIdentifier = normalizedIdentifier(normalizedQuery);
        const exactBenchmark = benchmarkHits.find(
          (hit) =>
            normalizedIdentifier(hit.name) === queryIdentifier ||
            normalizedIdentifier(hit.slug) === queryIdentifier,
        );
        if (exactBenchmark && /^[a-z0-9_-]+$/.test(exactBenchmark.slug)) {
          const scopedResults = await client.index(RESULTS_INDEX_UID).search(
            "",
            {
              filter: `benchmarkSlug = ${JSON.stringify(exactBenchmark.slug)}`,
              sort: ["score:desc"],
              limit: RESULT_LIMIT,
            },
            { signal: controller.signal },
          );
          resultHits = (
            scopedResults.hits as unknown as ModelResultSearchHit[]
          ).filter((hit) => allowedSlugs.has(hit.benchmarkSlug));
        }
        if (controller.signal.aborted) return;

        setState({
          status: "ready",
          benchmarks: benchmarkHits,
          results: resultHits,
          fallback: false,
        });
      } catch {
        if (controller.signal.aborted) return;
        setState({
          status: "ready",
          benchmarks: localMatches(localDocuments, normalizedQuery),
          results: [],
          fallback: true,
        });
      }
    }, 250);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [allowedSlugs, localDocuments, normalizedQuery, resultScopeFilter]);

  const hasQuery = normalizedQuery.length >= MIN_QUERY_LENGTH;
  const hasResults = state.benchmarks.length > 0 || state.results.length > 0;
  const catalogPresentation = presentation === "catalog";

  return (
    <div className="min-w-0" aria-busy={state.status === "loading"}>
      <label htmlFor={inputId} className="sr-only">
        Search {scopeLabel}, AI subjects, and results
      </label>
      <div
        className={
          trailingControls
            ? "flex flex-col gap-2 sm:flex-row sm:items-center"
            : undefined
        }
      >
        <div className="relative min-w-0 flex-1">
          <MagnifyingGlass
            size={catalogPresentation ? 15 : 22}
            aria-hidden="true"
            className={`pointer-events-none absolute top-1/2 -translate-y-1/2 opacity-50 ${
              catalogPresentation ? "left-3" : "left-4"
            }`}
          />
          <input
            id={inputId}
            type="search"
            value={query}
            maxLength={MAX_QUERY_LENGTH}
            onChange={(event) => {
              updateQuery(event.target.value);
              setState(EMPTY_STATE);
            }}
            placeholder="Search benchmarks, AI subjects, or use cases…"
            autoComplete="off"
            className={
              catalogPresentation
                ? "min-h-11 w-full rounded-md border border-[var(--line)] bg-white py-2 pl-9 pr-12 text-sm text-[var(--ink)] transition-colors placeholder:text-[var(--muted)] hover:border-[var(--muted)] focus:border-[var(--digital-blue)] focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] xl:min-h-10"
                : "w-full rounded-xl border border-black/25 bg-white py-4 pl-12 pr-14 text-base shadow-sm transition-colors placeholder:text-black/50 hover:border-black/45 focus:border-[var(--rd-cardinal)] focus:outline-none sm:text-lg [&::-webkit-search-cancel-button]:hidden"
            }
          />
          {query ? (
            <button
              type="button"
              onClick={() => {
                updateQuery("");
                setState(EMPTY_STATE);
              }}
              aria-label={`Clear ${scopeLabel} search`}
              className={`absolute right-1 top-1/2 flex -translate-y-1/2 items-center justify-center rounded-full opacity-55 transition-opacity hover:opacity-100 ${
                catalogPresentation ? "h-10 w-10" : "h-11 w-11"
              }`}
            >
              <X size={17} aria-hidden="true" />
            </button>
          ) : null}
        </div>
        {trailingControls}
      </div>

      {!catalogPresentation || query.trim() || state.status !== "idle" ? (
        <div
          className="mt-3 min-h-5 text-xs font-light opacity-60"
          aria-live="polite"
        >
          {state.status === "loading" ? (
            <span className="inline-flex items-center gap-1.5">
              <CircleNotch
                size={13}
                className="animate-spin"
                aria-hidden="true"
              />
              Searching…
            </span>
          ) : hasQuery && state.status === "ready" ? (
            hasResults ? (
              `Showing ${state.benchmarks.length} benchmark match${state.benchmarks.length === 1 ? "" : "es"} and ${state.results.length} aggregate result${state.results.length === 1 ? "" : "s"}.`
            ) : (
              "No matches. Try a broader AI subject, domain, or task."
            )
          ) : query.trim() ? (
            "Enter at least two characters to search."
          ) : (
            "Searches benchmark descriptions, domains, AI subjects, and aggregate scores."
          )}
        </div>
      ) : null}

      {state.fallback && hasQuery ? (
        <p className="mt-2 text-xs font-light text-[var(--rd-cardinal)]">
          Aggregate-result search is temporarily unavailable; showing benchmark
          matches only.
        </p>
      ) : null}

      {hasQuery && state.status !== "idle" && hasResults ? (
        <div
          className={`${catalogPresentation ? "mt-5" : "mt-7"} grid min-w-0 grid-cols-[minmax(0,1fr)] gap-x-12 gap-y-8 ${
            resultLayout === "responsive" ? "lg:grid-cols-2" : ""
          }`}
        >
          <section className="min-w-0" aria-labelledby={benchmarkHeadingId}>
            <div className="mb-1 flex items-baseline justify-between gap-4">
              <h3 id={benchmarkHeadingId} className="rd-mono opacity-65">
                Benchmarks
              </h3>
              <span className="font-mono text-xs opacity-45">
                {state.benchmarks.length}
              </span>
            </div>
            {state.benchmarks.length > 0 ? (
              <ul>
                {state.benchmarks.map((hit) => (
                  <BenchmarkHit
                    key={hit.slug}
                    hit={hit}
                    categoryLabels={categoryLabels}
                  />
                ))}
              </ul>
            ) : (
              <p className="border-t border-black/15 py-4 text-sm font-light opacity-55">
                No benchmark-level matches.
              </p>
            )}
          </section>

          {!state.fallback ? (
            <section className="min-w-0" aria-labelledby={subjectHeadingId}>
              <div className="mb-1 flex items-baseline justify-between gap-4">
                <h3 id={subjectHeadingId} className="rd-mono opacity-65">
                  Aggregate results
                </h3>
                <span className="font-mono text-xs opacity-45">
                  {state.results.length}
                </span>
              </div>
              {state.results.length > 0 ? (
                <ul>
                  {state.results.map((hit) => (
                    <ModelResultHit
                      key={hit.id}
                      hit={hit}
                      categoryLabels={categoryLabels}
                    />
                  ))}
                </ul>
              ) : (
                <p className="border-t border-black/15 py-4 text-sm font-light opacity-55">
                  No aggregate-result matches.
                </p>
              )}
              <p className="mt-3 text-xs font-light leading-relaxed opacity-50">
                Mean responses use each benchmark&rsquo;s own scale and are not
                directly comparable across benchmarks.
              </p>
            </section>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
