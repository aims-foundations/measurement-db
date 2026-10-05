import { getBenchmarkDetail } from "@/content/benchmark-details";
import { benchmarkCategories, benchmarks } from "@/content/measurement-db";
import type {
  LocalBenchmarkSearchDocument,
  ScaleType,
} from "@/lib/search-types";

function getScaleType(slug: string): ScaleType {
  const detail = getBenchmarkDetail(slug);
  if (detail?.categories?.length) return "Categorical";
  return detail?.isBinary ? "Binary" : "Graded";
}

function getReleaseYear(slug: string): number | null {
  const releaseDate = getBenchmarkDetail(slug)?.releaseDate;
  const year = Number(releaseDate?.slice(0, 4));
  return Number.isInteger(year) ? year : null;
}

/**
 * Build the small browser-side fallback corpus used by advanced search.
 *
 * The optional slug list is applied to the already-public benchmark catalog,
 * so a hand-maintained shelf cannot accidentally expose a hidden or
 * unpublished benchmark. The same resulting slugs scope Cloud result search.
 */
export function buildBenchmarkSearchData(slugs?: readonly string[]): {
  localDocuments: LocalBenchmarkSearchDocument[];
  categoryLabels: Record<string, string>;
} {
  const categoryLabels = Object.fromEntries(
    benchmarkCategories.map((category) => [category.id, category.label]),
  );
  const scope = slugs ? new Set(slugs) : null;
  const localDocuments: LocalBenchmarkSearchDocument[] = benchmarks
    .filter((benchmark) => !scope || scope.has(benchmark.slug))
    .map((benchmark) => {
      const detail = getBenchmarkDetail(benchmark.slug);
      const scaleType = getScaleType(benchmark.slug);
      const categoryText = benchmark.categories
        .map((category) => categoryLabels[category] ?? category)
        .join(" ");
      const institutionText =
        benchmark.institutions
          ?.map((institution) => institution.name)
          .join(" ") ?? "";

      return {
        slug: benchmark.slug,
        name: benchmark.name,
        description: benchmark.description,
        categories: benchmark.categories,
        license: benchmark.license,
        releaseYear: getReleaseYear(benchmark.slug),
        items: benchmark.items,
        models: benchmark.models,
        resultCount: detail?.matrixRows.length ?? 0,
        itemResponses: benchmark.itemResponses,
        scaleType,
        searchText: [
          benchmark.name,
          benchmark.description,
          detail?.description,
          categoryText,
          detail?.modality.join(" "),
          detail?.scaleLabel,
          institutionText,
        ]
          .filter(Boolean)
          .join(" ")
          .toLocaleLowerCase(),
      };
    });

  return { localDocuments, categoryLabels };
}
