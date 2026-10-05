import {
  type GalleryItem,
  type ScaleType,
  BenchmarkGalleryClient,
} from "@/components/benchmark-gallery-client";
import { getBenchmarkDetail } from "@/content/benchmark-details";
import { benchmarkCategories, benchmarks } from "@/content/measurement-db";
import { buildBenchmarkSearchData } from "@/lib/benchmark-search-data";
import type { LocalBenchmarkSearchDocument } from "@/lib/search-types";

export type BenchmarkGalleryData = {
  items: GalleryItem[];
  searchDocuments: LocalBenchmarkSearchDocument[];
  searchCategoryLabels: Record<string, string>;
};

/** Prepare one public benchmark-type variant without moving shelf membership
 * into the browser bundle. */
export function buildBenchmarkGalleryData(
  slugs?: readonly string[],
): BenchmarkGalleryData {
  const shelf = slugs ? new Set(slugs) : null;
  const items: GalleryItem[] = benchmarks
    .filter((benchmark) => !shelf || shelf.has(benchmark.slug))
    .map((benchmark) => {
      const detail = getBenchmarkDetail(benchmark.slug);
      const release = detail?.releaseDate ?? null; // e.g. "2024-08" or "2020"
      const year = release ? Number(release.slice(0, 4)) : null;
      const scaleType: ScaleType = detail?.categories?.length
        ? "Categorical"
        : detail?.isBinary
          ? "Binary"
          : "Graded";
      return {
        ...benchmark,
        observed: detail?.stats.observed ?? null,
        year: Number.isFinite(year) ? year : null,
        scaleType,
        attacks: detail?.attacks ?? null,
      };
    });
  const { categoryLabels, localDocuments } = buildBenchmarkSearchData(slugs);

  return {
    items,
    searchDocuments: localDocuments,
    searchCategoryLabels: categoryLabels,
  };
}

// Server wrapper: enrich each gallery card with its % of (subject × item) cells
// observed (from the generated detail data) so the client can sort/show it —
// without bundling the multi-MB benchmark-details.json into the client.
export function BenchmarkGallery({
  slugs,
  showLegend = true,
  searchScopeLabel = "this benchmark catalog",
}: {
  // Restrict this instance to one shelf. Undefined means every card.
  slugs?: readonly string[];
  // The filter guide is identical on every instance, so only the first page
  // section should render it.
  showLegend?: boolean;
  // Used by the embedded advanced search's visible and accessible copy.
  searchScopeLabel?: string;
} = {}) {
  const data = buildBenchmarkGalleryData(slugs);

  return (
    <BenchmarkGalleryClient
      items={data.items}
      categories={benchmarkCategories}
      showLegend={showLegend}
      searchDocuments={data.searchDocuments}
      searchCategoryLabels={data.searchCategoryLabels}
      searchScopeLabel={searchScopeLabel}
    />
  );
}
