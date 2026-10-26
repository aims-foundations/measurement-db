import type { Benchmark, BenchmarkCategoryId } from "@/content/measurement-db";
import type { GalleryEntry } from "./types";

/** Derive card data from the same manifest used by the benchmark page. */
export function recordCard(
  entry: GalleryEntry,
  previous?: Benchmark,
): Benchmark & { responses: number } {
  const { benchmark: metadata, tables } = entry.manifest;
  return {
    slug: metadata.benchmark_id,
    name: metadata.name,
    description:
      typeof metadata.one_line_description === "string"
        ? metadata.one_line_description
        : metadata.description,
    categories: metadata.domain as BenchmarkCategoryId[],
    license: metadata.license,
    code: typeof metadata.source_url === "string" ? metadata.source_url : "",
    paper: typeof metadata.paper_url === "string" ? metadata.paper_url : "",
    image: previous?.image ?? null,
    fit: previous?.fit ?? "cover",
    models: tables.subjects?.rows ?? 0,
    items: tables.items?.rows ?? 0,
    responses: tables.responses?.rows ?? 0,
    releaseDate:
      typeof metadata.release_date === "string" ? metadata.release_date : null,
    itemResponses: metadata.granularity === "item",
    saturation: previous?.saturation ?? null,
    institutions: previous?.institutions ?? null,
  };
}
