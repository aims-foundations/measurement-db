import "server-only";
import type { ChartBundle } from "@/components/matrix-viewer";
import type { BenchmarkDetail } from "@/content/benchmark-details";
import { getBenchmark } from "@/content/measurement-db";

export function benchmarkDataOrigin(slug: string): string | null {
  const origin =
    process.env.GALLERY_DATA_ORIGIN ??
    (process.env.NODE_ENV === "development" ? "http://127.0.0.1:3050" : null);
  return origin && getBenchmark(slug) ? origin : null;
}

export async function getBenchmarkBundle(slug: string) {
  const origin = benchmarkDataOrigin(slug);
  if (!origin)
    throw new Error("Set GALLERY_DATA_ORIGIN to the gallery table service.");
  const response = await fetch(`${origin}/${slug}/view`, {
    cache: "no-store",
    signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) throw new Error(`Could not load source tables for ${slug}`);
  const { detail, ...bundle } = (await response.json()) as ChartBundle & {
    detail: BenchmarkDetail;
  };
  return { detail, bundle };
}
