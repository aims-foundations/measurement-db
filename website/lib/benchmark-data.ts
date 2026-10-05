import "server-only";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import type { ChartBundle } from "@/components/matrix-viewer";
import type { BenchmarkDetail } from "@/content/benchmark-details";
import { getBenchmark } from "@/content/measurement-db";

export async function getBenchmarkBundle(slug: string) {
  if (!getBenchmark(slug)) throw new Error("Unknown benchmark");
  const path = join(
    process.cwd(),
    "public/benchmark-data",
    slug,
    "view.json.gz",
  );
  const bytes = await readFile(path);
  const { detail, ...bundle } = JSON.parse(
    gunzipSync(bytes).toString("utf8"),
  ) as ChartBundle & {
    detail: BenchmarkDetail;
  };
  return { detail, bundle };
}
