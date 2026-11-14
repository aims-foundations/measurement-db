import catalog from "./generated/gallery-records.json";
import previewJson from "./generated/record-previews.json";
import tracePreviewJson from "./generated/trace-previews.json";
import {
  matchingTracePreview,
  type TracePreview,
} from "@/lib/benchmark-records/trace-preview";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import {
  matchingPreviews,
  type RecordPreview,
} from "@/lib/benchmark-records/record-preview";

/** Only lightweight manifests are bundled with the page; table rows stay lazy. */
const previews = previewJson as unknown as Record<
  string,
  Record<string, RecordPreview>
>;
const tracePreviews = tracePreviewJson as unknown as Record<
  string,
  TracePreview
>;
export const galleryRecords: Record<string, GalleryEntry> = Object.fromEntries(
  Object.entries(catalog as Record<string, GalleryEntry>).map(
    ([slug, entry]) => [
      slug,
      {
        ...entry,
        manifest: {
          ...entry.manifest,
          // A cache for another source revision must never supply this record.
          previews: matchingPreviews(entry.manifest.tables, previews[slug]),
          tracePreview: matchingTracePreview(
            entry.manifest.tables.traces?.parquet?.download,
            tracePreviews[slug],
          ),
        },
      },
    ],
  ),
);

export function getGalleryRecords(slug: string): GalleryEntry | undefined {
  return galleryRecords[slug];
}

export function recordPageSummary(entry: GalleryEntry) {
  const { benchmark, tables } = entry.manifest;
  return {
    slug: benchmark.benchmark_id,
    name: benchmark.name,
    description: benchmark.description,
    domain: benchmark.domain,
    modality: benchmark.modality,
    license: benchmark.license,
    note: null,
    stats: {
      items: tables.items?.rows ?? 0,
      subjects: tables.subjects?.rows ?? 0,
    },
  };
}
