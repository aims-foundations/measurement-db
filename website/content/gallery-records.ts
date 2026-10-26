import catalog from "./generated/gallery-records.json";
import type { GalleryEntry } from "@/lib/benchmark-records/types";

/** Only lightweight manifests are bundled with the page; table rows stay lazy. */
export const galleryRecords = catalog as Record<string, GalleryEntry>;

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
