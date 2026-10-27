"use client";

import type { BenchmarkIrt } from "@/content/benchmark-irt";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import { LoadedGallery } from "./loaded-gallery";
import { ModelBrowser } from "./model-browser";
import { WindowBrowser } from "./window-browser";

/** Loading owns no visual decisions; the gallery receives complete canonical records. */
export function BenchmarkRecords({
  entry,
  irt,
}: {
  entry: GalleryEntry;
  irt?: Omit<BenchmarkIrt, "zByItem">;
}) {
  if (entry.manifest.windows) return <WindowBrowser entry={entry} irt={irt} />;
  if (
    (entry.manifest.tables.responses?.rows ?? 0) > 200000 &&
    entry.manifest.tables.responses?.bySubject
  )
    return <ModelBrowser entry={entry} irt={irt} />;
  return <LoadedGallery entry={entry} irt={irt} />;
}
