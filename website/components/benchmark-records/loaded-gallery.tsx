"use client";

import type { BenchmarkIrt } from "@/content/benchmark-irt";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import { RecordGallery } from "./record-gallery";
import { useGallery } from "./use-records";

export function LoadedGallery({
  entry,
  irt,
  subjectIds,
}: {
  entry: GalleryEntry;
  irt?: Omit<BenchmarkIrt, "zByItem">;
  subjectIds?: string[];
}) {
  const { source, state } = useGallery(entry, subjectIds);
  if (state.status === "loading")
    return (
      <p role="status" className="p-6 text-sm">
        Loading observation records…
      </p>
    );
  if (state.status === "error")
    return (
      <p role="alert" className="p-6 text-sm">
        {state.message}
      </p>
    );
  return (
    <RecordGallery
      source={source}
      model={state.value}
      metadata={entry.manifest.benchmark}
      irt={irt}
      paged={subjectIds !== undefined}
    />
  );
}
