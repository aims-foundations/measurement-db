"use client";

import { useEffect, useState } from "react";
import { RecordSource } from "@/lib/benchmark-records/client";
import { withBase } from "@/lib/base-path";
import type { GalleryEntry, TestletIndex } from "@/lib/benchmark-records/types";
import { CONTROL } from "./view-controls";

/** Jump to a group's first bounded page; never fetch the whole testlet. */
export function TestletNavigation({
  entry,
  onChange,
}: {
  entry: GalleryEntry;
  onChange: (page: number) => void;
}) {
  const [result, setResult] = useState<{
    entry: GalleryEntry;
    index?: TestletIndex;
    error?: string;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    const source = new RecordSource(withBase(entry.base), entry.manifest);
    source.testlets(controller.signal).then(
      (index) => {
        if (!controller.signal.aborted) setResult({ entry, index });
      },
      () => {
        if (!controller.signal.aborted)
          setResult({
            entry,
            error:
              "Testlet navigation could not load. Item pages remain available.",
          });
      },
    );
    return () => controller.abort();
  }, [entry]);
  const index = result?.entry === entry ? result.index : undefined;
  if (result?.entry === entry && result.error)
    return (
      <p className="text-xs" role="status">
        {result.error}
      </p>
    );
  return (
    <label className="flex min-w-0 items-center gap-2 text-xs">
      Go to testlet
      <select
        aria-label="Go to testlet"
        className={`${CONTROL} min-w-0 max-w-64`}
        value=""
        disabled={!index}
        onChange={(event) => {
          if (index && Object.hasOwn(index, event.target.value))
            onChange(index[event.target.value].pages[0]);
        }}
      >
        <option value="">
          {index
            ? `${Object.keys(index).length} testlets`
            : "Loading testlets…"}
        </option>
        {Object.entries(index ?? {}).map(([id, group]) => (
          <option key={id} value={id}>
            {id} · {group.items.toLocaleString()} items
          </option>
        ))}
      </select>
    </label>
  );
}
