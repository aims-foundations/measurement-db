"use client";

import { useEffect, useState, type ComponentType, type ReactNode } from "react";
import { readGallerySource } from "@/lib/gallery-source";
import type { AssetLink } from "@/lib/benchmark-records/types";
import { ItemContent } from "./item-content";

type ItemRecord = { content: string | null; asset_manifest: string | null };

/** Old chart bundles keep their matrices; selected items use exact source rows. */
export function SourceItemContent({
  base,
  itemId,
  benchmark,
  Text,
  renderAsset,
}: {
  base: string;
  itemId: string;
  benchmark: string;
  Text: ComponentType<{ text: string }>;
  renderAsset: (link: AssetLink) => ReactNode;
}) {
  const [record, setRecord] = useState<ItemRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    readGallerySource<ItemRecord>(
      base,
      "item-record",
      itemId,
      controller.signal,
    )
      .then((value) => {
        if (!controller.signal.aborted) setRecord(value);
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, [base, itemId, attempt]);
  if (error)
    return (
      <div role="alert" className="text-sm">
        <p>{error}</p>
        <button
          className="mt-2 underline"
          onClick={() => {
            setError(null);
            setAttempt(attempt + 1);
          }}
        >
          Retry item
        </button>
      </div>
    );
  if (!record)
    return (
      <p role="status" className="text-sm">
        Loading item…
      </p>
    );
  return (
    <ItemContent
      text={record.content}
      benchmark={benchmark}
      assets={JSON.parse(record.asset_manifest ?? "[]")}
      Text={Text}
      renderAsset={renderAsset}
    />
  );
}
