"use client";

import { useState } from "react";
import { useSearchParams } from "next/navigation";
import type { BenchmarkIrt } from "@/content/benchmark-irt";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import { useItemWindow } from "./use-records";
import { LoadedGallery } from "./loaded-gallery";
import { Pagination } from "./pagination";
import { CONTROL } from "./view-controls";
import { TestletNavigation } from "./testlet-navigation";

/** One bounded item window at a time, with every model and source item reachable. */
export function WindowBrowser({
  entry,
  irt,
}: {
  entry: GalleryEntry;
  irt?: Omit<BenchmarkIrt, "zByItem">;
}) {
  const params = useSearchParams();
  const total = entry.manifest.windows!.pages;
  const [page, setPage] = useState(() => {
    const requested = Number(params.get("itemPage"));
    return Number.isInteger(requested)
      ? Math.max(0, Math.min(total - 1, requested - 1))
      : 0;
  });
  const [item, setItem] = useState("");
  const [message, setMessage] = useState("");
  const { findItem, cancelFind } = useItemWindow(entry);
  const navigate = (next: number) => {
    cancelFind();
    setPage(Math.max(0, Math.min(total - 1, next)));
    setMessage("");
  };
  if (!total)
    return (
      <p className="text-sm">No items have been released for this benchmark.</p>
    );
  return (
    <>
      <div className="mb-5 space-y-3">
        <p className="text-xs text-[var(--muted)]">
          {(entry.manifest.tables.items?.rows ?? 0).toLocaleString()} items ·{" "}
          {entry.manifest.windows!.rows.toLocaleString()} observations.
          {total > 1 &&
            " Browse item pages to see all observations. Filters and means apply to the current page."}
          {!entry.manifest.windows!.rows &&
            " Item-level observations have not been released."}
        </p>
        {total > 1 && (
          <>
            <Pagination
              label="item pages"
              page={page}
              total={total}
              size={1}
              onChange={navigate}
            />
            <div className="flex flex-wrap items-center gap-3">
              {entry.manifest.windows?.testlets && (
                <TestletNavigation entry={entry} onChange={navigate} />
              )}
              <label className="flex items-center gap-2 text-xs">
                Go to page
                <input
                  aria-label="Go to item page"
                  type="number"
                  min={1}
                  max={total}
                  className={`${CONTROL} w-24`}
                  value={page + 1}
                  onChange={(event) => {
                    if (event.target.value)
                      navigate(Number(event.target.value) - 1);
                  }}
                />
              </label>
              <form
                className="flex min-w-0 flex-wrap items-center gap-2"
                onSubmit={async (event) => {
                  event.preventDefault();
                  setMessage("Finding item…");
                  try {
                    const range = await findItem(item.trim());
                    if (range) {
                      setPage(range[0]);
                      setMessage(
                        range[0] === range[1]
                          ? ""
                          : `This item's observations span pages ${range[0] + 1}–${range[1] + 1}.`,
                      );
                    } else setMessage("Item ID not found.");
                  } catch (error) {
                    if (error instanceof Error && error.name === "AbortError")
                      return;
                    setMessage(
                      error instanceof Error ? error.message : String(error),
                    );
                  }
                }}
              >
                <label className="text-xs">
                  Find an item
                  <input
                    className={`${CONTROL} ml-2 max-w-40`}
                    placeholder="Exact item ID"
                    value={item}
                    onChange={(event) => setItem(event.target.value)}
                  />
                </label>
                <button className={CONTROL} type="submit">
                  Find
                </button>
              </form>
            </div>
            {message && (
              <p role="status" className="text-xs">
                {message}
              </p>
            )}
          </>
        )}
      </div>
      <LoadedGallery key={page} entry={entry} irt={irt} windowPage={page} />
    </>
  );
}
