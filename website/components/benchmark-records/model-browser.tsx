"use client";

import { useState } from "react";
import type { BenchmarkIrt } from "@/content/benchmark-irt";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import { findSubjects, pageOf } from "@/lib/benchmark-records/model";
import { LoadedGallery } from "./loaded-gallery";
import { Pagination } from "./pagination";
import { useSubjects } from "./use-records";
import { CONTROL } from "./view-controls";

/** Bound browser memory by models; every item and attempt remains available. */
export function ModelBrowser({
  entry,
  irt,
}: {
  entry: GalleryEntry;
  irt?: Omit<BenchmarkIrt, "zByItem">;
}) {
  const state = useSubjects(entry);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  if (state.status !== "ready")
    return (
      <p role={state.status === "error" ? "alert" : "status"}>
        {state.status === "error" ? state.message : "Loading models…"}
      </p>
    );
  const subjects = findSubjects(state.value, query);
  const size = Math.max(
    1,
    Math.min(
      10,
      Math.floor(
        (200000 * state.value.length) / entry.manifest.tables.responses!.rows,
      ),
    ),
  );
  const selected = pageOf(subjects, page, size).map(
    (subject) => subject.subject_id,
  );
  return (
    <>
      <div className="mb-5 space-y-3">
        <p className="text-xs text-[var(--muted)]">
          All {state.value.length.toLocaleString()} models are available. Load a
          page or search for a model. Filters, ordering, and averages below
          apply to the models on this page.
        </p>
        <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
          Find a model
          <input
            type="search"
            className={CONTROL}
            placeholder="Name or subject ID"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
          />
        </label>
        <Pagination
          label="models"
          page={page}
          total={subjects.length}
          size={size}
          onChange={setPage}
        />
      </div>
      {selected.length ? (
        <LoadedGallery
          key={selected.join(",")}
          entry={entry}
          irt={irt}
          subjectIds={selected}
        />
      ) : (
        <p className="text-sm">No models match this search.</p>
      )}
    </>
  );
}
