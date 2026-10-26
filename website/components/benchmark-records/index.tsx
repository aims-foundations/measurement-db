"use client";

import { useMemo, useState } from "react";
import {
  findSubjects,
  observationsFor,
  pageOf,
  type CellSelection,
} from "@/lib/benchmark-records/model";
import { describeScale } from "@/lib/benchmark-records/scale";
import type {
  GalleryEntry,
  ResponseScale,
} from "@/lib/benchmark-records/types";
import { CellDetails } from "./cell-details";
import { MatrixTable } from "./matrix-table";
import { Pagination } from "./pagination";
import { useGallery } from "./use-records";

const ROWS_PER_PAGE = 12;
const TASKS_PER_PAGE = 12;

/** The controller owns selection state; components receive prepared records. */
export function BenchmarkRecords({ entry }: { entry: GalleryEntry }) {
  const { source, state } = useGallery(entry);
  const [query, setQuery] = useState("");
  const [rowPage, setRowPage] = useState(0);
  const [taskPage, setTaskPage] = useState(0);
  const [selection, setSelection] = useState<CellSelection | null>(null);
  const scale = useMemo(
    () => JSON.parse(entry.manifest.benchmark.response_scale) as ResponseScale,
    [entry],
  );
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
  const model = state.value;
  const subjects = findSubjects(model.subjects, query);
  return (
    <div>
      <p className="text-sm text-[var(--muted)]">
        {model.subjects.length.toLocaleString()} models ·{" "}
        {model.itemIds.length.toLocaleString()} tasks ·{" "}
        {model.observations.length.toLocaleString()} recorded attempts ·{" "}
        {model.graded.toLocaleString()} graded
      </p>
      <p className="mt-2 text-xs leading-relaxed text-[var(--muted)]">
        {describeScale(scale)}
      </p>
      <p className="mt-2 text-xs text-[var(--muted)]">
        — No recorded observation · ? Grade unavailable · a count opens every
        recorded attempt.
      </p>
      <label className="mt-5 flex flex-wrap items-center gap-3 text-sm">
        Find a model
        <input
          type="search"
          className="rounded border border-[var(--line)] bg-white px-3 py-2"
          placeholder="Name or subject ID"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setRowPage(0);
          }}
        />
      </label>
      <div className="my-4 flex flex-wrap gap-x-8 gap-y-3">
        <Pagination
          label="models"
          page={rowPage}
          total={subjects.length}
          size={ROWS_PER_PAGE}
          onChange={setRowPage}
        />
        <Pagination
          label="tasks"
          page={taskPage}
          total={model.itemIds.length}
          size={TASKS_PER_PAGE}
          onChange={setTaskPage}
        />
      </div>
      {subjects.length === 0 && (
        <p className="mb-3 text-sm">No models match this search.</p>
      )}
      <MatrixTable
        model={model}
        subjects={pageOf(subjects, rowPage, ROWS_PER_PAGE)}
        items={pageOf(model.itemIds, taskPage, TASKS_PER_PAGE)}
        itemOffset={taskPage * TASKS_PER_PAGE}
        onSelect={setSelection}
      />
      {selection && (
        <CellDetails
          key={`${selection.subject?.subject_id ?? ""}/${selection.itemId}`}
          source={source}
          selection={selection}
          observations={observationsFor(model, selection)}
          scale={scale}
          onClose={() => setSelection(null)}
        />
      )}
    </div>
  );
}
