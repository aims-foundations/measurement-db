"use client";

import { useMemo, useState } from "react";
import type { BenchmarkIrt } from "@/content/benchmark-irt";
import type { RecordSource } from "@/lib/benchmark-records/client";
import {
  observationsFor,
  pageOf,
  type CellSelection,
  type MatrixModel,
} from "@/lib/benchmark-records/model";
import { prepareContexts } from "@/lib/benchmark-records/context";
import { comparable, prepareView } from "@/lib/benchmark-records/view";
import { prepareComparisons } from "@/lib/benchmark-records/comparisons";
import { presentationFor } from "@/lib/benchmark-records/presentation";
import type {
  BenchmarkMetadata,
  ResponseScale,
} from "@/lib/benchmark-records/types";
import { IrtPanel } from "../irt-panel";
import { CellDetails } from "./cell-details";
import { MatrixTable } from "./matrix-table";
import { Pagination } from "./pagination";
import { Heatmap } from "./heatmap";
import { Legend } from "./legend";
import { ViewControls, type ViewSettings } from "./view-controls";

const ROWS_PER_PAGE = 40,
  ITEMS_PER_PAGE = 12;

/** Compose prepared views. Network requests and record transformations live elsewhere. */
export function RecordGallery({
  source,
  model,
  metadata,
  irt,
  paged = false,
}: {
  source: RecordSource;
  model: MatrixModel;
  metadata: BenchmarkMetadata;
  irt?: Omit<BenchmarkIrt, "zByItem">;
  paged?: boolean;
}) {
  const scale = useMemo(
    () => JSON.parse(metadata.response_scale) as ResponseScale,
    [metadata],
  );
  const [settings, setSettings] = useState<ViewSettings>({
    filters: {},
    query: "",
    order: comparable(scale) ? "grade" : "recorded",
    display: "heatmap",
    actual: false,
  });
  const [rowPage, setRowPage] = useState(0),
    [itemPage, setItemPage] = useState(0);
  const [selection, setSelection] = useState<CellSelection | null>(null);
  const contexts = useMemo(() => prepareContexts(model.observations), [model]);
  const presentation = presentationFor(metadata.benchmark_id);
  const pairs = useMemo(
    () => prepareComparisons(model, presentation),
    [model, presentation],
  );
  const view = useMemo(
    () =>
      prepareView(
        model,
        contexts,
        settings.filters,
        settings.query,
        settings.order,
        scale,
      ),
    [model, contexts, settings.filters, settings.query, settings.order, scale],
  );
  const subjects = useMemo(
    () => pageOf(view.subjects, rowPage, ROWS_PER_PAGE),
    [view, rowPage],
  );
  const update = (next: ViewSettings) => {
    setSettings(next);
    setSelection(null);
    setRowPage(0);
    setItemPage(0);
  };
  const publishedFit =
    irt &&
    metadata.response_type === "binary" &&
    irt.nTrain + irt.nTest === source.manifest.tables.responses?.rows &&
    irt.nItems === model.itemIds.length &&
    irt.nSubjects === source.manifest.tables.subjects?.rows &&
    model.subjects.every((subject) =>
      Object.hasOwn(irt.theta, subject.subject_id),
    )
      ? irt
      : undefined;
  return (
    <div>
      <p id="heatmap-help" className="mb-3 text-xs text-[var(--muted)]">
        Hover for model and item; click a cell for details. Arrow keys move
        between cells; Enter opens the selected cell.
      </p>
      <ViewControls
        dimensions={contexts.dimensions}
        settings={settings}
        canOrder={comparable(scale)}
        onChange={update}
        paged={paged}
      />
      <p className="mb-3 text-xs text-[var(--muted)]">
        {paged && "On this page: "}
        {view.subjects.length.toLocaleString()} models ·{" "}
        {view.itemIds.length.toLocaleString()} items ·{" "}
        {view.observations.length.toLocaleString()} recorded attempts ·{" "}
        {view.graded.toLocaleString()} graded
      </p>
      {view.subjects.length > ROWS_PER_PAGE && (
        <div className="mb-3">
          <Pagination
            label="models"
            page={rowPage}
            total={view.subjects.length}
            size={ROWS_PER_PAGE}
            onChange={(page) => {
              setRowPage(page);
              setSelection(null);
            }}
          />
        </div>
      )}
      {settings.display === "heatmap" ? (
        <Heatmap
          key={`${rowPage}/${JSON.stringify(settings.filters)}/${settings.order}/${settings.query}`}
          view={view}
          subjects={subjects}
          scale={scale}
          actual={settings.actual}
          theta={publishedFit?.theta}
          onSelect={setSelection}
        />
      ) : (
        <>
          <div className="mb-3">
            <Pagination
              label="items"
              page={itemPage}
              total={view.itemIds.length}
              size={ITEMS_PER_PAGE}
              onChange={setItemPage}
            />
          </div>
          <MatrixTable
            model={view}
            subjects={subjects}
            items={pageOf(view.itemIds, itemPage, ITEMS_PER_PAGE)}
            itemOffset={itemPage * ITEMS_PER_PAGE}
            scale={scale}
            onSelect={setSelection}
          />
        </>
      )}
      <Legend scale={scale} />
      {selection && (
        <CellDetails
          key={JSON.stringify([
            selection.subject?.subject_id,
            selection.itemId,
          ])}
          source={source}
          selection={selection}
          observations={observationsFor(view, selection)}
          scale={scale}
          presentation={presentation}
          model={model}
          pairs={pairs}
          onClose={() => setSelection(null)}
        />
      )}
      {publishedFit && (
        <div className="mt-6 border-t border-[var(--line)] pt-6">
          <IrtPanel irt={{ ...publishedFit, zByItem: {} }} />
          <p className="text-[0.6875rem] text-[var(--muted)]">
            Published fit for the full benchmark. Filters affect only the
            displayed observations.
          </p>
        </div>
      )}
    </div>
  );
}
