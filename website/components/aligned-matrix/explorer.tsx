"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { withBase } from "@/lib/base-path";
import { RecordSource } from "@/lib/benchmark-records/client";
import type { GalleryEntry } from "@/lib/benchmark-records/types";
import { loadMatrix, TestSource } from "@/lib/aligned-matrix/client";
import {
  fitRange,
  groupLabel,
  zoomRange,
  type ScoreMatrix,
} from "@/lib/aligned-matrix/model";
import type { Interval, SelectedCell } from "@/lib/aligned-matrix/types";
import { CONTROL } from "../benchmark-records/view-controls";
import { MatrixCanvas } from "./canvas";
import { TestDetails } from "./details";
import { COLORS, ROW_HEIGHT } from "./paint";
import { useRequest } from "./use-request";

const BASE = withBase("/benchmark-data/programbench/aligned-preview");
const number = (n: number) => n.toLocaleString("en-US");

export function AlignedPreview({ entry }: { entry: GalleryEntry }) {
  const revision =
    entry.manifest.tables.items?.parquet?.download?.match(
      /\/resolve\/([^/]+)\//,
    )?.[1] ?? "";
  const load = useCallback(
    (signal: AbortSignal) => loadMatrix(BASE, revision, signal),
    [revision],
  );
  const { state, retry } = useRequest(load);
  if (state.status === "loading")
    return (
      <p role="status" className="p-6 text-sm">
        Loading the compact scores for all tests…
      </p>
    );
  if (state.status === "error")
    return (
      <div role="alert" className="p-6 text-sm">
        {state.message}{" "}
        <button className={CONTROL} onClick={retry}>
          Retry matrix
        </button>
      </div>
    );
  return <Explorer matrix={state.value} entry={entry} />;
}

function Explorer({
  matrix,
  entry,
}: {
  matrix: ScoreMatrix;
  entry: GalleryEntry;
}) {
  const manifest = matrix.manifest;
  const [range, setRange] = useState<Interval>({
    start: 0,
    end: manifest.columns,
  });
  const [grader, setGrader] = useState("");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const detail = useRef<HTMLDivElement>(null);
  const source = useMemo(
    () => new RecordSource(withBase(entry.base), entry.manifest),
    [entry],
  );
  const tests = useMemo(() => new TestSource(BASE, manifest), [manifest]);
  const graders = useMemo(
    () =>
      [
        ...new Set(
          manifest.groups.flatMap((g) => g.setups.map((s) => s.grader)),
        ),
      ].sort(),
    [manifest],
  );
  const rowGraders = useMemo(
    () =>
      manifest.subjects.map((_, row) => [
        ...new Set(
          manifest.groups.flatMap((g) => {
            const setup = g.setups[g.rowSetups[row]];
            return setup ? [setup.grader] : [];
          }),
        ),
      ]),
    [manifest],
  );
  const rows = useMemo(
    () =>
      manifest.subjects
        .map((_, i) => i)
        .filter(
          (row) =>
            (!grader || rowGraders[row].includes(grader)) &&
            `${manifest.subjects[row].display_name} ${manifest.subjects[row].subject_id}`
              .toLowerCase()
              .includes(query.toLowerCase().trim()),
        ),
    [manifest, query, grader, rowGraders],
  );
  const span = range.end - range.start;
  const visibleGroup = manifest.groups.find(
    (g) => range.start >= g.start && range.end <= g.start + g.count,
  );
  const overview = range.start === 0 && range.end === manifest.columns;
  const changeRange = (next: Interval) => {
    setRange(next);
  };
  const select = (cell: SelectedCell) => {
    setSelected(cell);
    detail.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  return (
    <div>
      <div
        className="mb-5 flex flex-wrap gap-x-6 gap-y-2 text-sm"
        aria-label="Preview dataset counts"
      >
        <span>
          <strong className="font-mono">{number(manifest.columns)}</strong>{" "}
          tests
        </span>
        <span>
          <strong className="font-mono">{manifest.groups.length}</strong>{" "}
          testlets
        </span>
        <span>
          <strong className="font-mono">{manifest.subjects.length}</strong>{" "}
          model configurations
        </span>
        <span>
          <strong className="font-mono">{number(manifest.observations)}</strong>{" "}
          binary outcomes
        </span>
      </div>
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="flex w-full min-w-0 flex-col gap-1 text-xs text-[var(--muted)] sm:w-auto sm:flex-1">
          Go to testlet
          <select
            className={`${CONTROL} w-full min-w-0 max-w-full`}
            value={visibleGroup?.id ?? ""}
            aria-label="Go to testlet"
            onChange={(event) => {
              const group = manifest.groups.find(
                (g) => g.id === event.target.value,
              );
              changeRange(
                group
                  ? { start: group.start, end: group.start + group.count }
                  : { start: 0, end: manifest.columns },
              );
            }}
          >
            <option value="">All {manifest.groups.length} testlets</option>
            {manifest.groups.map((g) => (
              <option key={g.id} value={g.id}>
                {groupLabel(g)} · {number(g.count)} tests
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-[var(--muted)]">
          Grader
          <select
            className={CONTROL}
            value={grader}
            onChange={(e) => {
              setGrader(e.target.value);
              setSelected(null);
            }}
            aria-label="Grader version"
          >
            <option value="">All versions</option>
            {graders.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-xs text-[var(--muted)]">
          Find a model
          <input
            className={`${CONTROL} w-40 max-w-full`}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(null);
            }}
            placeholder="Name or ID"
          />
        </label>
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <button
          className={CONTROL}
          disabled={overview}
          onClick={() => changeRange({ start: 0, end: manifest.columns })}
        >
          All tests
        </button>
        <button
          className={CONTROL}
          disabled={span <= 1}
          onClick={() => changeRange(zoomRange(range, 0.25, manifest.columns))}
        >
          Zoom in
        </button>
        <button
          className={CONTROL}
          disabled={overview}
          onClick={() => changeRange(zoomRange(range, 4, manifest.columns))}
        >
          Zoom out
        </button>
        <span className="ml-auto text-xs tabular-nums text-[var(--muted)]">
          Tests {number(range.start + 1)}–{number(range.end)} of{" "}
          {number(manifest.columns)}
        </span>
      </div>
      {!overview && (
        <label className="mb-4 flex items-center gap-3 text-xs text-[var(--muted)]">
          Pan
          <input
            type="range"
            aria-label="Pan across tests"
            min={0}
            max={manifest.columns - span}
            value={range.start}
            step={1}
            onChange={(e) =>
              changeRange(
                fitRange(Number(e.target.value), span, manifest.columns),
              )
            }
            className="min-w-0 flex-1 accent-[var(--ink)]"
          />
        </label>
      )}
      <p
        id="aligned-help"
        className="mb-4 text-xs leading-relaxed text-[var(--muted)]"
      >
        Select the overview to zoom in; select an individual cell to inspect it.
        Drag horizontally or use Pan to move. Arrow keys navigate; Enter zooms
        or opens a cell. Shared headers mark testlets.
      </p>
      {rows.length ? (
        <div className="rounded-md border border-[var(--line)] bg-white p-2 sm:p-4">
          <div className="grid grid-cols-[82px_minmax(0,1fr)_36px] gap-x-2 sm:grid-cols-[190px_minmax(0,1fr)_52px] sm:gap-x-3">
            <div className="pb-2 text-[11px] text-[var(--muted)]">
              Model / grader
            </div>
            <div className="pb-2 text-[11px] text-[var(--muted)]">
              {visibleGroup ? groupLabel(visibleGroup) : "All testlets"}
            </div>
            <div
              className="pb-2 text-right text-[11px] text-[var(--muted)]"
              title="Fraction passed among observed tests in the visible range. Larger testlets contribute more tests; this is not the benchmark's task-averaged score."
            >
              Pass %
            </div>
            <div className="pt-10">
              {rows.map((row) => {
                const subject = manifest.subjects[row];
                return (
                  <div
                    key={row}
                    className="flex min-w-0 flex-col justify-center border-b border-[var(--line)]"
                    style={{ height: ROW_HEIGHT }}
                    title={`${subject.display_name} · ${subject.subject_id} · ${subject.harness ?? ""} ${subject.harness_version ?? ""} · grader ${rowGraders[row].join(", ")}`}
                  >
                    <span className="truncate text-[11px]">
                      {subject.display_name}
                    </span>
                    <span className="truncate font-mono text-[9px] text-[var(--muted)]">
                      {rowGraders[row].join("/")} ·{" "}
                      {subject.subject_id.slice(0, 6)}
                    </span>
                  </div>
                );
              })}
            </div>
            <MatrixCanvas
              matrix={matrix}
              rows={rows}
              range={range}
              grader={grader}
              selected={selected}
              onRange={changeRange}
              onSelect={select}
            />
            <div className="pt-10">
              {rows.map((row) => {
                const counts = matrix.counts(row, range, grader),
                  n = counts.passed + counts.failed;
                return (
                  <div
                    key={row}
                    className="flex items-center justify-end border-b border-[var(--line)] font-mono text-[11px] text-[var(--muted)]"
                    style={{ height: ROW_HEIGHT }}
                    title={`${number(counts.passed)} passed / ${number(n)} observed tests in view`}
                  >
                    {n ? ((100 * counts.passed) / n).toFixed(1) : "—"}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        <p className="py-10 text-sm" role="status">
          No model configurations match these filters.
        </p>
      )}
      <div
        className="mt-4 flex flex-wrap gap-5 text-xs text-[var(--muted)]"
        aria-label="Score legend"
      >
        {(
          [
            ["passed", "Pass"],
            ["failed", "Fail"],
            ["missing", "No observation in this selection"],
          ] as const
        ).map(([key, label]) => (
          <span className="inline-flex items-center gap-2" key={key}>
            <span
              className="h-3 w-3 rounded-sm"
              style={{ background: COLORS[key] }}
            />
            {label}
          </span>
        ))}
      </div>
      <p className="mt-3 text-xs leading-relaxed text-[var(--muted)]">
        At overview scale, blue, red, and gray bands show proportions within
        each strip. Pass % uses observed tests in the visible range, not a
        task-averaged benchmark score. Different recorded setups align by full
        test identifier; use the grader filter and cell details to inspect those
        differences.
      </p>
      <div ref={detail} className="scroll-mt-24">
        {selected && (
          <TestDetails
            matrix={matrix}
            tests={tests}
            source={source}
            selection={selected}
            onSelect={setSelected}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
      <details className="mt-6 rounded border border-[var(--line)] p-4 text-xs text-[var(--muted)]">
        <summary className="cursor-pointer font-medium">
          How test columns relate to the source items
        </summary>
        <p className="mt-3 leading-relaxed">
          The {number(manifest.canonicalItems)} canonical items remain
          unchanged. This preview aligns them into {number(manifest.columns)}{" "}
          columns using the complete upstream test identifier within each
          testlet, including its branch prefix. Every displayed outcome links to
          its original item, recorded setup, and attempt. This grouping is for
          display; no Rasch model is refitted.
        </p>
      </details>
    </div>
  );
}
