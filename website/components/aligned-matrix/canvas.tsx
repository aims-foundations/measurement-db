"use client";

import { useEffect, useRef, useState } from "react";
import {
  fitRange,
  groupAt,
  groupLabel,
  pixelBins,
  type ScoreMatrix,
} from "@/lib/aligned-matrix/model";
import type { Interval, SelectedCell } from "@/lib/aligned-matrix/types";
import { paint, ROW_HEIGHT } from "./paint";

export function MatrixCanvas({
  matrix,
  rows,
  range,
  grader,
  selected,
  onRange,
  onSelect,
}: {
  matrix: ScoreMatrix;
  rows: number[];
  range: Interval;
  grader: string;
  selected: SelectedCell | null;
  onRange: (range: Interval) => void;
  onSelect: (cell: SelectedCell) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; start: number; moved: boolean } | null>(
    null,
  );
  const [width, setWidth] = useState(1);
  const [hover, setHover] = useState<{
    row: number;
    range: Interval;
    viewport: Interval;
  } | null>(null);
  const [focus, setFocus] = useState<SelectedCell | null>(null);
  const span = range.end - range.start;
  const detailed = width / span >= 3;
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(1, Math.floor(entry.contentRect.width))),
    );
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (canvas.current)
      paint(
        canvas.current,
        width,
        matrix,
        rows,
        range,
        grader,
        focus ?? selected,
      );
  }, [width, matrix, rows, range, grader, selected, focus]);

  const point = (clientX: number, clientY: number) => {
    const rect = canvas.current!.getBoundingClientRect();
    const row = Math.floor((clientY - rect.top) / ROW_HEIGHT);
    if (row < 0 || row >= rows.length) return null;
    const bins = pixelBins(range, width);
    const index = Math.min(
      bins.length - 1,
      Math.max(
        0,
        Math.floor(((clientX - rect.left) / rect.width) * bins.length),
      ),
    );
    return { row: rows[row], range: bins[index], viewport: range };
  };
  const inspect = (cell: SelectedCell) => {
    setFocus(cell);
    if (detailed) onSelect(cell);
    else {
      const nextSpan = Math.max(1, Math.floor(width / 7));
      onRange(
        fitRange(cell.column - nextSpan / 2, nextSpan, matrix.manifest.columns),
      );
    }
  };
  const validFocus =
    focus &&
    rows.includes(focus.row) &&
    focus.column >= range.start &&
    focus.column < range.end
      ? focus
      : null;
  const current =
    (hover?.viewport === range && rows.includes(hover.row) ? hover : null) ??
    (validFocus
      ? {
          row: validFocus.row,
          range: { start: validFocus.column, end: validFocus.column + 1 },
        }
      : null);
  const counts = current && matrix.counts(current.row, current.range, grader);
  const group = current && groupAt(matrix.manifest.groups, current.range.start);

  return (
    <div ref={container} className="min-w-0">
      <div
        className="relative h-10 overflow-hidden border-b border-[var(--line)]"
        aria-label="Testlet headers"
      >
        {matrix.manifest.groups.map((group) => {
          const start = Math.max(range.start, group.start),
            end = Math.min(range.end, group.start + group.count);
          if (start >= end) return null;
          return (
            <button
              key={group.id}
              title={`${group.id} · ${group.count.toLocaleString()} tests`}
              aria-label={`Zoom to ${group.id}`}
              data-testlet-id={group.id}
              tabIndex={((end - start) / span) * width > 48 ? 0 : -1}
              onClick={() =>
                onRange({ start: group.start, end: group.start + group.count })
              }
              className="absolute bottom-0 h-full overflow-hidden border-l border-[var(--line)] bg-[var(--fog-light)] px-1 text-left text-[10px] leading-tight hover:bg-white focus-visible:z-10"
              style={{
                left: `${((start - range.start) / span) * 100}%`,
                width: `${((end - start) / span) * 100}%`,
              }}
            >
              {((end - start) / span) * width > 48 && (
                <span className="block truncate">
                  {groupLabel(group)}
                  <br />
                  {group.count.toLocaleString()} tests
                </span>
              )}
            </button>
          );
        })}
      </div>
      <canvas
        ref={canvas}
        role="grid"
        aria-label="Aligned test matrix"
        tabIndex={0}
        aria-rowcount={rows.length}
        aria-colcount={matrix.manifest.columns}
        aria-describedby="aligned-help aligned-hover"
        data-start={range.start}
        data-end={range.end}
        data-detail={detailed}
        className="block w-full touch-pan-y outline-offset-2 focus-visible:outline-2 focus-visible:outline-[var(--ink)]"
        style={{
          height: rows.length * ROW_HEIGHT,
          cursor: detailed ? "crosshair" : "zoom-in",
        }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          drag.current = { x: event.clientX, start: range.start, moved: false };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const gesture = drag.current;
          if (gesture && Math.abs(event.clientX - gesture.x) > 5)
            gesture.moved = true;
          if (gesture?.moved) {
            onRange(
              fitRange(
                gesture.start - ((event.clientX - gesture.x) / width) * span,
                span,
                matrix.manifest.columns,
              ),
            );
            setHover(null);
          } else setHover(point(event.clientX, event.clientY));
        }}
        onPointerUp={(event) => {
          const gesture = drag.current;
          drag.current = null;
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
          if (!gesture || gesture.moved) return;
          const cell = point(event.clientX, event.clientY);
          if (cell)
            inspect({
              row: cell.row,
              column: Math.floor((cell.range.start + cell.range.end - 1) / 2),
            });
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onPointerLeave={() => setHover(null)}
        onKeyDown={(event) => {
          const active = validFocus ?? { row: rows[0], column: range.start };
          if (!active || active.row === undefined) return;
          let { row, column } = active;
          const position = rows.indexOf(row);
          const step = detailed ? 1 : Math.max(1, Math.floor(span / 50));
          if (event.key === "ArrowLeft") column -= step;
          else if (event.key === "ArrowRight") column += step;
          else if (event.key === "ArrowUp")
            row = rows[Math.max(0, position - 1)];
          else if (event.key === "ArrowDown")
            row = rows[Math.min(rows.length - 1, position + 1)];
          else if (event.key === "Home") column = range.start;
          else if (event.key === "End") column = range.end - 1;
          else if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inspect(active);
            return;
          } else return;
          event.preventDefault();
          column = Math.min(matrix.manifest.columns - 1, Math.max(0, column));
          setFocus({ row, column });
          setHover(null);
          if (column < range.start || column >= range.end)
            onRange(fitRange(column - span / 2, span, matrix.manifest.columns));
        }}
      />
      <p
        id="aligned-hover"
        className="mt-2 min-h-12 break-words text-[11px] text-[var(--muted)]"
        aria-live="polite"
      >
        {current && counts && group ? (
          <>
            {matrix.manifest.subjects[current.row].display_name} ·{" "}
            {groupLabel(group)} · tests {current.range.start + 1}–
            {current.range.end}. {counts.passed} pass · {counts.failed} fail ·{" "}
            {counts.missing} unavailable.{" "}
            {detailed
              ? "Select to inspect this test and setup."
              : "Select to zoom into individual tests."}
          </>
        ) : detailed ? (
          "Individual tests · select a cell for its exact item and model answer."
        ) : (
          "Density overview · each vertical strip summarizes several tests. Select a region to zoom in."
        )}
      </p>
    </div>
  );
}
