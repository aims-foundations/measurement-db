"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import type { MatrixView } from "@/lib/benchmark-records/view";
import type { CellSelection } from "@/lib/benchmark-records/model";
import type { ResponseScale, Subject } from "@/lib/benchmark-records/types";
import { gradeLabel } from "@/lib/benchmark-records/scale";
import { paintHeatmap, ROW_HEIGHT } from "./paint-heatmap";
import { columnGroups } from "@/lib/benchmark-records/testlets";
import { columnAt, matrixLayout } from "@/lib/benchmark-records/matrix-layout";
import { TestletHeader, TESTLET_HEADER_HEIGHT } from "./testlet-header";
type Position = { row: number; col: number };

/** Draw the visible window. Selection reads records by ID, never canvas colors. */
export function HeatmapCanvas({
  view,
  subjects,
  scale,
  actual,
  onSelect,
}: {
  view: MatrixView;
  subjects: Subject[];
  scale: ResponseScale;
  actual: boolean;
  onSelect: (cell: CellSelection) => void;
}) {
  const box = useRef<HTMLDivElement>(null),
    canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0),
    [scroll, setScroll] = useState(0);
  const [active, setActive] = useState<Position>({ row: 0, col: 0 });
  const columns = view.itemIds.length,
    height = subjects.length * ROW_HEIGHT;
  const layout = useMemo(
    () => matrixLayout(columns, columnGroups(view), width, actual),
    [columns, view, width, actual],
  );
  const headerHeight = layout.grouped ? TESTLET_HEADER_HEIGHT : 0;
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    if (box.current) observer.observe(box.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx || !width || !height) return;
    paintHeatmap(
      ctx,
      view,
      subjects,
      scale,
      width,
      layout,
      actual ? scroll : 0,
    );
  }, [view, subjects, scale, actual, width, height, scroll, layout, columns]);
  const row = Math.min(active.row, subjects.length - 1),
    col = Math.min(active.col, columns - 1);
  const subject = subjects[row],
    itemId = view.itemIds[col];
  const attempts = view.cells.get(subject?.subject_id)?.get(itemId) ?? [];
  const value = !attempts.length
    ? view.partialItemIds?.includes(itemId)
      ? "No observation on this page"
      : "No recorded observation"
    : attempts.length > 1
      ? `${attempts.length} observations`
      : scale.kind === "mixed"
        ? String(attempts[0].response ?? "Grade unavailable")
        : gradeLabel(attempts[0].response, scale);
  const select = (position: Position) =>
    onSelect({
      subject: subjects[position.row],
      itemId: view.itemIds[position.col],
    });
  const locate = (
    event: React.MouseEvent<HTMLCanvasElement>,
  ): Position | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const column = columnAt(
      layout,
      event.clientX - rect.left + (actual ? scroll : 0),
    );
    if (column === null) return null;
    return {
      row: Math.min(
        subjects.length - 1,
        Math.max(0, Math.floor((event.clientY - rect.top) / ROW_HEIGHT)),
      ),
      col: column,
    };
  };
  const keyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select({ row, col });
      return;
    }
    const delta: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
      Home: [0, -col],
      End: [0, columns - 1 - col],
    };
    if (!delta[event.key]) return;
    event.preventDefault();
    const [dy, dx] = delta[event.key];
    const next = {
      row: Math.max(0, Math.min(subjects.length - 1, row + dy)),
      col: Math.max(0, Math.min(columns - 1, col + dx)),
    };
    setActive(next);
    if (actual && box.current)
      box.current.scrollLeft = Math.max(
        0,
        layout.positions[next.col] - width / 2,
      );
  };
  return (
    <div className="min-w-0">
      <div
        ref={box}
        className="overflow-x-auto outline-offset-2 focus-visible:outline-2"
        tabIndex={0}
        role="grid"
        aria-label="Response heatmap"
        aria-rowcount={subjects.length}
        aria-colcount={columns}
        aria-describedby="heatmap-help"
        onKeyDown={keyboard}
        onScroll={(e) => setScroll(e.currentTarget.scrollLeft)}
      >
        <div
          style={{
            width: layout.width || "100%",
            height: height + headerHeight,
          }}
        >
          <div
            className="sticky left-0 overflow-hidden"
            style={{ width, height: height + headerHeight }}
          >
            {layout.grouped && (
              <div
                className="relative overflow-hidden"
                style={{ height: headerHeight }}
                aria-label="Testlet groups"
              >
                {layout.groups.map((group) => {
                  const left = group.left - (actual ? scroll : 0);
                  const visibleLeft = Math.max(0, left);
                  const visibleWidth =
                    Math.min(width, left + group.width) - visibleLeft;
                  if (visibleWidth <= 0) return null;
                  return (
                    <div
                      key={group.start}
                      className="absolute top-0"
                      style={{
                        left: visibleLeft,
                        width: visibleWidth,
                      }}
                    >
                      <TestletHeader group={group} />
                    </div>
                  );
                })}
              </div>
            )}
            <canvas
              ref={canvas}
              aria-hidden="true"
              className="block cursor-crosshair"
              style={{ width, height }}
              onMouseMove={(event) => {
                const position = locate(event);
                if (position) setActive(position);
              }}
              onClick={(event) => {
                const position = locate(event);
                if (!position) return;
                setActive(position);
                select(position);
              }}
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute border border-black"
              style={{
                top: headerHeight + row * ROW_HEIGHT,
                left: layout.positions[col] - (actual ? scroll : 0),
                width: Math.max(2, layout.cellWidth),
                height: ROW_HEIGHT,
              }}
            />
          </div>
        </div>
        <div className="sr-only" role="row" aria-rowindex={row + 1}>
          <span role="gridcell" aria-colindex={col + 1}>
            {subject?.display_name}, item {col + 1}: {value}
          </span>
        </div>
      </div>
      <p
        className="mt-2 min-h-8 break-words text-[0.6875rem] text-[var(--muted)]"
        role="status"
        aria-live="polite"
      >
        {subject?.display_name} · Item {col + 1}: {value}
        <span className="block font-mono text-[0.6rem]">{itemId}</span>
      </p>
    </div>
  );
}
