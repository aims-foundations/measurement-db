"use client";

import Link from "next/link";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import {
  buildCoverageMatrix,
  matrixWindow,
  type CoverageMatrixData,
  type MatrixOrder,
} from "@/lib/coverage-network/matrix";
import type { CoverageNode, CoverageEdge } from "@/lib/coverage-network/model";
import { MatrixOverview } from "./matrix-overview";
import styles from "./matrix.module.css";

const ROWS = 12;
const count = (n: number) => n.toLocaleString("en-US");

export function CoverageMatrix({
  nodes,
  edges,
  selected,
  onSelect,
}: {
  nodes: CoverageNode[];
  edges: CoverageEdge[];
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const [order, setOrder] = useState<MatrixOrder>("coverage");
  const matrix = useMemo(
    () => buildCoverageMatrix(nodes, edges, order),
    [nodes, edges, order],
  );
  return (
    <div className={styles.matrix} data-matrix-links={matrix.points.length}>
      <div className={styles.heading}>
        <p>
          <strong>Evaluation coverage</strong>
          <br />
          Filled cells mark recorded evaluations; blank cells have no mapped
          record in this snapshot.
        </p>
        <label>
          Order by{" "}
          <select
            value={order}
            onChange={(event) => setOrder(event.target.value as MatrixOrder)}
          >
            <option value="coverage">Most connected</option>
            <option value="name">Name</option>
            <option value="group">Domain / provider</option>
          </select>
        </label>
      </div>
      {matrix.rows.length && matrix.columns.length ? (
        <MatrixDetail
          key={order}
          matrix={matrix}
          selected={selected}
          onSelect={onSelect}
        />
      ) : (
        <p className={styles.empty}>
          No mapped subject connections match these filters. Benchmarks without
          mapped subjects remain available in the search results.
        </p>
      )}
    </div>
  );
}

function MatrixDetail({
  matrix,
  selected,
  onSelect,
}: {
  matrix: CoverageMatrixData;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [columnsPerPage, setColumnsPerPage] = useState(10);
  const [position, setPosition] = useState({
    row: matrix.rowIndex.get(selected || "") ?? 0,
    column: matrix.columnIndex.get(selected || "") ?? 0,
    selection: selected,
  });
  const [cell, setCell] = useState<{ row: string; column: string } | null>(
    null,
  );
  // A selection in the shared search list brings its row or column into view.
  if (position.selection !== selected) {
    setPosition({
      row: matrix.rowIndex.get(selected || "") ?? position.row,
      column: matrix.columnIndex.get(selected || "") ?? position.column,
      selection: selected,
    });
  }
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      setColumnsPerPage(
        Math.max(
          3,
          Math.min(18, Math.floor((width - (width < 450 ? 110 : 165)) / 44)),
        ),
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const rowWindow = matrixWindow(matrix.rows.length, ROWS, position.row);
  const columnWindow = matrixWindow(
    matrix.columns.length,
    columnsPerPage,
    position.column,
  );
  const rows = matrix.rows.slice(rowWindow.start, rowWindow.end);
  const columns = matrix.columns.slice(columnWindow.start, columnWindow.end);
  const activeRow = cell
    ? matrix.rows.find((n) => n.id === cell.row)
    : undefined;
  const activeColumn = cell
    ? matrix.columns.find((n) => n.id === cell.column)
    : undefined;
  const activeItems =
    activeRow && activeColumn
      ? matrix.values.get(JSON.stringify([activeColumn.id, activeRow.id]))
      : undefined;
  function navigate(row: number, column: number) {
    setPosition({
      row: matrixWindow(matrix.rows.length, ROWS, row).start,
      column: matrixWindow(matrix.columns.length, columnsPerPage, column).start,
      selection: selected,
    });
  }
  function moveCell(
    event: KeyboardEvent<HTMLButtonElement>,
    row: number,
    column: number,
  ) {
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    const r = Math.max(0, Math.min(matrix.rows.length - 1, row + move[0]));
    const c = Math.max(
      0,
      Math.min(matrix.columns.length - 1, column + move[1]),
    );
    if (
      r < rowWindow.start ||
      r >= rowWindow.end ||
      c < columnWindow.start ||
      c >= columnWindow.end
    )
      navigate(
        r < rowWindow.start || r >= rowWindow.end ? r : rowWindow.start,
        c < columnWindow.start || c >= columnWindow.end
          ? c
          : columnWindow.start,
      );
    setCell({ row: matrix.rows[r].id, column: matrix.columns[c].id });
    requestAnimationFrame(() =>
      container.current
        ?.querySelector<HTMLButtonElement>(`[data-cell="${r}:${c}"]`)
        ?.focus(),
    );
  }
  const activeCellVisible =
    cell &&
    rows.some((n) => n.id === cell.row) &&
    columns.some((n) => n.id === cell.column);
  return (
    <div ref={container}>
      <MatrixOverview
        matrix={matrix}
        row={rowWindow.start}
        column={columnWindow.start}
        rowCount={rows.length}
        columnCount={columns.length}
        onNavigate={navigate}
      />
      <div className={styles.pagination}>
        <Pager
          label="Benchmarks"
          start={rowWindow.start}
          end={rowWindow.end}
          total={matrix.rows.length}
          onPrevious={() =>
            navigate(rowWindow.start - ROWS, columnWindow.start)
          }
          onNext={() => navigate(rowWindow.end, columnWindow.start)}
        />
        <Pager
          label="Subjects"
          start={columnWindow.start}
          end={columnWindow.end}
          total={matrix.columns.length}
          onPrevious={() =>
            navigate(rowWindow.start, columnWindow.start - columnsPerPage)
          }
          onNext={() => navigate(rowWindow.start, columnWindow.end)}
        />
      </div>
      <table
        className={styles.table}
        role="grid"
        aria-label="Benchmark-subject coverage matrix"
        aria-rowcount={matrix.rows.length + 1}
        aria-colcount={matrix.columns.length + 1}
      >
        <thead>
          <tr aria-rowindex={1}>
            <th scope="col" className={styles.corner}>
              Benchmark ↓<br />
              AI subject →
            </th>
            {columns.map((column, i) => (
              <th
                scope="col"
                key={column.id}
                aria-colindex={columnWindow.start + i + 2}
              >
                <button
                  type="button"
                  title={column.label}
                  aria-label={`Inspect ${column.label}`}
                  onClick={() => onSelect(column.id)}
                  className={
                    selected === column.id ? styles.selectedHeader : undefined
                  }
                >
                  {column.label}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={row.id} aria-rowindex={rowWindow.start + ri + 2}>
              <th scope="row">
                <button
                  type="button"
                  title={row.label}
                  className={
                    selected === row.id ? styles.selectedHeader : undefined
                  }
                  onClick={() => onSelect(row.id)}
                >
                  {row.label}
                </button>
              </th>
              {columns.map((column, ci) => {
                const items = matrix.values.get(
                  JSON.stringify([column.id, row.id]),
                );
                const active =
                  cell?.row === row.id && cell.column === column.id;
                const label = `${row.label} × ${column.label}: ${items === undefined ? "no mapped record" : count(items) + " items evaluated"}`;
                return (
                  <td
                    key={column.id}
                    aria-colindex={columnWindow.start + ci + 2}
                    aria-selected={active}
                  >
                    <button
                      type="button"
                      data-cell={`${rowWindow.start + ri}:${columnWindow.start + ci}`}
                      aria-label={label}
                      title={label}
                      data-observed={items !== undefined}
                      aria-pressed={active}
                      tabIndex={
                        active || (!activeCellVisible && ri === 0 && ci === 0)
                          ? 0
                          : -1
                      }
                      onClick={() =>
                        setCell({ row: row.id, column: column.id })
                      }
                      onKeyDown={(event) =>
                        moveCell(
                          event,
                          rowWindow.start + ri,
                          columnWindow.start + ci,
                        )
                      }
                    >
                      <span aria-hidden="true">
                        {items === undefined ? "" : "·"}
                      </span>
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className={styles.readout} aria-live="polite">
        {activeRow && activeColumn ? (
          <>
            <strong>
              {activeRow.label} × {activeColumn.label}
            </strong>
            <span>
              {activeItems === undefined
                ? "No mapped evaluation record in this snapshot."
                : `${count(activeItems)} items evaluated. This is a coverage count, not a score.`}
            </span>
            <Link href={`/${activeRow.slug}`}>Open benchmark →</Link>
          </>
        ) : (
          <span>
            Select a cell to read its benchmark, subject, and evaluated-item
            count. Arrow keys move between cells.
          </span>
        )}
      </div>
    </div>
  );
}

function Pager({
  label,
  start,
  end,
  total,
  onPrevious,
  onNext,
}: {
  label: string;
  start: number;
  end: number;
  total: number;
  onPrevious: () => void;
  onNext: () => void;
}) {
  return (
    <div className={styles.pager}>
      <span>
        {label}{" "}
        <strong>
          {start + 1}–{end}
        </strong>{" "}
        / {total}
      </span>
      <button
        type="button"
        aria-label={`Previous ${label.toLowerCase()}`}
        disabled={start === 0}
        onClick={onPrevious}
      >
        ←
      </button>
      <button
        type="button"
        aria-label={`Next ${label.toLowerCase()}`}
        disabled={end === total}
        onClick={onNext}
      >
        →
      </button>
    </div>
  );
}
