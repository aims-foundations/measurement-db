"use client";

import { useEffect, useRef } from "react";
import type { CoverageMatrixData } from "@/lib/coverage-network/matrix";
import styles from "./matrix.module.css";

/** Draw only observed links, with a fixed-size canvas instead of a dense DOM. */
export function MatrixOverview({
  matrix,
  row,
  column,
  rowCount,
  columnCount,
  onNavigate,
}: {
  matrix: CoverageMatrixData;
  row: number;
  column: number;
  rowCount: number;
  columnCount: number;
  onNavigate: (row: number, column: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    function draw() {
      if (!element) return;
      const { width, height } = element.getBoundingClientRect();
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      element.width = width * scale;
      element.height = height * scale;
      const ctx = element.getContext("2d");
      if (!ctx) return;
      ctx.scale(scale, scale);
      const x = width / matrix.columns.length,
        y = height / matrix.rows.length;
      ctx.fillStyle = "#f0f3f1";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#247f80";
      for (const point of matrix.points)
        ctx.fillRect(
          point.column * x,
          point.row * y,
          Math.max(1, x),
          Math.max(1, y),
        );
      ctx.fillStyle = "#8c15151a";
      ctx.fillRect(column * x, row * y, columnCount * x, rowCount * y);
      ctx.strokeStyle = "#8c1515";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        column * x + 0.75,
        row * y + 0.75,
        Math.max(2, columnCount * x - 1.5),
        Math.max(2, rowCount * y - 1.5),
      );
    }
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    draw();
    return () => observer.disconnect();
  }, [matrix, row, column, rowCount, columnCount]);
  return (
    <div className={styles.overview}>
      <div className={styles.overviewLabel}>
        <span>
          All {matrix.rows.length} benchmarks × {matrix.columns.length} subjects
        </span>
        <span>Subjects →</span>
      </div>
      <button
        type="button"
        className={styles.navigator}
        aria-label="Coverage overview. Click to navigate; arrow keys move the detail window."
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          onNavigate(
            Math.floor(
              ((event.clientY - bounds.top) / bounds.height) *
                matrix.rows.length -
                rowCount / 2,
            ),
            Math.floor(
              ((event.clientX - bounds.left) / bounds.width) *
                matrix.columns.length -
                columnCount / 2,
            ),
          );
        }}
        onKeyDown={(event) => {
          const directions: Record<string, [number, number]> = {
            ArrowDown: [rowCount, 0],
            ArrowUp: [-rowCount, 0],
            ArrowLeft: [0, -columnCount],
            ArrowRight: [0, columnCount],
          };
          const step = directions[event.key];
          if (step) {
            event.preventDefault();
            onNavigate(row + step[0], column + step[1]);
          }
        }}
      >
        <canvas ref={canvas} aria-hidden="true" />
      </button>
      <p>
        Benchmarks ↓{" "}
        <span>
          Click anywhere to inspect. The outlined window is shown below.
        </span>
      </p>
    </div>
  );
}
