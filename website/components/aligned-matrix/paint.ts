import { pixelBins, type ScoreMatrix } from "@/lib/aligned-matrix/model";
import type { Interval, SelectedCell } from "@/lib/aligned-matrix/types";

export const ROW_HEIGHT = 32;
export const COLORS = {
  passed: "#1f77b4",
  failed: "#d62728",
  missing: "#d8dadd",
};

/** Proportional bands retain all bin counts; colors are never averaged to purple. */
export function paint(
  canvas: HTMLCanvasElement,
  width: number,
  matrix: ScoreMatrix,
  rows: number[],
  range: Interval,
  grader: string,
  selected: SelectedCell | null,
) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(rows.length * ROW_HEIGHT * dpr);
  const context = canvas.getContext("2d");
  if (!context) return;
  const bins = pixelBins(range, width);
  const height = ROW_HEIGHT * dpr;
  for (let r = 0; r < rows.length; r++) {
    for (let i = 0; i < bins.length; i++) {
      const x = Math.round((i * canvas.width) / bins.length);
      const right = Math.round(((i + 1) * canvas.width) / bins.length);
      const counts = matrix.counts(rows[r], bins[i], grader);
      const total = bins[i].end - bins[i].start;
      const h = height - dpr;
      const blue = Math.round((h * counts.passed) / total);
      const red = Math.round((h * counts.failed) / total);
      context.fillStyle = COLORS.missing;
      context.fillRect(x, r * height, right - x, h);
      context.fillStyle = COLORS.passed;
      context.fillRect(x, r * height, right - x, blue);
      context.fillStyle = COLORS.failed;
      context.fillRect(x, r * height + blue, right - x, red);
    }
  }
  context.strokeStyle = "rgba(40, 45, 50, .45)";
  context.lineWidth = dpr;
  for (const group of matrix.manifest.groups) {
    if (group.start <= range.start || group.start >= range.end) continue;
    const x = Math.round(
      ((group.start - range.start) / (range.end - range.start)) * canvas.width,
    );
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, canvas.height);
    context.stroke();
  }
  if (
    selected &&
    selected.column >= range.start &&
    selected.column < range.end &&
    rows.includes(selected.row)
  ) {
    const x =
      ((selected.column - range.start) / (range.end - range.start)) *
      canvas.width;
    const w = canvas.width / (range.end - range.start);
    const y = rows.indexOf(selected.row) * height;
    context.strokeStyle = "#111";
    context.lineWidth = 2 * dpr;
    context.strokeRect(x, y + dpr, Math.max(w, 2 * dpr), height - 2 * dpr);
  }
}
