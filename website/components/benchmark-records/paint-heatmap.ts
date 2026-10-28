import type { MatrixView } from "@/lib/benchmark-records/view";
import type { ResponseScale, Subject } from "@/lib/benchmark-records/types";
import { gradeColor } from "@/lib/benchmark-records/colors";
import type { MatrixLayout } from "@/lib/benchmark-records/matrix-layout";

export const ROW_HEIGHT = 32;
const TILE_WIDTH = 2048;

/** Small native-resolution tiles keep colors exact without allocating a full dense image. */
export function paintHeatmap(
  ctx: CanvasRenderingContext2D,
  view: MatrixView,
  subjects: Subject[],
  scale: ResponseScale,
  width: number,
  layout: MatrixLayout,
  offset: number,
) {
  const height = subjects.length * ROW_HEIGHT,
    ratio = window.devicePixelRatio || 1;
  ctx.canvas.width = Math.round(width * ratio);
  ctx.canvas.height = Math.round(height * ratio);
  ctx.scale(ratio, ratio);
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  const { cellWidth } = layout;
  if (!cellWidth) return;
  const tile = document.createElement("canvas");
  tile.width = TILE_WIDTH;
  tile.height = ROW_HEIGHT;
  const pixels = tile.getContext("2d")!;
  subjects.forEach((subject, row) => {
    const cells = view.cells.get(subject.subject_id);
    for (const group of layout.groups) {
      const start =
        group.start +
        Math.max(0, Math.floor((offset - group.left) / cellWidth));
      const end = Math.min(
        group.start + group.length,
        group.start + Math.ceil((offset + width - group.left) / cellWidth),
      );
      for (let base = start; base < end; base += TILE_WIDTH) {
        const count = Math.min(TILE_WIDTH, end - base);
        pixels.fillStyle = "#fff";
        pixels.fillRect(0, 0, count, ROW_HEIGHT);
        for (let x = 0; x < count; x++) {
          const attempts = cells?.get(view.itemIds[base + x]) ?? [];
          const band = (ROW_HEIGHT - 4) / Math.max(1, attempts.length);
          attempts.forEach((record, i) => {
            pixels.fillStyle = gradeColor(record.response, scale);
            pixels.fillRect(x, 2 + i * band, 1, band);
          });
        }
        ctx.drawImage(
          tile,
          0,
          0,
          count,
          ROW_HEIGHT,
          layout.positions[base] - offset,
          row * ROW_HEIGHT,
          count * cellWidth,
          ROW_HEIGHT,
        );
      }
    }
    ctx.strokeStyle = "#e9e8e4";
    ctx.beginPath();
    ctx.moveTo(0, (row + 1) * ROW_HEIGHT - 0.5);
    ctx.lineTo(width, (row + 1) * ROW_HEIGHT - 0.5);
    ctx.stroke();
  });
  if (layout.grouped) {
    ctx.strokeStyle = "#9c9a94";
    for (const group of layout.groups) {
      for (const edge of [group.left - 0.5, group.left + group.width + 0.5]) {
        ctx.beginPath();
        ctx.moveTo(edge - offset, 0);
        ctx.lineTo(edge - offset, height);
        ctx.stroke();
      }
    }
  }
}
