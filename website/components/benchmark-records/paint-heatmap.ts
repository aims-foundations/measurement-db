import type { MatrixView } from "@/lib/benchmark-records/view";
import type { ResponseScale, Subject } from "@/lib/benchmark-records/types";
import { gradeColor } from "@/lib/benchmark-records/colors";

export const ROW_HEIGHT = 32;
const TILE_WIDTH = 2048;

/** Small native-resolution tiles keep colors exact without allocating a full dense image. */
export function paintHeatmap(
  ctx: CanvasRenderingContext2D,
  view: MatrixView,
  subjects: Subject[],
  scale: ResponseScale,
  width: number,
  cellWidth: number,
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
  const start = Math.max(0, Math.floor(offset / cellWidth));
  const end = Math.min(
    view.itemIds.length,
    Math.ceil((offset + width) / cellWidth),
  );
  const tile = document.createElement("canvas");
  tile.width = TILE_WIDTH;
  tile.height = ROW_HEIGHT;
  const pixels = tile.getContext("2d")!;
  subjects.forEach((subject, row) => {
    const cells = view.cells.get(subject.subject_id);
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
        base * cellWidth - offset,
        row * ROW_HEIGHT,
        count * cellWidth,
        ROW_HEIGHT,
      );
    }
    ctx.strokeStyle = "#e9e8e4";
    ctx.beginPath();
    ctx.moveTo(0, (row + 1) * ROW_HEIGHT - 0.5);
    ctx.lineTo(width, (row + 1) * ROW_HEIGHT - 0.5);
    ctx.stroke();
  });
}
