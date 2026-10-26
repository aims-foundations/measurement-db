import type { ConcreteScale, ResponseScale } from "./types";

export const EMPTY = "#ffffff";
export const UNGRADED = "#9e9e9e";
export const MIXED = "#8064a2";
const PALETTE = [
  "#2278b5",
  "#d95f02",
  "#7570b3",
  "#1b9e77",
  "#e7298a",
  "#a6761d",
];

/** Color is presentation only. The original grade is used for labels and selection. */
export function gradeColor(value: number | null, scale: ResponseScale): string {
  if (value === null) return UNGRADED;
  if (scale.kind === "mixed") return MIXED;
  if (scale.kind === "discrete" && scale.direction === "unordered") {
    const index = scale.values.findIndex(
      (v) =>
        Math.abs(v - value) <= Number.EPSILON * 8 * Math.max(1, Math.abs(v)),
    );
    return index < 0 ? MIXED : PALETTE[index % PALETTE.length];
  }
  const [min, max] = bounds(scale);
  // An unbounded domain has no honest fixed endpoints for a color ramp.
  if (min === null || max === null || min === max) return "#2278b5";
  let t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  if (scale.direction === "lower_is_better") t = 1 - t;
  const low = [214, 39, 40],
    middle = [242, 242, 242],
    high = [31, 119, 180];
  const a = t < 0.5 ? low : middle,
    b = t < 0.5 ? middle : high;
  t = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(", ")})`;
}
export function bounds(scale: ConcreteScale): [number | null, number | null] {
  return scale.kind === "discrete"
    ? [scale.values[0], scale.values.at(-1)!]
    : [scale.min, scale.max];
}
