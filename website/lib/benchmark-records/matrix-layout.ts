import type { ColumnGroup } from "./testlets";

/** One geometry for drawing, labels, pointer selection and keyboard scrolling. */
export function matrixLayout(
  columns: number,
  groups: ColumnGroup[],
  width: number,
  actual: boolean,
) {
  const grouped = groups.some((group) => group.testletId !== null);
  const boundaries = grouped
    ? new Set(groups.slice(1).map((group) => group.start))
    : new Set<number>();
  const gap = actual
    ? 8
    : Math.min(8, width / (2 * Math.max(1, boundaries.size)));
  const cellWidth = actual
    ? 8
    : Math.max(0, width - gap * boundaries.size) / Math.max(1, columns);
  const positions: number[] = [];
  let x = 0;
  for (let i = 0; i < columns; i++) {
    if (boundaries.has(i)) x += gap;
    positions.push(x);
    x += cellWidth;
  }
  return {
    positions,
    cellWidth,
    width: Math.max(width, x),
    grouped,
    groups: groups.map((group) => ({
      ...group,
      left: positions[group.start],
      width: cellWidth * group.length,
    })),
  };
}
export type MatrixLayout = ReturnType<typeof matrixLayout>;

export function columnAt(layout: MatrixLayout, x: number): number | null {
  const column = layout.positions.findIndex(
    (left) => x >= left && x < left + layout.cellWidth,
  );
  return column < 0 ? null : column;
}
