import type { CoverageEdge, CoverageNode } from "./model";

export type MatrixOrder = "coverage" | "name" | "group";

/** Sparse indexing preserves every visible benchmark, including empty rows. */
export function buildCoverageMatrix(
  nodes: CoverageNode[],
  edges: CoverageEdge[],
  order: MatrixOrder,
) {
  const degree = new Map<string, number>();
  for (const edge of edges)
    for (const id of [edge.source, edge.target])
      degree.set(id, (degree.get(id) || 0) + 1);
  const compare = (a: CoverageNode, b: CoverageNode) => {
    if (order === "coverage") {
      const difference = (degree.get(b.id) || 0) - (degree.get(a.id) || 0);
      if (difference) return difference;
    }
    if (order === "group") {
      const group = (n: CoverageNode) =>
        n.kind === "subject"
          ? n.provider || ""
          : [...n.categories].sort()[0] || "";
      const difference = group(a).localeCompare(group(b));
      if (difference) return difference;
    }
    return a.label.localeCompare(b.label) || a.id.localeCompare(b.id);
  };
  const rows = nodes.filter((n) => n.kind === "benchmark").sort(compare);
  const columns = nodes.filter((n) => n.kind === "subject").sort(compare);
  const rowIndex = new Map(rows.map((n, i) => [n.id, i]));
  const columnIndex = new Map(columns.map((n, i) => [n.id, i]));
  const values = new Map<string, number>();
  const points: { row: number; column: number }[] = [];
  for (const edge of edges) {
    const row = rowIndex.get(edge.target),
      column = columnIndex.get(edge.source);
    if (row === undefined || column === undefined)
      throw new Error("Matrix edge is missing an endpoint");
    values.set(JSON.stringify([edge.source, edge.target]), edge.items);
    points.push({ row, column });
  }
  return { rows, columns, rowIndex, columnIndex, values, points };
}

export type CoverageMatrixData = ReturnType<typeof buildCoverageMatrix>;

export function matrixWindow(total: number, size: number, start: number) {
  const length = Math.min(Math.max(1, size), total);
  const offset = Math.max(0, Math.min(start, total - length));
  return { start: offset, end: offset + length };
}
