import { findSubjects, prepareMatrix, type MatrixModel } from "./model";
import { matchesFilters, type Filters, type prepareContexts } from "./context";
import type { ResponseScale } from "./types";

export type Order = "grade" | "name" | "recorded";
export const comparable = (scale: ResponseScale) =>
  scale.kind !== "mixed" &&
  (scale.direction === "higher_is_better" ||
    scale.direction === "lower_is_better");

/** All summaries are derived from the selected observations, never from display pixels. */
export function prepareView(
  model: MatrixModel,
  contexts: ReturnType<typeof prepareContexts>,
  filters: Filters,
  query: string,
  order: Order,
  scale: ResponseScale,
) {
  const filtered = Object.values(filters).some(Boolean);
  const subjects = findSubjects(model.subjects, query);
  const visibleSubjects = new Set(
    subjects.map((subject) => subject.subject_id),
  );
  const observations = model.observations.filter(
    (record) =>
      visibleSubjects.has(record.subject_id) &&
      matchesFilters(contexts.fields.get(record.response_id)!, filters),
  );
  const observedItems = new Set(observations.map((record) => record.item_id));
  const itemIds = filtered
    ? model.itemIds.filter((id) => observedItems.has(id))
    : [...model.itemIds];
  const selected = prepareMatrix({
    subjects: model.subjects,
    itemIds,
    observations,
  });
  const rowTotals = new Map<string, [number, number]>(),
    columnTotals = new Map<string, [number, number]>();
  for (const record of observations) {
    if (record.response === null || !visibleSubjects.has(record.subject_id))
      continue;
    for (const [map, key] of [
      [rowTotals, record.subject_id],
      [columnTotals, record.item_id],
    ] as const) {
      const [sum, n] = map.get(key) ?? [0, 0];
      map.set(key, [sum + record.response, n + 1]);
    }
  }
  const mean = (map: Map<string, [number, number]>) =>
    new Map([...map].map(([id, [sum, n]]) => [id, sum / n]));
  const rowMeans = comparable(scale)
    ? mean(rowTotals)
    : new Map<string, number>();
  const columnMeans = comparable(scale)
    ? mean(columnTotals)
    : new Map<string, number>();
  if (order === "grade" && comparable(scale)) {
    const direction =
      scale.kind !== "mixed" && scale.direction === "lower_is_better" ? -1 : 1;
    const compare = (map: Map<string, number>, a: string, b: string) =>
      !map.has(a)
        ? Number(map.has(b))
        : !map.has(b)
          ? -1
          : direction * (map.get(a)! - map.get(b)!);
    subjects.sort((a, b) => compare(rowMeans, a.subject_id, b.subject_id));
    itemIds.sort((a, b) => compare(columnMeans, a, b));
  } else if (order === "name") {
    subjects.sort((a, b) => a.display_name.localeCompare(b.display_name));
    itemIds.sort((a, b) => a.localeCompare(b));
  }
  return { ...selected, subjects, itemIds, rowMeans, columnMeans };
}
export type MatrixView = ReturnType<typeof prepareView>;
