import { itemScale } from "./scale";
import type {
  AssetLink,
  GalleryRecords,
  GradingCriterion,
  Item,
  Observation,
  ResponseScale,
  Subject,
} from "./types";

export type MatrixModel = GalleryRecords & {
  cells: Map<string, Map<string, Observation[]>>;
  graded: number;
};
export type CellSelection = { subject: Subject | null; itemId: string };

/** Index original attempts. Pairwise observations follow exactly the same path. */
export function prepareMatrix(records: GalleryRecords): MatrixModel {
  const subjects = new Set(
    records.subjects.map((subject) => subject.subject_id),
  );
  const items = new Set(records.itemIds);
  const ids = new Set<string>();
  const cells: MatrixModel["cells"] = new Map();
  let graded = 0;
  for (const record of records.observations) {
    if (!subjects.has(record.subject_id) || !items.has(record.item_id))
      throw new Error(
        `Observation ${record.response_id} refers to an unknown model or task.`,
      );
    if (ids.has(record.response_id))
      throw new Error(`Duplicate observation ${record.response_id}.`);
    ids.add(record.response_id);
    if (!cells.has(record.subject_id)) cells.set(record.subject_id, new Map());
    const row = cells.get(record.subject_id)!;
    if (!row.has(record.item_id)) row.set(record.item_id, []);
    row.get(record.item_id)!.push(record);
    if (record.response !== null) graded++;
  }
  return { ...records, cells, graded };
}

export function observationsFor(
  model: MatrixModel,
  selection: CellSelection,
): Observation[] {
  return (
    model.cells
      .get(selection.subject?.subject_id ?? "")
      ?.get(selection.itemId) ?? []
  );
}

export function findSubjects(subjects: Subject[], query: string): Subject[] {
  const search = query.trim().toLowerCase();
  return subjects.filter((subject) =>
    `${subject.display_name} ${subject.subject_id}`
      .toLowerCase()
      .includes(search),
  );
}

export function pageOf<T>(
  records: readonly T[],
  page: number,
  size: number,
): T[] {
  return records.slice(page * size, (page + 1) * size);
}

export function prepareItem(item: Item, scale: ResponseScale) {
  return {
    item,
    criterion: JSON.parse(item.grading_criterion) as GradingCriterion,
    attachments: JSON.parse(item.asset_manifest ?? "[]") as AssetLink[],
    scale: itemScale(scale, item),
  };
}
