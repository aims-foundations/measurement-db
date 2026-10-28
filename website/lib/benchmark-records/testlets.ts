import type { GalleryRecords, TestletWindow } from "./types";

/** Preserve source group order while allowing item ordering within each group. */
export function groupItemOrder(
  recorded: string[],
  ordered: string[],
  testlets: TestletWindow[] = [],
): string[] {
  if (!testlets.length) return ordered;
  const membership = new Map(
    testlets.flatMap((group) =>
      group.itemIds.map((id) => [id, group.testletId] as const),
    ),
  );
  const ranks = new Map<string, number>();
  const key = (id: string) =>
    membership.has(id) ? `testlet:${membership.get(id)}` : `item:${id}`;
  for (const id of recorded)
    if (!ranks.has(key(id))) ranks.set(key(id), ranks.size);
  return [...ordered].sort((a, b) => ranks.get(key(a))! - ranks.get(key(b))!);
}

export type ColumnGroup = {
  testletId: string | null;
  start: number;
  length: number;
  totalItems: number;
  continuedBefore: boolean;
  continuedAfter: boolean;
};

/** Describe the current view or table subpage without joining unrelated items. */
export function columnGroups(
  records: Pick<GalleryRecords, "itemIds" | "testlets">,
  visible = records.itemIds,
): ColumnGroup[] {
  const membership = new Map(
    (records.testlets ?? []).flatMap((group) =>
      group.itemIds.map((id) => [id, group] as const),
    ),
  );
  const groups: ColumnGroup[] = [];
  const selected = new Set(visible);
  for (const [start, id] of visible.entries()) {
    const source = membership.get(id);
    const testletId = source?.testletId ?? null;
    const previous = groups.at(-1);
    if (previous && previous.testletId === testletId) {
      previous.length++;
      continue;
    }
    const full = records.itemIds.filter(
      (item) => membership.get(item)?.testletId === testletId,
    );
    groups.push({
      testletId,
      start,
      length: 1,
      totalItems: source?.totalItems ?? 0,
      continuedBefore: Boolean(
        source?.continuedBefore || (source && !selected.has(full[0])),
      ),
      continuedAfter: Boolean(
        source?.continuedAfter || (source && !selected.has(full.at(-1)!)),
      ),
    });
  }
  return groups;
}

export function testletLabel(group: ColumnGroup): string {
  if (group.testletId === null) return "Items";
  return `Testlet: ${group.testletId}${group.continuedBefore ? " (continued)" : ""}`;
}
