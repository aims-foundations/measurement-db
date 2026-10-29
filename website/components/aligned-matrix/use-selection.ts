import { useCallback } from "react";
import { groupAt, type ScoreMatrix } from "@/lib/aligned-matrix/model";
import type { TestSource } from "@/lib/aligned-matrix/client";
import type { RecordSource } from "@/lib/benchmark-records/client";
import { prepareMatrix } from "@/lib/benchmark-records/model";
import type { Observation } from "@/lib/benchmark-records/types";
import { useRequest } from "./use-request";

/** Resolve one logical cell back to its original item and source observation. */
export function useSelection(
  matrix: ScoreMatrix,
  tests: TestSource,
  source: RecordSource,
  row: number,
  column: number,
) {
  const load = useCallback(
    async (signal: AbortSignal) => {
      const subject = matrix.manifest.subjects[row];
      const group = groupAt(matrix.manifest.groups, column);
      const test = await tests.test(column, signal);
      const itemId = test.items[group.rowSetups[row]];
      if (!itemId || !matrix.value(row, column))
        return { test, model: null, itemId: null };
      const pages = await source.windowForItem(itemId, signal);
      if (!pages || pages[1] - pages[0] > 8)
        throw new Error(
          "Could not locate a bounded source record for this cell.",
        );
      const observations: Observation[] = [];
      for (let page = pages[0]; page <= pages[1]; page++) {
        const records = await source.window(page, signal);
        observations.push(
          ...records.observations.filter(
            (r) => r.item_id === itemId && r.subject_id === subject.subject_id,
          ),
        );
      }
      if (
        observations.length !== 1 ||
        observations[0].response !== matrix.value(row, column) - 1
      )
        throw new Error(
          "The source observation does not match the displayed cell.",
        );
      return {
        test,
        itemId,
        model: prepareMatrix({
          subjects: [subject],
          itemIds: [itemId],
          observations,
        }),
      };
    },
    [matrix, tests, source, row, column],
  );
  return useRequest(load);
}
