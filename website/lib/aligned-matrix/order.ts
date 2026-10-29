import { ScoreMatrix } from "./model";

/** Match the gallery's difficulty order: lower observed pass rate first.
 * Groups stay intact; missing observations never count as failed tests.
 * Ties keep the native test order and columns without observations go last.
 */
export function orderByDifficulty(
  matrix: ScoreMatrix,
  rows: number[],
  grader = "",
): ScoreMatrix {
  const { columns, subjects, groups } = matrix.manifest;
  const passed = new Uint32Array(columns),
    observed = new Uint32Array(columns);
  const order = new Uint32Array(columns);
  for (const group of groups) {
    const end = group.start + group.count;
    for (const row of rows) {
      if (grader && group.setups[group.rowSetups[row]]?.grader !== grader)
        continue;
      const offset = row * columns;
      for (let column = group.start; column < end; column++) {
        const score = matrix.values[offset + column];
        if (score) observed[column]++;
        if (score === 2) passed[column]++;
      }
    }
    const within = Uint32Array.from(
      { length: group.count },
      (_, i) => group.start + i,
    );
    within.sort((a, b) => {
      if (!observed[a] || !observed[b])
        return Number(!observed[a]) - Number(!observed[b]) || a - b;
      return passed[a] * observed[b] - passed[b] * observed[a] || a - b;
    });
    order.set(within, group.start);
  }
  const values = new Uint8Array(matrix.values.length);
  for (let row = 0; row < subjects.length; row++) {
    const offset = row * columns;
    for (let column = 0; column < columns; column++)
      values[offset + column] = matrix.values[offset + order[column]];
  }
  const sourceColumns = order.map((column) => matrix.sourceColumn(column));
  return new ScoreMatrix(matrix.manifest, values, sourceColumns);
}
