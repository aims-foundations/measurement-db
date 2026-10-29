import assert from "node:assert/strict";
import test from "node:test";
import "./load-typescript.mjs";
const { ScoreMatrix, pixelBins, fitRange, zoomRange, groupAt } =
  await import("../lib/aligned-matrix/model.ts");

function fixture(columns = 1027) {
  const values = Uint8Array.from(
    { length: columns * 2 },
    (_, i) => (i * 17 + Math.floor(i / 7)) % 3,
  );
  const manifest = {
    columns,
    subjects: [{ subject_id: "a" }, { subject_id: "b" }],
    passed: values.filter((v) => v === 2).length,
    failed: values.filter((v) => v === 1).length,
    observations: values.filter(Boolean).length,
    groups: [
      {
        id: "t1",
        start: 0,
        count: 513,
        setups: [{ grader: "old" }, { grader: "new" }],
        rowSetups: [0, 1],
      },
      {
        id: "t2",
        start: 513,
        count: columns - 513,
        setups: [{ grader: "new" }],
        rowSetups: [0, 0],
      },
    ],
  };
  return new ScoreMatrix(manifest, values);
}

test("block sums agree with every source cell across boundaries and grader filters", () => {
  const matrix = fixture();
  for (const row of [0, 1])
    for (const grader of ["", "old", "new"]) {
      for (let start = 0; start < 1027; start += 47)
        for (let end = start + 1; end <= 1027; end += 83) {
          const expected = { passed: 0, failed: 0, missing: 0 };
          for (let i = start; i < end; i++) {
            const value = matrix.value(row, i, grader);
            expected[
              value === 2 ? "passed" : value === 1 ? "failed" : "missing"
            ]++;
          }
          assert.deepEqual(
            matrix.counts(row, { start, end }, grader),
            expected,
          );
        }
    }
});

test("overview bins cover all tests once, including the final column", () => {
  const matrix = fixture();
  for (const width of [1, 13, 200, 1800]) {
    const range = { start: 27, end: 1027 };
    const bins = pixelBins(range, width);
    assert.equal(bins[0].start, range.start);
    assert.equal(bins.at(-1).end, range.end);
    const total = { passed: 0, failed: 0, missing: 0 };
    bins.forEach((bin, i) => {
      assert.ok(bin.end > bin.start);
      if (i) assert.equal(bin.start, bins[i - 1].end);
      const counts = matrix.counts(1, bin);
      for (const key of Object.keys(total)) total[key] += counts[key];
    });
    assert.deepEqual(total, matrix.counts(1, range));
  }
});

test("pan and anchored zoom stay inside data without losing the last test", () => {
  assert.deepEqual(fitRange(-40, 100, 1000), { start: 0, end: 100 });
  assert.deepEqual(fitRange(999, 100, 1000), { start: 900, end: 1000 });
  assert.deepEqual(zoomRange({ start: 100, end: 200 }, 0.1, 1000, 0), {
    start: 100,
    end: 110,
  });
  assert.deepEqual(zoomRange({ start: 100, end: 200 }, 50, 1000), {
    start: 0,
    end: 1000,
  });
  assert.equal(groupAt(fixture().manifest.groups, 513).id, "t2");
  assert.throws(() => groupAt(fixture().manifest.groups, 1027));
});

test("score dimensions and source totals must agree before rendering", () => {
  const matrix = fixture();
  assert.throws(
    () => new ScoreMatrix(matrix.manifest, matrix.values.slice(1)),
    /dimensions/,
  );
  assert.throws(
    () => new ScoreMatrix({ ...matrix.manifest, passed: 0 }, matrix.values),
    /source totals/,
  );
  const invalid = matrix.values.slice();
  invalid[0] = 3;
  assert.throws(() => new ScoreMatrix(matrix.manifest, invalid), /score code/);
});
