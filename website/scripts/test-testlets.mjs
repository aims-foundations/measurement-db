import assert from "node:assert/strict";
import test from "node:test";
import "./load-typescript.mjs";
const { columnGroups, testletLabel } =
  await import("../lib/benchmark-records/testlets.ts");
const { matrixLayout, columnAt } =
  await import("../lib/benchmark-records/matrix-layout.ts");
const { prepareMatrix } = await import("../lib/benchmark-records/model.ts");
const { prepareContexts } = await import("../lib/benchmark-records/context.ts");
const { prepareView } = await import("../lib/benchmark-records/view.ts");
const { RecordSource } = await import("../lib/benchmark-records/client.ts");

const scale = {
  kind: "discrete",
  values: [0, 1],
  direction: "higher_is_better",
};
const records = {
  subjects: [{ subject_id: "s", display_name: "Model" }],
  itemIds: ["b", "a", "d", "c", "ungrouped"],
  testlets: [
    {
      testletId: "task-1",
      itemIds: ["b", "a"],
      totalItems: 2,
      continuedBefore: false,
      continuedAfter: false,
    },
    {
      testletId: "task-2",
      itemIds: ["d", "c"],
      totalItems: 6,
      continuedBefore: true,
      continuedAfter: true,
    },
  ],
  observations: ["b", "a", "d", "c", "ungrouped"].map((item_id, i) => ({
    response_id: `r${i}`,
    item_id,
    subject_id: "s",
    response: i % 2,
    trial: 1,
    test_condition: "",
    interactors: null,
  })),
};

test("sorting retains testlet adjacency and every individual binary outcome", () => {
  const model = prepareMatrix(records),
    contexts = prepareContexts(records.observations);
  const before = structuredClone(records);
  for (const order of ["grade", "name", "recorded"]) {
    const view = prepareView(model, contexts, {}, "", order, scale);
    assert.deepEqual(view.itemIds.slice(0, 2).sort(), ["a", "b"]);
    assert.deepEqual(view.itemIds.slice(2, 4).sort(), ["c", "d"]);
    assert.deepEqual(view.observations, records.observations);
    assert.equal(view.itemIds.at(-1), "ungrouped");
  }
  assert.deepEqual(
    prepareView(model, contexts, {}, "", "name", scale).itemIds,
    ["a", "b", "c", "d", "ungrouped"],
  );
  assert.deepEqual(records, before);
});

test("table subpages mark continuation without assigning independent items a testlet", () => {
  const groups = columnGroups(records, ["a", "d"]);
  assert.equal(groups[0].continuedBefore, true);
  assert.equal(groups[0].continuedAfter, false);
  assert.equal(groups[1].continuedBefore, true);
  assert.equal(groups[1].continuedAfter, true);
  assert.match(testletLabel(groups[1]), /continued/);
  const all = columnGroups(records);
  assert.equal(all.at(-1).testletId, null);
  assert.equal(all[0].continuedBefore, false);
  assert.equal(all[0].length, 2);
});

test("filtering keeps the surviving items in their source groups", () => {
  const input = structuredClone(records);
  input.observations.forEach((row, i) => {
    row.trial = (i % 2) + 1;
  });
  const view = prepareView(
    prepareMatrix(input),
    prepareContexts(input.observations),
    { trial: JSON.stringify("2") },
    "",
    "grade",
    scale,
  );
  assert.deepEqual(view.itemIds, ["a", "c"]);
  assert.deepEqual(
    columnGroups(view).map((g) => g.testletId),
    ["task-1", "task-2"],
  );
  assert.equal(view.observations.length, 2);
});

test("fit and scroll geometry select the same item IDs and gaps select nothing", () => {
  for (const actual of [true, false]) {
    const layout = matrixLayout(5, columnGroups(records), 400, actual);
    for (let i = 0; i < 5; i++) {
      assert.equal(
        columnAt(layout, layout.positions[i] + layout.cellWidth / 2),
        i,
      );
    }
    assert.equal(columnAt(layout, layout.positions[2] - 1), null);
    assert.equal(columnAt(layout, -1), null);
    assert.equal(columnAt(layout, layout.width + 1), null);
    if (!actual) assert.equal(layout.width, 400);
  }
  const old = matrixLayout(
    5,
    columnGroups({ itemIds: records.itemIds }),
    400,
    false,
  );
  assert.equal(old.grouped, false);
  assert.deepEqual(old.positions, [0, 80, 160, 240, 320]);
});

test("bounded window loading carries testlets and navigation uses only a small index", async () => {
  const calls = [];
  const index = {
    "task-1": { pages: [0, 0], items: 2 },
    "task-2": { pages: [0, 2], items: 6 },
  };
  const source = new RecordSource(
    "/records",
    {
      tables: { subjects: { pages: 1 } },
      windows: { testlets: "testlets.json" },
    },
    async (url) => {
      calls.push(url);
      return new Response(
        JSON.stringify(
          url.endsWith("subjects/0.json")
            ? { s: records.subjects[0] }
            : url.endsWith("testlets.json")
              ? index
              : records,
        ),
      );
    },
  );
  const signal = new AbortController().signal;
  assert.deepEqual(await source.testlets(signal), index);
  assert.deepEqual(calls, ["/records/windows/testlets.json"]);
  assert.deepEqual((await source.window(0, signal)).testlets, records.testlets);
  assert.deepEqual(
    await new RecordSource("/legacy", { tables: {} }).testlets(signal),
    {},
  );
});
