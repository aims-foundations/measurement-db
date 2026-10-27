import assert from "node:assert/strict";
import "./load-typescript.mjs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

const root = process.argv[2];
if (!root)
  throw new Error("Pass the directory exported by tests/test_gallery_new.py.");
const expected = JSON.parse(
  await readFile(join(root, "expected-traces.json"), "utf8"),
);
const requests = [];
globalThis.self = globalThis;
globalThis.fetch = async (input, init) => {
  const url = new URL(input);
  const bytes = await readFile(join(root, url.pathname));
  if (url.pathname.endsWith(".json")) return new Response(bytes);
  if (url.pathname.endsWith(".gz")) return new Response(gunzipSync(bytes));
  const range = new Headers(init?.headers).get("range");
  assert.match(range, /^bytes=\d+-\d+$/);
  const [start, end] = range.slice(6).split("-").map(Number);
  assert.ok(start >= 0 && end < bytes.length);
  assert.ok(
    end - start + 1 < bytes.length,
    "A cell must not read the whole trace table",
  );
  requests.push(end - start + 1);
  return new Response(bytes.subarray(start, end + 1), {
    status: 206,
    headers: {
      "Content-Range": `bytes ${start}-${end}/${bytes.length}`,
    },
  });
};
await import("../lib/benchmark-records/parquet.worker.ts");
for (const [number, rows] of Object.entries(expected)) {
  const manifest = JSON.parse(
    await readFile(join(root, "lazy", number, "manifest.json"), "utf8"),
  );
  const parquet = manifest.tables.traces.parquet;
  for (const row of rows) {
    let result;
    self.postMessage = (value) => {
      result = value;
    };
    await self.onmessage({
      data: {
        base: `http://fixture.invalid/lazy/${number}/traces`,
        parquet,
        id: row.response_id,
      },
    });
    assert.ok(result && !result.error, result?.error);
    assert.deepEqual(
      result.value,
      row,
      "Every original trace field must survive a range read",
    );
  }
  let missing;
  self.postMessage = (value) => {
    missing = value;
  };
  await self.onmessage({
    data: {
      base: `http://fixture.invalid/lazy/${number}/traces`,
      parquet,
      id: "absent",
    },
  });
  assert.deepEqual(missing, { value: null });
}
console.log(
  `Exact canonical traces, reversed row order, null grades, trials, and absent IDs passed (${requests.length} byte ranges).`,
);

const records = JSON.parse(
  await readFile(join(root, "expected-records.json"), "utf8"),
);
const { readJson } = await import("../lib/benchmark-records/json.ts");
for (const [number, tables] of Object.entries(records)) {
  const manifest = JSON.parse(
    await readFile(join(root, "packed", number, "manifest.json"), "utf8"),
  );
  for (const [table, rows] of Object.entries(tables)) {
    const key = manifest.tables[table].key;
    for (const row of rows) {
      let result;
      self.postMessage = (value) => {
        result = value;
      };
      await self.onmessage({
        data: {
          base: `http://fixture.invalid/packed/${number}`,
          table,
          key,
          parquet: manifest.tables[table].parquet,
          pack: manifest.pack,
          id: row[key],
        },
      });
      assert.ok(result && !result.error, result?.error);
      const expected =
        table === "assets" ? { ...row, data: new Uint8Array(row.data) } : row;
      assert.deepEqual(
        result.value,
        expected,
        `Exact ${table} row must survive packed source lookup`,
      );
    }
  }
  let total = 0;
  for (let page = 0; page < manifest.windows.pages; page++) {
    const window = await readJson(
      `http://fixture.invalid/packed/${number}`,
      `windows/${page}.json`,
      manifest,
    );
    total += window.observations.length;
    assert.ok(window.observations.length <= manifest.windows.maxRows);
  }
  assert.equal(total, manifest.tables.responses?.rows ?? 0);
}
console.log(
  "Packed windows, exact item/trace/attachment rows, and all source bytes passed.",
);
