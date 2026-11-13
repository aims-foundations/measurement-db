import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import "./load-typescript.mjs";
const { RecordSource } = await import("../lib/benchmark-records/client.ts");
const { matchingTracePreview } =
  await import("../lib/benchmark-records/trace-preview.ts");

const directory = process.argv[2];
if (!directory) throw Error("Supply the Python-exported fixture directory");
const descriptor = JSON.parse(
  await fs.readFile(directory + "/descriptor.json", "utf8"),
);
const expected = JSON.parse(
  await fs.readFile(directory + "/expected.json", "utf8"),
);
const requests = [];
let corrupt = false;
const request = async (url, init) => {
  const file = /pack-(\d+)\.bin$/.exec(String(url));
  assert.ok(file, "Only display packs may load: " + url);
  const data = await fs.readFile(
    directory + "/preview/pack-" + file[1] + ".bin",
  );
  const range = new Headers(init?.headers).get("range");
  assert.match(range, /^bytes=\d+-\d+$/);
  const [, start, end] = range.match(/^bytes=(\d+)-(\d+)$/).map(Number);
  const body = new Uint8Array(data.subarray(start, end + 1));
  requests.push({
    url: String(url),
    bytes: body.length,
    expanded: gunzipSync(body).length,
  });
  if (corrupt) body[12] ^= 1;
  return new Response(body, {
    status: 206,
    headers: { "Content-Range": `bytes ${start}-${end}/${data.length}` },
  });
};
const manifest = {
  benchmark: { benchmark_id: "example" },
  tracePreview: descriptor,
  tables: {
    traces: {
      key: "response_id",
      parquet: {
        url: "/forbidden-original-source",
        download: descriptor.source,
        bytes: 999999999,
      },
    },
  },
};
const source = new RecordSource("/fixture", manifest, request);
const signal = () => new AbortController().signal;
const shown = await source.trace("wire-id", signal());
assert.equal(shown.kind, "paged");
const trace = shown.value;
assert.equal(
  requests.length,
  3,
  "First selection fetches only an ID index, header and first step page",
);
assert.ok(requests.reduce((n, r) => n + r.expanded, 0) < 1024 * 1024);
assert.ok(trace.firstPage.steps.length <= 32);
assert.equal(trace.header.summary.steps, expected.steps);
assert.equal(trace.header.summary.modelTurns, expected.modelTurns);
assert.equal(trace.header.summary.toolResults, expected.toolResults);
assert.equal(trace.header.trace.run.cost, 0);
assert.equal(trace.header.trace.native, undefined);
assert.ok(!JSON.stringify(trace).includes("Native-only-large-data"));

const last = await source.tracePage(
  trace,
  trace.header.pages.length - 1,
  signal(),
);
assert.equal(last.start + last.steps.length, expected.steps);
const field = last.fields.find((f) => f.step === 130 && f.path[0] === "output");
assert.ok(field);
let text = "";
for (let i = 0; i < field.parts.length; i++)
  text += await source.traceField(trace, field, i, signal());
assert.equal(
  text,
  expected.output,
  "Complete text reconstructs exactly, including Unicode and chunk boundaries",
);
const progress = [];
const found = await source.searchTrace(
  trace,
  "ΩΩΩneedle",
  signal(),
  (done, total) => progress.push([done, total]),
);
assert.deepEqual(found, { matches: [130], total: 1, complete: true });
assert.deepEqual(progress.at(-1), [
  trace.header.search.length,
  trace.header.search.length,
]);
const all = await source.searchTrace(trace, "Recorded answer", signal());
assert.equal(all.total, 130);

const cancelled = new AbortController();
cancelled.abort();
const before = requests.length;
await assert.rejects(source.trace("wire-id", cancelled.signal));
assert.equal(requests.length, before);
assert.equal(matchingTracePreview("changed-source", descriptor), undefined);
assert.equal(await source.trace("unknown-id", signal()), null);
corrupt = true;
await assert.rejects(source.trace("wire-id", signal()), /integrity/);
corrupt = false;
const wrongSource = new RecordSource(
  "/fixture",
  { ...manifest, tracePreview: { ...descriptor, page_bytes: 2 * 1024 * 1024 } },
  request,
);
const boundedBefore = requests.length;
await assert.rejects(wrongSource.trace("wire-id", signal()), /descriptor/);
assert.equal(requests.length, boundedBefore);
console.log(
  "Python/TypeScript trace transport: bounded first load, complete fields/search, cancellation, stale pins and corruption checks passed.",
);
