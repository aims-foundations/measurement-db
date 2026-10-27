import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import "./load-typescript.mjs";
import {
  readPreviewRecord,
  readPreviewPart,
  PagedRecordError,
} from "../lib/benchmark-records/record-preview.ts";
import {
  textPart,
  TEXT_PART_SIZE,
} from "../lib/benchmark-records/text-pages.ts";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
function fixture(envelope) {
  const buffers = [];
  let size = 0;
  function put(value) {
    const raw = Buffer.from(
      typeof value === "string" ? value : JSON.stringify(value),
    );
    const compressed = gzipSync(raw);
    const range = [0, size, compressed.length, raw.length, hash(compressed)];
    buffers.push(compressed);
    size += compressed.length;
    return range;
  }
  const fields =
    envelope.kind === "paged-record"
      ? { trace: [put("First 😺"), put("\nLast λ")] }
      : undefined;
  const record = put({ ...envelope, ...(fields ? { fields } : {}) });
  const index = put({ abc: record });
  const data = Buffer.concat(buffers);
  const preview = {
    source: "original",
    revision: "pinned",
    directory: "test",
    rows: 1,
    packs: [size],
    indexes: Array(256).fill(index),
  };
  const fetchRange = async (_, init) => {
    const [start, end] = init.headers.Range.slice(6).split("-").map(Number);
    return new Response(data.subarray(start, end + 1), {
      status: 206,
      headers: { "Content-Range": `bytes ${start}-${end}/${size}` },
    });
  };
  return { preview, fetchRange, record, data };
}

test("exact record values, missing IDs, and corruption checks", async (t) => {
  const record = {
    response_id: "abc",
    trace: "Unicode 😺\\n",
    grade: null,
    trial: 0,
  };
  const f = fixture({ kind: "record", record });
  t.mock.method(globalThis, "fetch", f.fetchRange);
  assert.deepEqual(
    await readPreviewRecord("/test", f.preview, "abc", "response_id"),
    record,
  );
  assert.equal(
    await readPreviewRecord("/test", f.preview, "absent", "response_id"),
    null,
  );
  f.data[f.record[1] + 12] ^= 1;
  await assert.rejects(
    readPreviewRecord("/test", f.preview, "abc", "response_id"),
    /integrity/,
  );
});

test("large text is paged without dropping Unicode or other fields", async (t) => {
  const f = fixture({
    kind: "paged-record",
    record: { response_id: "abc", metadata: "retained" },
  });
  t.mock.method(globalThis, "fetch", f.fetchRange);
  let paged;
  try {
    await readPreviewRecord("/test", f.preview, "abc", "response_id");
  } catch (error) {
    assert.ok(error instanceof PagedRecordError);
    paged = error.record;
  }
  assert.equal(paged.record.metadata, "retained");
  const parts = await Promise.all(
    paged.fields.trace.map((range) =>
      readPreviewPart(paged.base, paged.preview, range),
    ),
  );
  assert.equal(parts.join(""), "First 😺\nLast λ");
});

test("range and decompression sizes are enforced before large allocations", async (t) => {
  const f = fixture({ kind: "record", record: { response_id: "abc" } });
  const mock = t.mock.method(globalThis, "fetch", f.fetchRange);
  await assert.rejects(
    readPreviewPart("/test", f.preview, [0, 0, 1, 5 * 1024 ** 2, "x"]),
    /Invalid/,
  );
  assert.equal(mock.mock.callCount(), 0);
  mock.mock.mockImplementation(async () => new Response(f.data));
  await assert.rejects(
    readPreviewPart("/test", f.preview, f.record),
    /range \(200\)/,
  );
  mock.mock.mockImplementation(f.fetchRange);
  const smaller = [...f.record];
  smaller[3]--;
  await assert.rejects(readPreviewPart("/test", f.preview, smaller), /exceeds/);
  mock.mock.mockImplementation(async (_, init) => {
    const range = init.headers.Range.slice(6);
    return new Response(new Uint8Array(f.record[2] + 1), {
      status: 206,
      headers: { "Content-Range": `bytes ${range}/${f.data.length}` },
    });
  });
  await assert.rejects(
    readPreviewPart("/test", f.preview, f.record),
    /exceeds/,
  );
});

test("wrong record identities fail instead of displaying another answer", async (t) => {
  const f = fixture({ kind: "record", record: { response_id: "other" } });
  t.mock.method(globalThis, "fetch", f.fetchRange);
  await assert.rejects(
    readPreviewRecord("/test", f.preview, "abc", "response_id"),
    /ID does not match/,
  );
});

test("DOM text pagination preserves surrogate pairs at both boundaries", () => {
  const text =
    "a".repeat(TEXT_PART_SIZE - 1) +
    "😺" +
    "b".repeat(TEXT_PART_SIZE - 1) +
    "😺" +
    "tail";
  const parts = Array.from(
    { length: Math.ceil(text.length / TEXT_PART_SIZE) },
    (_, i) => textPart(text, i),
  );
  assert.equal(parts.join(""), text);
  for (const part of parts) assert.equal(part.isWellFormed(), true);
});

test("published preview manifests match canonical sources and bound every index range", async () => {
  const { readFile } = await import("node:fs/promises");
  const previews = JSON.parse(
    await readFile(
      new URL("../content/generated/record-previews.json", import.meta.url),
      "utf8",
    ),
  );
  const catalog = JSON.parse(
    await readFile(
      new URL("../content/generated/gallery-records.json", import.meta.url),
      "utf8",
    ),
  );
  for (const [slug, tables] of Object.entries(previews))
    for (const [table, preview] of Object.entries(tables)) {
      assert.match(preview.revision, /^[a-f0-9]{40}$/);
      assert.equal(
        preview.source,
        catalog[slug].manifest.tables[table].parquet.download,
      );
      assert.equal(
        preview.directory,
        `website-record-previews/${slug}/${table}`,
      );
      assert.equal(preview.indexes.length, 256);
      assert.ok(preview.rows > 0);
      for (const [pack, offset, bytes, expanded, hash] of preview.indexes) {
        assert.ok(pack >= 0 && pack < preview.packs.length);
        assert.ok(
          offset >= 0 && bytes > 0 && offset + bytes <= preview.packs[pack],
        );
        assert.ok(bytes <= 4 * 1024 ** 2 && expanded <= 4 * 1024 ** 2);
        assert.match(hash, /^[a-f0-9]{64}$/);
      }
    }
});
