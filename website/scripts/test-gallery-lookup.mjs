import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { readGalleryLookup } from "../lib/gallery-lookup.ts";

const digest = (value) => createHash("sha256").update(value).digest("hex");
const key = ["subject", "Unicode 雪", null, 0, "condition"];
const encoded = JSON.stringify(key);
const value = { ids: ["exact-response"], score: 0 };
const chunk = gzipSync(JSON.stringify({ [encoded]: value }), { mtime: 0 });
const pack = Buffer.concat([
  Buffer.from("prefix!"),
  chunk,
  Buffer.from("suffix!"),
]);
const packHash = digest(pack);
const index = {
  format: "gallery-lookup-pack-v1",
  families: {
    keys: { bits: 0, chunks: { "00": [0, 7, chunk.length, digest(chunk)] } },
  },
  packs: [{ file: `${packHash}.bin`, bytes: pack.length, sha256: packHash }],
};
const indexText = JSON.stringify(index);
const indexPath = `lookup-pack/index.${digest(indexText)}.json`;
let ranges = 0;
globalThis.fetch = async (url, options) => {
  if (url.endsWith("source.json.gz"))
    return Response.json(
      url.startsWith("/legacy") ? {} : { lookupPack: { index: indexPath } },
    );
  if (url.startsWith("/legacy")) return Response.json({ [encoded]: value });
  if (url.endsWith(".json")) return new Response(indexText);
  assert.equal(options.headers.Range, `bytes=7-${7 + chunk.length - 1}`);
  ranges++;
  if (url.startsWith("/ignored-range")) return new Response(pack);
  const headers = {
    "Content-Range": `bytes 7-${7 + chunk.length - 1}/${pack.length}`,
  };
  if (url.startsWith("/wrong-offset"))
    headers["Content-Range"] = `bytes 0-${chunk.length - 1}/${pack.length}`;
  if (url.startsWith("/outer-gzip")) headers["Content-Encoding"] = "gzip";
  const body = Buffer.from(chunk);
  if (url.startsWith("/corrupt")) body[body.length - 1] ^= 1;
  return new Response(body, { status: 206, headers });
};
assert.deepEqual(await readGalleryLookup("/valid", "keys", key), value);
assert.deepEqual(await readGalleryLookup("/legacy", "keys", key), value);
const before = ranges;
assert.equal(
  await readGalleryLookup("/valid", "absent-family", key),
  undefined,
);
assert.equal(ranges, before);
assert.equal(
  await readGalleryLookup("/valid", "keys", "missing-key"),
  undefined,
);
for (const mode of ["ignored-range", "wrong-offset", "outer-gzip", "corrupt"])
  await assert.rejects(readGalleryLookup(`/${mode}`, "keys", key), /chunk/i);
await assert.rejects(
  readGalleryLookup("/valid", "keys", key, AbortSignal.abort()),
);
console.log(
  "Packed and legacy lookups, exact Unicode/null keys, missing entries, abort, range and integrity failures passed.",
);
