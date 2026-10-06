import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";

const root = process.argv[2];
if (!root)
  throw new Error(
    "Pass the directory exported by test_gallery_hybrid_data.py.",
  );
const expected = JSON.parse(
  await fs.readFile(path.join(root, "expected.json"), "utf8"),
);
const requests = [];
globalThis.self = globalThis;
self.location = { origin: "http://fixture.invalid" };
globalThis.fetch = async (input, init) => {
  const url = new URL(input, self.location.origin);
  const source = url.pathname.match(
    /^\/measurement-db\/api\/gallery-source\/([^/]+)\/(\w+)$/,
  );
  if (source) {
    const range = new Headers(init?.headers).get("range");
    const match = range?.match(/^bytes=(\d+)-(\d+)$/);
    assert.ok(match, "Every HF read must be a byte-range request");
    const bytes = await fs.readFile(
      path.join(root, "tables", source[1], `${source[2]}.parquet`),
    );
    const start = Number(match[1]),
      end = Number(match[2]);
    assert.ok(start >= 0 && end < bytes.length);
    requests.push({ slug: source[1], table: source[2], size: end - start + 1 });
    return new Response(bytes.subarray(start, end + 1), { status: 206 });
  }
  assert.ok(url.pathname.startsWith("/measurement-db/benchmark-data/"));
  try {
    const bytes = await fs.readFile(
      path.join(
        root,
        "website/public",
        url.pathname.replace("/measurement-db/", ""),
      ),
    );
    return new Response(gunzipSync(bytes), { status: 200 });
  } catch (error) {
    if (error.code === "ENOENT") return new Response(null, { status: 404 });
    throw error;
  }
};
await import("../lib/gallery-source.worker.ts");
async function read(slug, kind, key) {
  let result;
  self.postMessage = (message) => {
    result = message;
  };
  await self.onmessage({
    data: { base: `/measurement-db/benchmark-data/${slug}`, kind, key },
  });
  assert.ok(result && !result.error, result?.error);
  return result.value;
}
for (const slug of ["binary", "normalized"]) {
  for (const [id, item] of Object.entries(expected[slug].items)) {
    const before = requests.length;
    const actual = await read(slug, "item", id);
    assert.deepEqual(actual, item);
    if (!expected.dynamic.includes(slug))
      assert.equal(requests.length, before, "Static prompts must not read HF");
    for (const [, src] of actual.content.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
      const image = await read(slug, "image", src);
      assert.equal(image.type, "image/png");
      assert.equal(
        Buffer.from(image.data).toString("base64"),
        expected[slug].image,
      );
    }
  }
  for (const { key, value } of expected[slug].answers)
    assert.deepEqual(await read(slug, "answer", key), value);
  assert.deepEqual(await read(slug, "answer", ["missing", "item", null, 1]), {
    trace: null,
  });
}
assert.ok(requests.some((r) => r.table === "assets"));
assert.ok(requests.some((r) => r.table === "traces"));
console.log(
  `Static and dynamic items, embedded and asset images, and exact traces passed (${requests.length} source ranges).`,
);
