import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
const helper = stripTypeScriptTypes(
  await fs.readFile(
    new URL("../lib/server/huggingface-range.ts", import.meta.url),
    "utf8",
  ),
);
const helperUrl = `data:text/javascript;base64,${Buffer.from(helper).toString("base64")}`;
const catalog = {
  example: {
    traces: {
      revision: "immutable",
      directory: "website-record-previews/example/traces",
      packs: [1000],
    },
  },
};
const code = stripTypeScriptTypes(
  await fs.readFile(
    new URL(
      "../app/api/record-preview/[slug]/[table]/[file]/route.ts",
      import.meta.url,
    ),
    "utf8",
  ),
)
  .replace(
    'import previewsJson from "@/content/generated/record-previews.json";',
    `const previewsJson=${JSON.stringify(catalog)};`,
  )
  .replace(
    'import { hiddenBenchmarkSet } from "@/content/curated/hidden-benchmarks";',
    "const hiddenBenchmarkSet = new Set();",
  )
  .replace('"@/lib/server/huggingface-range"', JSON.stringify(helperUrl));
const { GET } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`
);
const requests = [];
globalThis.fetch = async (url, init) => {
  requests.push({ url, init });
  return new Response(null, {
    status: 302,
    headers: { Location: "https://cdn.invalid/fixture" },
  });
};
const read = (
  slug = "example",
  table = "traces",
  file = "pack-0.bin",
  range = "bytes=10-19",
) =>
  GET(new Request("http://fixture.invalid/", { headers: { range } }), {
    params: Promise.resolve({ slug, table, file }),
  });
for (const args of [
  ["absent"],
  ["__proto__"],
  ["example", "__proto__"],
  ["example", "items"],
  ["example", "traces", "../../secret"],
  ["example", "traces", "pack-1.bin"],
])
  assert.equal((await read(...args)).status, 404);
assert.equal(
  (await read("example", "traces", "pack-0.bin", "bytes=0-1000")).status,
  416,
);
assert.equal(requests.length, 0);
const response = await read();
assert.equal(response.status, 307);
assert.equal(
  requests[0].url,
  "https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/immutable/website-record-previews/example/traces/pack-0.bin",
);
assert.equal(requests[0].init.method, "HEAD");
console.log(
  "Preview route allowlist, pinned artifact paths and bounded byte ranges passed.",
);
