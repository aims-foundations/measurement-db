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
    revision: "immutable",
    directory: "website-trace-previews/example/source/exporter",
    packs: [1000],
  },
};
catalog.hidden = catalog.example;
const code = stripTypeScriptTypes(
  await fs.readFile(
    new URL("../app/api/trace-preview/[slug]/[file]/route.ts", import.meta.url),
    "utf8",
  ),
)
  .replace(
    'import previewsJson from "@/content/generated/trace-previews.json";',
    `const previewsJson=${JSON.stringify(catalog)};`,
  )
  .replace(
    'import { hiddenBenchmarkSet } from "@/content/curated/hidden-benchmarks";',
    'const hiddenBenchmarkSet=new Set(["hidden"]);',
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
const read = (slug = "example", file = "pack-0.bin", range = "bytes=10-19") =>
  GET(new Request("http://fixture.invalid", { headers: { range } }), {
    params: Promise.resolve({ slug, file }),
  });
for (const args of [
  ["absent"],
  ["__proto__"],
  ["hidden"],
  ["example", "../../secret"],
  ["example", "pack-1.bin"],
])
  assert.equal((await read(...args)).status, 404);
assert.equal((await read("example", "pack-0.bin", "bytes=0-1000")).status, 416);
assert.equal(requests.length, 0);
assert.equal((await read()).status, 307);
assert.equal(
  requests[0].url,
  "https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/immutable/website-trace-previews/example/source/exporter/pack-0.bin",
);
assert.equal(requests[0].init.method, "HEAD");
console.log(
  "Trace display route: immutable packs, hidden sources, allowlisted paths and exact ranges passed.",
);
