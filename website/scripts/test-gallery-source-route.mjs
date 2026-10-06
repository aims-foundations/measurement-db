import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";

const code = await fs.readFile(
  new URL("../app/api/gallery-source/[slug]/[table]/route.ts", import.meta.url),
  "utf8",
);
const catalog = {
  example: {
    revision: "pinned-commit",
    directory: "example/formatted_tables",
    tables: { items: { size: 1000 } },
  },
};
const js = stripTypeScriptTypes(code).replace(
  'import sources from "@/content/generated/gallery-sources.json";',
  `const sources = ${JSON.stringify(catalog)};`,
);
const { GET } = await import(
  `data:text/javascript;base64,${Buffer.from(js).toString("base64")}`
);
const calls = [];
const token = process.env.HF_TOKEN;
process.env.HF_TOKEN = "fixture-server-token";
const responses = [];
globalThis.fetch = async (url, options) => {
  assert.equal(new URL(url).origin, "https://huggingface.co");
  assert.equal(
    options.headers.get("authorization"),
    "Bearer fixture-server-token",
  );
  assert.equal(
    options.headers.get("range"),
    options.method === "HEAD" ? null : "bytes=10-19",
  );
  calls.push(url);
  const response = responses.shift();
  if (response instanceof Error) throw response;
  assert.ok(response, "Unexpected source request");
  return response;
};
async function read(slug = "example", table = "items", range = "bytes=10-19") {
  const headers = range == null ? {} : { Range: range };
  return GET(
    new Request("https://preview.invalid/api/gallery-source", { headers }),
    { params: Promise.resolve({ slug, table }) },
  );
}
try {
  assert.equal((await read("unknown")).status, 404);
  assert.equal((await read("__proto__")).status, 404);
  assert.equal((await read("example", "subjects")).status, 404);
  for (const range of [
    null,
    "bytes=0-1000",
    "bytes=20-10",
    "bytes=0-9,20-29",
    "bytes=-10",
  ])
    assert.equal((await read("example", "items", range)).status, 416);
  assert.equal(calls.length, 0);
  responses.push(
    new Response(null, {
      status: 302,
      headers: { Location: "/api/resolve-cache/fixture" },
    }),
  );
  responses.push(
    new Response(null, {
      status: 302,
      headers: {
        Location: "https://cdn.example.invalid/file?signature=fixture",
      },
    }),
  );
  const redirect = await read();
  assert.equal(redirect.status, 307);
  assert.equal(
    redirect.headers.get("location"),
    "https://cdn.example.invalid/file?signature=fixture",
  );
  assert.equal(redirect.headers.get("authorization"), null);
  assert.equal(
    redirect.headers.get("cache-control"),
    "public, max-age=60, s-maxage=60",
  );
  assert.equal(
    calls[0],
    "https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/pinned-commit/example/formatted_tables/items.parquet",
  );
  assert.equal(calls[1], "https://huggingface.co/api/resolve-cache/fixture");
  responses.push(new Response(null, { status: 200 }));
  responses.push(
    new Response("0123456789", {
      status: 206,
      headers: { "Content-Range": "bytes 10-19/1000" },
    }),
  );
  const direct = await read();
  assert.equal(direct.status, 206);
  assert.equal(await direct.text(), "0123456789");
  responses.push(new Response("restricted", { status: 401 }));
  assert.equal((await read()).status, 502);
  responses.push(
    new Response("rate limit", {
      status: 429,
      headers: { ratelimit: '"resolvers";r=0;t=120' },
    }),
  );
  const limited = await read();
  assert.equal(limited.status, 429);
  assert.equal(limited.headers.get("retry-after"), "120");
  responses.push(new Error("authenticated request details"));
  assert.equal(await (await read()).text(), "Source download unavailable.");
  console.log(
    "Source route allowlist, range validation, pinned revision, redirects, and credential isolation passed.",
  );
} finally {
  if (token === undefined) delete process.env.HF_TOKEN;
  else process.env.HF_TOKEN = token;
}
