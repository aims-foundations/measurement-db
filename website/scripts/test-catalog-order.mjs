import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import "./load-typescript.mjs";

const { agenticFirst, laterInCatalog } =
  await import("../content/curated/catalog-order.ts");
const { compareCatalogOrder } = await import("../lib/catalog-order.ts");
const records = JSON.parse(
  readFileSync(
    new URL("../content/generated/gallery-records.json", import.meta.url),
    "utf8",
  ),
);
const catalog = Object.entries(records).map(([slug, record]) => {
  const benchmark = record.manifest.benchmark;
  return {
    slug,
    name: benchmark.name,
    year: Number.parseInt(benchmark.release_date) || null,
  };
});

test("editorial entries refer to published benchmarks with no duplicates or overlaps", () => {
  const entries = [...agenticFirst, ...laterInCatalog];
  assert.equal(new Set(entries).size, entries.length);
  for (const slug of entries) assert.ok(records[slug], slug);
});

test("the complete catalog retains every benchmark and defers static QA", () => {
  const ordered = catalog.toSorted(compareCatalogOrder).map((b) => b.slug);
  assert.deepEqual(ordered.slice(0, 5), [
    "programbench",
    "tau2_bench",
    "theagentcompany",
    "appworld",
    "swebench_live",
  ]);
  assert.deepEqual(ordered.slice(0, agenticFirst.length), [...agenticFirst]);
  assert.deepEqual(
    new Set(ordered.slice(-laterInCatalog.length)),
    new Set(laterInCatalog),
  );
  assert.deepEqual(new Set(ordered), new Set(catalog.map((b) => b.slug)));
  assert.ok(ordered.indexOf("mmlu") >= ordered.length - laterInCatalog.length);
});

test("filtered catalogs keep the same relative order", () => {
  const subset = new Set(["mmlu", "programbench", "arcagi", "scigym"]);
  const keep = (entry) => subset.has(entry.slug);
  assert.deepEqual(
    catalog.filter(keep).toSorted(compareCatalogOrder),
    catalog.toSorted(compareCatalogOrder).filter(keep),
  );
});

test("unlisted benchmarks sort deterministically by year, name, then slug", () => {
  const entries = [
    { slug: "new-b", name: "A", year: 2026 },
    { slug: "undated", name: "A", year: null },
    { slug: "old", name: "A", year: 2025 },
    { slug: "new-z", name: "Z", year: 2026 },
    { slug: "new-a", name: "A", year: 2026 },
  ];
  assert.deepEqual(
    entries.toSorted(compareCatalogOrder).map((b) => b.slug),
    ["new-a", "new-b", "new-z", "old", "undated"],
  );
});
