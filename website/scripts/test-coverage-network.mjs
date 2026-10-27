import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

const code = stripTypeScriptTypes(
  await readFile(
    new URL("../lib/coverage-network/model.ts", import.meta.url),
    "utf8",
  ),
  { mode: "transform" },
);
const { buildCoverageNetwork, filterCoverageNetwork } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`
);
const benchmarks = [
  { slug: "shared", name: "Shared", categories: ["coding"] },
  { slug: "unmapped", name: "Unmapped", categories: ["safety"] },
];
const subjects = [
  { name: "shared", provider: "Provider A", benchmarks: ["shared"] },
  { name: "second", provider: "Provider B", benchmarks: ["shared"] },
  { name: "hidden only", provider: null, benchmarks: ["hidden"] },
];
const marginals = [
  { slug: "shared", model: "shared", items: 10 }, // No grade required.
  { slug: "shared", model: "second", items: 20, sum: 5, n: 25 },
  { slug: "hidden", model: "hidden only", items: 30 },
];

test("edges use observed items, preserve canonical identities, and exclude hidden benchmarks", () => {
  const data = buildCoverageNetwork(benchmarks, subjects, marginals);
  assert.equal(data.nodes.length, 4);
  assert.equal(data.edges.length, 2);
  assert.ok(data.nodes.some((n) => n.id === "subject:shared"));
  assert.ok(data.nodes.some((n) => n.id === "benchmark:shared"));
  assert.equal(data.nodes.find((n) => n.id === "benchmark:unmapped").degree, 0);
  assert.equal(
    data.nodes.reduce((total, n) => total + n.degree, 0),
    4,
  );
  assert.equal(
    buildCoverageNetwork(benchmarks, subjects, [{ ...marginals[0], items: 0 }])
      .edges.length,
    0,
  );
});

test("inconsistent snapshots fail rather than fabricate or double-count edges", () => {
  assert.throws(
    () =>
      buildCoverageNetwork(benchmarks, subjects, [marginals[0], marginals[0]]),
    /Duplicate coverage edge/,
  );
  assert.throws(
    () => buildCoverageNetwork(benchmarks, [], marginals),
    /Unmapped coverage edge/,
  );
  assert.throws(
    () =>
      buildCoverageNetwork(benchmarks, subjects, [
        { ...marginals[0], items: NaN },
      ]),
    /Invalid coverage item count/,
  );
});

test("domain and provider filters preserve endpoints and handle unmapped benchmarks explicitly", () => {
  const data = buildCoverageNetwork(benchmarks, subjects, marginals);
  const filters = { domain: "", provider: "", showUnmapped: true };
  assert.equal(filterCoverageNetwork(data, filters).nodes.size, 4);
  const provider = filterCoverageNetwork(data, {
    ...filters,
    provider: "Provider A",
  });
  assert.deepEqual([...provider.nodes].sort(), [
    "benchmark:shared",
    "subject:shared",
  ]);
  assert.equal(provider.edges.length, 1);
  const safety = filterCoverageNetwork(data, { ...filters, domain: "safety" });
  assert.deepEqual([...safety.nodes], ["benchmark:unmapped"]);
  assert.equal(safety.edges.length, 0);
  assert.equal(
    filterCoverageNetwork(data, { ...filters, showUnmapped: false }).nodes.size,
    3,
  );
});

test("the real catalog snapshot has valid bipartite edges and retains unlinked benchmarks", async (t) => {
  const json = async (file) =>
    JSON.parse(
      await readFile(
        new URL(`../content/${file}.json`, import.meta.url),
        "utf8",
      ),
    );
  const [cards, hidden, subjects, summary] = await Promise.all([
    json("generated/benchmark-cards"),
    json("curated/hidden-benchmarks"),
    json("generated/model-timeline"),
    json("generated/chart-marginals"),
  ]);
  const visible = cards.filter((c) => !hidden.includes(c.slug));
  const data = buildCoverageNetwork(visible, subjects, summary.marginals);
  assert.equal(
    data.nodes.filter((n) => n.kind === "benchmark").length,
    visible.length,
  );
  assert.equal(
    data.nodes.reduce((sum, n) => sum + n.degree, 0),
    data.edges.length * 2,
  );
  const ids = new Set(data.nodes.map((n) => n.id));
  for (const edge of data.edges) {
    assert.ok(ids.has(edge.source) && ids.has(edge.target));
    assert.ok(
      edge.source.startsWith("subject:") &&
        edge.target.startsWith("benchmark:"),
    );
  }
  t.diagnostic(
    `${visible.length} benchmarks; ${data.nodes.length - visible.length} subjects; ${data.edges.length} links; ${data.nodes.filter((n) => n.kind === "benchmark" && !n.degree).length} unmapped benchmarks`,
  );
});
