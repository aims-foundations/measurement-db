import "./load-typescript.mjs";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
const { prepareItemContent } = await import("../lib/item-content.ts");
const { presentations } =
  await import("../lib/presentation-transforms/index.ts");

const directory = process.argv[2];
if (!directory)
  throw new Error(
    "Pass a directory containing inventory.json and <benchmark>.json item samples.",
  );
const inventory = JSON.parse(
  await fs.readFile(path.join(directory, "inventory.json"), "utf8"),
);
const report = [];
for (const entry of inventory) {
  assert.ok(
    presentations[entry.slug]?.item,
    `Missing registration: ${entry.slug}`,
  );
  const samples = JSON.parse(
    await fs.readFile(path.join(directory, `${entry.slug}.json`), "utf8"),
  );
  let formatted = 0;
  for (const sample of samples) {
    const before = JSON.stringify(sample);
    if (sample.content !== null) {
      const blocks = prepareItemContent(sample.content, entry.slug);
      assert.ok(
        blocks?.length,
        `${entry.slug}/${sample.item_id}: transformation failed`,
      );
      assert.doesNotThrow(() => JSON.stringify(blocks));
      formatted++;
    }
    assert.equal(
      JSON.stringify(sample),
      before,
      "Presentation must not alter a source record",
    );
  }
  report.push({
    benchmark: entry.slug,
    sampled: samples.length,
    formatted,
    nullContent: samples.length - formatted,
  });
}
await fs.writeFile(
  path.join(directory, "presentation-results.json"),
  JSON.stringify(report, null, 2),
);
console.log(
  `Presentation checks passed for ${report.length} benchmarks and ${report.reduce((n, r) => n + r.sampled, 0)} sampled items. Original records unchanged.`,
);
