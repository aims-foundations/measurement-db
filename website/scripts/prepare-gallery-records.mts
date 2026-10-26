/** Install the Python export as static data. No rendering or data interpretation. */
import { cp, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { GalleryEntry, Manifest } from "../lib/benchmark-records/types.ts";

const source = process.argv[2];
if (!source)
  throw new Error(
    "Usage: node --experimental-strip-types scripts/prepare-gallery-records.mts <export-directory>",
  );
const website = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(website, "public/benchmark-records");
const stage = `${output}.staging-${process.pid}`;
const catalog = JSON.parse(
  await readFile(join(source, "index.json"), "utf8"),
) as { id: string; manifest: string }[];
const entries: Record<string, GalleryEntry> = {};

try {
  await mkdir(stage, { recursive: true });
  for (const entry of catalog) {
    if (
      !/^[a-z0-9][a-z0-9_-]*$/.test(entry.id) ||
      !/^\d+\/manifest\.json$/.test(entry.manifest)
    )
      throw new Error(`Invalid exported benchmark path: ${entry.id}`);
    if (Object.hasOwn(entries, entry.id))
      throw new Error(`Duplicate benchmark ${entry.id}`);
    const directory = join(source, dirname(entry.manifest));
    const manifest = JSON.parse(
      await readFile(join(directory, "manifest.json"), "utf8"),
    ) as Manifest;
    if (manifest.benchmark.benchmark_id !== entry.id)
      throw new Error(`Mismatched benchmark ${entry.id}`);
    await cp(directory, join(stage, entry.id), {
      recursive: true,
      dereference: true,
    });
    entries[entry.id] = { base: `/benchmark-records/${entry.id}`, manifest };
  }
  await rm(output, { recursive: true, force: true });
  await rename(stage, output);
  await writeFile(
    join(website, "content/generated/gallery-records.json"),
    JSON.stringify(entries, null, 2) + "\n",
  );
  console.log(
    `Prepared ${catalog.length} benchmark galleries: ${catalog.map((entry) => entry.id).join(", ")}`,
  );
} finally {
  await rm(stage, { recursive: true, force: true });
}
