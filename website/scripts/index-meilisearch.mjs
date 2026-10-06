import { ErrorStatusCode, Meilisearch, MeilisearchApiError } from "meilisearch";
import {
  BENCHMARKS_INDEX_SETTINGS,
  BENCHMARKS_INDEX_UID,
  RESULTS_INDEX_SETTINGS,
  RESULTS_INDEX_UID,
  buildSearchDocuments,
} from "./meilisearch-documents.mjs";

const DRY_RUN = process.argv.includes("--dry-run");
const TASK_TIMEOUT_MS = 10 * 60 * 1000;
const STAGING_SUFFIX = "_next";

function requiredEnvironment(name, fallbackName) {
  const value = process.env[name]?.trim() || process.env[fallbackName]?.trim();
  if (!value) {
    throw new Error(
      `Missing ${name}${fallbackName ? ` (or ${fallbackName})` : ""}`,
    );
  }
  return value;
}

function validateHost(host) {
  let url;
  try {
    url = new URL(host);
  } catch {
    throw new Error("MEILISEARCH_HOST must be a valid URL");
  }

  const local = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(local && url.protocol === "http:")) {
    throw new Error(
      "MEILISEARCH_HOST must use HTTPS (HTTP is allowed only for localhost)",
    );
  }
}

async function waitForTask(taskPromise, label) {
  const task = await taskPromise.waitTask({
    timeout: TASK_TIMEOUT_MS,
    interval: 1_000,
  });
  if (task.status !== "succeeded") {
    const detail = task.error?.message ? `: ${task.error.message}` : "";
    throw new Error(`${label} ${task.status}${detail}`);
  }
  return task;
}

function isMissingIndex(error) {
  return (
    error instanceof MeilisearchApiError &&
    error.cause?.code === ErrorStatusCode.INDEX_NOT_FOUND
  );
}

async function indexExists(client, uid) {
  try {
    await client.getRawIndex(uid);
    return true;
  } catch (error) {
    if (isMissingIndex(error)) return false;
    throw error;
  }
}

async function ensureIndex(client, uid, primaryKey) {
  if (await indexExists(client, uid)) return;
  await waitForTask(client.createIndex(uid, { primaryKey }), `Creating ${uid}`);
}

async function deleteIndexIfPresent(client, uid) {
  if (!(await indexExists(client, uid))) return;
  await waitForTask(client.deleteIndex(uid), `Deleting ${uid}`);
}

async function prepareIndex({
  client,
  liveUid,
  primaryKey,
  settings,
  documents,
}) {
  const stagingUid = `${liveUid}${STAGING_SUFFIX}`;
  await deleteIndexIfPresent(client, stagingUid);
  await waitForTask(
    client.createIndex(stagingUid, { primaryKey }),
    `Creating ${stagingUid}`,
  );

  const stagingIndex = client.index(stagingUid);
  await waitForTask(
    stagingIndex.updateSettings(settings),
    `Configuring ${stagingUid}`,
  );
  await waitForTask(
    stagingIndex.addDocuments(documents),
    `Adding documents to ${stagingUid}`,
  );

  const stats = await stagingIndex.getStats();
  if (stats.numberOfDocuments !== documents.length) {
    throw new Error(
      `${stagingUid} has ${stats.numberOfDocuments} documents; expected ${documents.length}`,
    );
  }

  const smoke = await stagingIndex.search("", { limit: 1 });
  if (documents.length > 0 && smoke.hits.length !== 1) {
    throw new Error(`${stagingUid} smoke search returned no documents`);
  }

  return stagingUid;
}

async function sync() {
  const { benchmarks, results } = buildSearchDocuments();
  console.log(
    `Prepared ${benchmarks.length} public benchmarks and ${results.length} aggregate model results.`,
  );

  if (DRY_RUN) {
    console.log("Validation complete; no Meilisearch data was changed.");
    return;
  }

  const host = requiredEnvironment(
    "MEILISEARCH_HOST",
    "NEXT_PUBLIC_MEILISEARCH_HOST",
  );
  const apiKey = requiredEnvironment("MEILISEARCH_ADMIN_KEY");
  validateHost(host);

  const client = new Meilisearch({
    host,
    apiKey,
    timeout: 15_000,
    defaultWaitOptions: { timeout: TASK_TIMEOUT_MS, interval: 1_000 },
  });

  await Promise.all([
    ensureIndex(client, BENCHMARKS_INDEX_UID, "slug"),
    ensureIndex(client, RESULTS_INDEX_UID, "id"),
  ]);

  const [benchmarksStagingUid, resultsStagingUid] = await Promise.all([
    prepareIndex({
      client,
      liveUid: BENCHMARKS_INDEX_UID,
      primaryKey: "slug",
      settings: BENCHMARKS_INDEX_SETTINGS,
      documents: benchmarks,
    }),
    prepareIndex({
      client,
      liveUid: RESULTS_INDEX_UID,
      primaryKey: "id",
      settings: RESULTS_INDEX_SETTINGS,
      documents: results,
    }),
  ]);

  // One request swaps both related indexes atomically, so users cannot see a
  // new benchmark corpus paired with stale result rows (or the reverse).
  await waitForTask(
    client.swapIndexes([
      { indexes: [BENCHMARKS_INDEX_UID, benchmarksStagingUid] },
      { indexes: [RESULTS_INDEX_UID, resultsStagingUid] },
    ]),
    "Publishing search indexes",
  );

  const [benchmarkStats, resultStats] = await Promise.all([
    client.index(BENCHMARKS_INDEX_UID).getStats(),
    client.index(RESULTS_INDEX_UID).getStats(),
  ]);
  if (
    benchmarkStats.numberOfDocuments !== benchmarks.length ||
    resultStats.numberOfDocuments !== results.length
  ) {
    throw new Error("Published Meilisearch document counts failed validation");
  }

  await Promise.all([
    deleteIndexIfPresent(client, benchmarksStagingUid),
    deleteIndexIfPresent(client, resultsStagingUid),
  ]);

  console.log(
    `Published ${benchmarks.length} benchmarks and ${results.length} results to Meilisearch.`,
  );
}

sync().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Meilisearch sync failed: ${message}`);
  process.exitCode = 1;
});
