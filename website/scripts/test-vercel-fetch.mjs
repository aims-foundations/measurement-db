import assert from "node:assert/strict";
import { Readable } from "node:stream";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const timers = require("node:timers/promises");
const delays = [];
timers.setTimeout = async (milliseconds, value, options) => {
  options?.signal?.throwIfAborted();
  delays.push(milliseconds);
  return value;
};

let calls = 0;
let active = 0;
let maximum = 0;
let failure;
let transientFailures = 1;
globalThis.fetch = async (input, options) => {
  calls++;
  active++;
  maximum = Math.max(active, maximum);
  await new Promise((resolve) => setTimeout(resolve, 5));
  active--;
  if (failure) throw failure;
  if (String(input).endsWith("/files")) {
    assert.equal(options.body.toString(), "payload");
    if (calls <= transientFailures)
      throw Object.assign(new TypeError("fetch failed"), {
        cause: { code: "UND_ERR_CONNECT_TIMEOUT" },
      });
  }
  return new Response("ok");
};
await import("./vercel-fetch.cjs");
const responses = await Promise.all([
  fetch("https://api.vercel.com/v2/files", {
    method: "POST",
    body: Readable.from(["pay", "load"]),
  }),
  fetch("https://api.vercel.com/v13/deployments", {
    method: "POST",
    body: "{}",
  }),
  fetch("https://api.vercel.com/v2/user"),
]);
assert.equal(maximum, 1);
assert.equal(calls, 4);
assert.deepEqual(delays, [1000]);
assert.deepEqual(await Promise.all(responses.map((r) => r.text())), [
  "ok",
  "ok",
  "ok",
]);

// A single upload can outlast the former four-attempt budget. Its body must
// remain replayable across every connection attempt.
calls = 0;
transientFailures = 5;
delays.length = 0;
assert.equal(
  await (
    await fetch("https://api.vercel.com/v2/files", {
      method: "POST",
      body: Readable.from(["pay", "load"]),
    })
  ).text(),
  "ok",
);
assert.equal(calls, 6);
assert.deepEqual(delays, [1000, 2000, 4000, 8000, 10000]);

for (const [code, attempts] of [
  ["UND_ERR_SOCKET", 1],
  ["UND_ERR_CONNECT_TIMEOUT", 12],
]) {
  calls = 0;
  delays.length = 0;
  failure = Object.assign(new TypeError("fetch failed"), { cause: { code } });
  await assert.rejects(
    fetch("https://api.vercel.com/v13/deployments"),
    failure,
  );
  assert.equal(calls, attempts);
  assert.equal(delays.length, attempts - 1);
  assert.ok(delays.every((milliseconds) => milliseconds <= 10000));
}
calls = 0;
const controller = new AbortController();
controller.abort(new Error("Deployment canceled"));
await assert.rejects(
  fetch("https://api.vercel.com/v2/user", { signal: controller.signal }),
  controller.signal.reason,
);
assert.equal(calls, 1);
calls = 0;
await assert.rejects(fetch("https://example.invalid/"), failure);
assert.equal(calls, 1);
failure = undefined;
assert.equal(
  await (await fetch("https://api.vercel.com/v2/user")).text(),
  "ok",
);
console.log(
  "Vercel request serialization, upload replay, and bounded retries passed.",
);
