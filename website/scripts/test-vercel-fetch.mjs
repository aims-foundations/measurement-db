import assert from "node:assert/strict";
import { Readable } from "node:stream";

let calls = 0;
let active = 0;
let maximum = 0;
let failure;
globalThis.fetch = async (input, options) => {
  calls++;
  active++;
  maximum = Math.max(active, maximum);
  await new Promise((resolve) => setTimeout(resolve, 5));
  active--;
  if (failure) throw failure;
  if (String(input).endsWith("/files")) {
    assert.equal(options.body.toString(), "payload");
    if (calls === 1)
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
assert.deepEqual(await Promise.all(responses.map((r) => r.text())), [
  "ok",
  "ok",
  "ok",
]);

for (const [code, attempts] of [
  ["UND_ERR_SOCKET", 1],
  ["UND_ERR_CONNECT_TIMEOUT", 4],
]) {
  calls = 0;
  failure = Object.assign(new TypeError("fetch failed"), { cause: { code } });
  await assert.rejects(
    fetch("https://api.vercel.com/v13/deployments"),
    failure,
  );
  assert.equal(calls, attempts);
}
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
