// Preview experiment only: record counts and byte sizes, never headers or bodies.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs = require("node:fs");
const originalFetch = globalThis.fetch;
globalThis.fetch = async function (input, options) {
  const url = new URL(input instanceof Request ? input.url : input);
  if (url.hostname !== "api.vercel.com") return originalFetch(input, options);
  let body = options?.body;
  if (
    body &&
    !Buffer.isBuffer(body) &&
    typeof body[Symbol.asyncIterator] === "function"
  ) {
    const pieces = [];
    for await (const piece of body) pieces.push(Buffer.from(piece));
    body = Buffer.concat(pieces);
    options = { ...options, body };
  }
  const row = {
    path: url.pathname,
    method: options?.method || "GET",
    bodyBytes:
      typeof body === "string"
        ? Buffer.byteLength(body)
        : body?.byteLength || 0,
    started: new Date().toISOString(),
  };
  const began = performance.now();
  try {
    const response = await originalFetch(input, options);
    row.status = response.status;
    if (row.method === "POST" && url.pathname.endsWith("/deployments")) {
      const result = await response
        .clone()
        .json()
        .catch(() => ({}));
      row.missingFiles =
        result.error?.missing?.length ?? result.error?.missing_files?.length;
      row.errorCode = result.error?.code;
      row.deploymentId = result.id;
    }
    return response;
  } catch (error) {
    row.errorCode = error.cause?.code || error.code || error.name;
    throw error;
  } finally {
    row.seconds = (performance.now() - began) / 1000;
    if (process.env.VERCEL_UPLOAD_METRICS)
      fs.appendFileSync(
        process.env.VERCEL_UPLOAD_METRICS,
        JSON.stringify(row) + "\n",
      );
  }
};
