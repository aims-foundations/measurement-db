// Shared runners intermittently time out when Vercel opens many connections.
// eslint-disable-next-line @typescript-eslint/no-require-imports -- Node's --require hook loads this CommonJS preload.
const { setTimeout: delay } = require("node:timers/promises");
const originalFetch = globalThis.fetch;
let pending = Promise.resolve();

globalThis.fetch = function (input, options) {
  const url = new URL(input instanceof Request ? input.url : input);
  if (url.hostname !== "api.vercel.com") return originalFetch(input, options);
  const task = pending.then(async () => {
    if (
      options?.body &&
      typeof options.body[Symbol.asyncIterator] === "function"
    ) {
      const chunks = [];
      for await (const chunk of options.body) chunks.push(Buffer.from(chunk));
      options = { ...options, body: Buffer.concat(chunks) };
    }
    for (let attempt = 0; ; attempt++) {
      try {
        return await originalFetch(input, options);
      } catch (error) {
        if (error.cause?.code !== "UND_ERR_CONNECT_TIMEOUT" || attempt === 11)
          throw error;
        const waitMs = Math.min(1000 * 2 ** attempt, 10_000);
        console.error(
          `Retrying Vercel connection timeout before sending the request (${attempt + 2}/12, in ${waitMs / 1000}s).`,
        );
        await delay(waitMs, undefined, { signal: options?.signal });
      }
    }
  });
  pending = task.then(
    () => {},
    () => {},
  );
  return task;
};
