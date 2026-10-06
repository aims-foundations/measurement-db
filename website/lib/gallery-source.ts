export function readGallerySource<T>(
  base: string,
  kind: string,
  key: unknown,
  signal: AbortSignal,
): Promise<T> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const worker = new Worker(
      new URL("./gallery-source.worker.ts", import.meta.url),
    );
    const stop = () => {
      worker.terminate();
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
    };
    const abort = () => {
      stop();
      reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
    const timeout = setTimeout(() => {
      stop();
      reject(new Error("The source download took too long. Please try again."));
    }, 120000);
    worker.onmessage = ({
      data,
    }: MessageEvent<{ value: T; error?: string }>) => {
      stop();
      if (data.error) reject(new Error(data.error));
      else resolve(data.value);
    };
    worker.onerror = (event) => {
      stop();
      reject(new Error(event.message || "Could not read the source table."));
    };
    worker.postMessage({ base, kind, key });
  });
}
