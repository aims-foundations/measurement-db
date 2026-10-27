/** A worker keeps Parquet decompression off the rendering thread. */
import { withBase } from "../base-path";
import type { JsonTransport } from "./json";
import {
  PagedRecordError,
  type PagedRecord,
  type RecordPreview,
} from "./record-preview";
export function readParquetRecord<T>(
  base: string,
  parquet: { url: string; bytes: number },
  id: string,
  signal: AbortSignal,
  options?: JsonTransport & {
    table: string;
    key: string;
    slug?: string;
    preview?: RecordPreview;
  },
): Promise<T | null> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./parquet.worker.ts", import.meta.url));
    const stop = () => {
      worker.terminate();
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
    };
    const abort = () => {
      stop();
      reject(signal.reason);
    };
    const timeout = setTimeout(() => {
      stop();
      reject(
        new Error("The source download timed out. Select the cell to retry."),
      );
    }, 120000);
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = ({
      data,
    }: MessageEvent<{
      value: T | null;
      error?: string;
      paged?: PagedRecord;
    }>) => {
      stop();
      if (data.paged) reject(new PagedRecordError(data.paged));
      else if (data.error) reject(new Error(data.error));
      else resolve(data.value);
    };
    worker.onerror = (event) => {
      stop();
      reject(new Error(event.message));
    };
    const url = parquet.url.startsWith("/")
      ? withBase(parquet.url)
      : parquet.url;
    worker.postMessage({
      base: new URL(base, location.href).href,
      parquet: { ...parquet, url: new URL(url, location.href).href },
      id,
      ...options,
      previewBase:
        options?.preview &&
        new URL(
          withBase(`/api/record-preview/${options.slug}/${options.table}`),
          location.href,
        ).href,
    });
  });
}
