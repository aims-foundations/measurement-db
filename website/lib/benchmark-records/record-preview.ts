/** Independently compressed source records. No rendering or answer interpretation. */
export type PreviewRange = [number, number, number, number, string];
export type RecordPreview = {
  source: string;
  revision: string;
  directory: string;
  rows: number;
  packs: number[];
  indexes: PreviewRange[];
};
export type PagedRecord = {
  record: Record<string, unknown>;
  fields: Record<string, PreviewRange[]>;
  base: string;
  preview: RecordPreview;
};
export class PagedRecordError extends Error {
  readonly record: PagedRecord;
  constructor(record: PagedRecord) {
    super("This record is available in smaller parts.");
    this.record = record;
  }
}

const MAX_PAYLOAD = 4 * 1024 * 1024;

/** Reject an oversized or ignored range before consuming its body. */
async function readRange(
  url: string,
  start: number,
  length: number,
  total: number,
  expanded: number,
  hash: string,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (
    ![start, length, total, expanded].every(Number.isSafeInteger) ||
    start < 0 ||
    length < 1 ||
    expanded < 0 ||
    start + length > total ||
    length > MAX_PAYLOAD ||
    expanded > MAX_PAYLOAD
  )
    throw new Error("Invalid source record range.");
  const response = await fetch(url, {
    signal,
    headers: { Range: `bytes=${start}-${start + length - 1}` },
  });
  if (
    response.status !== 206 ||
    response.headers.get("Content-Range") !==
      `bytes ${start}-${start + length - 1}/${total}`
  ) {
    await response.body?.cancel();
    throw new Error(
      `Could not load the requested record range (${response.status}).`,
    );
  }
  const compressed = await readBounded(response.body!, length, signal);
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", compressed)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
  if (compressed.byteLength !== length || digest !== hash)
    throw new Error("Source record integrity check failed.");
  return readBounded(
    new Blob([compressed])
      .stream()
      .pipeThrough(new DecompressionStream("gzip")),
    expanded,
    signal,
  );
}

async function readBounded(
  stream: ReadableStream<Uint8Array>,
  length: number,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  const reader = stream.getReader();
  const output = new Uint8Array(length);
  let offset = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > length)
        throw new Error("Source record exceeds its declared size.");
      output.set(value, offset);
      offset += value.length;
    }
  } finally {
    await reader.cancel();
  }
  if (offset !== length)
    throw new Error("Source record size differs from its index.");
  return output;
}

export async function readPreviewPart(
  base: string,
  preview: RecordPreview,
  range: PreviewRange,
  signal?: AbortSignal,
): Promise<string> {
  const [pack, offset, bytes, expanded, hash] = range;
  if (!Number.isSafeInteger(pack) || pack < 0 || !preview.packs[pack])
    throw new Error("Unknown source record pack.");
  return new TextDecoder("utf-8", { fatal: true }).decode(
    await readRange(
      `${base}/pack-${pack}.bin`,
      offset,
      bytes,
      preview.packs[pack],
      expanded,
      hash,
      signal,
    ),
  );
}

export async function readPreviewRecord(
  base: string,
  preview: RecordPreview,
  id: string,
  key: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown> | null> {
  const bucket = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(id)),
  )[0];
  const index = JSON.parse(
    await readPreviewPart(base, preview, preview.indexes[bucket], signal),
  ) as Record<string, PreviewRange>;
  if (!Object.hasOwn(index, id)) return null;
  const value = JSON.parse(
    await readPreviewPart(base, preview, index[id], signal),
  );
  if (value.record?.[key] !== id)
    throw new Error("Source record ID does not match the selection.");
  if (value.kind === "paged-record")
    throw new PagedRecordError({
      record: value.record,
      fields: value.fields,
      base,
      preview,
    });
  if (value.kind !== "record") throw new Error("Unknown source record format.");
  return value.record;
}
