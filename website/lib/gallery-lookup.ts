export type GallerySource = {
  dynamicItems: boolean;
  tables: Record<string, { size: number; columns: string[] }>;
  lookupPack?: { index: string };
};

type PackIndex = {
  format: "gallery-lookup-pack-v1";
  families: Record<
    string,
    {
      bits: number;
      chunks: Record<string, [number, number, number, string]>;
    }
  >;
  packs: { file: string; bytes: number; sha256: string }[];
};

// Only cache the small descriptors, never expanded lookup dictionaries.
const descriptors = new Map<string, Promise<unknown>>();

async function json<T>(url: string, signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted();
  const cached = descriptors.get(url);
  if (cached) return (await cached) as T;
  const pending = (async () => {
    const response = await fetch(url, { signal });
    if (!response.ok)
      throw new Error(`Could not load gallery index (${response.status}).`);
    return response.json();
  })();
  descriptors.set(url, pending);
  try {
    return (await pending) as T;
  } catch (error) {
    descriptors.delete(url);
    throw error;
  }
}

export function gallerySource(base: string, signal?: AbortSignal) {
  return json<GallerySource>(`${base}/source.json.gz`, signal);
}

function hex(bytes: ArrayBuffer) {
  return Array.from(new Uint8Array(bytes), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}

export async function readGalleryLookup<T>(
  base: string,
  kind: string,
  key: unknown,
  signal?: AbortSignal,
): Promise<T | undefined> {
  signal?.throwIfAborted();
  const encoded = JSON.stringify(key);
  const hash = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(encoded)),
  );
  const source = await gallerySource(base, signal);
  if (!source.lookupPack) {
    const bucket = hash[0].toString(16).padStart(2, "0");
    const response = await fetch(`${base}/${kind}/${bucket}.json.gz`, {
      signal,
    });
    if (response.status === 404) return undefined;
    if (!response.ok)
      throw new Error(`Could not load ${kind} data (${response.status}).`);
    return ((await response.json()) as Record<string, T>)[encoded];
  }
  const indexPath = source.lookupPack.index;
  if (!/^lookup-pack\/index\.[a-f0-9]{64}\.json$/.test(indexPath))
    throw new Error("Invalid gallery pack index path.");
  const index = await json<PackIndex>(`${base}/${indexPath}`, signal);
  if (index.format !== "gallery-lookup-pack-v1")
    throw new Error("Unsupported gallery pack format.");
  const family = index.families[kind];
  if (!family) return undefined;
  if (!Number.isInteger(family.bits) || family.bits < 0 || family.bits > 8)
    throw new Error("Invalid gallery bucket count.");
  const bucket = (hash[0] >> (8 - family.bits)).toString(16).padStart(2, "0");
  const chunk = family.chunks[bucket];
  if (!chunk) return undefined;
  const [packId, offset, length, sha256] = chunk;
  const pack = index.packs[packId];
  if (
    !pack ||
    !/^[a-f0-9]{64}\.bin$/.test(pack.file) ||
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(length) ||
    offset < 0 ||
    length <= 0 ||
    !Number.isSafeInteger(pack.bytes) ||
    offset + length > pack.bytes
  )
    throw new Error("Invalid gallery chunk location.");
  const end = offset + length - 1;
  const response = await fetch(`${base}/lookup-pack/${pack.file}`, {
    signal,
    headers: { Range: `bytes=${offset}-${end}` },
  });
  const encoding = response.headers.get("Content-Encoding");
  if (
    response.status !== 206 ||
    (encoding && encoding !== "identity") ||
    response.headers.get("Content-Range") !==
      `bytes ${offset}-${end}/${pack.bytes}`
  ) {
    await response.body?.cancel();
    throw new Error(
      `Could not read the requested gallery chunk (${response.status}).`,
    );
  }
  const compressed = await response.arrayBuffer();
  if (
    compressed.byteLength !== length ||
    hex(await crypto.subtle.digest("SHA-256", compressed)) !== sha256
  )
    throw new Error("Gallery chunk integrity check failed.");
  signal?.throwIfAborted();
  const entries = (await new Response(
    new Blob([compressed])
      .stream()
      .pipeThrough(new DecompressionStream("gzip")),
  ).json()) as Record<string, T>;
  signal?.throwIfAborted();
  return entries[encoded];
}
