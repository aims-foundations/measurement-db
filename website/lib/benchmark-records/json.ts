/** JSON transport, including independently compressed ranges in static packs. */
export type JsonTransport = { compressed?: boolean; pack?: { index: string } };
type PackIndex = {
  format: "canonical-record-pack-v1";
  packs: { file: string; bytes: number }[];
  entries: Record<string, [number, number, number, number, string]>;
};

export async function readJson<T>(
  base: string,
  path: string,
  transport: JsonTransport,
  signal?: AbortSignal,
  request: typeof fetch = (...args) => fetch(...args),
): Promise<T> {
  if (!transport.pack) {
    const response = await request(
      `${base}/${path}${transport.compressed ? ".gz" : ""}`,
      { signal },
    );
    if (!response.ok)
      throw new Error(`Could not load benchmark data (${response.status}).`);
    return response.json();
  }
  const descriptor = await request(`${base}/${transport.pack.index}`, {
    signal,
  });
  if (!descriptor.ok)
    throw new Error(`Could not load record index (${descriptor.status}).`);
  const index = (await descriptor.json()) as PackIndex;
  const location = index.entries[path];
  if (index.format !== "canonical-record-pack-v1" || !location)
    throw new Error(`Missing record page ${path}.`);
  const [number, offset, length, expanded, hash] = location;
  const pack = index.packs[number];
  if (
    !pack ||
    !/^pack-\d+\.bin$/.test(pack.file) ||
    offset < 0 ||
    length < 1 ||
    offset + length > pack.bytes
  )
    throw new Error("Invalid record page location.");
  const end = offset + length - 1;
  const response = await request(`${base}/${pack.file}`, {
    signal,
    headers: { Range: `bytes=${offset}-${end}` },
  });
  if (
    response.status !== 206 ||
    response.headers.get("Content-Range") !==
      `bytes ${offset}-${end}/${pack.bytes}`
  ) {
    await response.body?.cancel();
    throw new Error("The server did not return the requested record range.");
  }
  const bytes = await response.arrayBuffer();
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
  if (bytes.byteLength !== length || digest !== hash)
    throw new Error("Record page integrity check failed.");
  const raw = await new Response(
    new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")),
  ).arrayBuffer();
  if (raw.byteLength !== expanded)
    throw new Error("Record page size differs from its index.");
  signal?.throwIfAborted();
  return JSON.parse(new TextDecoder().decode(raw)) as T;
}
