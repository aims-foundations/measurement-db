/** Fetch a gzipped JSON asset and parse it. Decompresses client-side with
 *  DecompressionStream; if the host already decoded it (Content-Encoding), falls
 *  back to the raw text. Returns null on any failure (missing file, bad JSON).
 *
 *  Shared by the matrix viewer (trace shards, lazy click data) and the joined
 *  matrix, which are separate components but read the same kind of asset. */
export async function fetchGzJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    let text: string;
    try {
      const stream = new Blob([buf])
        .stream()
        .pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(stream).text();
    } catch {
      text = new TextDecoder().decode(buf);
    }
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}
