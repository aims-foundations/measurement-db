/** Resolve one pinned HF file; credentials remain on the server. */
export async function serveHuggingFaceRange(
  request: Request,
  url: string,
  size: number,
) {
  const range = request.headers.get("range");
  const match = range?.match(/^bytes=(\d+)-(\d+)$/);
  if (!match || Number(match[1]) > Number(match[2]) || Number(match[2]) >= size)
    return new Response("A valid byte range is required.", { status: 416 });
  const headers = new Headers();
  if (process.env.HF_TOKEN)
    headers.set("Authorization", `Bearer ${process.env.HF_TOKEN}`);
  try {
    for (let hop = 0; hop < 3; hop++) {
      let response = await fetch(url, {
        headers,
        method: "HEAD",
        redirect: "manual",
        cache: "no-store",
        signal: AbortSignal.timeout(30000),
      });
      const location = response.headers.get("location");
      if (location && [301, 302, 303, 307, 308].includes(response.status)) {
        await response.body?.cancel();
        const target = new URL(location, url);
        if (target.origin === "https://huggingface.co") {
          url = target.href;
          continue;
        }
        return new Response(null, {
          status: 307,
          headers: {
            Location: target.href,
            "Cache-Control": "public, max-age=60, s-maxage=60",
          },
        });
      }
      if (response.status === 200) {
        headers.set("Range", range!);
        response = await fetch(url, {
          headers,
          redirect: "manual",
          cache: "no-store",
          signal: AbortSignal.timeout(30000),
        });
      }
      if (response.status === 206)
        return new Response(response.body, {
          status: 206,
          headers: {
            "Content-Type": "application/octet-stream",
            "Cache-Control": "no-store",
            "Content-Range": response.headers.get("content-range")!,
          },
        });
      await response.body?.cancel();
      if (response.status === 429)
        return new Response(
          "Hugging Face is temporarily rate-limiting downloads.",
          {
            status: 429,
            headers: {
              "Retry-After":
                response.headers.get("retry-after") ??
                response.headers.get("ratelimit")?.match(/;t=(\d+)/)?.[1] ??
                "60",
            },
          },
        );
      console.warn("Gallery source request failed", response.status);
      break;
    }
  } catch {
    /* Report the source failure without exposing authenticated request details. */
  }
  return new Response("Source download unavailable.", { status: 502 });
}
