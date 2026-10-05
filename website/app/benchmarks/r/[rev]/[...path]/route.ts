// Streams the derived gallery assets (trace shards, click data, matrix PNGs,
// item images, card images, institution logos) from the HuggingFace dataset
// repo, attaching
// HF_TOKEN server-side so the repo's gating stays enforced and the credential
// never reaches the browser. See lib/gallery-assets.ts for why the revision is
// baked into the URL path.
//
// SECURITY: the path validation below is load-bearing, not hygiene. This
// handler holds a token with read access to a private repo, so an unvalidated
// path would turn it into an open proxy that anyone could use to pull whatever
// that token can reach. Keep the allowlist tight and fail closed.
import { GALLERY_REPO } from "@/lib/gallery-assets";
import { hiddenBenchmarkSet } from "@/content/curated/hidden-benchmarks";

const HF_DATASETS = "https://huggingface.co/datasets";

/** Only these top-level directories are reachable. */
const ALLOWED_ROOTS = new Set([
  "traces",
  "data",
  "matrices",
  "item-images",
  // Institution logos. They used to be static files under public/institutions/;
  // they now publish with the rest of the gallery, so this handler serves them.
  "institutions",
  // tau_voice call recordings, one Opus clip per response cell. Served to an
  // <audio> element, so unlike the other roots these are ranged-fetched by the
  // browser — see the Range pass-through below.
  "audio",
  // Per-cell payloads for the joined matrix (one character per response),
  // fetched by components/joined-matrix.tsx. This handler fails closed, so a
  // root missing from this list 400s and the joined matrix silently renders
  // nothing — the page still works, which is why the omission is easy to miss.
  "joined",
]);

/** Extensions permitted for the top-level <slug> card images. */
const CARD_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp"];

// Matrix filenames are condition values run through generate_benchmark_gallery.sanitize(),
// which only replaces ';', '=' and '/'. EVERYTHING else in the source data
// survives verbatim — spaces, ':', '&', '|', ',', '(', ')', '@', and accented
// or combining Unicode. 2,503 of the 14,692 matrix files contain at least one
// such character (e.g. "dataset_dot_type_text recognition en_trial_4.png",
// "source_leaderboard_aggregate_pass@1.png"), so a character allowlist here
// silently 400s a sixth of the gallery.
//
// Rejecting traversal is what actually matters, and that needs only three
// things: no path separators inside a segment, no "." / ".." segments, and no
// control characters. Everything else is made inert by encodeURIComponent when
// the upstream URL is built. The real access control is the ALLOWED_ROOTS
// allowlist and the .parquet check below, not the shape of the filename.
// eslint-disable-next-line no-control-regex
const UNSAFE_SEGMENT_RE = /[/\\\x00-\x1f\x7f]/;

/** A published commit SHA, or the un-pinned bootstrap value. */
const REV_RE = /^(?:main|[0-9a-f]{7,40})$/;

const CONTENT_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  // .json.gz is served as opaque bytes: the client gunzips it itself with
  // DecompressionStream (components/matrix-viewer.tsx:158). Labelling it
  // Content-Encoding: gzip would make the browser decompress it first and
  // break that path, so we never forward the upstream encoding header.
  gz: "application/octet-stream",
  opus: "audio/ogg",
};

/**
 * Returns the validated repo-relative path, or null to reject. Fails closed:
 * anything not positively recognised is refused.
 */
function safePath(rev: string, segments: string[]): string | null {
  if (!REV_RE.test(rev)) return null;
  if (segments.length === 0) return null;

  for (const seg of segments) {
    if (seg === "" || seg === "." || seg === "..") return null;
    if (UNSAFE_SEGMENT_RE.test(seg)) return null;
  }

  const last = segments[segments.length - 1];
  // Belt and braces: the gallery repo holds no parquets, but if it ever did,
  // this handler must not be the way they get out.
  if (last.endsWith(".parquet")) return null;

  if (segments.length === 1) {
    // Top level holds only the <slug> card images. Mostly .png, but 8 of them
    // are .jpg/.jpeg (ai2d_test, csedb, cybench, jailbreakbench,
    // medhelm_med_qa, mmbench_v11, soar_arc, swe_rebench) — restricting this
    // to .png silently 404s those cards.
    if (!CARD_EXTENSIONS.some((ext) => last.endsWith(ext))) return null;
  } else if (!ALLOWED_ROOTS.has(segments[0])) {
    return null;
  }

  return segments.join("/");
}

/**
 * True when the (already-validated) path belongs to a benchmark on the hidden
 * list. Every asset is slug-keyed: top-level card images are "<slug>.png",
 * the per-slug roots hold either a "<slug>/" directory (traces, item-images,
 * audio, some matrices) or a "<slug>.<ext>" file (data, joined, the rest of
 * matrices). Slugs never contain ".", so the stem before the first dot IS the
 * slug. "institutions" holds shared logos, not per-benchmark files, and is
 * deliberately not checked.
 */
function isHidden(segments: string[]): boolean {
  if (segments.length > 1 && segments[0] === "institutions") return false;
  const keyed = segments.length === 1 ? segments[0] : segments[1];
  return hiddenBenchmarkSet.has(keyed.split(".")[0]);
}

function contentTypeFor(path: string): string {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  return CONTENT_TYPES[ext] ?? "application/octet-stream";
}

export async function GET(
  req: Request,
  ctx: { params: Promise<{ rev: string; path: string[] }> },
) {
  const { rev, path } = await ctx.params;
  const rel = safePath(rev, path ?? []);
  if (!rel) {
    return new Response("Bad request", { status: 400 });
  }

  // Hidden benchmarks 404 like any missing file — same status and body as the
  // upstream-404 branch below, so a hidden slug is indistinguishable from one
  // that never existed. Checked at every revision, including "main".
  if (isHidden(path)) {
    return new Response("Not found", { status: 404 });
  }

  // LOCAL PREVIEW ONLY. The revision in the URL pins the asset set to a
  // PUBLISHED commit, so a file generate_benchmark_gallery.py just wrote is unreachable until
  // it is published — and because the page drops the PNG matrix whenever
  // `hasJoined` is set, an unpublished joined payload renders an empty section
  // rather than an obvious error. Setting GALLERY_LOCAL_DIR serves the same
  // paths off disk instead, so a rebuild can be inspected before it ships.
  //
  // Opt-in by env var rather than NODE_ENV: `next build && next start` runs in
  // production mode, which is exactly the mode a preview needs. Unset (every
  // deployment) this branch does not exist. `rel` has already been through
  // safePath, so it carries no traversal segments.
  const localDir = process.env.GALLERY_LOCAL_DIR;
  if (localDir) {
    const { readFile } = await import("node:fs/promises");
    const { join } = await import("node:path");
    try {
      const bytes = await readFile(join(localDir, rel));
      return new Response(new Uint8Array(bytes), {
        headers: {
          "Content-Type": contentTypeFor(rel),
          "Cache-Control": "no-store",
          "X-Content-Type-Options": "nosniff",
        },
      });
    } catch {
      // Fall through to the proxy: only some of the gallery may be on disk.
    }
  }

  const token = process.env.HF_TOKEN;
  if (!token) {
    // Loud on the server, vague to the client.
    console.error("HF_TOKEN is not set — gallery assets cannot be served.");
    return new Response("Gallery assets unavailable", { status: 500 });
  }

  const upstream =
    `${HF_DATASETS}/${GALLERY_REPO}/resolve/${rev}/` +
    rel.split("/").map(encodeURIComponent).join("/");

  let res: Response;
  try {
    // HF redirects to a signed CDN URL; fetch drops Authorization on the
    // cross-origin hop, which is correct — the signed URL carries its own auth.
    // Forward Range so <audio> can seek within a clip without pulling the whole
    // file. Only Range — never the caller's other headers, which must not reach
    // HF with our token attached.
    const range = req.headers.get("range");
    res = await fetch(upstream, {
      headers: {
        Authorization: `Bearer ${token}`,
        ...(range ? { Range: range } : {}),
      },
      redirect: "follow",
    });
  } catch (err) {
    console.error(`gallery proxy: upstream fetch failed for ${rel}`, err);
    return new Response("Upstream unavailable", { status: 502 });
  }

  if (res.status === 404) {
    return new Response("Not found", { status: 404 });
  }
  if (!res.ok || !res.body) {
    // Never surface the upstream body — it can carry HF auth diagnostics.
    console.error(`gallery proxy: upstream ${res.status} for ${rel}`);
    return new Response("Upstream error", { status: 502 });
  }

  // A pinned SHA can be cached forever; "main" can change under us, so give it
  // a short TTL with revalidation instead.
  const cacheControl =
    rev === "main"
      ? "public, max-age=0, s-maxage=300, stale-while-revalidate=3600"
      : "public, max-age=31536000, s-maxage=31536000, immutable";

  const headers = new Headers({
    "Content-Type": contentTypeFor(rel),
    "Cache-Control": cacheControl,
    "X-Content-Type-Options": "nosniff",
  });
  const len = res.headers.get("content-length");
  if (len) headers.set("Content-Length", len);
  // Range plumbing: a 206 must carry its Content-Range or the browser treats
  // the partial body as the whole file. Accept-Ranges advertises seek support.
  const contentRange = res.headers.get("content-range");
  if (contentRange) headers.set("Content-Range", contentRange);
  const acceptRanges = res.headers.get("accept-ranges");
  if (acceptRanges) headers.set("Accept-Ranges", acceptRanges);

  return new Response(res.body, { status: res.status === 206 ? 206 : 200, headers });
}
