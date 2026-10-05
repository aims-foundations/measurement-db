// Proxied card images, institution logos, and audio at one published revision.
import revision from "@/content/generated/gallery-revision.json";

import { BASE_PATH } from "./base-path";

/** HF dataset repo holding the published assets. */
export const GALLERY_REPO: string = revision.repo;

/**
 * Commit SHA the site reads assets at, written by
 * scripts/render_website/publish_gallery_assets.py. The literal "main" is the
 * un-pinned bootstrap/dev value; the proxy serves it with a short TTL instead
 * of an immutable one, since its contents can change under it.
 */
export const GALLERY_REVISION: string = revision.revision;

/**
 * Build the proxied URL for a gallery asset. Accepts either a bare relative
 * path ("traces/hle/abc.json.gz") or the historical root-absolute form
 * ("/benchmarks/traces/hle/abc.json.gz") that benchmark-details.json still
 * stores for matrices, so callers can pass either without special-casing.
 *
 * Use this instead of withBase() for anything under public/benchmarks/.
 */
export function assetUrl(path: string): string {
  const rel = path.replace(/^\/?(?:benchmarks\/)?/, "");
  return `${BASE_PATH}/benchmarks/r/${GALLERY_REVISION}/${rel}`;
}
