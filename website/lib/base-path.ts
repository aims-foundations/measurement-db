// Single source of truth for the path prefix this site is served under.
// Imported by next.config.ts (basePath) AND by client code that builds asset /
// fetch URLs at runtime.
//
// Next.js auto-prefixes basePath for next/link, next/router, and next/image,
// but NOT for plain fetch() or `new Image()`. Those must be wrapped in
// withBase() so the request hits <origin>/measurement-db/... (which the main
// site proxies here) rather than <origin>/... (which does not).
export const BASE_PATH = "/measurement-db";

/**
 * Prefix a root-absolute public path (e.g. "/benchmarks/data/x.json.gz") with
 * BASE_PATH. Use for fetch() and `new Image()` only — do NOT use for next/image
 * or next/link, which apply basePath themselves (double-prefixing breaks them).
 */
export function withBase(path: string): string {
  return `${BASE_PATH}${path.startsWith("/") ? path : `/${path}`}`;
}
