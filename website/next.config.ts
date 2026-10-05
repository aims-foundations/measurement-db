import type { NextConfig } from "next";
import { BASE_PATH } from "./lib/base-path";

// This site is served under the /measurement-db path prefix. When deployed, its
// pages and assets all live under <origin>/measurement-db, so the main AIMS site
// can forward aimslab.stanford.edu/measurement-db here with a single passthrough
// rewrite (same pattern as Benchmark Caliper). Visiting the origin root returns
// 404 by design — only /measurement-db/* is served.
const nextConfig: NextConfig = {
  basePath: BASE_PATH,
  reactCompiler: true,
  poweredByHeader: false,
  typedRoutes: true,
  outputFileTracingIncludes: {
    "/[slug]": ["./public/benchmark-data/*/view.json.gz"],
  },
  async headers() {
    return [
      {
        source: "/benchmark-data/:path*.json.gz",
        headers: [
          { key: "Content-Type", value: "application/json; charset=utf-8" },
          { key: "Content-Encoding", value: "gzip" },
        ],
      },
    ];
  },
  images: {
    // 85 for photographic card images where the default 75 shows compression.
    qualities: [75, 85],
  },
};
// NOTE on dev-server file watching: this repo lives on a Lustre mount
// (/lfs/...), which does not deliver inotify events, so Turbopack's watcher
// never sees edits and `next dev` serves stale pages until restarted.
// `watchOptions.pollIntervalMs` was tried and does not help (the Rust poll
// watcher also fails here). The `dev` script therefore runs webpack with
// WATCHPACK_POLLING=true, whose JS-level polling works on this mount;
// `dev:turbo` keeps the fast Turbopack path for filesystems with working
// inotify.

export default nextConfig;
