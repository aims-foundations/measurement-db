"use client";

import { Meilisearch } from "meilisearch";
import searchConfig from "@/meilisearch.config.json";

export const BENCHMARKS_INDEX_UID = searchConfig.benchmarksIndex;
export const RESULTS_INDEX_UID = searchConfig.resultsIndex;

let publicClient: Meilisearch | null | undefined;

function validPublicHost(host: string): boolean {
  try {
    const url = new URL(host);
    const local = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
    return url.protocol === "https:" || (local && url.protocol === "http:");
  } catch {
    return false;
  }
}

/**
 * Return the browser-safe search client, or null when Cloud is not configured.
 * The key must have only the `search` action for the two public indexes. Admin
 * credentials belong in CI and must never use the NEXT_PUBLIC_ prefix.
 */
export function getPublicSearchClient(): Meilisearch | null {
  if (publicClient !== undefined) return publicClient;

  const host = process.env.NEXT_PUBLIC_MEILISEARCH_HOST?.trim() ?? "";
  const apiKey = process.env.NEXT_PUBLIC_MEILISEARCH_SEARCH_KEY?.trim() ?? "";
  if (!host || !apiKey || !validPublicHost(host)) {
    publicClient = null;
    return publicClient;
  }

  publicClient = new Meilisearch({ host, apiKey, timeout: 5_000 });
  return publicClient;
}
