"use client";

import { useSyncExternalStore } from "react";

const MEDIA_QUERY = "(prefers-reduced-motion: reduce)";

function subscribe(onStoreChange: () => void) {
  const mediaQueryList = window.matchMedia(MEDIA_QUERY);
  const handleChange = () => onStoreChange();

  mediaQueryList.addEventListener("change", handleChange);
  return () => mediaQueryList.removeEventListener("change", handleChange);
}

function getSnapshot() {
  return window.matchMedia(MEDIA_QUERY).matches;
}

/**
 * Hydration-safe reduced-motion hook. Returns `false` during SSR and the
 * initial client render so the server and client trees always match, then
 * resolves to the real preference after mount.
 */
export function useReducedMotionSafe(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}
