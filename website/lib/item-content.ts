import { presentationFor } from "./presentation-transforms";

/** Missing or incompatible adapters leave the exact original text available. */
export function prepareItemContent(text: string, benchmark: string) {
  try {
    return presentationFor(benchmark).item?.(text) ?? null;
  } catch {
    return null;
  }
}
