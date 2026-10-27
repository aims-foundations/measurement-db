export { presentationFor } from "../presentation-transforms";
export type { Presentation } from "../presentation-transforms/types";

export function displayText(
  source: string | null,
  path?: string[],
): string | null {
  if (source === null || !path) return source;
  try {
    let value: unknown = JSON.parse(source);
    for (const key of path) {
      if (!value || typeof value !== "object" || !Object.hasOwn(value, key))
        return source;
      value = (value as Record<string, unknown>)[key];
    }
    return typeof value === "string" ? value : source;
  } catch {
    return source;
  }
}
