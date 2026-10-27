/** Decode structured text for display without coercing IDs or numeric strings. */
export function structuredValue(value: unknown): unknown {
  for (let layer = 0; layer < 3 && typeof value === "string"; layer++) {
    try {
      const decoded: unknown = JSON.parse(value);
      if (
        typeof decoded !== "string" &&
        !Array.isArray(decoded) &&
        !isRecord(decoded)
      )
        break;
      value = decoded;
    } catch {
      break;
    }
  }
  return value;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function fieldLabel(key: string): string {
  const words = key.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/[_-]/g, " ");
  return (words.charAt(0).toUpperCase() + words.slice(1)).replace(
    /\b(id|url|json)\b/gi,
    (word) => word.toUpperCase(),
  );
}

const FIELD_LISTS = new Set([
  "item_features",
  "subject_features_extra",
  "test_condition",
  "interactors",
]);

/** Only declared metadata fields use key=value syntax; ambiguous text stays intact. */
export function metadataFields(
  record: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(record).map(([key, value]) => {
      if (
        !FIELD_LISTS.has(key) ||
        typeof value !== "string" ||
        isRecord(structuredValue(value))
      )
        return [key, value];
      const parts = value
        .split(";")
        .map((part) => /^\s*([\w.-]+)=([\s\S]*)$/.exec(part));
      const keys = parts.map((part) => part?.[1]);
      if (parts.some((part) => !part) || new Set(keys).size !== keys.length)
        return [key, value];
      return [
        key,
        Object.fromEntries(parts.map((part) => [part![1], part![2]])),
      ];
    }),
  );
}
