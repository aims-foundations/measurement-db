import type { Observation } from "./types";

export type Fields = Record<string, string>;
export type Filters = Record<string, string>;
export type Dimension = {
  key: string;
  label: string;
  values: (string | null)[];
};

/** Canonical contexts may be JSON objects or semicolon-delimited key=value fields. */
export function contextValues(text: unknown): Record<string, unknown> {
  if (typeof text !== "string" || !text) return {};
  try {
    const value: unknown = JSON.parse(text);
    if (value && typeof value === "object" && !Array.isArray(value))
      return value as Record<string, unknown>;
  } catch {
    /* A key=value context does not need JSON decoding. */
  }
  return Object.fromEntries(
    text.split(";").flatMap((part) => {
      const at = part.indexOf("=");
      return at < 0
        ? [["value", part]]
        : [[part.slice(0, at).trim(), part.slice(at + 1).trim()]];
    }),
  );
}

export function parseFields(text: unknown): Fields {
  return Object.fromEntries(
    Object.entries(contextValues(text)).map(([key, value]) => [
      key,
      typeof value === "string" ? value : JSON.stringify(value),
    ]),
  );
}

export function observationFields(observation: Observation): Fields {
  return {
    trial: String(observation.trial),
    ...Object.fromEntries(
      Object.entries(parseFields(observation.test_condition)).map(([k, v]) => [
        `condition.${k}`,
        v,
      ]),
    ),
    ...Object.fromEntries(
      Object.entries(parseFields(observation.interactors)).map(([k, v]) => [
        `interactor.${k}`,
        v,
      ]),
    ),
  };
}

export function prepareContexts(observations: Observation[]) {
  const fields = new Map(
    observations.map((record) => [
      record.response_id,
      observationFields(record),
    ]),
  );
  const keys = new Set([...fields.values()].flatMap(Object.keys));
  const dimensions: Dimension[] = [...keys].map((key) => ({
    key,
    label:
      key === "trial"
        ? "Trial"
        : key.replace(/^(condition|interactor)\./, "").replaceAll("_", " "),
    values: [
      ...new Set([...fields.values()].map((record) => record[key] ?? null)),
    ].sort((a, b) =>
      (a ?? "").localeCompare(b ?? "", undefined, { numeric: true }),
    ),
  }));
  return { fields, dimensions };
}

export function matchesFilters(fields: Fields, filters: Filters): boolean {
  return Object.entries(filters).every(
    ([key, token]) => !token || JSON.stringify(fields[key] ?? null) === token,
  );
}
