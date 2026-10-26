/** Explicit display mappings for source payloads; they never change canonical records. */
export type Presentation = {
  promptPath?: string[];
  answerPath?: string[];
  pairing?: { subjectFeature: string; contextFields: string[] };
};
const presentations: Record<string, Presentation> = {
  aegis: { promptPath: ["prompt"], answerPath: ["source_record", "response"] },
  real_pocqi: {
    answerPath: ["native_answer", "answer_markdown"],
    // The release records two slots for each rating, identified by its source row.
    pairing: {
      subjectFeature: "source_provider",
      contextFields: ["source_row", "axis", "render_mode"],
    },
  },
};
const DEFAULT_PRESENTATION: Presentation = {};
export const presentationFor = (slug: string): Presentation =>
  presentations[slug] ?? DEFAULT_PRESENTATION;

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
