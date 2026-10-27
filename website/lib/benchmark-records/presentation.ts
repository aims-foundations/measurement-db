/** Explicit display mappings for source payloads; they never change canonical records. */
export type Presentation = {
  promptPath?: string[];
  itemFormat?: "mmlu";
  answerPath?: string[];
  answerFormat?: "mmlu";
  pairing?: { subjectFeature: string; contextFields: string[] };
};
const presentations: Record<string, Presentation> = {
  mmlu: { itemFormat: "mmlu", answerFormat: "mmlu" },
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

/** MMLU releases four answer-letter log-likelihoods, not a generated completion. */
export function answerScores(source: string | null, format?: "mmlu") {
  if (!source || format !== "mmlu") return;
  try {
    const record = JSON.parse(source)?.source_record;
    const choices: unknown = record?.choices,
      scores: unknown = record?.predictions;
    if (
      !Array.isArray(choices) ||
      !Array.isArray(scores) ||
      scores.length !== 4 ||
      choices
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .join() !== "A,B,C,D" ||
      !scores.every(
        (value) => typeof value === "number" && Number.isFinite(value),
      )
    )
      return;
    return choices.map((choice: string, i) => ({
      label: choice.trim(),
      score: scores[i] as number,
    }));
  } catch {
    // An unfamiliar trace retains the general structured display.
  }
}
