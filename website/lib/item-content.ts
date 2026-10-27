import { isRecord, structuredValue } from "./structured-content";

export type ItemDocument = {
  value: unknown;
  multipleChoice?: {
    question: string;
    choices: { label: string; text: string }[];
    context: string;
    details: Record<string, unknown>;
  };
};

/** MMLU stores the full few-shot prompt and four answer-letter continuations. */
function mmluQuestion(value: Record<string, unknown>) {
  const { prompt, candidate_continuations: candidates, ...details } = value;
  if (
    typeof prompt !== "string" ||
    !Array.isArray(candidates) ||
    candidates.map((v) => (typeof v === "string" ? v.trim() : "")).join() !==
      "A,B,C,D"
  )
    return;

  // Completed answers delimit demonstrations; the final unanswered block is
  // the evaluated question. Keep every preceding character in the disclosure.
  const examples = [...prompt.matchAll(/^Answer: [A-D][ \t]*\r?\n\r?\n/gm)];
  const last = examples.at(-1);
  const start = last ? last.index! + last[0].length : 0;
  const question = prompt.slice(start);
  const ending = /\r?\nAnswer:[ \t]*$/.exec(question);
  if (!ending) return;
  const body = question.slice(0, ending.index);
  const options = [...body.matchAll(/^([A-D])\.[ \t]+/gm)];
  // Ambiguous or unfamiliar source formats stay in the general field view.
  if (options.map((m) => m[1]).join() !== "A,B,C,D") return;
  const stem = body.slice(0, options[0].index).trim();
  const choices = options.map((match, i) => ({
    label: match[1],
    text: body
      .slice(
        match.index! + match[0].length,
        options[i + 1]?.index ?? body.length,
      )
      .trim(),
  }));
  if (!stem || choices.some((choice) => !choice.text)) return;
  return { question: stem, choices, context: prompt.slice(0, start), details };
}

export function prepareItemContent(
  text: string,
  format?: "mmlu",
): ItemDocument {
  const value = structuredValue(text);
  const multipleChoice =
    format === "mmlu" && isRecord(value) ? mmluQuestion(value) : undefined;
  return { value, ...(multipleChoice ? { multipleChoice } : {}) };
}
