import { isRecord, structuredValue } from "../../structured-content";
import { readable } from "../shared";
import type { ItemTransform } from "../types";

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

export const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  const choice = isRecord(value) ? mmluQuestion(value) : undefined;
  if (!choice) return readable(text);
  return [
    { kind: "text", title: "Question", text: choice.question },
    { kind: "choices", title: "Answer choices", choices: choice.choices },
    ...(choice.context
      ? [
          {
            kind: "details" as const,
            title: "Instructions and example questions",
            blocks: [{ kind: "text" as const, text: choice.context }],
          },
        ]
      : []),
    ...(Object.keys(choice.details).length
      ? [
          {
            kind: "details" as const,
            title: "Request details",
            blocks: [{ kind: "fields" as const, value: choice.details }],
          },
        ]
      : []),
  ];
};

/** MMLU releases four answer-letter log-likelihoods, not a generated completion. */
export function answerScores(source: string) {
  if (!source) return;
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

const presentation = { item, answerScores };

export default presentation;
