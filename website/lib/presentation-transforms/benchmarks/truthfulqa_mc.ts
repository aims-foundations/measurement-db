import { isRecord, structuredValue } from "../../structured-content";
import { readable } from "../shared";
import type { ItemTransform } from "../types";

const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (
    !isRecord(value) ||
    typeof value.prompt !== "string" ||
    !Array.isArray(value.candidate_continuations) ||
    !value.candidate_continuations.every((v) => typeof v === "string")
  )
    return readable(text);
  const { prompt, candidate_continuations: candidates, ...rest } = value;
  // These are complete answer continuations, not MMLU-style letter tokens.
  return [
    { kind: "text", title: "Prompt", text: prompt },
    {
      kind: "choices",
      title: "Candidate answers",
      choices: candidates.map((text, i) => ({ label: String(i + 1), text })),
    },
    { kind: "fields", value: rest },
  ];
};
const presentation = { item };

export default presentation;
