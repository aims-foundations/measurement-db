import { isRecord, structuredValue } from "../../structured-content";
import { readable } from "../shared";
import type { ItemTransform } from "../types";

const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  const labels = ["A", "B", "C", "D", "E"];
  if (
    !isRecord(value) ||
    typeof value.question !== "string" ||
    labels.some((label) => typeof value[`option${label}`] !== "string")
  )
    return readable(text);
  const rest = Object.fromEntries(
    Object.entries(value).filter(
      ([key]) =>
        key !== "question" && !labels.some((label) => key === `option${label}`),
    ),
  );
  return [
    { kind: "text", title: "Question", text: value.question },
    {
      kind: "choices",
      title: "Answer choices",
      choices: labels.map((label) => ({
        label,
        text: value[`option${label}`] as string,
      })),
    },
    { kind: "fields", value: rest },
  ];
};
const presentation = { item };

export default presentation;
