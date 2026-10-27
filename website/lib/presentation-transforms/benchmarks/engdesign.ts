/** engdesign: preserve source fields; raw content is displayed separately. */
import { readable } from "../shared";
import { isRecord, structuredValue } from "../../structured-content";
import type { Block, ItemTransform } from "../types";

const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (!isRecord(value) || typeof value.output_structure !== "string")
    return readable(text);
  return Object.entries(value).map(
    ([title, value]): Block =>
      title === "output_structure"
        ? {
            kind: "code",
            title: "Output structure",
            text: value as string,
            language: "python",
          }
        : { kind: "fields", title, value },
  );
};
const presentation = { item };

export default presentation;
