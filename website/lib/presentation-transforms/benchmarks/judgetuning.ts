/** judgetuning: preserve source fields; raw content is displayed separately. */
import { readable } from "../shared";
import { isRecord, structuredValue } from "../../structured-content";
import type { Block, ItemTransform } from "../types";

// Some instructions in this mixed collection are unfenced Python programs.
const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (!isRecord(value)) return readable(text);
  return Object.entries(value).map(
    ([title, value]): Block =>
      title === "instruction" &&
      typeof value === "string" &&
      /^\s*(?:import \w|from [\w.]+ import |def \w+\(|class \w+[(:])/.test(
        value,
      )
        ? { kind: "code", title, text: value, language: "python" }
        : { kind: "fields", title, value },
  );
};
const presentation = { item };

export default presentation;
