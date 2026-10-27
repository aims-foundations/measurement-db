/** Separate the released configuration from the instruction; preserve other blocks. */
import { multimedia } from "../shared";
import type { ItemTransform } from "../types";
const item: ItemTransform = (text) =>
  (multimedia(text) ?? []).flatMap((block) => {
    const prefix = "Published initial task configuration:\n";
    if (block.kind !== "text" || !block.text.startsWith(prefix)) return [block];
    try {
      const value: unknown = JSON.parse(block.text.slice(prefix.length));
      return [{ kind: "fields", title: "Initial task configuration", value }];
    } catch {
      return [block];
    }
  });
const presentation = {
  item,
  answerNotice:
    "The release includes evaluator records, but does not include the agent’s answer or interaction transcript.",
};
export default presentation;
