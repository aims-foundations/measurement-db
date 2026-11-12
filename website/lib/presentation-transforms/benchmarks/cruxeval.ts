/** cruxeval: preserve source fields; raw content is displayed separately. */
import type { Block, ItemTransform } from "../types";

const item: ItemTransform = (text) => {
  const blocks: Block[] = [];
  let start = 0;
  for (const match of text.matchAll(/\[PYTHON\]([\s\S]*?)\[\/PYTHON\]/g)) {
    if (match.index > start)
      blocks.push({ kind: "text", text: text.slice(start, match.index) });
    blocks.push({ kind: "code", text: match[1], language: "python" });
    start = match.index + match[0].length;
  }
  if (start < text.length)
    blocks.push({ kind: "text", text: text.slice(start) });
  return blocks;
};
const presentation = { item };

export default presentation;
