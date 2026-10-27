/** matharena: preserve source fields; raw content is displayed separately. */
import type { Block, ItemTransform } from "../types";

const item: ItemTransform = (text) => {
  const blocks: Block[] = [];
  let start = 0;
  for (const match of text.matchAll(
    /\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}/g,
  )) {
    if (match.index > start)
      blocks.push({ kind: "text", text: text.slice(start, match.index) });
    blocks.push({
      kind: "code",
      title: "Diagram source",
      language: "latex",
      text: match[0],
    });
    start = match.index + match[0].length;
  }
  if (start < text.length)
    blocks.push({ kind: "text", text: text.slice(start) });
  return blocks;
};
const presentation = { item };

export default presentation;
