/** The released web page is source HTML, displayed as code rather than prose. */
import { isRecord, structuredValue } from "../../structured-content";
import { readable } from "../shared";
import type { ItemTransform, Block } from "../types";
const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (!isRecord(value) || typeof value.cleaned_html !== "string")
    return readable(text);
  const { query, cleaned_html, ...rest } = value;
  const blocks: Block[] = [];
  if (query !== undefined)
    blocks.push({ kind: "fields", title: "Instruction", value: query });
  blocks.push({
    kind: "details",
    title: "Web page HTML",
    blocks: [{ kind: "code", language: "html", text: cleaned_html }],
  });
  if (Object.keys(rest).length) blocks.push({ kind: "fields", value: rest });
  return blocks;
};
const presentation = { item };
export default presentation;
