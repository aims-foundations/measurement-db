/** aider: preserve source fields; raw content is displayed separately. */
import { readable } from "../shared";
import { isRecord, structuredValue } from "../../structured-content";
import type { Block, ItemTransform } from "../types";

// This release stores escaped source-file strings inside the item JSON.
function sourceText(text: string): string {
  if (text.includes("\n")) return text;
  try {
    return JSON.parse(`"${text}"`) as string;
  } catch {
    return text;
  }
}
const languages: Record<string, string> = {
  py: "python",
  cpp: "cpp",
  cc: "cpp",
  h: "cpp",
  hpp: "cpp",
  js: "javascript",
  ts: "typescript",
  java: "java",
  rs: "rust",
  go: "go",
  rb: "ruby",
  sh: "bash",
};
const item: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (!isRecord(value) || !isRecord(value.initial_files)) return readable(text);
  const { initial_files, instructions, ...rest } = value;
  const blocks: Block[] =
    typeof instructions === "string"
      ? [
          {
            kind: "text",
            title: "Instructions",
            text: sourceText(instructions),
          },
        ]
      : instructions === undefined
        ? []
        : [{ kind: "fields", title: "Instructions", value: instructions }];
  for (const [path, content] of Object.entries(initial_files))
    blocks.push(
      typeof content === "string"
        ? {
            kind: "code",
            title: path,
            text: sourceText(content),
            language: languages[path.split(".").at(-1) ?? ""],
          }
        : { kind: "fields", title: path, value: content },
    );
  if (Object.keys(rest).length) blocks.push({ kind: "fields", value: rest });
  return blocks;
};
const presentation = { item };

export default presentation;
