import { isRecord, structuredValue } from "../structured-content";
import type { Block, ItemTransform } from "./types";

/** Formatting only: unknown keys, false, zero, null, and list order survive. */
export const readable: ItemTransform = (text) => {
  const value = structuredValue(text);
  return typeof value === "string"
    ? [{ kind: "text", text: value }]
    : [{ kind: "fields", value }];
};

/** Reorder declared source fields; long context can collapse without being omitted. */
export function sections(
  fields: { key: string; title: string; collapsed?: boolean }[],
): ItemTransform {
  return (text) => {
    const value = structuredValue(text);
    if (!isRecord(value)) return readable(text);
    const blocks: Block[] = [];
    for (const { key, title, collapsed } of fields) {
      if (!Object.hasOwn(value, key)) continue;
      blocks.push(
        collapsed
          ? {
              kind: "details",
              title,
              blocks: [{ kind: "fields", value: value[key] }],
            }
          : { kind: "fields", title, value: value[key] },
      );
    }
    const rest = Object.fromEntries(
      Object.entries(value).filter(
        ([key]) => !fields.some((field) => field.key === key),
      ),
    );
    if (Object.keys(rest).length) blocks.push({ kind: "fields", value: rest });
    return blocks.length ? blocks : readable(text);
  };
}

/** Select explicit fields, retaining the rest below them. No key-name guessing. */
export function textFields(...keys: string[]): ItemTransform {
  return (text) => {
    const value = structuredValue(text);
    if (!isRecord(value) || keys.some((key) => typeof value[key] !== "string"))
      return readable(text);
    const remaining = Object.fromEntries(
      Object.entries(value).filter(([key]) => !keys.includes(key)),
    );
    return [
      ...keys.map(
        (key): Block => ({
          kind: "text",
          title: key,
          text: value[key] as string,
        }),
      ),
      ...(Object.keys(remaining).length
        ? [{ kind: "fields" as const, value: remaining }]
        : []),
    ];
  };
}

export const questionOptions: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (
    !isRecord(value) ||
    typeof value.question !== "string" ||
    !Array.isArray(value.options) ||
    !value.options.every((v) => typeof v === "string")
  )
    return readable(text);
  const { question, options, ...rest } = value;
  return [
    { kind: "text", title: "Question", text: question },
    {
      kind: "choices",
      title: "Answer choices",
      choices: options.map((text, i) => ({
        label: String.fromCharCode(65 + i),
        text,
      })),
    },
    ...(Object.keys(rest).length
      ? [{ kind: "fields" as const, value: rest }]
      : []),
  ];
};

/** Released multimedia block order is meaningful. Retain extra block fields. */
export const multimedia: ItemTransform = (text) => {
  const value = structuredValue(text);
  if (!isRecord(value) || !Array.isArray(value.multimedia_elements))
    return readable(text);
  const blocks: Block[] = [];
  for (const block of value.multimedia_elements) {
    if (!isRecord(block)) return readable(text);
    const { content_type, text: body, location, ...rest } = block;
    if (
      content_type === "text/plain" &&
      typeof body === "string" &&
      location === undefined
    ) {
      blocks.push({ kind: "text", text: body });
    } else if (
      typeof content_type === "string" &&
      typeof location === "string" &&
      body === undefined
    ) {
      blocks.push({ kind: "media", path: location, mediaType: content_type });
    } else return readable(text);
    if (Object.keys(rest).length) blocks.push({ kind: "fields", value: rest });
  }
  const rest = Object.fromEntries(
    Object.entries(value).filter(([key]) => key !== "multimedia_elements"),
  );
  if (Object.keys(rest).length) blocks.push({ kind: "fields", value: rest });
  return blocks.length ? blocks : readable(text);
};

/** Explicit curation markers are preserved in raw text and separated visually. */
export const markedAside: ItemTransform = (text) => {
  const match = /<!--(preview|meta)[\s\S]*?-->/.exec(text);
  if (!match) return readable(text);
  return [
    { kind: "text", text: text.slice(0, match.index) },
    {
      kind: "details",
      title:
        match[1] === "preview"
          ? "Rendered preview · not shown to the model"
          : "Curation metadata · not shown to the model",
      blocks: [
        { kind: "text", text: text.slice(match.index + match[0].length) },
      ],
    },
  ];
};
