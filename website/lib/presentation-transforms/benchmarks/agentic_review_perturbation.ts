import type { ItemTransform } from "../types";

// Full LaTeX documents are source, not Markdown or an inline math expression.
const item: ItemTransform = (text) => [
  { kind: "code", title: "Manuscript source", language: "latex", text },
];
const presentation = { item };

export default presentation;
