import type { ItemTransform } from "../types";

// Full LaTeX documents are source, not Markdown or an inline math expression.
const item: ItemTransform = (text) => [
  { kind: "source", title: "Manuscript source", text },
];
const presentation = { item };

export default presentation;
