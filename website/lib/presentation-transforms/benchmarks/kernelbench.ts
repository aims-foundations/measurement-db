/** kernelbench: preserve source fields; raw content is displayed separately. */
import type { ItemTransform } from "../types";

const item: ItemTransform = (text) => [
  { kind: "code", text, language: "python", title: "Reference implementation" },
];
const presentation = { item };

export default presentation;
