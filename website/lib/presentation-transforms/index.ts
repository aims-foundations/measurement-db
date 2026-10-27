import { presentations } from "./registry";
import type { Presentation } from "./types";
export { presentations } from "./registry";

const RAW: Presentation = {};
export const presentationFor = (slug: string): Presentation =>
  presentations[slug] ?? RAW;
