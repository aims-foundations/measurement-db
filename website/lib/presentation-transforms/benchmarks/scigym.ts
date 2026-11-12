/** SciGym releases an indented prompt with an embedded SBML/XML document. */
import { readable } from "../shared";
import type { ItemTransform } from "../types";

const prose = (text: string) => text.replace(/^ {16}/gm, "").trim();
const item: ItemTransform = (text) => {
  const xml = /<\?xml\b[\s\S]*?<sbml\b[\s\S]*?<\/sbml>/.exec(text);
  if (!xml || !text.includes("## Task Info")) return readable(text);
  return [
    { kind: "text", text: prose(text.slice(0, xml.index)) },
    { kind: "code", language: "xml", text: xml[0] },
    { kind: "text", text: prose(text.slice(xml.index + xml[0].length)) },
  ];
};
const presentation = { item, answerPath: ["final_model"] };
export default presentation;
