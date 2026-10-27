/** stanford_orb: preserve source fields; raw content is displayed separately. */
import { sections } from "../shared";

const presentation = {
  item: sections([
    { key: "capture", title: "Capture" },
    { key: "input_scope", title: "Input scope" },
    { key: "input_archives", title: "Source archives" },
    { key: "cameras", title: "Camera parameters", collapsed: true },
  ]),
};

export default presentation;
