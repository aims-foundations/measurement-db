/** tau2_bench: preserve source fields; raw content is displayed separately. */
import { sections } from "../shared";

const presentation = {
  item: sections([
    { key: "task", title: "Item specification" },
    {
      key: "environment_info",
      title: "Environment and policy",
      collapsed: true,
    },
    {
      key: "user_guidelines",
      title: "Simulated-user guidelines",
      collapsed: true,
    },
  ]),
};

export default presentation;
