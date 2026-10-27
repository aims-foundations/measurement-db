import { textFields } from "../shared";

const presentation = {
  item: textFields("prompt"),
  answerPath: ["source_record", "response"],
};

export default presentation;
