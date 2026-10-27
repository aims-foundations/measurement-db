import { readable } from "../shared";

const presentation = {
  item: readable,
  answerPath: ["native_answer", "answer_markdown"],
  // Source row links the two perspectives; slot is not a comparison identity.
  pairing: {
    subjectFeature: "source_provider",
    contextFields: ["source_row", "axis", "render_mode"],
  },
};

export default presentation;
