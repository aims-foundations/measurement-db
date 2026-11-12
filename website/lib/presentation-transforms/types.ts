/** Temporary display documents, never written back to benchmark data. */
export type Block =
  | { kind: "code"; text: string; language?: string; title?: string }
  | { kind: "text" | "source"; text: string; title?: string }
  | { kind: "fields"; value: unknown; title?: string }
  | {
      kind: "choices";
      choices: { label: string; text: string }[];
      title?: string;
    }
  | { kind: "media"; path: string; mediaType: string }
  | { kind: "details"; title: string; blocks: Block[] };

export type ItemTransform = (text: string) => Block[] | null;
export type AnswerScore = { label: string; score: number };
export type Presentation = {
  item?: ItemTransform;
  answerPath?: string[];
  answerNotice?: string;
  answerScores?: (text: string) => AnswerScore[] | undefined;
  pairing?: { subjectFeature: string; contextFields: string[] };
};
