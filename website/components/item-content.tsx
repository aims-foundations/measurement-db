import type { ComponentType } from "react";
import { prepareItemContent } from "@/lib/item-content";
import { RichText } from "./benchmark-records/rich-text";
import { StructuredContent } from "./structured-content";

type TextRenderer = ComponentType<{ text: string }>;

export function ItemContent({
  text,
  format,
  Text = RichText,
}: {
  text: string;
  format?: "mmlu";
  Text?: TextRenderer;
}) {
  const document = prepareItemContent(text, format);
  const choice = document.multipleChoice;
  if (!choice) return <StructuredContent value={document.value} Text={Text} />;
  return (
    <div className="space-y-4">
      <div aria-label="Question">
        <Text text={choice.question} />
      </div>
      <ol aria-label="Answer choices" className="space-y-2">
        {choice.choices.map(({ label, text }) => (
          <li
            key={label}
            className="flex min-w-0 gap-3 rounded border border-[var(--line)] px-3 py-2"
          >
            <span className="shrink-0 text-sm font-semibold">{label}.</span>
            <div className="min-w-0">
              <Text text={text} />
            </div>
          </li>
        ))}
      </ol>
      {choice.context && (
        <details className="border-t border-[var(--line)] pt-3">
          <summary className="cursor-pointer text-xs font-medium">
            Instructions and example questions
          </summary>
          <div className="mt-3">
            <Text text={choice.context} />
          </div>
        </details>
      )}
      {Object.keys(choice.details).length > 0 && (
        <details className="border-t border-[var(--line)] pt-3">
          <summary className="cursor-pointer text-xs font-medium">
            Request details
          </summary>
          <div className="mt-3">
            <StructuredContent value={choice.details} Text={Text} />
          </div>
        </details>
      )}
    </div>
  );
}
