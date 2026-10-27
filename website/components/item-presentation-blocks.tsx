import type { ComponentType, ReactNode } from "react";
import type { Block } from "@/lib/presentation-transforms/types";
import { fieldLabel } from "@/lib/structured-content";
import { StructuredContent } from "./structured-content";
import { CodeBlock } from "./code-block";

export function ItemPresentationBlocks({
  blocks,
  Text,
  media,
}: {
  blocks: Block[];
  Text: ComponentType<{ text: string }>;
  media: (path: string, mediaType: string) => ReactNode;
}) {
  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.kind === "details")
          return (
            <details key={index} className="border-t border-[var(--line)] pt-3">
              <summary className="cursor-pointer text-xs font-medium">
                {block.title}
              </summary>
              <div className="mt-3">
                <ItemPresentationBlocks
                  blocks={block.blocks}
                  Text={Text}
                  media={media}
                />
              </div>
            </details>
          );
        if (block.kind === "media")
          return <div key={index}>{media(block.path, block.mediaType)}</div>;
        return (
          <section key={index} className="min-w-0" aria-label={block.title}>
            {block.title && (
              <h5 className="mb-2 text-xs font-medium text-[var(--muted)]">
                {block.kind === "code" ? block.title : fieldLabel(block.title)}
              </h5>
            )}
            {block.kind === "code" ? (
              <CodeBlock text={block.text} language={block.language} />
            ) : block.kind === "text" ? (
              <Text text={block.text} />
            ) : block.kind === "source" ? (
              <pre className="overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed">
                {block.text}
              </pre>
            ) : block.kind === "fields" ? (
              <StructuredContent value={block.value} Text={Text} />
            ) : block.kind === "choices" ? (
              <ol aria-label={block.title ?? "Choices"} className="space-y-2">
                {block.choices.map(({ label, text }, i) => (
                  <li
                    key={i}
                    className="flex min-w-0 gap-3 rounded border border-[var(--line)] px-3 py-2"
                  >
                    <span className="shrink-0 text-sm font-semibold">
                      {label}.
                    </span>
                    <div className="min-w-0">
                      <Text text={text} />
                    </div>
                  </li>
                ))}
              </ol>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
