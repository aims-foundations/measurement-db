"use client";
import { useState } from "react";
import { CodeBlock } from "../code-block";
import { TextPages } from "../text-pages";
import "katex/dist/katex.min.css";
import { normalizeMath } from "@/lib/benchmark-records/text";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks";
import rehypeKatex from "rehype-katex";
import { MarkdownCodeBlock } from "../code-block";

/** Keep JSON as code; render authored Markdown and explicit math without executing HTML. */
export function RichText({ text }: { text: string }) {
  if (text.length > 65536)
    return (
      <TextPages text={text}>{(part) => <RichText text={part} />}</TextPages>
    );
  if (/^\s*<\?xml\b/.test(text) || /^\s*<sbml\b[\s\S]*<\/sbml>\s*$/.test(text))
    return <CodeBlock text={text} language="xml" />;
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    /* Plain text and Markdown are rendered below. */
  }
  if (json && typeof json === "object")
    return (
      <pre className="overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed">
        {JSON.stringify(json, null, 2)}
      </pre>
    );
  const normalized = normalizeMath(text);
  return (
    <div className="break-words text-[0.8125rem] leading-6 [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_h1]:my-3 [&_h1]:font-semibold [&_h2]:my-3 [&_h2]:font-semibold [&_h3]:my-2 [&_h3]:font-semibold [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_pre]:my-3 [&_pre]:overflow-auto [&_pre]:rounded [&_pre]:bg-black/5 [&_pre]:p-3 [&_code]:text-xs [&_table]:block [&_table]:overflow-auto [&_td]:border [&_td]:p-2 [&_th]:border [&_th]:p-2 [&_.katex-display]:overflow-auto">
      <ReactMarkdown
        remarkPlugins={[
          remarkGfm,
          [remarkMath, { singleDollarTextMath: false }],
          ...(/^(?:#{1,6} |```|[-*] )/m.test(text) ? [] : [remarkBreaks]),
        ]}
        rehypePlugins={[[rehypeKatex, { strict: "ignore", trust: false }]]}
        components={{
          pre: MarkdownCodeBlock,
          a: (props) => (
            <a
              {...props}
              className="text-[var(--lagunita)] underline"
              rel="noreferrer"
            />
          ),
          img: ({ alt }) => (
            <span className="italic text-[var(--muted)]">
              {alt ? `Image: ${alt}` : "Image"} (see attachments)
            </span>
          ),
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

export function SourceText({ text, label }: { text: string; label: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details
      onToggle={(event) => setOpen(event.currentTarget.open)}
      className="mt-3 text-xs text-[var(--muted)]"
    >
      <summary className="cursor-pointer">
        Original {label.toLowerCase()}
      </summary>
      {open && (
        <div aria-label={`Original ${label.toLowerCase()}`}>
          <CodeBlock text={text} language="Original text" />
        </div>
      )}
    </details>
  );
}
