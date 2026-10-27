"use client";
import { TextPages } from "./text-pages";
import { isValidElement, type ReactNode } from "react";

/** Display source code verbatim; never evaluate it or run math processing on it. */
export function CodeBlock({
  text,
  language,
}: {
  text: string;
  language?: string;
}) {
  if (text.length > 65536)
    return (
      <TextPages text={text}>
        {(part) => <CodeBlock text={part} language={language} />}
      </TextPages>
    );
  return (
    <div className="my-3 min-w-0 overflow-hidden rounded-md border border-[var(--line)] bg-[var(--fog-light)]">
      <div className="border-b border-[var(--line)] px-3 py-1.5 text-[0.625rem] uppercase tracking-wide text-[var(--muted)]">
        {language || "Code"}
      </div>
      <pre className="!m-0 max-h-96 overflow-auto !bg-transparent p-3 text-xs leading-5">
        <code
          className={language ? `language-${language}` : undefined}
          data-language={language}
          style={{
            whiteSpace: "pre",
            fontFamily: "var(--font-mono, monospace)",
          }}
        >
          {text}
        </code>
      </pre>
    </div>
  );
}

/** Markdown already identifies code fences and their optional language. */
export function MarkdownCodeBlock({ children }: { children?: ReactNode }) {
  if (isValidElement<{ className?: string; children?: ReactNode }>(children)) {
    const language = /language-([^\s]+)/.exec(
      children.props.className ?? "",
    )?.[1];
    return (
      <CodeBlock
        text={String(children.props.children ?? "")}
        language={language}
      />
    );
  }
  return <pre className="overflow-auto font-mono text-xs">{children}</pre>;
}
