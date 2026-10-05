"use client";

import { useState } from "react";
import { Check, Copy } from "@phosphor-icons/react";

// Matches a BibTeX field line: leading indent, field name, `=`, and value.
const FIELD_RE = /^(\s*)([A-Za-z][\w-]*)(\s*=\s*.*)$/;

// Minimal BibTeX highlighter: field names in cardinal, everything else
// (the @type{key,, values, and closing brace) in digital-blue — matching the
// textbook citation style.
function HighlightedLine({ line }: { line: string }) {
  const match = line.trim().startsWith("@") ? null : FIELD_RE.exec(line);
  if (match) {
    const [, indent, field, rest] = match;
    return (
      <>
        {indent}
        <span className="text-[var(--cardinal)]">{field}</span>
        <span className="text-[var(--digital-blue)]">{rest}</span>
      </>
    );
  }
  return <span className="text-[var(--digital-blue)]">{line}</span>;
}

// Copy-to-clipboard BibTeX block for the measurement-db citation. Client
// component because it needs the clipboard API and local "copied" state.
export function CitationBlock({ bibtex }: { bibtex: string }) {
  const [copied, setCopied] = useState(false);
  const lines = bibtex.split("\n");

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(bibtex);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard unavailable (e.g. insecure context) — silently no-op.
    }
  }

  return (
    <div className="relative overflow-hidden rounded-md border border-[var(--line)] bg-[var(--fog-light)]">
      <div className="flex min-h-12 items-center justify-between gap-3 border-b border-[var(--line)] px-3 py-2 sm:px-4">
        <span className="font-mono text-[0.68rem] uppercase tracking-wider text-[var(--muted)]">
          BibTeX
        </span>
        <button
          type="button"
          onClick={handleCopy}
          aria-label={copied ? "Copied" : "Copy citation"}
          className="inline-flex min-h-11 items-center gap-1.5 rounded-md border border-[var(--line)] bg-white px-3 py-1.5 text-xs font-medium text-[var(--ink)] transition hover:border-[var(--muted)] hover:bg-[var(--fog-light)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] sm:min-h-9"
        >
          {copied ? (
            <Check size={14} weight="bold" aria-hidden="true" />
          ) : (
            <Copy size={14} weight="regular" aria-hidden="true" />
          )}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div
        className="overflow-x-auto px-3 py-4 sm:px-4 sm:py-5"
        aria-label="BibTeX citation"
        role="region"
        tabIndex={0}
      >
        <pre className="font-mono text-xs leading-5 sm:text-sm sm:leading-7">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="flex items-start">
                <span
                  aria-hidden="true"
                  className="mr-3 w-4 shrink-0 select-none text-right text-[var(--muted)]/50 sm:mr-5"
                >
                  {i + 1}
                </span>
                <span className="min-w-0 whitespace-pre-wrap [overflow-wrap:anywhere] sm:min-w-max sm:whitespace-pre sm:[overflow-wrap:normal]">
                  <HighlightedLine line={line} />
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
