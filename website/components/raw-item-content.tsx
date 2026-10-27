"use client";

import { useState } from "react";

/** React escapes the original string. Never parse, trim, truncate, or reserialize it. */
export function RawItemContent({
  text,
  collapsed = false,
}: {
  text: string | null;
  collapsed?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const content =
    text === null ? (
      <p className="text-sm text-[var(--muted)]">No text component released.</p>
    ) : (
      <pre
        aria-label="Original item"
        className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded bg-black/5 p-3 text-xs leading-relaxed"
      >
        {text}
      </pre>
    );
  if (!collapsed) return content;
  return (
    <details
      className="mt-4 border-t border-[var(--line)] pt-3 text-xs text-[var(--muted)]"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer">Original item</summary>
      {open && <div className="mt-3">{content}</div>}
    </details>
  );
}
