"use client";

import { useState, type ReactNode } from "react";
import { Pagination } from "./benchmark-records/pagination";
import { textPart, TEXT_PART_SIZE } from "@/lib/benchmark-records/text-pages";

export function TextPages({
  text,
  children,
}: {
  text: string;
  children: (part: string) => ReactNode;
}) {
  const [selected, setSelected] = useState(0);
  const total = Math.ceil(text.length / TEXT_PART_SIZE);
  const page = Math.min(selected, total - 1);
  return (
    <div className="min-w-0 space-y-3">
      <Pagination
        label="text parts"
        page={page}
        total={total}
        size={1}
        onChange={setSelected}
      />
      {children(textPart(text, page))}
    </div>
  );
}
