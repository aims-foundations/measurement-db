"use client";

import { useMemo, type ComponentType, type ReactNode } from "react";
import { prepareItemContent } from "@/lib/item-content";
import type { Block } from "@/lib/presentation-transforms/types";
import type { AssetLink } from "@/lib/benchmark-records/types";
import { RichText } from "./benchmark-records/rich-text";
import { ItemPresentationBlocks } from "./item-presentation-blocks";
import { ItemAttachments } from "./item-attachments";
import { RawItemContent } from "./raw-item-content";

function mediaPaths(blocks: Block[]): string[] {
  return blocks.flatMap((block) =>
    block.kind === "media"
      ? [JSON.stringify([block.path, block.mediaType])]
      : block.kind === "details"
        ? mediaPaths(block.blocks)
        : [],
  );
}

/** The same custom-first/raw-second panel is used by every benchmark viewer. */
export function ItemContent({
  text,
  benchmark,
  assets = [],
  renderAsset,
  Text = RichText,
}: {
  text: string | null;
  benchmark: string;
  assets?: AssetLink[];
  renderAsset?: (link: AssetLink) => ReactNode;
  Text?: ComponentType<{ text: string }>;
}) {
  const blocks = useMemo(
    () => (text === null ? null : prepareItemContent(text, benchmark)),
    [text, benchmark],
  );
  const referenced = new Set(mediaPaths(blocks ?? []));
  const asset =
    renderAsset ??
    ((link: AssetLink) => (
      <p className="text-xs">
        {link.path} · {link.media_type} · {link.role}
      </p>
    ));
  return (
    <div className="min-w-0">
      {blocks?.length ? (
        <div
          aria-label="Formatted item"
          className="max-h-[32rem] overflow-auto break-words"
        >
          <ItemPresentationBlocks
            blocks={blocks}
            Text={Text}
            media={(path, type) => {
              const link = assets.find(
                (entry) => entry.path === path && entry.media_type === type,
              );
              return link ? (
                asset(link)
              ) : (
                <p className="text-xs">
                  {path} ({type}) · Attachment unavailable
                </p>
              );
            }}
          />
        </div>
      ) : (
        <RawItemContent text={text} />
      )}
      <ItemAttachments
        assets={assets.filter(
          (link) =>
            !referenced.has(JSON.stringify([link.path, link.media_type])),
        )}
        renderAsset={asset}
      />
      {Boolean(blocks?.length) && <RawItemContent text={text} collapsed />}
    </div>
  );
}
