"use client";

import { useState, type ReactNode } from "react";
import type { AssetLink } from "@/lib/benchmark-records/types";
import { Pagination } from "./benchmark-records/pagination";

/** Large capture datasets must not download every attachment on selection. */
export function ItemAttachments({
  assets,
  renderAsset,
}: {
  assets: AssetLink[];
  renderAsset: (link: AssetLink) => ReactNode;
}) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  if (!assets.length) return null;
  const showImages =
    assets.length <= 4 &&
    assets.every(
      (link) =>
        ["input", "input_image"].includes(link.role) &&
        link.media_type.startsWith("image/"),
    );
  return (
    <section
      className="mt-4 border-t border-[var(--line)] pt-3"
      aria-label="Item attachments"
    >
      <h5 className="mb-2 text-xs font-medium">
        Attachments ({assets.length.toLocaleString()})
      </h5>
      {assets.slice(page * 10, page * 10 + 10).map((link) => (
        <div key={link.path} className="my-2 min-w-0 text-xs">
          <button
            type="button"
            className="break-all text-left text-[var(--lagunita)] underline"
            onClick={() =>
              setSelected(selected === link.path ? null : link.path)
            }
          >
            {link.path}
          </button>
          <span className="ml-2 text-[var(--muted)]">
            {link.role} · {link.media_type}
          </span>
          {(showImages || selected === link.path) && renderAsset(link)}
        </div>
      ))}
      {assets.length > 10 && (
        <Pagination
          label="attachments"
          page={page}
          size={10}
          total={assets.length}
          onChange={(next) => {
            setPage(next);
            setSelected(null);
          }}
        />
      )}
    </section>
  );
}
