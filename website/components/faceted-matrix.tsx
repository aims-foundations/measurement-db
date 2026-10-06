"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  observationKey,
  type JoinedPick,
  type CellKeys,
} from "./joined-matrix";

// Draw the faceted_matrix layout supplied with the page.

const PASS: [number, number, number] = [31, 119, 180];
const FAIL: [number, number, number] = [214, 39, 40];
const NONE: [number, number, number] = [233, 232, 228];

function cellColor(ch: string): [number, number, number] {
  if (ch === ".") return NONE;
  return ch === "1" ? PASS : FAIL;
}

type Panel = { key: string; label: string; colIds: string[] };

type FacetRow = {
  sid: string;
  name: string;
  band: string;
  panels: Record<string, CellKeys & { bits: string; cond: string | null }>;
  pass: number;
  n: number;
};

type Band = { key: string; panels: Panel[]; rows: FacetRow[] };

export type FacetedData = {
  rowKind?: string;
  bandDim: string;
  bandLabels?: Record<string, Record<string, string>>;
  bands: Band[];
};

type Props = {
  data: FacetedData;
  onPick: (p: JoinedPick) => void;
  /** subject_id -> fitted Rasch ability. A leading θ column when present, the
   *  same as the joined view. */
  theta?: Record<string, number> | null;
  /** What 1 and 0 MEAN when they are not pass/fail — see JoinedMatrix. */
  binaryLabels?: { zero: string; one: string } | null;
};

/** A band's heading. `bandLabels` is keyed by dim then value, the same map the
 *  joined view uses, so "task=dermatology" reads as the phrase someone wrote
 *  rather than the raw value. */
function bandHeading(
  band: string,
  labels: Record<string, Record<string, string>> | undefined,
): string {
  const i = band.indexOf("=");
  if (i < 0) return band;
  const k = band.slice(0, i);
  const v = band.slice(i + 1);
  return labels?.[k]?.[v] ?? v;
}

/** One row of one panel, painted as a 1px-tall canvas scaled with
 *  image-rendering: pixelated — the same trick the PNG matrices and the joined
 *  view use, so a click knows its cell without sampling a pixel. */
function Strip({
  bits,
  onHover,
  onLeave,
  onPick,
  label,
}: {
  bits: string;
  onHover: (idx: number, e: React.MouseEvent) => void;
  onLeave: () => void;
  onPick: (idx: number) => void;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const n = bits.length;
    cv.width = n;
    cv.height = 1;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(n, 1);
    for (let i = 0; i < n; i++) {
      const c = cellColor(bits[i]);
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [bits]);

  function idxAt(e: React.MouseEvent) {
    const b = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return Math.max(
      0,
      Math.min(
        bits.length - 1,
        Math.floor(((e.clientX - b.left) / b.width) * bits.length),
      ),
    );
  }

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      tabIndex={0}
      className="block h-[15px] w-full cursor-crosshair rounded-[1px]"
      style={{ imageRendering: "pixelated" }}
      onMouseMove={(e) => onHover(idxAt(e), e)}
      onMouseLeave={onLeave}
      onClick={(e) => onPick(idxAt(e))}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onPick(0);
        }
      }}
    />
  );
}

export function FacetedMatrix({ data, onPick, theta, binaryLabels }: Props) {
  const [tip, setTip] = useState<{
    x: number;
    y: number;
    text: string;
    passed: boolean;
    observed: boolean;
  } | null>(null);

  const totals = useMemo(() => {
    if (!data) return { rows: 0, cells: 0, observed: 0 };
    let rows = 0;
    let cells = 0;
    let observed = 0;
    for (const b of data.bands) {
      rows += b.rows.length;
      for (const r of b.rows) {
        for (const p of b.panels) {
          const rp = r.panels[p.key];
          cells += p.colIds.length;
          if (rp) {
            for (const ch of rp.bits) if (ch !== ".") observed++;
          }
        }
      }
    }
    return { rows, cells, observed };
  }, [data]);

  if (!data || !data.bands?.length) return null;

  const one = binaryLabels?.one ?? "passed";
  const zero = binaryLabels?.zero ?? "failed";

  return (
    <div className="relative">
      <p className="mb-2 text-[0.8125rem] text-[var(--muted)]">
        {totals.rows.toLocaleString()} subjects in {data.bands.length} groups,
        each with its own items. {totals.observed.toLocaleString()} of{" "}
        {totals.cells.toLocaleString()} cells measured (
        {((100 * totals.observed) / (totals.cells || 1)).toFixed(1)}%) — pale
        cells were never measured, not failures.
      </p>

      {/* The figure scrolls in its own box rather than down the page, so a
          clicked cell and the detail panel below stay on screen together. */}
      <div className="max-h-[70vh] overflow-auto rounded-md border border-[var(--line)] bg-white px-4 pb-4 pt-4">
        <div className="min-w-[640px]">
          {data.bands.map((band) => {
            // Per BAND, which is the whole point: bands differ in width (24
            // items vs 32) and in panel count, so one shared template cannot
            // serve them.
            // 300px, not the joined view's 170px: these row labels carry the
            // subject AND the arm it was measured under, and the arm is the
            // thing that decides which panels a row can fill — truncating it
            // away leaves two visually identical rows with no way to tell which
            // is which.
            const grid = `${theta ? "40px " : ""}300px ${band.panels
              .map((p) => `minmax(60px, ${p.colIds.length}fr)`)
              .join(" ")} 56px`;
            const heading = bandHeading(
              `${data.bandDim}=${band.key}`,
              data.bandLabels,
            );
            return (
              <div key={band.key} className="mb-6 last:mb-0">
                <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 border-b border-[var(--line)] pb-1 text-[0.8125rem] font-semibold text-[var(--ink)]">
                  {heading}
                  <span className="font-normal text-[var(--muted)] tabular-nums">
                    {band.rows.length.toLocaleString()} subjects ×{" "}
                    {band.panels[0]?.colIds.length ?? 0} items
                  </span>
                </div>

                {/* Column headings repeat per band. They have to: the panels
                    and the items beneath them are this band's, not the
                    figure's. */}
                <div
                  className="grid items-end gap-x-2.5 pb-1"
                  style={{ gridTemplateColumns: grid }}
                >
                  {theta ? (
                    <div
                      className="text-right text-[0.6875rem] text-[var(--muted)]"
                      title="Fitted ability (theta), per subject"
                    >
                      θ
                    </div>
                  ) : null}
                  <div />
                  {band.panels.map((p) => (
                    <div
                      key={p.key}
                      /* min-w-0 lets a heading wrap inside its own track
                         instead of overlapping its neighbour. */
                      className="min-w-0 break-words text-[0.6875rem] text-[var(--ink)]"
                    >
                      {p.label}
                    </div>
                  ))}
                  <div />
                </div>

                {band.rows.map((row) => (
                  <div
                    key={row.sid}
                    className="grid items-center gap-x-2.5 py-[1.5px]"
                    style={{ gridTemplateColumns: grid }}
                  >
                    {theta ? (
                      <div className="text-right text-[0.6875rem] tabular-nums text-[var(--muted)]">
                        {theta[row.sid] != null
                          ? theta[row.sid].toFixed(2)
                          : ""}
                      </div>
                    ) : null}
                    <div
                      title={row.name}
                      className="truncate text-right text-[0.6875rem] text-[var(--ink)]"
                    >
                      {row.name}
                    </div>
                    {band.panels.map((p) => {
                      const rp = row.panels[p.key];
                      // A missing panel is drawn as a full run of unobserved
                      // rather than a "not run" chip: for haiid that is 280 of
                      // 558 rows in each post-advice panel, and 280 chips would
                      // shout where the pale band should stay quiet.
                      const bits = rp?.bits ?? ".".repeat(p.colIds.length);
                      return (
                        <Strip
                          key={p.key}
                          bits={bits}
                          label={`${row.name}, ${heading}, ${p.label}: ${row.pass} of ${row.n}`}
                          onHover={(idx, e) => {
                            const ch = bits[idx];
                            setTip({
                              x: e.clientX,
                              y: e.clientY,
                              text: `${row.name} · ${p.label} · item ${idx + 1}`,
                              passed: ch === "1",
                              observed: ch !== ".",
                            });
                          }}
                          onLeave={() => setTip(null)}
                          onPick={(idx) => {
                            const ch = bits[idx];
                            if (ch === "." || !rp) return;
                            const itemId = p.colIds[idx];
                            if (!itemId) return;
                            onPick({
                              keyRef: rp.keyRef,
                              keyIndex: idx,
                              key: observationKey(rp, idx, itemId),
                              subjectId: row.sid,
                              itemId,
                              cond: rp.cond,
                              trial: "1",
                              passed: ch === "1",
                            });
                          }}
                        />
                      );
                    })}
                    <div className="text-[0.6875rem] tabular-nums text-[var(--muted)]">
                      {row.n ? `${Math.round((100 * row.pass) / row.n)}%` : ""}
                    </div>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {tip ? (
        <div
          className="pointer-events-none fixed z-50 rounded border border-[var(--line)] bg-white px-2 py-1 text-[0.6875rem] text-[var(--ink)] shadow-sm"
          style={{ left: tip.x + 12, top: tip.y + 12 }}
        >
          {tip.text}
          <span className="ml-1 text-[var(--muted)]">
            — {tip.observed ? (tip.passed ? one : zero) : "never measured"}
          </span>
        </div>
      ) : null}
    </div>
  );
}
