"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  observationKey,
  type JoinedPick,
  type CellKeys,
} from "./joined-matrix";

// Draw the session_matrix layout supplied with the page.

/** Matches generate_benchmark_gallery's PASS/FAIL and the pin colours in matrix-viewer. */
const PASS = "#1f77b4";
const FAIL = "#d62728";

const CELL = 6; // px, the drawn square
const PITCH = 7; // cell + 1px gutter
const SESSION_GAP = 9; // the break that reads as "new session"
const LINE = 12; // wrapped-line pitch
// Insets that keep the selection highlight inside the canvas. The caret rises
// 8.4px above a cell and carries a 2.5px halo stroke; the halo ring extends
// 3.1px past the cell on every side.
const TOP_PAD = 11;
const BOTTOM_PAD = 5;
const PAD_X = 4;

type SessionRow = {
  sid: string;
  band: string;
  name: string;
  blocks: Record<string, CellKeys & { bits: string; cond: string | null }>;
  cols: string[];
  segs: number[];
  segLabels: string[];
  subjects: string[];
  pass: number;
  n: number;
};

export type SessionData = {
  bandDims: string[];
  bandLabels?: Record<string, Record<string, string>>;
  blocks: { key: string; colIds: string[] }[];
  trials: Record<string, SessionRow[]>;
  rowKind?: string;
};

type Placed = { x: number; y: number; i: number; seg: number };

/** Lay a row's cells into wrapped lines, keeping each session whole when it
 *  fits. A session longer than the full width wraps inside itself — the only
 *  case where a block breaks, and unavoidable at 521 turns. */
function layout(segs: number[], width: number) {
  const cells: Placed[] = [];
  // The selection highlight is drawn OUTSIDE the cell it marks, so the track is
  // inset on every side to keep the halo ring and the caret inside the canvas —
  // otherwise the first and last cell of a line get a clipped ring, and a
  // top-line caret is cut off entirely.
  const right = width - PAD_X;
  let x = PAD_X;
  let y = TOP_PAD;
  let i = 0;
  segs.forEach((len, seg) => {
    const need = len * PITCH;
    const gap = x > PAD_X ? SESSION_GAP : 0;
    if (x + gap + need <= right) x += gap;
    else if (need <= right - PAD_X) {
      x = PAD_X;
      y += LINE;
    } else if (x > PAD_X) {
      x = PAD_X;
      y += LINE;
    }
    for (let k = 0; k < len; k++) {
      if (x + PITCH > right + 1) {
        x = PAD_X;
        y += LINE;
      }
      cells.push({ x, y, i, seg });
      i += 1;
      x += PITCH;
    }
  });
  return { cells, height: y + CELL + BOTTOM_PAD };
}

function bandLabel(
  band: string,
  dims: string[],
  labels: Record<string, Record<string, string>> | undefined,
) {
  const [dim, ...rest] = band.split("=");
  const value = rest.join("=");
  const hand = labels?.[dim]?.[value];
  if (hand) return hand;
  return dims.length === 1 ? value : `${dim} ${value}`;
}

type Props = {
  data: SessionData;
  onPick: (p: JoinedPick) => void;
  binaryLabels?: { zero: string; one: string } | null;
};

export function SessionStrip({ data, onPick, binaryLabels }: Props) {
  const [minCells, setMinCells] = useState(1);
  const [sel, setSel] = useState<{ row: string; i: number } | null>(null);
  const [tip, setTip] = useState<{
    x: number;
    y: number;
    text: string;
    passed: boolean;
  } | null>(null);

  const rows = useMemo(() => {
    const all = data?.trials?.["1"] ?? [];
    return all.filter((r) => r.n >= minCells);
  }, [data, minCells]);

  const oneLabel = binaryLabels?.one ?? "accepted";
  const zeroLabel = binaryLabels?.zero ?? "pushed back";

  if (!data || data.rowKind !== "session" || !rows.length) return null;

  // Bands keep the payload's row order (the renderer already sorted them).
  const bands: { key: string; rows: SessionRow[] }[] = [];
  for (const r of rows) {
    const last = bands[bands.length - 1];
    if (last && last.key === r.band) last.rows.push(r);
    else bands.push({ key: r.band, rows: [r] });
  }

  const shownCells = rows.reduce((s, r) => s + r.n, 0);
  const shownPass = rows.reduce((s, r) => s + r.pass, 0);

  return (
    <div className="mb-10">
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="flex items-center gap-1.5 text-xs text-[var(--muted)]">
          <i
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-[2px]"
            style={{ background: PASS }}
          />
          {oneLabel}
        </span>
        <span className="flex items-center gap-1.5 text-xs text-[var(--muted)]">
          <i
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-[2px]"
            style={{ background: FAIL }}
          />
          {zeroLabel}
        </span>
        <span className="text-xs text-[var(--muted)]">
          {rows.length.toLocaleString()} developers ·{" "}
          {shownCells.toLocaleString()} prompts ·{" "}
          {((shownPass / shownCells) * 100).toFixed(1)}% {oneLabel}
        </span>
        <label className="ml-auto flex items-center gap-2 text-xs text-[var(--muted)]">
          Min prompts
          <select
            className="rounded border border-[var(--line)] bg-white px-1.5 py-0.5 text-xs"
            onChange={(e) => setMinCells(Number(e.target.value))}
            value={minCells}
          >
            <option value={1}>all</option>
            <option value={10}>10+</option>
            <option value={50}>50+</option>
            <option value={200}>200+</option>
          </select>
        </label>
      </div>

      {/* Same cap the joined matrix uses: 182 developers is several screens of
          strips, and the cell detail this panel fills sits BELOW them — so the
          strips scroll in their own box rather than pushing the detail off the
          page. The band heading sticks to the top of the box (its own pt-4 is
          what rows pass under), so a strip is never read without knowing which
          scaffold it belongs to. */}
      <div className="max-h-[70vh] overflow-auto rounded-md border border-[var(--line)] bg-white px-4 pb-4">
        {bands.map((band) => (
          <section key={band.key}>
            <h4 className="sticky top-0 z-10 mb-1 border-b border-[var(--line)] bg-white pt-4 pb-1 text-xs font-semibold">
              {bandLabel(band.key, data.bandDims, data.bandLabels)}
              <span className="ml-2 font-normal text-[var(--muted)]">
                {band.rows.length} developer{band.rows.length === 1 ? "" : "s"}{" "}
                · {band.rows.reduce((s, r) => s + r.n, 0).toLocaleString()}{" "}
                prompts
              </span>
            </h4>
            {band.rows.map((row) => (
              <StripRow
                key={`${row.band}/${row.name}`}
                onLeave={() => setTip(null)}
                onPickCell={(i, passed) => {
                  setSel({ row: `${row.band}/${row.name}`, i });
                  onPick({
                    key: observationKey(row.blocks.all, i, row.cols[i]),
                    subjectId: row.sid,
                    itemId: row.cols[i],
                    cond: null,
                    trial: "1",
                    passed,
                  });
                }}
                onTip={setTip}
                row={row}
                selected={
                  sel && sel.row === `${row.band}/${row.name}` ? sel.i : null
                }
                zeroLabel={zeroLabel}
                oneLabel={oneLabel}
              />
            ))}
          </section>
        ))}
      </div>

      {tip ? (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded bg-[var(--fg)] px-2 py-1 text-[11px] leading-snug text-[var(--bg)] shadow"
          style={{ left: tip.x + 14, top: tip.y - 10 }}
        >
          {tip.text}
        </div>
      ) : null}
    </div>
  );
}

function StripRow({
  row,
  selected,
  onPickCell,
  onTip,
  onLeave,
  oneLabel,
  zeroLabel,
}: {
  row: SessionRow;
  selected: number | null;
  onPickCell: (i: number, passed: boolean) => void;
  onTip: (t: { x: number; y: number; text: string; passed: boolean }) => void;
  onLeave: () => void;
  oneLabel: string;
  zeroLabel: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const placed = useRef<Placed[]>([]);
  const bits = row.blocks.all?.bits ?? "";

  const draw = useCallback(() => {
    const cv = ref.current;
    if (!cv) return;
    const width = cv.clientWidth;
    if (!width) return;
    const { cells, height } = layout(row.segs, width);
    placed.current = cells;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    cv.style.height = `${height}px`;
    const g = cv.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    let cur = "";
    for (const c of cells) {
      const want = bits[c.i] === "1" ? PASS : FAIL;
      if (want !== cur) {
        g.fillStyle = want;
        cur = want;
      }
      g.fillRect(c.x, c.y, CELL, CELL);
    }
    if (selected != null) {
      const q = cells[selected];
      if (q) {
        // A thin dark ring is the wrong tool at this size: the cell is 6px with
        // a 1px gutter, and black has its LEAST contrast against exactly the two
        // colours the data uses. So the highlight is built from separation and a
        // pointer rather than from line weight.
        //
        //   1. a white halo ring detaches the cell from its neighbours — on a
        //      dense strip, empty space reads louder than a darker outline;
        //   2. a thin ink ring on top gives it a crisp edge against the halo;
        //   3. a caret above points at it, so the eye finds the cell without
        //      scanning — this is what survives when the strip is 300 cells wide.
        const cx = q.x + CELL / 2;
        g.lineJoin = "miter";
        g.strokeStyle = "#ffffff";
        g.lineWidth = 3;
        g.strokeRect(q.x - 2.5, q.y - 2.5, CELL + 5, CELL + 5);
        g.strokeStyle = "#0b0b0b";
        g.lineWidth = 1.25;
        g.strokeRect(q.x - 2.5, q.y - 2.5, CELL + 5, CELL + 5);

        // Caret sits in the gap above the cell (TOP_PAD on line 0, the
        // inter-line gap otherwise), haloed so it stays legible over a cell.
        const tipY = q.y - 4;
        const drawCaret = () => {
          g.beginPath();
          g.moveTo(cx, tipY);
          g.lineTo(cx - 3.2, tipY - 4.4);
          g.lineTo(cx + 3.2, tipY - 4.4);
          g.closePath();
        };
        drawCaret();
        g.strokeStyle = "#ffffff";
        g.lineWidth = 2.5;
        g.stroke();
        drawCaret();
        g.fillStyle = "#0b0b0b";
        g.fill();
      }
    }
  }, [bits, row.segs, selected]);

  useEffect(() => {
    draw();
    const ro = new ResizeObserver(() => draw());
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, [draw]);

  function hit(ev: React.MouseEvent<HTMLCanvasElement>) {
    const cv = ref.current;
    if (!cv) return null;
    const r = cv.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    for (const c of placed.current)
      if (
        mx >= c.x - 1 &&
        mx <= c.x + CELL + 1 &&
        my >= c.y - 1 &&
        my <= c.y + CELL + 1
      )
        return c;
    return null;
  }

  const rate = row.n ? Math.round((row.pass / row.n) * 100) : 0;

  return (
    <div className="flex items-start gap-3 py-[1px]">
      <div className="flex w-44 flex-none items-baseline gap-2 pt-px font-mono text-[11px]">
        <span className="flex-1 truncate" title={row.name}>
          {row.name}
        </span>
        <span className="tabular-nums text-[var(--muted)]">{row.n}</span>
        <span className="w-9 text-right tabular-nums text-[var(--muted)]">
          {rate}%
        </span>
      </div>
      <canvas
        aria-label={`${row.name}: ${row.n} prompts across ${row.segs.length} sessions, ${rate}% ${oneLabel}`}
        className="min-w-0 flex-1 cursor-crosshair"
        onClick={(ev) => {
          const c = hit(ev);
          if (c) onPickCell(c.i, bits[c.i] === "1");
        }}
        onMouseLeave={onLeave}
        onMouseMove={(ev) => {
          const c = hit(ev);
          if (!c) return onLeave();
          const passed = bits[c.i] === "1";
          const turnInSeg =
            c.i - row.segs.slice(0, c.seg).reduce((s, v) => s + v, 0) + 1;
          onTip({
            x: ev.clientX,
            y: ev.clientY,
            passed,
            text: `${row.name} · session ${row.segLabels[c.seg]} · turn ${turnInSeg} of ${row.segs[c.seg]} — ${passed ? oneLabel : zeroLabel}`,
          });
        }}
        ref={ref}
        role="img"
      />
    </div>
  );
}
