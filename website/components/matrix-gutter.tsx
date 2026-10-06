"use client";

import { useEffect, useRef } from "react";

/* ============================================================
   MatrixGutter — a numeric column flanking the response matrix,
   one value per matrix ROW (= one per subject).

   Two of them frame the matrix: the fitted ability θ on the left
   and the subject's observed accuracy on the right, both in the
   matrix's own row order (weakest subject at the top).

   Rows can be 2px tall on a 260-subject benchmark, so the column
   is drawn as a canvas ONE PIXEL PER ROW and stretched to the
   matrix height — the same trick the joined matrix uses for its
   strips. That keeps it aligned with the PNG at any row count,
   with no per-row DOM. Numbers are overlaid only once rows are
   tall enough to read (TEXT_MIN_ROW), which caps the overlay at
   a few dozen nodes.
   ============================================================ */

const TEXT_MIN_ROW = 9; // px — below this a number is unreadable, bars only
const BAR = "rgba(0,0,0,0.24)"; // reads as a quiet margin, not a second matrix
const AXIS = "rgba(0,0,0,0.14)";

/** The q-th quantile of |values|, ignoring nulls. 0 for an empty column. */
function quantileAbs(values: (number | null)[], q: number): number {
  const mags = values
    .filter((v): v is number => v != null)
    .map(Math.abs)
    .sort((a, b) => a - b);
  if (!mags.length) return 0;
  return mags[Math.min(mags.length - 1, Math.floor(q * (mags.length - 1)))];
}

/** Item difficulty z, drawn along the matrix's COLUMNS.
 *
 *  The transpose of MatrixGutter, and it stays a separate component because the
 *  axis it runs along behaves differently: a matrix has tens of rows and
 *  thousands of columns, so a per-column number is never legible and the strip
 *  is bars only — the exact z is read from the hover tooltip instead.
 *
 *  Columns are ordered hardest-first (generate_benchmark_gallery sorts by ascending solve
 *  rate) and z rises with difficulty, so this reads as a descending ramp on
 *  essentially every benchmark. What it shows is the SHAPE of that ramp: a long
 *  flat tail means a benchmark whose items are mostly interchangeable, a steep
 *  drop means difficulty is concentrated in a few items. */
export function MatrixColumnStrip({
  values,
  width,
  height,
}: {
  /** One value per matrix column, left to right. Null = no fitted z. */
  values: (number | null)[];
  width: number;
  height: number;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const n = values.length;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !n) return;
    canvas.width = n; // one canvas pixel column per matrix column
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, n, height);

    // The zero line is meaningful here and not just decoration: theta is
    // mean-centred, so z = 0 is the difficulty at which the AVERAGE subject has
    // an even chance. Items below the line are ones the average model handles;
    // items above it are ones it does not.
    const mid = Math.round(height / 2);
    ctx.fillStyle = AXIS;
    ctx.fillRect(0, mid, n, 1);

    // 90th percentile, not 95th: z has long tails (an item nobody solved has no
    // finite difficulty) and scaling to them flattens the typical item to two
    // or three pixels — the part of the curve actually worth reading.
    const span = Math.max(1e-6, quantileAbs(values, 0.9));
    const half = mid - 1;
    ctx.fillStyle = BAR;
    values.forEach((v, i) => {
      if (v == null) return;
      const len = Math.max(
        -half,
        Math.min(half, Math.round((v / span) * half)),
      );
      // Drawn DOWNWARD for positive z: the strip sits under the matrix, so a
      // harder item growing toward the reader matches the eye's expectation.
      if (len >= 0) ctx.fillRect(i, mid, 1, Math.max(1, len));
      else ctx.fillRect(i, mid + len, 1, Math.max(1, -len));
    });
  }, [values, height, n]);

  if (!n) return null;
  return (
    <div className="mt-1 flex items-center gap-1.5" style={{ width }}>
      <canvas
        ref={ref}
        aria-label="Item difficulty (z), per item, hardest first"
        className="block"
        style={{
          width: Math.max(0, width - 14),
          height,
          imageRendering: "auto",
        }}
      />
      <span className="shrink-0 text-[0.625rem] leading-none text-[var(--muted)]">
        z
      </span>
    </div>
  );
}

export type MatrixGutterProps = {
  /** One value per matrix row, top to bottom. Null = not available for that row. */
  values: (number | null)[];
  /** Column width in px. */
  width: number;
  /** Height in px — must be the rendered matrix height, or rows drift apart. */
  height: number;
  /** "diverging" draws from a centre line (θ, which is signed and centred at 0);
   *  "fill" draws from the left edge (a rate in [0,1]). */
  scale: "diverging" | "fill";
  format: (v: number) => string;
  /** Which side of the matrix this sits on — the numbers hug the matrix. */
  side: "left" | "right";
  /** Column heading, shown above the bars. */
  title: string;
  /** Full-word heading for screen readers and the title attribute. */
  label: string;
  /** Row currently under the cursor, highlighted to tie the value to the cell. */
  hoverRow?: number | null;
};

export function MatrixGutter({
  values,
  width,
  height,
  scale,
  format,
  side,
  title,
  label,
  hoverRow,
}: MatrixGutterProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const n = values.length;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !n) return;
    canvas.width = width;
    canvas.height = n; // one canvas pixel row per matrix row
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, n);

    // θ is signed, so it needs a zero line to read against; a rate does not.
    const mid = Math.round(width / 2);
    if (scale === "diverging") {
      ctx.fillStyle = AXIS;
      ctx.fillRect(mid, 0, 1, n);
    }

    // A shared denominator per column: bar lengths are comparable ACROSS rows,
    // which is the only comparison this column is for.
    //
    // For θ that denominator is the 95th percentile of |θ|, not the max. A
    // subject that answered every item correctly has no finite ability — the
    // fit only walks its θ out until the logit saturates, landing at 10 or more
    // — and scaling to that one row would squash every genuinely-measured
    // ability into a couple of pixels. Bars past the percentile clip to the
    // column edge, which reads as "off the scale", which is what they are.
    const span =
      scale === "diverging" ? Math.max(1e-6, quantileAbs(values, 0.95)) : 1;

    ctx.fillStyle = BAR;
    values.forEach((v, i) => {
      if (v == null) return;
      if (scale === "diverging") {
        const half = mid - 1;
        const len = Math.max(
          -half,
          Math.min(half, Math.round((v / span) * half)),
        );
        // A bar is never shorter than 1px: a near-zero θ should still read as a
        // drawn value rather than as a missing row.
        if (len >= 0) ctx.fillRect(mid, i, Math.max(1, len), 1);
        else ctx.fillRect(mid + len, i, Math.max(1, -len), 1);
      } else {
        ctx.fillRect(0, i, Math.max(1, Math.round(v * (width - 1))), 1);
      }
    });
  }, [values, width, height, scale, n]);

  if (!n) return null;
  const rowH = height / n;
  const showText = rowH >= TEXT_MIN_ROW;
  const fontPx = Math.min(10, Math.floor(rowH) - 1);

  return (
    <div className="shrink-0" style={{ width }}>
      <div
        className={`mb-1 h-3.5 whitespace-nowrap text-[0.625rem] leading-none text-[var(--muted)] ${
          side === "left" ? "text-right" : "text-left"
        }`}
        title={label}
      >
        {title}
      </div>
      <div className="relative" style={{ width, height }}>
        <canvas
          ref={ref}
          aria-label={label}
          className="block"
          style={{ width, height, imageRendering: "pixelated" }}
        />
        {showText
          ? values.map((v, i) =>
              v == null ? null : (
                <span
                  key={i}
                  className={`pointer-events-none absolute flex items-center tabular-nums text-[var(--ink)] ${
                    side === "left" ? "justify-end" : "justify-start"
                  }`}
                  style={{
                    top: i * rowH,
                    height: rowH,
                    left: 0,
                    width,
                    fontSize: fontPx,
                    lineHeight: 1,
                    // The bars sit behind the numbers, so the numbers need a
                    // hairline of breathing room from the matrix edge.
                    paddingRight: side === "left" ? 3 : 0,
                    paddingLeft: side === "right" ? 3 : 0,
                  }}
                >
                  {format(v)}
                </span>
              ),
            )
          : null}
        {hoverRow != null && hoverRow >= 0 && hoverRow < n ? (
          <div
            className="pointer-events-none absolute left-0 bg-[var(--ink)]/10"
            style={{ top: hoverRow * rowH, height: Math.max(1, rowH), width }}
          />
        ) : null}
      </div>
    </div>
  );
}
