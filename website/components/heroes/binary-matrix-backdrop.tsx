"use client";

import { useEffect, useMemo, useRef } from "react";

/* ============================================================
   BinaryMatrixBackdrop — interactive response-matrix watermark.

   Three cell states, drawn over a transparent grid:
     • correct   — HAI blue
     • incorrect — fig purple
     • unobserved — empty (page background shows through)

   Two stacked layers render the same per-cell state:
     1. Base   — faint baseline, always visible
     2. Highlight — bright + slightly scaled, masked by a radial
        gradient at the cursor. Cells "pop" as the spotlight
        passes over them.

   Pointer movement only updates two CSS custom properties
   (`--mx`, `--my`) on the container; cells themselves never
   re-render or restyle on every frame.
   ============================================================ */

const BACKDROP_ROWS = 30;
const BACKDROP_COLS = 96;

const BASE_OPACITY = 0.12;
const HIGHLIGHT_OPACITY = 0.95;
const HIGHLIGHT_RADIUS_PX = 50;
const HIGHLIGHT_CORE = 0.45; // share of radius at full opacity before fade
const HIGHLIGHT_SCALE = 1.4;
const HIGHLIGHT_LIFT_PX = 1;

// Trail: sample cursor position every TRAIL_INTERVAL_MS ms and keep the last
// TRAIL_LENGTH positions in a ring buffer. The mask is built from one radial
// gradient per slot; older slots have lower alpha. When the cursor stops or
// leaves, the slots roll over to the current/off-screen position and the tail
// fades out smoothly over TRAIL_LENGTH × TRAIL_INTERVAL_MS ms.
const TRAIL_LENGTH = 6;
const TRAIL_INTERVAL_MS = 45;
const TRAIL_ALPHAS = [1.0, 0.78, 0.6, 0.44, 0.3, 0.18];

const CORRECT_COLOR = "color-mix(in srgb, #005fa3 92%, white)";
const INCORRECT_COLOR = "color-mix(in srgb, #72307c 84%, white)";

type CellState = "correct" | "incorrect" | "unobserved";

function hash(r: number, c: number, salt: number): number {
  const s = Math.sin(r * 12.9898 + c * 78.233 + salt * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

function buildCellStates(rows: number, cols: number): CellState[] {
  const out: CellState[] = [];
  for (let r = 0; r < rows; r++) {
    const ability = 0.45 + ((rows - 1 - r) / (rows - 1)) * 0.45;
    for (let c = 0; c < cols; c++) {
      // Observation gate: roughly half of cells are observed.
      if (hash(r, c, 1) > 0.55) {
        out.push("unobserved");
        continue;
      }
      const difficulty = 0.3 + (c / (cols - 1)) * 0.55;
      const logit = (ability - difficulty) * 5;
      const p = 1 / (1 + Math.exp(-logit));
      out.push(hash(r, c, 2) < p ? "correct" : "incorrect");
    }
  }
  return out;
}

function BackdropGrid({
  variant,
  cellStates,
  cols,
}: {
  variant: "base" | "highlight";
  cellStates: CellState[];
  cols: number;
}) {
  const isHighlight = variant === "highlight";
  return (
    // Row tracks intentionally not declared. Each cell carries
    // `aspect-square`, so rows auto-size to (container_width / cols) and
    // cells stay genuinely square regardless of the hero's aspect ratio.
    // `place-items-center` on the parent vertically centres the grid in
    // the available height; the outer wrapper's `overflow-hidden` clips
    // any spill at the top/bottom edges, which gives a "matrix extends
    // beyond the viewport" feel.
    <div
      className="grid w-full gap-[5px] p-6"
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
      }}
    >
      {cellStates.map((state, i) => {
        if (state === "unobserved")
          return <div key={i} className="aspect-square" />;
        const color = state === "correct" ? CORRECT_COLOR : INCORRECT_COLOR;
        return (
          <div
            key={i}
            className="aspect-square rounded-[2px]"
            style={{
              backgroundColor: color,
              transform: isHighlight
                ? `translate3d(0, -${HIGHLIGHT_LIFT_PX}px, 0) scale(${HIGHLIGHT_SCALE})`
                : undefined,
            }}
          />
        );
      })}
    </div>
  );
}

export function BinaryMatrixBackdrop({
  className = "",
  rows = BACKDROP_ROWS,
  cols = BACKDROP_COLS,
}: {
  className?: string;
  rows?: number;
  cols?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cellStates = useMemo(() => buildCellStates(rows, cols), [rows, cols]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Touch devices have no meaningful hover spotlight, and reduced-motion
    // users should not pay for a continuously updating pointer trail. The base
    // matrix remains visible in both cases.
    if (
      window.matchMedia(
        "(hover: none), (pointer: coarse), (prefers-reduced-motion: reduce)",
      ).matches
    ) {
      return;
    }

    const history: { x: number; y: number }[] = Array.from(
      { length: TRAIL_LENGTH },
      () => ({ x: -9999, y: -9999 }),
    );
    let lastX = -9999;
    let lastY = -9999;

    const onMove = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect();
      lastX = e.clientX - rect.left;
      lastY = e.clientY - rect.top;
    };

    const onLeave = () => {
      lastX = -9999;
      lastY = -9999;
    };

    const tick = window.setInterval(() => {
      history.pop();
      history.unshift({ x: lastX, y: lastY });
      for (let i = 0; i < TRAIL_LENGTH; i++) {
        el.style.setProperty(`--mx${i}`, `${history[i]!.x}px`);
        el.style.setProperty(`--my${i}`, `${history[i]!.y}px`);
      }
    }, TRAIL_INTERVAL_MS);

    window.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("pointerleave", onLeave);

    return () => {
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerleave", onLeave);
      window.clearInterval(tick);
    };
  }, []);

  // One radial gradient per past cursor position, composited additively so
  // overlapping samples brighten and older samples fade. The mask updates
  // every TRAIL_INTERVAL_MS ms as new positions roll into the ring buffer.
  const maskValue = TRAIL_ALPHAS.map((alpha, i) => {
    return `radial-gradient(circle ${HIGHLIGHT_RADIUS_PX}px at var(--mx${i}, -9999px) var(--my${i}, -9999px), rgba(0,0,0,${alpha}) 0%, rgba(0,0,0,${alpha}) ${HIGHLIGHT_CORE * 100}%, transparent 100%)`;
  }).join(", ");

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`.trim()}
    >
      {/* Base layer — faint baseline, always visible */}
      <div
        className="absolute inset-0 flex items-center"
        style={{ opacity: BASE_OPACITY }}
      >
        <BackdropGrid variant="base" cellStates={cellStates} cols={cols} />
      </div>

      {/* Highlight layer — bright + scaled, masked to a soft circle.
          A single drop-shadow filter on the layer gives the cells their
          glow halo. Much cheaper than per-cell box-shadow: the browser
          runs one filter pass over the already-masked result (a handful
          of visible cells) instead of computing shadows for every cell. */}
      <div
        className="absolute inset-0 flex items-center"
        style={{
          opacity: HIGHLIGHT_OPACITY,
          WebkitMaskImage: maskValue,
          maskImage: maskValue,
          // Each trail layer in the mask adds to the previous one — overlap
          // brightens, decay accumulates. Without this the last gradient
          // would simply paint over (source-over) the older ones.
          WebkitMaskComposite: "source-over",
          maskComposite: "add",
          // Two stacked drop-shadows: a dark offset shadow gives elevation
          // (cells feel lifted off the page), a tinted glow gives the
          // luminous "emerging" quality.
          filter:
            "drop-shadow(0 2px 8px rgba(0, 0, 0, 0.20)) drop-shadow(0 0 6px color-mix(in srgb, #9cf2f2 60%, transparent))",
          willChange: "filter, mask-image",
        }}
      >
        <BackdropGrid variant="highlight" cellStates={cellStates} cols={cols} />
      </div>
    </div>
  );
}
