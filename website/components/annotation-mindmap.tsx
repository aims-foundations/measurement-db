import type {
  Agreement,
  Band,
  Leaf,
  LeafStatus,
} from "@/content/annotations";
import { leafStatusLabel } from "@/content/annotations";

/* ============================================================
   AnnotationMindmap — every coded item as a two-sided tree with
   the benchmark at the centre.

   Layout is COMPUTED, not authored: no coordinate appears in the
   data. Each leaf carries its verbatim question, its status and
   detail, and every selection tally, so leaf blocks differ in
   height; a band sits at the mean anchor of its own leaves and
   the root at the middle of the canvas. Connectors are plain
   elbows rather than curves.

   SVG <text> does not wrap, so lines are broken here at build
   time from a character-width estimate and emitted as <tspan>s.
   The estimate is deliberately conservative: over-wrapping costs
   a little height, under-wrapping would overflow the column.

   Server component — the page is static, so this runs once per
   build.
   ============================================================ */

const W = 1760;
const CENTER = W / 2;
const ROOT_HALF = 95;
const BAND_HALF = 78;
const BAND_X = { left: 610, right: W - 610 };
const MARK_X = { left: 420, right: W - 420 };
const TEXT_X = { left: 402, right: W - 402 };
const COL_W = 392;
const PAD = 30;

// Line heights and the character-width estimate used to break each run of text.
const LINE = { q: 18, d: 15, v: 15 };
const CHARS = {
  q: Math.floor(COL_W / 7.0), // 13.5px semibold
  d: Math.floor(COL_W / 5.75), // 11.5px regular
  v: Math.floor(COL_W / 6.1), // 10.5px mono + label
};
const LEAF_GAP = 26;
const BAND_GAP = 40;

type Side = "left" | "right";

const SIDE_OF: Record<string, Side> = {
  durability: "right",
  validity: "right",
  integrity: "left",
  neutrality: "left",
  verification: "left",
};

const MARK_FILL: Record<LeafStatus, string> = {
  ok: "#00917F",
  part: "#C27C00",
  gap: "#B8233A",
  risk: "#3B5FC4",
  desc: "#8A949C",
};

/** Greedy wrap on spaces. A single word longer than the limit is left intact
 *  rather than hyphenated — the column has slack and codes must stay verbatim. */
function wrap(text: string, maxChars: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines.length ? lines : [""];
}

type Block = {
  key: string;
  status: LeafStatus;
  question: string[];
  detail: string[];
  votes: { tally: string; lines: string[] }[];
  /** Per-question agreement line, or null when the question is uncoded. */
  agree: string | null;
  /** y of the status mark and the first question line. */
  anchor: number;
  height: number;
};

function buildBlock(band: Band, leaf: Leaf): Block {
  const question = wrap(leaf.question, CHARS.q);
  const detail = wrap(
    `${leaf.status === "desc" ? "Descriptive" : leafStatusLabel[leaf.status]} — ${leaf.detail}`,
    CHARS.d,
  );
  const votes = (leaf.votes ?? []).map((vote) => ({
    tally: `${vote.k}/${vote.n}`,
    lines: wrap(vote.code, CHARS.v - 5),
  }));
  const voteLines = votes.reduce((sum, v) => sum + v.lines.length, 0);
  const agree =
    leaf.qaa != null && leaf.qraa != null
      ? `QAA ${leaf.qaa.toFixed(3)}  ·  QRAA ${leaf.qraa.toFixed(3)}${
          leaf.kind === "single" ? `  (1 of ${leaf.options})` : ""
        }`
      : null;
  const height =
    question.length * LINE.q +
    6 +
    detail.length * LINE.d +
    (votes.length ? 6 + voteLines * LINE.v : 0) +
    (agree ? 6 + LINE.v : 0);
  return {
    key: `${band.id}-${leaf.name}`,
    status: leaf.status,
    question,
    detail,
    votes,
    agree,
    anchor: 0,
    height,
  };
}

type PlacedBand = {
  id: string;
  label: string;
  index: number;
  side: Side;
  y: number;
  blocks: Block[];
};

function placeSide(
  entries: { band: Band; index: number }[],
  side: Side,
  height: number,
): PlacedBand[] {
  let y = 0;
  const placed = entries.map(({ band, index }) => {
    const blocks = band.leaves.map((leaf) => {
      const block = buildBlock(band, leaf);
      block.anchor = y + 10;
      y += block.height + LEAF_GAP;
      return block;
    });
    y += BAND_GAP - LEAF_GAP;
    const mid = blocks.length
      ? (blocks[0].anchor + blocks[blocks.length - 1].anchor) / 2
      : y;
    return { id: band.id, label: band.label, index, side, y: mid, blocks };
  });
  const used = y - (BAND_GAP - LEAF_GAP);
  const offset = PAD + (height - 2 * PAD - used) / 2;
  for (const band of placed) {
    band.y += offset;
    for (const block of band.blocks) block.anchor += offset;
  }
  return placed;
}

function sideHeight(entries: { band: Band; index: number }[]): number {
  let total = 0;
  for (const { band } of entries) {
    for (const leaf of band.leaves) {
      total += buildBlock(band, leaf).height + LEAF_GAP;
    }
    total += BAND_GAP - LEAF_GAP;
  }
  return total - (BAND_GAP - LEAF_GAP);
}

function elbow(x1: number, y1: number, x2: number, y2: number): string {
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} H${mx} V${y2} H${x2}`;
}

function StatusMark({
  status,
  x,
  y,
}: {
  status: LeafStatus;
  x: number;
  y: number;
}) {
  const fill = MARK_FILL[status];
  if (status === "desc") return <circle cx={x} cy={y} r={4} fill={fill} />;
  if (status === "risk")
    return (
      <rect
        x={x - 5}
        y={y - 5}
        width={10}
        height={10}
        fill={fill}
        transform={`rotate(45 ${x} ${y})`}
      />
    );
  return (
    <>
      <rect
        x={x - 5.5}
        y={y - 5.5}
        width={11}
        height={11}
        rx={1.5}
        fill={status === "ok" ? fill : "transparent"}
        stroke={fill}
        strokeWidth={1.75}
      />
      {status === "part" ? (
        <rect x={x - 5.5} y={y - 5.5} width={5.5} height={11} fill={fill} />
      ) : null}
    </>
  );
}

/** The key, rendered above the plot as HTML so it wraps on narrow screens. */
function Key() {
  const entries: { status: LeafStatus; label: string; gloss: string }[] = [
    { status: "ok", label: "Documented", gloss: "public evidence answers it" },
    { status: "part", label: "Partial", gloss: "some evidence, with a gap" },
    { status: "gap", label: "Not located", gloss: "no evidence found" },
    {
      status: "risk",
      label: "Disclosed risk",
      gloss: "a stated conflict or unequal access",
    },
    {
      status: "desc",
      label: "Descriptive",
      gloss: "an unordered category, no status",
    },
  ];
  const terms = [
    [
      "QAA",
      "question annotator agreement — mean majority share over this question's options",
    ],
    [
      "QRAA",
      "the same question answered at random — 0.750 for any checkbox question, lower for single-select with more options",
    ],
    ["BAA", "benchmark annotator agreement — the QAA figures pooled across the whole instrument"],
    ["RAA", "the chance baseline pooled the same way"],
  ];
  return (
    <div className="mt-8 flex flex-col gap-4">
      <div className="flex flex-wrap gap-x-7 gap-y-2.5">
        {entries.map((entry) => (
          <span
            key={entry.status}
            className="flex items-baseline gap-2 text-sm font-light"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
              <StatusMark status={entry.status} x={6} y={6} />
            </svg>
            <span className="rd-mono !text-[0.62rem] uppercase tracking-wider">
              {entry.label}
            </span>
            <span className="opacity-60">{entry.gloss}</span>
          </span>
        ))}
      </div>
      <div className="flex flex-col gap-1.5 border-t border-black/10 pt-3.5">
        {terms.map(([abbr, gloss]) => (
          <span
            key={abbr}
            className="flex items-baseline gap-2.5 text-sm font-light"
          >
            <span className="rd-mono !text-[0.62rem] w-12 shrink-0">
              {abbr}
            </span>
            <span className="opacity-60">{gloss}</span>
          </span>
        ))}
        <span className="mt-1 text-sm font-light opacity-55">
          A score at or near its random baseline means the coders were no more
          consistent than chance, not that the benchmark did badly.
        </span>
      </div>
    </div>
  );
}

export function AnnotationMindmap({
  benchmarkName,
  bands,
  agreement,
}: {
  benchmarkName: string;
  bands: readonly Band[];
  agreement?: Agreement;
}) {
  if (bands.length === 0) return null;

  const indexed = bands.map((band, index) => ({ band, index }));
  const right = indexed.filter(
    ({ band }) => (SIDE_OF[band.id] ?? "right") === "right",
  );
  // Reversed so the left column reads upward from the lowest-numbered band.
  const left = indexed
    .filter(({ band }) => SIDE_OF[band.id] === "left")
    .reverse();

  const H = Math.max(sideHeight(right), sideHeight(left)) + PAD * 2;
  const placed = [
    ...placeSide(right, "right", H),
    ...placeSide(left, "left", H),
  ];
  const rootY = H / 2;

  return (
    <>
      <Key />
      <figure className="mt-7 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full min-w-[72rem]"
          role="img"
          aria-label={`${benchmarkName}: five codebook parts branching into their coded items, each with its question, status and per-option selection tallies.`}
        >
          {placed.map((band) => {
            const bx = BAND_X[band.side];
            const rootEdge =
              band.side === "right" ? CENTER + ROOT_HALF : CENTER - ROOT_HALF;
            const inner = band.side === "right" ? bx - BAND_HALF : bx + BAND_HALF;
            const outer = band.side === "right" ? bx + BAND_HALF : bx - BAND_HALF;
            return (
              <g key={`edges-${band.id}`} stroke="currentColor" strokeOpacity={0.22} strokeWidth={1.25} fill="none">
                <path d={elbow(rootEdge, rootY, inner, band.y)} />
                {band.blocks.map((block) => (
                  <path
                    key={`edge-${block.key}`}
                    d={elbow(outer, band.y, MARK_X[band.side], block.anchor)}
                  />
                ))}
              </g>
            );
          })}

          <g>
            <rect
              x={CENTER - ROOT_HALF}
              y={rootY - 20}
              width={ROOT_HALF * 2}
              height={40}
              rx={20}
              fill="currentColor"
            />
            <text
              x={CENTER}
              y={rootY + 6}
              textAnchor="middle"
              className="fill-[var(--rd-sage)] text-[15px] font-semibold"
            >
              {benchmarkName}
            </text>
            {/* Instrument-wide agreement, split because multi- and
                single-select have different chance baselines. */}
            {agreement ? (
              <text
                textAnchor="middle"
                className="rd-mono fill-current text-[10px] tabular-nums"
                opacity={0.62}
              >
                <tspan x={CENTER} y={rootY + 44}>
                  RAA {agreement.multiRandom.toFixed(3)} multi ·{" "}
                  {agreement.singleRandom.toFixed(3)} single
                </tspan>
                <tspan x={CENTER} y={rootY + 60}>
                  BAA {agreement.multiObserved.toFixed(3)} multi ·{" "}
                  {agreement.singleObserved.toFixed(3)} single
                </tspan>
                <tspan x={CENTER} y={rootY + 76} opacity={0.75}>
                  {agreement.multiBoxes} boxes · {agreement.singleCount}{" "}
                  single-select
                </tspan>
              </text>
            ) : null}
          </g>

          {placed.map((band) => (
            <g key={`band-${band.id}`}>
              <rect
                x={BAND_X[band.side] - BAND_HALF}
                y={band.y - 13}
                width={BAND_HALF * 2}
                height={26}
                rx={13}
                fill="currentColor"
                fillOpacity={0.07}
                stroke="currentColor"
                strokeOpacity={0.2}
              />
              <text
                x={BAND_X[band.side]}
                y={band.y + 4}
                textAnchor="middle"
                className="rd-mono fill-current text-[10.5px] uppercase"
                style={{ letterSpacing: "0.07em" }}
              >
                {band.index + 1} · {band.label}
              </text>
            </g>
          ))}

          {placed.flatMap((band) =>
            band.blocks.map((block) => {
              const tx = TEXT_X[band.side];
              const anchor = band.side === "right" ? "start" : "end";
              let y = block.anchor;
              const qy = y + 4;
              y += block.question.length * LINE.q + 6;
              const dy = y;
              y += block.detail.length * LINE.d + 6;
              const vy = y;
              return (
                <g key={`leaf-${block.key}`}>
                  <StatusMark
                    status={block.status}
                    x={MARK_X[band.side]}
                    y={block.anchor}
                  />
                  <text
                    x={tx}
                    textAnchor={anchor}
                    className="fill-current text-[13.5px] font-semibold"
                  >
                    {block.question.map((line, i) => (
                      <tspan key={line + i} x={tx} y={qy + i * LINE.q}>
                        {line}
                      </tspan>
                    ))}
                  </text>
                  <text
                    x={tx}
                    textAnchor={anchor}
                    className="fill-current text-[11.5px]"
                    opacity={0.68}
                  >
                    {block.detail.map((line, i) => (
                      <tspan key={line + i} x={tx} y={dy + i * LINE.d}>
                        {line}
                      </tspan>
                    ))}
                  </text>
                  {block.votes.length > 0
                    ? (() => {
                        let row = 0;
                        return block.votes.map((vote) => {
                          const first = row;
                          row += vote.lines.length;
                          return (
                            <text
                              key={vote.tally + vote.lines[0]}
                              x={tx}
                              textAnchor={anchor}
                              className="fill-current text-[10.5px]"
                              opacity={0.6}
                            >
                              {vote.lines.map((line, i) => (
                                <tspan
                                  key={line + i}
                                  x={tx}
                                  y={vy + (first + i) * LINE.v}
                                >
                                  {i === 0 ? `${vote.tally}  ${line}` : line}
                                </tspan>
                              ))}
                            </text>
                          );
                        });
                      })()
                    : null}
                  {block.agree ? (
                    <text
                      x={tx}
                      y={
                        vy +
                        block.votes.reduce((s, v) => s + v.lines.length, 0) *
                          LINE.v +
                        6
                      }
                      textAnchor={anchor}
                      className="rd-mono fill-current text-[9.5px] tabular-nums"
                      opacity={0.5}
                    >
                      {block.agree}
                    </text>
                  ) : null}
                </g>
              );
            }),
          )}
        </svg>
      </figure>
    </>
  );
}
