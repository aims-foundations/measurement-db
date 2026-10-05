"use client";

import Image from "next/image";
import "katex/dist/katex.min.css";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, {
  defaultUrlTransform,
  type Components,
} from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type {
  BenchmarkConditions,
  MatrixCategory,
  MatrixCondition,
} from "@/content/benchmark-details";
import type { BenchmarkIrt } from "@/content/benchmark-irt";
import { withBase } from "@/lib/base-path";
import { IrtPanel } from "./irt-panel";
import {
  JoinedMatrix,
  type JoinedPick,
  type JoinedData,
  type ObservationKey,
} from "./joined-matrix";
import { FacetedMatrix, type FacetedData } from "./faceted-matrix";
import { SessionStrip, type SessionData } from "./session-strip";
import { MatrixColumnStrip, MatrixGutter } from "./matrix-gutter";

/* ============================================================
   MatrixViewer — interactive response-matrix heatmap.

   • Fit-to-width ⇄ full-resolution (scroll) toggle.
   • Condition dropdowns (e.g. attack / category / judge): when a
     benchmark stores multiple binary results per (subject, item)
     cell, each condition is its own BINARY matrix and the
     dropdowns (left of "Actual size") switch between them. Each
     condition is sorted independently (rows by its own row mean,
     columns by its own column mean, both over that slice's
     observed cells), so switching changes the axis ORDER and SIZE,
     not just the colors — every axis-derived value below reads
     from the selected condition, and any hover/pin is dropped on
     switch since the indices no longer mean the same cell.
   • Hover: the subject (row) label floats as a tooltip beside the
     cursor, the item (column) label sits in a bottom strip, with
     crosshair guides — all aligned to the hovered cell.
   • Click a cell to pin it and show its details below the matrix:
     the item prompt + gold answer, the subject + score, the cell
     value, and — when the benchmark publishes traces — how that
     model actually answered. Click the same cell again to dismiss.

   The PNG holds every cell; hover maps the cursor to (row, col)
   and reads the response by sampling the pixel colour. Item
   prompts and exact model answers are loaded from the build on click.
   ============================================================ */

const STRIP = 30; // px, bottom axis (item)
const Z_STRIP = 30; // px, item-difficulty strip under the matrix

async function readCellData<T>(
  dataUrl: string,
  kind: "item" | "answer",
  key: string | ObservationKey,
  signal: AbortSignal,
): Promise<T | undefined> {
  const encoded = JSON.stringify(key);
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(encoded),
  );
  const bucket = new Uint8Array(digest)[0].toString(16).padStart(2, "0");
  const response = await fetch(`${dataUrl}/${kind}/${bucket}.json.gz`, {
    signal,
  });
  if (!response.ok) throw new Error(`Could not load ${kind} data.`);
  const entries = (await response.json()) as Record<string, T>;
  return entries[encoded];
}
/* Display-time colour swap for BINARY matrices, so we can recolour without
   re-rendering any PNG. The matrices bake RED=(214,39,40)=correct(1),
   BLUE=(31,119,180)=incorrect(0), GRAY=(158,158,158)=unobserved. This exact
   linear feColorMatrix (zero offset) maps
        RED → BLUE,  BLUE → RED,  GRAY → WHITE
   and — being purely linear — maps every anti-aliased blend of those three
   colours to the blend of their targets, so cell boundaries stay clean with no
   colour fringing. color-interpolation-filters MUST be sRGB (the coefficients
   were solved in sRGB byte space; the linearRGB default would distort them).
   The offscreen sampling canvas reads the raw (unfiltered) PNG, so cell
   classification is unaffected — only the on-screen colours change. */
const COLOR_SWAP_ID = "matrix-binary-color-swap";
const COLOR_SWAP_MATRIX = [
  -0.18308, 1.70144, 0.09556, 0, 0, 0.33193, 3.31151, -2.02951, 0, 0, 0.67756,
  2.45148, -1.51512, 0, 0, 0, 0, 0, 1, 0,
].join(" ");

type MatrixViewerProps = {
  slug: string;
  bundle: ChartBundle;
  width: number;
  height: number;
  alt: string;
  rows: string[];
  rowIds: string[] | null;
  rowScores: (number | null)[];
  colIds: string[] | null;
  colP: (number | null)[] | null;
  nItems: number;
  isBinary: boolean;
  hasTraces: boolean;
  /** This benchmark ships per-cell call recordings (see generate_benchmark_gallery.py). */
  audio?: {
    path: string;
    subjects: Record<string, string>;
    items: Record<string, string>;
  };
  conditions?: BenchmarkConditions | null;
  categories?: MatrixCategory[] | null;
  binaryLabels?: { zero: string; one: string } | null;
  /** Rasch fit for this benchmark — the AUC panel above the matrix and the θ
   *  column beside it. Absent for graded benchmarks, which are not fitted. */
  irt?: BenchmarkIrt | null;
};

type CellValue = { label: string; color: string };
type Hover = {
  rx: number;
  ry: number;
  row: number;
  col: number;
  value: CellValue;
};
/** A pinned cell. `row`/`col` index the CURRENTLY SELECTED slice's axes, which
 *  is all the PNG matrix can report. The joined matrix instead names the cell
 *  outright: its columns are per-block and shared across conditions, so a
 *  positional index would resolve to the wrong item. When the ids are present
 *  they win, and the positions are re-derived from them. */
type Pin = {
  row: number;
  col: number;
  value: CellValue;
  subjectId?: string;
  itemId?: string;
  /** Pairwise matrices only. A cell there is a COMPARISON, so it has two
   *  answers: the first-named model's and its opponent's. `cond2` is the
   *  opponent's own condition for the same battle (each side records the other
   *  as its opponent), which is what its trace is keyed by. */
  subjectId2?: string;
  cond2?: string | null;
  key?: ObservationKey;
  key2?: ObservationKey;
};
type ItemContent = { content: string | null; answer: string | null };

type LazyData = {
  items: Record<string, ItemContent>;
  colIds: string[] | null;
  colP: (number | null)[] | null;
  /** Per-condition column axis, keyed by the same "dim=val;…" the selector
   *  builds. `colIdx` indexes into `colIds` — each condition is sorted by its own
   *  column mean and covers only the items it observed, so both the order and the
   *  length differ per condition. Absent for unsliced benchmarks. */
  slices: Record<string, { colIdx: number[]; colP: (number | null)[] }> | null;
};

export type ChartBundle = {
  chart: JoinedData | FacetedData | SessionData | null;
  axes: LazyData;
  matrices: Record<
    string,
    {
      width: number;
      height: number;
      pixels: [number, number, number, number][];
      keys: Record<number, ObservationKey>;
    }
  >;
};

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

function classify(
  r: number,
  g: number,
  b: number,
  isBinary: boolean,
  categories?: MatrixCategory[] | null,
  binaryLabels?: { zero: string; one: string } | null,
): CellValue {
  const categorical = Boolean(categories && categories.length);
  // Binary matrices are displayed through the RED→BLUE / BLUE→RED / GRAY→WHITE
  // filter (see COLOR_SWAP_MATRIX), so the swatch colours reported here must
  // match what the viewer actually shows — not the raw PNG pixel. Only the
  // colours flip; the pixel-read semantics (redder ⇒ 1) are unchanged.
  if (Math.abs(r - g) < 16 && Math.abs(g - b) < 16 && r > 120 && r < 198) {
    return {
      label: "Not evaluated",
      color: isBinary && !categorical ? "#ffffff" : "#9e9e9e",
    };
  }
  // Nominal categories: snap the sampled pixel to the nearest category colour.
  if (categories && categories.length) {
    let best = categories[0];
    let bestD = Infinity;
    for (const c of categories) {
      const [cr, cg, cb] = hexToRgb(c.color);
      const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    return { label: best.label, color: best.color };
  }
  const redder = r - b >= 0;
  if (isBinary) {
    // Display swap: the "1" pixel is red but shown blue, the "0" pixel is blue
    // but shown red. Labels keep their meaning; only the swatch colour flips.
    if (binaryLabels) {
      return redder
        ? { label: `${binaryLabels.one} (1)`, color: "#1f77b4" }
        : { label: `${binaryLabels.zero} (0)`, color: "#d62728" };
    }
    return redder
      ? { label: "Correct (1)", color: "#1f77b4" }
      : { label: "Incorrect (0)", color: "#d62728" };
  }
  return redder
    ? { label: "Higher response", color: "#d62728" }
    : { label: "Lower response", color: "#1f77b4" };
}

function pctText(p: number | null | undefined): string {
  return p === null || p === undefined ? "N/A" : `${Math.round(p * 100)}%`;
}

function conditionKey(dims: string[], sel: Record<string, string>): string {
  return dims.map((d) => `${d}=${sel[d]}`).join(";");
}

/* ---- Condition selection ----------------------------------------------
   Dimensions are RAGGED: a benchmark's conditions often carry different dim
   sets, so a slice that lacks a dim is keyed with the sentinel below (mirrors
   MISSING_DIM in scripts/render_website/generate_benchmark_gallery.py). It is offered in the
   dropdown like any other value.

   The consequence is that the dropdowns span a cross-product wider than the set
   of slices that exist — visit_bench has an `opponent` only under the pairwise
   protocol, so pairing `human_correctness_rating` with a named opponent names
   nothing. Rather than stranding the user on an empty panel, changing one
   dropdown SNAPS the others to the nearest slice that does exist, keeping the
   dim the user just set. Filtering each dropdown to its locally-valid values
   instead would be worse: with mutually exclusive dims (block XOR experiment)
   every single-dim step out of a valid slice is invalid, so the menus would
   strand the user in whichever half they started in. */
const MISSING_DIM = "n/a";

/** A slice's selection over `dims`, with absent dims filled in. Read from the
 *  stored `sel` rather than parsed back out of the key, because a collapsed
 *  dimension's value is a raw test_condition that may itself contain "=" or
 *  ";" — writing such a key is unambiguous, re-parsing it is not. */
function slotsOf(dims: string[], m: MatrixCondition): Record<string, string> {
  const out: Record<string, string> = {};
  for (const d of dims) out[d] = m.sel?.[d] ?? MISSING_DIM;
  return out;
}

/** `want` if it names a real slice, else the existing slice agreeing with it on
 *  the most dims — never anything that disagrees on `pinned` (the dim the user
 *  just chose), so their choice is honoured rather than undone. */
function snapSel(
  conditions: BenchmarkConditions,
  want: Record<string, string>,
  pinned: string | null,
): Record<string, string> {
  if (conditions.matrices[conditionKey(conditions.dims, want)]) return want;
  let best: Record<string, string> | null = null;
  let bestScore = -1;
  for (const m of Object.values(conditions.matrices)) {
    const slots = slotsOf(conditions.dims, m);
    if (pinned !== null && slots[pinned] !== want[pinned]) continue;
    let score = 0;
    for (const d of conditions.dims) if (slots[d] === want[d]) score++;
    if (score > bestScore) {
      bestScore = score;
      best = slots;
    }
  }
  return best ?? want; // no slice honours `pinned` — leave it; the empty panel
} // below is then the truthful answer

/** A collapsed dimension's value is a raw test_condition, so it reads as
 *  "block=standalone". Show it as "block: standalone" — the dim name is the
 *  meaningful half here (it is what distinguishes the batches), so it stays. */
function condLabel(v: string): string {
  return v.split(";").join(" · ").split("=").join(": ");
}

/** "attack, category, or judge" — the dims this benchmark actually has, for the
 *  empty panel, which used to name those three whatever the benchmark was. */
function listDims(dims: string[]): string {
  if (dims.length === 1) return dims[0];
  return `${dims.slice(0, -1).join(", ")} or ${dims[dims.length - 1]}`;
}

/* ============================================================
   MathText — renders a text blob (item prompt, gold answer, model
   answer) as Markdown with any embedded LaTeX typeset by KaTeX.
   Many benchmarks store their prompts as full markdown documents
   (headings, bold, lists, fenced code), others as plain text with
   TeX; both render cleanly here.

   Math delimiters, found by our own strict pre-pass and normalised
   to remark-math's $$-form before the markdown parse:
     \begin{env}…\end{env}  display math (align*, cases, matrix, …)
     $$…$$   \[…\]          display math (own centered line)
     \(…\)   $…$            inline math

   The pre-pass — not remark-math's own single-$ mode — decides
   what counts as inline math: the content may not start/end with
   whitespace and the closing $ may not be followed by a digit, so
   prose like "costs $5 and $10 more" is never mis-typeset.
   remark-breaks keeps single newlines as line breaks, so plain-text
   prompts keep their layout — but only for prompts that don't read
   as full markdown documents. A doc with headings or fenced code
   was authored with markdown's soft-wrap convention, so its single
   newlines are ~80-column source wrapping, not layout; hard-breaking
   them renders ragged mid-sentence lines (e.g. ceobench's README
   prompt). Those get standard markdown paragraph flow instead.
   KaTeX errors render inline in red rather than crashing the panel.
   ============================================================ */

const MATH_RE =
  // \begin{env}…\end{env} | $$…$$ | \[…\] | \(…\) | $…$ (not preceded by \, $
  // or a word char; not followed by a digit or another $ — keeps dollar
  // amounts as text)
  /(\\begin\{([a-zA-Z*]+)\}[\s\S]+?\\end\{\2\})|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$\w])\$((?:\\[\s\S]|[^$])+?)\$(?![\d$])/g;

/** Rewrite every math span to the $$-delimited form remark-math parses
 *  (single-$ parsing is disabled there — this pre-pass is the sole judge of
 *  what is math). Display math is set off as its own block so it typesets
 *  centered on its own line. */
function normalizeMathDelimiters(source: string): string {
  // Some benchmarks store their text with literal "\n" escape sequences
  // rather than real newlines. Unescape those — but never when a letter
  // follows, which would clobber TeX commands (\neq, \nabla, \nu, …).
  source = source.replace(/\\n(?![a-zA-Z])/g, "\n");
  let out = "";
  let last = 0;
  for (const m of source.matchAll(MATH_RE)) {
    const [raw, env, , dollars, brackets, parens, inline] = m;
    const display =
      env !== undefined || dollars !== undefined || brackets !== undefined;
    // The environment keeps its \begin…\end wrapper — KaTeX parses it whole.
    const value = env ?? dollars ?? brackets ?? parens ?? inline ?? "";
    // Reject sloppy inline-$ matches ("… $5 and 3$ …"): real inline math
    // hugs its delimiters.
    if (inline !== undefined && /^\s|\s$/.test(inline)) continue;
    out += source.slice(last, m.index);
    out += display ? `\n\n$$\n${value}\n$$\n\n` : `$$${value}$$`;
    last = m.index + raw.length;
  }
  return out + source.slice(last);
}

/** Markdown element styling, matched to the panel's compact type scale. */
const MD_COMPONENTS: Components = {
  h1: (p) => (
    <h1
      className="mt-4 mb-2 text-[0.9375rem] font-semibold first:mt-0"
      {...p}
    />
  ),
  h2: (p) => (
    <h2 className="mt-4 mb-2 text-[0.875rem] font-semibold first:mt-0" {...p} />
  ),
  h3: (p) => (
    <h3
      className="mt-3 mb-1.5 text-[0.8125rem] font-semibold first:mt-0"
      {...p}
    />
  ),
  h4: (p) => (
    <h4
      className="mt-3 mb-1.5 text-[0.8125rem] font-semibold first:mt-0"
      {...p}
    />
  ),
  p: (p) => <p className="my-2 first:mt-0 last:mb-0" {...p} />,
  ul: (p) => <ul className="my-2 list-disc pl-5" {...p} />,
  ol: (p) => <ol className="my-2 list-decimal pl-5" {...p} />,
  li: (p) => <li className="my-0.5" {...p} />,
  pre: (p) => (
    <pre
      className="my-2 overflow-x-auto rounded bg-[var(--fog-light)] p-3"
      {...p}
    />
  ),
  code: (p) => <code className="font-mono text-[0.75rem]" {...p} />,
  a: (p) => <a className="text-[var(--lagunita)] underline" {...p} />,
  img: ({ node: _node, src, alt, ...p }) => (
    // Plain <img>, not next/image: item images (e.g. edu_circuit_hw's
    // handwritten solutions, extracted to /benchmarks/item-images/ by
    // generate_benchmark_gallery) are static exports with unknown dimensions. The srcs are
    // embedded in the click-data payloads, so they arrive as root-absolute
    // /benchmarks/... paths and route through the gallery proxy; any other
    // site-relative src just needs the basePath prefix, and external URLs are
    // left alone.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      className="my-2 max-h-[24rem] max-w-full rounded border border-[var(--line)]"
      src={
        typeof src !== "string"
          ? src
          : src.startsWith("/benchmarks/")
            ? withBase(src)
            : src.startsWith("/")
              ? withBase(src)
              : src
      }
      alt={alt ?? "item image"}
      loading="lazy"
      {...p}
    />
  ),
  blockquote: (p) => (
    <blockquote
      className="my-2 border-l-2 border-[var(--line)] pl-3 text-[var(--muted)]"
      {...p}
    />
  ),
  hr: (p) => <hr className="my-3 border-[var(--line)]" {...p} />,
  table: (p) => <table className="my-2 border-collapse" {...p} />,
  th: (p) => (
    <th
      className="border border-[var(--line)] px-2 py-1 text-left font-semibold"
      {...p}
    />
  ),
  td: (p) => <td className="border border-[var(--line)] px-2 py-1" {...p} />,
};

/** What an item's trailing aside is, which decides the box it gets. */
type AsideKind = "preview" | "meta";

const ASIDE_HEADINGS: Record<AsideKind, string> = {
  preview: "Rendered preview · not shown to the model",
  meta: "Curation metadata · not shown to the model",
};

/** Split an item's content into [prompt, aside, kind] on a `<!--preview: …-->`
 *  or `<!--meta: …-->` marker.
 *
 *  Some benchmarks carry material the subject was NOT shown. arc_agi_3 appends
 *  a picture of its grid purely so a human can read the item (`preview`);
 *  agents_last_exam appends the task-card fields and the identifiers that make
 *  the run reproducible, which the curation guide asks it to carry (`meta`).
 *  Either one rendered in the prompt's box reads as part of the prompt, and a
 *  heading saying otherwise is easy to skim past, so the marker lets the panel
 *  put it in a box of its own. Content with no marker is returned unchanged
 *  with a null aside, which is most benchmarks. */
function splitAside(content: string): [string, string | null, AsideKind] {
  for (const kind of ["preview", "meta"] as AsideKind[]) {
    const at = content.indexOf(`<!--${kind}`);
    if (at < 0) continue;
    const end = content.indexOf("-->", at);
    if (end < 0) continue;
    return [
      content.slice(0, at).trimEnd(),
      content.slice(end + 3).trim() || null,
      kind,
    ];
  }
  return [content, null, "preview"];
}

/** True when the text reads as an authored markdown document (ATX headings
 *  or fenced code blocks) rather than a plain-text prompt — such documents
 *  use single newlines as source wrapping, so remark-breaks must not turn
 *  them into hard <br>s. */
function looksLikeMarkdownDoc(text: string): boolean {
  return /^#{1,6}\s\S/m.test(text) || /^\s*(`{3,}|~{3,})/m.test(text);
}

/** Fence top-level JSON blocks so they render as code.
 *
 *  Several benchmarks store structured task specs (tau_voice's item prompts and
 *  gold answers, core_codereasoning, worldcentralbanks, algotune) as pretty-
 *  printed JSON embedded in an otherwise-markdown blob. Markdown mangles that
 *  badly and inconsistently: 2-space-indented lines collapse into one run-on
 *  paragraph, while 4-space-indented ones become accidental indented code
 *  blocks, so a single object renders as half prose and half code.
 *
 *  This is presentation only — the stored item content is never rewritten. A run
 *  is fenced ONLY if it starts with `{`/`[` alone on a line, ends at the matching
 *  brace in column 0, and `JSON.parse` accepts it, so prose can't be swallowed by
 *  accident. Text that already carries a fence is returned untouched. Verified
 *  lossless (fences stripped == original) over 2,769 samples spanning 40
 *  benchmarks plus tau_voice's transcripts. */
function fenceJsonBlocks(text: string): string {
  if (text.includes("```")) return text;
  const lines = text.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const open = lines[i];
    if (open === "{" || open === "[") {
      const close = open === "{" ? "}" : "]";
      let end = -1;
      for (let j = i + 1; j < lines.length; j++) {
        if (lines[j] === close) {
          end = j;
          break;
        }
      }
      if (end > i) {
        const block = lines.slice(i, end + 1).join("\n");
        try {
          JSON.parse(block);
          out.push("```json", block, "```");
          i = end;
          continue;
        } catch {
          // Not JSON — fall through and emit the line verbatim.
        }
      }
    }
    out.push(open);
  }
  return out.join("\n");
}

function MathText({ text }: { text: string }) {
  const normalized = useMemo(() => normalizeMathDelimiters(text), [text]);
  // Deliberately measured on the UNFENCED text: fencing would otherwise flip a
  // prose-plus-JSON blob to "doc" and silently drop the soft line breaks in its
  // prose. Whether newlines are meaningful is a property of the source, not of
  // what we did to it for display.
  const isDoc = useMemo(() => looksLikeMarkdownDoc(normalized), [normalized]);
  const source = useMemo(() => fenceJsonBlocks(normalized), [normalized]);
  return (
    <ReactMarkdown
      urlTransform={(url, key) =>
        key === "src" && /^data:image\/(?:png|jpeg|gif|webp);base64,/.test(url)
          ? url
          : defaultUrlTransform(url)
      }
      remarkPlugins={[
        [remarkMath, { singleDollarTextMath: false }],
        remarkGfm,
        ...(isDoc ? [] : [remarkBreaks]),
      ]}
      rehypePlugins={[[rehypeKatex, { strict: "ignore" }]]}
      components={MD_COMPONENTS}
    >
      {source}
    </ReactMarkdown>
  );
}

/** Renders children only when `omit` is false. Benchmarks that ship the joined
 *  matrix drop the per-condition matrix and its one-dropdown-per-dimension
 *  toolbar entirely: the joined view shows every response with nothing to
 *  select, so the slice picker is redundant on those pages. Every other
 *  benchmark is unaffected and keeps its matrix. */
function OmitWhen({
  omit,
  children,
}: {
  omit: boolean;
  children: React.ReactNode;
}) {
  return omit ? null : <>{children}</>;
}

/** Flanks the matrix with its two numeric columns, or renders it untouched when
 *  there is no fit to show. The `head` pad matches the gutters' heading line so
 *  row 0 of a gutter starts level with row 0 of the image. */
const GUTTER_HEAD = 18; // px — MatrixGutter's h-3.5 heading + mb-1

function GutterRow({
  show,
  gap,
  head,
  left,
  right,
  children,
}: {
  show: boolean;
  gap: number;
  head: number;
  left: React.ReactNode;
  right: React.ReactNode;
  children: React.ReactNode;
}) {
  if (!show) return <>{children}</>;
  // The matrix sits inside a 1px-bordered box, so its first pixel row starts
  // one pixel lower than a bare canvas would. Without this the columns drift
  // half a row out of step on a tall matrix.
  const BORDER = 1;
  return (
    <div className="flex items-start justify-center" style={{ gap }}>
      <div style={{ paddingTop: BORDER }}>{left}</div>
      <div className="min-w-0" style={{ paddingTop: head }}>
        {children}
      </div>
      <div style={{ paddingTop: BORDER }}>{right}</div>
    </div>
  );
}

export function MatrixViewer({
  slug,
  bundle,
  width,
  height,
  alt,
  rows,
  rowIds,
  rowScores,
  colIds,
  colP,
  nItems,
  isBinary,
  hasTraces,
  audio,
  conditions,
  categories,
  binaryLabels,
  irt,
}: MatrixViewerProps) {
  const dataUrl = withBase(`/benchmark-data/${slug}`);
  const hasJoined = Boolean(bundle.chart);
  const axes = bundle.axes;
  const [directImage, setDirectImage] = useState<{
    key: string;
    src: string;
  } | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);
  const [directAnswers, setDirectAnswers] = useState<{
    key: string;
    texts: (string | null)[];
  } | null>(null);
  const rowKind = bundle.chart?.rowKind ?? "subject";
  const categorical = Boolean(categories && categories.length);
  const [actual, setActual] = useState(false);
  const [hover, setHover] = useState<Hover | null>(null);
  const [pin, setPin] = useState<Pin | null>(null);
  const [items, setItems] = useState<Record<string, ItemContent>>({});
  // Seed from `default`, but snapped: a payload built before `default` named
  // every dim leaves a <select> showing its own first option instead, which for
  // a ragged benchmark is a pairing that never existed — the bug this page had.
  const [sel, setSel] = useState<Record<string, string>>(() =>
    conditions
      ? snapSel(
          conditions,
          Object.fromEntries(
            conditions.dims.map((d) => [
              d,
              conditions.default[d] ?? MISSING_DIM,
            ]),
          ),
          null,
        )
      : {},
  );
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  // Resolve the active matrix: a selected condition overrides the defaults.
  const comboKey = conditions ? conditionKey(conditions.dims, sel) : null;
  const cond =
    conditions && comboKey ? (conditions.matrices[comboKey] ?? null) : null;
  const missing = Boolean(conditions) && cond === null;
  const effW = cond?.matrixSize[0] ?? width;
  const effH = cond?.matrixSize[1] ?? height;

  const imageKey = comboKey ?? "";
  const matrixUrl = directImage?.key === imageKey ? directImage.src : null;

  useEffect(() => {
    if (missing || hasJoined) return;
    const data = bundle.matrices[imageKey];
    if (!data) return;
    const frame = requestAnimationFrame(() => {
      const cells = document.createElement("canvas");
      cells.width = data.width;
      cells.height = data.height;
      const ctx = cells.getContext("2d")!;
      const pixels = ctx.createImageData(data.width, data.height);
      for (let i = 0; i < data.width * data.height; i++) {
        pixels.data.set([158, 158, 158, 255], i * 4);
      }
      for (const [offset, r, g, b] of data.pixels)
        pixels.data.set([r, g, b, 255], offset * 4);
      ctx.putImageData(pixels, 0, 0);
      const canvas = document.createElement("canvas");
      canvas.width = effW;
      canvas.height = effH;
      const scaled = canvas.getContext("2d")!;
      scaled.imageSmoothingEnabled = false;
      scaled.drawImage(cells, 0, 0, effW, effH);
      setDirectImage({ key: imageKey, src: canvas.toDataURL() });
    });
    return () => cancelAnimationFrame(frame);
  }, [bundle, imageKey, effW, effH, missing, hasJoined]);

  // Row axis. A condition ships its axis as indices into the parent's `rows` /
  // `rowIds` (the union over all conditions), so its rows are both reordered and
  // narrowed to the subjects it actually observed. `rowIdx` is absent for
  // unsliced benchmarks and for legacy payloads predating per-condition axes;
  // both fall back to the parent axis as-is.
  const rowIdx = cond?.rowIdx ?? null;
  const effRows = useMemo(
    () => (rowIdx ? rowIdx.map((i) => rows[i]) : rows),
    [rowIdx, rows],
  );
  const effRowIds = useMemo(
    () => (rowIdx && rowIds ? rowIdx.map((i) => rowIds[i]) : rowIds),
    [rowIdx, rowIds],
  );
  const effRowScores = cond?.rowScores ?? rowScores;
  const nSubjects = effRows.length;

  // Each condition indexes into the global item axis carried by the bundle.
  const perSliceCols = cond?.nItems != null;
  const slice =
    comboKey && axes.slices ? (axes.slices[comboKey] ?? null) : null;
  const baseColIds = colIds ?? axes.colIds ?? null;
  const effColIds = useMemo(() => {
    if (!perSliceCols) return baseColIds;
    return slice && baseColIds ? slice.colIdx.map((i) => baseColIds[i]) : null;
  }, [perSliceCols, slice, baseColIds]);
  const effColP = perSliceCols
    ? (slice?.colP ?? null)
    : (cond?.colP ?? colP ?? axes.colP ?? null);
  const effNItems = cond?.nItems ?? nItems;

  // Fit-to-width layout: the matrix is drawn at square cells (side = available
  // width / items, capped at CELL_MAX so tiny benchmarks stay compact) and its
  // bordered box shrink-wraps it exactly — no dead space. Item-rich matrices
  // whose columns are sub-pixel keep full width with a MIN_ROW row floor so
  // they render as the familiar band; subject-heavy matrices render as
  // (at most MAX_H-tall) towers — rows always mean models. Sizing is measured
  // in JS off the outer frame — CSS aspect-ratio + min constraints transfer
  // across axes and overflow.
  //
  // When the Rasch fit is available the matrix is flanked by two numeric
  // columns (θ left, observed accuracy right), so they come out of the same
  // width budget — the image is sized against what is LEFT of the frame, not
  // the whole frame, or the three together overflow the card.
  const MAX_H = 600;
  const CELL_MAX = 16;
  const MIN_ROW = 4;
  const BAND_MIN = 24; // matrices with 1-2 subjects render a visible band, not a hairline
  const GUTTER_L = 48; // θ
  const GUTTER_R = 42; // accuracy
  const GUTTER_GAP = 6;
  const [panelW, setPanelW] = useState<number | null>(null);
  useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      setPanelW(entries[0].contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [missing]);

  // Gutters align to rows of the FITTED image, so they are dropped in
  // "actual size" mode (where the image scrolls inside its box and its rows no
  // longer map to a fixed height) and whenever there is no fit to show.
  const theta = irt?.theta ?? null;
  const showGutters =
    !!theta && !actual && !missing && effRowIds != null && nSubjects > 0;
  const gutterW = showGutters ? GUTTER_L + GUTTER_R + 2 * GUTTER_GAP : 0;
  const availW = Math.max(0, (panelW ?? 0) - gutterW);

  // Both gutters are indexed by the SELECTED slice's row axis, so switching
  // condition re-orders them with the matrix. θ comes from the global fit and is
  // looked up by subject_id; a subject the fit never saw stays blank.
  const thetaByRow = useMemo(
    () => (effRowIds ?? []).map((id) => theta?.[id] ?? null),
    [effRowIds, theta],
  );
  const accByRow = useMemo(
    () => (effRowIds ?? []).map((_, i) => effRowScores?.[i] ?? null),
    [effRowIds, effRowScores],
  );

  const zByItem = irt?.zByItem;
  const effZ = useMemo(
    () =>
      zByItem && effColIds ? effColIds.map((id) => zByItem[id] ?? null) : null,
    [zByItem, effColIds],
  );
  // Drawn only when it spans the axis it sits under — a length mismatch means
  // the two disagree about what a column is.
  const showZStrip = !!effZ && !actual && !missing && effZ.length === effNItems;

  const cellW = availW ? availW / effNItems : 0;
  let fitW = availW;
  let fitH = 0;
  if (cellW >= MIN_ROW) {
    // Individually visible cells: strictly square.
    const side = Math.min(CELL_MAX, cellW, MAX_H / nSubjects);
    fitW = effNItems * side;
    fitH = nSubjects * side;
  } else {
    // Sub-pixel columns: a full-width texture band. The BAND_MIN floor keeps
    // a tiny roster (1-2 subjects) from drawing a near-invisible hairline.
    fitH = Math.max(BAND_MIN, Math.min(MAX_H, nSubjects * MIN_ROW));
  }
  const fitStyle: React.CSSProperties = panelW
    ? {
        width: fitW,
        height: fitH,
        // Crisp cell edges once cells are visibly sized; browser smoothing
        // looks better for sub-pixel-column textures.
        imageRendering: fitW / effNItems >= 3 ? "pixelated" : undefined,
      }
    : { width: "100%", height: "auto" }; // one pre-measure frame

  // Offscreen canvas at native resolution for pixel reads.
  useEffect(() => {
    if (missing || !matrixUrl) {
      ctxRef.current = null;
      return;
    }
    const img = new window.Image();
    img.src = matrixUrl;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = effW;
      canvas.height = effH;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, effW, effH);
      ctxRef.current = ctx;
    };
    return () => {
      ctxRef.current = null;
    };
  }, [matrixUrl, effW, effH, missing]);

  // Switching condition keeps the axis but changes every cell value — drop any
  // stale hover/pin so the panel never shows a value from another condition.
  function selectCondition(dim: string, value: string) {
    setSel((prev) =>
      conditions
        ? snapSel(conditions, { ...prev, [dim]: value }, dim)
        : { ...prev, [dim]: value },
    );
    setHover(null);
    setPin(null);
  }

  function cellAt(e: React.MouseEvent<HTMLImageElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = Math.min(
      0.999,
      Math.max(0, (e.clientX - rect.left) / rect.width),
    );
    const fy = Math.min(
      0.999,
      Math.max(0, (e.clientY - rect.top) / rect.height),
    );
    const col = Math.min(effNItems - 1, Math.floor(fx * effNItems));
    const row = Math.min(nSubjects - 1, Math.floor(fy * nSubjects));
    let value: CellValue = { label: "N/A", color: "#9e9e9e" };
    const ctx = ctxRef.current;
    if (ctx) {
      const [r, g, b] = ctx.getImageData(
        Math.min(effW - 1, Math.floor(fx * effW)),
        Math.min(effH - 1, Math.floor(fy * effH)),
        1,
        1,
      ).data;
      value = classify(r, g, b, isBinary, categories, binaryLabels);
    }
    return { row, col, value, fx, fy };
  }

  function handleMove(e: React.MouseEvent<HTMLImageElement>) {
    const frame = frameRef.current;
    if (!frame) return;
    const frameRect = frame.getBoundingClientRect();
    const { row, col, value } = cellAt(e);
    setHover({
      rx: e.clientX - frameRect.left,
      ry: e.clientY - frameRect.top,
      row,
      col,
      value,
    });
  }

  function handleJoinedPick(pick: JoinedPick) {
    // Keep the selection aligned with the clicked row, including its audio clip.
    if (conditions) {
      const next: Record<string, string> = { ...sel };
      if (pick.cond) {
        for (const part of pick.cond.split(";")) {
          const [k, v] = part.split("=");
          if (k && v !== undefined && conditions.dims.includes(k.trim()))
            next[k.trim()] = v.trim();
        }
      }
      if (conditions.dims.includes("trial"))
        next.trial = String(pick.key?.[3] ?? pick.trial);
      setSel(next);
    }
    setPin({
      row: -1,
      col: -1,
      subjectId: pick.key?.[0] ?? pick.subjectId,
      subjectId2: pick.key2?.[0] ?? pick.subjectId2,
      cond2: pick.cond2,
      key: pick.key,
      key2: pick.key2,
      itemId: pick.key?.[1] ?? pick.itemId,
      // A graded benchmark reports the level it actually landed on; only a binary
      // one collapses to the pass/fail pair (and its binaryLabels renaming).
      value: pick.level ?? {
        label: pick.passed
          ? (binaryLabels?.one ?? "1")
          : (binaryLabels?.zero ?? "0"),
        color: pick.passed ? "#1f77b4" : "#d62728",
      },
    });
  }

  function handleClick(e: React.MouseEvent<HTMLImageElement>) {
    const { row, col, value } = cellAt(e);
    setPin((prev) =>
      prev && prev.row === row && prev.col === col ? null : { row, col, value },
    );
  }

  // A pin from the joined matrix names its cell by id; one from the PNG names it
  // by position. Resolve both to ids AND positions so the detail panel below
  // serves either source unchanged.
  const itemId = pin
    ? (pin.itemId ?? (effColIds ? (effColIds[pin.col] ?? null) : null))
    : null;
  const subjectId = pin
    ? (pin.subjectId ?? (effRowIds ? (effRowIds[pin.row] ?? null) : null))
    : null;
  const pinCol =
    pin && pin.itemId && effColIds
      ? effColIds.indexOf(pin.itemId)
      : (pin?.col ?? -1);
  const pinRow =
    pin && pin.subjectId && effRowIds
      ? effRowIds.indexOf(pin.subjectId)
      : (pin?.row ?? -1);
  const subjectId2 = pin?.subjectId2 ?? null;

  const observation =
    pin?.key ??
    (pin
      ? bundle.matrices[imageKey]?.keys[pin.row * effNItems + pin.col]
      : null) ??
    null;
  const answerQuery = JSON.stringify([observation, pin?.key2 ?? null]);
  useEffect(() => {
    if (!itemId) return;
    const controller = new AbortController();
    void readCellData<ItemContent>(dataUrl, "item", itemId, controller.signal)
      .then((item) => {
        if (!item) throw new Error("Could not load item content.");
        if (!controller.signal.aborted)
          setItems((previous) => ({ ...previous, [itemId]: item }));
      })
      .catch((error: Error) => {
        if (!controller.signal.aborted) setDataError(error.message);
      });
    return () => controller.abort();
  }, [dataUrl, itemId]);

  useEffect(() => {
    if (!hasTraces) return;
    const keys = JSON.parse(answerQuery) as (ObservationKey | null)[];
    const controller = new AbortController();
    void Promise.all(
      keys.map(async (key) => {
        if (!key) return null;
        const answer = await readCellData<{ trace: string | null }>(
          dataUrl,
          "answer",
          key,
          controller.signal,
        );
        return answer?.trace ?? null;
      }),
    )
      .then((texts) => {
        if (!controller.signal.aborted) {
          setDirectAnswers({ key: answerQuery, texts });
          setDataError(null);
        }
      })
      .catch((error: Error) => {
        if (!controller.signal.aborted) {
          setDataError(error.message);
          setDirectAnswers({ key: answerQuery, texts: [null, null] });
        }
      });
    return () => controller.abort();
  }, [dataUrl, hasTraces, answerQuery]);

  function resolveAnswer(index: number) {
    const ready = directAnswers?.key === answerQuery;
    const text = ready ? directAnswers.texts[index] : null;
    const state: "loading" | "answer" | "none" = !hasTraces
      ? "none"
      : !ready
        ? "loading"
        : text != null
          ? "answer"
          : "none";
    return { state, text };
  }
  const { state: answerState, text: answerText } = resolveAnswer(0);
  const answer2 = resolveAnswer(1);

  /* One pane per model whose answer this cell has. A normal cell has one; a
     pairwise cell has two, and naming them matters there — "Model answer" twice
     would leave the reader to guess which side is which. */
  const nameOfSubject = (sid: string | null) => {
    if (!sid || !effRowIds) return null;
    const i = effRowIds.indexOf(sid);
    return i >= 0 ? (effRows[i] ?? null) : null;
  };
  const answerPanes = subjectId2
    ? [
        {
          // "First"/"Second" match the legend ("first model preferred"), and
          // say nothing about who won this cell — which model leads is fixed
          // for the whole row, while the outcome changes cell to cell.
          key: "a",
          heading: "First model" as string,
          name: nameOfSubject(subjectId),
          state: answerState,
          text: answerText,
        },
        {
          key: "b",
          heading: "Second model",
          name: nameOfSubject(subjectId2),
          state: answer2.state,
          text: answer2.text,
        },
      ]
    : [
        {
          key: "a",
          heading: "Model answer",
          name: null as string | null,
          state: answerState,
          text: answerText,
        },
      ];

  // Existing recordings retain their published IDs and filename convention.
  const audioValues: Record<string, string | undefined> = {
    ...sel,
    subject_id: audio?.subjects[subjectId ?? ""],
    item_id: audio?.items[itemId ?? ""],
  };
  const audioSrc =
    audio && audioValues.subject_id && audioValues.item_id
      ? withBase(
          audio.path.replace(/\{(\w+)\}/g, (_, key: string) =>
            (audioValues[key] ?? "").replace(/[;=/]/g, "_"),
          ),
        )
      : null;
  const pinned = pin
    ? {
        subject: effRows[pinRow] ?? `row ${pinRow + 1}`,
        subjectScore: effRowScores[pinRow] ?? null,
        itemIndex: pinCol,
        solveRate: effColP ? (effColP[pinCol] ?? null) : null,
        itemId,
        item: itemId ? items[itemId] : null,
        value: pin.value,
      }
    : null;

  function titleCase(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  return (
    <div>
      {isBinary && !categorical ? (
        <svg aria-hidden="true" className="absolute h-0 w-0" focusable="false">
          <filter id={COLOR_SWAP_ID} colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix" values={COLOR_SWAP_MATRIX} />
          </filter>
        </svg>
      ) : null}
      {/* The Rasch summary heads the section: it describes the whole matrix
          below it, in either rendering. Absent for graded benchmarks. */}
      {irt ? <IrtPanel irt={irt} /> : null}

      {/* Joined matrix first: every response at once, nothing to choose but
          trial. The per-condition matrix and its dropdowns follow, collapsed. */}
      {hasJoined && rowKind === "session" ? (
        <SessionStrip
          data={bundle.chart as SessionData}
          onPick={handleJoinedPick}
          binaryLabels={binaryLabels}
        />
      ) : null}

      {hasJoined && rowKind === "faceted" ? (
        <FacetedMatrix
          data={bundle.chart as FacetedData}
          theta={theta}
          onPick={handleJoinedPick}
          binaryLabels={binaryLabels}
        />
      ) : null}

      {hasJoined && !["session", "faceted"].includes(rowKind) ? (
        <JoinedMatrix
          data={bundle.chart as JoinedData}
          theta={theta}
          /* The GLOBAL row axis (union over every condition), not effRowIds —
             that one is the selected slice's rows, so on a voice slice it omits
             the text-baseline subjects and their names fall back to raw
             subject_id hashes. The joined matrix spans all conditions. */
          rowIds={rowIds}
          rows={rows}
          onPick={handleJoinedPick}
          zByItem={zByItem}
          /* 1 is not always "passed" — see the prop's note. handleJoinedPick
             already renames the pinned cell; without this the hover tip that
             precedes the click still said the opposite. */
          binaryLabels={binaryLabels}
        />
      ) : null}

      {dataError ? (
        <p role="alert" className="my-3 text-sm text-red-700">
          {dataError}
        </p>
      ) : null}
      <OmitWhen omit={!!hasJoined}>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <p className="text-xs text-[var(--muted)]">
            {actual ? "Full resolution. Scroll to inspect." : "Fit to width."}{" "}
            Hover for subject &amp; item; click a cell for details.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {/* Condition dropdowns — left of "Actual size" */}
            {conditions
              ? conditions.dims.map((dim) => (
                  <label
                    key={dim}
                    className="flex items-center gap-1 text-xs text-[var(--muted)]"
                  >
                    <span className="sr-only sm:not-sr-only">
                      {titleCase(dim)}
                    </span>
                    <select
                      value={sel[dim] ?? ""}
                      onChange={(ev) => selectCondition(dim, ev.target.value)}
                      aria-label={titleCase(dim)}
                      className="rounded-md border border-[var(--line)] bg-white px-2 py-1 text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--muted)]"
                    >
                      {conditions.values[dim].map((v) => (
                        <option key={v} value={v}>
                          {dim === conditions.collapsed ? condLabel(v) : v}
                        </option>
                      ))}
                    </select>
                  </label>
                ))
              : null}
            <button
              type="button"
              onClick={() => setActual((v) => !v)}
              aria-pressed={actual}
              className="shrink-0 rounded-md border border-[var(--line)] px-2.5 py-1 text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--muted)] hover:bg-[var(--fog-light)]"
            >
              {actual ? "Fit to width" : "Actual size"}
            </button>
          </div>
        </div>

        {missing ? (
          <div className="flex min-h-[160px] items-center justify-center rounded-md border border-dashed border-[var(--line)] bg-[var(--fog-light)] p-6 text-center text-sm text-[var(--muted)]">
            This combination wasn&rsquo;t evaluated. Try a different{" "}
            {conditions ? listDims(conditions.dims) : "condition"}.
          </div>
        ) : (
          /* Frame: θ gutter + matrix + accuracy gutter + bottom strip (item) */
          <div
            ref={frameRef}
            className="relative"
            style={{ paddingBottom: STRIP }}
            onMouseLeave={() => setHover(null)}
          >
            <GutterRow
              show={showGutters}
              gap={GUTTER_GAP}
              head={GUTTER_HEAD}
              left={
                <MatrixGutter
                  values={thetaByRow}
                  width={GUTTER_L}
                  height={fitH}
                  scale="diverging"
                  format={(v) => v.toFixed(2)}
                  side="left"
                  title="θ"
                  label="Fitted ability (theta), per subject"
                  hoverRow={hover?.row ?? null}
                />
              }
              right={
                <MatrixGutter
                  values={accByRow}
                  width={GUTTER_R}
                  height={fitH}
                  scale="fill"
                  format={(v) => pctText(v)}
                  side="right"
                  title="acc"
                  label="Observed accuracy, per subject"
                  hoverRow={hover?.row ?? null}
                />
              }
            >
              <div
                className={`rounded-md border border-[var(--line)] bg-white ${
                  actual
                    ? "max-h-[75vh] overflow-auto"
                    : "mx-auto w-fit max-w-full overflow-hidden"
                }`}
              >
                <Image
                  key={matrixUrl}
                  // `unoptimized` renders a plain <img> and, unlike optimized
                  // next/image, is NOT basePath-prefixed — so the matrix PNG path
                  // must be wrapped (same as the offscreen Image() above).
                  src={
                    matrixUrl ??
                    "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
                  }
                  alt={alt}
                  width={effW}
                  height={effH}
                  unoptimized
                  onMouseMove={handleMove}
                  onClick={handleClick}
                  className={`block cursor-crosshair ${
                    actual
                      ? "h-auto min-w-full max-w-none"
                      : "max-h-full max-w-full"
                  }`}
                  style={{
                    ...(actual ? {} : fitStyle),
                    // Recolour binary matrices at display time (RED↔BLUE, GRAY→WHITE)
                    // without touching the underlying PNG. Graded/categorical matrices
                    // keep their own colour scale, so the filter is skipped for them.
                    ...(isBinary && !categorical
                      ? { filter: `url(#${COLOR_SWAP_ID})` }
                      : {}),
                  }}
                />
              </div>
              {showZStrip ? (
                <div className="mx-auto w-fit max-w-full">
                  <MatrixColumnStrip
                    values={effZ as (number | null)[]}
                    width={fitW}
                    height={Z_STRIP}
                  />
                </div>
              ) : null}
            </GutterRow>

            {hover ? (
              <>
                {/* crosshair guides */}
                <div
                  className="pointer-events-none absolute z-10 border-t border-[var(--ink)]/40"
                  style={{ left: 0, right: 0, top: hover.ry }}
                />
                <div
                  className="pointer-events-none absolute z-10 border-l border-[var(--ink)]/40"
                  style={{ top: 0, bottom: STRIP, left: hover.rx }}
                />
                {/* subject label as a floating tooltip above the cursor */}
                <div
                  className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-[calc(100%+10px)]"
                  style={{ left: hover.rx, top: hover.ry }}
                >
                  <span className="whitespace-nowrap rounded bg-[var(--ink)] px-1.5 py-0.5 text-[0.625rem] font-medium text-white shadow-sm">
                    {effRows[hover.row]}
                    {/* On a 250-subject matrix the gutters are 2px per row, far
                        too tight for a printed number — so the hover tooltip is
                        where θ and the row's accuracy are actually legible. */}
                    {showGutters ? (
                      <span className="font-normal text-white/70">
                        {thetaByRow[hover.row] != null
                          ? ` · θ ${thetaByRow[hover.row]?.toFixed(2)}`
                          : ""}
                        {accByRow[hover.row] != null
                          ? ` · ${pctText(accByRow[hover.row])}`
                          : ""}
                      </span>
                    ) : null}
                  </span>
                </div>
                {/* item label in bottom strip, aligned to column */}
                <div
                  className="pointer-events-none absolute bottom-0 z-20 -translate-x-1/2"
                  style={{ left: hover.rx }}
                >
                  <span className="whitespace-nowrap rounded bg-[var(--ink)] px-1.5 py-0.5 text-[0.625rem] font-medium text-white">
                    #{(hover.col + 1).toLocaleString("en-US")}
                    {!categorical && effColP && effColP[hover.col] != null
                      ? ` · ${pctText(effColP[hover.col])}`
                      : ""}
                    {/* The strip shows the shape of the difficulty curve; the
                        exact z for one item is only readable here. */}
                    {effZ && effZ[hover.col] != null ? (
                      <span className="font-normal text-white/70">
                        {` · z ${effZ[hover.col]?.toFixed(2)}`}
                      </span>
                    ) : null}
                  </span>
                </div>
              </>
            ) : null}
          </div>
        )}
      </OmitWhen>

      {/* Pinned cell details */}
      {pinned ? (
        <div className="mt-4 rounded-md border border-[var(--lagunita)]/40 bg-[var(--fog-light)] p-4 sm:p-5">
          <div className="mb-3 flex items-start justify-between gap-3">
            <h4 className="type-h3 text-[1rem] text-[var(--ink)]">
              Cell detail
            </h4>
            <button
              type="button"
              onClick={() => setPin(null)}
              className="rounded-md border border-[var(--line)] bg-white px-2 py-0.5 text-xs text-[var(--muted)] hover:text-[var(--ink)]"
            >
              Close
            </button>
          </div>

          {/* Compact metadata strip: subject · score · cell value · condition */}
          <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded bg-white px-3 py-2.5">
            <div className="flex items-baseline gap-1.5">
              <span className="text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                Subject
              </span>
              <span className="font-mono text-sm text-[var(--ink)]">
                {pinned.subject}
              </span>
              {!categorical && pinned.subjectScore !== null ? (
                <span className="text-xs text-[var(--muted)]">
                  ({pctText(pinned.subjectScore)}
                  {isBinary ? " accuracy" : ""} overall)
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                Response
              </span>
              <span
                className="inline-block h-3.5 w-3.5 rounded-sm border border-[var(--line)]"
                style={{ backgroundColor: pinned.value.color }}
              />
              <span className="text-sm text-[var(--ink)]">
                {pinned.value.label}
              </span>
            </div>
            {conditions ? (
              <div className="flex items-baseline gap-1.5">
                <span className="text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                  Condition
                </span>
                <span className="text-xs text-[var(--ink)]">
                  {conditions.dims.map((d) => `${d}=${sel[d]}`).join(" · ")}
                </span>
              </div>
            ) : null}
          </div>

          {/* The recorded call for this cell, where one exists. Leads the panel
              so the call can be played while reading the task and transcript
              below. Keyed exactly like that transcript, so the audio and the
              text are always the same run. Hidden if the clip 404s — text-
              modality cells of a voice benchmark have a transcript but no
              recording. */}
          {audioSrc ? (
            <div className="mb-4">
              <p className="mb-1 text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                Call recording
              </p>
              <audio
                key={audioSrc}
                controls
                preload="none"
                src={audioSrc}
                className="w-full"
                onError={(e) => {
                  (e.currentTarget.parentElement as HTMLElement).style.display =
                    "none";
                }}
              />
            </div>
          ) : null}

          {/* Item prompt — full width, LaTeX-rendered */}
          <div>
            <p className="mb-1 flex flex-wrap items-baseline gap-x-3 text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
              <span>
                Item #{(pinned.itemIndex + 1).toLocaleString("en-US")}
                {!categorical ? ` · ${pctText(pinned.solveRate)} solved` : ""}
              </span>
              {pinned.itemId ? (
                <span className="font-mono normal-case tracking-normal text-[0.625rem]">
                  {pinned.itemId}
                </span>
              ) : null}
            </p>
            {pinned.item?.content ? (
              (() => {
                const [prompt, aside, kind] = splitAside(pinned.item.content);
                return (
                  <>
                    <div className="max-h-[32rem] overflow-auto break-words rounded bg-white p-4 text-[0.8125rem] leading-6 text-[var(--ink)]">
                      <MathText text={prompt} />
                    </div>
                    {/* Separate box, separate heading: the aside is a reading
                        aid or a curation record, and sharing the prompt's box is
                        what makes readers take it for something the model was
                        shown. */}
                    {aside ? (
                      <div className="mt-2 max-h-[24rem] overflow-auto rounded border border-dashed border-[var(--line)] bg-[var(--fog-light)] p-4">
                        <p className="mb-2 text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                          {ASIDE_HEADINGS[kind]}
                        </p>
                        <div className="break-words text-[0.8125rem] leading-6 text-[var(--ink)]">
                          <MathText text={aside} />
                        </div>
                      </div>
                    ) : null}
                  </>
                );
              })()
            ) : (
              <p className="rounded bg-white p-4 text-[0.8125rem] text-[var(--muted)]">
                {itemId && !Object.hasOwn(items, itemId) && !dataError
                  ? "Loading item…"
                  : "Prompt not available for this item."}
              </p>
            )}
            {pinned.item?.answer ? (
              <div className="mt-2 break-words text-[0.8125rem] leading-6 text-[var(--ink)]">
                <span className="text-[var(--muted)]">Correct answer:</span>
                <div className="text-[var(--palo-alto)]">
                  <MathText text={pinned.item.answer} />
                </div>
              </div>
            ) : null}
          </div>

          {/* The clicked model's actual answer to this item — only for
              benchmarks that publish answer traces. Full width, LaTeX-rendered.
              A pairwise cell is a COMPARISON, so it shows both models' answers
              side by side: seeing one of them alone cannot tell a reader why the
              vote went the way it did. */}
          {hasTraces ? (
            <div
              className={`mt-4 grid gap-4 ${answerPanes.length > 1 ? "lg:grid-cols-2" : ""}`}
            >
              {answerPanes.map((pane) => (
                <div key={pane.key}>
                  <p className="mb-1 flex items-baseline gap-2 text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                    {pane.heading}
                    {pane.name ? (
                      <span className="normal-case tracking-normal text-[var(--ink)]">
                        {pane.name}
                      </span>
                    ) : null}
                  </p>
                  {pane.state === "loading" ? (
                    <p className="rounded bg-white p-4 text-[0.8125rem] text-[var(--muted)]">
                      Loading answer…
                    </p>
                  ) : pane.state === "answer" ? (
                    <div className="max-h-[32rem] overflow-auto break-words rounded bg-white p-4 text-[0.8125rem] leading-6 text-[var(--ink)]">
                      <MathText text={pane.text as string} />
                    </div>
                  ) : (
                    <p className="rounded bg-white p-4 text-[0.8125rem] text-[var(--muted)]">
                      No answer recorded for this model on this item.
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
