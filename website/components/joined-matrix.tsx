"use client";

import { useEffect, useMemo, useRef, useState } from "react";

// Draw the joined_matrix layout supplied with the page.

const PASS: [number, number, number] = [31, 119, 180];
const FAIL: [number, number, number] = [214, 39, 40];
const NONE: [number, number, number] = [233, 232, 228];
// Unobserved on a GRADED matrix. The binary NONE above is nearly the same value
// as graded_color()'s fog midpoint (#f2f2f2), so on a graded scale a mid-scale
// response and an unmeasured cell would look alike. The PNG matrices already
// solve this by painting unobserved as this grey, and the page's non-binary
// legend already shows that swatch — so match it rather than invent a third.
const NONE_GRADED: [number, number, number] = [158, 158, 158];

/** One entry per distinct response value, in ascending order — index into this
 *  is exactly the character stored in a cell. Absent for binary benchmarks. */
type Level = { value: number; label: string; color: string };

const LEVEL_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz";

function hexToRgbTriple(hex: string): [number, number, number] {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** The colour for one stored cell character. Graded benchmarks index into the
 *  `levels` the build emitted (already coloured by graded_color, so identical to
 *  the PNG); binary keeps the pass/fail pair. */
function cellColor(
  ch: string,
  ramp: [number, number, number][] | null,
  none?: [number, number, number] | null,
): [number, number, number] {
  if (ch === ".") return none ?? (ramp ? NONE_GRADED : NONE);
  if (!ramp) return ch === "1" ? PASS : FAIL;
  const i = LEVEL_CHARS.indexOf(ch);
  return ramp[i] ?? NONE_GRADED;
}

type Block = { key: string; colIds: string[] };
export type ObservationKey = [
  subjectId: string,
  itemId: string,
  condition: string | null,
  trial: number,
  interactors?: string | null,
];

export type CellKeys = {
  key?: ObservationKey;
  key2?: ObservationKey;
  keys?: Record<number, ObservationKey>;
  keys2?: Record<number, ObservationKey>;
};

export function observationKey(
  cells: CellKeys,
  index: number,
  itemId: string,
  second = false,
): ObservationKey | undefined {
  const exact = (second ? cells.keys2 : cells.keys)?.[index];
  if (exact) return exact;
  const shared = second ? cells.key2 : cells.key;
  if (!shared) return undefined;
  const key: ObservationKey = [...shared];
  key[1] = itemId;
  return key;
}

type RowBlock = CellKeys & {
  bits: string;
  cond: string | null;
  /** Pair rows only: the SECOND model's condition for the same cell, which is
   *  the mirror of `cond` (each side of a battle records the other as its
   *  opponent). Needed to look up that model's answer. */
  cond2?: string;
};
type JoinedRow = {
  sid: string;
  band: string;
  rowcond: string;
  blocks: Record<string, RowBlock>;
  pass: number;
  n: number;
  /** Pair rows only — see JoinedData.rowKind. */
  sid2?: string;
  /** An explicit row label. Pair rows carry one because a pair has no subject
   *  whose display name would do. */
  name?: string;
  /** An explicit right-gutter score, when `pass`/`n` do not describe the row. */
  score?: string;
  /** Per-row item ids, positionally aligned with `bits`. Pair rows need this
   *  because every row covers a DIFFERENT set of items — a column position is
   *  not a shared item — so the block's single colIds axis cannot serve them. */
  cols?: string[];
};
export type JoinedData = {
  /** Per-dim value→label map from benchmark-overrides.json, e.g.
   *  { speech_complexity: { control: "Clean voice" } }. Raw dim values like
   *  "control_audio" are unreadable on their own; the override is hand-written
   *  per benchmark so this component stays generic. */
  bandLabels?: Record<string, Record<string, string>>;
  blockDim: string | null;
  /** Noun for the per-block column count, e.g. "environments". Null keeps the
   *  bare number every other benchmark shows. */
  blockUnit?: string | null;
  bandDims: string[];
  dimKinds: Record<string, string>;
  blocks: Block[];
  trials: Record<string, JoinedRow[]>;
  /** Graded scales only. Absent ⇒ binary, and cells are the "0"/"1" pair. */
  levels?: Level[];
  /** What one row IS. Absent or "subject" is the normal matrix. "pair" is the
   *  crossed-opponent format written by write_pairwise: a row is two models and
   *  its cells are the items they were compared on, so there is no single
   *  ability to show and the label needs more room than a model name. */
  rowKind?: "subject" | "pair" | "session" | "faceted";
  /** Overrides the colour of an unobserved cell. Pair rows pad short bars to
   *  the track width, and the graded default (#9e9e9e) is close enough to this
   *  matrix's tie grey that the padding would read as a tied battle. */
  noneColor?: string;
};

export type JoinedPick = {
  key?: ObservationKey;
  key2?: ObservationKey;
  subjectId: string;
  itemId: string;
  cond: string | null;
  trial: string;
  passed: boolean;
  /** Pair rows only: the other model in the comparison, and its own condition
   *  for this cell. A battle has two answers, and showing only the first-named
   *  model's would hide the half the reader clicked to compare against. */
  subjectId2?: string;
  cond2?: string | null;
  /** Graded scales only: the level the cell actually holds, so the detail panel
   *  can name it instead of rounding the response to passed/failed. */
  level?: { label: string; color: string } | null;
};

type Props = {
  data: JoinedData;
  rowIds: string[] | null;
  rows: string[];
  onPick: (p: JoinedPick) => void;
  /** subject_id -> fitted Rasch ability, from the benchmark's IRT fit. When
   *  present it becomes a leading θ column, mirroring the score column on the
   *  right. Null for graded benchmarks, which are not fitted. */
  theta?: Record<string, number> | null;
  /** What 1 and 0 MEAN on this benchmark, when they are not pass/fail.
   *
   *  Several benchmarks measure the presence of a bad outcome, so 1 is the worse
   *  result: helm_bold and helm_real_toxicity_prompts both set
   *  { one: "toxic continuation", zero: "no toxic continuation" }. Without this
   *  the hover tip calls a toxic continuation "passed" and paints it the same
   *  blue as a correct answer — stating the opposite of the measurement. The PNG
   *  view has honoured this since it shipped; the joined view did not receive it
   *  at all, so the two views contradicted each other on the same page. */
  binaryLabels?: { zero: string; one: string } | null;
  zByItem?: Record<string, number> | null;
};

type BandPart = { setting: string | null; text: string };

/** One band header, split into its settings.
 *
 *  A hand-written label in benchmark-overrides.json stands on its own —
 *  "speech_complexity=control" → "Clean voice", where naming the setting again
 *  would only add noise. Anything WITHOUT a label keeps its setting name, because
 *  the bare value is usually unreadable: τ²-bench's five condition dims render as
 *  "default · default · gpt-4.1-2025-04-14 · 964ef7ae · n/a", where nothing says
 *  which "default" is which, that the model name is the user simulator rather than
 *  the subject, or that the hex is a build id. Prefixing each with its setting is
 *  the floor, not the goal — the goal is that someone writes the labels. */
function bandParts(
  band: string,
  labels?: Record<string, Record<string, string>>,
): BandPart[] {
  if (!band) return [{ setting: null, text: "all responses" }];
  return band.split(";").map((p) => {
    const i = p.indexOf("=");
    if (i < 0) return { setting: null, text: p };
    const k = p.slice(0, i);
    const v = p.slice(i + 1);
    const mapped = labels?.[k]?.[v];
    return mapped ? { setting: null, text: mapped } : { setting: k, text: v };
  });
}

/** Flat text of the same header, for aria-label and title attributes. */
function bandLabel(
  band: string,
  labels?: Record<string, Record<string, string>>,
): string {
  return bandParts(band, labels)
    .map((p) => (p.setting ? `${p.setting}: ${p.text}` : p.text))
    .join(" · ");
}

/** Fill value generate_benchmark_gallery writes for a row whose condition lacks a dim. */
const MISSING_DIM = "n/a";

/** A column block's heading, split into the name and an optional gloss.
 *
 *  Block keys are raw setting values, and a raw value is often a code word:
 *  ALE's item axis blocks on `tier`, whose values `near-term`, `full-spectrum`
 *  and `last-exam` name difficulty tiers a reader has no way to decode. The
 *  same `bandLabels` map that names band values names these — keyed by the
 *  block dim — and a label may carry a "Name · what it means" gloss, whose
 *  second half renders muted so the heading still scans short. */
function blockLabel(
  key: string,
  blockDim: string | null | undefined,
  labels: Record<string, Record<string, string>> | undefined,
): [string, string | null] {
  const text = (blockDim && labels?.[blockDim]?.[key]) || key;
  const at = text.indexOf(" · ");
  return at < 0 ? [text, null] : [text.slice(0, at), text.slice(at + 3)];
}

/** The settings that made this row its own row but are NOT already named by the
 *  band header above it: model variants (reasoning effort, temperature,
 *  scaffold version) and any item dim that lost the column-block slot.
 *
 *  Without this the row label is the model's display name alone, so a model
 *  swept over two reasoning efforts renders as two adjacent rows reading
 *  "Claude Opus 4.5" and "Claude Opus 4.5" with nothing to tell them apart.
 *  `condition` dims are dropped (the band header already states them) and
 *  `constant` dims are dropped (identical on every row, so pure noise) — what
 *  is left is exactly what distinguishes two rows sharing one model name.
 *  Rendered in brackets after the name: "Claude Opus 4.5 (effort=high)". */
function variantLabel(
  rowcond: string,
  dimKinds: Record<string, string>,
  varying: Set<string> | undefined,
  labels?: Record<string, Record<string, string>>,
  subjectName?: string,
  nameIsAmbiguous?: boolean,
): string {
  if (!rowcond) return "";
  const parts: string[] = [];
  for (const p of rowcond.split(";")) {
    const i = p.indexOf("=");
    if (i < 0) continue;
    const k = p.slice(0, i);
    const v = p.slice(i + 1);
    const kind = dimKinds?.[k];
    if (kind === "condition" || kind === "constant") continue;
    // Constant across this band, so the band header already pins it down.
    if (!varying?.has(k)) continue;
    // This row's condition simply does not carry the dim — a bracket reading
    // "effort=n/a" is less informative than no bracket at all.
    if (v === MISSING_DIM || v === "") continue;
    const shown = labels?.[k]?.[v] ?? v;
    // The loader joins routed subject features back into the display dims:
    // the scaffold both distinguishes `gpt-5-5 (Codex)` as a subject and
    // remains queryable without parsing the label. Printing both would give a
    // redundant "gpt-5-5 (OpenClaw) (harness=OpenClaw)" bracket, so omit it
    // when the subject name already says it.
    if (!nameIsAmbiguous && namedBy(subjectName, shown, v)) continue;
    parts.push(`${k}=${shown}`);
  }
  return parts.join(", ");
}

/** Does the subject's display name already carry this dim value?
 *
 *  Compared on alphanumerics only, so `harness=openclaw` matches the label
 *  "OpenClaw" inside "gpt-5-5 (OpenClaw)" and punctuation/case differences
 *  between the build's raw value, the override's label, and the display name
 *  do not defeat the match. Both the rendered label and the raw value are
 *  tried, since an override may rename a value the subject label spells raw.
 *
 *  Only ever consulted when the name is unique within its band — see
 *  `duplicateNamesByBand`. A coincidental substring (effort `max` inside the
 *  model "Qwen3.7 Max") would otherwise hide the one thing telling two rows
 *  apart; requiring uniqueness first means the worst case is a dropped
 *  bracket on a row nothing else collides with. */
function namedBy(
  name: string | undefined,
  shown: string,
  raw: string,
): boolean {
  if (!name) return false;
  const flat = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const hay = flat(name);
  if (!hay) return false;
  return [shown, raw].some((v) => {
    const needle = flat(v);
    return needle.length > 0 && hay.includes(needle);
  });
}

/** band → the display names that more than one row in it carries.
 *
 *  A name that appears twice in a band is exactly the case the variant bracket
 *  exists for, so nothing may be dropped from those rows' brackets. */
function duplicateNamesByBand(
  rows: JoinedRow[],
  label: (sid: string) => string,
): Map<string, Set<string>> {
  const counts = new Map<string, Map<string, number>>();
  for (const row of rows) {
    let seen = counts.get(row.band);
    if (!seen) {
      seen = new Map();
      counts.set(row.band, seen);
    }
    const name = label(row.sid);
    seen.set(name, (seen.get(name) ?? 0) + 1);
  }
  const out = new Map<string, Set<string>>();
  for (const [band, seen] of counts) {
    const dupes = new Set<string>();
    for (const [name, n] of seen) if (n > 1) dupes.add(name);
    out.set(band, dupes);
  }
  return out;
}

/** band → the row dims that actually take more than one value inside it.
 *
 *  A dim that is constant throughout a band is already implied by the band
 *  header, so bracketing it onto every row is noise: every one of tau_voice's
 *  voice bands is scaffold=discrete_time_audio_native_agent, and printing that
 *  on all 24 rows says nothing the header did not. Only a dim that DIFFERS
 *  between two rows of the same band is something a reader cannot otherwise
 *  resolve — that is the one worth spending label width on. */
function varyingDimsByBand(rows: JoinedRow[]): Map<string, Set<string>> {
  const seen = new Map<string, Map<string, Set<string>>>();
  for (const row of rows) {
    let dims = seen.get(row.band);
    if (!dims) {
      dims = new Map();
      seen.set(row.band, dims);
    }
    for (const p of row.rowcond.split(";")) {
      const i = p.indexOf("=");
      if (i < 0) continue;
      const k = p.slice(0, i);
      let vals = dims.get(k);
      if (!vals) {
        vals = new Set();
        dims.set(k, vals);
      }
      vals.add(p.slice(i + 1));
    }
  }
  const out = new Map<string, Set<string>>();
  for (const [band, dims] of seen) {
    const multi = new Set<string>();
    for (const [k, vals] of dims) if (vals.size > 1) multi.add(k);
    out.set(band, multi);
  }
  return out;
}

/** Item difficulty across one block's columns, drawn under the matrix.
 *
 *  Diverging from a zero line, which is meaningful and not decoration: theta is
 *  mean-centred, so z = 0 is the difficulty at which the AVERAGE subject has an
 *  even chance. Bars below the line are items the average model handles, above
 *  are ones it does not.
 *
 *  Unlike the response strips above it, this one is NOT interactive — the exact
 *  z for a column reaches the reader through the cell tooltip, which already
 *  names the item. */
const Z_HEIGHT = 26;

function ZStrip({ z }: { z: (number | null)[] }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const n = z.length;
    cv.width = n;
    cv.height = Z_HEIGHT;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, n, Z_HEIGHT);

    const mid = Math.round(Z_HEIGHT / 2);
    ctx.fillStyle = "rgba(0,0,0,0.14)";
    ctx.fillRect(0, mid, n, 1);

    // 90th percentile, not the max: an item nobody solved has no finite
    // difficulty, and letting those few set the scale flattens the part of the
    // curve worth reading down to a couple of pixels.
    const mags = z
      .filter((v): v is number => v != null)
      .map(Math.abs)
      .sort((a, b) => a - b);
    if (!mags.length) return;
    const span = Math.max(
      1e-6,
      mags[Math.min(mags.length - 1, Math.floor(0.9 * (mags.length - 1)))],
    );
    const half = mid - 1;
    ctx.fillStyle = "rgba(0,0,0,0.24)";
    z.forEach((v, i) => {
      if (v == null) return;
      const len = Math.max(
        -half,
        Math.min(half, Math.round((v / span) * half)),
      );
      if (len >= 0) ctx.fillRect(i, mid, 1, Math.max(1, len));
      else ctx.fillRect(i, mid + len, 1, Math.max(1, -len));
    });
  }, [z]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Item difficulty (z) across this block, hardest items lowest"
      className="block w-full"
      style={{ height: Z_HEIGHT }}
    />
  );
}

function Strip({
  bits,
  ramp,
  none,
  onHover,
  onLeave,
  onPick,
  label,
}: {
  bits: string;
  ramp: [number, number, number][] | null;
  none: [number, number, number] | null;
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
      const c = cellColor(bits[i], ramp, none);
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [bits, ramp, none]);

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

export function JoinedMatrix({
  data,
  rowIds,
  rows,
  onPick,
  theta,
  binaryLabels,
  zByItem,
}: Props) {
  const [selectedTrial, setTrial] = useState<string | null>(null);
  const trial =
    selectedTrial ??
    (data
      ? Object.keys(data.trials).sort((a, b) => Number(a) - Number(b))[0]
      : null);
  const [tip, setTip] = useState<{
    level?: { label: string; color: string } | null;
    x: number;
    y: number;
    text: string;
    passed: boolean;
    z?: number | null;
  } | null>(null);

  const nameOf = useMemo(() => {
    const m = new Map<string, string>();
    if (rowIds) rowIds.forEach((id, i) => m.set(id, rows[i] ?? id));
    return m;
  }, [rowIds, rows]);

  // A miss here means the caller handed us a row axis that does not cover every
  // subject in the payload (e.g. the selected slice's rows rather than the
  // global ones). Showing the raw subject_id hash is useless to a reader, so say
  // so plainly and shorten it.
  const label = (sid: string) =>
    nameOf.get(sid) ?? `unknown subject ${sid.slice(0, 6)}`;

  if (data === undefined || data === null || !trial) return null;

  const list = data.trials[trial] ?? [];
  const trialKeys = Object.keys(data.trials).sort(
    (a, b) => Number(a) - Number(b),
  );
  // A row is two models, not one: there is no single ability to head it with
  // (θ belongs to a subject), and "gemini-2.5-flash vs claude-opus-4-20250514"
  // needs more room than a model name.
  const pairwise = data.rowKind === "pair";
  const noneTriple = data.noneColor ? hexToRgbTriple(data.noneColor) : null;

  // Blocks are sized by item count, so the widths stay honest: a 50-item block
  // is visibly narrower than a 114-item one.
  // θ leads the row when the benchmark has a Rasch fit, so the row reads
  // ability → responses → observed score, left to right.
  //
  // BLOCK_MIN is a floor, not a width. While the blocks fit, `fr` divides the
  // available space and nothing scrolls — the normal case, and the one that
  // should never grow a scrollbar. Only when the floors no longer fit does the
  // grid overflow its `overflow-x-auto` parent and scroll horizontally, which is
  // strictly better than the alternative it replaces: mmlu's 57 tasks squeezed
  // to ~12px each, thinner than the word "law", under headings that cannot be
  // read at all. The figure's own width is unchanged either way, so a wide
  // benchmark scrolls rather than making every page wider.
  const BLOCK_MIN = 60; // px — roughly what a block heading needs to be legible
  const grid = `${theta && !pairwise ? "40px " : ""}${
    pairwise ? "268px" : "170px"
  } ${data.blocks
    .map((b) => `minmax(${BLOCK_MIN}px, ${b.colIds.length}fr)`)
    .join(" ")} 60px`;

  const zOfBlock = new Map<string, (number | null)[]>();
  for (const block of data.blocks) {
    const z = block.colIds.map((id) => zByItem?.[id] ?? null);
    if (z.some((value) => value !== null)) zOfBlock.set(block.key, z);
  }
  // Never on a pairwise matrix: there each ROW carries its own `cols`, so a
  // column position is not a shared item and a single strip beneath the block
  // would describe none of the rows above it.
  const hasZ = zOfBlock.size > 0 && !pairwise;

  // Precompute where a band starts. Mutating a `lastBand` local while mapping
  // would be a render-phase mutation (react-hooks/immutability).
  const withBand = list.map((row, i) => ({
    row,
    showBand: i === 0 || row.band !== list[i - 1].band,
  }));
  const bandVarying = varyingDimsByBand(list);
  const bandDuplicateNames = duplicateNamesByBand(list, label);

  const levels = data.levels ?? null;
  const ramp = levels ? levels.map((l) => hexToRgbTriple(l.color)) : null;
  const levelAt = (ch: string) => {
    if (!levels || ch === ".") return null;
    const l = levels[LEVEL_CHARS.indexOf(ch)];
    return l ? { label: l.label, color: l.color } : null;
  };
  /* `row.pass` is the SUM of the row's responses. On a binary scale that is the
     number passed, so "12/89" reads correctly. On a graded scale it is a point
     total, and "150/89" is nonsense — the same two numbers say something true as
     a mean, which is what the row's score actually is. */
  /* A pair row states its own score, because neither reading above fits: it is
     graded (four outcomes, so `levels` is set) but the number that means
     something is a COUNT — how many of the pair's items the first-named model
     won — not a mean over a scale. */
  const rowScore = (row: JoinedRow) =>
    row.score ??
    (levels
      ? `${(row.pass / Math.max(1, row.n)).toFixed(2)} avg`
      : `${row.pass}/${row.n}`);

  return (
    <div className="mb-5">
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="flex items-center gap-2 text-[0.8125rem] text-[var(--muted)]">
          Trial
          <select
            value={trial}
            onChange={(e) => {
              setTip(null);
              setTrial(e.target.value);
            }}
            className="rounded border border-[var(--line)] bg-white px-2 py-1 text-[0.8125rem] text-[var(--ink)]"
          >
            {trialKeys.map((k) => (
              <option key={k} value={k}>
                trial {k} ({data.trials[k].length} rows)
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* The figure scrolls in its own box rather than down the page. A
          250-subject benchmark is several screens tall, and clicking a cell
          fills the detail panel BELOW this box — which the reader could only
          reach by scrolling past every remaining row. Capping the box keeps the
          cell and its content on screen together. Padding moves off the
          container and onto the header so the header can stick flush to the
          top: its own pt-4 is what rows pass under. */}
      <div className="max-h-[70vh] overflow-auto rounded-md border border-[var(--line)] bg-white px-4 pb-4">
        <div className="min-w-[640px]">
          <div
            className="sticky top-0 z-10 grid items-center gap-x-2.5 border-b border-[var(--line)] bg-white pt-4"
            style={{ gridTemplateColumns: grid }}
          >
            {theta ? (
              <div
                className="pb-1 text-right text-[0.6875rem] text-[var(--muted)]"
                title="Fitted ability (theta), per subject"
              >
                θ
              </div>
            ) : null}
            <div />
            {data.blocks.map((b) => {
              const [head, gloss] = blockLabel(
                b.key,
                data.blockDim,
                data.bandLabels,
              );
              return (
                <div
                  key={b.key}
                  /* min-w-0 lets the heading wrap inside its own track. A grid
                     item defaults to min-width:auto, so a long heading pushes
                     past the block it belongs to and overlaps its neighbour
                     rather than wrapping — arc_agi_3's level headings collided
                     exactly that way. */
                  className="min-w-0 break-words pb-1 text-[0.6875rem] text-[var(--ink)]"
                >
                  {head}
                  {/* The bare count is ambiguous once the block name does not
                     imply what is being counted ("Level 8  4"). `blockUnit`
                     names the unit, so the heading reads "Level 8, 4
                     environments"; benchmarks that leave it unset keep the
                     bare number. */}
                  <span
                    className={`text-[var(--muted)] tabular-nums ${
                      data.blockUnit ? "" : "ml-1"
                    }`}
                  >
                    {data.blockUnit
                      ? `, ${b.colIds.length} ${data.blockUnit}`
                      : b.colIds.length}
                  </span>
                  {/* The gloss says what the block MEANS — `last-exam` alone does
                      not tell a reader it is the hardest tier. Muted and after
                      the count, so the block still scans as a short heading. */}
                  {gloss ? (
                    <span className="ml-1 font-normal text-[var(--muted)]">
                      {gloss}
                    </span>
                  ) : null}
                </div>
              );
            })}
            <div />
          </div>

          {withBand.map(({ row, showBand }) => {
            const subjectName = label(row.sid);
            // A row that names itself needs no variant bracket: the bracket
            // exists to tell two rows sharing a model name apart, and a pair
            // label already carries both models.
            const variant = row.name
              ? ""
              : variantLabel(
                  row.rowcond,
                  data.dimKinds,
                  bandVarying.get(row.band),
                  data.bandLabels,
                  subjectName,
                  bandDuplicateNames.get(row.band)?.has(subjectName),
                );
            const name =
              row.name ??
              (variant ? `${subjectName} (${variant})` : subjectName);
            return (
              <div key={`${row.sid}|${row.sid2 ?? ""}|${row.rowcond}`}>
                {showBand ? (
                  <div className="mb-1.5 mt-3 flex flex-wrap items-baseline gap-x-1.5 border-b border-[var(--line)] pb-1 text-[0.75rem] font-semibold text-[var(--ink)]">
                    {bandParts(row.band, data.bandLabels).map((part, i) => (
                      <span
                        key={`${part.setting ?? ""}|${part.text}`}
                        className="whitespace-nowrap"
                      >
                        {i > 0 ? (
                          <span className="mr-1.5 font-normal text-[var(--line)]">
                            ·
                          </span>
                        ) : null}
                        {/* The setting name is context, the value is the content —
                            so the name recedes and the value keeps the weight. */}
                        {part.setting ? (
                          <span className="font-normal text-[var(--muted)]">
                            {part.setting}{" "}
                          </span>
                        ) : null}
                        {part.text}
                      </span>
                    ))}
                  </div>
                ) : null}
                <div
                  className="grid items-center gap-x-2.5 py-[1.5px]"
                  style={{ gridTemplateColumns: grid }}
                >
                  {/* θ is per SUBJECT, not per row: a benchmark whose rows are
                      (subject × condition) repeats the same ability down the
                      band, which is exactly what the fit says — one ability,
                      several conditions. */}
                  {theta && !pairwise ? (
                    <div className="text-right text-[0.6875rem] tabular-nums text-[var(--muted)]">
                      {theta[row.sid] != null ? theta[row.sid].toFixed(2) : ""}
                    </div>
                  ) : null}
                  {/* The label column is a fixed 170px, so a name carrying a
                      variant bracket clips — `title` keeps the full text
                      reachable on hover rather than silently losing the part
                      that distinguishes this row from the one above it. */}
                  <div
                    title={name}
                    className="truncate text-right text-[0.6875rem] text-[var(--ink)]"
                  >
                    {name}
                  </div>
                  {data.blocks.map((b) => {
                    const rb = row.blocks[b.key];
                    // Screen readers and the hover tip get the readable name too,
                    // not the raw key the label above no longer shows.
                    const blockName = blockLabel(
                      b.key,
                      data.blockDim,
                      data.bandLabels,
                    )[0];
                    if (!rb)
                      return (
                        <div
                          key={b.key}
                          className="flex h-[15px] items-center justify-center rounded-[1px] bg-[var(--fog-light)] text-[0.5625rem] text-[var(--muted)]"
                        >
                          not run
                        </div>
                      );
                    return (
                      <Strip
                        key={b.key}
                        bits={rb.bits}
                        ramp={ramp}
                        none={noneTriple}
                        label={`${name}, ${bandLabel(row.band, data.bandLabels)}, ${blockName}: ${rowScore(row)}`}
                        onHover={(idx, e) => {
                          const ch = rb.bits[idx];
                          setTip({
                            x: e.clientX,
                            y: e.clientY,
                            text: `${name} · ${blockName} · item ${idx + 1}`,
                            passed: ch === "1",
                            level: levelAt(ch),
                            // The strip below shows the shape of the difficulty
                            // curve; this is where one item's z is readable.
                            z: zOfBlock.get(b.key)?.[idx] ?? null,
                          });
                        }}
                        onLeave={() => setTip(null)}
                        onPick={(idx) => {
                          const ch = rb.bits[idx];
                          if (ch === ".") return;
                          // A pair row's items differ per row, so its own
                          // `cols` axis is authoritative; the block's shared
                          // axis serves every other benchmark.
                          const itemId = row.cols?.[idx] ?? b.colIds[idx];
                          if (!itemId) return;
                          onPick({
                            key: observationKey(rb, idx, itemId),
                            key2: observationKey(rb, idx, itemId, true),
                            subjectId: row.sid,
                            subjectId2: row.sid2,
                            itemId,
                            cond: rb.cond,
                            cond2: rb.cond2 ?? null,
                            trial,
                            passed: ch === "1",
                            level: levelAt(ch),
                          });
                        }}
                      />
                    );
                  })}
                  <div className="text-[0.6875rem] tabular-nums text-[var(--muted)]">
                    {rowScore(row)}
                  </div>
                </div>
              </div>
            );
          })}

          {/* Item difficulty, once at the bottom: z is a property of the item,
              so it is the same for every band above it. Same grid template, so
              each block's strip sits exactly under its own columns. */}
          {hasZ ? (
            <div
              className="mt-3 grid items-center gap-x-2.5 border-t border-[var(--line)] pt-2"
              style={{ gridTemplateColumns: grid }}
            >
              {theta && !pairwise ? <div /> : null}
              <div
                className="truncate text-right text-[0.6875rem] text-[var(--muted)]"
                title="Fitted item difficulty (z). Above the line: harder than the average subject can handle."
              >
                item difficulty z
              </div>
              {data.blocks.map((b) => {
                const z = zOfBlock.get(b.key);
                return z ? <ZStrip key={b.key} z={z} /> : <div key={b.key} />;
              })}
              <div />
            </div>
          ) : null}
        </div>
      </div>

      {tip ? (
        <div
          className="pointer-events-none fixed z-30 rounded border border-[var(--line)] bg-white px-2.5 py-1.5 text-[0.75rem] shadow-lg"
          style={{ left: tip.x + 12, top: tip.y - 38 }}
        >
          <div className="text-[var(--ink)]">
            {tip.text}
            {tip.z != null ? (
              <span className="text-[var(--muted)]">
                {` · z ${tip.z.toFixed(2)}`}
              </span>
            ) : null}
          </div>
          {/* A graded cell names its level; only binary reduces to passed/failed.
              The fog midpoint is near-white, so the swatch carries the colour and
              the text stays in ink rather than becoming unreadable. */}
          {tip.level ? (
            <div className="flex items-center gap-1.5 font-semibold text-[var(--ink)]">
              <span
                className="inline-block h-2.5 w-2.5 rounded-[2px] border border-[var(--line)]"
                style={{ backgroundColor: tip.level.color }}
              />
              {tip.level.label}
            </div>
          ) : (
            <div
              className="font-semibold"
              style={{
                color: tip.passed ? "rgb(31,119,180)" : "rgb(214,39,40)",
              }}
            >
              {tip.passed
                ? (binaryLabels?.one ?? "passed")
                : (binaryLabels?.zero ?? "failed")}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
