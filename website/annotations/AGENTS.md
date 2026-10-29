# Benchmark annotations — working notes

State of the private-benchmark verifiability annotations and the website pages
that display them. Read this before touching anything under `/annotations`.

## What this is

A survey of how far private AI benchmarks can be checked from outside, coded by
three annotators against **Codebook v3.1**. It backs the paper *Trustworthy
Private AI Benchmarks Require External Verification*.

The instrument codes **public verifiability** — how strongly available evidence
lets outsiders assess a benchmark's claims. Per the codebook's own rule, scores
describe *the strength of verifiability evidence, not the benchmark's actual
quality, fairness, or validity*. A `Not located` code means no supporting
evidence was found after an adequate search; it does **not** establish that a
practice is absent. Website copy must not drift into reading these as a quality
ranking.

Two framing points from the paper that the pages depend on:

- **Durability is not a fourth claim.** It is the constraint that motivates
  withholding information, so it sets the disclosure limits the other claims
  work within. It is descriptive and explicitly excluded from any scale score.
- **Verifiability is not verification.** Making a claim checkable is not the
  same as someone having checked it.

## Current state

**Focus is FrontierMath.** Finish it properly before spreading effort across the
other three — if the framing gets challenged, we'd rather have paid that cost
once.

| Benchmark | Context | Items | Summary | Tree |
|---|---|---|---|---|
| FrontierMath | done | 18 leaves, verbatim + tallies | done | done |
| SWE-Bench Pro | done | — | — | — |
| METR Time Horizon / HCAST | done | — | — | — |
| GDPval | done | — | — | — |

All four are annotated in all five CSV parts by all three annotators. Only
FrontierMath has been brought through to the site.

## Where things live

```
website/annotations/
  Part 1..5 *.csv            raw form exports — the record, tracked in git
  frontiermath-sources.md    38 normalized sources from the FrontierMath records
  test.html                  standalone mock the design came from (gitignored)

website/content/annotations.ts         all page data, hand-maintained
website/components/annotation-mindmap.tsx   the tree, layout computed at build
website/app/annotations/               index + /[slug] report pages
```

Nothing fetches Google Sheets at build time. The CSVs are read by hand when
regenerating figures, and the results are written into `annotations.ts`.

**No coordinates live in the data.** The mindmap computes its own layout: leaf
blocks size themselves from their content, a band sits at the mean anchor of its
leaves, the root at the middle. Adding a leaf reflows the diagram. Text is
wrapped at build time from a character-width estimate, because SVG `<text>` does
not wrap — if a line overflows, adjust `CHARS` in the component.

## Agreement figures

Majority share per selectable box: the fraction of annotators agreeing with the
most common answer. A box nobody ticked counts — all three agreed to leave it,
so it scores 1.0.

- **QAA / QRAA** — per question: observed, and the same question answered at
  random. QRAA is 0.750 for any checkbox question (each box is an independent
  y/n: 2 unanimous + 6 split × 2/3, over 8 outcomes). For single-select with N
  options it is `(N² + 3N − 1) / 3N²`.
- **BAA / RAA** — the same two pooled across the instrument, kept separate for
  multi- and single-select because their baselines differ.

FrontierMath, current: multi 0.9348 over 92 boxes (random 0.750); single 0.8333
over 4 questions (random 0.5949).

Per-question values reproduce the working sheet exactly. The pooled multi figure
does not — see the open questions below.

## What's left

**FrontierMath**

1. Fill the Validity, Integrity, Neutrality and Verification cells in the
   summary table from the coded evidence. Several findings in the CSVs never
   made it into the report card — see the open-questions block on the page.
2. Resolve the open questions listed there (9 at last count).
3. Decide whether the per-leaf `detail` sentences should carry the tallies'
   nuance, or stay short with the tree doing that work.

**The other three benchmarks**

4. Transcribe context + leaves + tallies the same way. The CSV parsing is
   straightforward; the editorial statuses are not — each needs a curator to
   read the evidence notes, not just the selections.

**Pipeline**

5. The CSV → `annotations.ts` path is currently manual. If this grows past four
   benchmarks it wants a generator under `scripts/render_website/`, following
   the existing `generate_*.py` → `content/generated/*.json` pattern.
6. Option lists come from the Codebook v3.1 sheet, not from the responses — a
   checkbox nobody ticked is invisible in the exports. Any generator needs the
   codebook as a second input.

## Open questions

- **Codebook version.** The forms are v3.1 (Validity 4 items, Integrity 4,
  Neutrality 3, Verification 4). Codebook v4 has 5/5/4 plus Durability and
  Verification subscales. The site reports v3.1. Decide which is canonical
  before any subscale percentages are published.
- **Three claims or five bands.** The summary table shows four claims; the tree
  shows all five codebook parts. Both are defensible together, but the labels
  and band order must agree.
- **Pooled BAA doesn't reconcile.** 92 boxes agreed, share totals don't: 86.0
  here against 85.3 in the working sheet. 85.3 cannot be produced by majority
  shares of 1 and 2/3 over 92 binary boxes — it needs 20.1 split boxes — so
  roughly two boxes are treated differently somewhere. Likely the blank
  handling on `Verification scope`, where two annotators left the question
  empty and this code reads empty as "nothing ticked".
- **`other` / `unclear` boxes** are currently scored, which is what gets the
  count to 92. They're free text, so agreement on them is not well defined, and
  the Part 1 working sheet leaves them blank instead.
- **Conditional questions.** `Verification scope` enters the pooled BAA but has
  no leaf, because it only applies when an external check occurred.

## Conventions

- Annotator disagreement is a property of our coding process, not a finding
  about the benchmark. Keep it out of strengths/concerns.
- "What would make it checkable" is deliberately not "what would fix it":
  disclosure trades against durability, and a claim can be made checkable
  through controlled inspection or non-disclosing verification rather than
  publication.
- Questions and option codes are stored **verbatim** from the forms. Counting is
  case-insensitive (coders typed both `items/tasks` and `Items/tasks`); the most
  common spelling is kept.
- Blocks marked as working notes — the summary brief and the open-questions
  panel — are drafting aids and come out before publication.
