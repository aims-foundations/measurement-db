# Benchmark annotations — working notes

State of the private-benchmark verifiability annotations and the website pages
that display them. Read this before touching anything under `/annotations`.

## What this is

A survey of how far private AI benchmarks can be checked from outside, coded by
three annotators against **Codebook v3.1** (final — v4 exists in the sheet but is
not what was coded). It backs the paper *Trustworthy Private AI Benchmarks
Require External Verification*.

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
  work within. It is descriptive and excluded from any scale score.
- **Verifiability is not verification.** Making a claim checkable is not the
  same as someone having checked it.

## Current state

**Focus is FrontierMath.** Finish it properly before spreading effort across the
other three — if the framing gets challenged, we'd rather have paid that cost
once.

| Benchmark | Context | Tree | Summary table | Evidence quality |
|---|---|---|---|---|
| FrontierMath | done | 18 leaves | 4 claims filled | 129 codes rated |
| SWE-Bench Pro | done | — | — | — |
| METR Time Horizon / HCAST | done | — | — | — |
| GDPval | done | — | — | — |

All four are annotated in all five CSV parts by all three annotators. Only
FrontierMath has been brought through to the site.

**Annotators are anonymised to initials** — MZ, AP, NT — everywhere in
`annotations.ts` and on the page. `frontiermath-sources.md` still spells the
names out in its scope section and per-source attributions; decide whether that
should be stripped before the repo goes public. The site footer also credits a
team member by name, which is unrelated to annotator anonymity.

## Page structure (`/annotations/frontiermath`)

```
header            operator · domain · version · stamp-free meta row
                  Hidden / Run by / Release / Verification (links only when found)
                  summary + its drafting brief
By claim          4 rows: Validity, Integrity, Neutrality, Verification
                  × Strengths | Concerns | What would make it checkable
By item           mindmap (computed layout) with verbatim questions, selection
                  tallies, QAA/QRAA per question, BAA/RAA under the root
                  → working-notes panel, 9 open
Evidence quality  rubric pulldown (verbatim) → heatmap → All codes + Summary slot
                  → band pulldowns → question → code → per-annotator scores
                  → working-notes panel, 1 open
Sources           38 titled links
```

## Where things live

```
website/annotations/
  Part 1..5 *.csv            raw form exports — the record, tracked in git
  Evidence quality rubric.pdf  the rubric; its text is transcribed into the site
  frontiermath-sources.md    38 normalized sources from the FrontierMath records
  test.html                  standalone mock the design came from (gitignored)

website/content/annotations.ts            all page data, hand-maintained
website/components/annotation-mindmap.tsx tree, layout computed at build
website/components/evidence-quality.tsx   rubric + band/question/code pulldowns
website/components/evidence-heatmap.tsx   question × dimension grid
website/app/annotations/                  index + /[slug] report pages
```

Nothing fetches Google Sheets at build time. The CSVs are parsed by hand when
regenerating figures, and the results are written into `annotations.ts`.

**No coordinates live in the data.** The mindmap computes its own layout: leaf
blocks size themselves from their content, a band sits at the mean anchor of its
leaves, the root at the middle. Text is wrapped at build time from a
character-width estimate, because SVG `<text>` does not wrap — if a line
overflows, adjust `CHARS` in the component.

## Agreement figures (the tree)

Majority share per selectable box: the fraction of annotators agreeing with the
most common answer. A box nobody ticked counts — all three agreed to leave it,
so it scores 1.0.

- **QAA / QRAA** — per question: observed, and the same question answered at
  random. QRAA is 0.750 for any checkbox question (each box is an independent
  y/n: 2 unanimous + 6 split × 2/3, over 8 outcomes). For single-select with N
  options it is `(N² + 3N − 1) / 3N²`.
- **BAA / RAA** — the same two pooled across the instrument, kept separate for
  multi- and single-select because their baselines differ.

FrontierMath: multi 0.9348 over 92 boxes (random 0.750); single 0.8333 over 4
questions (random 0.5949). Per-question values reproduce the working sheet
exactly; the pooled multi figure does not — see open questions.

Option lists come from the Codebook v3.1 sheet, **not** from the responses: a
checkbox nobody ticked is invisible in the exports. The 92 count includes the
`other`/`unclear` options and the conditional `Verification scope` question.

## Evidence quality (Rubric v1)

Every **selected** code carries four scores. The rubric is explicit: *"Record
each score separately. Do not add them up or assign a band."* So means are taken
**per dimension only** — mean V, mean S, mean P, mean C — and never across them.
Nothing in the code should ever produce a single blended number.

| | Scale | |
|---|---|---|
| V Verifiability | 1–3 | Stated · Inspectable · Confirmed |
| S Source independence | 0–3 | Unattributed · Interested · Operator official · Independent |
| P Precision and fit | 0–3 | Off-target · Adjacent · On-target · Exact |
| C Corroboration | 0–3 | Contradicted · Single · Same party · Cross-party |
| Search completeness | 0–3 | for codes **not** selected: Minimal · Partial · Complete · Targeted |

**V floors at 1, the others at 0.** A pale V in the heatmap is therefore a worse
result than the same colour in another column; the caption says so rather than
normalising each column away.

`N/A`, `none documented` and `not reported` are exempt from rating by the
rubric — which is also why they are hidden at 0/n in the tree.

FrontierMath, 129 rated codes: mean V 1.45 · S 1.98 · P 1.76 · C 1.50. Read
together: the documentation is consistent and operator-official, and most of it
cannot be checked from outside.

## Known defects

**Code transfer error — 30 rows (NT 12, MZ 3 by code, 30 rendered).** Rationale
is attached to a code by matching the annotator's note header against the option
text. NT writes shorthand headers ("Measured capability and target domain" for
two separate codes) and sometimes groups several codes under a header naming
none of them. Token-overlap matching recovered most (81% → 88% coverage) but not
all. The page labels the rest `Code transfer error` in cardinal — **this is a
pipeline bug, not missing annotation; every rated code was written up.** Two
ways to finish: attach an unmatched block to every code selected for that
question (cheap, over-attributes), or match on quote text rather than header
(precise, more work). Needs a decision.

## What's left

**FrontierMath**

1. Fill the `Summary: [fill out later]` slot in the evidence-quality section.
2. Resolve the working-notes panels — 9 open under the tree, 1 under evidence
   quality.
3. Fix the code transfer error above.

**The other three benchmarks**

4. Transcribe context, leaves, tallies and evidence quality the same way. CSV
   parsing is straightforward; the editorial statuses are not — each needs a
   curator reading the evidence notes, not just the selections.

**Charts**

5. **Spider plots are deliberately deferred.** They need at least two benchmarks
   to be worth drawing — a four-axis radar for one benchmark is four numbers
   drawn as a shape. When built: Validity and Integrity only for content codes,
   unfilled outlines (a filled polygon's area is an implicit sum of V+S+P+C,
   which the rubric forbids), fixed axis order, and N/A drops an axis rather
   than plotting as 0.
6. Evidence quality **is** comparable across all five bands, unlike the content
   codes — it rates the citation, not the benchmark's merit. That is why the
   heatmap spans every band while the claim table does not.

**Pipeline**

7. The CSV → `annotations.ts` path is manual. Past four benchmarks it wants a
   generator under `scripts/render_website/`, following the existing
   `generate_*.py` → `content/generated/*.json` pattern. It needs the codebook
   sheet as a second input for the option lists.

## Open questions

- **Pooled BAA doesn't reconcile.** 92 boxes agreed, share totals don't: 86.0
  here against 85.3 in the working sheet. 85.3 cannot be produced by majority
  shares of 1 and 2/3 over 92 binary boxes — it needs 20.1 split boxes — so
  roughly two boxes are treated differently somewhere. Likely the blank handling
  on `Verification scope`, where two annotators left the question empty and this
  code reads empty as "nothing ticked".
- **`other` / `unclear` boxes** are scored, which is what gets the count to 92.
  They're free text, so agreement on them is not well defined, and the Part 1
  working sheet leaves them blank instead.
- **Conditional questions.** `Verification scope` enters the pooled BAA but has
  no leaf, because it only applies when an external check occurred.
- **Display vs maths.** Exclusive codes are hidden at 0/n in the tree but still
  count toward the 92 and the QAA, so the visible rows no longer sum to the
  stated denominator.
- **Search completeness is coded but not shown.** It rates how hard an annotator
  looked before concluding a code had no support, which is what makes a negative
  finding credible. Different scale from the four dimensions, and it attaches to
  unselected codes, so it needs its own place.
- **Three claims or five bands.** The summary table shows four claims; the tree
  and heatmap show all five codebook parts. Defensible together, but labels and
  band order must agree.

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
- Status marks differ in **shape as well as colour**, so they survive greyscale
  and colour-vision deficiency. Keep it that way.
- Blocks marked as working notes — the summary brief and the open-questions
  panels — are drafting aids and come out before publication.
- `.rd-mono` sets its own font weight, so plain `font-bold` on a mono span does
  nothing. Use `!font-bold`.
