# Canonical benchmark viewer

Python exports the original table rows and lookup indexes. TypeScript decides
how to display them, using the website's existing typography, heatmap colors,
and detail-panel styles. Every published benchmark uses this viewer. No PNG, legacy chart bundle, or
color-to-grade decoding is used by its response matrix.

| Module                                                                    | Responsibility                                                    |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `types.ts`, `client.ts`                                                   | Canonical fields, HTTP, bounded caching, exact ID lookups         |
| `parquet.ts`, `parquet.worker.ts`                                         | Cancelable worker and exact trace reads using Parquet byte ranges |
| `model.ts`                                                                | Validate references and index original observations               |
| `context.ts`, `view.ts`                                                   | Condition filters, selected axes, summaries, stable ordering      |
| `scale.ts`, `colors.ts`                                                   | Declared scale labels and presentation colors                     |
| `presentation.ts`, `comparisons.ts`                                       | Explicit source text mappings and unambiguous comparison links    |
| `catalog.ts`                                                              | Page-card metadata from the same manifest                         |
| `components/benchmark-records/use-records.ts`                             | Loading states, cancellation, and attachment cleanup              |
| `index.tsx`, `record-gallery.tsx`                                         | Loading boundary and view composition                             |
| `window-browser.tsx`, `loaded-gallery.tsx`                                | Navigate bounded item pages and search the full model registry    |
| `heatmap.tsx`, `heatmap-canvas.tsx`, `paint-heatmap.ts`                   | Layout, pointer/keyboard interaction, visible canvas tiles        |
| `matrix-table.tsx`, `view-controls.tsx`, `legend.tsx`                     | Accessible table alternative, controls, declared grade legend     |
| `cell-details.tsx`, `answer-panel.tsx`, `rich-text.tsx`, `attachment.tsx` | Item, grading rule, reference, Markdown/math, answers, media      |

Data modules do not import React or access the DOM. Rendering components do not
issue HTTP requests. Layout and styling changes do not require rerunning Python.

## Presentation rules

- Every cell retains all its attempts. The heatmap stacks their colors; it does
  not turn multiple responses into a synthetic cell grade. Null grades and
  absent observations have different colors and descriptions.
- Hover, keyboard navigation, and click selection resolve original IDs directly.
  Actual-size mode draws only visible columns. Small canvas tiles avoid creating
  one full-resolution image of the entire matrix. Hover does not repaint it.
- Filters use observation conditions, interactors, and trial. Subject metadata
  remains attached to the model. All models remain searchable. Item pages include unobserved items, so the
  complete item registry remains reachable even without released responses.
- Mean-grade ordering and margins require a common, explicitly ordered scale.
  Means use graded observations in the selected view. Unordered and mixed scales
  have no cross-item mean. Mixed and unbounded scales use neutral overview colors;
  task details always show the exact grade and that task's declared scale.
- Answers join on `response_id`. A pair of answer panels requires unique
  reciprocal opponents with matching item, trial, and comparison context.
  Trial counters alone never pair repeated comparisons. Ambiguous or unmatched
  observations remain individually accessible.
- Every benchmark page uses `components/item-content.tsx`: a customized view
  followed by a closed **Original item** section containing the exact source text.
  An unregistered or failed customization shows the original text directly.
- Temporary transformations live only in `lib/presentation-transforms/benchmarks/`.
  Each published benchmark has a separate module. Most select a shared formatter
  that preserves all fields; specialized parsing (such as MMLU) stays in its own file.
  `registry.ts` is the explicit registration list. Remove an entry and its module
  to revert that benchmark to raw text; no data regeneration is necessary.
- `shared.ts` supplies ordinary text/Markdown, labeled structured fields, variable
  answer choices, and ordered multimedia blocks. These are display documents,
  not a new persisted data schema. Unknown fields remain visible. There are no
  benchmark-name branches in the shared React components or Python exporter.
- `raw-item-content.tsx` escapes and displays the original string without JSON
  reserialization, trimming, truncation, Markdown, or math processing. The large
  raw text DOM is mounted only when expanded. A missing text component is explicit.
- `source-item-content.tsx` remains only in the retired legacy viewer. Published
  benchmark pages read exact original items through the canonical record source.
- The same source record drives custom and raw views. Images follow the released
  multimedia order, and remaining attachments are paginated and opened on demand.
  Source-specific answer and comparison mappings also live in their corresponding
  benchmark modules.
- `rich-text.tsx` renders explicit LaTeX with KaTeX and accessible MathML. Inline
  and fenced code bypass math normalization; common bare commands such as
  `\boxed{A}` retain balanced arguments. `code-block.tsx` preserves indentation
  and labels declared languages. Benchmark adapters handle source-specific code
  wrappers (CRUXEval), whole Python programs (KernelBench), and source files (Aider).
- `components/structured-content.tsx` remains shared by metadata and answers.
  Formatting does not change item, model, observation, or response identities.
- Existing Rasch summaries and subject abilities can be supplied separately.
  They describe the published full-benchmark analysis, not a refit of the current
  filters; incompatible counts/subject IDs prevent displaying them.

## Build and preview

Use Node 24, as in the website CI.

```bash
python scripts/render_website/generate_benchmark_gallery_new.py \
  /path/to/aegis/formatted_tables /path/to/real_pocqi/formatted_tables \
  --output /tmp/gallery-records
pnpm --dir website gallery:prepare /tmp/gallery-records
pnpm --dir website gallery:test
pnpm --dir website dev
```

Open `/measurement-db` for the normal catalog, or `/measurement-db/gallery-preview`
for links to just the prepared benchmarks.
Their ordinary `/measurement-db/<slug>` pages use the new viewer. Preparation
copies data to ignored `public/benchmark-records/` and writes lightweight
manifests to `content/generated/gallery-records.json`. The checked-in default is
empty. Keep the prepared manifest and its static files together in deployments;
do not commit a local manifest without a corresponding deployment preparation.

## Normal website preparation

CI prepares canonical records for every published benchmark; there is no migration
allowlist or legacy matrix fallback. The existing catalog, search, and cards retain
their design. `NEXT_PUBLIC_GALLERY_RECORDS_ONLY=1` is available for isolated local
experiments with a subset of exported records.

`record_windows.py` groups unchanged observations by item using a disk-backed
sort. Each window holds at most 256 items, 4,096 observations, and 2 MiB of expanded
JSON, using the same limits for every benchmark. Items with more attempts span
explicitly labeled partial pages. Filters and means describe the current page.
Reciprocal comparison links are disabled for partial items because uniqueness
cannot be established from an incomplete set of attempts. Full published Rasch
fits remain separately labeled. Items without observations still have pages.

`record_parquet.py` indexes exact original items, traces, shared attempts, and attachments. Their
content stays in pinned Parquet and loads on selection through the authenticated
source route. `record_packs.py` packs independently compressed JSON pages into
static byte ranges, avoiding hundreds of thousands of deployment files. The
browser verifies each range checksum before decoding it. No HF credentials reach
the client. Source reads stop before downloading or decompressing more than
16 MiB for one selected record; oversized source pages offer the original pinned
Parquet table for download. Values are never silently truncated. UI changes leave
these record bundles unchanged.

Schema v4 observations may include `attempt_id`. The answer panel uses that link
to read the shared execution from `attempts`; otherwise it reads the legacy
response-specific `traces` record. Both use the same lazy loading, cache, and
bounded-preview transport. `items.testlet_id` is retained as source data; it does
not change paging or imply that a page contains a complete testlet. Page means
still describe the displayed item outcomes, not a benchmark's official aggregate.

`scripts/ci/check_gallery_records.py` verifies every source lookup position and
compares full-record checksums, counts, and item coverage for all observation
windows. The native grades, contexts, trials, identities, and source fields are
preserved. `canonical_gallery.py` packages these verified records. The legacy
chart and analysis preparation has an independent cache identity, allowing the
existing fitted results and catalog artifacts to be reused when only canonical
record preparation changes.

The standalone exporter accepts `--parquet-base <pinned-source-base>` for lazy
source records and bounded item windows. Without it, small local demonstrations
can still export self-contained JSON records and attachment files.

Offline tests: `pytest tests/test_gallery_new.py`, `pnpm --dir website gallery:test`,
and the canonical Parquet fixture read in `.github/workflows/website.yml`.

## Presentation validation

`pnpm --dir website gallery:test` checks fallback behavior, catalog registration,
MMLU parsing, variable choice counts, media order, and retained unknown fields.
The hybrid worker fixtures check exact raw item text for static and dynamic data,
including Unicode, embedded images, and exporter-appended images.

`node website/scripts/check-item-presentations.mjs /path/to/samples` exercises
all registered transformations against an inventory plus per-benchmark item samples.
The initial check used 6,106 sampled items from all 270 tables available at HF
revision `54074453da7b9911be0f48e4c560be547bc84d21`. This is structural coverage,
not a claim to have manually reviewed every item or reconstructed every prompt.
Future registrations without a module retain the raw-text fallback.

## Oversized source pages

The original Parquet reader budgets **16 MiB of downloaded plus decompressed
page bytes per lookup**, not 16 MiB of answer text. A small answer can share a
large dictionary or data page. Keep that guard: removing it can allocate hundreds
of megabytes for one click, especially problematic on phones.

`scripts/render_website/record_previews.py` prepares an optional exact-record
cache only for rows whose original page reads exceed that budget. Records are
independently gzipped in 64 MiB packs on Hugging Face, outside the Vercel build.
The browser reads two small byte ranges (one ID bucket and one record), verifies
their SHA256 and sizes, and decompresses them in a worker. Each read is capped at
4 MiB compressed and expanded. Records larger than 2 MiB expose their long text
fields in complete, ordered parts of at most 32,768 Unicode characters; no text
is dropped. Conversation lists and long text also paginate DOM rendering.

The lightweight `content/generated/record-previews.json` pins the immutable HF
artifact revision and the exact canonical source URL. A stale source pin is
ignored. IDs absent from this optional cache use the normal Parquet lookup.
Item and answer loading are independent, and failed lookups have a retry button.
The cache does not cover binary attachments or invent missing released answers.

Prepare and publish only when source data changes, using the same local tables
as the canonical gallery exporter (requires pyarrow, fastparquet, huggingface_hub):

```bash
python -m scripts.render_website.prepare_record_previews \
  --tables /path/to/pinned-tables --output /path/to/new-record-cache \
  --publish-branch preview/record-loading-YYYYMMDD
```

Omit `--publish-branch` for local preparation and verification. Completed tables
can be resumed only with the same source hash. Publication checks the local source
against the canonical HF file hash, uploads to the explicit preview branch,
verifies uploaded file hashes, then writes the frontend manifest. Commit that
small manifest with the frontend change. Ordinary presentation deployments reuse
it and do not rerun Python or upload the packs. If table bytes change, prepare a
new output directory; do not reuse a cache from a different source snapshot.
