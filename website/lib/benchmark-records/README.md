# Canonical benchmark viewer

Python exports the original table rows and lookup indexes. TypeScript decides
how to display them, using the website's existing typography, heatmap colors,
and detail-panel styles. No PNG, legacy chart bundle, or color-to-grade decoding
is required by this viewer.

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
| `model-browser.tsx`, `loaded-gallery.tsx`                                 | Search the full registry and load a bounded page of models        |
| `heatmap.tsx`, `heatmap-canvas.tsx`, `paint-heatmap.ts`                   | Layout, pointer/keyboard interaction, visible canvas tiles        |
| `matrix-table.tsx`, `view-controls.tsx`, `legend.tsx`                     | Accessible table alternative, controls, declared grade legend     |
| `cell-details.tsx`, `answer-panel.tsx`, `rich-text.tsx`, `attachment.tsx` | Task, grading rule, reference, Markdown/math, answers, media      |

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
  remains attached to the model. Registry axes remain visible in the unfiltered
  view, including unobserved models and tasks.
- Mean-grade ordering and margins require a common, explicitly ordered scale.
  Means use graded observations in the selected view. Unordered and mixed scales
  have no cross-item mean. Mixed and unbounded scales use neutral overview colors;
  task details always show the exact grade and that task's declared scale.
- Answers join on `response_id`. A pair of answer panels requires unique
  reciprocal opponents with matching item, trial, and comparison context.
  Trial counters alone never pair repeated comparisons. Ambiguous or unmatched
  observations remain individually accessible.
- `presentation.ts` contains explicit mappings for nested source text and release
  comparison identifiers. For example, ReAL-PoCQI's source row identifies a
  rating, while its slot identifies the perspective. Unknown payloads remain
  readable as text/JSON. The full original prompt and trace are always available.
- Both viewers use `lib/task-content.ts` and `components/task-content.tsx` to
  present structured tasks as labeled fields, lists, and conversation messages.
  MMLU's explicit format shows the evaluated question and answer choices first,
  with instructions, demonstrations, and request details in disclosures.
  Ambiguous question boundaries fall back to displaying every source field.
  Original task text remains accessible; this formatting never changes exports.
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

`content/curated/canonical-record-benchmarks.json` selects the migrated benchmarks:
AEGIS, ReAL-PoCQI, and MMLU. CI's existing preparation job exports their records
from the same hash-verified source tables used by the rest of the catalog.
`scripts/ci/check_gallery_records.py` compares every exported field and every
lazy trace position to its source, including null grades and subject-page coverage.
Validation reports ship beside the records as `records-validation.json`.

The records, indexes, and lightweight manifests are included in the existing
hashed artifacts. The normal Vercel build restores those artifacts; it does not
regenerate data. The full catalog, search, cards, and other benchmark pages keep
their existing data and presentation. A new manifest switches only that detail
page to the canonical viewer. The temporary directory is not the production
catalog. For isolated local experiments, `NEXT_PUBLIC_GALLERY_RECORDS_ONLY=1`
restricts the catalog to the prepared entries.

Above 200,000 responses, the viewer searches the full subject registry and loads
up to ten models at a time. Every task, repetition, and condition for those models
is retained. Page filters and means are explicitly scoped to those models;
published Rasch summaries still describe the full benchmark. This bounds MMLU's
initial load without sampling its subjects or items. A single model with millions
of observations will need further partitioning before that benchmark is migrated.

Prompts, references, answers, and attachments load on selection. CI leaves traces
in pinned Parquet and indexes only their IDs and byte locations. The existing
server route authenticates HF access; the worker downloads just the required
pages and returns the complete original row. The standalone exporter also accepts
`--parquet-base <public-source-base>` for range-accessible source files. Omit it
to export self-contained JSON traces. Never expose HF credentials to the client.

Offline tests: `pytest tests/test_gallery_new.py`, `pnpm --dir website gallery:test`,
and the canonical Parquet fixture read in `.github/workflows/website.yml`.
