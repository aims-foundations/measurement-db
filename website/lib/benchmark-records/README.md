# Canonical benchmark viewer

Python exports the original table rows and lookup indexes. TypeScript decides
how to display them, using the website's existing typography, heatmap colors,
and detail-panel styles. No PNG, legacy chart bundle, or color-to-grade decoding
is required by this viewer.

| Module                                                                    | Responsibility                                                 |
| ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `types.ts`, `client.ts`                                                   | Canonical fields, HTTP, bounded caching, exact ID lookups      |
| `model.ts`                                                                | Validate references and index original observations            |
| `context.ts`, `view.ts`                                                   | Condition filters, selected axes, summaries, stable ordering   |
| `scale.ts`, `colors.ts`                                                   | Declared scale labels and presentation colors                  |
| `presentation.ts`, `comparisons.ts`                                       | Explicit source text mappings and unambiguous comparison links |
| `catalog.ts`                                                              | Page-card metadata from the same manifest                      |
| `components/benchmark-records/use-records.ts`                             | Loading states, cancellation, and attachment cleanup           |
| `index.tsx`, `record-gallery.tsx`                                         | Loading boundary and view composition                          |
| `heatmap.tsx`, `heatmap-canvas.tsx`, `paint-heatmap.ts`                   | Layout, pointer/keyboard interaction, visible canvas tiles     |
| `matrix-table.tsx`, `view-controls.tsx`, `legend.tsx`                     | Accessible table alternative, controls, declared grade legend  |
| `cell-details.tsx`, `answer-panel.tsx`, `rich-text.tsx`, `attachment.tsx` | Task, grading rule, reference, Markdown/math, answers, media   |

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

Open `/measurement-db/gallery-preview` for links to the prepared benchmarks.
Their ordinary `/measurement-db/<slug>` pages use the new viewer. Preparation
copies data to ignored `public/benchmark-records/` and writes lightweight
manifests to `content/generated/gallery-records.json`. The checked-in default is
empty. Keep the prepared manifest and its static files together in deployments;
do not commit a local manifest without a corresponding deployment preparation.

Benchmarks without a new manifest retain the existing viewer. For an isolated
preview containing only prepared benchmarks, set
`NEXT_PUBLIC_GALLERY_RECORDS_ONLY=1` when building. The production preparation
workflow and source selection remain separate from this experimental migration.

The client loads one benchmark's subjects and response records, then indexes
them. Prompts, references, answers, and attachments load on selection. Very large
response tables will need a query service or browser Parquet reader behind
`RecordSource`; the rendering components need not change for that transition.
