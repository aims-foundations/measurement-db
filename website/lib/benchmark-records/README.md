# Canonical benchmark viewer

The Python exporter writes the original table rows and lookup indexes. This
TypeScript implementation owns how those records become a model-by-task view.
It does not use the old generator's chart layouts, cell encodings, or inferred
comparison pairs.

| Module                                                   | Responsibility                                             |
| -------------------------------------------------------- | ---------------------------------------------------------- |
| `types.ts`                                               | The exported table and manifest types                      |
| `client.ts`                                              | HTTP, bounded caching, paged reads, and exact ID lookups   |
| `model.ts`                                               | Pure indexing, filtering, pagination, and item preparation |
| `scale.ts`                                               | Labels from declared benchmark/item scales                 |
| `catalog.ts`                                             | Page-card metadata from the same manifest                  |
| `components/benchmark-records/use-records.ts`            | Request lifecycle, cancellation, and loading/error states  |
| `components/benchmark-records/index.tsx`                 | View selection state and composition                       |
| `matrix-table.tsx`, `cell-details.tsx`, `pagination.tsx` | Small React views                                          |

The data modules do not import React or access the DOM. Rendering components do
not issue HTTP requests. Changes to table layout do not require rerunning Python.
Tests exercise the pure model and the HTTP client independently of React.

Every cell contains its original attempts. A repeated trial or different
opponent adds an observation to the cell; it never overwrites another attempt.
Missing grades stay null. Answers join on `response_id`, never source row order.
Subject/item features remain in their respective records. Mixed scales use the
item's declared domain, and no average is invented for unordered categories.

## Build and preview

```bash
python scripts/render_website/generate_benchmark_gallery_new.py \
  /path/to/aegis/formatted_tables /path/to/real_pocqi/formatted_tables \
  --output /tmp/gallery-records
pnpm --dir website gallery:prepare /tmp/gallery-records
pnpm --dir website gallery:test
pnpm --dir website dev
```

Open `/measurement-db/gallery-preview` for links to the prepared benchmarks.
Their ordinary `/measurement-db/<slug>` pages use the new viewer. The preparation
command copies data to ignored `public/benchmark-records/` and writes lightweight
manifests to `content/generated/gallery-records.json`. The checked-in default is
an empty catalog: prepare the data before building a deployment using this view.
Keep the generated manifest and its static data together in that deployment.

Benchmarks without a new manifest retain the existing viewer during migration.
For an isolated preview containing only prepared benchmarks, set
`NEXT_PUBLIC_GALLERY_RECORDS_ONLY=1` when building. This does not change the
production data-preparation workflow or its source selection.

The client loads one benchmark's subject and response records, then indexes
them once. Prompts, references, answers, and attachments load on demand. Table
and attempt pagination limit DOM work without dropping records. Very large
response tables will need a query service or browser Parquet reader behind
`RecordSource`; that change need not affect the components or the data model.
