# Continuous test matrix experiment

`/testlet-preview` is an isolated ProgramBench preview. The regular benchmark
pages and canonical Parquet tables are unchanged.

- `scripts/render_website/prepare_aligned_test_preview.py` prepares a display
  index from the existing tables. A column is the full native test identifier
  within its testlet, including the branch prefix. The export rejects repeated
  attempts, ambiguous item mappings, and nonbinary or duplicate cell outcomes.
- `lib/aligned-matrix/` owns the compact score model, exact interval counts,
  range geometry, and lazy test-to-item lookup. Scores use one byte per cell:
  0 means no observation, 1 means fail, 2 means pass. The browser verifies the
  score checksum and full-source totals before rendering.
  `order.ts` sorts columns within each testlet by increasing observed pass rate
  among the selected models and grader, matching the gallery's difficulty
  convention. Missing observations are excluded, ties keep native order, and
  tests with no selected observations appear last. A display-to-source index
  keeps cell details attached to the exact original test after sorting.
- `explorer.tsx` composes controls and model rows. `canvas.tsx` handles pointer
  and keyboard navigation; `paint.ts` draws the current viewport. Overview bins
  cover every visible column exactly once. Bands show proportions, while a
  sufficiently zoomed viewport shows individual binary cells.
- `details.tsx` displays recorded setup variants. `use-selection.ts` resolves a
  cell to its canonical item and checks its observation against the displayed
  score. Existing `RecordSource` and `CellDetails` load and render the original
  item and shared model attempt; no full prompts or traces enter the overview.

The current snapshot has 247,930 columns, 200 testlets, 28 model configurations,
1,483,092 canonical item variants, and 6,119,913 binary observations. Its score
file is 494,881 compressed bytes / 6,942,040 decoded bytes. The experimental
loader caps decoded score data at 12 MiB; larger matrices need score tiles.
Native test names and item mappings use bounded pages in static range packs,
with at most four decoded pages cached.

Run the exporter from the repository root (requires the already-built tables):

```sh
python -m scripts.render_website.prepare_aligned_test_preview \
  --tables benchmarks/programbench/formatted_tables \
  --output website/public/benchmark-data/programbench/aligned-preview \
  --source-revision a7a2d0b153057662838c26b88f3597413ddbd9c5
```

Prepared assets can be reused for UI-only changes. The preview workflow checks
their hashes before deployment; it does not rebuild benchmarks or publish HF
tables. Python tests verify identity preservation and reject lossy alignment.
`pnpm --dir website gallery:test` includes interval, filter, and bin conservation
tests for the frontend model.

Rows retain distinct subject configurations, and the grader filter does not
claim that different prompts or environments are equivalent. Pass percentages
refer to observed tests in the visible interval; they are not the benchmark's
task-averaged leaderboard score. Visual alignment does not change item identity
or fit a new Rasch model.
