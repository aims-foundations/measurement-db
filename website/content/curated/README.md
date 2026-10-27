# Benchmark introductions

`benchmark-copy.json` contains reader-facing copy, keyed by stable benchmark
slug. Each entry has a one-sentence `description` and, when needed, a shorter
`name`. The catalog, detail-page introduction, page metadata, coverage views,
timeline, and search use the same text through `measurement-db.ts`. The search
index exporter reads the same file.

- Use a recognizable benchmark name. Retain a short qualifier when it
  distinguishes a release, task subset, or another catalog entry.
- Explain what the participant or model does and what is assessed. Aim for
  one sentence of at most 160 characters. Avoid implementation and import notes.
- Describe the actual imported task, not every capability claimed by its paper.
  Keep essential distinctions such as ungraded examples and selected showcases.
- Check the benchmark's `metadata.yaml` and released-data description before
  editing. Do not infer capabilities from its name alone.

The full dataset name and description from the pinned data remain available
under **About this data** on the benchmark page. Original names also remain
searchable. IDs, measurement records, and published metadata are unchanged.
New benchmarks fall back to their exported metadata until copy is reviewed;
this file does not determine which benchmarks are visible.

Copy edits need only a website build. Do not regenerate Parquet tables or
gallery assets for an editorial change.
