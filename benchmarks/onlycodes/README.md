# Onlycodes

3,474 released attempts across the SWE-bench and artifact suites; 12 measured comparisons.

This folder is the curation package for one source study. Its task suites retain
separate `benchmark_id` values; evaluation collections retain `experiment_id`.
The folder name is not substituted for either identifier. See the
[shared pilot design](../../docs/intervention-pilot/README.md) and
[pilot results](../../docs/intervention-pilot/results-v2.md).

- `study.json`: authoritative study design, intervention definitions, source
  revision, and availability limits. Its original analysis-scope IDs are mapped
  to the v2 experiment IDs in the shared `catalog-mapping.json`.
- `metadata.yaml`: package description, publication status, and shared manifest links.
- `build.py`: rebuild entry point using the common pilot implementation.
- `formatted_tables/`: generated study tables using `intervention-pilot-v2`.
  `manifest.json` records table hashes, membership, and the shared catalog location.

Rebuild from the repository root:

```bash
python benchmarks/onlycodes/build.py --offline
```

This rebuilds the shared three-study pilot from cached, hash-verified inputs and
exports this study's tables into this folder. Omit `--offline` to fetch missing
pinned inputs. No model inference is performed. The common source cache and
catalogs remain under `artifacts/intervention_pilot/`; the large source archive
is not copied into each package. Use `--package-root /path/to/packages` to direct
the generated study tables elsewhere. The main pilot command exports all three.

The shared `benchmarks.parquet`, `experiments.parquet`, and
`experiment_benchmarks.parquet` are written once in
`artifacts/intervention_pilot/release_v2/catalog/`.
Study tables preserve the same IDs as the derived benchmark-partitioned export.
Do not ingest both layouts as separate observations.

These are prototype tables with explicit metadata gaps. `release: withheld`
keeps them out of production publishing and canonical reproduction CI while the
new schema is being integrated; it does not describe access to the public source
study. Generated tables remain ignored by Git, following repository convention.
