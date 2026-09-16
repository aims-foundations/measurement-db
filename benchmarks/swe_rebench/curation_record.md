# SWE-rebench curation record

The reviewed snapshot contains two cached column projections from the public SWE-rebench item bank and OpenHands trajectory release. It records one subject configuration (Qwen3-Coder-480B-A35B-Instruct with OpenHands v0.54.0), 6,306 grading-specific items, 67,074 responses, and 67,018 final-patch traces. Only released non-null `resolved` outcomes are included, with the historical binary mapping retained.

## Grading evidence

The item stimulus is the released problem statement. The reference solution is the released patch; the rule requires the FAIL_TO_PASS tests to pass while preserving PASS_TO_PASS tests. The verifier carries the released harness inputs, test patch, test lists, and Docker image. No patch is rerun or regraded during curation.

## Migration decisions

The former content-based registry held 6,271 items. Including grading in identity separates 35 additional instruments with identical stimuli but different grading definitions. Every released attempt, outcome, task text, and trace is preserved. Repeated trajectories receive consecutive trials in source order within their final subject–item pair; 668 trial assignments change because the old registry merged distinct instruments.

The original downloader materialized selected columns rather than complete upstream Parquet files. `sources.inputs` pins the resulting local projection bytes. On a cache miss the builder repeats that projection and requires the same manifest hash; a changed upstream release or serialization requires an explicit snapshot review.

## Source snapshot and reproducibility

`metadata.yaml` is the source manifest: every cached input has its original URL, byte size, and SHA-256 hash. The historical downloads did not record upstream commit IDs; these are byte-pinned snapshots, not claims of a recovered upstream revision. A changed upstream download fails integrity validation instead of silently updating the release. `raw/_provenance.json` is retired after a successful build.

`test.py` reads the existing tables without downloading or modifying data. It checks the shared dataset contract, the reviewed table fingerprints in `testdata/characterization.json`, and every released grade/trace against the cached source records. Migration review also compares every observation with the pre-migration snapshot by subject configuration, task content, grade, and trace, preserving multiplicities.

The six canonical table schemas apply here without benchmark-specific columns. This release has subjects, items, benchmarks, responses, and traces; it has no attachment assets. Traces link to observations through `response_id`. Item and dependent response IDs change when grading information or the effective response scale changes.
