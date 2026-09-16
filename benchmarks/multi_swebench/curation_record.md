# Multi-SWE-bench curation record

The reviewed snapshot contains 647 cached artifacts from the Multi-SWE-bench experiments repository, its Hugging Face item bank, and SWE-bench Verified. It records 82 subject configurations, 2,126 grading-specific items, 57,975 responses, and 57,184 traces. Resolved/unresolved membership is the binary outcome; prediction patches supply traces when available. Harnesses remain subject attributes and language remains an item attribute. The earliest released run date for each subject configuration is retained.

## Migration decisions

The former content-based registry held 2,078 items. Including grading in identity separates 48 additional instruments with the same stimulus but different grading definitions. No released attempt, grade, task text, or trace is added or removed. Repeated trials are numbered in deterministic source order within each final subject–item pair; 1,447 assignments differ from the old merged-item numbering.

The original 12,000-character stimulus cap, 4,000-character reference/test patch caps, and 50-name PASS_TO_PASS sample are preserved. Complete source artifacts remain in `raw/`; the stored abbreviated verifier should not be mistaken for a complete executable test harness. Reference patches belong in `grading_criterion.reference_answer`; the success rule belongs in `grading_criterion.rule`.

Legacy filenames containing plus signs or parentheses are escaped to portable manifest paths. `archive_layout.original_paths` retains their original source names. Their bytes are unchanged. One captured prediction file (`java__20250426_MopenHands_Gemini-2.5-Pro.jsonl`) is a Git LFS pointer, not prediction data. As in the original build, its outcomes are retained and its unavailable patch traces remain absent. The source test verifies this single known gap explicitly; migration does not fetch a new source payload.

## Source snapshot and reproducibility

`metadata.yaml` is the source manifest: every cached input has its original URL, byte size, and SHA-256 hash. The historical downloads did not record upstream commit IDs; these are byte-pinned snapshots, not claims of a recovered upstream revision. A changed upstream download fails integrity validation instead of silently updating the release. `raw/_provenance.json` is retired after a successful build.

`test.py` reads the existing tables without downloading or modifying data. It checks the shared dataset contract, the reviewed table fingerprints in `testdata/characterization.json`, and every released grade/trace against the cached source records. Migration review also compares every observation with the pre-migration snapshot by subject configuration, task content, grade, and trace, preserving multiplicities.

The six canonical table schemas apply here without benchmark-specific columns. This release has subjects, items, benchmarks, responses, and traces; it has no attachment assets. Traces link to observations through `response_id`. Item and dependent response IDs change when grading information or the effective response scale changes.
