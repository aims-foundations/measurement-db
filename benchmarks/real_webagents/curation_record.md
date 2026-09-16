# REAL curation record

The reviewed snapshot contains the cached REAL verified-model API response and 112 upstream task definitions. It records 33 subject configurations, 233 items, 4,449 responses, and 1,242 traces. The grade is one exactly when `evalsFailed` is empty, with `accuracy >= 100` as the historical fallback when that field is absent. Entries with no accuracy remain excluded. `retrievedAnswer` supplies a trace except for empty values and the placeholders `Done` and `No response`. Website remains an item attribute.

## Grading evidence

For the 112 tasks with definitions, `grading_criterion.rule` contains the released `evals` list. The verifier identifies the existing deterministic checker or its LLM rubric procedure. These lists are rules, not reference solutions.

The other 121 tasks have `v2.` identifiers and no task definition in the captured source. Their criterion states the known all-checks-pass requirement, and their verifier explicitly describes the deterministic mapping from provider verdict fields. The original task checker and its rubrics remain unavailable; this record does not claim to reconstruct them or infer them from older similarly named tasks.

## Migration decisions

The subjects, item count, responses, trial assignments, and traces are unchanged. Previously missing grading fields for v2 tasks now describe the available provider-result mapping and its limitations.

## Source snapshot and reproducibility

`metadata.yaml` is the source manifest: every cached input has its original URL, byte size, and SHA-256 hash. The historical downloads did not record upstream commit IDs; these are byte-pinned snapshots, not claims of a recovered upstream revision. A changed upstream download fails integrity validation instead of silently updating the release. `raw/_provenance.json` is retired after a successful build.

`test.py` reads the existing tables without downloading or modifying data. It checks the shared dataset contract, the reviewed table fingerprints in `characterization.yaml`, and every released grade/trace against the cached source records. Migration review also compares every observation with the pre-migration snapshot by subject configuration, task content, grade, and trace, preserving multiplicities.

The six canonical table schemas apply here without benchmark-specific columns. This release has subjects, items, benchmarks, responses, and traces; it has no attachment assets. Traces link to observations through `response_id`. Item and dependent response IDs change when grading information or the effective response scale changes.

Validation expectations are now stored beside metadata in `characterization.yaml`,
using the shared schema and table hash algorithm. The previous source audits and
migration assertions passed before conversion; release-specific assertions remain
in `test.py`. Source claims distinguish reported quantities from those independently
derived from the archived release. This format change does not alter curated data.
