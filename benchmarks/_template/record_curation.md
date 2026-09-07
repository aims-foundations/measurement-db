# Example Benchmark curation record

Replace this guidance with a concise, evidence-backed record of the completed
curation. Cite immutable provider URLs or revisions wherever possible.

## Summary

Summarize what the benchmark measures, its item bank, response scale, and the
paper's main contribution.

## Qualification and coverage

Record where exact administered item content and provider-released per-item
responses were found. For items with external files, distinguish files actually
delivered to the subject from source documents or other supporting artifacts,
and state whether their exact bytes were preserved in `assets.parquet`. State
the included subjects, splits, trials, and traces, plus every known coverage gap.

## Ingested sources and provenance

For every ingested artifact, record:

- provider-owned URL and pinned revision or checksum;
- what it contributes: items, assets, subjects, responses, traces, or grader;
- evidence that the location is the benchmark provider's own release; and
- any linkage assumptions between separately released artifacts.

Also list the archives, leaderboards, Hugging Face organizations, repositories,
supplements, and other release locations searched.

## Harness, environment, and grading

Document the scaffold and version, tools, turn or step budget, model settings,
environment or image, grader, metric, and success threshold for each response
source. Explicitly identify any response file that cannot be attributed to a
harness. For agentic benchmarks, describe whether the released environment can
be reconstructed and why one shared or per-item Docker image is appropriate.

## Processing decisions

Explain how upstream item and subject identities, repeated trials, features,
conditions, response values, reference answers, and traces map into the output
tables. When applicable, document each attachment's stable logical path, media
type, role, ordering, and whether it was directly administered. `assets.parquet`
and `traces.parquet` are optional sidecars; do not create either when the source
has no corresponding data. Do not infer unreleased configuration values.

## Deliberately excluded

List every discovered third-party rerun or re-evaluation that was not ingested,
including its additional coverage and why it was rejected. If none were found,
state that and summarize the search performed.

## Source reconciliation

For each `metadata.yaml` source claim, report the expected value or table, the
observed curated value, the comparison rule, and the result. Preserve explained
source disagreements and partial or non-comparable claims rather than changing
an expectation to make a test pass.

## Verification and publication

Record the successful build and characterization-test commands, schema audit,
model-registry coverage, asset byte/digest and redistribution review when
applicable, the Hugging Face data pull-request URL and review status, and any
remaining follow-up.
