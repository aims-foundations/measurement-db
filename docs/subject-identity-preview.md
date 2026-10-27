# Subject identity data preview

The preview combines the current canonical gallery with the subject fixes in
`635a74e2955d970b48837cee1a109a9ef96845c3`. The complete machine-readable build
and publication evidence is in [subject-identity-preview.json](subject-identity-preview.json).

The updated tables for 65 public benchmarks are published at HF commit
[`9a8a4e098dd39b02f02104f08362f99c6b8cb250`](https://huggingface.co/datasets/aims-foundations/measurement-db/commit/9a8a4e098dd39b02f02104f08362f99c6b8cb250),
on `preview/subject-identities-20261027`. All 522 uploaded table and descriptor
files were verified against their remote blob identities. Production HF `main`
remained at `54074453da7b9911be0f48e4c560be547bc84d21`.

All 65 builders have passed full logical-table comparisons with the reviewed
identity migration, covering 19,352,542 observations. Three earlier author builds
were reused and revalidated; the other 62 were rebuilt during this release.
Items, grades, traces, and configuration distinctions are preserved. The
withheld MedArabiq, OSWorld, and SWE-Together datasets are excluded.

The website uses newly validated record bundles for those 65 benchmarks and
reuses the existing bundles for the other 205. Every source download is pinned
to the new HF commit. The model timeline, coverage graph, and benchmark summaries
were refreshed from the complete subjects tables. Existing Rasch estimates for
32 affected benchmarks were translated to the new subject IDs; their original
fit provenance is retained.

The index now contains 901 registry-normalized model names, 3,849 benchmark-model
links, and 53 benchmarks without a registry mapping. The increase from 781 names
reflects newly recognized models after removing benchmark decorations. Harness
and configuration differences remain in separate subject records.

Reproduction notes:

- Each normal builder verified its declared source hashes. Original captured
  inputs were reused for changed paper HTML and slow archive transfers; the
  evidence lists those inputs and their checksums.
- `rouge-chinese==1.0.3` also installs files under the `rouge` module. In LexEval's
  isolated Python 3.14.6 environment, `rouge==1.0.1` was reinstalled after the
  shared requirements and its module files checked against its distribution's
  `RECORD`. The initial conflicting-scorer output was rejected. The complete
  rerun matches all 1,076,280 published observations after identity translation.
- Gallery preparation validated every observation and the source Parquet
  positions used for loading item and trace details. Existing visualizations
  are reused; this release requires no presentation changes.
