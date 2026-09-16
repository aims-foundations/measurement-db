# AI Measurement Data Bank

The AI community invests considerable effort in designing and running evaluations. Yet item-level measurement data—what system answered which item, under what conditions, and with what outcome—remain scattered across papers, repositories, leaderboards, and incompatible file formats.

The AI Measurement Data Bank is community-owned infrastructure for turning fragmented evaluation results into shared, reusable evidence. It supports systematic efforts to compare benchmarks, study validity and reliability, track capabilities over time, and design better evaluations.

- [Explore the AI Measurement Data Bank](https://aimslab.stanford.edu/measurement-db)
- [Access the released data](https://huggingface.co/datasets/aims-foundations/measurement-db)

The [`benchmarks/`](benchmarks) directory contains the six public curation
examples: MathArena, MMDocRAG, Multi-SWE-bench, REAL, ResearchCodeBench, and
SWE-rebench. Each includes its builder, source manifest, read-only tests, and
curation record. Generated data remain on Hugging Face rather than in Git.

## Contributing

Curating a benchmark is not simply a data-cleaning task. It requires reconstructing what was measured, which systems were evaluated, how the evaluation was conducted, and what each recorded outcome means. Contributors will gain firsthand experience with the structure and limitations of modern AI evaluation data. For participants in the Predictive AI Evaluation Competition, this work can also inform the development of prediction methods and expand the public evidence available for training them. For benchmark authors, curation makes their results easier to discover, compare, and reuse. For measurement researchers and practitioners, it creates a common foundation for studying the validity, reliability, and generalizability of AI evaluations. Under appropriate conditions, contributors will also be eligible for the AI Measurement Data award under the NeurIPS 2026 Predictive AI Evaluation Competition.

We welcome contributions from benchmark authors, evaluation researchers, practitioners, and students. The complete and maintained instructions are available in the [Instructions for Item-level Measurement Data Curation](link). Questions may be submitted by opening an issue. Completed curation work may be submitted for review through a pull request to this repository.


The six public datasets follow schema version 3, defined in
[`parquet_schemas.yaml`](parquet_schemas.yaml). This release replaces legacy
grading fields, fixes table types, and includes grading protocols in item
identity. Item and dependent response IDs therefore change; download all tables
from the same Hugging Face revision when joining them. On Hugging Face the
response table is named `response.parquet`; local builders write `responses.parquet`.
Each item requires a `grading_criterion` and a `verifier`. The criterion is a
JSON object containing a `reference_answer`, a `rule`, or both (nonempty strings);
the verifier describes the grader or implementation that applies it. Both enter
item identity. These fields are stored in `items.parquet`; graded observations are stored
in `responses.parquet`. Available model outputs are stored in `traces.parquet`
and join to their exact response through `response_id`. Benchmark source links
use `benchmarks.parquet.source_url`.
An item identifies both its stimulus and its grading protocol: changing the
criterion, judge, grading implementation, or effective response scale produces
a distinct item ID. Uniform scales remain stored once in benchmark metadata.

`response_scale` is structured metadata: for example, `{kind: discrete, values:
[0, 1]}` for binary outcomes or `{kind: interval, min: -10, max: 10}` for a
bounded score. A `{kind: mixed}` benchmark specifies a concrete `response_scale`
inside each item's `grading_criterion`. Grades must be finite and within their
declared scale. Optional `meanings` describe discrete values (e.g., `{"0":
"failure", "1": "success"}`); `direction` is `higher_is_better`,
`lower_is_better`, `unordered`, or null when unknown. These annotations never
reverse or rescale grades. `categorical` is derived for binary, error-presence,
Likert, ordinal, and continuous types; conflicting declarations fail. Fractions,
rates, and mixed scales require an explicit classification.
An explicit `None` records an ungraded attempt, while no row
means no available observation. Canonical Parquet types are fixed, including
nullable `float64` grades and `int64` trial indices. See
[`parquet_schemas.yaml`](parquet_schemas.yaml) for the complete contract.

Each of the six table types has one fixed schema across all benchmarks: the
same columns, order, types, and nullability. Unknown values use nullable fields;
benchmark-specific columns are rejected. Keep source evidence in the pinned
raw inputs and document grade transformations in the builder. Derived analysis
outputs belong outside the canonical tables. Traces and assets are optional
tables when a benchmark has no such data; when present, their schemas are fixed.

`validate_dataset(tables, expected_benchmark_id=...)` checks a complete benchmark
across its tables: keys and references, grade domains, trace and asset links,
and summary statistics recomputed from the rows. Completed builders run this
check before replacing their outputs. Migration and publishing tools import
the same validator; validating individual tables alone is insufficient.

`metadata.yaml` records benchmark facts and source provenance. New definitions
use `build.contract_version: 2`: `sources.upstream` lists source URLs and known
revisions (or null), and `sources.archive` identifies an HF dataset repository,
an immutable commit, and `<slug>/raw`. Optional `sources.notes` records provenance
limitations. Unknown fields, embedded file inventories, mutable archive revisions,
and untyped `archive_layout`/`expectations` sections are rejected. Keep parsing
rules in `build.py` and regression expectations in the characterization test.

Archive the reviewed raw inputs first, then record the resulting commit in
metadata. The shared builder restores missing files and verifies their sizes
and content hashes against that commit's file tree, including existing cached
files. `self.source_files` lists the verified raw-relative input paths for the
row-building hook; use it when discovering files so unrelated local caches do
not affect the build. The Hub must be reachable to obtain the pinned file tree;
raw inputs are downloaded only when missing. No additional provenance file is
maintained. Contract 1 remains available for older, unmigrated private builders.

Check new definitions without downloading data:

```bash
python -m scripts.build_measurement_tables.validate_benchmark_metadata --require-current benchmarks/*/metadata.yaml
```

GitHub checks run this validator and the snapshot tests for every pull request.

Optional `benchmark.version` records the provider's benchmark release in
`benchmarks.parquet.version` (for example, `version: "2.0"`). Omit it or use
`null` when unknown; schema versions, `build.contract_version`, and source
revisions describe separate concepts.

Set `benchmark.release_date` explicitly to `null` when unknown. Known dates use
quoted `YYYY-MM` or `YYYY-MM-DD` strings with valid calendar values; retain the
available precision rather than guessing a month or day.

## Restoring archived sources

The [Hugging Face repository](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main)
stores the six public source snapshots under `<benchmark>/raw/`. From a checkout
of this repository, install the dependencies and run a builder:

```bash
pip install -r requirements.txt
python benchmarks/real_webagents/build.py
```

The builder uses the archive revision in its local `metadata.yaml`; later changes
to the upstream sources or HF `main` do not change its inputs. Replace
`real_webagents` with another benchmark to rebuild it. Do not combine a builder
with metadata from a different build-contract version.

## License

To the extent that AIMS holds copyright or database rights, the original curation contributions in the AI Measurement Data Bank—including their selection, organization, standardized schema, metadata, and normalization work—are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). 

Individual benchmarks and other third-party materials are not relicensed under CC BY-SA 4.0. They retain their original licenses and terms, as identified in each benchmark's metadata. Those upstream terms govern the corresponding material.

## Citation

If you use the data curated in AI Measurement Data Bank, please cite:

```bibtex
@misc{measurementdb2026,
  title        = {The AI Measurement Data Bank},
  author       = {Truong, Nhi and Truong, Sang T. and Koyejo, Sanmi},
  year         = {2026},
  howpublished = {\url{https://aimslab.stanford.edu/measurement-db}},
  note         = {AIMS Lab, Stanford University}
}
```
