# Three-paper pilot with separate benchmark and experiment identities

The completed pilot contains five shared benchmark records, three experiment
records, 368 task records, and 15,174 observed attempts. Every response carries
both `benchmark_id` and `experiment_id`. The experiment catalog includes
AgentArch as a reported evaluation whose original response data were not
located; it does not create placeholder attempts.

| Study | Experiment ID | Benchmark IDs | Released attempts analyzed |
| --- | --- | --- | ---: |
| [Onlycodes](https://arxiv.org/abs/2607.10569) | `onlycodes:main` | `swebench`, `onlycodes_artifact` | 3,474 |
| [Tool-interface study](https://arxiv.org/abs/2608.11386) | `tool_architecture:main_live` | `swebench_live` | 11,700 |
| [AgentArch](https://arxiv.org/abs/2509.10769) | `agentarch:main` | `agentarch_time_off`, `agentarch_customer_routing` | 0: original outputs unavailable in inspected release |

The two Onlycodes task suites are analysis scopes within its common three-arm
experiment. AgentArch's two use cases share its reported architecture experiment.
These grouping choices are explicit in `catalog-mapping.json`; they are not
inferred from filenames. The earlier analysis files retain their original scope
identifiers so the measured contrasts and uncertainty calculations are unchanged.

## Tables and files

The source studies have three curation folders in the repository's existing
`benchmarks/` tree:

```text
benchmarks/
  onlycodes/            # 193 task records, 3,474 responses, 12 comparisons
  tool_architecture/    # 65 task records, 11,700 responses, 15 comparisons
  agentarch/            # 110 input cases and architecture design tables
```

Each folder contains `build.py`, `metadata.yaml`, the authoritative `study.json`,
and generated `formatted_tables/`. These folders group source studies, while
`benchmark_id` continues to identify the task suite and `experiment_id` the
evaluation collection. A study package can include multiple benchmark IDs.
The generated `formatted_tables/manifest.json` lists table hashes and points to
the shared catalogs; it does not duplicate benchmark or experiment catalog rows.

The shared catalogs and an equivalent export partitioned by task-suite IDs are
under `artifacts/intervention_pilot/release_v2/`:

```text
catalog/
  benchmarks.parquet               # 5 records, written once
  experiments.parquet              # 3 records, written once
  experiment_benchmarks.parquet     # 5 cohort links
  arms.parquet                     # 9 observed-condition definitions
  comparisons.parquet              # 27 measured within-model contrasts
  design_configurations.parquet    # 18 AgentArch architecture templates
  design_relations.parquet         # 39 possible single-axis changes
benchmarks/<benchmark_id>/
  formatted_tables/items.parquet
  experiments/<encoded_experiment_id>/formatted_tables/
    cohort_items.parquet
    subjects.parquet               # Observed experiments only
    responses.parquet              # Both IDs; original nullable grades
    response_provenance.parquet    # Source row, repetition, arm, trace reference
    response_metrics.parquet       # Resource metrics and explicit derived scores
schema.json
sources.lock.json
catalog-mapping.json
verification.json
```

There are no one-row benchmark or experiment metadata files in the data
partitions. Experiment IDs are URL-encoded in directory names only; their
original values are retained in the tables. Six Onlycodes system descriptions
are reused across its two task-suite partitions; overall there are 27 distinct
observed subject descriptions, not 33 independently defined subjects.
The study packages deduplicate those shared subjects locally. Both layouts
preserve the same response IDs and must not be combined as additional runs.

The response schema is:

```text
response_id, experiment_id, benchmark_id, subject_id, item_id,
trial, test_condition, interactors, response
```

Its uniqueness rule includes `experiment_id`. An automated test demonstrates
that the same subject, item, and trial can appear in two experiments with
different response IDs. This is a synthetic validator test, not an additional
experiment inserted into the released data. Grade corrections retain attempt
identity; reanalysis should link to the existing observation.

## Evidence preserved

- **Onlycodes:** all 3,474 PASS/FAIL outcomes are present. One failed attempt has
  missing token and cost measurements; its grade stays available. The existing
  efficiency analysis reports task exclusions for affected comparisons. The
  two Claude Code versions remain separate subject descriptions.
- **Tool-interface study:** all 11,700 main-cohort records are present, including
  548 null/unknown recorded grades. The author's all-attempt success rule is a
  separate metric. All main-cohort trace member references were checked. The
  additional 30-repeat and other-task cohorts were not combined with this
  release; overlap and outcome-revision questions require separate auditing.
- **AgentArch:** 60 time-off and 50 customer-routing input cases are included,
  along with their released ground-truth structures. The 18 supported architecture
  templates exclude compact-memory ReAct. The 39 relations enumerate changes to
  one architectural axis; they are not empirical effect estimates, and their
  reference/target direction does not express an ordering of quality. Mocked
  tool data were not mistaken for model responses. The pinned GitHub revision
  still had no result files, releases, or tags when rechecked.

The AgentArch input counts describe the inspected repository release, not a
verified reconstruction of which exact cases were used in every published run.

## Validation and remaining boundary

The full build verified source hashes, explicit Arrow types and nullability,
primary keys, experiment/benchmark membership, item ownership, response identity,
subject and arm references, matched comparison cohorts, metadata counts, and
Parquet round trips including schema metadata. The focused nine-test suite
passed. The earlier 27 contrast calculations were preserved; all 54 tool-interface
consistency values still match the author's reproduction output.

These are **pilot tables**, marked `intervention-pilot-v2`, with provisional IDs.
They are not production-canonical imports. The 193 Onlycodes task records are
source references with missing full stimuli; the 65 tool-interface records
contain released task text; the 110 AgentArch records contain input utterances
and ground-truth structures. Full prompts, assets, grading implementations, and
complete historical system settings still require the canonical curation audit.
Each record carries its metadata-completeness status. Benchmark catalog IDs also
need reconciliation with the existing production registry.

Run from the repository root:

```bash
python scripts/analyze_measurements/intervention_pilot.py --offline
python -m unittest tests.test_intervention_pilot_tables -v
```

The first command regenerates both the analysis and v2 tables from verified
cached inputs, including all three study packages. A package's `build.py` runs
the same shared analysis and exports only that study's local tables. Use
`--package-root /path/to/packages` for an alternate export destination. The
package metadata marks these pilot outputs `release: withheld` until production
schema integration and metadata auditing are complete.
Omit `--offline` to fetch missing pinned inputs. No agents or
model APIs are invoked. The production schema, published datasets, and website
are unchanged.
