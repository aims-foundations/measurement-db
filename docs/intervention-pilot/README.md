# Pilot: studies, experiments, and intervention comparisons

This pilot uses released experiments and results. It does not run agents. It
maps three papers into study records and reanalyzes the available item-level
outcomes from two of them. The second pilot release implements separate benchmark
and experiment IDs, shared metadata catalogs, and typed response partitions.
See [the completed v2 pilot](results-v2.md). This remains a prototype with explicit
metadata gaps; it does not migrate the production schema.

The three source-study packages now live alongside existing packages:
[Onlycodes](../../benchmarks/onlycodes/README.md),
[Tool-interface study](../../benchmarks/tool_architecture/README.md), and
[AgentArch](../../benchmarks/agentarch/README.md). Each contains its authoritative
`study.json`, package `metadata.yaml`, `build.py`, and generated `formatted_tables/`.
The earlier artifacts-only organization did not provide these curation folders.

## Agreed direction: benchmark and experiment are separate identities

The proposed `responses.parquet` has **both `benchmark_id` and `experiment_id`**.
This supersedes the earlier proposal to rename one into the other. A benchmark
identifies the reusable task suite/protocol release; an experiment identifies a
particular collection of evaluations under a documented design. Independent
experiments can evaluate the same subject on the same item without changing
either the subject's identity or the item's identity.

Illustrative rows (not pilot observations):

| benchmark_id | experiment_id | subject_id | item_id | trial | response |
| --- | --- | --- | --- | ---: | ---: |
| swebench | original_evaluation | agent_x | task_y | 1 | 1 |
| swebench | independent_replication | agent_x | task_y | 1 | 0 |

Both rows are valid with identical test conditions and interactors. The proposed
logical uniqueness key is `(experiment_id, subject_id, item_id, trial,
test_condition, interactors)`. `response_id` remains the primary key and traces
continue to reference it. `benchmark_id` is retained on responses for direct
filtering and partitioning, and must agree with the referenced item's owner.
New response identity derivation must include experiment identity. Item identity
remains benchmark/protocol-scoped; it must not incorporate experiment identity.
`trial` counts repetitions within the experiment and the remaining cell keys.

A standard benchmark evaluation has an experiment ID identifying its actual
evaluation release, even when no intervention comparisons are specified. A
task-only release can have benchmark and item records without experiment or
response records. An experiment may have several arms and model strata; a new
pairwise comparison does not create a new experiment or duplicate observations.
A later analysis that reuses an existing run links to that observation rather
than registering it as an independent attempt.

## Shared metadata catalogs and data partitions

Use one curation folder per source study under the existing `benchmarks/` tree:

```text
benchmarks/
  onlycodes/
  tool_architecture/
  agentarch/
    build.py
    metadata.yaml
    study.json
    formatted_tables/
```

Each of the three folders has that structure. A folder groups source provenance
and build logic; it can contain more than one task suite or experiment. Its name
does not redefine `benchmark_id` or `experiment_id`. The pilot writes each study's
items, cohorts, configurations, responses, and comparisons together, including
the relevant IDs in each table. AgentArch has design tables and inputs only.

Use a **single `benchmarks.parquet` and a single `experiments.parquet` per database
release**, with one row per benchmark/protocol release and one row per experiment
respectively. The catalogs are shared tables, not one-row Parquet files copied
into every folder. Both conventional and intervention evaluations retain the
same curation, provenance, reproducibility, and schema-validation standards.

| Table | Responsibility |
| --- | --- |
| `benchmarks.parquet` | Benchmark ID, name, version, task/protocol description, original benchmark citation, and scale metadata |
| `experiments.parquet` | Experiment ID, optional study grouping, design, evaluation dates when known, experiment citation, pinned result provenance, and result availability |
| `experiment_benchmarks.parquet` | Experiment-to-benchmark membership and cohort/subset references when experiments span multiple benchmarks |
| `items.parquet` | Reusable benchmark-scoped stimuli, grading criteria, verifiers, and item IDs |
| `subjects.parquet` | Evaluated system configurations; unchanged configurations can retain their subject IDs across experiments |
| `responses.parquet` | Both benchmark and experiment IDs, subject/item IDs, repetitions, conditions, and recorded grades |
| `comparisons.parquet` | Reference/target arms or configuration sets, experiment and cohort scope, intervention specifications, and analysis policy |

An experiment-to-benchmark link is useful even before results are available;
membership should not depend solely on inferring it from observed responses.
Multiple benchmarks in one experiment are not forced into a single benchmark
field on its metadata row. Selection manifests should resolve the actual item
IDs for each cohort.

The pilot also retains a derived export partitioned by logical benchmark ID:

```text
catalog/
  benchmarks.parquet
  experiments.parquet
  experiment_benchmarks.parquet
benchmarks/<benchmark_id>/
  formatted_tables/
    items.parquet
    assets.parquet                     # When available
  experiments/<experiment_id>/
    formatted_tables/
      subjects.parquet
      responses.parquet
      comparisons.parquet              # When available
      traces.parquet                   # When available
```

Those exported folders partition the data; they do not replace the three source
study packages above or define identities. If an
experiment spans several benchmarks, its data can occupy several benchmark
partitions while its definition appears once in the experiments catalog. Shared
subjects must agree across shards. Cross-benchmark comparisons need an explicit
cohort scope and cannot assume that source task names establish item equality.

Source builders and editable YAML metadata can remain modular in their curation
packages. A deterministic collection step generates the shared catalogs from
those definitions, rejecting conflicting records with the same ID. Avoid a
second manually maintained catalog. Validate a package against the relevant
catalog rows, and publish catalogs and data shards together at a pinned release.
Self-contained exports may bundle metadata rows for portability, but those are
derived subsets of the catalog rather than separate authoritative copies.

The current `benchmarks.parquet` mixes instrument metadata with evaluation
summaries. Separate those meanings during migration: response counts, evaluated
subject counts, max trial, result availability, and coverage belong to an
experiment or experiment-by-benchmark scope. Full item-bank size and evaluated
cohort size are different quantities. Benchmark-wide aggregate summaries, if
needed for the website, should be explicitly derived with deduplication and
denominator rules. Coverage must use the intended experiment cohort, not a
cross-product of every subject and every item ever associated with a benchmark.

## Migration boundary

The current implementation has no experiment column and requires one local
benchmark metadata row per package. This direction requires a coordinated schema
migration: table definitions, response ID generation, traces, validators,
builders, catalog assembly, reproduction checks, and readers. It is not just a
column rename or concatenation of today's metadata files.

Legacy releases can receive explicit experiment records representing the
evaluations they already contain. A package with several distinct evaluations
requires evidence-based mapping. Retain existing published response IDs through
an explicit legacy identity version where feasible; new experiment-aware IDs
need a versioned policy. Existing benchmark-scoped item IDs should be reused
when they already refer to the correct instrument/protocol. Study-scoped legacy
item namespaces may require reviewed aliases rather than an automatic relabel.

The two-ID/global-catalog model is now implemented in the separate
`intervention-pilot-v2` schema. `catalog-mapping.json` records the mapping from
the earlier study analysis scopes to five benchmark records and three experiment
records. The Onlycodes experiment spans two benchmarks, as does AgentArch. These
IDs remain provisional until reconciled with the production catalog. Item IDs
use benchmark, source task ID, and protocol reference, not canonical complete
stimulus hashing. Subject IDs use available configuration descriptions without
including study or experiment ID; full configuration auditing remains pending.

Generated study tables are in `benchmarks/<study>/formatted_tables/`. Shared
catalogs and the equivalent benchmark-partitioned export are under
`artifacts/intervention_pilot/release_v2/`. These are two layouts of the same
records and must not be imported as independent observations. The older
analysis intermediates remain under `artifacts/intervention_pilot/studies/` to
preserve the verified metric calculations; their `experiment_id` columns denote
the original finer analysis scopes. Neither format is production canonical.
Every v2 Parquet file carries `measurement_db.pilot_schema=intervention-pilot-v2`.
Explicit schemas live in `scripts/analyze_measurements/intervention_pilot_tables.py`
and are exported to `release_v2/schema.json`.

## Scope and evidence

| Study | Pilot scope | Verified released observations | Status |
| --- | --- | ---: | --- |
| [Onlycodes](https://arxiv.org/abs/2607.10569) | Two task suites, two model-harness systems, three arms, three repetitions | 3,474 | 12 within-system comparisons analyzed |
| [The Devil Is in the Interface](https://arxiv.org/abs/2608.11386) | Main SWE-bench Live cohort: 65 tasks, three actor models, six interfaces, ten repetitions | 11,700 | 15 within-model comparisons analyzed |
| [AgentArch](https://arxiv.org/abs/2509.10769) | 110 released input cases, 18 supported configurations, 39 single-axis design relations | 0 imported | Design and inputs mapped; original run outcomes not located in inspected release |

The full tool-interface archive was downloaded and its author-published SHA-256
verified. The pilot uses the processed outcome/efficiency inputs and confirms
that all 11,700 main-cohort trace filenames exist. It does not audit every trace
content or analyze the additional-task and extra-consistency cohorts.

AgentArch's pinned repository contains task inputs, mocked tools, and executable
configurations. Its illustrated `results/` directory is absent from that tree,
and no GitHub releases or tags were present at inspection. This is a bounded
availability finding, not a claim that results exist nowhere else. No response
rows were fabricated from paper percentages or mocked inputs. See its
[release audit](../../benchmarks/agentarch/release-audit.json).

## Analysis

The pilot averages repetitions within each task and arm, then differences the
task means within each model/system. Matching is on source task identity inside
each experiment; repetition indices do not establish shared random streams.
Onlycodes compares all three arm pairs. The interface study compares each of
the five alternatives with BashOnly. These are analyst-defined pilot contrasts.

Reported uncertainty is a pointwise 95% percentile bootstrap over matched task
blocks, with 5,000 draws and deterministic seeds. It is exploratory, has no
multiple-comparison correction, and is conditional on the released cohorts and
protocols. It does not establish a universal causal effect or estimate all
sources of within-task stochastic uncertainty. Repeated attempts are not treated
as independent tasks.

For the interface study, `grade_observed` preserves the Figure 4 input's
true/false/null statuses. `success_all_attempts` implements the separate Table 2
policy, under which unknown/unfinished outcomes count as failures. There are
548 unknown recorded statuses. Both are retained so an analyst can see the
denominator choice. The pilot does not silently overwrite unknown grades.

One Onlycodes failed attempt lacks token and cost fields. It remains in success
analysis. For each affected efficiency contrast, its task is excluded because
not every repetition has that metric; the output reports this exclusion.
Source-adjusted cost is retained as its own accounting measure.

Illustrative success results, with all attempts in the denominator:

| Study / system / contrast | Reference | Target | Difference, percentage points | Pointwise 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Onlycodes SWE / Claude / code-only minus native tools | 51.67% | 50.00% | -1.67 | [-6.00, 2.33] |
| Onlycodes SWE / Codex / code-only minus native tools | 46.33% | 46.67% | +0.33 | [-3.67, 4.00] |
| Interface / Qwen / Atomic minus BashOnly | 14.62% | 19.54% | +4.92 | [1.85, 8.46] |
| Interface / Kimi / Atomic minus BashOnly | 38.00% | 38.92% | +0.92 | [-2.46, 4.92] |
| Interface / Sonnet / Atomic minus BashOnly | 41.69% | 40.92% | -0.77 | [-3.69, 2.15] |

These examples are within-study contrasts, not cross-paper rankings. Full
results include input/output tokens, source-adjusted cost or steps, task-level
gains/losses, and reconstructed consistency at k=5,7,9. For example, the Qwen
Atomic contrast has higher task-level success rates on 15 tasks, lower rates on
5, and ties on 45. This is the kind of item-level pattern that aggregate scores
alone obscure.

## What the pilot changes about the proposed format

1. **Arms are not subjects.** An arm can include several models and historical
   harness versions. Onlycodes needs nine distinct available configuration
   descriptions, even though it has three arm labels. Claude artifact runs
   include two harness versions; versions match across arms within each
   task/repetition block, and 32 tasks span versions across repetitions.
2. **A comparison can change several components.** The Python interface also
   changes its controller loop. NLSearch adds a search subagent. Preserve a list
   of changes and implementation evidence instead of claiming a single isolated
   mechanism from a short arm label.
3. **Treatment settings have constraints.** AgentArch skips compact-memory ReAct
   combinations. Its 18 configurations are a constrained design, not a full
   24-cell Cartesian product. Unsupported, unrun, unknown, and observed failure
   are separate statuses.
4. **Outcome definition belongs in the comparison.** The same attempt may enter
   an all-attempt success statistic and an evaluated-only statistic differently.
   Preserve the source status, scoring policy, and denominator.
5. **Separate benchmark identity from experiment identity.** The current package
   namespace can separate studies by duplicating their item namespaces. The
   proposed model instead reuses the same benchmark/item identity across
   experiments and distinguishes their observations through `experiment_id`.
   Reanalysis of an existing run references that observation rather than
   registering another independent attempt.
6. **Task reuse requires protocol checks.** Source task IDs and prompt hashes
   are useful linkage evidence, not proof of equal item identity. Canonical
   items are benchmark-scoped and include grading protocol and effective scale.
   A shared benchmark name does not guarantee an interchangeable task subset,
   environment, or grader.

No global intervention order or distance was imposed. The manifests record
targets, operations, values, and compound changes. Later dose sweeps can add
typed numerical parameters and units without replacing this representation.

## Reproduction and validation

From the repository root:

```bash
python scripts/analyze_measurements/intervention_pilot.py
# With cached inputs; performs no network access:
python scripts/analyze_measurements/intervention_pilot.py --offline
```

Dependencies: pandas, numpy, pyarrow, requests. The first run downloads the
748,538,616-byte archive and the small pinned source files listed in
`sources.lock.json`. Raw inputs and generated results are ignored by Git.

The earlier analysis stage produces `experiments.parquet`, `configurations.parquet`,
`observations.parquet`, `comparisons.parquet`, `task_effects.parquet`,
`effects.csv`, and `coverage.csv`. The interface study also produces
`consistency.csv`. These are prototype tables with no canonical-schema claim.

The v2 stage additionally writes the shared catalogs, one item table per
benchmark, and response/subject/provenance/metric partitions for each observed
experiment-benchmark pair. AgentArch gets input-cohort and architecture records
but no response partition. Source statuses remain nullable in `responses`;
derived all-attempt success is stored separately in `response_metrics`.

Run the focused identity and integrity tests with:

```bash
python -m unittest tests.test_intervention_pilot_tables -v
```

The nine tests cover reuse of a subject/item/trial across experiments,
within-experiment duplicates, grade corrections, unknown grades versus derived
zeros, cohort ownership, cross-experiment arm references, absent-results
semantics, schema enforcement, and faithful serialization of YAML date values.

The completed run verified input checksums; unique source keys; expected task
and repetition coverage; matching task sets; configuration references; grade
policy agreement; trace member existence; and readable Parquet outputs. An
independent invocation of the author's Table 2 reproduction script agreed with
all 54 reconstructed consistency cells. All 27 success contrasts also matched
direct panel-level differences. Verification details are saved under
`artifacts/intervention_pilot/verification.json`.

The next measurement step is to inspect heterogeneous task gains/losses and
repeat variability in these two outcome releases. Production ingestion still
needs the full stimulus/configuration/grader audit, reconciliation of provisional
IDs, and migration of production builders/readers to the shared-catalog model.
AgentArch's original item-level outcomes are still needed for the same response
analysis. No agent evaluations were rerun and no production data were published.
