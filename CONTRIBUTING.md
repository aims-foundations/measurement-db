# Contributing to measurement-db

Thanks for your interest in contributing! `measurement-db` is the data-curation
pipeline maintained by the [AIMS Foundations](https://github.com/aims-foundations)
as the dataset layer under our broader Measurement Science stack. It is the
upstream of the HuggingFace dataset
[aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
that is consumed by
[torch_measure](https://github.com/aims-foundations/torch_measure)
and other downstream analyses. Contributions from outside the lab are very
welcome.

* [Project Overview](#project-overview)
* [Ways to Contribute](#ways-to-contribute)
* [Getting Involved](#getting-involved)
* [Development Environment Setup](#development-environment-setup)
* [Adding a New Benchmark](#adding-a-new-benchmark)
* [Schema and Invariants](#schema-and-invariants)
* [Public / Private Split](#public--private-split)
* [Submitting a Pull Request](#submitting-a-pull-request)
* [Citation and Attribution](#citation-and-attribution)

## Project Overview

`measurement-db` curates benchmark response data (AI systems × items × trials)
into a single long-form schema so downstream consumers can fit IRT and factor
models, do cold-start prediction, and run validity audits without rewriting a
parser per benchmark. Each benchmark directory at the repo root contains one
self-contained `build.py` that:

1. Downloads raw upstream data into `{benchmark}/raw/` (gitignored).
2. Resolves per-row subjects and items against the shared registry helpers.
3. Writes `{benchmark}/responses.parquet` (and optionally `traces.parquet`).
4. Writes registry contributions to `{benchmark}/_contrib/`.

A post-step (`scripts/merge_registry.py`) unions every contrib into the
canonical `_registry/{subjects,items,benchmarks}.parquet`.

Data files never live in git — they live on HuggingFace. Git tracks only the
code that builds the data, the schema docs, and the manifest.

See [DATA_FORMAT.md](DATA_FORMAT.md) for the full schema, benchmark inventory,
and per-field vocabularies (modality, domain, response_type, …).

## Ways to Contribute

**New benchmarks**
* Add a `<name>/build.py` that ingests an upstream benchmark into the
  long-form schema. See [Adding a New Benchmark](#adding-a-new-benchmark)
  below — the pattern is small and largely mechanical.
* Propose moving a dataset from `BENCHMARKS_PENDING` back to `BENCHMARKS`
  once upstream finally publishes the missing per-item or per-model data.

**Extend existing benchmarks**
* Back-fill `content` for items that currently have `content=None` when the
  upstream actually exposes prompt text.
* Back-fill `correct_answer` for items where the grader has a ground-truth
  reference (MCQ keys, reference patches, reference translations, test I/O).
* Extract `trace` (raw model output) into `{benchmark}/traces.parquet` when
  upstream publishes the scored generation.
* Curate subject metadata (`provider`, `hub_repo`, `params`, `release_date`)
  in `_registry/subjects.parquet`.

**Diagnostics and audits**
* Extensions to `scripts/audit_benchmarks.py` — new invariant checks,
  per-benchmark sanity reports, content-quality heuristics.
* New diagnostic scripts under `scripts/` — e.g. cross-benchmark duplicate
  detection, subject-resolution quality, response-distribution drift.

**Schema and infrastructure**
* `response_type` / `modality` / `domain` vocabulary additions when a new
  benchmark genuinely doesn't fit the current set.
* Build-time performance fixes (fewer API calls, better caching).
* Documentation, tutorials, and worked examples.

Not everything happens through a pull request — feel free to drop into our
[Discord](https://discord.gg/F6xbEwvvhb) to discuss ideas first, especially
for proposals that touch the schema or introduce a new `response_type`.

## Getting Involved

**Find something to work on.** Browse
[open issues](https://github.com/aims-foundations/measurement-db/issues) and
comment on one you'd like to take so we don't duplicate work. The
"Benchmark inventory" section of [DATA_FORMAT.md](DATA_FORMAT.md) shows
current per-benchmark status — datasets with `content% < 100%`, `trace%` empty,
or `correct_answer` null are all fair game for incremental curation.

**Ask questions.** [Discord](https://discord.gg/F6xbEwvvhb) is best for quick
questions, design discussion, and general chat. Open a GitHub issue for bug
reports, schema-change proposals, and anything that should have a permanent
record.

## Development Environment Setup

```bash
git clone https://github.com/aims-foundations/measurement-db
cd measurement-db
pip install -r requirements.txt

# Optional: build one dataset to verify your environment.
python reproduce.py mtbench
```

Raw upstream data downloads land under `{benchmark}/raw/`; outputs land under
`{benchmark}/responses.parquet` and `{benchmark}/_contrib/`. All of these are
gitignored.

Run all ready benchmarks in parallel:

```bash
python reproduce.py -j 8
```

Or a specific subset:

```bash
python reproduce.py bfcl swebench mmlupro
```

After a build, consolidate registry contributions and regenerate the
inventory table in `DATA_FORMAT.md`:

```bash
python scripts/merge_registry.py
python scripts/audit_benchmarks.py --out /tmp/inventory.md
```

## Adding a New Benchmark

1. Create `<name>/build.py`. Copy a similar existing build (e.g. `mtbench/build.py`
   for a simple HF judgment set; `livecodebench/build.py` for a multi-file coding
   benchmark; `matharena/build.py` for a multi-config math benchmark) and adapt
   the download + parse logic.

2. Declare metadata in the module-level `INFO = {...}` dict:

   ```python
   INFO = {
       'description': '...',              # one-line summary
       'testing_condition': '...',        # any interpretation quirks
       'paper_url': 'https://arxiv.org/abs/YYMM.NNNNN',
       'data_source_url': '...',          # upstream repo / HF dataset
       'subject_type': 'model',
       'item_type': 'task',
       'license': 'MIT',                  # SPDX identifier if known
       'citation': '...',
       'tags': [],
       'modality': ['text'],              # ["text"] / ["text", "image"] / ["grid"] / etc.
       'domain': ['general'],             # see DATA_FORMAT.md vocabulary
       'response_type': 'binary',         # binary / likert_5 / win_rate / fraction / ...
       'response_scale': '{0, 1}',        # free-form description of value set
       'categorical': True,               # finite enumerable responses?
       'release_date': 'YYYY-MM',         # benchmark release date
   }
   ```

3. Follow the build contract. Each `build.py` MUST:
   - Call `resolve_subject(raw_label)` for every AI-system row. Subjects must
     be AI systems — models, agents, judges, classifiers. Human annotators,
     prompts, and samples are **not** valid subjects (see [Schema and
     Invariants](#schema-and-invariants)).
   - Call `register_item(benchmark_id, raw_item_id, content, correct_answer=...)`
     for every item.
   - Call `get_benchmark_id(name, ..., modality=..., domain=...,
     response_type=..., response_scale=..., categorical=..., paper_url=...,
     release_date=...)` once, passing the INFO fields through.
   - Emit `{benchmark}/responses.parquet` with columns `subject_id, item_id,
     benchmark_id, trial, test_condition, response, correct_answer, trace`.
   - Call `ensure_unique_trials(df)` before `to_parquet` to enforce the
     primary-key invariant `(subject_id, item_id, trial, test_condition)`.
   - If upstream publishes raw model generations, also emit
     `{benchmark}/traces.parquet` with the same primary key + a `trace` column
     (see `mtbench/build.py` for the pattern).
   - Call `registry_save(_BENCHMARK_DIR / "_contrib")` at the end.

4. Add `"<name>"` to one of the three lists in `reproduce.py`:
   - `BENCHMARKS` — real per-(subject, item) response data, ready for IRT.
   - `BENCHMARKS_AGGREGATE` — multi-subject data where cells are aggregate
     rates across trials / sub-benchmarks.
   - `BENCHMARKS_PENDING` — pipeline exists but upstream data is missing,
     blocked, or pending a schema decision.

5. Test locally:

   ```bash
   python reproduce.py <name>
   python scripts/merge_registry.py
   python scripts/audit_benchmarks.py | grep <name>
   ```

   Verify the audit row shows `PK-dup = 0`, non-zero `rows`, expected
   `content%` / `binary%` for your benchmark.

6. Regenerate the inventory section of `DATA_FORMAT.md`:

   ```bash
   python scripts/audit_benchmarks.py --out /tmp/inventory.md
   ```

   and paste into `DATA_FORMAT.md` under the `## Benchmark inventory` heading.

## Schema and Invariants

The full schema lives in [DATA_FORMAT.md](DATA_FORMAT.md). The invariants
every `build.py` must preserve:

* **Subject rule.** Subjects are AI systems (models, agents, judges,
  classifiers). Human annotators, prompts-as-subjects, or sample-index-as-subject
  do not belong here — such datasets should be in `BENCHMARKS_PENDING` with a
  `BUILD_SKIPPED.md` explaining why.
* **Primary key.** `(subject_id, item_id, trial, test_condition)` must be
  unique in `responses.parquet`. The `ensure_unique_trials(df)` helper bumps
  `trial` to satisfy this if upstream has repeat observations of the same
  cell.
* **Items are identity-by-content.** `item_id = sha256(benchmark_id + normalized_content)[:16]`.
  Two benchmarks that happen to share a prompt get different `item_id`s
  because `benchmark_id` is in the hash. Items under multiple conditions
  (same content, different `test_condition`) share the same `item_id` — the
  condition lives on the response.
* **Data does not live in git.** `_registry/*.parquet`, `responses.parquet`,
  `traces.parquet`, and everything under `{benchmark}/raw/` or
  `{benchmark}/_contrib/` are all gitignored. They live on HuggingFace and
  are reproducible from `build.py` + upstream data.

## Public / Private Split

This repo is maintained as two GitHub repositories that share a manifest:

* [`aims-foundations/measurement-db-private`](https://github.com/aims-foundations/measurement-db-private)
  — every benchmark, including datasets used for competitions and
  incentive-aligned evaluations.
* [`aims-foundations/measurement-db`](https://github.com/aims-foundations/measurement-db)
  — the subset of benchmarks marked `release: public` in `manifest.yaml`.

`sync_to_public.py` copies tracked code files for release=public benchmarks
from the private repo into the public one. **If you are an external
contributor**, you'll work against the public repo — adding or improving any
benchmark there is equivalent to adding it in both (we'll sync privately-visible
changes back).

Per-benchmark release visibility is set in `manifest.yaml` (`release: public`
vs. `release: private`). Every benchmark in the repo has been audited; the
`release` field is a separate publish-to-public decision made by the
maintainers — open an issue if you'd like a benchmark promoted to public.

## Submitting a Pull Request

1. Fork the repo and create a feature branch off `main`.
2. Make your changes. Before opening the PR, verify:
   - [ ] `python reproduce.py <your-benchmark>` succeeds
   - [ ] `python scripts/audit_benchmarks.py` shows `PK-dup = 0` for your benchmark
   - [ ] `content%` is 100% wherever upstream makes prompt text available
   - [ ] `correct_answer` is set wherever a ground-truth answer exists upstream
   - [ ] `INFO` dict includes `modality`, `domain`, `response_type`,
         `response_scale`, `categorical`, `paper_url`, `release_date`, `license`
   - [ ] New benchmark is listed in one of the three `reproduce.py` lists
   - [ ] Commit messages are short, single-line, and describe the change
   - [ ] **No committed data files** — `responses.parquet`, `traces.parquet`,
         `_contrib/`, `raw/`, and `_registry/*.parquet` are all gitignored
3. Open a pull request against `main`.
4. CI will run a minimal build-and-audit for benchmarks you touched. A
   maintainer will review; most PRs need one approval before merge.
5. We squash-merge PRs by default.

## Citation and Attribution

If your contribution is substantial (e.g. a new multi-benchmark audit tool,
a schema extension, or onboarding a large new benchmark family) and leads to
inclusion in a paper derived from `measurement-db`, we are glad to discuss
author credit. See the citation block in the
[README](README.md) for the current dataset citation.
