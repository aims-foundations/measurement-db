# AI Measurement Data Bank

The AI community invests considerable effort in designing and running evaluations. Yet item-level measurement data—what system answered which item, under what conditions, and with what outcome—remain scattered across papers, repositories, leaderboards, and incompatible file formats.

The AI Measurement Data Bank is community-owned infrastructure for turning fragmented evaluation results into shared, reusable evidence. It supports systematic efforts to compare benchmarks, study validity and reliability, track capabilities over time, and design better evaluations.

- [Explore the AI Measurement Data Bank](https://aimslab.stanford.edu/measurement-db)
- [Access the released data](https://huggingface.co/datasets/aims-foundations/measurement-db)

The [`benchmarks/`](benchmarks) directory contains the six released examples—MathArena,
MMDocRAG, Multi-SWE-bench, REAL, ResearchCodeBench, and SWE-rebench—and additional
benchmarks migrated to the same tabular builder contract. Each includes source
metadata, a reviewed characterization, and a curation record. Publishing a builder
here does not imply a Hugging Face data release. Raw inputs, generated tables,
and model-fitting outputs are excluded from Git.

## Website

The website lives in [`website/`](website), with its rendering scripts in
[`scripts/render_website/`](scripts/render_website). See the
[gallery guide](scripts/render_website/README.md) for installation, local preview,
data refresh, and validation commands. Viewer data is built directly from
`aims-foundations/measurement-db` on Hugging Face and included in the Vercel
deployment; no private-bank fallback or separate gallery service is used.

Merging changes under `benchmarks/` into `main` automatically refreshes and
deploys the website using the tables already published on public HF. The workflow
discovers new HF benchmarks automatically and retains the website's explicit
hidden-benchmark list. It does not execute benchmark builders or upload tables;
the benchmark reproducibility workflow remains a separate correctness check.
If tables are released after the merge, manually run **Deploy website to Vercel**
on `main` to refresh the website.

PRs targeting `main` from branches in this repository deploy a preview when they
change benchmark or website files. Open the **Deploy website to Vercel** check
to find its website link. You can also run the workflow manually on a feature
branch; only `main` deploys production. Fork and Dependabot PR deployments are
skipped because GitHub does not provide the required deployment secrets.

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

`metadata.yaml` records benchmark facts and upstream provenance. Under build
contract 2, `sources.upstream` lists source URLs and revisions. Named entries also
declare which files to download and their destinations under `raw/`. Keep transformations
in `build.py` and reviewed expectations in `characterization.yaml`.
Optional `build.parameters` groups string constants such as source paths, marker
patterns, and verbatim prompt text inside `metadata.yaml`. The shared builder
validates their structure and exposes them as `self.build_parameters`, including
when importing fresh runs with `--source`. ResearchCodeBench provides an example.

Each public builder explicitly selects its inputs in a short hook:

```python
def download(self):
    return self.fetch_sources("results", "tasks")
```

The names refer to `sources.upstream` entries in `metadata.yaml`. For GitHub and
author-owned HF repositories, `revision` is a full commit SHA; `files` contains
full-match path expressions and raw-relative destination templates. Named captures
can be inserted into the destination, and `{path}` preserves the upstream path.
For a single HTTP URL, specify `file`, `size`, and `sha256` instead. See REAL for
both forms. MathArena uses `fetch_sources("*")` to select all its named releases.
Unnamed entries document additional references without downloading them.

For GitHub files stored with Git LFS, set `git_lfs: true` on that source.
The downloader verifies the pointer against the pinned commit, then downloads
the large file and verifies its declared SHA-256 and size.

For file-level DVC releases, `dvc_index: <registry source name>` reads pointers
from a named GitHub source at the same commit. Set the data source URL to the
HTTPS DVC cache root; `files` matches original paths without `.dvc`, and
`tree_sha256` pins the selected paths, sizes and MD5 hashes. The downloader
verifies both the original pointers and the downloaded files. Directory pointers
are not supported.

Encrypted JSON releases can use the shared `read_gpg_json` reader (requires
GnuPG). The provider's public password belongs in metadata; decoding uses an
isolated temporary directory and leaves captured inputs unchanged.

Public GCS sources use a bucket URL, an object `prefix`, the same `files` rules,
and a `tree_sha256` fingerprint of the selected object paths, generations, sizes,
MD5 checksums and content encodings. Downloads select each recorded generation
and verify its bytes; a changed inventory requires review. Gzip-encoded objects
stay compressed under `raw/` with a `.gz` suffix, so their captured bytes match
the provider's checksum.

For HELM releases, `helm_index: {source: release, group: <scenario>}` selects
run directories from a separately declared, checksum-pinned HTTP manifest.
The GCS prefix is the project root (for example, `safety/`); file patterns then
match the manifest's versioned run paths. This preserves the release's actual
model panel without listing hundreds of file URLs in metadata.
Omit `group` to select the complete release panel, including all its tasks.

For static transcript sites, `html_index: <source name>` selects same-site links
from a separately pinned HTML index. The `files` patterns select pages, and
`tree_sha256` pins their paths, sizes and SHA-256 content hashes. The downloader
checks the complete selection, including cached files, before building tables.

JSON manifests use `json_index: {source: manifest, records: [models, runs],
path: "{run_id}.json"}` to select files beneath the source URL. `records`
walks nested object/list fields, and `path` uses values from each resulting
record. `source` can also name another JSON-indexed collection, for example when
each trajectory lists its screenshots. The manifest checksum and each collection's
`tree_sha256` pin membership and file contents; duplicate paths, changed bytes,
dependency cycles and paths outside the source are rejected.

Public data APIs can declare a JSON POST body with `request_json` on a pinned
HTTP source. A JSON index can supply `query` parameter templates for an API
endpoint while `path` names each local document. HTML embedded in those JSON
documents can link source files through, for example,
`html_index: {source: items, field: itemHTML, tag: img, attribute: src}`.
The same content hashes and immutable-cache checks apply to these API responses.

Public Google Drive folder URLs support the same `files` rules. Their
`tree_sha256` pins selected relative paths, Drive file IDs, sizes and SHA-256
content hashes. The shared loader visits the complete folder tree and checks
cached bytes as well as downloads; unavailable folders or a changed selection
stop the build. No per-file URL list or Google login is needed for public folders.

The shared downloader creates `raw/`, verifies files against the upstream repository's
hashes or the declared HTTP checksum, and fetches missing inputs. Existing raw files
with different contents cause an error rather than being overwritten. `self.source_files`
lists the verified paths; use it so unrelated local caches do not enter a build.
Repository inventories require network access, including when files are cached.
Contract 1 remains available for older builders.

Check new definitions without downloading data:

```bash
python -m scripts.build_measurement_tables.validate_benchmark_metadata --require-current benchmarks/*/metadata.yaml
```

Each benchmark also requires a colocated `characterization.yaml`, governed by
[`characterization_schema.yaml`](characterization_schema.yaml). It contains
reviewed table counts and content hashes plus a nonempty `source_claims` section.
Claims cite precise locations in papers, official posts, or released data and
identify whether values were reported or independently derived. The shared
checker requires every claim to be verified by the benchmark's source audit.
See [the characterization guide](characterization.md) for fields and review steps.

```bash
python -m scripts.build_measurement_tables.validate_characterization --benchmarks-dir benchmarks
```

GitHub checks validate both definitions and run the shared unit tests for every
pull request. Full benchmark tests compare generated tables and source evidence.

Use one entrypoint for dataset checks: `python tests/test_benchmark_datasets.py <slug>`
or `python tests/test_benchmark_datasets.py --all`. The runner reads each benchmark's
characterization and includes its source-specific checks from `tests/benchmarks/`.
Missing data fail by default; `--allow-missing` skips absent tables on code-only
checkouts. `--list` lists the reviewed datasets covered by the runner.

Optional `benchmark.version` records the provider's benchmark release in
`benchmarks.parquet.version` (for example, `version: "2.0"`). Omit it or use
`null` when unknown; schema versions, `build.contract_version`, and source
revisions describe separate concepts.

Set `benchmark.release_date` explicitly to `null` when unknown. Known dates use
quoted `YYYY-MM` or `YYYY-MM-DD` strings with valid calendar values; retain the
available precision rather than guessing a month or day.

Every `metadata.yaml` must explicitly set `benchmark.release` to `public` or
`withheld` in this repository. `public` records the decision to publish the
formatted tables under their source terms. `withheld` prevents publication and
skips reproduction CI; it requires a nonempty `benchmark.release_reason`.
Keep the underlying terms in `license` and evidence in `curation_record.md`.
Private benchmarks belong in the separate repository; shared tooling still
recognizes `private`, but this repository's metadata check rejects it. The
template starts withheld. Missing legacy inline decisions also default to
withheld. `release_reason` is authoring metadata and adds no Parquet column.

The [2026-10-04 release inventory](docs/release-reconciliation-2026-10-04.csv)
records all 290 previous values, decisions, reasons, and existing license and
curation evidence. It is a migration snapshot; `metadata.yaml` remains the
current source of truth. The repository policy is public unless the reviewed
evidence identifies a concrete redistribution restriction, unmet source-access
condition, sensitive content in the formatted data, or unavailable exact author
inputs needed for reproducibility. A missing or historical
private flag, an unknown license, and noncommercial or attribution conditions
alone do not make a benchmark withheld. Unclear licensing remains documented
in the inventory and source metadata; public is a repository decision, not a
new license grant or a claim of legal clearance. Earlier holds based only on
unclear licensing are superseded explicitly in release_reason. Raw-only holds
remain in the curation records where the formatted tables exclude that material.
No private definitions were moved and no publication tooling was changed in
this migration. Upload review must still respect the original source terms.

Source reproducibility holds record the missing or changed input and the
conditions for restoring public release in `release_reason` and the curation
record. Preserve builders, pinned source hashes and historical measurements.
Restore public release after verified author-source recovery and strict table
reproduction, or a separately reviewed version update using immutable author
inputs. Temporary connection failures or long download times alone do not
justify withholding a benchmark. A reproducibility hold does not change its
recorded license or imply a redistribution restriction.

Fixed verifier descriptions belong in the optional `grading.verifiers` mapping
in `metadata.yaml`, exposed to builders through `self.grading`. Each named
description is a nonempty JSON-compatible object; its fields describe the
upstream grading protocol. Builders select and serialize these descriptions into
the existing item verifier specs. An optional `grading.fallback_rule` records the
known criterion when item-specific rules are unavailable; `grading.rule` records
a criterion shared by all items. Item-specific rules
come from the source tables, and transformations that derive response values
remain in the builder. REAL provides an example; no Parquet columns are added.

## Building from DataFrames

`BenchmarkBuild.build_tables()` accepts a benchmark-specific transformation of
raw inputs into pandas DataFrames. Return a dictionary containing `subjects`,
`items`, and `responses`, with optional `traces`. The
[REAL builder](benchmarks/real_webagents/build.py) demonstrates JSON normalization,
joins, grading, and trace selection before registration. The
[Multi-SWE-bench builder](benchmarks/multi_swebench/build.py) follows the same
staged method with complete JSONL records, explicit uniqueness checks, and
patches linked to their source runs.

| Input table | Required columns | Optional columns |
| --- | --- | --- |
| `subjects` | `subject_key`, `raw_label` | `features`, `access_date` |
| `items` | `item_key`, `raw_item_id`, `content`, `grading_criterion`, `verifier` | `attachments`, `features`, `verifier_features` |
| `responses` | `response_key`, `subject_key`, `item_key`, `response` | `trial`, `test_condition`, `interactors` |
| `traces` | `response_key`, `trace` | — |

Keys are temporary string or integer identifiers linking these input tables;
they are not exported. Each table's own key must be unique and non-null, and
every reference must resolve. The shared layer derives canonical IDs using
the existing registration rules, so different source keys can resolve to the
same subject or item. If `trial` is omitted, attempts are numbered from one in
response-table order within each canonical subject, item, condition, and
interactor combination. Supply `trial` explicitly when the source records it.
Trace rows link through `response_key` and preserve the complete text.

Other columns use the corresponding `add_subject`, `add_item`, and `add_response`
arguments, including `ExactMatcher`/`Judge` verifier objects. Missing DataFrame
cells become `None`; grading-scale and finite-value checks still apply. Item
attachments use the existing asset handling, and benchmark statistics are
computed centrally. These are in-memory input tables; the six published
Parquet schemas remain unchanged. Existing builders can continue implementing
`build_subject_item_response_rows()` with the `add_*` methods.

## Building and restoring sources

Clone this repository into a directory named `measurement_db`, install the dependencies,
and run a builder. It downloads directly from the authors' upstream sources and writes
the validated tables under `benchmarks/<slug>/formatted_tables/`:

```bash
pip install -r requirements.txt
python benchmarks/real_webagents/build.py
```

Replace `real_webagents` with another benchmark to rebuild it. Normal public builds
do not need access to the MeasurementDB HF dataset. Upstream sources may have their
own access requirements. Revisions and checksums preserve the selected release;
if an unversioned endpoint changes, the build fails instead of silently accepting
different observations.

Our [HF archive](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/main)
preserves historical inputs when upstream sources change or disappear. Select it explicitly:

```bash
python benchmarks/real_webagents/build.py --archive
```

Archive restoration requires access to that HF dataset and uses the shared pinned
revision. `MEASUREMENT_DB_SOURCE_REVISION` selects another full archive commit SHA;
`MEASUREMENT_DB_SOURCE_REPO` selects a different archive repository and requires an
explicit revision. These existing environment overrides also select archive mode.
Older private definitions without named selections retain their existing archive behavior.

MeasurementDB maintains reproducible data curation from captured upstream inputs.
Running new model evaluations and maintaining their execution environments are
outside this repository's scope. Source provenance, released traces, and the
checks needed to rebuild and audit the curated tables remain part of the project.

### Checking published table bytes in CI

The `Benchmark table reproduction` workflow checks PR creation, PR updates,
and branch pushes, and rebuilds the benchmark folders changed under `benchmarks/`.
It reads the full Git diff instead of GitHub's 300-file path-filter window,
including metadata, documentation, nested files, renames, and deletions.
Template changes and new branches check all builders. A manual run can select a
single benchmark or all benchmarks. Events with no benchmark changes run only
the offline CI tests and change detection, without accessing HF or rebuilding.

Set the repository's **`HF_TOKEN` Actions secret** to a read token whose account
has access to `aims-foundations/measurement-db`. The dataset is gated; missing
credentials or access prevent reference comparison but do not block upstream
builds. Author-hosted gated sources still require their own access through this
token. GitHub sources use the automatically supplied `GITHUB_TOKEN`.
Fork PRs are skipped; their reviewed commits
need a branch in this repository to run this check with repository secrets.

Selection first reads each changed benchmark's `metadata.yaml`. Withheld
benchmarks are excluded from the job matrix before build dependency installation,
source downloads, or reference access. The complete report lists each as
**SKIPPED (WITHHELD)** with its reason; skips are neither passes nor failures.
An all-withheld selection runs no reproduction jobs and does not resolve HF.
Missing, invalid, duplicate, or private release declarations fail selection.
Direct `verify` commands apply the same release check before executing a builder.

Each run with public benchmarks attempts to resolve the HF `migration/tabular-builders-20260924` branch
to one immutable commit shared by all jobs. If that preparation fails, upstream
builds still run and successful builds report **REFERENCE ERROR**. Jobs never
fall back to independently resolving a moving HF branch.
Each selected public benchmark first executes `benchmarks/<slug>/build.py`
in a temporary source checkout, downloading the authors' inputs declared in
`sources.upstream` in the benchmark's `metadata.yaml`. The HF commit supplies
only the expected output tables; it does not select the build's raw inputs.
Only after a successful upstream build does the job list and download those
reference tables and compare them. Missing reference tables therefore cannot
prevent an upstream build from being attempted.
The check clears `MEASUREMENT_DB_SOURCE_REPO`, `MEASUREMENT_DB_SOURCE_REVISION`,
and `MEASUREMENT_DB_SOURCE_MANIFEST` so inherited archive overrides cannot
redirect those downloads. Existing local raw inputs and tables are never reused
or modified. The comparison uses `<slug>/formatted_tables/*.parquet`
on HF, or the older flat `<slug>/*.parquet` layout when the modern directory is
absent. Only the legacy filename `response.parquet` maps to `responses.parquet`;
no table bytes are normalized or rewritten.

Success requires the exact same set of tables and byte-identical files, including
optional traces and assets, row order, and Parquet metadata. Missing references,
missing builders, build errors, extra tables, and any byte difference fail.
SHA-256 values for both files appear in the job log. Jobs continue independently
after another benchmark fails. The workflow uses up to 16 concurrent reproduction jobs;
changes exceeding GitHub's 256-job matrix limit are split into small batches.

Jobs use Python 3.11 unless `build.parameters.runtime.python_version` declares
a quoted version in the benchmark's metadata. Batches contain only benchmarks
using the same interpreter. LawBench pins Python 3.12.12 because its native
scorer's floating-point summation must reproduce the published grades; use that
version for local LawBench builds too.

The final **Benchmark report** job publishes a table on the workflow run's
**Summary** page, with one row per selected benchmark: overall result, upstream
build status, comparison status, elapsed time,
failure details, and a link to its job log. Each grouped job also publishes its
own summary. The **benchmark-run-report-<attempt>** artifact contains
`summary.md`, `results.csv`, and `results.json` with full failure messages.
Results are checkpointed after each benchmark and uploaded even when a job
fails. A passing benchmark remains visible if another benchmark in its group
fails. **BUILD FAILED** identifies an upstream download/build failure.
**COMPARISON FAILED** means the build passed but comparison failed, including
byte differences and missing or extra generated tables. **REFERENCE MISSING**
means the build passed but published comparison tables are absent;
**REFERENCE ERROR** covers reference access, resolution, and network failures.
Neither reference status counts as a reproduction pass; the workflow remains
unsuccessful unless every selected benchmark builds and matches its references.
**NOT RUN** means verification never started; **INTERRUPTED** means it
started without completing; **NOT REPORTED** means no usable artifact arrived.
Missing reports are never counted as passes. Reruns use the latest result for
each group at the same source and HF revisions, retaining prior successful
groups when only failed jobs are rerun. A forcibly cancelled run or an unavailable
runner can prevent the final report from being published; already uploaded group
reports remain available. Existing historical runs do not gain these reports.

All workflow jobs target a native Linux x64 self-hosted runner with the custom label
`measurement-db-reproduction`. The runner runs as a systemd user service on
`skampere1`, with its installation, runner home, tool cache, downloads,
and temporary builds under
`/lfs/skampere1/0/sttruong/aims/measurement-db-ghaction`. It needs no Docker,
interactive login, or AFS credentials. Each job creates a fresh Python virtual
environment. Sixteen independent runner instances process one job each. The original
instance lives at the installation root, and the other 15 live under
`runners/02` through `runners/16`. Each has its own checkout, Python tool
cache, download cache, and temporary directory. Full runs still group benchmarks
in pairs; each pair runs sequentially within its job, so at most 16 benchmarks
are being rebuilt at once. Reproduction jobs configure OpenMP, OpenBLAS, MKL,
NumExpr, and Arrow CPU pools to use two threads, and Arrow I/O pools to use four.
These are library settings, not an operating-system CPU quota. Larger pools
require more runner installations and a higher `max-parallel` setting.

An existing run keeps the workflow configuration from its original commit.
GitHub's **Re-run jobs** also uses that commit. To apply a new parallelism limit,
start a fresh **Run workflow** dispatch on the updated branch. A new manual run
on the same branch cancels the previous manual run through the workflow's
concurrency group; preserve any needed results before replacing an active run.

Before activating this runner for the public repository, configure **Settings >
Actions > General > Fork pull request workflows** to require approval for all
outside collaborators. Review all workflow and executable changes before
approving a fork run. The fork condition in this workflow prevents accidental
execution; it is not an access-control boundary because a PR can modify workflow
files. The service runs as `sttruong` and has that account's filesystem access.
It needs no sudo: user lingering keeps systemd running after logout and starts
it at boot. A user crontab restores the service definition under `/run/user`
after reboot, avoiding the AFS home directory. The installer is
`deployment/install-user-service.py` in the installation directory. Use
`deployment/runnerctl status`, `logs`, `stop`, or `start` to manage all 16; append
an instance number from `1` to `16` to manage one instance. `stop` also prevents cron from
restarting the selected instances. Trusted push/PR events and manual
dispatches are picked up automatically once the workflow is pushed.

The CI constraints pin pandas and PyArrow to versions recorded in the published
migration tables. To run the same comparison locally after authenticating to HF:

```bash
python -m pip install -r requirements.txt -c scripts/ci/reproduction-constraints.txt
python scripts/ci/benchmark_reproduction.py verify xstest
```

Pass `--revision <HF-commit-SHA>` to repeat a comparison against an exact snapshot.

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

## Local benchmark layout

Each benchmark keeps its builder and metadata at the folder root, captured upstream
inputs in `raw/`, and generated Parquet tables in `formatted_tables/`:

```text
benchmarks/<benchmark>/
  build.py
  metadata.yaml
  characterization.yaml
  raw/
  formatted_tables/
    benchmarks.parquet
    subjects.parquet
    items.parquet
    responses.parquet
    traces.parquet       # when available
    assets.parquet       # when available
```

Builders write to `formatted_tables/` automatically. Generated files remain ignored
by Git. Hugging Face keeps its existing `<benchmark>/<table>.parquet` paths; the
publisher maps the local table folder to those paths.
