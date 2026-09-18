# Reproducing benchmark evaluations

A pilot checks whether released code can execute and grade fresh model attempts
on three tasks selected randomly before execution. A model may fail every task
while the evaluation works correctly. Three tasks do not establish reproducibility
of the whole benchmark or reproduce its historical scores.

## Common layout

```
benchmarks/<benchmark>/
  metadata.yaml                    # Upstream URLs and revisions
  reproducibility.yaml             # Environment, model, commands, limitations
  raw/                             # Immutable upstream code, inputs and assets
  reproduction_checks/
    check.sh                       # Complete clone-to-run procedure
    selection.json                 # Candidate population, seed and frozen tasks
    luna.json                      # Model, inference settings, prices and dollar cap
    helpers/                       # Explicitly identified custom code
    run_results/<run-id>/           # Logs, full outputs, usage, writable work/
      native/                      # Inputs for the existing builder, when exportable
      tables/                      # Separately validated fresh observations
```

`raw/` and `run_results/` are excluded from Git; preserve them with the dataset
archive. Captures are never patched or reformatted. Installations and runtime
writes use `work/`. A different source revision receives a new capture path.
The procedure records hashes before capture, after capture and after execution.

Published pilot records are available in the
[Hugging Face dataset](https://huggingface.co/datasets/aims-foundations/measurement-db/blob/main/reproduction_results.md).
Each benchmark's `raw/reproducibility/` contains its upstream source archives;
`reproduction_checks/run_results/` contains the saved procedures, selections,
logs, API usage, native results and fresh tables where available. Extracted source
copies, installed environments and build caches can be recreated by `check.sh`
and are omitted from the archive. Earlier failed attempts remain documented.

## Run and review

Use Linux with Git, network access, and enough disk space for upstream dependencies.
The manifest and Bash procedure declare each benchmark's Python and, when needed,
Docker requirements. Keep the repository and caches on a data volume.
Read the benchmark's `check.sh` before running; its comments identify every custom
helper and adaptation. From the repository root:

```bash
benchmark=multi_swebench
bash "benchmarks/$benchmark/reproduction_checks/check.sh" --prepare-only
# Set OPENAI_API_KEY privately in the environment before fresh inference.
bash "benchmarks/$benchmark/reproduction_checks/check.sh"
```

Each invocation creates a new run directory. `REPRO_RUN_DIR` can name a new
directory explicitly. A run retains its scripts and configuration in `procedure/`,
stage logs, `result.json`, and per-task `task_results.json`. Upstream controls test
the grader before paid inference. Selected tasks are never replaced because they
are difficult, unavailable, or fail. A task whose reference solution fails its
control remains an environment/grading issue rather than a model failure.

The model uses medium reasoning through a local API proxy. Native prompts,
tools, stopping rules and grading remain upstream-owned. The proxy supplies
provider compatibility parameters and reserves estimated dollars before each
request, including retries. The ledger at `reproduction_checks/run_results/api-budget.json`
is shared across attempts; rerunning does not reset spending. Missing provider
usage retains the reservation. The cap is conservative estimated API spending,
not a provider billing guarantee. Upstream Docker, package and network costs are
not API spending. Only the native client receives a dummy local key; the real
credential stays in the proxy process and is not saved with the run.

`graded` requires fresh inference and agreement with saved-output regrading.
For an LLM judge, distinguish reparsing the saved judgment from requesting a new
stochastic judgment; the manifest and task record identify which was tested.
`budget_exhausted`, `blocked`, and `error` are not graded failures. A native step
limit may yield a recorded failure without executing a submission grader; this
is labeled separately and does not establish grader reproducibility. Full native
traces remain in the run, including invalid generated code. The trace syntax
auditor flags recognized JSON/code errors without dropping attempts.

## Building fresh tables

Builders accepting `--source` use their existing parser and the same shared
schema/integrity validator, without downloading or replacing released data:

```bash
python "benchmarks/$benchmark/build.py" --source /path/to/run/native --output /path/to/run/tables
```

The source directory must have the benchmark's upstream input layout. Fresh run
exports record the actual model and settings in `subject_settings.json`; they
must never inherit a historical model label. Release-specific characterization
counts apply to the published dataset, not a three-task pilot. Each written table retains its canonical schema; an assets table is written when assets are present. Setup failures and ungraded interruptions are
kept in run records and are not converted into observed successes or failures.

## Automated checks

GitHub Actions validates contracts, frozen random selections, Bash syntax,
immutable-input handling, trace syntax auditing and API budget accounting using
offline fixtures. It does not spend API dollars or launch untrusted benchmark
containers. Shared implementations live here; the private extension imports or
links these files instead of maintaining copies.
