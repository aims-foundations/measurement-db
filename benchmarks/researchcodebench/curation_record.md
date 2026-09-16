# ResearchCodeBench — curation record

Curated 2026-09-07. The measurement rows come only from the benchmark
providers' released greedy-run artifact. Later third-party re-evaluations,
adapters, and mirrors are documented below and were not ingested.

## Paper and qualification decision

**ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning
Research Code** ([project](https://researchcodebench.github.io/),
[arXiv 2506.02314](https://arxiv.org/abs/2506.02314),
[OpenReview 3k70Vt0YFS](https://openreview.net/forum?id=3k70Vt0YFS),
[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cd0d0a873cc3e601c76f46dccc3d4c5f-Abstract-Datasets_and_Benchmarks_Track.html))
measures whether a model can implement a novel machine-learning contribution
from its paper and surrounding reference code.

The providers purposively selected 20 recent papers from ICLR, NeurIPS, CVPR,
and arXiv. They favored papers whose central contribution was clearly described,
cleanly implemented in a public repository, and suitable for independently
testable code-completion tasks. The tasks were co-developed or checked with
paper authors or domain experts. This is a curated coverage sample of recent ML
research, not a probability sample of papers or software-engineering work.

Each task removes one named region from a reference implementation. Regions may
be nested, allowing both coarse and fine-grained versions of a contribution.
The model receives the paper, the target source file with one region replaced by
a TODO block, and any declared context files. The submitted code is inserted
back into the original project and evaluated by execution-based correctness
tests. The provider describes a hybrid of equivalence tests and targeted unit or
integration tests, rather than string similarity or an LLM judge.

ResearchCodeBench qualifies for item-level ingestion because the providers
released:

- the annotated code and paper/context material needed to reconstruct the
  subject-visible tasks;
- the reference implementations removed from those tasks; and
- a complete per-model-configuration, per-snippet pass/fail matrix for the
  headline greedy run.

## Provider-owned sources and immutable revision

The official project page links directly to the providers'
[`PatrickHua/ResearchCodeBench`](https://github.com/PatrickHua/ResearchCodeBench)
repository and the paper. The build uses commit
[`2758001c2ff84fc25c546339d65479ed058b0265`](https://github.com/PatrickHua/ResearchCodeBench/commit/2758001c2ff84fc25c546339d65479ed058b0265),
dated 2025-05-16. This is the first repository commit containing the released
`2025-05-12-17-13-20` result artifact.

The provider archive is pinned as:

```text
URL:    https://codeload.github.com/PatrickHua/ResearchCodeBench/tar.gz/2758001c2ff84fc25c546339d65479ed058b0265
file:   researchcodebench-2758001c2ff84fc25c546339d65479ed058b0265.tar.gz
size:   1,085,442 bytes
SHA256: 0ea1c8e5ccd1f223a168d47e9a744a9d5f0baca80225c75036dfe3c38557a857
```

Its relevant layout is:

```text
ResearchCodeBench-2758001c2ff84fc25c546339d65479ed058b0265/
  README.md
  environment.yml
  main.py
  run_greedy.sh
  core/
  pset/<paper>/...
  outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json
```

The archive has 303 members: 244 files and 59 directories. The selected legacy
prompt inputs comprise 113 pset files across all 20 papers: 86 Python files, 19
`paper2code_paper.tex` files, and eight `paper2code.yaml` files. Every selected
file and the result JSON are byte-identical to the previous downloads from the
mutable `main` branch. A later provider commit, `db3d16d...`, also matches those
selected bytes, but the result-introduction commit provides the clearer
historical revision boundary.

The released result member is
[`overall_stats.json`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json):

```text
size:   1,735,830 bytes
SHA256: 612e6329b42037b8a04944174e673799b9f84df898390bd154996aaa339e8eea
```

The provider-owned project-site repository publishes the same JSON under its
[`leaderboard`](https://github.com/ResearchCodeBench/ResearchCodeBench.github.io/blob/5410f0f8779e80cb13764cfd1b3dc87e0f34b651/leaderboard/data/overall_stats.json);
that copy is byte-identical and corroborates the repository provenance. It is
not a second response source.

Useful executable provenance at the pinned revision includes:

- [`run_greedy.sh`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/run_greedy.sh),
  which enumerates the 20 papers, 32 model labels, and main run flags;
- [`run_inference.py`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/core/annotation/utils/run_inference.py),
  which constructs the prompt and extracts generated code;
- [`problem.py`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/core/annotation/models/problem.py)
  and [`file.py`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/core/annotation/models/file.py),
  which load papers/context and mask snippets;
- [`pset.py`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/core/annotation/models/pset.py),
  which inserts completions, runs tests, and summarizes scores;
- [`run_shell_command.py`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/core/annotation/utils/run_shell_command.py),
  which defines the final pass predicate; and
- [`environment.yml`](https://github.com/PatrickHua/ResearchCodeBench/blob/2758001c2ff84fc25c546339d65479ed058b0265/environment.yml),
  which describes the shared Conda environment.

The repository and README do not declare a repository-wide benchmark license.
Some constituent research-code files carry their own MIT or Apache notices,
but those do not establish a license for the combined benchmark release. The
project website's CC BY-SA notice covers the website, not this archive.
Measurement-DB therefore records the benchmark license as `unknown`.

## Released and curated coverage

The ingested run is the provider's with-paper, greedy headline matrix. It is a
complete rectangle before model-registry normalization.

| component | coverage |
|---|---:|
| papers | 20 |
| items | 212 |
| raw model configurations | 32 |
| canonical subjects | 31 |
| responses | 6,784 |
| response value `1` | 2,451 |
| response value `0` | 4,333 |
| trial-1 rows | 6,572 |
| trial-2 rows | 212 |
| released or curated traces | 0 |

Every raw model configuration has exactly one released completion for every
item (`completion_idx = 0`). Each item consequently has 32 observations. The
released exit codes are 2,451 zeros, 4,272 ones, and 61 timeout codes (`124`);
all passing cells have exit code zero. Across the 212 unique snippets, the
provider records 1,449 executable reference lines.

The paper also analyzes ResearchCodeBench-HARD, a derived 109-item bottom-half
difficulty subset, and a derived contamination-safer group of 13 papers. Those
views do not add response cells and are not encoded as new Measurement-DB
settings. The paper's without-paper ablation is not present as an item-level
matrix in the pinned released artifact. The build therefore includes the full
212-item with-paper result matrix only.

## Harness, tools, budgets, and environment

The released `run_greedy.sh` invokes generation and testing with:

```text
n_completions = 1
temperature = 0
max_retries = 100
timeout_seconds = 60  (the main.py default, not overridden)
summarize_results = true
wo_paper = false      (the flag is absent)
```

The generation harness sends one user message and no system message. It exposes
no tools, repository shell, test feedback, repair loop, or interactive user to
the model. The prompt contains the paper, flattened declared context files, the
source file with exactly one target masked, fixed implementation instructions,
and a worked indentation example. The harness takes the last fenced `python`
block from the response as the submitted implementation.

The shared client routes requests through OpenAI-compatible clients for OpenAI,
Anthropic, Google, xAI, DeepSeek, and OpenRouter. The global run requests
temperature zero, but configuration-specific keyword arguments take precedence:

- O1 High, O3 High, and O3-mini High use temperature 1 and high reasoning;
- Grok 3 Mini Beta High and Gemini 2.5 Flash Preview request high reasoning;
- Claude 3.5 Sonnet and Claude 3.7 Sonnet explicitly request 8,192 completion
  tokens; and
- most other configurations have no released explicit output-token limit.

The HTTP client has a 60-second request timeout. Request failures may be retried
up to 100 times, and empty responses have a separate retry path. These retries
are transport behavior, not additional benchmark trials.

The provider README says the benchmark runs on a CPU-only machine. The pinned
`environment.yml` names a Python 3.11 Conda environment with PyTorch 2.1.2,
torchvision 0.16.2, NumPy 1.26, and a mixture of version-pinned, Git-revision-
pinned, range-constrained, and floating dependencies. No executed container
digest, operating-system image, hardware description, dependency lock, API
access dates, or per-run environment manifest accompanies the result JSON.

Legacy Measurement-DB subject rows leave `harness`, `reasoning_effort`,
`harness_version`, `access_date`, and extra subject features null. Although the
provider code supplies partial evidence for some of those fields, populating
them now would change subject identity and is outside this structural migration.

## Grader and metrics

For each completion, the provider copies the full paper project to a temporary
cache, replaces only the target snippet with the extracted code, and executes
the paper-specific Python test entry point through the shell `timeout` utility.
The headline run uses the sequential test path and a 60-second limit per test
process.

The released `passed` predicate requires both:

1. process success with exit code zero; and
2. no occurrence of `Error:`, `Exception:`, `Traceback`, or `failed` in stdout
   or stderr.

No LLM grader is involved. Measurement-DB translates `passed` to the categorical
response value `1` and all other cells to `0`.

The provider reports two aggregates:

- vanilla pass rate: the percentage of snippets whose first completion passes;
- Scaled Pass@1: the same pass indicator weighted by executable lines in the
  reference snippet.

Scaled Pass@1 is the paper's primary headline metric. Measurement-DB stores the
underlying per-snippet binary outcomes. An ordinary mean over these rows
reproduces vanilla Pass@1, not the line-weighted headline metric. The source's
per-cell line counts and aggregate `line_rates` remain validation/provenance
information rather than separate response values.

## Item processing and identity

The transformation preserves the historical Measurement-DB behavior:

- item raw identity is `<paper>::<snippet-name>`;
- all annotated Python files are scanned for matching `paper2code` marker pairs;
- if a name occurs more than once for a paper, the nonempty occurrence with the
  longest reference body wins;
- the target body is replaced by the provider's two-line TODO placeholder;
- all remaining `paper2code` marker lines are removed while non-target snippet
  implementations remain visible;
- context paths declared by `paper2code.yaml` are flattened before the target
  source file;
- `reference_answer` is the removed implementation with nested marker lines
  stripped; and
- `paper` is routed as an item feature, leaving response `test_condition` null.

All 212 released `(paper, snippet)` keys have a nonempty matched reference
implementation. Response values are taken directly from the released completion
records; the builder does not rerun the grader or infer outcomes from the
reference code.

The current metadata schema permits only `contract_version` under `build`, so it
cannot encode the legacy `SETTING_CLASSES = {"paper": "item"}` declaration.
The builder instead passes `paper` directly to `add_item(features=...)`, which
preserves the same item features and null response conditions.

### Gemini preview identity collision

The raw labels `GEMINI_2_5_PRO_PREVIEW_03_25` and
`GEMINI_2_5_PRO_PREVIEW_05_06` both resolve through the existing model registry
to the same canonical Google Gemini 2.5 Pro subject. The earlier label creates
trial 1 and has 125 passing cells; the later label creates trial 2 and has 136
passing cells. The canonical subject retains the earlier display label and
release date.

These are two provider model configurations, not two random samples from one
stochastic run. Their historical trial assignment is nevertheless preserved
because splitting or renaming the subject would change established identifiers.
No model-registry entry was added or changed for this migration.

### Preserved GPS paper-omission bug

GPS is the one paper directory whose paper is stored as
`pset/GPS/paper2code_paper.md`. The provider's `Problem.parse_problem` explicitly
falls back from `.tex` to `.md`, so the released upstream run supplied that
Markdown paper to GPS.

The legacy Measurement-DB downloader, however, selected only
`paper2code_paper.tex`, and its prompt reconstruction also looked only for that
filename. Its six GPS items therefore use the no-paper prompt header and omit
the Markdown paper even though their response values came from the provider's
with-paper run. Earlier benchmark prose incorrectly implied that every
reconstructed item included its paper.

This is a pre-existing semantic bug, not part of the schema migration. The
migrated builder deliberately retains the same 113-file selection and the same
six GPS prompts so item content and identifiers remain unchanged. Adding the
Markdown paper would alter those items and requires separate review and a new
accepted characterization.

## Traces and other coverage gaps

The released `overall_stats.json` retains only the per-cell `passed` flag, exit
code, executable-line count, and completion index, plus aggregate model
statistics and model configuration data. The pinned result directory does not
contain raw completions, API request/response envelopes, parsed submissions,
stdout, stderr, token usage, costs, or the generated run manifest. Consequently
all response traces are null and no `traces.parquet` is produced.

Provider test programs exist in the archive, but the historical builder
deliberately excluded `paper2code_test.py` files and did not register executable
item verifiers. The migrated item verifier remains null. Reconstructing a
present-day executable verifier would be useful future work but would be a
semantic expansion, not structural preservation.

Other known limitations are:

- the aggregate artifact does not bind the run to an executed Git commit or
  environment digest;
- exact endpoint revisions, SDK versions, access dates, and most output budgets
  are unavailable;
- the two raw Gemini configurations are represented as trials of one canonical
  subject;
- GPS stimuli do not faithfully reconstruct the provider run, as documented
  above;
- exit codes and line counts are not carried into response extensions; and
- no human baseline was released because the tasks require specialist
  implementation expertise.

## Deliberately excluded third-party artifacts

### Harbor adapter and parity re-evaluations

The [Harbor ResearchCodeBench adapter](https://github.com/harbor-framework/harbor/tree/main/adapters/research-code-bench)
converts the 212 snippets into Dockerized coding-agent tasks. Its
[`parity_experiment.json`](https://github.com/harbor-framework/harbor/blob/a6586232a330a98f1d1f0e2c5ca833e44a57acd3/adapters/research-code-bench/parity_experiment.json)
and [published parity dataset](https://huggingface.co/datasets/harborframework/parity-experiments/blob/f2b758254561c1511ef194341aa627757daa7691/adapters/research-code-bench/README.md)
report three trials each for Codex with GPT-4o-mini and GPT-4.1-mini under both a
direct-API reconstruction and the Harbor agent harness.

These are later third-party measurements with different model releases, three
trials, an agent/container harness, and trajectory artifacts. The Harbor adapter
also relaxes the GMFlow numerical threshold from `1e-7` to `1e-3`, producing a
documented divergence in its line-weighted metric. Neither its response rows nor
its trajectories were ingested. The `introvoyz041/parity-experiments` copy is a
duplicate mirror of the Harbor material and adds no independent coverage.

### OpenReward environment port

[`GeneralReasoning/env-research-code-bench`](https://github.com/GeneralReasoning/env-research-code-bench/tree/36099173fbc43a58bb160fb118472a06dcd0024c)
ports the Harbor tasks to a third-party OpenReward environment. It presents 212
Dockerized, multi-turn tasks with `bash`, `view`, `str_replace`, `create_file`,
and one-shot `submit_answer` tools, and documents different resource limits.
This is an agent-environment transformation, not the provider's released
single-turn model run, and it supplies no new provider-attributable response
matrix. It was not used.

### REVERE re-evaluation

[`REVERE`](https://arxiv.org/abs/2603.20667)
([project](https://revere-acceleron.github.io/revere/)) evaluates
ResearchCodeBench in a later research-coding framework. It uses alternate
baseline and adaptive prompts, offline and online exposure regimes, a
34/34/144 split for one setting, repeated runs, and evolving cross-task memory.
Its reported ResearchCodeBench results therefore measure a materially different
harness and adaptation process. They were not joined to the provider's greedy
matrix.

### Modified repository fork

[`cxiong-ship-it/ResearchCodeBench`](https://github.com/cxiong-ship-it/ResearchCodeBench)
is a downstream fork that changes numerical tolerances in eight test files. It
does not publish a replacement provider response matrix, and its altered grader
is outside this release. No files or outcomes from the fork were used.

Searches also found unrelated projects with similar names, including
`anote-ai/Research-CodeBench`; they do not implement or rerun this benchmark.
No additional provider-owned item-level response matrix was found beyond the
pinned GitHub artifact and its byte-identical official-site copy.

## Verification and publication history

Provider history:

- 2025-05-12: timestamp of the released headline run directory;
- 2025-05-16: first provider commit containing `overall_stats.json`;
- 2025-06-02: arXiv v1 publication;
- 2025: publication in the NeurIPS Datasets and Benchmarks Track.

Measurement-DB history:

- 2026-07-25: the original ResearchCodeBench builder was introduced;
- 2026-08-01 and 2026-08-09: description and databank-schema adjustments were
  applied without changing the provider matrix;
- 2026-09-07: the legacy builder was reproduced under a compatible historical
  runtime, immutable provider inputs were pinned, and the benchmark was migrated
  to the current folder/metadata/characterization structure.

The isolated legacy rebuild reproduced `items.parquet`, `subjects.parquet`, and
`responses.parquet` byte-for-byte from the cached provider inputs. Re-downloading
from provider `main` at `db3d16d...` also matched the cache, and the selected
inputs match the adopted `2758001c...` archive exactly. The prior benchmark-table
difference was only the reviewed correction from an unsupported MIT license to
`unknown`.

The migrated transformation preserves item IDs, subject IDs, prompts, reference
answers, response values, trial assignment, item features, and null patterns.
The current shared runtime necessarily regenerates all 6,784 `response_id`
values under its newer hash contract; this identifier-only difference was
reviewed and explicitly approved during migration. No raw downloads, generated
Parquet files, credentials, or third-party rerun data belong in the migration
diff.

Validation expectations are now stored beside metadata in `characterization.yaml`,
using the shared schema and table hash algorithm. The previous source audits and
migration assertions passed before conversion; release-specific assertions remain
in `test.py`. Source claims distinguish reported quantities from those independently
derived from the archived release. This format change does not alter curated data.
