# MathArena curation record

Migration reviewed on 2026-09-12. The user authorized correcting obvious legacy
defects as well as moving to the current schema. This is therefore a documented
corrective migration, not a claim of byte-for-byte or ID-for-ID equivalence.

## Measurement and qualification

MathArena measures mathematical problem solving on recent competitions. For
final-answer tasks, an observation is the provider's per-attempt correctness
verdict. For proof tasks, it is the awarded points divided by the maximum for
one released rubric criterion and one judge slot. These fractions are not an
average over judges, an overall proof score, or a best-of-N success rate.

The paper and descriptive benchmark facts are recorded in `metadata.yaml`.
The benchmark qualifies for item-level ingestion because the provider releases
problem/prompt inputs, identifiable AI-system configurations, attempted solutions,
and per-attempt verdicts or rubric grades. We import these observations; no model
or judge is called during a build.

## Provider-owned provenance and released coverage

The existing 27 `MathArena/*_outputs` repositories remain the only measurement
sources. The source manifest pins every dataset revision and every train shard
and dataset card by exact size and SHA-256. The dataset cards identify the
[official project repository](https://github.com/eth-sri/matharena) as the runner
and describe `model_config` as the configuration path used for an observation.
They also distinguish `user_message`, `answer`, and serialized grading details.
The build verifies cached files as well as newly downloaded files and fails on
missing or corrupt artifacts instead of silently publishing partial coverage.

The selected provider revisions reproduce the historical observation multiset.
They are not a refresh to today's leaderboard. The train counts in the pinned
cards are independently reconciled against all downloaded shards by the tests.
The exact manifests and output counts are not duplicated here: source facts
live in YAML; reviewed row counts and full-table fingerprints live in
`characterization.yaml`; parsing and migration invariants remain in `test.py`.

The retained families are AIME (including the released combined and split
datasets), HMMT, BRUMO, CMIMC, SMT, APEX and its shortlist, ArXivMath, six
Kangaroo grade bands, and the USAMO, IMO, IMC, Putnam and Miklos proof releases.
Coverage is whatever per-model/per-problem attempts the pinned releases contain,
not a rectangular matrix or an assumption of four trials everywhere. Separate
datasets can overlap in problems and observations; we preserve their source
scope and do not treat the combined matrix as independent competition samples.

## Harness, tools, budgets, environment and graders

The provider's [runner documentation](https://github.com/eth-sri/matharena#readme)
describes conventional model runs and agent configurations. Released labels and
configuration paths distinguish plain models, agents, self-checking systems,
best-of-32 configurations, and explicit reasoning levels. The source prompt is
preserved from `user_message`; Kangaroo prompts use provider-specific multimodal
message encodings whose image payloads are recovered exactly. `problem` is the
fallback only if the released user message is absent. Neither reference answers
nor model outputs may substitute for a missing prompt.

The pinned outputs do not bind every run to immutable model-configuration file
contents or a harness commit. A path alone does not establish its historical
temperature, token/tool budget, system prompt, toolchain, sandbox, inference API,
access date, or software environment. These are not inferred from current config
files. The subject retains both the released name and `model_config`; explicit
parenthetical low/medium/high/xhigh/max levels populate `reasoning_effort`.
Unspecified effort stays null. Harness name/version and access date remain null
where no sufficiently specific run binding is released. Agent and best-of-N
labels remain distinct identities, not silently renamed to their base models.
The turn/modality metadata covers text and visual tasks and the released mix of
plain and agentic configurations; it does not assert that every observation
used tools or multiple turns.

Final-answer grades use the released `correct` boolean, not a new parser or
threshold. Proofs use human grades: the released proof instructions explicitly
describe human evaluation, and the official runner documentation identifies
the retained proof competitions as human-verified. Anonymized `judge_1` and
`judge_2` slots are retained within each instrument; they do not identify a
particular person across competitions. The item verifier preserves the exact
criterion title, grading scheme, maximum points and judge slot. Grader feedback
is observation-specific and stays on the response row. Missing/non-numeric
points or maxima, zero maxima, and non-finite values do not yield measurements.
The legacy clipping of finite fractions into [0, 1] is retained.

## Identity and processing decisions

- One subject per released model-name/configuration pair. Raw labels remain
  intact. The shared nullable `normalized_name` and `provider` fields permit
  honestly unmapped subjects; no model-registry entries were added or guessed.
  A future registry change changes subject IDs and needs separate review.
- An item includes the complete released textual prompt, competition/problem
  scope, and its grading instrument. Different prompt instructions are different
  stimuli. Shared NFC/edge-whitespace normalization still coalesces text-only
  whitespace variants, retaining the first released text. Competition scope is
  an explicit feature, because `raw_item_id` alone
  does not disambiguate content-based IDs. Ground-truth spelling variants and
  rubric variants also enter verifier identity; first-registration wins must not
  silently replace a record's reference answer or grading scheme.
- Kangaroo attachments retain exact decoded image bytes in the content-addressed
  assets table. Logical attachment paths and released image detail settings are
  part of item identity. We neither OCR nor redraw the images. The two released
  detail settings can create two items linking the same image bytes.
- Proof judge and criterion identity live in the item verifier under the current
  schema, not in `test_condition`. The current build schema accepts only
  `contract_version`; the former `setting_classes` field is not reintroduced.
- Trial is exactly the released zero-based `idx_answer` plus one. No inferred
  seeds, reassigned trial numbers, or post-registration duplicate repair are
  used. Explicit source/configuration/item identities make these keys unique.
  Test conditions and interactors remain null when not recorded.
- Each response retains its manifest shard key, zero-based shard row, competition,
  problem, original attempt index, and applicable judge/criterion positions.
  These fields make every observation auditable back to the pinned release.
- Traces retain the full released `answer`, or `parsed_answer` if no answer text
  is available. For a proof attempt, the solution trace is attached only to the
  first usable criterion in judge/criterion order, as in the old allocation
  rule. Other criteria can be associated through their common source pointer.
  The trace table uses the shared composite response key. Full conversation/tool
  logs remain in the pinned raw shards but are not substituted for solution text.

## Legacy defects and accepted differences

Before replacing the old implementation, its code, selected-column raw caches
and historical `.hf-cache/matharena` Parquet tables were inspected. The cache
held 448 registered items, 104 subjects and 89,567 responses. Its logical tables
are fingerprinted separately in the characterization record.

The old implementation could not run unchanged against the moved/current shared
runtime. A read-only replay of its transformation using the preserved historical
subject identities and pinned provider releases reproduced the old item IDs
and the complete observation multiset after excluding collision-dependent trial
numbers. It did not reproduce the exact old trial assignment: the old numbering
depended on row order and on incorrectly merged items. We do not label that
replay an exact-table rebuild.

The authorized corrections are:

- Recovering all image questions instead of collapsing Kangaroo into the five
  answer letters A–E; enforcing competition scope and preserving prompt variants.
- Decoding JSON-string rubric lists for USAMO, IMO and IMC, restoring 3,377
  criterion scores. All 89,482 final verdicts and the 85 already-retained native
  proof grades remain unchanged in value. There is no fresh grading.
- Retaining model configuration/effort distinctions and moving proof instruments
  to items as the current schema requires.
- Retaining full prompts and solution traces instead of the former 4,000/8,000
  character caps, and preserving per-record reference/rubric variants.
- Preserving original attempt indices instead of collision-driven renumbering.
  Consequently item IDs, subject IDs, response IDs, trial structure, metadata,
  coverage statistics and trace keys are intentionally not legacy-equivalent.

The corrected output contains 92,944 response rows, 90,005 attributed solution
traces, all 104 source configurations and 168 exact image payloads. The reviewed
snapshot, rather than these summary numbers, is the complete acceptance record.

## Gaps and exclusions

We do not synthesize omitted competition problems, missing attempts or absent
rubric criteria. No competition-level score is expanded into item responses.
The released IMO subset, for example, is not assumed to be the complete contest.
Criterion-level scores should not be equally averaged as if they were equal-weight
whole problems: maxima, judge coverage and rubric lengths vary.

Third-party reruns, leaderboard scrapes, model-vendor score announcements,
mirrors, reconstructed answers and later provider releases are deliberately
excluded. No new measurement source, third-party run, registry update or
Hugging Face upload is part of this migration. Unknown harness configuration
remains a documented provenance gap, not an inferred setting. Global analyses
that intentionally require normalized model identities may exclude the unmapped
subjects even though this benchmark retains their observations.

## Verification and publication history

Verification uses the current shared runtime in the sibling `measurement_db`
checkout. The only runtime edits are nullable model/provider schema fields and
the corresponding registration documentation; `build_base.py` is unchanged.
The metadata test now locates the central schema/template in that shared
checkout. Nullable-subject regressions check successful writes, response links,
and distinct fallback identities for two unknown labels.

The public Viewer feature declarations and their tests also include the new
response provenance/feedback columns. No publisher implementation was changed.

Final checks on the corrected build:

- `python -m unittest tests.test_benchmark_metadata -v`: 24 tests pass.
- `python benchmarks/matharena/build.py`: succeeds from all pinned artifacts.
  Repeated after accepting the snapshot; all six logical table fingerprints
  remain identical (1,755 items, 104 subjects, 92,944 responses).
- `MEASUREMENT_DB_FULL_TEST=1 python benchmarks/matharena/test.py -v`: all 11
  tests pass before and after the repeat build, including exhaustive source
  reconciliation and exact image-byte checks.
- The nullable-subject regression and measurement-table tests: 12 tests pass
  with the shared checkout on `PYTHONPATH`. Viewer-card tests: 17 tests pass.
- `python -m unittest discover -s tests -v`: 308 tests run, with nine unrelated
  failures. Supplying `PYTHONPATH=../measurement_db` reduces this to eight:
  two obsolete local-template checks, suites for `exploitbench_v8_r1`,
  `financebench`, `frontieror`, `swe_together`, and `swebench`, and the existing
  attribution-manifest freshness check. The extra failure without that path
  is EDIT-Bench's unchanged relocated-runtime import. The eight shared-path
  failures were also observed in the pre-migration suite run; no other
  benchmark's snapshot was refreshed.
- The requested `scripts/audit_benchmarks.py` no longer exists. Its current
  replacement, `scripts/validate_benchmarks/audit_benchmark_inventory.py`,
  succeeds with the shared checkout on `PYTHONPATH`. MathArena is `ok`, with
  valid assets and zero duplicate response keys. Audit output was directed to
  a temporary file, not a tracked artifact.
- `ruff check benchmarks/matharena tests`: MathArena is clean; 40 existing
  findings remain in repository tests. The MathArena and metadata-test-only
  lint check passes. These unrelated test files were not bulk-reformatted.
- `git diff --check`: passes in both worktrees. Raw files, images, generated
  Parquet tables and Python caches remain ignored. Nothing is staged or
  published; existing unrelated user changes were preserved.

MathArena's migration and local verification are complete. The repository-wide
definition of done remains blocked on the unrelated suite/lint failures above.

Validation expectations are now stored beside metadata in `characterization.yaml`,
using the shared schema and table hash algorithm. The previous source audits and
migration assertions passed before conversion; release-specific assertions remain
in `test.py`. Source claims distinguish reported quantities from those independently
derived from the archived release. This format change does not alter curated data.
