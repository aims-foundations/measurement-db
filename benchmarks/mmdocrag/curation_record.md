# MMDocRAG curation record

## Measurement and qualification

MMDocRAG measures document question answering with retrieved text and image
evidence: producing an answer that is fluent, factual, logically supported,
coherent across text and images, and appropriately cited. This curation uses
the provider's released per-question answer-quality judgments. It does not
re-run inference, re-grade answers, or curate the separate retrieval metrics.

The provider describes a larger QA bank and broader experimental coverage.
The curated slice is the released evaluation questions and judge files, not
every training example, retriever, or headline model count in the paper.
It qualifies as item-level measurement data because each released judgment
identifies a question and the filename identifies the evaluated system and
condition. The record's `model` field identifies the **judge**, not the subject.

The accepted output has 2,000 items, 68 subjects, 342,695 attempt rows, and
328,286 answer traces. Of those attempts, 342,566 have usable grades and 129
are retained with null grades. Exact counts, coverage by condition, categories,
Reviewed counts and full-table fingerprints are in
[characterization.yaml](characterization.yaml); detailed identity, null-pattern,
and migration assertions remain in `test.py`.

## Provider-owned provenance

[metadata.yaml](metadata.yaml) is authoritative for bibliographic metadata,
license, release configuration, provider URLs, source revision, local filenames,
byte sizes, and SHA-256 pins. None of those pins is defined in Python.

The sources are the benchmark authors' own GitHub release: two gold question
files, 173 per-model evaluation files, and 191 raw-answer files. The migration
pinned the existing cached release to its matching provider commit; it did not
replace the data. All 366 files were checked against that commit's Git blob
identities, and the builder verifies their declared byte sizes and SHA-256
hashes. Acquisition is now a fixed manifest instead of discovery from a moving
branch. Unlisted local files cannot add measurements to a build.

The `sources.references` entries give pinned provider-owned evidence:

- `README.md`: benchmark scope, released data/results, usage, and the distinct
  code/data license notices.
- `inference_api.py`, `inference_checkpoint.py`, and `inference_wrapper.py`:
  public inference entry points, model wrappers, and prompt construction.
- `eval_llm_judge.py`, `inference_wrapper.py`, and the evaluation-answer prompt:
  judge invocation, submitted fields, JSON parsing, and grading rubric.
- `eval_all.py`: provider aggregate computation, which is not identical to
  the legacy curator's handling of malformed or noisy judge keys.
- The pure-text and multimodal inference prompts: instructions used by the
  reference harness to combine retrieved evidence with the question.

The provider also links hosted question/image data from its README. Those
assets are not new migration inputs; adding them to items would change item
content and identity and requires a separate review.

## Harness, tools, budgets, environment, grader, and metric

The released conditions are pure-text or multimodal input with either of the
two quote-count settings. They remain response-level `test_condition` values,
not additional subjects or duplicated items. Every released cell has trial 1;
missing cells are not synthesized. Interactors remain null.

The reference harness supports API inference and local checkpoint inference;
the latter uses Swift with PyTorch or vLLM paths depending on the mode. The
README documents a Python 3.9 environment and ms-swift. These are evidence
about the public harness, not a complete versioned environment for every
released cell. The release does not attach a per-cell execution manifest
establishing all dependency versions, hardware, endpoints, seeds, token/time
budgets, or sampling parameters. Wrapper defaults are not promoted into
historical run facts.

Labels such as `no-think` and fine-tuning suffixes remain part of the released
subject labels. Reasoning-effort, harness, and access-date fields remain null.
The subsequent approved registry curation preserves the literal `no-think`
variant as a subject feature, without claiming a numeric effort, token budget,
or verified server-side behavior. See the registry-completion section below.

The released judge labels are checked by the regression assertions in `test.py`. The public
judge wrapper defaults to the dated GPT-4o version recorded in the release,
combines the question and gold answers with the submitted answer, and requests
a JSON object. Successful JSON decoding does not guarantee that the object
contains all required dimensions. This explains why released attempts can
have unusable grades even though answer text exists.

The five dimensions are Fluency, Citation Quality, Text-Image Coherence,
Reasoning Logic, and Factuality, each scored from 0 through 5. The curator keeps
the legacy arithmetic: sum the first valid value for each dimension, in
released JSON order, divide by five dimensions, then divide by the scale
maximum. It does not change floating-point rounding or apply a new threshold.
Zero is a genuine observed grade, not a substitute for missing information.

Dimension keys are lowercased and stripped of non-alphanumeric characters.
This retains grades with padded, quoted, or de-spaced keys. Booleans,
nonnumeric values, and out-of-range numbers do not supply a dimension. A later
valid duplicate is usable only if no earlier valid value supplied that
dimension. All five dimensions are required; a partial average is not used.

## Processing and identity decisions

Gold records prefer the 20-quote source, then fill missing question IDs from
the 15-quote source; first occurrence wins. Short-answer lists are joined with
comma-space separators. The released question ID produces `q_id::<id>` as the
raw item key. Item content remains the question alone and the item reference
remains the short gold answer. All legacy item columns and item IDs compare
exactly with the migrated tables.

Evaluation filenames, including the older double-extension variants, identify
the subject, mode, and quote count. Records without a question ID or without a
dictionary judge payload do not become rows. Empty dictionaries do become
ungraded attempts. The pinned release is validated without adding fallback
questions or silently accepting unexpected question IDs.

The old subject helper merged case-only aliases and could fill missing model
metadata from a later spelling. The current helper finalizes metadata before
deriving the subject ID. The migration therefore preserves all five existing
case-only alias groups and resolves the two initially unmapped InternVL
spellings through the existing lowercase registry aliases before registration.
The declared alias mappings live in `archive_layout.subject_aliases`. Their
display names change only in capitalization; their existing canonical model,
provider, and release date are preserved. Trace lookup still uses the original
evaluation filename's spelling, not the registry alias.

Approved structural-migration differences, before registry completion, were:

- All subject IDs use the current canonical-name/row-metadata hash instead of
  the historical cleaned-label hash. The 68-subject grouping is unchanged;
  the snapshot records a one-to-one old/new identity crosswalk.
- The two InternVL display-name capitalization changes described above.
- Existing current-registry entries enrich `gpt-o3-mini` and `qwen3-14b`, whose
  canonical name/provider/release date were null in the legacy tables. No
  registry file was changed. The other 66 subjects retain their model metadata.
- The current schema adds derived response IDs, nullable interactors and item
  feature/asset fields, and explicit nullable subject configuration fields;
  it removes retired subject and benchmark columns. No absent configuration
  is inferred. Responses and traces are identical after the subject crosswalk.
- Benchmark license and publication configuration honor the user's existing
  pre-migration Python edits, rather than the stale values in the saved
  Parquet table. Their values now live only in metadata.yaml. This is not an
  upload authorization and no publication was performed.

For traces, the first existing **exact-case** filename wins: the
`<model>_<mode>_quotes<N>_response.jsonl` form precedes
`<model>_<mode>_response_quotes<N>.jsonl`. Within that file the last duplicate
question ID wins. Nonempty strings, including whitespace-only strings, are
preserved verbatim. No filename case folding, OCR substitution, answer cleanup,
or matching through the judge's model label is introduced.

## Coverage gaps and pre-existing issues left unchanged

The 129 ungraded attempts comprise 112 empty dictionaries, eight nonempty
payloads without usable dimensions, and nine partially usable payloads.
The old build docstring incorrectly described all of the first 120 as empty
dictionaries; this record corrects that explanation, not the scoring behavior.
Of the ungraded attempts, 120 have released nonempty answer traces and nine
do not. Null attempts are retained, not dropped, zero-filled, or re-judged.

Released coverage is not a complete subject-by-question-by-condition grid.
Traces are also incomplete. In particular, some answer filenames differ in
case from their evaluation filenames; the legacy exact-case matching leaves
those traces absent. Fixing that would change released trace attribution and
is deliberately outside this migration. Extra raw-answer files, including OCR
variants, do not create additional response cells.

Items do not include the retrieved text, page images, complete assembled
inference prompt, or an executable verifier. These are limitations of the
legacy representation, not evidence that the benchmark was text-only or
ungraded. Adding attachments, contexts, or a verifier would change item
identity and needs separate curation approval.

The historical `testing_condition` description is preserved in YAML. Its
simple association between inference mode and judge vision mode should not
be treated as an independently verified per-cell harness setting: the
released judge labels are the evidence retained by source reconciliation.
Likewise, the registry's model release dates are consumed as registry facts,
not independently corrected in this migration.

The provider's aggregate script uses literal dimension names and zero defaults.
The legacy curator already normalized key noise and retained incomplete
judgments as null; reproducing provider aggregate headlines would therefore
require a different measurement policy. This migration intentionally keeps
the accepted curator policy.

Inspection confirms that normalization affects 443 fully graded outcomes
relative to literal-key zero filling, increasing those outcomes by about 0.33
on average on the normalized scale. This was already true of the legacy
output; it is not a migration score change.

## Exclusions

No third-party reruns, leaderboard reconstructions, newly generated answers,
or inferred results are included. Only the provider release in the pinned
manifest supplies measurement rows. No expanded survey of independent reruns
is claimed. Training examples, image archives, retriever evaluations, and
unpaired answer files do not expand the measured slice.

## Registry completion after the structural migration

The user subsequently approved registering the 41 remaining released names,
preserving distinct evaluated variants, and reviewing the resulting identity
changes. This is a separate enrichment of the verified migration baseline.

Exactly 41 keys were added to the sibling runtime's
`scripts/build_measurement_tables/map_model_registry.json`. No existing entry
was edited, renamed, or deleted. Thirty-five additions reuse the complete
metadata of existing canonical models, including the existing release-date
evidence. Six additions introduce distinct canonical models: QwQ Plus and the
five provider-released MMDocRAG LoRA checkpoints.

The pinned provider README's `model_dict`, API model list, and checkpoint list
(already linked through `sources.references` in metadata.yaml) reconcile the
released abbreviations with the model families and sizes. Important decisions:

| Released label(s) | Registration decision and evidence |
| --- | --- |
| `gemini-2.0-flash-tk` | Use the existing **Google Gemini 2.0 Flash Thinking** identity, not ordinary Flash. The README explicitly expands `tk` to Flash-Think and the API list identifies the thinking endpoint. |
| `llama4-mave-17b-128e` | Use existing **Meta Llama 4 Maverick** metadata; the API list spells out the Maverick 17B/128E endpoint. Scout remains distinct. |
| `minicpm-o-2.6-8b` | Use existing **OpenBMB MiniCPM-o 2.6**, not the separately registered MiniCPM-V line. |
| `qwen-qwq-plus` | Register **Alibaba QwQ Plus**, not QwQ Max or the open-weight QwQ 32B. The provider API list identifies `qwq-plus`; Alibaba's [model documentation](https://www.alibabacloud.com/help/en/model-studio/qwq-plus) and [release history](https://www.alibabacloud.com/help/en/model-studio/newly-released-models) confirm the product and its release date. |
| `qwen2.5-{3,7,14,32,72}b-ft` | Register five distinct **MMDocRAG Qwen2.5 …B LoRA** models. The provider checkpoint list explicitly identifies these as its own fine-tunes, not aliases of the unchanged base checkpoints. |
| The five `*-no-think` labels | Reuse the underlying Qwen3 or QVQ-Max canonical metadata, and preserve the released variant before subject registration. Do not create fictitious new base models or infer a reasoning-effort value. |

The remaining mappings preserve the provider's explicit size, tier, version,
and VL/non-VL distinctions. They cover the two DeepSeek distillations, Gemini
Pro, Grok 3, InternVL sizes/versions, Janus-Pro, Llama sizes/versions, base
Qwen2.5 sizes, Qwen2.5-VL sizes, Qwen-VL Plus/Max, and QVQ-Max. Existing canonical
metadata was copied consistently rather than independently re-dating aliases.

The LoRA records use `Community`, following the registry convention for
benchmark-specific derivatives; their canonical names identify MMDocRAG as the
derivative, rather than implying that Alibaba released the adapted weights.
The provider-hosted adapter configuration files identify their corresponding
Qwen2.5-Instruct bases and `LORA` adaptation. Their release dates are based on
the provider-hosted upload histories (all five uploaded on the same day), not
the earlier base-model release date:

- [3B checkpoint history](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-3B-Instruct_lora/commits/main)
- [7B checkpoint history](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-7B-Instruct_lora/commits/main)
- [14B checkpoint history](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-14B-Instruct_lora/commits/main)
- [32B checkpoint history](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-32B-Instruct_lora/commits/main)
- [72B checkpoint history](https://huggingface.co/MMDocIR/MMDocRAG_Qwen2.5-72B-Instruct_lora/commits/main)

Only documentation and small adapter metadata were inspected; no checkpoint
weights or new benchmark measurement sources were downloaded or ingested.

For `no-think`, `archive_layout.subject_features` stores the literal released
variant, materialized as `released_variant=no-think` in
`subject_features_extra`. This is deliberately weaker than asserting a
server-side thinking configuration: the public wrapper supports an
`enable_thinking` argument, but the released files are not complete execution
manifests. In particular, QVQ support cannot be inferred from a Qwen3 flag.
Unsuffixed runs retain null features; no default thinking behavior is guessed.

Identity preflight found 68 distinct final subjects with these decisions. This
matters because the fine-tuned/base 7B pair has 2,386 overlapping attempt keys,
and the Qwen3 14B/no-think pair has 3,998. Those are separate released
measurements, not duplicate rows to drop or convert into additional trials.

Exact before/after comparison established:

- All 27 previously mapped subject rows retain their values and IDs.
- The 41 newly mapped subjects acquire canonical metadata and new IDs; only
  the five explicit no-think variants acquire a subject feature.
- Item and benchmark tables are exactly unchanged. All 342,695 response values,
  their 129 null scores, trials, conditions, reference answers, and all 328,286
  traces are unchanged after the one-to-one subject-ID crosswalk.
- Exactly 186,849 dependent response IDs change; measurement null patterns
  and the legacy-anchored response/trace fingerprints remain unchanged.

The characterization was updated only for the reviewed registry metadata,
subject feature, and ID differences. `registry_enrichment` retains the preceding
subject IDs and table fingerprints so this enrichment does not erase the
earlier accepted migration baseline.

A read-only scan of the other cached subject tables found one affected
benchmark: MathArena also uses the exact `gemini-2.0-pro` label. Its existing
outputs were not rebuilt or rewritten. Its next rebuild will need to review
the corresponding model metadata/ID enrichment. Benchmarks without cached
subject tables were not covered by this scan.

Separate existing-registry follow-up, not changed here: Alibaba's deployment
history dates QVQ-Max earlier than the launch-blog date already used by its
canonical registry entry. The new aliases inherit that existing entry
consistently; resolving the historical date convention is not silently bundled
into adding aliases.

## Verification and publication history

Before implementation changes, the original builder and all five generated
tables were saved outside the repository. The old entry point could not run
against the relocated current shared runtime (`build_base` import failure).
The original transformation was nevertheless replayed unmodified against the
cached release with a legacy-registration adapter, checking item registrations
against the saved item bank. Its response and trace tables matched the saved
tables exactly. The baseline was inspected rather than guessed.

The migrated build verifies the same provider bytes through immutable source
pins. Exact DataFrame comparison establishes unchanged legacy item fields and
IDs, plus unchanged response and trace fields after the approved subject
crosswalk. The initial subject comparison permitted only the two registry
enrichments described under structural migration; the subsequent 41-label
registry completion was compared separately as documented above.
Characterization expectations were recorded after comparison, not accepted
simply because the new builder ran.

The shared runtime must include the approved nullable-response contract change:
an explicit `None` is a released but ungraded attempt, while invalid values
such as booleans, strings, or floating-point NaN are still rejected at the
authoring boundary. The schema permits the resulting null score, and contract
tests cover trace retention and stable response IDs for both all-null and
mixed scored/ungraded builds. No shared registration helper was modified for
alias resolution or the subsequent registry completion.

The current central metadata schema accepts `build.contract_version` but no
longer accepts the retired `setting_classes` field. The YAML follows that
current contract rather than restoring a field from an older template.

Initial structural-migration verification on 2026-09-12:

- `python -m unittest tests.test_benchmark_metadata -v`: passed, 24 tests.
- `python benchmarks/mmdocrag/build.py`: passed with the pinned cached sources;
  all source byte sizes and hashes verified. Unmapped model labels are reported
  as advisory warnings and retain nullable registry fields.
- `MEASUREMENT_DB_FULL_TEST=1 python benchmarks/mmdocrag/test.py -v`: passed,
  10 tests, including independent reconciliation of every released judge cell
  and selected trace, and legacy-anchored fingerprints.
- The nullable-response regression tests and `tests.test_hash_measurement_ids`
  passed, 11 tests, with `PYTHONPATH=../measurement_db` for the sibling runtime.
- `python -m unittest discover -s tests -v`: ran 309 tests, with nine failures
  outside MMDocRAG. Two contract tests still look for runtime/template files
  at their former private-repository paths. EDIT-Bench and SWE-bench local
  tests have obsolete runtime imports. ExploitBench and FinanceBench lack the
  `fitz` dependency. FrontierOR and SWE-Together have stale generated tables
  relative to their characterization. The attribution manifest is stale for
  the concurrent MathArena migration; MMDocRAG's own metadata reference was
  updated. These unrelated migrations and environment repairs were not folded
  into this change.
- The requested `scripts/audit_benchmarks.py` path no longer exists. Its current
  equivalent, `PYTHONPATH=../measurement_db python
  scripts/validate_benchmarks/audit_benchmark_inventory.py`, passed: MMDocRAG is
  `ok` with zero duplicate keys. The inventory has no invalid datasets with
  the relocated runtime on the import path; unbuilt datasets remain missing.
- `ruff check benchmarks/mmdocrag`: passed. The requested broader
  `ruff check benchmarks/mmdocrag tests` reports 40 existing test-file findings.
  Removing this migration's nullable-response test in memory reproduces the
  same 12 existing findings in that contract-test file; the new test adds none.
- `git diff --check`: passed in both repositories. Generated data and caches
  remain ignored, and unrelated worktree changes were preserved.

Registry-completion verification on the same date:

- Rebuild passed without any unmapped-subject warnings; all 68 subjects now
  have registry metadata, while the fine-tuned and no-think subjects remain
  distinct.
- Full-test mode passed all 12 MMDocRAG tests, including the new registry-row
  concordance, subject-ID derivation, and variant non-merging checks.
- Central metadata validation again passed all 24 tests. The current inventory
  audit, benchmark-local Ruff check, and both repositories' diff checks passed.
- The full repository suite again ran 309 tests with the same nine unrelated
  failures listed above. The broader Ruff command still reports the same 40
  existing findings in `tests/`.

The benchmark-local migration is verified; the repository-wide green-test and
green-lint requirements remain outstanding because of the issues above.
No raw data, generated Parquet, caches, credentials, or downloaded archives
belong in the migration diff. No Hugging Face upload or commit was performed.

Validation expectations are now stored beside metadata in `characterization.yaml`,
using the shared schema and table hash algorithm. The previous source audits and
migration assertions passed before conversion; release-specific assertions remain
in `test.py`. Source claims distinguish reported quantities from those independently
derived from the archived release. This format change does not alter curated data.
