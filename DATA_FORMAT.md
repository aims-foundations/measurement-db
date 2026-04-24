# measurement-db data format

Long-form, registry-backed schema for storing evaluation data.

## Directory layout

```
measurement-db-private/
├── _registry/
│   ├── __init__.py         # Python helpers (resolve_subject, register_item, ...)
│   ├── subjects.parquet    # one row per distinct model ever seen
│   ├── benchmarks.parquet  # one row per benchmark
│   └── items.parquet       # one row per distinct item across all benchmarks
├── {dataset}/
│   ├── build.py            # ingests raw data, writes responses.parquet
│   ├── audit.py            # optional; gating criterion for public release
│   ├── raw/                # gitignored; reproducible via build.py
│   └── responses.parquet   # the canonical long-form table
├── manifest.yaml            # dataset → {status, domain}
├── sync_to_public.py        # manifest-gated sync
└── README.md
```

The `build.py` file should generate the long form. We should let go the generation of wide form and the heatmap. 

---

## Tables

### `_registry/subjects.parquet`

Registry of every model (AI test-taker) that has ever appeared in any benchmark.

| column | type | nullable | description |
|---|---|---|---|
| `subject_id` | string | no | Primary key. `sha256(normalized_raw_label)[:16]` as a first cut; migrates to `sha256(hub_repo + "@" + revision)[:16]` once revisions are curated. |
| `display_name` | string | no | Human-friendly label for plots, leaderboards. |
| `provider` | string | yes | `meta`, `openai`, `anthropic`, `google`, etc. |
| `hub_repo` | string | yes | HuggingFace repo id for open-weight models, e.g. `meta-llama/Llama-2-13b-chat-hf`. |
| `revision` | string | yes | Git SHA of the HF repo snapshot evaluated, or for API models a provider-emitted version string like `gpt-4-0613`. |
| `params` | string | yes | Parameter count, e.g. `7B`, `70B`, `unknown`. |
| `release_date` | date | yes | Provider's release date. |
| `raw_labels_seen` | list[string] | no | Audit trail of raw strings from source data that resolved to this subject. |
| `notes` | string | yes | Free-form — e.g. known quirks, license. |

**Nullability rationale:** `hub_repo` / `revision` / `release_date` require human curation (reading HF pages or provider docs). Nullable today so build.py isn't blocked; backfilled via a separate curation PR process. The `subject_id` stays stable across backfills because it's derived from the initial raw-label hash, not from `hub_repo`.

### `_registry/benchmarks.parquet`

| column | type | nullable | description |
|---|---|---|---|
| `benchmark_id` | string | no | Primary key, typically same as folder name (e.g. `mtbench`). |
| `name` | string | no | Display name (e.g. "MT-Bench"). |
| `version` | string | yes | Version string if the benchmark has one. |
| `license` | string | yes | SPDX identifier if known. |
| `source_url` | string | yes | Upstream repo or paper. |
| `description` | string | yes | One-line description. |
| `modality` | list[string] | no | Input modalities required to solve items: `"text"`, `"image"`, `"grid"`, `"gui_screenshot"`, `"audio"`. Defaults to `["text"]`. Multimodal benchmarks list multiple. |
| `domain` | list[string] | no | Subject areas: `"software_engineering"`, `"mathematics"`, `"science"`, `"medicine"`, `"law"`, `"finance"`, `"safety"`, `"preference"`, `"tool_use"`, `"gui_agent"`, `"cybersecurity"`, `"general"`, `"translation"`, `"summarization"`, `"ner"`, `"cultural"`, `"reasoning"`, `"reward_modeling"`, `"ml_engineering"`. Defaults to `["general"]`. Multi-domain benchmarks list multiple. |
| `response_type` | string | no | How the grader emits the response. Controlled vocabulary: `"binary"`, `"likert_5"`, `"likert_10"`, `"win_rate"`, `"ordinal"`, `"fraction"`, `"continuous_bounded"`, `"continuous_unbounded"`, `"error_presence"`, `"mixed"`. Defaults to `"binary"`. |
| `response_scale` | string | no | Free-form description of the response value set, e.g. `"{0, 1}"`, `"{1, 2, 3, 4, 5}"`, `"{0, 1/8, 2/8, ..., 1}"`, `"k/N fraction, N=#test_cases"`, `"[-18, 18] continuous"`. Defaults to `"{0, 1}"`. |
| `categorical` | bool | no | `True` if the response set is finitely enumerable in a small way (binary, likert, ordinal, win-rate, error-presence). `False` for truly continuous responses or fractions with variable denominator (reward scores, rubric-weighted sums, k/N with varying N). Downstream IRT can filter on this to pick the right model family (dichotomous/polytomous vs. continuous). Defaults to `True`. |

**Authoring note:** Each `build.py` declares `modality`, `domain`, `response_type`, `response_scale`, and `categorical` in its `INFO = {...}` dict and passes them through `get_benchmark_id(..., modality=INFO.get("modality"), domain=INFO.get("domain"), response_type=INFO.get("response_type"), response_scale=INFO.get("response_scale"), categorical=INFO.get("categorical"))`. Keeping these fields on `INFO` means all semantic metadata for a dataset lives in its own `build.py` — a single source of truth, and clone-and-modify workflows carry the metadata automatically.

### `_registry/items.parquet`

Pure item registry — one row per distinct prompt/question, benchmark-scoped.
Holds what the item *is*, not under what conditions it was evaluated.

| column | type | nullable | description |
|---|---|---|---|
| `item_id` | string | no | Primary key. `sha256(benchmark_id + "::" + normalized_content)[:16]`. |
| `benchmark_id` | string | no | Foreign key to `benchmarks`. |
| `raw_item_id` | string | no | Original ID in upstream data (for traceability). |
| `content` | string | yes | Prompt / question text. Null for benchmarks that don't expose per-item content. |
| `correct_answer` | string | yes | Ground truth, if one exists. Null for preference / judge benchmarks. |
| `content_hash` | string | yes | `sha256(normalized_content)[:16]` — makes cross-benchmark duplicate detection a simple equality query. |

### `{dataset}/responses.parquet`

The long-form data. One row per (subject, item, trial, test_condition). Lives
at the benchmark root, not inside `processed/`.

| column | type | nullable | description |
|---|---|---|---|
| `subject_id` | string | no | FK → `subjects.subject_id`. |
| `item_id` | string | no | FK → `items.item_id`. |
| `benchmark_id` | string | no | Denormalized FK → `benchmarks.benchmark_id`. Constant within a single responses file; included so cross-dataset unions are self-describing. |
| `trial` | int32 | no | 1-indexed. Use `1` for single-trial observations; increment for repeated measurements (reruns, additional annotators of the same (subject, item, condition) cell). |
| `test_condition` | string | yes | Observation-level condition, e.g. `few-shot=0`, `temperature=0.7`, `skill=coherence`, `variant=chosen`. Use when the *same item* (same content, same `item_id`) is scored under multiple conditions. Item-level variants (follow-up turns with different prompt text, different task framings) should instead produce distinct `item_id`s via distinct content — not be encoded as `test_condition`. |
| `response` | float64 | no | The scalar outcome. For binary tasks: 0/1. For scored tasks: the score. |
| `correct_answer` | string | yes | Denormalized from `items.correct_answer` — lets simple scoring queries skip the items join. Null if no ground truth exists. |
| `trace` | string | yes | Raw model output text (if available). Null when not collected. |
| `metadata` | struct | yes | Optional nested struct for per-response metadata (latency, tokens, etc.). |

**Primary key:** `(subject_id, item_id, trial, test_condition)`, with `test_condition` `NULL` treated as a single condition value. Every dataset must satisfy this uniqueness invariant.

Storage: Parquet with snappy compression. For large datasets with traces, split traces into `traces.parquet` with `(subject_id, item_id, trial, test_condition, trace)` so the main responses table stays small.

---

## ID derivation rules

```python
def subject_id(raw_label: str) -> str:
    return sha256(normalize(raw_label).encode()).hexdigest()[:16]

def item_id(benchmark_id: str, content: str) -> str:
    return sha256(f"{benchmark_id}::{normalize(content)}".encode()).hexdigest()[:16]

def content_hash(content: str) -> str:
    return sha256(normalize(content).encode()).hexdigest()[:16]

def normalize(s: str) -> str:
    # Strip whitespace, NFC-normalize, lowercase for subject labels;
    # preserve case for item content (case matters for prompts).
    ...
```

IDs are **deterministic from inputs** — rerunning build.py produces identical IDs for the same raw data. This is what makes the system robust to rebuilds.

**When raw labels are inconsistent** (e.g. two benchmarks call the same model `GPT-4` and `gpt-4-0613`): the subject gets registered twice with two different `subject_id`s. A curator later merges them by editing `raw_labels_seen` in the registry (moves the aliases under one subject and deletes the duplicate). Queries that need to treat them as one model rely on `hub_repo`/`revision` after the backfill.

---

## Build-time invariants

Each `build.py` MUST:

1. **Subjects are AI systems.** Every row's `subject_id` must resolve to a model, agent, judge, or classifier. Datasets whose "subjects" are prompts, samples, or human annotators do not fit this schema — they belong in the items registry only (or in `BENCHMARKS_PENDING` until a real AI-subject data source exists).
2. Call `resolve_subject(raw_label)` for every raw AI-system label. With `auto_register=True`, this creates a new subject entry if none matches; with `auto_register=False`, it raises `UnknownSubject`.
3. Call `register_item(benchmark_id, raw_item_id, content)` for every item. Idempotent — returns the same `item_id` on re-registration. Note: `register_item` does NOT take `test_condition` — conditions live on responses.
4. Call `get_benchmark_id(name, ...)` to register the benchmark once.
5. Write the final `responses.parquet` referencing only resolved `subject_id`, `item_id`, and `benchmark_id`.
6. The table must satisfy the primary-key invariant: every `(subject_id, item_id, trial, test_condition)` tuple appears at most once. If upstream has multiple observations of the same (subject, item) cell — repeat runs, multiple annotators — encode them as separate `trial` values (1, 2, 3, ...), not duplicate rows.

---

## Migration plan

- **Phase 1 (current):** cleaning the build.py for each dataset to generate the long-form `responses.parquet` and the registries, while start to deprecate the wide-form CSV and the heatmap PNG. The package is early enough that this won't impact downstrem consumers. 
- **Phase 2:** downstream consumers (torch_measure loaders, analytics) switched to read `responses.parquet`.

---

## Query patterns

**Single-model scores on one benchmark:**

```python
import duckdb
duckdb.sql("""
    SELECT AVG(r.response)
    FROM 'mtbench/responses.parquet' r
    JOIN '_registry/subjects.parquet' s USING (subject_id)
    WHERE s.display_name = 'Llama-2-70B-Chat'
""").df()
```

**Cross-benchmark leaderboard:**

```python
duckdb.sql("""
    SELECT s.display_name, r.benchmark, AVG(r.response) AS mean_score
    FROM '{mtbench,alpacaeval}/responses.parquet' r
    JOIN '_registry/subjects.parquet' s USING (subject_id)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
```

(DuckDB reads the glob directly and stamps `benchmark` from the path.)

---

## Benchmark inventory

Two auto-generated tables describing the benchmarks in `reproduce.py:BENCHMARKS`.
Regenerate in place with:

```bash
python scripts/audit_benchmarks.py --out /tmp/inventory.md
# then paste the contents of /tmp/inventory.md below this heading.
```

Columns in the **registry** table come from `_registry/benchmarks.parquet`
(populated by each build's `INFO` dict). Columns in the **inventory** table
come from the current `responses.parquet` on disk plus `_registry/items.parquet`:

- `rows` — number of observations
- `subjects` / `items` — distinct `subject_id` / `item_id` counts
- `PK-dup` — duplicate `(subject_id, item_id, trial, test_condition)` tuples; must be **0**
- `binary%` — fraction of responses in `{0.0, 1.0}`; rest are continuous
- `content%` — fraction of this benchmark's registered items (in `_registry/items.parquet`) with non-null `content`
- `range` — observed `[min, max]` of the response
- `max trial` — highest `trial` value seen (> 1 = repeated measurements / multi-annotator)
- `test_condition` — ✓ if any row uses a non-null test_condition

### Benchmark registry

Registry-level metadata per benchmark (from `_registry/benchmarks.parquet`). Every field is declared in the benchmark's own `build.py:INFO` dict and threaded through `get_benchmark_id()` at build time.

| benchmark_id | name | license | modality | domain | response_type | response_scale | categorical |
|---|---|---|---|---|---|---|:---:|
| afrieval | afrieval (MasakhaNER v1+v2) | Apache-2.0 | text | ner, multilingual | binary | {0, 1} | ✓ |
| afrimedqa | AfriMed-QA | CC-BY-NC-SA-4.0 | text | medicine, multilingual | binary | {0, 1} | ✓ |
| agentdojo | AgentDojo | MIT | text | tool_use, safety | binary | {0, 1} | ✓ |
| ai2d_test | AI2D_TEST | CC-BY-SA-4.0 | text, image | science | binary | {0, 1} | ✓ |
| androidworld | AndroidWorld | Apache-2.0 | text, gui_screenshot | gui_agent | binary | {0, 1} | ✓ |
| appworld | AppWorld | Apache-2.0 | text, gui_screenshot | gui_agent, tool_use | fraction | aggregate completion rate per (split, level, metric) | — |
| arena_hard | Arena-Hard-Auto | Apache-2.0 | text | preference | win_rate | {0, 0.125, 0.25, ..., 1} (8-level judge scale) | ✓ |
| bbq | BBQ | CC-BY-4.0 | text | safety | binary | {0, 1} | ✓ |
| bfcl | BFCL | Apache-2.0 | text | tool_use | binary | {0, 1} | ✓ |
| biggen | BiGGen-Bench | CC-BY-SA-4.0 | text | general | likert_5 | {-1, 1, 2, 3, 4, 5} (-1 = N/A) | ✓ |
| bridging_gap | Bridging-the-Gap African Languages | Apache-2.0 | text | multilingual, cultural | binary | {0, 1} | ✓ |
| chatgpt_drift | ChatGPT Drift (LLMDrift) | Apache-2.0 | text | general | binary | {0, 1} | ✓ |
| clinebench | ClineBench | unknown | text | software_engineering | fraction | hardcoded pass rates from READMEs | — |
| corebench | CORE-Bench | MIT | text | science | fraction | k/N per paper (questions correct / total) | — |
| cruxeval | CRUXEval | MIT | text | software_engineering | binary | {0, 1} | ✓ |
| cybench | Cybench | Apache-2.0 | text | cybersecurity | fraction | per-task k/N for subtask_fractional mode; binary for unguided/subtask_guided | — |
| editbench | EDIT-Bench | unknown | text | software_engineering | fraction | k/N from pytest (tests passed / total) | — |
| faithcot | FaithCoT-BENCH | nan | text | reasoning, safety | binary | {0, 1} | ✓ |
| financebench | FinanceBench | CC-BY-NC-4.0 | text | finance | binary | {0, 1} | ✓ |
| flask | FLASK | MIT | text | general | likert_5 | {-1, 1, 2, 3, 4, 5} (-1 = N/A) | ✓ |
| gaia | GAIA | CC-BY-4.0 | text, image | general | fraction | k/N HAL runs per task | — |
| hallusionbench | HallusionBench | BSD-3-Clause | text, image | safety | binary | {0, 1} | ✓ |
| helm_afr | HELM African MMLU + Winogrande | Apache-2.0 | text | multilingual | binary | {0, 1} | ✓ |
| helm_cleva | HELM CLEVA (Chinese) | Apache-2.0 | text | multilingual | binary | {0, 1} | ✓ |
| helm_thaiexam | HELM Thai Exam | Apache-2.0 | text | multilingual | binary | {0, 1} | ✓ |
| hle | Humanity's Last Exam | MIT | text | general, reasoning | binary | {0, 1} | ✓ |
| igakuqa | IgakuQA | MIT | text | medicine | binary | {0, 1} | ✓ |
| indeterminacy | Indeterminacy Experiments | CC-BY-4.0 | text | summarization | binary | {0, 1} | ✓ |
| jailbreakbench | JailbreakBench | MIT | text | safety | binary | {0, 1} | ✓ |
| judgebench | JudgeBench | MIT | text | reward_modeling, preference | binary | {0, 1} | ✓ |
| kmmlu | KMMLU | CC-BY-ND-4.0 | text | general, multilingual | binary | {0, 1} | ✓ |
| kormedmcqa | KorMedMCQA | CC-BY-4.0 | text | medicine, multilingual | binary | {0, 1} | ✓ |
| livebench | LiveBench | Apache-2.0 | text | general | fraction | k/N per category (questions correct / total) | — |
| livecodebench | LiveCodeBench | CC-BY-4.0 | text | software_engineering | binary | {0, 1} | ✓ |
| machiavelli | MACHIAVELLI | MIT | text | safety | continuous_unbounded | raw metric points per dimension, scale varies (e.g. power [-917, 1001]) | — |
| matharena | MathArena | unknown | text | mathematics | mixed | binary for AIME family (per-attempt correct); continuous fraction points/max for rubric comps (USAMO/IMO/IMC/Putnam/Miklos, per-criterion) | — |
| mathvista_mini | MathVista MINI | CC-BY-SA-4.0 | text, image | mathematics | binary | {0, 1} | ✓ |
| mlebench | MLE-bench | unknown | text | ml_engineering | ordinal | {0.5, 2, 3} complexity tiers | ✓ |
| mmbench_v11 | MMBench V1.1 | Apache-2.0 | text, image | general | binary | {0, 1} | ✓ |
| mme | MME | unknown | text, image | general | binary | {0, 1} | ✓ |
| mmlupro | MMLU-Pro | MIT | text | general, reasoning | binary | {0, 1} | ✓ |
| mmmu_dev_val | MMMU (dev+val) | Apache-2.0 | text, image | general | binary | {0, 1} | ✓ |
| mtbench | MT-Bench | CC-BY-4.0 | text | preference | likert_10 | {1, 2, ..., 10} | ✓ |
| osworld | OSWorld | Apache-2.0 | text, gui_screenshot | gui_agent | fraction | [0, 1] per-task rubric scorer output | — |
| paperbench | PaperBench | MIT | text | science | continuous_bounded | [0, 1] weighted rubric score (leaf binaries not published) | — |
| personalllm | PersonalLLM | MIT | text | preference | continuous_unbounded | [-18, 18] reward-model score | — |
| preference_dissection | Preference Dissection | unknown | text | preference | binary | {0, 1} | ✓ |
| prm800k | PRM800K | MIT | text | mathematics, reward_modeling | binary | {0, 1} | ✓ |
| rakuda | Rakuda | CC-BY-SA-4.0 | text | multilingual | ordinal | {0, 0.1, 0.2, ..., 0.9, 1} (judge score / 10) | ✓ |
| rewardbench | RewardBench | ODC-BY | text | reward_modeling | binary | {0, 0.5, 1} (0.5 = judge tie, <1% of rows) | ✓ |
| rewardbench2 | RewardBench 2 | ODC-BY | text | reward_modeling | binary | {0, 0.25, 0.33, 0.5, 1} (fractions = ties, ~2% of rows) | ✓ |
| sib200 | SIB-200 | CC-BY-SA-4.0 | text | multilingual | binary | {0, 1} | ✓ |
| summeval | SummEval | MIT | text | summarization | likert_5 | {1, 2, 3, 4, 5} | ✓ |
| swebench | SWE-bench Verified | MIT | text | software_engineering | binary | {0, 1} | ✓ |
| swebench_full | SWE-bench Full | MIT | text | software_engineering | binary | {0, 1} | ✓ |
| swebench_java | SWE-bench Java | Apache-2.0 | text | software_engineering | binary | {0, 1} | ✓ |
| swebench_multilingual | SWE-bench Multilingual | MIT | text | software_engineering, multilingual | binary | {0, 1} | ✓ |
| taubench | TAU-bench | MIT | text | tool_use | binary | {0, 1} | ✓ |
| tengu | Tengu-Bench | Apache-2.0 | text | multilingual | ordinal | {0, 0.1, 0.2, ..., 0.9, 1} (judge score / 10) | ✓ |
| terminal_bench | Terminal-Bench | Apache-2.0 | text | software_engineering | binary | {0, 1} | ✓ |
| theagentcompany | TheAgentCompany | MIT | text, gui_screenshot | tool_use | fraction | binary per checkpoint for 75% of checkpoints; weighted rubric for the rest | — |
| toolbench | ToolBench | Apache-2.0 | text | tool_use | mixed | binary for StableToolBench; continuous [0,1] for SambaNova paper Table 9 | — |
| tumlu | TUMLU | unknown | text | multilingual | binary | {0, 1} | ✓ |
| ultrafeedback | UltraFeedback | MIT | text | preference | likert_5 | {1, 2, 3, 4, 5} | ✓ |
| visualwebarena | VisualWebArena | MIT | text, gui_screenshot | gui_agent | binary | {0, 1} | ✓ |
| vl_rewardbench | VL-RewardBench | MIT | text, image | reward_modeling | binary | {0, 1} | ✓ |
| wildbench | WildBench | CC-BY-4.0 | text | preference | likert_10 | {1, 2, ..., 10} | ✓ |
| wmt_mqm | WMT MQM | Apache-2.0 | text | translation, multilingual | error_presence | {0, 1} per (category, severity) bucket | ✓ |
| workarena | WorkArena | Apache-2.0 | text, gui_screenshot | gui_agent | fraction | binary for AgentRewardBench, aggregate rates for leaderboard+paper sources | — |

### Benchmark inventory

_Snapshot: 69 datasets in BENCHMARKS • 66 ready • 3 empty (upstream gap) • 0 missing • 16,906,973 total response rows._

| dataset | rows | subjects | items | PK-dup | binary% | content% | range | max trial | test_condition | modality | domain |
|---|---:|---:|---:|---:|---:|---:|---|---:|:---:|---|---|
| afrieval | 219,289 | 12 | 32,518 | 0 | 100% | 100% | [0.00, 1.00] | 42 | — | text | ner, multilingual |
| afrimedqa | 110,930 | 30 | 6,910 | 0 | 100% | 100% | [0.00, 1.00] | 1 | ✓ | text | medicine, multilingual |
| agentdojo | 69,796 | 29 | 1,081 | 0 | 100% | 100% | [0.00, 1.00] | 14 | ✓ | text | tool_use, safety |
| ai2d_test | 770,916 | 254 | 3,088 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text, image | science |
| androidworld | 348 | 3 | 116 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text, gui_screenshot | gui_agent |
| appworld | 288 | 18 | 16 | 0 | 15% | 100% | [0.00, 1.00] | 1 | — | text, gui_screenshot | gui_agent, tool_use |
| arena_hard | 35,990 | 72 | 500 | 0 | 14% | 100% | [0.00, 1.00] | 1 | — | text | preference |
| bbq | 409,444 | 7 | 56,578 | 0 | 100% | 100% | [0.00, 1.00] | 5 | ✓ | text | safety |
| bfcl | 441,086 | 93 | 4,133 | 0 | 100% | 100% | [0.00, 1.00] | 14 | — | text | tool_use |
| biggen | 305,935 | 103 | 764 | 0 | 14% | 100% | [-1.00, 5.00] | 2 | ✓ | text | general |
| bridging_gap | 190,836 | 3 | 21,134 | 0 | 100% | 100% | [0.00, 1.00] | 6 | — | text | multilingual, cultural |
| chatgpt_drift | 0 | — | — | — | — | 100% | — | — | — | text | general |
| clinebench | 26 | 3 | 12 | 0 | 81% | 100% | [0.00, 1.00] | 1 | — | text | software_engineering |
| corebench | 1,956 | 15 | 270 | 0 | 93% | 100% | [0.00, 1.00] | 3 | — | text | science |
| cruxeval | 16,000 | 1 | 1,600 | 0 | 100% | 100% | [0.00, 1.00] | 10 | ✓ | text | software_engineering |
| cybench | 960 | 8 | 40 | 0 | 81% | 100% | [0.00, 1.00] | 1 | ✓ | text | cybersecurity |
| editbench | 23,328 | 44 | 533 | 0 | 70% | 100% | [0.00, 1.00] | 2 | — | text | software_engineering |
| faithcot | 2,519 | 4 | 340 | 0 | 100% | 100% | [0.00, 1.00] | 2 | ✓ | text | reasoning, safety |
| financebench | 2,400 | 16 | 150 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | finance |
| flask | 76,009 | 15 | 1,696 | 0 | 14% | 100% | [-1.00, 5.00] | 2 | ✓ | text | general |
| gaia | 18,060 | 3191 | 173 | 0 | 42% | 100% | [0.00, 1.00] | 2 | ✓ | text, image | general |
| hallusionbench | 0 | — | — | — | — | — | — | — | — | text, image | safety |
| helm_afr | 747,560 | 23 | 32,741 | 0 | 100% | 100% | [0.00, 1.00] | 6 | — | text | multilingual |
| helm_cleva | 23,312 | 4 | 5,822 | 0 | 100% | 100% | [0.00, 1.00] | 3 | — | text | multilingual |
| helm_thaiexam | 23,730 | 42 | 561 | 0 | 100% | 100% | [0.00, 1.00] | 2 | — | text | multilingual |
| hle | 13,339 | 19 | 1,792 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | general, reasoning |
| igakuqa | 7,355 | 5 | 1,471 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | medicine |
| indeterminacy | 65,012 | 9 | 200 | 0 | 100% | 100% | [0.00, 1.00] | 10 | ✓ | text | summarization |
| jailbreakbench | 1,800 | 4 | 100 | 0 | 100% | 100% | [0.00, 1.00] | 1 | ✓ | text | safety |
| judgebench | 2,897 | 19 | 350 | 0 | 100% | 100% | [0.00, 1.00] | 1 | ✓ | text | reward_modeling, preference |
| kmmlu | 875,750 | 25 | 35,015 | 0 | 100% | 100% | [0.00, 1.00] | 2 | ✓ | text | general, multilingual |
| kormedmcqa | 18,002 | 7 | 3,009 | 0 | 100% | 100% | [0.00, 1.00] | 1 | ✓ | text | medicine, multilingual |
| livebench | 60,372 | 166 | 494 | 0 | 75% | 80% | [0.00, 1.00] | 3 | ✓ | text | general |
| livecodebench | 326,530 | 72 | 1,055 | 0 | 100% | 100% | [0.00, 1.00] | 10 | ✓ | text | software_engineering |
| machiavelli | 9,274 | 12 | 30 | 0 | 4% | 100% | [-917.89, 1001.00] | 1 | ✓ | text | safety |
| matharena | 86,206 | 97 | 448 | 0 | 100% | 100% | [0.00, 1.00] | 216 | ✓ | text | mathematics |
| mathvista_mini | 263,000 | 263 | 874 | 0 | 100% | 100% | [0.00, 1.00] | 44 | — | text, image | mathematics |
| mlebench | 14,241 | 33 | 75 | 0 | 72% | 100% | [0.00, 3.00] | 45 | — | text | ml_engineering |
| mmbench_v11 | 1,180,617 | 251 | 3,579 | 0 | 100% | 100% | [0.00, 1.00] | 34 | ✓ | text, image | general |
| mme | 535,810 | 232 | 1,983 | 0 | 100% | 100% | [0.00, 1.00] | 43 | ✓ | text, image | general |
| mmlupro | 564,750 | 48 | 13,542 | 0 | 100% | 100% | [0.00, 1.00] | 2 | — | text | general, reasoning |
| mmmu_dev_val | 204,632 | 253 | 896 | 0 | 100% | 100% | [0.00, 1.00] | 4 | — | text, image | general |
| mtbench | 5,436 | 34 | 160 | 0 | 20% | 100% | [1.00, 10.00] | 1 | — | text | preference |
| osworld | 27,766 | 77 | 369 | 0 | 97% | 100% | [0.00, 1.00] | 1 | — | text, gui_screenshot | gui_agent |
| paperbench | 539 | 9 | 20 | 0 | 14% | 100% | [0.00, 0.68] | 3 | — | text | science |
| personalllm | 832,160 | 8 | 10,174 | 0 | 0% | 100% | [-18.00, 18.25] | 3 | ✓ | text | preference |
| preference_dissection | 167,680 | 32 | 4,890 | 0 | 100% | 100% | [0.00, 1.00] | 12 | — | text | preference |
| prm800k | 561,715 | 11 | 11,268 | 0 | 100% | 100% | [0.00, 1.00] | 442 | ✓ | text | mathematics, reward_modeling |
| rakuda | 53,488 | 551 | 40 | 0 | 19% | 100% | [0.00, 1.00] | 1 | ✓ | text | multilingual |
| rewardbench | 450,735 | 118 | 2,733 | 0 | 99% | 100% | [0.00, 1.00] | 4 | ✓ | text | reward_modeling |
| rewardbench2 | 349,192 | 188 | 1,824 | 0 | 98% | 100% | [0.00, 1.00] | 2 | ✓ | text | reward_modeling |
| sib200 | 52,836 | 2 | 31,640 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | multilingual |
| summeval | 51,200 | 16 | 100 | 0 | 2% | 100% | [1.00, 5.00] | 5 | ✓ | text | summarization |
| swebench | 67,000 | 134 | 500 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | software_engineering |
| swebench_full | 55,056 | 24 | 2,275 | 0 | 100% | 100% | [0.00, 1.00] | 2 | — | text | software_engineering |
| swebench_java | 5,464 | 54 | 170 | 0 | 100% | 100% | [0.00, 1.00] | 1 | — | text | software_engineering |
| swebench_multilingual | 74,694 | 94 | 2,414 | 0 | 100% | 100% | [0.00, 1.00] | 2 | — | text | software_engineering, multilingual |
| taubench | 12,812 | 12 | 214 | 0 | 100% | 100% | [0.00, 1.00] | 392 | — | text | tool_use |
| tengu | 180,837 | 551 | 120 | 0 | 45% | 100% | [0.00, 1.00] | 1 | ✓ | text | multilingual |
| terminal_bench | 68,797 | 148 | 89 | 0 | 100% | 100% | [0.00, 1.00] | 11 | — | text | software_engineering |
| theagentcompany | 9,700 | 19 | 554 | 0 | 99% | 100% | [0.00, 1.00] | 1 | — | text, gui_screenshot | tool_use |
| toolbench | 7,924 | 39 | 774 | 0 | 97% | 100% | [0.00, 1.00] | 2 | ✓ | text | tool_use |
| tumlu | 143,316 | 16 | 7,486 | 0 | 100% | 100% | [0.00, 1.00] | 1 | ✓ | text | multilingual |
| ultrafeedback | 1,009,730 | 17 | 63,932 | 0 | 10% | 100% | [1.00, 5.00] | 2 | ✓ | text | preference |
| visualwebarena | 600 | 3 | 98 | 0 | 100% | 100% | [0.00, 1.00] | 2 | ✓ | text, gui_screenshot | gui_agent |
| vl_rewardbench | 0 | — | — | — | — | 100% | — | — | — | text, image | reward_modeling |
| wildbench | 113,566 | 71 | 1,024 | 0 | 1% | 100% | [1.00, 10.00] | 1 | ✓ | text | preference |
| wmt_mqm | 4,883,886 | 80 | 9,124 | 0 | 100% | 100% | [0.00, 1.00] | 33 | ✓ | text | translation, multilingual |
| workarena | 539 | 22 | 129 | 0 | 90% | 100% | [0.00, 1.00] | 1 | ✓ | text, gui_screenshot | gui_agent |
