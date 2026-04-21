# measurement-db data format

Long-form, registry-backed schema for storing evaluation data (models × items × responses).

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
│   ├── processed/
│   │   ├── responses.parquet        # the canonical long-form table
│   │   ├── response_matrix.csv      # wide form, regenerated for convenience
│   │   └── response_matrix*.png     # heatmaps
│   └── (intermediate artifacts)
├── manifest.yaml            # dataset → {status, domain}
├── sync_to_public.py        # manifest-gated sync
└── README.md
```

Wide-form CSVs and heatmap PNGs are **secondary artifacts** regenerated from `responses.parquet` during `build.py`. The long-form table is the source of truth.

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

### `_registry/items.parquet`

| column | type | nullable | description |
|---|---|---|---|
| `item_id` | string | no | Primary key. `sha256(benchmark_id + "::" + normalized_content)[:16]`. |
| `benchmark_id` | string | no | Foreign key to `benchmarks`. |
| `raw_item_id` | string | no | Original ID in upstream data (for traceability). |
| `content` | string | yes | Prompt / question text. Null for benchmarks that don't expose per-item content. |
| `correct_answer` | string | yes | Ground truth, if one exists. Null for preference / judge benchmarks. |
| `test_condition` | string | yes | e.g. `turn=1`, `few-shot=0`, `temperature=0.7`. Use when a single raw item appears under multiple conditions. |
| `content_hash` | string | yes | `sha256(normalized_content)[:16]` — makes cross-benchmark duplicate detection a simple equality query. |

### `{dataset}/processed/responses.parquet`

The long-form data. M×N rows per dataset (M subjects × N items × k trials).

| column | type | nullable | description |
|---|---|---|---|
| `subject_id` | string | no | FK → `subjects.subject_id`. |
| `item_id` | string | no | FK → `items.item_id`. |
| `trial` | int32 | no | 1-indexed. Use `1` for single-trial benchmarks. |
| `response` | float64 | no | The scalar outcome. For binary tasks: 0/1. For scored tasks: the score. |
| `trace` | string | yes | Raw model output text (if available). Null when not collected. |
| `metadata` | struct | yes | Optional nested struct for per-response metadata (latency, tokens, etc.). |

Storage: Parquet with snappy compression. For large datasets with traces, split traces into `traces.parquet` with `(subject_id, item_id, trial, trace)` so the main responses table stays small.

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

1. Call `resolve_subject(raw_label)` for every raw model label. With `auto_register=True`, this creates a new subject entry if none matches; with `auto_register=False`, it raises `UnknownSubject`.
2. Call `register_item(benchmark_id, raw_item_id, content)` for every item. Idempotent — returns the same `item_id` on re-registration.
3. Call `get_benchmark_id(name, ...)` to register the benchmark once.
4. Write the final `responses.parquet` referencing only resolved `subject_id` and `item_id` values.
5. Also regenerate `response_matrix.csv` and `response_matrix.png` from the long form, as secondary artifacts.

---

## Migration plan

- **Phase 1 (current):** long-form alongside wide-form. Both produced by `build.py`. Tools read from whichever they prefer.
- **Phase 2:** downstream consumers (torch_measure loaders, analytics) switched to read `responses.parquet`.
- **Phase 3:** wide-form CSV dropped; PNGs remain as visualization-only artifacts.

---

## Query patterns

**Single-model scores on one benchmark:**

```python
import duckdb
duckdb.sql("""
    SELECT AVG(r.response)
    FROM 'mtbench/processed/responses.parquet' r
    JOIN '_registry/subjects.parquet' s USING (subject_id)
    WHERE s.display_name = 'Llama-2-70B-Chat'
""").df()
```

**Cross-benchmark leaderboard:**

```python
duckdb.sql("""
    SELECT s.display_name, r.benchmark, AVG(r.response) AS mean_score
    FROM '{mtbench,alpacaeval,aegis}/processed/responses.parquet' r
    JOIN '_registry/subjects.parquet' s USING (subject_id)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
```

(DuckDB reads the glob directly and stamps `benchmark` from the path.)
