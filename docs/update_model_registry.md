# Update the model registry and statistics

`artifacts/render_website/model_statistics.csv` ranks the underlying models in
the measurement-db gallery by how many benchmark items each has been asked.
Because every benchmark names its own subjects differently (for example,
`gpt-4o-2024-08-06`, `openai/gpt-4o`, and `chatgpt` may all mean GPT-4o), the
numbers only make sense after each subject string is mapped to one canonical
model. That mapping lives in
`scripts/build_measurement_tables/map_model_registry.json`.

Here's what you should know when new benchmarks are added to the gallery: The item-counting is mechanical; the only part needing judgment is mapping new, never-seen subject strings to a model.

## Files

| File | Role |
|------|------|
| `scripts/build_measurement_tables/map_model_registry.json` | Durable crosswalk from raw subject strings to canonical model metadata. |
| `artifacts/render_website/model_statistics.csv` | Generated model-statistics report. |
| `scripts/render_website/.hf-cache/<slug>/…` | Per-benchmark Parquets downloaded by the gallery generator. |
| `website/content/generated/benchmark-cards.json` | Published, site-visible benchmark list. |

## Steps

### 1. Download the new benchmarks' data
```bash
cd <repo-root>
python3 scripts/render_website/generate_benchmark_gallery.py all
```

### 2. Run the rebuild script (below)
It regenerates the CSV and prints any subject strings not yet in the registry.
```bash
python3 scripts/render_website/generate_model_statistics.py
```
- If it prints `✓ every subject resolved` → the CSV is done. Stop here.
- If it prints a `⚠ N subject string(s) not in the registry` list → go to step 3. (Those subjects were still written to the CSV as their own rows with `company=Unknown` — placeholders until you map them.)

### 3. Map each flagged subject string

For every flagged string, decide its canonical `model` and `company`, then add
one entry to `scripts/build_measurement_tables/map_model_registry.json`:
```json
"the-exact-new-subject-string": { "model": "OpenAI GPT-4o", "company": "OpenAI" }
```
The key must be the subject string byte-for-byte as printed. Then re-run step 2; the flagged strings now fold into the right model rows.

## Mapping rules (how to fill in `model` + `company`)

Granularity is **family-level** — merge trivial variants, keep real differences.

- **Merge into one model:** date snapshots and serving/quant/effort suffixes.
  `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `openai/gpt-4o`, `gpt-4o (high)`,
  `-turbo`, `-instruct`, `-chat`, `-hf`, `-fc`, `:thinking` → **OpenAI GPT-4o**.
- **Keep distinct:** parameter sizes (`8B` vs `70B` vs `405B`), tiers (`gpt-4o` vs
  `gpt-4o-mini`; Gemini `pro` vs `flash`), major/minor versions (Claude `3` vs `3.5`;
  GPT-`4` vs `4.1`; Qwen`2` vs `2.5`), and variant lines (base vs `Coder`/`Math`/`VL`).
- **Agent scaffolds** — SWE-bench-style `20241022_tools_claude-3-5-sonnet`,
  `agentless+deepseek-r1`, `openhands+gpt-4o`: map to the **embedded base model**.
  If no base model is identifiable (`nfactorial`, `factory_code_droid`), use
  `"model": "Agent: <name>"`, `"company": "(agent scaffold)"`.
- **Not a real / unpublished / unidentifiable model** — baselines (`random`), run or
  dataset labels (`long_context-IGG-Baseline`, `stanford-online-all-v4-s3`),
  underspecified names (`qwen_base`): **keep the string as-is** as its own `model`,
  `"company": "Unknown"`. Do **not** guess.
- **Company** = the vendor/lab (`OpenAI`, `Anthropic`, `Google`, `Meta`, `Alibaba`,
  `Mistral`, `DeepSeek`…). Match the spelling already used in the registry for that
  vendor. Use `"Unknown"` if you can't identify it.

**Golden rule: when unsure, do not guess — keep the string as its own row
(`company: Unknown`).** A wrong merge silently corrupts a real model's totals.

## CSV schema

Sorted by `total_items_asked`, descending.

| Column | Meaning |
|--------|---------|
| `rank` | position by `total_items_asked` |
| `company` | vendor/lab of the model |
| `model` | canonical family-level model name |
| `total_items_asked` | distinct `item_id`s the model was asked, **deduped per benchmark** (aliases mapping to the same model are collapsed, so items aren't counted once per alias), then summed across benchmarks |
| `total_responses` | total response rows (≥ items; multi-condition benchmarks re-ask items) |
| `n_benchmarks` | number of benchmarks the model appears in |

## `generate_model_statistics.py`

The deterministic implementation lives at
`scripts/render_website/generate_model_statistics.py`. It uses no network or
LLM calls.

**Dedup:** item counts are collapsed per `(model, benchmark)`. When several
subject strings in one benchmark map to the same canonical model (e.g. `gpt-4o`
and `gpt-4o-2024-08-06` → "OpenAI GPT-4o"), that benchmark's items count ONCE for
the model — not once per alias — so `total_items_asked` stays a true distinct-item
count and the leaderboard isn't biased toward models with many alias strings. This
works by tagging every response row with its canonical `model` and then taking
`groupby("model")["item_id"].nunique()` (grouping by *model*, not *subject_id*, so
aliases land in one group). `total_responses` stays summed across aliases — it
counts runs, which are legitimately additive. Keeping the implementation in one
source file avoids a documentation copy drifting out of sync.
