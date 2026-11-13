# Plan: a standard trace format and a structured trace viewer

Status: ProgramBench released; converter-family migrations in progress. PR #15 provides the deployed schema, converter helpers,
mini-swe-agent converter and structured website viewer. PR #16 adds `StructuredTraceBuild`,
write-time validation and format-aware paging. PR #17 migrates ProgramBench; other builders
retain legacy traces until they explicitly migrate. The survey and measurements below describe the
original 2026-11-09 research spike; refresh them against pinned inputs for each migration.

The viewer and schema landed in [#15](https://github.com/aims-foundations/measurement-db/pull/15).
The builder contract follows in [#16](https://github.com/aims-foundations/measurement-db/pull/16)
(`feat/structured-trace-build` → `main`). The ProgramBench pilot is
[#17](https://github.com/aims-foundations/measurement-db/pull/17)
(`feat/programbench-structured-traces` → `main`). The reader and abstract contract are merged;
the ProgramBench release is merged and deployed on main.

## 1. Goal

Formalize the `trace` column of `traces.parquet`. Today it is free-form text: each builder wraps
its upstream data in its own JSON envelope, and no two benchmarks agree. This plan replaces that
with one documented, validated JSON format, **`trace/1`**, for every row of every benchmark,
produced by each benchmark's builder.

With a standard format, the website can show **how a model actually worked through a task**
when a reader clicks a matrix cell (for example on `/measurement-db/programbench`). That includes
the task, every model turn, every tool call and its output, harness interventions, and what the
grader observed. It reads like a chat transcript with tool cards instead of a raw JSON dump, and
one renderer serves every benchmark.

At the end of the migration, every benchmark's `trace` conforms to `trace/1`, the website has one
trace renderer, and no code for the old format remains.

## 2. Decisions

Agreed on 2026-11-09:

1. **Repurpose `trace`, no new column.** The structured value replaces today's content in the
   existing `trace` column. There is no separate trajectory column or table, and the word
   "trajectory" is used only where it is an upstream name (mini-swe-agent's
   `trajectory.messages`).
2. **One format for every row.** Plain answers, agent runs and pairwise sides all use `trace/1`.
3. **Lossless through `native`.** Source fields that the mapping does not use are kept in an
   optional `native` field on each step and on the trace. The UI does not show them yet; the data
   is there if that changes.
4. **The raw record stays reproducible.** Builders download from pinned upstream sources and are
   verified byte for byte, so the original records can always be regenerated with `build.py`.
5. **Convert in the builders, in Python.** Each benchmark's `build.py` produces `trace/1` values,
   with shared converters for common formats. Benchmark authors and curators can review and fix
   the mapping like any other formatted data. The website contains no benchmark-specific
   knowledge.
6. **Run and evaluation are separate sections.** `run` holds what the harness recorded; `evaluation`
   holds what the grader observed. The trace never carries a score: `responses.response` remains
   the only grade.
7. **Staged delivery**: many small branches and PRs (section 11), not one.
8. **Clean finish.** Once every benchmark is converted, the legacy display path, the format marker
   and the whole-trace truncation are deleted, and `trace/1` becomes mandatory.

## 3. Today

### 3.1 Data path

- `scripts/render_website/generate_benchmark_gallery.py` builds, per benchmark, an `answer`
  lookup (observation key → physical parquet row) and a `traces-pages` page index
  (`write_source_pages()`).
- In the browser, `website/lib/gallery-source.worker.ts` resolves the clicked cell's key, reads
  only the needed parquet page of `traces.parquet` through the byte-range proxy
  `website/app/api/gallery-source/[slug]/[table]/route.ts` (which adds `HF_TOKEN` server side),
  cuts the string to `TRACE_MAX_CHARS = 50000` characters and returns `{ trace }`.
  `truncate_trace()` / `read_answer()` in `generate_benchmark_gallery.py` mirror the cut for the
  output-parity tests (`tests/test_gallery_output_parity.py`).
- `website/components/matrix-viewer.tsx` shows the string in a 32rem-tall box through `MathText`
  (Markdown + KaTeX), one pane per model, two for pairwise cells.

### 3.2 Problems

1. **Truncation breaks agent traces.** Agent traces are large JSON documents (programbench rows
   are 0.2–2.5 MB). Cutting at 50,000 characters leaves invalid, partial JSON rendered as Markdown,
   and most of each run is never visible.
2. **No structure.** Tool calls, outputs, harness messages, token usage, cost and grading detail
   are buried in benchmark-specific JSON.
3. **Grading context is lost.** programbench `sheepla__pingu.926d475` × gemini-3-1-pro shows 0%,
   but its trace says `evaluation.error_code = "compile_failed"` and all 515 tests are `not_run`.
   The submission failed to build and the tests never ran. Today's page cannot show that.
4. **Clicks download far too much.** A click reads one whole parquet data page, and the table
   writer uses pyarrow's defaults, which pack up to 1,024 rows into one page. programbench's trace
   column has 8 pages for 4,965 rows, so one click downloads **277 MB (median) to 747 MB**,
   through the Vercel proxy and inside the worker's 120 s timeout. See section 6.
5. **No shared format.** `parquet_schemas.yaml` describes `trace` as "raw model output text";
   dataset users must reverse-engineer every benchmark's envelope.

## 4. What the traces look like (research findings)

267 of the 270 benchmarks in `website/content/generated/gallery-sources.json` publish
`traces.parquet` (104.6 GB in total; largest: alignment_faking 18.5 GB, risebench 9.9 GB, mmlu
8.8 GB). The 75 benchmarks marked `multi_turn` in `metadata.yaml` were scanned in full (32 GB,
revision of 2026-11-02):

- None uses a shared format, and none uses LiteLLM's logging format. LiteLLM response objects
  appear only embedded in programbench and theagentcompany.
- Formats can differ inside one benchmark: programbench mixes three message dialects (OpenAI
  chat 36%, OpenAI Responses 40%, Anthropic blocks 24%) depending on the model client of each run.

| Group | Count | Benchmarks |
|---|---:|---|
| A. OpenAI-chat-style messages at a benchmark-specific path | 10 | agentdojo, taubench, tau2_bench, scale_mrt, openbiorq, swe_smith, mtbbench, autoresearchbench, find_interp, radar |
| B. Real agent runs in a custom structure | 13 | programbench, theagentcompany, visualwebarena, workarena, os_harm, bigfinancebench, dtap_bench, matharena_platform, nyu_ctf_bench, hle, intercode, coffeebench, mypcbench |
| C. Conversations stored as text or one-off shapes | 11 | goodai_ltm_benchmark, wikihow_agent, genai_learning, algotune, aider, appworld, scigym, workbench_revisited, medagentboard, live_agent_risk, confagents |
| D. No agent run (answer, patch, scores or summaries only) | 41 | e.g. matharena, multi_swebench, swe_rebench, kernelbench, prm800k, exploitgym, dataclawbench, frontieror |

The other ~190 benchmarks with traces are single-turn and were not scanned; they are expected to
look like group D (one answer per row) and will be surveyed before their stage.

Group A: where the messages are (JSONPath into the parsed `trace`; `<json>` = a JSON-encoded
string that must be parsed again):

| Benchmark | Rows | Messages path | Notes |
|---|---:|---|---|
| agentdojo | 69,384 | `$` (bare list) | some messages use `tool_call` (singular) |
| taubench | 4,330 | `$.result.traj` | `$.logs[].inputs.messages` duplicates per-call inputs |
| tau2_bench | 23,342 | `$.simulation.messages` | per-message `timestamp`, `cost`, `usage`; simulated user |
| scale_mrt | 51,491 | `$.native_record.conversation` | also `$.native_record.monitor_conversation` (a monitor model) |
| openbiorq | 7,227 | `$.trace` | assistant rounds only; tool results are not in the source |
| swe_smith | 49,897 | `$.renderings[].messages<json>` | 18% structured `tool_calls`; 82% SWE-agent tool use in text |
| mtbbench | 8,412 | `$.conversation` | no tools |
| autoresearchbench | 115 | `$.native_pass.messages` | no tools |
| find_interp | 10,397 | `$.files["history.json"]<json>` | no tools |
| radar | 1 | `$.native_record.llm_messages` | no tools |

Group B shapes:

| Benchmark(s) | Rows | Shape |
|---|---:|---|
| programbench | 4,965 | mini-swe-agent trajectory, details in section 7.4 |
| theagentcompany | 3,057 | OpenHands event list (`source`, `action`, `observation`, `timestamp`, `tool_call_metadata.model_response`) |
| visualwebarena, workarena | 1,075 | BrowserGym/AgentLab `record.steps[]`: `action`, `reasoning`, `axtree`, per-step `chat_messages`, `stats` |
| os_harm | 1,740 | OSWorld `record.steps[]`: `response`, pyautogui `actions`, `a11y_tree`, `screenshot_file` |
| bigfinancebench | 9,000 | `run_records[].record.steps[]`: `assistant_text`, `tool_calls`, `tool_results`, tokens, `cost_usd`, `wallclock_seconds` |
| dtap_bench | 106,570 | `related_histories[].record.trajectory[]` of `{step_id, role: agent/tool/user, action, state}` |
| matharena_platform | 32,503 | `record.all_messages<json>` with roles `user`/`assistant`/`tool_response`, `tool_name`, `arguments`, `tool_call_id` |
| nyu_ctf_bench | 2,979 | multi-agent: `released_traces[].source_record.{autoprompter,planner,executors[]}`, roles `MessageRole.*`, `tool_call`/`tool_result` |
| hle | 14,588 | typed `events[]` (`web-search-call`, `web-search-result`, …) with `toolCallId` |
| intercode | 17,740 | `record.turn_history` with parallel `actions`/`observations`/`rewards` arrays |
| coffeebench, mypcbench | 139 | `thought`/`action`/`observation` steps |

Pairwise benchmarks (arena_140k, sciarena, visit_bench, alpacaeval, …) already store **one
response row per side**, each with its own trace, and name the other model in
`interactors="opponent=…"`. Some builders (e.g. arena_140k) put the whole battle record, both
sides, on every row.

Benchmark slugs in this repo's `benchmarks/` can differ slightly from Hugging Face folder names;
six GitHub benchmarks (bountybench, cybench, osworld, scivisagentbench, prediction_arena,
arena_140k) are not on Hugging Face at all.

## 5. The `trace/1` format

### 5.1 Rules

- **One format for every row.** A plain answer is a trace with one model step; an agent run is a
  trace with many; a row with only a submission or scores may have no steps at all.
- **Lossless.** Every source field is either mapped or kept in the `native` field of the step (or
  the trace) it came from. Content the converter does not recognize becomes an `event` step named
  `unrecognized`, with its source JSON in `native` (section 5.4).
- **`native` holds no copies.** A source field whose value is an exact copy of something already
  mapped is dropped, not kept in `native` (mini-swe-agent's `extra.actions`, which repeats the
  tool calls, and Anthropic `tool_result` blocks' `text`, which repeats `content`). The converter
  checks that the values are equal before dropping one, and keeps the field when they differ. In
  the programbench sample, `native` was 3.36 of 7.82 GB uncompressed, mostly such copies.
- **Omit what the source does not provide.** Optional fields are left out, not written as `null`,
  and never invented.
- **Plain strings.** Content, reasoning, arguments and outputs are stored in full as plain
  strings. Size caps are a display concern and live only in the website (section 8.2).
- **No score.** `responses.response` is the only grade. The `evaluation` section holds the
  evidence behind it: test results, error codes, judge output.
- **No duplicated task.** If the task message's text is identical to the item's `content`, the
  trace omits it; the cell view already shows the item above the trace. A system prompt or task
  that differs from the item (programbench's harness prompt, for example) stays in the trace.
- **One side per trace.** In pairwise benchmarks, a trace holds only its own model's side. The
  opponent's side is the opponent's row; the vote or judge output goes in `evaluation`.
- **Other parties are labeled.** Messages from a simulated user (tau2_bench), an attacker
  (agentdojo) or a human (arena) carry a `participant` that matches the name used in
  `interactors`.

### 5.2 Shape

The normative definition is a JSON Schema, `trace_schema.json`, next to `parquet_schemas.yaml`.
The shape, in TypeScript notation (`?` = optional, omitted when absent):

```ts
type Trace = {
  schema: "trace/1";
  converter: string;                    // e.g. "mini_swe_agent@1", "text@1"
  timing?: "recorded" | "partial" | "synthetic";  // absent: no timing information
  steps: Step[];
  run?: {
    status?: string;                    // harness exit status, e.g. "Submitted", "TimeExceeded"
    submission?: string; exception?: string; traceback?: string;
    cost?: number; apiCalls?: number;
    promptTokens?: number; completionTokens?: number; durationS?: number;
  };
  evaluation?: {                        // absent when the attempt was not graded
    errorCode?: string;                 // e.g. "compile_failed"
    errorDetails?: string;              // e.g. the build log
    passed?: number; total?: number; notRun?: number;
    tests?: { name: string; status: string }[];
    judge?: { content: string; reasoning?: string; participant?: string };
  };
  native?: Record<string, unknown>;     // unmapped trace-level source fields
};

type Common = { agent?: string; native?: Record<string, unknown> };  // agent: e.g. "monitor", "planner"

type Step = Common & (
  | { type: "message"; label: "system" | "task" | "harness" | "user";
      participant?: string; content: string }
  | { type: "llm"; model?: string; content?: string; reasoning?: string;
      toolCalls?: { id?: string; name?: string; arguments: string }[];
      usage?: { prompt?: number; completion?: number; reasoning?: number;
                cacheRead?: number; cacheWrite?: number };
      cost?: number; start?: number; end?: number }
  | { type: "tool"; callId?: string; name?: string; arguments?: string;
      output: string; exitCode?: number; error?: string; start?: number; end?: number }
  | { type: "event"; name: string; content?: string; start?: number; end?: number }
);
```

- `timing`: `recorded` = source timestamps; `partial` = durations only; `synthetic` = order only.
  The UI shows durations only for `recorded`/`partial`.
- `agent` groups steps of multi-agent runs and second conversations (scale_mrt's monitor,
  nyu_ctf_bench's planner and executors) into sections.
- Times are Unix seconds. Arguments are kept as the source string (usually JSON text).

### 5.3 Examples

A plain answer (task omitted because it equals the item content):

```json
{"schema": "trace/1", "converter": "text@1", "steps": [{"type": "llm", "content": "The answer is 42."}]}
```

One pairwise side with a judge vote:

```json
{"schema": "trace/1", "converter": "arena@1",
 "steps": [{"type": "llm", "model": "model-a", "content": "Here is a haiku…"}],
 "evaluation": {"judge": {"participant": "human", "content": "model_a"}}}
```

An agent run (abridged):

```json
{"schema": "trace/1", "converter": "mini_swe_agent@1", "timing": "recorded",
 "steps": [
   {"type": "message", "label": "system", "content": "You are a helpful assistant…"},
   {"type": "message", "label": "task", "content": "Rebuild this program…"},
   {"type": "llm", "model": "gemini-3-1-pro", "content": "Let me look around.",
    "toolCalls": [{"id": "c1", "name": "bash", "arguments": "{\"command\": \"ls\"}"}],
    "usage": {"prompt": 2103, "completion": 41}, "cost": 0.004, "start": 1761903000.1, "end": 1761903004.9,
    "native": {"response_id": "…", "system_fingerprint": "…"}},
   {"type": "tool", "callId": "c1", "name": "bash", "arguments": "{\"command\": \"ls\"}",
    "output": "Cargo.toml\nsrc\n", "exitCode": 0, "start": 1761903004.9, "end": 1761903005.2}
 ],
 "run": {"status": "Submitted", "cost": 1.92, "apiCalls": 88},
 "evaluation": {"errorCode": "compile_failed", "errorDetails": "error[E0432]…", "passed": 0, "total": 515, "notRun": 515}}
```

### 5.4 Unknown and missing content

Today nothing counts as unknown, because nothing is parsed: the website shows the first 50,000
characters of whatever the builder wrote. With parsing, three cases occur:

| Case | Examples | Handling |
|---|---|---|
| Unrecognized shapes | programbench: none in 1.42M messages. Likely elsewhere: new event types in hle, `MessageRole.*` roles in nyu_ctf_bench, harness versions newer than the converter | `event` step named `unrecognized`, source JSON in `native`; counted (section 9) |
| Known variants a converter can miss | programbench's 718 format-error replies stored on a user message (an early parser dropped 349 of them); agentdojo's singular `tool_call`; swe_smith's tool use in text | mapped explicitly; one fixture per variant |
| Missing data | openbiorq has no tool results; programbench row 4697 is an ungraded attempt (`evaluation: null`) | the field or section is omitted |

### 5.5 Documentation, schema and migration marker

- `schema.md` and the `trace` description in `parquet_schemas.yaml` describe `trace/1`, with the
  three examples of section 5.3.
- During the migration, files differ in format. Each `traces.parquet` written in the new format
  carries the parquet metadata key `measurement_db.trace_format = "trace/1"`, and every value
  carries `"schema": "trace/1"`. Files without the key are legacy. The gallery generator copies
  the marker into `gallery-sources.json`, so the worker chooses a renderer from it and never from
  content.
- At the end (stage 12), `PARQUET_SCHEMA_VERSION` goes from `"3"` to `"4"`, `validate_table`
  rejects any `trace` that is not `trace/1`, and the marker is removed.

## 6. Storage layout: one row per data page

Measured on the full programbench `traces.parquet` (2.89 GB, the pinned revision), rewritten
with pyarrow `write_batch_size=1, data_page_size=1` (each row flushes its own page), dictionary
encoding off for `trace`, still one row group. The repo's `write_source_pages()` indexed the
rewritten files unchanged; hyparquet 1.31.2 reads made the way `readRow()` does matched pyarrow
byte for byte.

| Layout | Pages | Download per click (median / p99 / max) | `trace` column size |
|---|---:|---|---:|
| Today (pyarrow defaults), raw content | 8 | 277 MB / 747 MB / 747 MB | 2.89 GB |
| One row per page, raw content | 4,965 | 0.33 MB / 3.65 MB / 6.86 MB | 2.88 GB |
| One row per page, prototype structured content (no `native`) | 4,964 | 0.14 MB / 0.95 MB / 1.88 MB | 1.09 GB |

Reading one row took 3–30 ms instead of about 850 ms (local disk). The page index grows from 1 KB
to about 56 KB.

- The paging change is independent of the format work, and is its precondition: without it, no
  trace display is usable on programbench. It applies to `traces.parquet` only.
- The structured size above excludes `native`. Measured on the full programbench sample, with
  `native`: raw 9.04 GB → `trace/1` 7.82 GB uncompressed (0.86×), of which `native` is 3.36 GB;
  2.43 GB compressed against 2.88 GB raw. Most of `native` was copies of mapped values, which
  section 5.1 now excludes.

## 7. Builders

### 7.1 Shared code

- `build_base.py::StructuredTraceBuild` is the opt-in abstract builder contract. Each migrated
  builder implements `convert_trace(source, *, context) -> ConvertedTrace`. `TraceContext` is
  immutable and contains the final observation identity and item text; `ConvertedTrace` holds
  the `trace/1` document and accounting counters. Benchmark-specific parsing stays in `build.py`.
  Shared format converters are optional helpers called from this method.
- The shared pipeline invokes conversion once for each available trace after response assembly.
  Both `build_tables()` and `build_subject_item_response_rows()` use this path. Missing traces
  remain absent; conversion does not create or modify response rows, grades or identities.
- `scripts/build_measurement_tables/trace_format.py`: step constructors, `native` collection
  (a helper that returns the source fields not consumed by the mapping), the task-omission rule,
  JSON Schema validation against `trace_schema.json` (`jsonschema` is already a dependency), and
  the accounting counters.
- `scripts/build_measurement_tables/trace_converters/<name>.py`: converters for formats shared by
  several benchmarks: `text` (plain answers), `chat` (group A), `mini_swe_agent`, `openhands`,
  `browsergym`. Each has a version (`chat@1`) written into `converter`.
- Benchmark-specific mapping stays in that benchmark's `build.py`, so a curator can read the
  whole mapping in the file they own.

### 7.2 In a builder

A migrated builder inherits `StructuredTraceBuild`. Its existing `trace` input can contain
native dictionaries, message lists or text; the shared pipeline handles serialization. The
following simplified ProgramBench example illustrates the hook; its actual adapter also
preserves provider metadata and complete test-result identities:

```python
from measurement_db.build_base import ConvertedTrace, StructuredTraceBuild
from measurement_db.scripts.build_measurement_tables.trace_converters import mini_swe_agent

class ProgramBench(StructuredTraceBuild):
    def convert_trace(self, source, *, context):
        record = json.loads(source) if isinstance(source, str) else source
        value, counts = mini_swe_agent.convert(
            record["trajectory"], evaluation=record.get("evaluation"),
            native={k: v for k, v in record.items() if k not in {"trajectory", "evaluation"}},
        )
        return ConvertedTrace(value, counts)
```

Legacy `BenchmarkBuild` subclasses keep accepting only trace strings and retain their existing
output. Inheriting `StructuredTraceBuild` is the migration opt-in; no per-benchmark metadata flag
or external postprocessing command is needed. ProgramBench is the first audited migration;
additional converter families follow incrementally.

### 7.3 Write-time validation

In `BenchmarkBuild._validate_and_write_tables`, before `traces.parquet` is written:

1. Every value is validated against `trace_schema.json`.
2. Accounting: emitted unrecognized events are counted independently and checked against the
   converter's counters. Unrecognized events and reported unmatched tool results default to
   zero tolerance. A builder may declare reviewed finite totals with
   `TRACE_ALLOWANCES = {"unrecognized": n, "unmatched_tool_results": m}`. Invalid counters and
   parse failures always fail the build. A present trace cannot be converted to `None`.
3. Counts (rows, steps by type, unrecognized, unmatched) are printed in the build log.
4. The file gets the `measurement_db.trace_format` marker (section 5.5) and the one-row-per-page
   layout (section 6).

The writer validates structured JSON before opening its output and keeps the existing table
columns and schema-version metadata. It disables dictionary encoding only for the trace column.
Legacy writes retain their prior options and bytes. Tests cover row/table builder parity,
unchanged non-trace files and identities, missing/ungraded traces, failure handling, accounting
and the page index consumed by the website.

### 7.4 Converters

`chat` (group A). Options: the messages path, optional field mapping. The first `system` message
→ `message/system`; the first `user` message → `message/task` (unless identical to the item);
later `user` messages → `message/user` (with `participant` where the benchmark has a simulated
user or attacker); each `assistant` message → `llm` step (content, `tool_calls` or singular
`tool_call`); each `tool` message → `tool` step matched by `tool_call_id`, with name and arguments
taken from the matching call. Unmatched tool results are kept and counted. tau2_bench: per-message
`timestamp`/`cost`/`usage`, `timing: "recorded"`. scale_mrt: the monitor conversation as steps
with `agent: "monitor"`. swe_smith text tool use: kept as text in `chat@1`.

`mini_swe_agent` (programbench). Port of the verified prototype parser (`normalize()` and the
`parse_*` functions, section 14), already Python, checked against all 4,965 rows: LLM step counts
match the agent's own
`model_stats.api_calls` except for calls the agent never saved, every tool result matched its
call, and no message was unrecognized.

- Input: `trajectory.messages`; also `trajectory.info` (`model_stats`, `config.model.model_name`,
  `exit_status`) and the evaluation record (`test_results[]` with `name`/`status`,
  `error_code`, `error_details`).
- **Dialect per message:**
  - `object === "response"` → OpenAI Responses API turn. `output[]` items: `function_call`
    (`call_id`, `name`, `arguments`), `message` (text from `content[]`), `reasoning` (`summary` +
    `content` text; usually empty). Usage: `input_tokens`, `output_tokens`,
    `output_tokens_details.reasoning_tokens`, `input_tokens_details.cached_tokens`. End:
    `extra.timestamp`, else `completed_at`. Start hint: `created_at`.
  - `type === "function_call_output"` → tool result (`call_id`, `output`).
  - `role === "assistant"` with a `tool_calls` key, or with string/null content → chat
    completions turn: `tool_calls[].function.{name,arguments}`, usage from
    `extra.response.usage` (`prompt_tokens`, `completion_tokens`,
    `completion_tokens_details.reasoning_tokens`, `prompt_tokens_details.cached_tokens`),
    reasoning from `extra.response.choices[0].message.reasoning_content`, start hint
    `extra.response.created`.
  - `role === "assistant"` with list content → Anthropic blocks: `text`, `thinking`,
    `redacted_thinking`, `tool_use` (`id`, `name`, `input`). Usage from `extra.response.usage`.
    **Prompt tokens = `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`**
    (Anthropic's `input_tokens` excludes cached tokens).
  - `role === "tool"` → chat tool result (`tool_call_id`).
  - `role === "user"` whose content list contains `tool_result` blocks → Anthropic tool results;
    pair the i-th `tool_result` with `extra.observations[i]` for `returncode`/`timestamp`.
  - `role === "user"` with `extra.response` → the model answered without calling a tool, and newer
    agent versions stored that reply on the format-error message. Emit an `llm` step from
    `extra.response` (Responses object if `object === "response"`, else chat completion), then
    the user message as a harness message. **Both shapes occur** (chat in glm-5.2,
    gemini-3-6/3-7-flash, opus-4.8, muse-spark-1.3-xhigh; Responses in muse-spark-1.1/1.2/1.3-max).
  - other `system`/`user` messages → `message` steps (first user = task, later = harness).
  - `role === "exit"` → `run`: `extra.exit_status`, `extra.submission`, `extra.exception_str`,
    `extra.traceback`.
- **Cost** per LLM step is `extra.cost` of that message (absent on format-error replies).
  A cost of exactly 0 on a step or run that used tokens means the harness did not price the
  model (about 20% of programbench runs, e.g. every Claude Opus run), not that it was free:
  omit `cost` and keep the raw 0 in `native`.
- **Tool output** is wrapped as `<returncode>N</returncode>\n<output>\n…</output>\n`. Strip the
  wrapper only when the whole string matches
  `^<returncode>(-?\d+)</returncode>\n<output>\n?([\s\S]*)</output>\n?$`; otherwise keep it raw.
  Exit code from `extra.returncode`; error from `extra.exception_info` / `extra.exception`.
- **Timing** (`recorded`): each step starts where the previous step ended and ends at its own
  timestamp; the first LLM step starts at its start hint. If a start is after its end, set
  start = end. Steps without an end time use their start.
- **Ungraded attempts** (e.g. row 4697, `evaluation: null`): no `evaluation` section. The
  prototype converter crashed on this row; the parser itself handles it.
- Expected volume: 696,537 LLM steps (718 from format-error replies) and 703,996 tool steps
  across 4,965 rows; up to 1,000+ steps per run.

Remaining converters, in stage order (section 11):
1. `chat` for group A.
2. `openhands` (theagentcompany): agent actions with `tool_call_metadata` → `llm` steps,
   observations → `tool` steps, `timestamp` → `recorded`.
3. `browsergym` (visualwebarena, workarena): one `llm` step per step (`reasoning`, `action`) and an
   `event` for the observation (URL, last action error; the axtree in `native`).
4. bigfinancebench, dtap_bench, matharena_platform, nyu_ctf_bench (`agent` per sub-agent), os_harm,
   hle, intercode, coffeebench/mypcbench: mapping in each `build.py`.
5. Group C: a mapping per benchmark (goodai_ltm_benchmark and wikihow_agent are simple prefix
   splits). Conversations stored only as text become one `llm` step, not guessed turns.
6. Group D and the single-turn benchmarks: `text` (answer → one `llm` step), with patches and
   final programs in `run.submission` and stored scores or summaries in `evaluation`.
7. Pairwise builders: drop the opponent's side from each row, and put votes or judge output in
   `evaluation.judge`.

### 7.5 Keeping Hugging Face in sync

Reproduction is byte-exact (`scripts/ci/benchmark_reproduction.py::compare_tables`), and
`changed_benchmarks()` reruns only benchmarks whose `benchmarks/<slug>/` folder changed. A change
to a shared converter, `trace_format.py` or the table writer therefore changes the output of
every dependent benchmark without CI noticing. Plan:

- `changed_benchmarks()` also selects every benchmark whose `build.py` imports a changed module
  under `scripts/build_measurement_tables/trace_converters/` or `trace_format.py`. Changes to the
  writer select all benchmarks with traces.
- Every converted benchmark is rebuilt and re-uploaded in its stage, and its revision in
  `gallery-sources.json` is updated, before the website relies on it.

## 8. Website

### 8.1 Worker

`gallery-source.worker.ts` reads the trace row as today. Then:

- **`trace/1` files** (marker from `gallery-sources.json`): parse the JSON, apply display caps
  (section 8.2), return `{ trace: null, structured: Trace }`. If parsing fails, return the raw string as today and
  log a console warning with the slug and response key. A broken value must never hide a trace.
- **Legacy files**: unchanged: the 50,000-character cut and `{ trace: string }`, so
  `tests/test_gallery_output_parity.py` and `website/scripts/test-gallery-source*.mjs` stay
  valid until the cleanup stage.

### 8.2 Display caps

The dataset stores full text; the worker caps fields for display and records the original
length (`{ text, truncatedFrom }` in the website's own types):
- tool output, message content, reasoning: 20,000 characters each (tune after measuring);
- test list: at most 5,000 entries, non-passed first;
- whole trace: if the capped result is over ~5 MB, lower the per-field cap and retry.

`native` is not sent to the main thread in the first release.

### 8.3 Trace viewer

New component `website/components/trace-viewer.tsx`, rendered by `matrix-viewer.tsx` in place of
the `MathText` answer pane when the worker returns a structured trace. Use the site's Tailwind
setup and CSS variables (`--ink`, `--muted`, `--palo-alto`, …) and match the surrounding pane
styling. Behavior proven in the prototype:

1. **Header:** the matrix row's subject name, with the provider's model ID beside it in small
   type; run status badge (neutral: its vocabulary is the harness's own, so only a recorded
   exception turns it red), tests passed `x / y` with a bar, an **"evaluation:
   compile_failed"**-style badge when `evaluation.errorCode` is set, **"no tests ran"** when
   `notRun === total`, and chips for cost, model turns, API calls (only if different from turns),
   tool calls, prompt/output tokens and duration. A missing cost reads "not recorded". The matrix score from `responses.parquet` stays
   as today; the trace complements it and is never used to recompute it.
2. **Step strip:** one thin tick per step, click to jump, search hits marked on top, the current
   step marked below. Only the steps a reader hunts for are colored (tool non-zero exit, tool
   error, harness message); the rest are one neutral color, because model turns and tool calls
   alternate on agent runs and coloring both reads as a single band. Long runs have 1,000+
   steps, so this is the main navigation.
3. **Timeline:**
   - system prompt and task: collapsed to ~8 lines with "Show all N lines";
   - model turn: text, reasoning in a collapsed `<details>`, one card per tool call. A tool call
     whose arguments are exactly `{"command": "…"}` shows the command as a code block; otherwise
     pretty-printed JSON. Corner metadata: tokens in/out (+reasoning, cached), cost, latency;
   - tool result, indented under its call: tool name, exit-code badge (green 0, amber non-zero,
     red error), duration, output clamped to ~16 lines with "Show all N lines". The command is
     already on the model turn above, so the result repeats its first line only when that turn
     made several calls or the result matches none of them;
   - harness message (e.g. "No tool calls found…" format errors) in a distinct accent color,
     labeled "sent by the agent harness, not a person"; other participants labeled by
     `participant`;
   - steps with an `agent` grouped under that agent's heading (e.g. "Monitor");
   - an empty model response shown explicitly as "(empty response: no text and no tool call)".
   - A trace with a single model step and nothing else renders as a plain answer, like today.
4. **Search within the trace:** highlights matches (built from text nodes; never inject trace
   content as HTML), counts matching steps, Enter / Shift+Enter to move, auto-expands clamped
   blocks that contain a match. No other keyboard shortcuts: the site has none, so they would be
   undiscoverable; Escape closes full screen.
5. **Result section:** run status, exception and traceback, submission; evaluation error code and
   details; judge output; filterable list of non-passed tests (status containing "fail"/"error"
   in red, others such as `not_run` in amber). Label totals as "tests in this run's evaluation".
6. **Space:** inline, the trace is part of the page, with no scroll area of its own (nested
   scrolling trapped the mouse wheel). The search bar and step strip stick to the top of the page,
   just below the site header, which publishes its visible height as `--rd-header-offset`; the
   toolbar has a "Back to matrix" control. "Open full screen" shows the same viewer as an overlay
   with its own scroll area. A dedicated route (`/[slug]/trace/[responseKey]`) with step deep
   links (`…#s42`) is a later stage.
   - **Item prompt:** when the trace contains a `task` message, the cell's item prompt is folded
     into a "Item prompt" disclosure, since the run starts with the task as the agent received
     it. The item header and correct answer stay visible. The pane heading is "Run" instead of
     "Model answer".
7. **Pairwise cells:** two stacked trace viewers rather than side by side.
8. **Performance:** do not use `content-visibility: auto` for step cards; it made jump-to-step land
   in the wrong place in the prototype. Plain rendering of 375 steps took about 25 ms. Consider
   virtualization only if runs with several thousand steps are slow.
9. **Accessibility and theme:** step strip ticks need labels; badges carry text, not only color;
   respect `prefers-reduced-motion`. The existing panes use hard-coded `bg-white`; check light and
   dark mode.

## 9. Validation

| Layer | What | Where |
|---|---|---|
| Schema | Every value validates against `trace_schema.json` | build time, every row (7.3) |
| Accounting | Unrecognized, unmatched and unparsable counts within the allowance | build time, every row (7.3) |
| Converter fixtures | Real source records with expected `trace/1` output, one per dialect and variant | `tests/fixtures/traces/<converter>/`, pytest |
| Cross-checks | Counts against an independent counter in the source: programbench LLM steps vs `model_stats.api_calls` (known gap: format-error turns older agents never saved); bigfinancebench summed tokens vs `total_prompt_tokens`/`total_completion_tokens` | converter tests and the stage PR |
| Reproduction | Byte-exact rebuild of every converted benchmark, including dependents of changed shared code (7.5) | `benchmark-reproduction.yml` |
| Website logic | Display caps, legacy path, fallback on unparsable values | Node test `website/scripts/test-trace-display.mjs`, added to `website.yml` |
| UI | Manual check on the section 14 runs in a preview deployment, light and dark mode, phone width | stage PRs |

programbench fixtures must cover: chat, Responses, Anthropic, both format-error reply shapes, an
`evaluation.errorCode` run, a non-`Submitted` exit and an ungraded attempt.

Trade-offs of validating in the builders: every row is checked on every build and nothing reaches
Hugging Face unvalidated, but a converter fix needs a rebuild, an upload and a revision bump
before readers see it. The website layer stays thin, so its tests stay small.

## 10. Size and performance

- One row per page (section 6) bounds a click to roughly one row's compressed size.
- Structured content without `native` was 0.38× the raw size for programbench; with `native`
  (before the no-copies rule) it was 0.86× (section 6). Record per-benchmark sizes in each
  conversion PR.
- The website does not send `native` to the main thread, but the worker still downloads it as part
  of the value. If `native` makes clicks slow for some benchmark, options are to trim what that
  converter keeps or to revisit storing `native` separately (needs a new decision).

## 11. Stages

Each stage is its own branch and PR, reviewable and releasable on its own.

| Stage | PR | Work | Done when |
|---|---|---|---|
| 1 | Trace paging | Table writer writes `traces.parquet` one row per page (section 6); CI selects all benchmarks with traces when the writer changes | writer tests pass; `write_source_pages()` indexes a rewritten fixture; gallery parity tests pass |
| 2 | Re-upload with new paging | Rebuild and upload benchmarks with large traces (programbench first), update pinned revisions | a programbench click downloads ≤ ~7 MB in a preview deployment; display unchanged |
| 3 | `trace/1` foundation | `trace_schema.json`, `schema.md`, `trace_format.py`, write-time validation and marker, `add_response` dict support, CI dependent selection (7.5) | unit tests pass; no benchmark output changes |
| 4 | programbench converter | `mini_swe_agent` converter, `build.py` change, fixtures, rebuild and upload | 4,965 rows, 0 parse failures, 0 unrecognized, step totals as in 7.4; size with `native` recorded |
| 5 | Worker path | Marker in `gallery-sources.json`, worker `trace/1` path, display caps, fallback | existing gallery tests pass unchanged; new Node test passes |
| 6 | Trace viewer UI | `trace-viewer.tsx` + `matrix-viewer.tsx` integration | section 8.3 items 1–5 on the section 14 runs; section 12 criteria met |
| 7 | `chat` converter | Group A, a few benchmarks per PR, each rebuilt and uploaded | each passes section 9 |
| 8 | Group B | One PR per converter or benchmark, in section 7.4 order | each passes section 9 |
| 9 | Group C | One PR per benchmark | each passes section 9 |
| 10 | Pairwise | Pairwise builders, one side per trace, judge output in `evaluation` | each passes section 9 |
| 11 | Group D and single-turn | `text` converter; survey first; batches of benchmarks per PR | all 267 benchmarks with traces carry the marker |
| 12 | Cleanup | Delete the legacy worker path, the 50,000-character cut, `truncate_trace()`/`read_answer()` legacy code and parity fixtures, the marker; `PARQUET_SCHEMA_VERSION` 3 → 4; `validate_table` requires `trace/1` | no legacy trace code remains; all tests pass |
| 13 | Optional: full-screen route | `/[slug]/trace/[responseKey]` with step deep links | |

Stages 1–2 help every benchmark with large traces and can ship before the rest. Follow the repo's
usual PR flow; the website build steps are in `scripts/render_website/README.md`.

## 12. Acceptance criteria for the first release (stages 1–6)

- Clicking any programbench cell shows the structured viewer, falling back to the raw display only
  for a value that cannot be parsed.
- The gemini-3-1-pro × `sheepla__pingu.926d475` cell shows "evaluation: compile_failed" and "no
  tests ran", with the build log in the result section.
- Search, step strip, collapsible outputs and the sticky toolbar work on the longest run in the
  prototype set (muse-spark-1.2-xhigh × `sheepla__pingu.926d475`, 375 steps).
- All existing gallery tests pass; benchmarks not yet converted look exactly as before.

## 13. Risks and open questions

1. **Transfer time through the proxy.** Bytes per click are measured (section 6); end-to-end time
   through the Vercel proxy is not. Measure in stage 2.
2. **`native` size.** Measured on the programbench sample (section 6): 43% of the `trace/1`
   bytes, mostly copies of mapped values. Re-measure after the no-copies rule (section 5.1) in
   stage 4; it bounds how much smaller than the raw file a converted file can get.
3. **Re-upload volume.** Converting everything rewrites 104.6 GB of traces and rebuilds 267
   benchmarks. Shared-code changes that select many dependents make CI runs long; batch them.
4. **Format evolution.** Once published, `trace/1` is a contract for dataset users. Additive
   optional fields can stay `trace/1`; anything else is `trace/2` with a migration.
5. **Licensing.** The programbench page notes its result repositories do not declare
   redistribution terms. Conversion changes the layout of published traces, not what is
   published. Confirm with the team before publishing traces for benchmarks that do not show them
   today.
6. **Reasoning text is mostly absent.** Only some runs contain readable reasoning (in
   programbench, glm-5.2). The UI must not suggest missing reasoning is an error.
7. **Test totals differ per model on the same task** (programbench `sitkevij__hex.61ae69b`:
   1,164 / 877 / 823), from the source data and not investigated; the UI labels totals as "tests in
   this run's evaluation".
8. **Task-omission edge cases.** Whitespace or templating differences between the item content and
   the task message keep the task in the trace. That is acceptable (duplication, not loss).

## 14. ProgramBench builder migration

ProgramBench opts into `StructuredTraceBuild`; its `convert_trace` hook is in
`benchmarks/programbench/build.py`. The hook adapts the three mini-swe-agent
dialects, retains full branch/test names and repeated test results, and keeps
provider timestamps, content-block signatures and tool-call metadata in
`native`. An empty published verdict map stays distinct from an absent grade.
Measurement construction and the pinned upstream inputs are unchanged.

The native audit accepts both legacy envelopes and `trace/1`. It independently
checks source identities, configuration, test records, recorded turn counts,
tool associations and source scalar/text coverage. Coverage permits text
substrings when content blocks are flattened; this is not an exact JSON-tree
round trip. The source snapshots remain the reconstruction reference.

Review this stack in order: trace viewer (#15), abstract builder contract (#16),
then ProgramBench. Compare the migration with its parent builder using the same
inputs and pinned writer environment. Non-trace files must remain byte identical;
only trace payloads, the format marker and trace page layout should change.
Publish converted tables to the canonical dataset after the compatible reader
is deployed. The normal reproduction check then compares the migrated builder
against that published revision before the builder PR is merged.

For isolated review, `deploy-website.yml` accepts `preview_gallery_revision`, an
immutable commit in the gallery repository containing `preview/manifest.json`.
The manifest must explicitly identify itself as preview-only and match the
checked-out code commit and tree. Every archive/member checksum is verified.
This path rejects production and main before downloading data; it does not
promote a gallery or substitute for the ordinary reproduction receipt.
Rebuild only the migrated benchmark bundle and shared catalog, retaining the
accepted bundles for other benchmarks. Pin the converted source tables on a
separate HF branch so browser range requests exercise the new traces.

## 15. Reference material

The research artifacts are kept outside this repo, because they contain benchmark data and the
prototype is not production code. Ask the plan's author for them.

- **Prototype viewer:** a standalone HTML page with 12 programbench runs, showing the target
  behavior of section 8.3. Its data uses the earlier field names (`outcome`, `final`).
- **Prototype parsers:** `normalize()` and the `parse_*` functions for the three mini-swe-agent
  dialects, the verified reference for `mini_swe_agent` (section 7.4), and the converter that
  feeds the prototype viewer.
- **Format survey:** a per-benchmark table of the 75 scanned benchmarks, every structural
  signature with example rows, and the scanner.
- **Paging experiment (section 6):** scripts that rewrite a traces file one row per page, index it
  with the repo's `write_source_pages()`, measure page sizes, and read rows the way the worker does.
- **Prototype runs** (programbench response IDs, useful for manual checks):
  - `318db1aaee7dc64c`: gemini-3-1-pro × `sheepla__pingu.926d475` (compile_failed, 0 tests run)
  - `8acab3ff317104fb`: muse-spark-1.2-xhigh × `sheepla__pingu.926d475` (375 steps, format errors)
  - `39197df399b4fad8`: opus-4-7 × `sitkevij__hex.61ae69b` (Anthropic blocks, 389 failing tests)

## 13. Converter-family rollout

Work continues on `feat/trace-converter-families`, based on
`codex/independent-gallery-workflow`. Reproduction is manually triggered;
gallery preparation and preview deployment run independently.
Builder-only PRs do not prepare the full published gallery: their converted data
is reviewed through a targeted migration preview. Main pushes after benchmark
publication and explicit gallery runs still prepare the published catalog.

| Family | Representative builders | State |
|---|---|---|
| mini-swe-agent | ProgramBench | Released |
| Chat messages and tool calls | tau-bench, AgentDojo | First migration batch |
| OpenHands events | TheAgentCompany | 3,057 traces validated; staged preview |
| BrowserGym steps | WorkArena, VisualWebArena | Pilot implemented; validation recorded in the PR |
| Custom and multi-agent records | OS-Harm, NYU CTF, others | Separate adapters after source review |
| Text, submissions and pairwise sides | Survey each source shape before choosing pilots | Planned |

The chat batch implements `chat@1` through each builder's `convert_trace()` hook.
Benchmark envelopes and HAL-specific extraction stay in the builders; the shared
converter handles message roles, content blocks, reasoning, usage and tool calls.
It retains unused fields and references to mapped values, allowing an independent
audit to recover every original message, including null fields and provider
extensions. A task identical to item content is recoverable through the item link.

HAL's duplicate provider wrappers are counted once per recorded response ID.
Incompatible planner, executor or retry histories remain separate labeled
contexts. Complete parallel-call proposals survive harness pruning. An ID-less
result may use the executed call explicitly recorded in the harness history;
empty IDs do not become invented identifiers. Indistinguishable repeated calls
are preserved in source order and reported in the conversion counters. Sierra's
known simulator messages are labeled separately. HAL API user roles remain
unattributed, since they may contain harness instructions as well as customer text.
The saved-reply count stays in native provenance; it is not presented as a
complete API-call total. Separate context snapshots can repeat historical turns.

For each family, first migrate representatives, then check full message/envelope
recovery, source order, call associations, schema validation, observation identity,
and unchanged measurement tables. Check the staged viewer on desktop and mobile,
including search, step navigation, tool output, participant labels and errors.
Only then expand the family to additional builders.

Migration previews convert the pinned published legacy traces with the final
builder hook and retain the existing measurement files and identities. Fixture
builds exercise the complete shared builder path. This preview validation is
separate from a fresh upstream rebuild: the latter remains available as a manual
release check before canonical publication. Preview data and galleries use
isolated HF branches and immutable revisions; they do not update the canonical
dataset or the accepted production gallery.

### OpenHands pilot

TheAgentCompany implements `StructuredTraceBuild.convert_trace()` using `openhands@1`.
Recorded provider response IDs deduplicate parallel action records; tool outputs
link through the native event `cause` and recorded call ID. Untimed event records
retain their original timestamps without assuming a timezone. Provider Unix
timestamps give partial timing where recorded. Recall, condensation and orphan
harness errors remain explicit events.

The release also contains OWL role-play records and MUSE text transcripts.
The builder maps recorded role-play pairs and nested tool results separately;
plain transcripts remain one complete text step with no guessed turn boundaries.
Source-field references support independent display validation, and the complete
native input remains recoverable byte-for-byte for the existing source audit.

The pinned pilot has 2,707 OpenHands event streams, 175 OWL role-play traces, and
175 MUSE text transcripts. All original inputs and displayed fields passed
independent recovery checks, all converted documents passed schema validation,
and the four measurement files and seven observation identity columns stayed
unchanged. A source-only census found 76,372 distinct saved provider responses;
all appear in the event-action coverage, with 112 copied action records deduplicated.

### BrowserGym pilot

WorkArena and VisualWebArena share `browsergym@1` through their builder hooks.
Each saved snapshot becomes an observation event, followed by its recorded
last-action error when present, then the current reasoning/action output.
Terminal snapshots with no agent output remain observation-only. Accessibility
trees, DOM geometry, recorded image paths and complete input histories remain
in the recoverable native record. No tool IDs, missing images or timestamps are
invented. Token and cost fields retain their recorded per-step scope; zero cost
with nonzero tokens is kept natively without claiming the call was free.

The full builder tests retain repeated human ratings, independent automatic
grades, disagreement cases, and exact original image assets. Native audits accept
both legacy envelopes and structured traces while rejecting changed source
fields, grades, associations and assets.

### Text pilot

MathArena implements the same abstract hook through `text@1`: one complete
recorded answer becomes one model-output step. JSON-looking mathematical strings
remain literal text. No turns, timestamps, test results or judge records are
invented. The existing selection of answer versus parsed answer, first usable
grade link, proof-criterion expansion and input images stays unchanged.
Full builder fixtures verify exact measurement and asset files; the native
characterization audit recovers the original answer from either trace format.
