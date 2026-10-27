import assert from "node:assert/strict";
import "./load-typescript.mjs";
import test from "node:test";
import { createRequire } from "node:module";

test("math markup and its stylesheet use the same KaTeX version", () => {
  const require = createRequire(import.meta.url);
  const markdown = createRequire(require.resolve("rehype-katex"));
  assert.equal(require("katex").version, markdown("katex").version);
});

async function loadModule(name) {
  const url = new URL(`../lib/benchmark-records/${name}.ts`, import.meta.url)
    .href;
  return { url, exports: await import(url) };
}
const scaleModule = await loadModule("scale");
const { gradeLabel, describeScale } = scaleModule.exports;
const modelModule = await loadModule("model", { scale: scaleModule });
const { prepareMatrix, observationsFor, findSubjects, pageOf, prepareItem } =
  modelModule.exports;
const baseModule = await loadModule("../base-path");
const parquetModule = await loadModule("parquet", {
  "../base-path": baseModule,
});
const { RecordSource } = (
  await loadModule("client", { parquet: parquetModule })
).exports;
const { recordCard } = (await loadModule("catalog")).exports;
const structuredModule = await loadModule("../structured-content");
const { structuredValue, metadataFields } = structuredModule.exports;
const { prepareItemContent } = (
  await loadModule("../item-content", {
    "./structured-content": structuredModule,
  })
).exports;
const signal = () => new AbortController().signal;

test("MMLU answer scores preserve the native choice order and exact values", async () => {
  const { answerScores } =
    await import("../lib/presentation-transforms/benchmarks/mmlu.ts");
  const source = JSON.stringify({
    source_record: {
      choices: [" A", " B", " C", " D"],
      predictions: [-3.21, -4.005, -1.8, -7],
    },
  });
  assert.deepEqual(answerScores(source), [
    { label: "A", score: -3.21 },
    { label: "B", score: -4.005 },
    { label: "C", score: -1.8 },
    { label: "D", score: -7 },
  ]);
  const { presentationFor: lookup } = (await loadModule("presentation"))
    .exports;
  assert.equal(lookup("unregistered").answerScores, undefined);
  for (const value of [
    null,
    "invalid",
    JSON.stringify({
      source_record: { choices: ["A", "B"], predictions: [1, 2] },
    }),
  ])
    assert.equal(answerScores(value), undefined);
});

test("structured display decodes nested JSON without coercing IDs or losing false and null", () => {
  const value = {
    id: "000123",
    enabled: false,
    score: 0,
    missing: null,
    extra: ["x"],
  };
  assert.deepEqual(
    structuredValue(JSON.stringify(JSON.stringify(value))),
    value,
  );
  for (const text of [
    "000123",
    "123",
    "false",
    "null",
    "{malformed",
    "Plain **Markdown**",
  ])
    assert.equal(structuredValue(text), text);
  assert.equal(structuredValue(false), false);
  assert.equal(structuredValue(null), null);
});

test("metadata fields retain ownership and preserve ambiguous field lists intact", () => {
  const source = {
    content: "query=a;b=c",
    test_condition: "split=test;seed=0",
    interactors: "opponent=model-a",
    subject_features_extra: "flag=false;empty=",
    item_features: "note=sentence; with a semicolon",
  };
  const before = structuredClone(source);
  assert.deepEqual(metadataFields(source), {
    ...source,
    test_condition: { split: "test", seed: "0" },
    interactors: { opponent: "model-a" },
    subject_features_extra: { flag: "false", empty: "" },
  });
  assert.deepEqual(source, before);
  assert.deepEqual(metadataFields({ test_condition: "key=one;key=two" }), {
    test_condition: "key=one;key=two",
  });
});

test("MMLU shows the evaluated question and retains examples and request details", () => {
  const context =
    "Instructions.\n\nExample?\nA. One\nB. Two\nC. Three\nD. Four\nAnswer: B\n\n";
  const source = {
    prompt:
      context +
      "Read this passage.\n\nWhich statement follows?\nA. First\ncontinued\nB. Second\nC. Third\nD. Fourth\nAnswer:",
    candidate_continuations: [" A", " B", " C", " D"],
    request_protocol: "Score each continuation.",
    extra_metadata: { seed: 0 },
  };
  const result = prepareItemContent(JSON.stringify(source), "mmlu");
  assert.equal(
    result[0].text,
    "Read this passage.\n\nWhich statement follows?",
  );
  assert.deepEqual(result[1].choices, [
    { label: "A", text: "First\ncontinued" },
    { label: "B", text: "Second" },
    { label: "C", text: "Third" },
    { label: "D", text: "Fourth" },
  ]);
  assert.equal(result[2].blocks[0].text, context);
  assert.deepEqual(result[3].blocks[0].value, {
    request_protocol: source.request_protocol,
    extra_metadata: { seed: 0 },
  });
  assert.equal(
    prepareItemContent(JSON.stringify(source), "unregistered"),
    null,
  );
});

test("MMLU supports prompts without demonstrations and Windows line endings", () => {
  const source = {
    prompt: "Question?\r\nA. One\r\nB. Two\r\nC. Three\r\nD. Four\r\nAnswer:",
    candidate_continuations: ["A", "B", "C", "D"],
  };
  const result = prepareItemContent(JSON.stringify(source), "mmlu");
  assert.equal(result[0].text, "Question?");
  assert.equal(
    result.some((block) => block.kind === "details"),
    false,
  );
  assert.deepEqual(
    result[1].choices.map((c) => c.text),
    ["One", "Two", "Three", "Four"],
  );
});

test("ambiguous or incomplete questions keep all fields instead of guessing", () => {
  for (const prompt of [
    "Quotation\nA. Author's name\nQuestion?\nA. One\nB. Two\nC. Three\nD. Four\nAnswer:",
    "Question?\nA. One\nB. Two\nC. Three\nAnswer:",
    "Question?\nA. One\nB. Two\nC. Three\nD. Four\nAnswer: A",
  ]) {
    const source = { prompt, candidate_continuations: ["A", "B", "C", "D"] };
    assert.deepEqual(prepareItemContent(JSON.stringify(source), "mmlu"), [
      { kind: "fields", value: source },
    ]);
  }
});

test("general structured tasks preserve messages, nesting, and empty or false values", () => {
  const source = {
    messages: [
      { role: "user", content: "**Question**\n\n$x^2$", name: "Test" },
    ],
    options: ["Yes", "No"],
    environment: { count: 0, enabled: false, missing: null, empty: "" },
  };
  assert.deepEqual(prepareItemContent(JSON.stringify(source), "tau2_bench"), [
    { kind: "fields", value: source },
  ]);
  for (const text of [
    "# Question\n\nText",
    "{'prompt': 'Unparsed dictionary'}",
    '{"prompt":',
  ])
    assert.deepEqual(prepareItemContent(text, "tau2_bench"), [
      { kind: "text", text },
    ]);
});

function records() {
  return {
    subjects: [
      { subject_id: "a", display_name: "Same name" },
      { subject_id: "b", display_name: "Same name" },
      { subject_id: "empty", display_name: "Unobserved" },
    ],
    itemIds: ["i", "unanswered"],
    observations: ["a", "b"].flatMap((subject_id) =>
      Array.from({ length: 24 }, (_, i) => ({
        response_id: `${subject_id}-${i + 1}`,
        subject_id,
        item_id: "i",
        trial: i + 1,
        test_condition: "temperature=0",
        interactors: `opponent=${subject_id === "a" ? "b" : "a"}`,
        response: i === 0 ? null : subject_id === "a" ? 0.5 : 1,
      })),
    ),
  };
}

test("single and pairwise records retain all repeated and ungraded attempts", () => {
  const source = records();
  const before = structuredClone(source);
  const model = prepareMatrix(source);
  assert.equal(model.graded, 46);
  assert.equal(model.subjects.length, 3);
  assert.equal(model.itemIds.length, 2);
  for (const subject of source.subjects.slice(0, 2)) {
    const attempts = observationsFor(model, { subject, itemId: "i" });
    assert.equal(attempts.length, 24);
    assert.equal(attempts[0].response, null);
    assert.equal(attempts.at(-1).trial, 24);
  }
  assert.deepEqual(
    observationsFor(model, { subject: source.subjects[2], itemId: "i" }),
    [],
  );
  assert.deepEqual(source, before);
});

test("invalid foreign keys and duplicate identities fail visibly", () => {
  const source = records();
  source.observations[0].subject_id = "absent";
  assert.throws(() => prepareMatrix(source), /unknown model or task/);
  const duplicates = records();
  duplicates.observations.push(duplicates.observations[0]);
  assert.throws(() => prepareMatrix(duplicates), /Duplicate observation/);
});

test("model search keeps distinct configurations and pagination loses no records", () => {
  const source = records();
  assert.equal(findSubjects(source.subjects, " SAME ").length, 2);
  assert.equal(findSubjects(source.subjects, "empty")[0].subject_id, "empty");
  assert.deepEqual(
    Array.from({ length: 5 }, (_, page) =>
      pageOf(source.observations, page, 10),
    ).flat(),
    source.observations,
  );
});

test("declared labels and mixed item scales preserve native numeric values", () => {
  const binary = {
    kind: "discrete",
    values: [0, 1],
    meanings: { 0: "safe", 1: "unsafe" },
    direction: "lower_is_better",
  };
  assert.equal(gradeLabel(1, binary), "unsafe (1)");
  assert.equal(gradeLabel(null, binary), "Grade unavailable");
  assert.match(describeScale(binary), /Lower is better/);
  assert.equal(
    gradeLabel(0.10000000000000002, {
      kind: "discrete",
      values: [0, 0.1, 1],
      meanings: { ".1": "ignored", 0.1: "partial" },
    }),
    "partial (0.10000000000000002)",
  );
  const item = {
    item_id: "i",
    grading_criterion: JSON.stringify({
      reference_answer: null,
      rule: "rating",
      response_scale: { kind: "interval", min: 0, max: 100 },
    }),
    asset_manifest: null,
  };
  const prepared = prepareItem(item, { kind: "mixed" });
  assert.equal(gradeLabel(37.5, prepared.scale), "37.5");
  assert.deepEqual(prepared.attachments, []);
  assert.equal(
    describeScale({ kind: "discrete", values: [0, 1], direction: "unordered" }),
    "0 · 1 · Unordered categories",
  );
});

function transport() {
  const dataset = records();
  const manifest = {
    benchmark: {},
    tables: {
      subjects: { pages: 1 },
      items: { pages: 1 },
      responses: { pages: 2 },
      traces: { pages: 2 },
      assets: { pages: 1 },
    },
  };
  const files = {
    "subjects/0.json": Object.fromEntries(
      dataset.subjects.map((s) => [s.subject_id, s]),
    ),
    "items/index.json": { i: 0, unanswered: 0 },
    "items/0.json": {
      i: { item_id: "i", content: "Exact task" },
      unanswered: { item_id: "unanswered" },
    },
    "responses/0.json": Object.fromEntries(
      dataset.observations.slice(0, 24).map((r) => [r.response_id, r]),
    ),
    "responses/1.json": Object.fromEntries(
      dataset.observations.slice(24).map((r) => [r.response_id, r]),
    ),
    "traces/index.json": { "a-1": 1, "b-1": 0 },
    "traces/0.json": { "b-1": { response_id: "b-1", trace: "Answer B" } },
    "traces/1.json": { "a-1": { response_id: "a-1", trace: "Answer A" } },
    "assets/index.json": { image: 0 },
    "assets/0.json": { image: { asset_id: "image", data_url: "0.bin" } },
  };
  const calls = [];
  const fetch = async (url, options) => {
    options.signal.throwIfAborted();
    const path = url.replace("/records/", "");
    calls.push(path);
    if (path === "assets/0.bin") return new Response(new Uint8Array([1, 2, 3]));
    return Object.hasOwn(files, path)
      ? Response.json(files[path])
      : new Response(null, { status: 404 });
  };
  return {
    dataset,
    manifest,
    files,
    calls,
    fetch,
    source: new RecordSource("/records", manifest, fetch),
  };
}

test("initial load fetches axes and observations; content and answers remain lazy", async () => {
  const fixture = transport();
  assert.deepEqual(await fixture.source.load(signal()), fixture.dataset);
  assert.equal(
    fixture.calls.some((path) => path.startsWith("traces/")),
    false,
  );
  assert.equal(fixture.calls.includes("items/0.json"), false);
  assert.equal(
    (await fixture.source.get("items", "i", signal())).content,
    "Exact task",
  );
  assert.equal(
    (await fixture.source.get("traces", "a-1", signal())).trace,
    "Answer A",
  );
  assert.equal(
    (await fixture.source.get("traces", "b-1", signal())).trace,
    "Answer B",
  );
  assert.equal(await fixture.source.get("traces", "a-2", signal()), null);
  const calls = fixture.calls.length;
  await fixture.source.get("traces", "a-1", signal());
  assert.equal(fixture.calls.length, calls);
});

test("default transport does not bind browser fetch to the record source", async (t) => {
  const fixture = transport();
  t.mock.method(globalThis, "fetch", function (...args) {
    assert.equal(this, undefined);
    return fixture.fetch(...args);
  });
  const source = new RecordSource("/records", fixture.manifest);
  assert.deepEqual(await source.load(signal()), fixture.dataset);
});

test("subject pages fetch only needed responses, retaining all attempts and unobserved axes", async () => {
  const f = transport();
  f.manifest.tables.responses.bySubject = "subjects.json";
  f.files["responses/subjects.json"] = { a: [0], b: [1] };
  // A physical page may straddle two subjects. Filter it without changing records.
  f.files["responses/0.json"]["b-1"] = f.dataset.observations[24];
  const a = await f.source.load(signal(), ["a"]);
  assert.deepEqual(a.observations, f.dataset.observations.slice(0, 24));
  assert.deepEqual(a.itemIds, f.dataset.itemIds);
  assert.deepEqual(a.allSubjects, f.dataset.subjects);
  assert.equal(f.calls.includes("responses/1.json"), false);
  const b = await f.source.load(signal(), ["b"]);
  assert.deepEqual(b.observations, f.dataset.observations.slice(24));
  const empty = await f.source.load(signal(), ["empty"]);
  assert.equal(empty.observations.length, 0);
  assert.equal(empty.subjects[0].subject_id, "empty");
  assert.deepEqual(empty.itemIds, f.dataset.itemIds);
});

test("prepared gzip transport keeps the same records and exact lazy ID lookup", async () => {
  const f = transport();
  f.manifest.compressed = true;
  const source = new RecordSource("/records", f.manifest, (url, options) => {
    assert.ok(url.endsWith(".json.gz"));
    return f.fetch(url.slice(0, -3), options);
  });
  assert.deepEqual(await source.load(signal()), f.dataset);
  assert.equal((await source.get("traces", "a-1", signal())).trace, "Answer A");
});

test("missing optional tables remain empty, but failed reads are errors", async () => {
  const fixture = transport();
  delete fixture.manifest.tables.responses;
  delete fixture.manifest.tables.traces;
  assert.equal((await fixture.source.load(signal())).observations.length, 0);
  assert.equal(await fixture.source.get("traces", "a-1", signal()), null);
  delete fixture.files["items/0.json"];
  await assert.rejects(fixture.source.get("items", "i", signal()), /404/);
  fixture.files["items/0.json"] = { i: { item_id: "i" } };
  assert.equal((await fixture.source.get("items", "i", signal())).item_id, "i");
});

test("aborting a selection does not leave a failed request in the cache", async () => {
  const fixture = transport();
  const controller = new AbortController();
  let cancel = true;
  const source = new RecordSource(
    "/records",
    fixture.manifest,
    async (url, options) => {
      const response = await fixture.fetch(url, options);
      if (cancel) {
        cancel = false;
        controller.abort();
      }
      return response;
    },
  );
  await assert.rejects(source.get("items", "i", controller.signal), {
    name: "AbortError",
  });
  assert.equal(
    (await source.get("items", "i", signal())).content,
    "Exact task",
  );
});

test("attachments retain bytes and declared media type", async () => {
  const { source } = transport();
  const result = await source.asset(
    {
      asset_id: "image",
      path: "image.png",
      media_type: "image/png",
      role: "input",
    },
    signal(),
  );
  assert.equal(result.url, "/records/assets/0.bin");
  assert.equal(result.blob.type, "image/png");
  assert.deepEqual(
    new Uint8Array(await result.blob.arrayBuffer()),
    new Uint8Array([1, 2, 3]),
  );
});

test("page cards use exported counts while retaining curated artwork", () => {
  const entry = {
    manifest: {
      benchmark: {
        benchmark_id: "b",
        name: "New name",
        description: "Description",
        domain: ["safety"],
        granularity: "item",
      },
      tables: {
        subjects: { rows: 3 },
        items: { rows: 14 },
        responses: { rows: 48 },
      },
    },
  };
  const card = recordCard(entry, {
    name: "Old name",
    image: "/logo.svg",
    fit: "contain",
  });
  assert.equal(card.name, "New name");
  assert.equal(card.models, 3);
  assert.equal(card.responses, 48);
  assert.equal(card.image, "/logo.svg");
});

const contextModule = await loadModule("context");
const { prepareContexts, parseFields } = contextModule.exports;
const { prepareView } = (
  await loadModule("view", { model: modelModule, context: contextModule })
).exports;
const { gradeColor, EMPTY, UNGRADED } = (await loadModule("colors")).exports;
const { prepareComparisons } = (
  await loadModule("comparisons", { context: contextModule })
).exports;
const { displayText, presentationFor } = (await loadModule("presentation"))
  .exports;
const { normalizeMath } = (await loadModule("text")).exports;
const orderedScale = {
  kind: "discrete",
  values: [0, 0.5, 1],
  direction: "higher_is_better",
};

test("filters, axes, means, and ordering describe exactly the selected observations", () => {
  const original = records(),
    before = structuredClone(original),
    model = prepareMatrix(original),
    contexts = prepareContexts(original.observations);
  const all = prepareView(model, contexts, {}, "", "grade", orderedScale);
  assert.deepEqual(
    all.subjects.map((s) => s.subject_id),
    ["a", "b", "empty"],
  );
  assert.equal(all.rowMeans.get("a"), 0.5);
  assert.deepEqual(all.itemIds, ["i", "unanswered"]);
  const filtered = prepareView(
    model,
    contexts,
    { trial: JSON.stringify("2") },
    "",
    "grade",
    orderedScale,
  );
  assert.equal(filtered.observations.length, 2);
  assert.equal(filtered.graded, 2);
  assert.deepEqual(filtered.itemIds, ["i"]);
  const search = prepareView(
    model,
    contexts,
    {},
    "empty",
    "grade",
    orderedScale,
  );
  assert.equal(search.observations.length, 0);
  assert.equal(search.graded, 0);
  assert.deepEqual(search.itemIds, ["i", "unanswered"]);
  const lower = prepareView(model, contexts, {}, "", "grade", {
    ...orderedScale,
    direction: "lower_is_better",
  });
  assert.deepEqual(
    lower.subjects.map((s) => s.subject_id),
    ["b", "a", "empty"],
  );
  assert.deepEqual(original, before);
});

test("mixed and unordered scales do not produce cross-item means or grade ordering", () => {
  const model = prepareMatrix(records()),
    contexts = prepareContexts(model.observations);
  for (const scale of [
    { kind: "mixed" },
    { ...orderedScale, direction: "unordered" },
  ]) {
    const view = prepareView(model, contexts, {}, "", "grade", scale);
    assert.equal(view.rowMeans.size, 0);
    assert.equal(view.columnMeans.size, 0);
  }
});

test("JSON and key=value contexts stay distinct from subject metadata", () => {
  assert.deepEqual(parseFields("setting=a=b;mode=fast"), {
    setting: "a=b",
    mode: "fast",
  });
  assert.deepEqual(parseFields('{"axis":"accuracy","source_row":0}'), {
    axis: "accuracy",
    source_row: "0",
  });
  const source = records();
  source.observations[0].test_condition = null;
  const dimensions = prepareContexts(source.observations).dimensions;
  assert.ok(
    dimensions
      .find((d) => d.key === "condition.temperature")
      .values.includes(null),
  );
  assert.ok(dimensions.find((d) => d.key === "interactor.opponent"));
  assert.equal(
    dimensions.some((d) => d.key.includes("display_name")),
    false,
  );
});

test("colors use declared direction and keep null, absent, and midpoint distinct", () => {
  assert.equal(gradeColor(null, orderedScale), UNGRADED);
  assert.notEqual(gradeColor(0.5, orderedScale), EMPTY);
  assert.notEqual(gradeColor(0.5, orderedScale), UNGRADED);
  assert.equal(
    gradeColor(0, { ...orderedScale, direction: "lower_is_better" }),
    gradeColor(1, orderedScale),
  );
  const categorical = {
    kind: "discrete",
    values: [0, 1, 2],
    direction: "unordered",
  };
  assert.equal(
    new Set(categorical.values.map((value) => gradeColor(value, categorical)))
      .size,
    3,
  );
  assert.equal(
    gradeColor(100, { kind: "interval", min: 0, max: null }),
    gradeColor(1000, { kind: "interval", min: 0, max: null }),
  );
});

test("comparison links require a unique reciprocal context across all trials", () => {
  const repeated = prepareMatrix(records());
  assert.equal(
    prepareComparisons(repeated, {}).size,
    0,
    "trial counters alone cannot pair repetitions",
  );
  const observations = repeated.observations.filter((r) => r.trial === 2);
  const single = prepareMatrix({ ...repeated, observations });
  const pairs = prepareComparisons(single, {});
  assert.equal(pairs.get("a-2").response_id, "b-2");
  assert.equal(pairs.get("b-2").response_id, "a-2");
  const duplicate = { ...observations[0], response_id: "duplicate", trial: 3 };
  assert.equal(
    prepareComparisons(
      prepareMatrix({
        ...repeated,
        observations: [...observations, duplicate],
      }),
      {},
    ).size,
    0,
  );
  const differentTrials = [{ ...observations[0], trial: 3 }, observations[1]];
  assert.equal(
    prepareComparisons(
      prepareMatrix({ ...repeated, observations: differentTrials }),
      {},
    ).size,
    0,
  );
});

test("source-specific comparison and text mappings are explicit and preserve original payloads", () => {
  const source = records();
  source.observations = source.observations
    .slice(1, 2)
    .concat(source.observations.slice(25, 26));
  for (const [i, subject] of source.subjects.slice(0, 2).entries())
    subject.subject_features_extra = `source_provider=provider${i}`;
  source.observations.forEach((record, i) => {
    record.interactors = `opponent=provider${1 - i}`;
    record.test_condition = JSON.stringify({
      axis: "accuracy",
      render_mode: "qa_text_only",
      source_row: 12,
      slot: i ? "slot_b_provider" : "slot_a_provider",
    });
  });
  const presentation = presentationFor("real_pocqi");
  const pairs = prepareComparisons(prepareMatrix(source), presentation);
  assert.equal(pairs.get("a-2").response_id, "b-2");
  const text = JSON.stringify({
    native_answer: { answer_markdown: "**Answer**" },
    private_example: "still in original",
  });
  assert.equal(displayText(text, presentation.answerPath), "**Answer**");
  assert.equal(displayText(text, ["absent"]), text);
  assert.equal(
    displayText("plain text", presentation.answerPath),
    "plain text",
  );
});

test("math formatting preserves currency, code, and raw source text", () => {
  assert.equal(normalizeMath("Cost $5 and $10 more."), "Cost $5 and $10 more.");
  assert.equal(normalizeMath("Solve $x+1$."), "Solve $$x+1$$.");
  assert.equal(normalizeMath("`$x$` and $y$"), "`$x$` and $$y$$");
  assert.equal(normalizeMath(String.raw`\[x^2\]`), "\n$$\nx^2\n$$\n");
  assert.equal(normalizeMath("$$x$$"), "$$x$$");
});

test("comparison identity preserves JSON value types", () => {
  const source = records();
  source.observations = source.observations
    .slice(1, 2)
    .concat(source.observations.slice(25, 26));
  source.observations[0].test_condition = '{"id":1}';
  source.observations[1].test_condition = '{"id":"1"}';
  assert.equal(prepareComparisons(prepareMatrix(source), {}).size, 0);
});

test("an undeclared value is never relabeled as the nearest category", () => {
  assert.equal(
    gradeLabel(2, {
      kind: "discrete",
      values: [0, 1],
      meanings: { 0: "unsafe", 1: "safe" },
    }),
    "2",
  );
});

test("unregistered, removed, or failed customizations fall back to the original text", async () => {
  const { presentations } =
    await import("../lib/presentation-transforms/index.ts");
  const text = '  {"prompt":"<script>never execute</script>","empty":""}\n';
  assert.equal(prepareItemContent(text, "new-benchmark"), null);
  const saved = presentations.aegis;
  try {
    delete presentations.aegis;
    assert.equal(prepareItemContent(text, "aegis"), null);
    presentations.aegis = {
      item: () => {
        throw new Error("Incompatible source");
      },
    };
    assert.equal(prepareItemContent(text, "aegis"), null);
  } finally {
    presentations.aegis = saved;
  }
});

test("registered benchmark modules cover the published catalog", async () => {
  const { readFile } = await import("node:fs/promises");
  const { presentations } =
    await import("../lib/presentation-transforms/index.ts");
  const cards = JSON.parse(
    await readFile(
      new URL("../content/generated/benchmark-cards.json", import.meta.url),
      "utf8",
    ),
  );
  for (const card of cards)
    assert.equal(typeof presentations[card.slug]?.item, "function", card.slug);
});

test("question options and continuation lists retain variable counts and extra fields", () => {
  for (const size of [1, 4, 10, 13]) {
    const options = Array.from({ length: size }, (_, i) => `Option ${i}`);
    const question = {
      question: "Question?",
      options,
      extra: { enabled: false },
    };
    const blocks = prepareItemContent(JSON.stringify(question), "mmlupro");
    assert.deepEqual(
      blocks[1].choices.map((c) => c.text),
      options,
    );
    assert.deepEqual(blocks[2].value, { extra: { enabled: false } });
    const bundle = {
      prompt: "Full prompt including examples",
      candidate_continuations: options,
      request_protocol: "Score independently",
    };
    const continuations = prepareItemContent(
      JSON.stringify(bundle),
      "truthfulqa_mc",
    );
    assert.equal(continuations[0].text, bundle.prompt);
    assert.deepEqual(
      continuations[1].choices.map((c) => c.text),
      options,
    );
    assert.deepEqual(continuations[2].value, {
      request_protocol: bundle.request_protocol,
    });
  }
});

test("multimedia formatting preserves block order and unknown fields", () => {
  const source = {
    multimedia_elements: [
      { content_type: "text/plain", text: "A)", extra: 0 },
      { content_type: "image/png", location: "a.png", source: "released" },
      { content_type: "text/plain", text: "B)" },
      { content_type: "image/png", location: "b.png" },
    ],
    metadata: { original: true },
  };
  const blocks = prepareItemContent(JSON.stringify(source), "agc_bench");
  assert.deepEqual(
    blocks.map((b) => b.kind),
    ["text", "fields", "media", "fields", "text", "media", "fields"],
  );
  assert.deepEqual(
    blocks.filter((b) => b.kind === "media").map((b) => b.path),
    ["a.png", "b.png"],
  );
  assert.deepEqual(blocks.at(-1).value, { metadata: source.metadata });
  const unusual = {
    multimedia_elements: [{ content_type: "unknown", missing: true }],
  };
  assert.deepEqual(prepareItemContent(JSON.stringify(unusual), "wildvision"), [
    { kind: "fields", value: unusual },
  ]);
});

test("sectioned context keeps simulator roles, camera data, and unfamiliar fields", () => {
  const source = {
    task: { user_scenario: { instructions: "Ask for help" } },
    environment_info: { policy: "Original policy", enabled: false },
    user_guidelines: "Simulate the user",
    future_field: [0, null],
  };
  const blocks = prepareItemContent(JSON.stringify(source), "tau2_bench");
  assert.deepEqual(blocks[0].value, source.task);
  assert.deepEqual(blocks[1].blocks[0].value, source.environment_info);
  assert.equal(blocks[2].blocks[0].value, source.user_guidelines);
  assert.deepEqual(blocks[3].value, { future_field: source.future_field });
  const capture = {
    capture: "scene",
    cameras: {
      train: [
        {
          matrix: [
            [0, 1],
            [1, 0],
          ],
        },
      ],
    },
    input_scope: "Released capture context",
  };
  const orb = prepareItemContent(JSON.stringify(capture), "stanford_orb");
  assert.equal(orb[0].value, capture.capture);
  assert.deepEqual(orb.at(-1).blocks[0].value, capture.cameras);
});

test("item windows load only the selected page and preserve complete axes and partial-item labels", async () => {
  const calls = [];
  const pages = {
    "subjects/0.json": { s: { subject_id: "s", display_name: "Single model" } },
    "windows/8.json": {
      itemIds: ["i"],
      observations: [],
      partialItemIds: ["i"],
    },
  };
  const source = new RecordSource(
    "/records",
    { tables: { subjects: { pages: 1 } }, windows: { pages: 1000 } },
    async (url) => {
      calls.push(url);
      const value = pages[url.slice("/records/".length)];
      assert.ok(value, `Unexpected download: ${url}`);
      return new Response(JSON.stringify(value));
    },
  );
  const result = await source.window(8, signal());
  assert.deepEqual(result.itemIds, ["i"]);
  assert.deepEqual(result.partialItemIds, ["i"]);
  assert.equal(result.subjects.length, 1);
  assert.deepEqual(
    new Set(calls),
    new Set(["/records/subjects/0.json", "/records/windows/8.json"]),
  );
});

test("partial items cannot create a false reciprocal comparison", async () => {
  const { prepareComparisons } = (await loadModule("comparisons")).exports;
  const model = prepareMatrix({
    subjects: [
      { subject_id: "a", display_name: "A" },
      { subject_id: "b", display_name: "B" },
    ],
    itemIds: ["i"],
    partialItemIds: ["i"],
    observations: [
      {
        response_id: "a",
        subject_id: "a",
        item_id: "i",
        trial: 1,
        test_condition: null,
        interactors: "opponent=b",
        response: 1,
      },
      {
        response_id: "b",
        subject_id: "b",
        item_id: "i",
        trial: 1,
        test_condition: null,
        interactors: "opponent=a",
        response: 0,
      },
    ],
  });
  assert.equal(prepareComparisons(model, {}).size, 0);
  assert.equal(
    prepareComparisons({ ...model, partialItemIds: [] }, {}).size,
    2,
  );
});

test("math remains math while fenced and tagged Python retain source whitespace", () => {
  const python = "def f(x):\n    return '$x$'\n";
  assert.equal(
    normalizeMath("~~~python\n" + python + "~~~\nThen $x^2$."),
    "~~~python\n" + python + "~~~\nThen $$x^2$$.",
  );
  assert.equal(normalizeMath("```python\n" + python), "```python\n" + python);
  assert.equal(
    normalizeMath(String.raw`Use \(\frac{1}{2}\) and \[x^2\].`),
    "Use $$\\frac{1}{2}$$ and \n$$\nx^2\n$$\n.",
  );
  const blocks = prepareItemContent(
    "Instructions\n[PYTHON]" + python + "[/PYTHON]\nMore",
    "cruxeval",
  );
  assert.equal(blocks[1].kind, "code");
  assert.equal(blocks[1].text, python);
  assert.equal(blocks[1].language, "python");
  assert.equal(prepareItemContent(python, "kernelbench")[0].text, python);
  const files = prepareItemContent(
    JSON.stringify({
      instructions: "Implement this.",
      initial_files: { "main.py": python },
      extra: false,
    }),
    "aider",
  );
  assert.equal(files[1].text, python);
  assert.equal(files[1].language, "python");
  assert.deepEqual(files[2].value, { extra: false });
});

test("bare balanced math commands render without changing commands inside longer code fences", () => {
  assert.equal(
    normalizeMath(String.raw`Use \boxed{A} or \frac{1}{\sqrt{2}}.`),
    String.raw`Use $$\boxed{A}$$ or $$\frac{1}{\sqrt{2}}$$.`,
  );
  assert.equal(
    normalizeMath(String.raw`Incomplete \frac{1} and \boxed{a`),
    String.raw`Incomplete \frac{1} and \boxed{a`,
  );
  assert.equal(
    normalizeMath("````python\n```\n$literal$\n\\boxed{A}\n````\n$x$"),
    "````python\n```\n$literal$\n\\boxed{A}\n````\n$$x$$",
  );
  assert.equal(
    normalizeMath("``$code$ ` still code`` and $x$"),
    "``$code$ ` still code`` and $$x$$",
  );
});

test("source page limits reject large downloads and decompression before allocation", async () => {
  const { budgetSource } = (await loadModule("source-budget")).exports;
  let reads = 0,
    decodes = 0;
  const source = budgetSource(
    {
      byteLength: 10000,
      slice(start, end) {
        reads++;
        return new ArrayBuffer(end - start);
      },
    },
    {
      GZIP(input, length) {
        decodes++;
        return new Uint8Array(length);
      },
    },
    100,
  );
  assert.equal(source.file.slice(0, 30).byteLength, 30);
  assert.throws(
    () => source.compressors.GZIP(new Uint8Array(1), 1000),
    /preview limit/,
  );
  assert.equal(decodes, 0);
  assert.throws(() => source.file.slice(0, 10000), /preview limit/);
  assert.equal(reads, 1);
});
