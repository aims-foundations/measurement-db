import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

async function loadModule(name, dependencies = {}) {
  let code = stripTypeScriptTypes(
    await readFile(
      new URL(`../lib/benchmark-records/${name}.ts`, import.meta.url),
      "utf8",
    ),
    { mode: "transform" },
  );
  for (const [name, module] of Object.entries(dependencies))
    code = code.replaceAll(`"./${name}"`, JSON.stringify(module.url));
  const url = `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
  return { url, exports: await import(url) };
}
const scaleModule = await loadModule("scale");
const { gradeLabel, describeScale } = scaleModule.exports;
const modelModule = await loadModule("model", { scale: scaleModule });
const { prepareMatrix, observationsFor, findSubjects, pageOf, prepareItem } =
  modelModule.exports;
const { RecordSource } = (await loadModule("client")).exports;
const { recordCard } = (await loadModule("catalog")).exports;
const signal = () => new AbortController().signal;

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
  assert.equal(result.image.type, "image/png");
  assert.deepEqual(
    new Uint8Array(await result.image.arrayBuffer()),
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
