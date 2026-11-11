import assert from "node:assert/strict";
import { displayTrace, TESTS_MAX } from "../lib/traces/display.ts";

// Long fields are capped for display, and the original length is recorded.
const shown = displayTrace(
  {
    schema: "trace/1",
    converter: "fixture@1",
    native: { big: "x".repeat(100) },
    steps: [
      { type: "message", label: "task", content: "short" },
      {
        type: "llm",
        content: "a".repeat(30),
        toolCalls: [{ id: "c1", arguments: "b".repeat(30) }],
        native: { id: 1 },
      },
      { type: "tool", callId: "c1", output: "測".repeat(30), exitCode: 0 },
    ],
    evaluation: {
      tests: Array.from({ length: TESTS_MAX + 2 }, (_, i) => ({
        name: `t${i}`,
        status: i < 3 ? "failure" : "passed",
      })),
    },
  },
  10,
);
assert.equal(shown.native, undefined, "native is not sent to the page");
assert.equal(shown.steps[1].native, undefined);
assert.equal(shown.steps[0].content, "short");
assert.equal(shown.steps[0].truncated, undefined);
assert.equal(shown.steps[1].content, "a".repeat(10));
assert.deepEqual(shown.steps[1].truncated, { content: 30, arguments: 30 });
assert.equal(shown.steps[1].toolCalls[0].arguments, "b".repeat(10));
assert.equal(shown.steps[2].output, "測".repeat(10));
assert.deepEqual(shown.steps[2].truncated, { output: 30 });
// The test list keeps non-passed tests first and records the full count.
assert.equal(shown.evaluation.tests.length, TESTS_MAX);
assert.equal(shown.evaluation.testsTotal, TESTS_MAX + 2);
assert.deepEqual(
  shown.evaluation.tests.slice(0, 3).map((t) => t.status),
  ["failure", "failure", "failure"],
);
// Counts per status cover every test, including the ones dropped from the list.
assert.deepEqual(shown.evaluation.testCounts, {
  failure: 3,
  passed: TESTS_MAX - 1,
});

// Anything that is not a trace/1 value is rejected, so the worker falls back to raw text.
for (const value of [
  null,
  "text",
  { schema: "trace/2", steps: [] },
  { schema: "trace/1" },
])
  assert.throws(() => displayTrace(value));

console.log("trace/1 display caps, test ordering and rejection passed.");
