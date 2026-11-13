// General normalized schema design; independent of any benchmark export.

(() => {
  const root = document.getElementById("mdb-schema-expanded");
  const tables = {
    benchmarks: {
      fields:
        "benchmark_id! name version? license source_url description one_line_description? modality:list<string> domain:list<string> multi_single_turn response_type response_scale categorical:bool paper_url? release_date? granularity release? n_response_values:int64? benchmark_features? n_subjects:int64 n_items:int64 n_responses:int64 max_trial:int64 coverage:float64 has_reference_answer:bool",
      preview: [
        "benchmark_id",
        "name",
        "version",
        "domain",
        "modality",
        "response_scale",
      ],
      refs: {},
    },
    contents: {
      fields: "content_id! media_type text? asset_id?",
      preview: ["content_id", "media_type", "text", "asset_id"],
      refs: { asset_id: ["assets", "asset_id"] },
    },
    verifiers: {
      fields:
        "verifier_id! verifier_type name? version? instructions_content_id? implementation_content_id? configuration? provenance?",
      preview: [
        "verifier_id",
        "verifier_type",
        "name",
        "version",
        "configuration",
      ],
      refs: {
        instructions_content_id: ["contents", "content_id"],
        implementation_content_id: ["contents", "content_id"],
      },
    },
    subjects: {
      fields:
        "subject_id! display_name normalized_name? provider? release_date? access_date? harness? reasoning_effort? harness_version? subject_features_extra?",
      preview: [
        "subject_id",
        "display_name",
        "provider",
        "harness",
        "reasoning_effort",
      ],
      refs: {},
    },
    items: {
      fields:
        "item_id! benchmark_id raw_item_id? content_id? asset_manifest? item_features? testlet_id? provenance?",
      preview: [
        "item_id",
        "benchmark_id",
        "content_id",
        "raw_item_id",
        "testlet_id",
        "item_features",
      ],
      refs: {
        benchmark_id: ["benchmarks", "benchmark_id"],
        content_id: ["contents", "content_id"],
      },
      notes: {
        asset_manifest:
          "JSON attachment links to assets.asset_id, when present.",
      },
    },
    checks: {
      fields:
        "check_id! item_id verifier_id raw_check_id? criterion_content_id? reference_content_id? parameters? response_scale check_features? provenance?",
      preview: [
        "check_id",
        "item_id",
        "verifier_id",
        "criterion_content_id",
        "parameters",
        "response_scale",
      ],
      refs: {
        item_id: ["items", "item_id"],
        verifier_id: ["verifiers", "verifier_id"],
        criterion_content_id: ["contents", "content_id"],
        reference_content_id: ["contents", "content_id"],
      },
    },
    attempts: {
      fields:
        "attempt_id! benchmark_id subject_id item_id? raw_attempt_id? trace? provenance?",
      preview: ["attempt_id", "benchmark_id", "subject_id", "item_id", "trace"],
      refs: {
        benchmark_id: ["benchmarks", "benchmark_id"],
        subject_id: ["subjects", "subject_id"],
        item_id: ["items", "item_id"],
      },
    },
    responses: {
      fields:
        "response_id! benchmark_id subject_id check_id attempt_id? trial:int64 test_condition? interactors? response:float64? rater_id? grading_evidence_content_id? provenance?",
      preview: [
        "response_id",
        "benchmark_id",
        "subject_id",
        "check_id",
        "attempt_id",
        "response",
        "trial",
      ],
      refs: {
        benchmark_id: ["benchmarks", "benchmark_id"],
        subject_id: ["subjects", "subject_id"],
        check_id: ["checks", "check_id"],
        attempt_id: ["attempts", "attempt_id"],
        grading_evidence_content_id: ["contents", "content_id"],
      },
    },
    assets: {
      fields: "asset_id! benchmark_id byte_size:int64 data:binary",
      refs: { benchmark_id: ["benchmarks", "benchmark_id"] },
    },
  };
  tables.assets.preview = ["asset_id", "benchmark_id", "byte_size", "data"];
  tables.items.preview.push("asset_manifest");
  tables.items.refs.asset_manifest = [
    "assets",
    "asset_id",
    "Nested JSON attachment references, not a scalar foreign key.",
  ];
  // Show the logical schema independently of any benchmark's exported data.
  const descriptions = {
    benchmarks:
      "Benchmark identity, version, licensing, provenance, domain, modality and summary metadata.",
    assets:
      "Exact bytes of external files, such as images, documents, audio or code. Contents can reference an asset directly; items can reference multiple assets through asset_manifest.",
    contents:
      "Reusable text, structured content or an asset reference. The content ID identifies the payload independently of the items or grading definitions that use it.",
    items:
      "Task identity, its benchmark, content, features and attachment references. testlet_id optionally groups related tasks.",
    verifiers:
      "The grading implementation: program, LLM or human, with its version, configuration, instruction or implementation references, and provenance.",
    checks:
      "Connects an item to a verifier and a grading criterion or reference, including parameters and the applicable score scale.",
    subjects:
      "The evaluated systems or participants, their identities, and relevant evaluation configurations.",
    attempts:
      "A generated answer or execution, including its trace, stored once even when multiple checks assess it.",
    responses:
      "Individual recorded grades linked to checks and subjects, with an optional execution link. Repeated assessments, conditions and missing grades are preserved.",
  };
  for (const [name, t] of Object.entries(tables)) {
    t.status = "";
    t.description = descriptions[name];
  }
  const names = Object.keys(tables);
  const edges = names.flatMap((child) =>
    Object.entries(tables[child].refs).map(([field, [parent]]) => {
      const declaration = tables[child].fields
        .split(" ")
        .find((raw) => raw.replace(/[!?]/g, "").split(":")[0] === field);
      return [
        parent,
        child,
        field,
        declaration?.includes("?") ? "optional" : "required",
      ];
    }),
  );
  // Kahn topological sort uses every declared relationship, independent of data.
  const children = Object.fromEntries(names.map((n) => [n, new Set()])),
    indegree = Object.fromEntries(names.map((n) => [n, 0]));
  for (const child of names)
    for (const [parent] of Object.values(tables[child].refs))
      if (names.includes(parent) && !children[parent].has(child)) {
        children[parent].add(child);
        indegree[child]++;
      }
  const queue = names.filter((n) => indegree[n] === 0),
    topological = [];
  while (queue.length) {
    const parent = queue.shift();
    topological.push(parent);
    for (const child of children[parent])
      if (--indegree[child] === 0) queue.push(child);
  }
  if (topological.length !== names.length)
    throw Error(
      "Schema dependencies contain a cycle; a topological layout is not possible.",
    );
  const distance = {};
  for (const n of [...topological].reverse())
    distance[n] = Math.max(0, ...[...children[n]].map((c) => distance[c] + 1));
  const maxDistance = Math.max(...Object.values(distance)),
    rank = Object.fromEntries(names.map((n) => [n, maxDistance - distance[n]]));
  // The first three ranks form a compact, internally ordered definition group.
  const column = Object.fromEntries(
    names.map((n) => [n, Math.max(0, rank[n] - 2)]),
  );
  const row = {
    benchmarks: 0,
    assets: 1,
    contents: 2,
    verifiers: 0,
    items: 1,
    subjects: 2,
    checks: 0,
    attempts: 2,
    responses: 1,
  };
  names.sort((a, b) => rank[a] - rank[b] || row[a] - row[b]);
  root.dataset.layers = JSON.stringify(
    Object.fromEntries(names.map((n) => [n, rank[n]])),
  );
  const parse = (raw) => ({
    name: raw.replace(/[!?]/g, "").split(":")[0],
    type: raw.includes(":") ? raw.replace("?", "").split(":")[1] : "string",
    nullable: raw.includes("?"),
    pk: raw.includes("!"),
  });
  const escape = (s) =>
    String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  const state = {
    selected: "assets",
    fields: "keys",
    focus: false,
    expanded: [],
  };
  const map = root.querySelector(".s-map"),
    grid = root.querySelector(".s-nodes"),
    svg = root.querySelector(".s-wires"),
    detail = root.querySelector(".s-detail");
  function showFields(name) {
    const t = tables[name],
      all = t.fields.split(" ").map(parse);
    if (state.fields === "all" || state.expanded.includes(name)) return all;
    if (state.fields === "keys")
      return all.filter(
        (f) => t.refs[f.name] || (f.pk && edges.some((e) => e[0] === name)),
      );
    return [...new Set([...t.preview, ...Object.keys(t.refs)])].map((n) =>
      all.find((f) => f.name === n),
    );
  }
  function renderNodes() {
    grid.innerHTML = names
      .map((name) => {
        const t = tables[name],
          fields = showFields(name),
          remaining = t.fields.split(" ").length - fields.length;
        return `<article class="s-node ${state.selected === name ? "is-selected" : ""}" data-table="${name}"><button type="button" class="s-node-head cursor-interaction" data-inspect="${name}" aria-pressed="${state.selected === name}"><span class="s-node-name">${name}</span><span class="s-count">${t.status}</span></button><div class="s-fields">${fields.map((f) => `<div class="s-field" data-column="${f.name}"><span class="s-key ${f.pk ? "pk" : t.refs[f.name] ? "fk" : ""}">${f.pk ? "PK" : t.refs[f.name] ? "FK" : "·"}</span><span class="s-field-name">${f.name}</span></div>`).join("")}</div>${state.fields !== "all" ? `<button type="button" class="s-more cursor-interaction" data-expand="${name}">${state.expanded.includes(name) ? "Collapse fields" : remaining ? "Show all fields" : "All fields shown"}</button>` : ""}</article>`;
      })
      .join("");
    grid.querySelectorAll("[data-expand]").forEach((b) => {
      if (
        !state.expanded.includes(b.dataset.expand) &&
        showFields(b.dataset.expand).length ===
          tables[b.dataset.expand].fields.split(" ").length
      ) {
        b.disabled = true;
        return;
      }
      b.addEventListener("click", () => {
        const name = b.dataset.expand;
        state.expanded = state.expanded.includes(name)
          ? state.expanded.filter((n) => n !== name)
          : [...state.expanded, name];
        renderNodes();
        wireInspect();
      });
    });
    grid.querySelectorAll("[data-table]").forEach((n) => {
      n.style.setProperty("--s-rank", column[n.dataset.table] + 1);
      n.style.setProperty("--s-order", row[n.dataset.table] + 2);
      n.dataset.rank = rank[n.dataset.table];
      n.querySelector(".s-count").textContent = tables[n.dataset.table].status;
      n.querySelectorAll("[data-column=asset_manifest] .s-key").forEach(
        (k) => (k.textContent = "JSON"),
      );
    });
    grid.insertAdjacentHTML(
      "afterbegin",
      ["Definitions", "Tasks & actors", "Checks & runs", "Grades"]
        .map(
          (label, i) =>
            `<div class="s-stage-title" style="grid-column:${i + 1}">${label}</div>`,
        )
        .join(""),
    );
    wireInspect();
    scheduleDraw();
  }
  function wireInspect() {
    root.querySelectorAll("[data-inspect]").forEach((b) => {
      b.onclick = () => select(b.dataset.inspect);
    });
  }
  function select(name) {
    state.selected = name;
    root
      .querySelectorAll("[data-inspect]")
      .forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.inspect === name)),
      );
    grid
      .querySelectorAll(".s-node")
      .forEach((n) =>
        n.classList.toggle("is-selected", n.dataset.table === name),
      );
    renderDetail();
    scheduleDraw();
  }
  function renderDetail() {
    const name = state.selected,
      t = tables[name],
      related = edges.filter((e) => e[0] === name || e[1] === name);
    detail.innerHTML = `<div class="s-detail-head"><h3>${name}.parquet</h3><span class="s-count">${t.status}</span></div><p>${t.description}</p><div class="s-relations">${related.map(([parent, child, key]) => `<button type="button" class="s-join-button cursor-interaction" data-go="${parent === name ? child : parent}">${child}.${key} → ${parent}</button>`).join("")}</div><details><summary class="cursor-interaction">Column types, nullability and references</summary><table><thead><tr><th>Column</th><th>Type</th><th>Reference / key</th></tr></thead><tbody>${t.fields
      .split(" ")
      .map((raw) => {
        const f = parse(raw),
          ref = t.refs[f.name];
        return `<tr><td><span class="s-field-name">${f.name}</span>${f.nullable ? "<small>Nullable</small>" : ""}</td><td>${escape(f.type)}</td><td>${f.pk ? '<span class="s-key pk">PK</span> ' : ""}${ref ? `<button type="button" class="s-ref cursor-interaction" data-go="${ref[0]}">${ref[0]}.${ref[1]}</button>${ref[2] ? `<small>${ref[2]}</small>` : ""}` : f.pk ? "" : "—"}${t.notes?.[f.name] ? `<small>${t.notes[f.name]}</small>` : ""}</td></tr>`;
      })
      .join("")}</tbody></table></details>`;
    detail.querySelector(".s-count").textContent = t.status;
    detail
      .querySelectorAll("[data-go]")
      .forEach((b) => (b.onclick = () => select(b.dataset.go)));
  }
  // Orthogonal routing uses free lanes around the measured table rectangles.
  // Crossing, shared-segment and bend costs discourage overlapping links.
  class Heap {
    constructor() {
      this.a = [];
    }
    push(v) {
      const a = this.a;
      let i = a.length;
      a.push(v);
      while (i) {
        const p = (i - 1) >> 1;
        if (a[p][0] <= v[0]) break;
        a[i] = a[p];
        i = p;
      }
      a[i] = v;
    }
    pop() {
      const a = this.a,
        out = a[0],
        last = a.pop();
      if (a.length) {
        let i = 0;
        while (i * 2 + 1 < a.length) {
          let c = i * 2 + 1;
          if (c + 1 < a.length && a[c + 1][0] < a[c][0]) c++;
          if (a[c][0] >= last[0]) break;
          a[i] = a[c];
          i = c;
        }
        a[i] = last;
      }
      return out;
    }
  }
  const round = (n) => Math.round(n * 10) / 10;
  function geometry() {
    const area = map.getBoundingClientRect(),
      boxes = {};
    grid.querySelectorAll(".s-node").forEach((n) => {
      const r = n.getBoundingClientRect(),
        fields = {};
      n.querySelectorAll("[data-column]").forEach((f) => {
        const b = f.getBoundingClientRect();
        fields[f.dataset.column] = round(b.top - area.top + b.height / 2);
      });
      boxes[n.dataset.table] = {
        x: round(r.left - area.left),
        y: round(r.top - area.top),
        w: round(r.width),
        h: round(r.height),
        fields,
      };
    });
    return { w: area.width, h: area.height, boxes };
  }
  function routingGraph(g) {
    const xs = [10, g.w - 10],
      ys = [10, g.h - 10],
      ports = {};
    const boxes = Object.values(g.boxes);
    for (const [name, b] of Object.entries(g.boxes)) {
      ys.push(round(b.y - 15), round(b.y + b.h + 15));
      xs.push(round(b.x - 9), round(b.x + b.w + 9));
      const keys = new Set([
        parse(tables[name].fields.split(" ")[0]).name,
        ...edges.filter((e) => e[1] === name).map((e) => e[2]),
      ]);
      for (const field of keys) {
        const y = b.fields[field] ?? round(b.y + 30);
        ports[`${name}:${field}`] = [
          { x: round(b.x - 15), y, ax: b.x, ay: y, dir: 0 },
          { x: round(b.x + b.w + 15), y, ax: round(b.x + b.w), ay: y, dir: 0 },
        ];
        ports[`${name}:${field}`].forEach((p) => {
          xs.push(p.x);
          ys.push(p.y);
        });
      }
    }
    const X = [...new Set(xs.map(round))].sort((a, b) => a - b),
      Y = [...new Set(ys.map(round))].sort((a, b) => a - b),
      points = [],
      index = new Map();
    const blocked = (x, y) =>
      boxes.some(
        (b) =>
          x > b.x - 7 && x < b.x + b.w + 7 && y > b.y - 7 && y < b.y + b.h + 7,
      );
    for (let yi = 0; yi < Y.length; yi++)
      for (let xi = 0; xi < X.length; xi++) {
        const x = X[xi],
          y = Y[yi];
        if (blocked(x, y)) continue;
        index.set(`${xi},${yi}`, points.length);
        points.push({ x, y, xi, yi, neighbors: [] });
      }
    const segmentBlocked = (a, b) =>
      boxes.some((r) =>
        a.y === b.y
          ? a.y > r.y - 7 &&
            a.y < r.y + r.h + 7 &&
            Math.max(a.x, b.x) > r.x - 7 &&
            Math.min(a.x, b.x) < r.x + r.w + 7
          : a.x > r.x - 7 &&
            a.x < r.x + r.w + 7 &&
            Math.max(a.y, b.y) > r.y - 7 &&
            Math.min(a.y, b.y) < r.y + r.h + 7,
      );
    points.forEach((p, i) => {
      for (const [dx, dy, dir] of [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 1],
        [0, -1, 1],
      ]) {
        const j = index.get(`${p.xi + dx},${p.yi + dy}`);
        if (j === undefined || segmentBlocked(p, points[j])) continue;
        const q = points[j];
        p.neighbors.push({
          j,
          dir,
          length: Math.abs(p.x - q.x) + Math.abs(p.y - q.y),
          key: i < j ? `${i}-${j}` : `${j}-${i}`,
        });
      }
    });
    Object.values(ports)
      .flat()
      .forEach((p) => {
        p.index = index.get(`${X.indexOf(p.x)},${Y.indexOf(p.y)}`);
      });
    return {
      points,
      ports,
      horizontal:
        getComputedStyle(grid).gridTemplateColumns.split(" ").length === 4,
    };
  }
  function route(graph, parent, child, field, usage, nodeDirections) {
    const pk = parse(tables[parent].fields.split(" ")[0]).name;
    const forward = graph.horizontal && column[parent] < column[child];
    const { points, ports } = graph,
      starts = ports[`${parent}:${pk}`].filter(
        (p) => p.index !== undefined && (!forward || p.x > p.ax),
      ),
      ends = ports[`${child}:${field}`].filter(
        (p) => p.index !== undefined && (!forward || p.x < p.ax),
      ),
      targets = new Map(ends.map((p) => [p.index, p]));
    const size = points.length * 2,
      dist = new Float64Array(size).fill(Infinity),
      prev = new Int32Array(size).fill(-1),
      startPort = new Array(size),
      heap = new Heap();
    for (const p of starts) {
      const s = p.index * 2 + p.dir;
      dist[s] = 15;
      startPort[s] = p;
      heap.push([15, s]);
    }
    let endState = -1;
    while (heap.a.length) {
      const [cost, s] = heap.pop();
      if (cost > dist[s]) continue;
      const i = Math.floor(s / 2),
        direction = s % 2;
      if (targets.has(i)) {
        endState = s;
        break;
      }
      for (const e of points[i].neighbors) {
        if (forward && points[e.j].x < points[i].x) continue;
        const next = e.j * 2 + e.dir,
          cross = [...(nodeDirections.get(`${i}:${1 - e.dir}`) || [])].some(
            (owner) => owner !== parent,
          )
            ? 5000
            : 0,
          used = [...(usage.get(e.key) || [])].filter(
            (owner) => owner !== parent,
          ).length;
        const value =
          cost +
          e.length +
          (direction !== e.dir ? 16 : 0) +
          cross +
          used * e.length * 20;
        if (value < dist[next]) {
          dist[next] = value;
          prev[next] = s;
          startPort[next] = startPort[s];
          heap.push([value, next]);
        }
      }
    }
    if (endState < 0) return null;
    const indices = [];
    for (let s = endState; s >= 0; s = prev[s]) indices.push(Math.floor(s / 2));
    indices.reverse();
    const start = startPort[endState],
      end = targets.get(Math.floor(endState / 2));
    for (let i = 1; i < indices.length; i++) {
      const a = indices[i - 1],
        b = indices[i],
        dir = points[a].y === points[b].y ? 0 : 1,
        key = a < b ? `${a}-${b}` : `${b}-${a}`;
      if (!usage.has(key)) usage.set(key, new Set());
      usage.get(key).add(parent);
      for (const point of [a, b]) {
        const k = `${point}:${dir}`;
        if (!nodeDirections.has(k)) nodeDirections.set(k, new Set());
        nodeDirections.get(k).add(parent);
      }
    }
    const raw = [
        { x: start.ax, y: start.ay },
        ...indices.map((i) => points[i]),
        { x: end.ax, y: end.ay },
      ],
      path = [];
    raw.forEach((p) => {
      while (path.length > 1) {
        const a = path[path.length - 2],
          b = path[path.length - 1];
        if ((a.x === b.x && b.x === p.x) || (a.y === b.y && b.y === p.y))
          path.pop();
        else break;
      }
      path.push({ x: p.x, y: p.y });
    });
    return path;
  }
  function crossings(routes) {
    let count = 0;
    for (let a = 0; a < routes.length; a++)
      for (let b = a + 1; b < routes.length; b++) {
        const A = routes[a].points,
          B = routes[b].points;
        for (let i = 1; i < A.length; i++)
          for (let j = 1; j < B.length; j++) {
            const p = A[i - 1],
              q = A[i],
              r = B[j - 1],
              s = B[j];
            if (
              p.y === q.y &&
              r.x === s.x &&
              r.x > Math.min(p.x, q.x) &&
              r.x < Math.max(p.x, q.x) &&
              p.y > Math.min(r.y, s.y) &&
              p.y < Math.max(r.y, s.y)
            )
              count++;
            if (
              p.x === q.x &&
              r.y === s.y &&
              p.x > Math.min(r.x, s.x) &&
              p.x < Math.max(r.x, s.x) &&
              r.y > Math.min(p.y, q.y) &&
              r.y < Math.max(p.y, q.y)
            )
              count++;
          }
      }
    return count;
  }
  let pending = false,
    cacheKey = "",
    cached = [];
  const settings = { edgeWidth: 2.4 };
  function scheduleDraw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      draw();
    });
  }
  function draw() {
    const g = geometry();
    if (!g.w) return;
    const key = JSON.stringify(g);
    if (key !== cacheKey) {
      cacheKey = key;
      const graph = routingGraph(g);
      let best = null;
      // Try several deterministic routing orders, keeping the fewest crossings.
      const span = (e) =>
        Math.abs(g.boxes[e[0]].y - g.boxes[e[1]].y) +
        Math.abs(g.boxes[e[0]].x - g.boxes[e[1]].x);
      for (const order of [
        edges,
        [...edges].reverse(),
        [...edges].sort((a, b) => span(a) - span(b)),
        [...edges].sort((a, b) => span(b) - span(a)),
        [...edges].sort((a, b) => a[0].localeCompare(b[0])),
        [...edges].sort((a, b) => a[1].localeCompare(b[1])),
      ]) {
        const usage = new Map(),
          dirs = new Map(),
          routes = [];
        for (const e of order) {
          const points = route(graph, e[0], e[1], e[2], usage, dirs);
          if (points) routes.push({ edge: e, points });
        }
        const score =
          crossings(routes) * 10000 +
          routes.reduce((s, r) => s + r.points.length, 0) +
          (edges.length - routes.length) * 1e6;
        if (!best || score < best.score) best = { routes, score };
      }
      cached = best.routes;
      svg.dataset.crossings = String(crossings(cached));
      svg.dataset.routed = String(cached.length);
    }
    svg.setAttribute("viewBox", `0 0 ${g.w} ${g.h}`);
    svg.replaceChildren();
    const ns = "http://www.w3.org/2000/svg";
    const visible = cached
      .filter(
        (r) =>
          !state.focus ||
          r.edge[0] === state.selected ||
          r.edge[1] === state.selected,
      )
      .sort(
        (a, b) =>
          Number(a.edge.includes(state.selected)) -
          Number(b.edge.includes(state.selected)),
      );
    for (const r of visible) {
      const active =
          r.edge[0] === state.selected || r.edge[1] === state.selected,
        p = document.createElementNS(ns, "path"),
        pts = r.points;
      p.setAttribute(
        "d",
        pts.map((v, i) => `${i ? "L" : "M"}${v.x},${v.y}`).join(" "),
      );
      p.setAttribute(
        "class",
        `s-edge${active ? " is-selected" : ""}${r.edge[3] === "optional" ? " optional" : ""}`,
      );
      p.dataset.join = `${r.edge[1]}.${r.edge[2]} -> ${r.edge[0]}`;
      if (active) p.style.strokeWidth = settings.edgeWidth;
      svg.appendChild(p);
      const a = pts[pts.length - 2],
        b = pts[pts.length - 1],
        dx = Math.sign(b.x - a.x),
        dy = Math.sign(b.y - a.y),
        arrow = document.createElementNS(ns, "path");
      arrow.setAttribute(
        "d",
        `M${b.x},${b.y} L${b.x - dx * 6 - dy * 3},${b.y - dy * 6 + dx * 3} L${b.x - dx * 6 + dy * 3},${b.y - dy * 6 - dx * 3} Z`,
      );
      arrow.setAttribute("class", `s-arrow${active ? " selected" : ""}`);
      svg.appendChild(arrow);
    }
  }
  root.querySelector("[data-fields]").value = state.fields;
  root.querySelector("[data-focus]").checked = state.focus;
  root.querySelector("[data-fields]").onchange = (e) => {
    state.fields = e.target.value;
    state.expanded = [];
    renderNodes();
  };
  root.querySelector("[data-focus]").onchange = (e) => {
    state.focus = e.target.checked;
    scheduleDraw();
  };
  renderNodes();
  renderDetail();
  wireInspect();
  new ResizeObserver(scheduleDraw).observe(map);

  // Report only content height; the parent validates the sending iframe.
  let reportedHeight = 0;
  function reportHeight() {
    const height = Math.ceil(document.body.getBoundingClientRect().height);
    if (height === reportedHeight) return;
    reportedHeight = height;
    if (window.parent !== window)
      window.parent.postMessage(
        { type: "measurement-db:schema-height", height },
        "*",
      );
  }
  new ResizeObserver(reportHeight).observe(document.body);
  window.addEventListener("message", (event) => {
    if (
      event.source === window.parent &&
      event.data?.type === "measurement-db:schema-measure"
    ) {
      reportedHeight = 0;
      reportHeight();
    }
  });
  window.addEventListener("load", reportHeight);
  reportHeight();
})();
