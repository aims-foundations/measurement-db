"use client";

import Image from "next/image";
import Link from "next/link";
import { BenchmarkSearchClient } from "@/components/benchmark-search-client";
import { withBase } from "@/lib/base-path";
import type { LocalBenchmarkSearchDocument } from "@/lib/search-types";
import {
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { ArrowUpRight, CaretDown, Info } from "@phosphor-icons/react";
import {
  type Benchmark,
  type BenchmarkCategory,
  // measurementDb, // used only by the temporarily-hidden "Build script" link
} from "@/content/measurement-db";

/** A benchmark enriched (server-side) with its % of (subject × item) cells
 *  observed, so the gallery can sort/show it without importing the big JSON. */
export type ScaleType = "Binary" | "Graded" | "Categorical";

export type GalleryItem = Benchmark & {
  observed: number | null;
  year: number | null;
  scaleType: ScaleType;
  attacks: number | null;
};

const SCALE_TYPES: ScaleType[] = ["Binary", "Graded", "Categorical"];

// Benchmarks we cannot judge are classified as unknown; selecting a known
// saturation value excludes them from the results.
type SaturationKey = "yes" | "no" | "unknown";
const SATURATION_OPTIONS: {
  key: Exclude<SaturationKey, "unknown">;
  label: string;
}[] = [
  { key: "yes", label: "Saturated" },
  { key: "no", label: "Not saturated" },
];
const saturationKey = (s: boolean | null): SaturationKey =>
  s === null ? "unknown" : s ? "yes" : "no";

type SortKey = "group" | "subjects" | "items" | "name";

const SORT_CHOICES = [
  { value: "group", label: "Year · newest first" },
  { value: "name:asc", label: "Name · A to Z" },
  { value: "name:desc", label: "Name · Z to A" },
  { value: "subjects:desc", label: "AI subjects · high to low" },
  { value: "subjects:asc", label: "AI subjects · low to high" },
  { value: "items:desc", label: "Items · high to low" },
  { value: "items:asc", label: "Items · low to high" },
] as const;

type FilterPanelKey =
  | "domain"
  | "scale"
  | "evidence"
  | "released"
  | "more"
  | "guide"
  | "mobile";

// Benchmarks are grouped by release year, newest first, with everything
// before 2023 collapsed into a single "Earlier" bucket (undated cards land
// there too). A benchmark's domain is a *filter*, not its shelf — so one that
// spans several domains (e.g. AlgoTune → math + coding) appears once, under its
// year, and surfaces under every matching domain chip.
const EARLIEST_YEAR = 2023;
const EARLIER_KEY = "earlier";

function yearGroupKey(year: number | null): string {
  return year !== null && year >= EARLIEST_YEAR ? String(year) : EARLIER_KEY;
}

// Build script link is temporarily hidden — re-enable together with the
// "Build script" <a> block in BenchmarkCard and the `FileCode` import.
// function buildScriptHref(slug: string): string {
//   return `${measurementDb.repoHref}/blob/main/${slug}/build.py`;
// }

// ---------------------------------------------------------------------------
//  Deterministic thumbnail fallback (used only if a cover image is missing)
// ---------------------------------------------------------------------------

const COLS = 14;
const ROWS = 8;
const GAP = 2;
const CELL = 20;
const PAD = 6;
const VB_W = COLS * CELL + 2 * PAD;
const VB_H = ROWS * CELL + 2 * PAD;
const OPACITY_TIERS = [0, 0, 0.14, 0.32, 0.58, 0.92] as const;

function hashString(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function makeRng(seed: number): () => number {
  let state = seed || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0xffffffff;
  };
}

function BenchmarkThumbnail({ slug, color }: { slug: string; color: string }) {
  const rng = makeRng(hashString(slug));
  const cells: { x: number; y: number; opacity: number }[] = [];
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const tier = OPACITY_TIERS[Math.floor(rng() * OPACITY_TIERS.length)];
      if (tier === 0) continue;
      cells.push({ x: PAD + col * CELL, y: PAD + row * CELL, opacity: tier });
    }
  }
  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="h-full w-full"
      role="presentation"
      aria-hidden="true"
      preserveAspectRatio="xMidYMid slice"
    >
      <rect width={VB_W} height={VB_H} fill={color} fillOpacity={0.06} />
      {cells.map((cell) => (
        <rect
          key={`${cell.x}-${cell.y}`}
          x={cell.x}
          y={cell.y}
          width={CELL - GAP}
          height={CELL - GAP}
          rx={3}
          fill={color}
          fillOpacity={cell.opacity}
        />
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
//  Card
// ---------------------------------------------------------------------------

// Small institution mark overlaid on the thumbnail corner — the affiliations
// of the reference paper's first two authors. Logo files are fetched by
// scripts/render_website/generate_benchmark_affiliations.py; an institution without a
// usable mark falls back to an initials chip (hover shows the full name).
function InstitutionBadge({
  name,
  logo,
}: {
  name: string;
  logo: string | null;
}) {
  if (logo) {
    return (
      <span
        title={name}
        className="flex h-[30px] w-[30px] items-center justify-center overflow-hidden"
      >
        <Image
          src={withBase(logo)}
          alt={name}
          width={30}
          height={30}
          unoptimized
          // white halo keeps dark marks readable over dark thumbnails
          className="h-full w-full object-contain [filter:drop-shadow(0_0_2px_rgba(255,255,255,0.9))_drop-shadow(0_1px_2px_rgba(0,0,0,0.25))]"
        />
      </span>
    );
  }
  return (
    <span
      title={name}
      role="img"
      aria-label={name}
      className="flex h-[30px] w-[30px] items-center justify-center rounded-full bg-white/70 text-[13px] font-semibold text-[var(--ink)] shadow-sm backdrop-blur-[2px]"
    >
      {(name.replace(/[^A-Za-z0-9]/g, "")[0] ?? "?").toUpperCase()}
    </span>
  );
}

function CompactStat({ value, label }: { value: string; label: string }) {
  return (
    <div className="inline-flex items-baseline gap-1.5">
      <dt className="sr-only">{label}</dt>
      <dd className="m-0 inline-flex items-baseline gap-1.5">
        <span className="font-mono text-xs text-[var(--ink)]">{value}</span>
        <span className="text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
          {label}
        </span>
      </dd>
    </div>
  );
}

function BenchmarkCard({
  benchmark,
  color,
}: {
  benchmark: GalleryItem;
  color: string;
}) {
  return (
    <article
      aria-labelledby={`benchmark-card-${benchmark.slug}`}
      className="panel group relative isolate flex min-h-40 h-full flex-col overflow-hidden p-0 transition-[box-shadow,transform] duration-200 hover:-translate-y-0.5 hover:shadow-md motion-reduce:transform-none motion-reduce:transition-none"
      style={{
        backgroundColor: `color-mix(in srgb, ${color} 7%, white)`,
      }}
    >
      {/* The benchmark artwork is now a decorative watermark. Keeping it as a
          real Next Image preserves lazy loading and optimization while the
          textual card content remains the accessible benchmark identity. */}
      <div
        className="pointer-events-none absolute inset-0 overflow-hidden"
        aria-hidden="true"
      >
        {benchmark.image ? (
          <Image
            src={withBase(benchmark.image)}
            alt=""
            fill
            sizes="(min-width: 1280px) 22vw, (min-width: 640px) 45vw, 90vw"
            className={`${
              benchmark.fit === "contain"
                ? "object-contain p-7"
                : "object-cover object-center"
            } grayscale contrast-75 opacity-[0.12] transition-[filter,opacity] duration-300 group-hover:grayscale-[60%] group-hover:opacity-[0.17] motion-reduce:transition-none`}
          />
        ) : (
          <div className="absolute inset-0 opacity-[0.12] grayscale">
            <BenchmarkThumbnail slug={benchmark.slug} color={color} />
          </div>
        )}
      </div>

      {/* A deterministic white wash—not automatic image sampling—keeps text
          contrast stable across dark artwork, white diagrams, and logos. */}
      <div
        className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.28)_0%,rgba(255,255,255,0.62)_48%,rgba(255,255,255,0.92)_100%)]"
        aria-hidden="true"
      />

      {benchmark.institutions?.length ? (
        <div className="absolute right-3 top-3 z-20 flex gap-1.5">
          {benchmark.institutions.slice(0, 2).map((inst) => (
            <InstitutionBadge
              key={inst.name}
              name={inst.name}
              logo={inst.logo}
            />
          ))}
        </div>
      ) : null}

      <div className="relative z-10 flex min-h-40 flex-1 flex-col p-4">
        <h4
          id={`benchmark-card-${benchmark.slug}`}
          className="type-h3 pr-20 text-[1.2rem] leading-snug"
        >
          <Link
            href={`/${benchmark.slug}`}
            title={`${benchmark.name}: response matrix, sample items, and subjects`}
            className="inline-flex items-start gap-1.5 text-[var(--ink)] transition-colors hover:text-[var(--lagunita)] focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--digital-blue)]"
          >
            {benchmark.name}
            <ArrowUpRight
              size={14}
              weight="regular"
              aria-hidden="true"
              className="mt-1 shrink-0 opacity-45 transition-opacity duration-200 group-hover:opacity-80"
            />
          </Link>
        </h4>

        <p className="mt-1.5 line-clamp-1 text-[0.8125rem] leading-5 text-[var(--muted)]">
          {benchmark.description}
        </p>

        <div className="mt-auto pt-2.5">
          <dl className="flex flex-wrap gap-x-3 gap-y-1 border-t border-black/15 pt-2">
            <CompactStat
              value={benchmark.items.toLocaleString("en-US")}
              label={benchmark.items === 1 ? "item" : "items"}
            />
            <CompactStat
              value={benchmark.models.toLocaleString("en-US")}
              label={benchmark.models === 1 ? "AI subject" : "AI subjects"}
            />
            {benchmark.attacks ? (
              <CompactStat
                value={benchmark.attacks.toLocaleString("en-US")}
                label={benchmark.attacks === 1 ? "attack" : "attacks"}
              />
            ) : null}
          </dl>
        </div>
      </div>
    </article>
  );
}

function YearHeader({ label, count }: { label: string; count: number }) {
  return (
    <div className="mb-5 flex items-baseline gap-3 border-b border-[var(--line)] pb-3">
      <h3 className="type-h3 text-[1.25rem] text-[var(--ink)]">{label}</h3>
      <span className="font-mono text-sm text-[var(--muted)]">{count}</span>
    </div>
  );
}

function FilterDisclosure({
  controlId,
  panelKey,
  label,
  summary,
  open,
  onToggle,
  children,
  align = "left",
  panelClassName = "w-72",
}: {
  controlId: string;
  panelKey: FilterPanelKey;
  label: ReactNode;
  summary?: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
  align?: "left" | "right";
  panelClassName?: string;
}) {
  const triggerId = `${controlId}-trigger`;
  const panelId = `${controlId}-panel`;
  return (
    <div className="relative shrink-0">
      <button
        id={triggerId}
        data-filter-trigger={panelKey}
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={onToggle}
        className={`inline-flex min-h-10 items-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] ${
          open || summary
            ? "border-[var(--ink)] bg-white text-[var(--ink)]"
            : "border-[var(--line)] bg-white text-[var(--muted)] hover:border-[var(--muted)] hover:text-[var(--ink)]"
        }`}
      >
        {label}
        {summary ? (
          <span className="rounded-full bg-[var(--fog-light)] px-1.5 py-0.5 font-mono text-[0.625rem] text-[var(--ink)]">
            {summary}
          </span>
        ) : null}
        <CaretDown
          size={12}
          weight="bold"
          aria-hidden="true"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open ? (
        <div
          id={panelId}
          data-filter-panel={panelKey}
          role="region"
          aria-labelledby={triggerId}
          className={`absolute top-[calc(100%+0.5rem)] z-50 max-w-[calc(100vw-2rem)] rounded-lg border border-[var(--line)] bg-white p-4 shadow-xl ${
            align === "right" ? "right-0" : "left-0"
          } ${panelClassName}`}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}

function CheckOption({
  checked,
  onChange,
  children,
  color,
}: {
  checked: boolean;
  onChange: () => void;
  children: ReactNode;
  color?: string;
}) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-2 rounded px-1.5 text-sm text-[var(--ink)] transition-colors hover:bg-[var(--fog-light)] xl:min-h-8">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-4 w-4 shrink-0 accent-[var(--lagunita)]"
      />
      {color ? (
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
      ) : null}
      <span>{children}</span>
    </label>
  );
}

function FilterGuide() {
  return (
    <dl className="grid gap-3 text-xs leading-relaxed text-[var(--muted)]">
      <div>
        <dt className="font-semibold text-[var(--ink)]">Filter logic</dt>
        <dd>
          A benchmark can match any selected domain, scale, or saturation value.
          Different filter groups combine together.
        </dd>
      </div>
      <div>
        <dt className="font-semibold text-[var(--ink)]">
          Item-level responses
        </dt>
        <dd>
          Limits results to benchmarks with public per-subject, per-item
          response data rather than aggregate results alone.
        </dd>
      </div>
      <div>
        <dt className="font-semibold text-[var(--ink)]">Saturation</dt>
        <dd>
          Saturated means the best system averages at least 90%; benchmarks
          without enough evidence are excluded when this filter is active.
        </dd>
      </div>
      <div>
        <dt className="font-semibold text-[var(--ink)]">Counts</dt>
        <dd>
          Items are evaluation prompts or tasks; AI subjects are the evaluated
          models or agents; attacks are distinct adversarial methods.
        </dd>
      </div>
    </dl>
  );
}

// ---------------------------------------------------------------------------
//  Gallery with filter + sort controls
// ---------------------------------------------------------------------------

export function BenchmarkGalleryClient({
  items,
  categories,
  showLegend = true,
  searchDocuments,
  searchCategoryLabels,
  searchScopeLabel,
  searchQuery,
  onSearchQueryChange,
  metaAnalysisId,
}: {
  items: GalleryItem[];
  categories: readonly BenchmarkCategory[];
  // The filter/count guide is the same everywhere. With more than one gallery
  // on a page, only the first should carry it.
  showLegend?: boolean;
  searchDocuments: LocalBenchmarkSearchDocument[];
  searchCategoryLabels: Record<string, string>;
  searchScopeLabel: string;
  searchQuery?: string;
  onSearchQueryChange?: (query: string) => void;
  metaAnalysisId?: string;
}) {
  const controlsId = useId();
  const controlsRef = useRef<HTMLDivElement>(null);
  const [openPanel, setOpenPanel] = useState<FilterPanelKey | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [scaleSel, setScaleSel] = useState<Set<ScaleType>>(new Set());
  const [onlyItemResponses, setOnlyItemResponses] = useState(false);
  const [satSel, setSatSel] = useState<Set<SaturationKey>>(new Set());
  const [minYear, setMinYear] = useState<number | null>(null);
  const [maxYear, setMaxYear] = useState<number | null>(null);
  const [minSubjects, setMinSubjects] = useState<number | null>(null);
  const [minItems, setMinItems] = useState<number | null>(null);
  const [minAttacks, setMinAttacks] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("group");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    if (!openPanel) return;

    function closeOnOutsidePointer(event: PointerEvent) {
      if (
        controlsRef.current &&
        !controlsRef.current.contains(event.target as Node)
      ) {
        setOpenPanel(null);
      }
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      const trigger = [
        ...(controlsRef.current?.querySelectorAll<HTMLElement>(
          `[data-filter-trigger="${openPanel}"]`,
        ) ?? []),
      ].find((candidate) => candidate.getClientRects().length > 0);
      setOpenPanel(null);
      requestAnimationFrame(() => trigger?.focus());
    }

    document.addEventListener("pointerdown", closeOnOutsidePointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [openPanel]);

  useEffect(() => {
    const wideLayout = window.matchMedia("(min-width: 1280px)");
    const closeOnLayoutChange = () => setOpenPanel(null);
    wideLayout.addEventListener("change", closeOnLayoutChange);
    return () => wideLayout.removeEventListener("change", closeOnLayoutChange);
  }, []);

  const years = useMemo(() => {
    const ys = items.map((b) => b.year).filter((y): y is number => y !== null);
    return [...new Set(ys)].sort((a, b) => a - b);
  }, [items]);

  const presentScales = useMemo(
    () => SCALE_TYPES.filter((s) => items.some((b) => b.scaleType === s)),
    [items],
  );

  const filtersActive =
    selected.size > 0 ||
    scaleSel.size > 0 ||
    onlyItemResponses ||
    satSel.size > 0 ||
    minYear !== null ||
    maxYear !== null ||
    minSubjects !== null ||
    minItems !== null ||
    minAttacks !== null;

  const filterGroupCount =
    Number(selected.size > 0) +
    Number(scaleSel.size > 0) +
    Number(onlyItemResponses || satSel.size > 0) +
    Number(minYear !== null || maxYear !== null) +
    Number(minSubjects !== null || minItems !== null || minAttacks !== null);

  function resetFilters() {
    setSelected(new Set());
    setScaleSel(new Set());
    setOnlyItemResponses(false);
    setSatSel(new Set());
    setMinYear(null);
    setMaxYear(null);
    setMinSubjects(null);
    setMinItems(null);
    setMinAttacks(null);
  }

  const catById = useMemo(
    () => new Map(categories.map((c) => [c.id, c])),
    [categories],
  );

  // Only show category chips that actually have benchmarks.
  const presentCategories = useMemo(
    () =>
      categories.filter((c) => items.some((b) => b.categories.includes(c.id))),
    [categories, items],
  );

  const hasAttacks = useMemo(
    () => items.some((benchmark) => benchmark.attacks !== null),
    [items],
  );

  // A benchmark's accent = its primary (first, most-specific) category color.
  const colorFor = useMemo(
    () => (b: GalleryItem) => catById.get(b.categories[0])?.color ?? "#007c92",
    [catById],
  );

  const filtered = useMemo(() => {
    return items.filter((b) => {
      if (selected.size > 0 && !b.categories.some((c) => selected.has(c)))
        return false;
      if (scaleSel.size > 0 && !scaleSel.has(b.scaleType)) return false;
      if (onlyItemResponses && !b.itemResponses) return false;
      if (satSel.size > 0 && !satSel.has(saturationKey(b.saturation)))
        return false;
      if (minYear !== null && (b.year === null || b.year < minYear))
        return false;
      if (maxYear !== null && (b.year === null || b.year > maxYear))
        return false;
      if (minSubjects !== null && b.models < minSubjects) return false;
      if (minItems !== null && b.items < minItems) return false;
      if (minAttacks !== null && (b.attacks === null || b.attacks < minAttacks))
        return false;
      return true;
    });
  }, [
    items,
    selected,
    scaleSel,
    onlyItemResponses,
    satSel,
    minYear,
    maxYear,
    minSubjects,
    minItems,
    minAttacks,
  ]);

  function toggleScale(s: ScaleType) {
    setScaleSel((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  function toggleSaturation(k: SaturationKey) {
    setSatSel((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  }

  function numInput(value: number | null, set: (n: number | null) => void) {
    return (
      <input
        type="number"
        min={0}
        step={1}
        inputMode="numeric"
        value={value ?? ""}
        onChange={(e) => set(e.target.value ? Number(e.target.value) : null)}
        placeholder="any"
        className="min-h-11 w-16 rounded-md border border-[var(--line)] bg-white px-2 py-1 text-xs text-[var(--ink)] transition-colors hover:border-[var(--muted)] xl:min-h-8"
      />
    );
  }

  const sorted = useMemo(() => {
    const arr = [...filtered];
    const mul = dir === "asc" ? 1 : -1;
    if (sortKey === "name") {
      arr.sort((a, b) => a.name.localeCompare(b.name) * mul);
    } else {
      const v = (b: GalleryItem) =>
        sortKey === "subjects" ? b.models : sortKey === "items" ? b.items : 0;
      arr.sort((a, b) => (v(a) - v(b)) * mul);
    }
    return arr;
  }, [filtered, sortKey, dir]);

  // Group by release year (newest first); pre-2023 + undated collapse to "Earlier".
  const groups = useMemo(() => {
    const byKey = new Map<string, GalleryItem[]>();
    for (const b of filtered) {
      const key = yearGroupKey(b.year);
      const bucket = byKey.get(key);
      if (bucket) bucket.push(b);
      else byKey.set(key, [b]);
    }
    const recent = [...byKey.keys()]
      .filter((k) => k !== EARLIER_KEY)
      .sort((a, b) => Number(b) - Number(a));
    const keys = byKey.has(EARLIER_KEY) ? [...recent, EARLIER_KEY] : recent;
    return keys.map((key) => ({
      key,
      label: key === EARLIER_KEY ? "Earlier (pre-2023)" : key,
      items: byKey.get(key) ?? [],
    }));
  }, [filtered]);

  const grouped = sortKey === "group";

  const sortChoice =
    sortKey === "group" ? "group" : (`${sortKey}:${dir}` as const);

  const evidenceCount = Number(onlyItemResponses) + satSel.size;
  const moreCount =
    Number(minSubjects !== null) +
    Number(minItems !== null) +
    Number(minAttacks !== null);
  const releasedSummary =
    minYear !== null && maxYear !== null
      ? `${minYear}–${maxYear}`
      : minYear !== null
        ? `≥${minYear}`
        : maxYear !== null
          ? `≤${maxYear}`
          : undefined;

  function togglePanel(panel: FilterPanelKey) {
    setOpenPanel((current) => (current === panel ? null : panel));
  }

  function changeSort(value: string) {
    if (value === "group") {
      setSortKey("group");
      return;
    }
    const [nextKey, nextDir] = value.split(":") as [
      Exclude<SortKey, "group">,
      "asc" | "desc",
    ];
    setSortKey(nextKey);
    setDir(nextDir);
  }

  function toggleCat(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function domainFilters() {
    return (
      <fieldset>
        <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-[var(--muted)]">
          Domain
        </legend>
        <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
          Matches any selected domain.
        </p>
        <div className="mt-2 max-h-64 space-y-0.5 overflow-y-auto pr-1">
          {presentCategories.map((category) => (
            <CheckOption
              key={category.id}
              checked={selected.has(category.id)}
              onChange={() => toggleCat(category.id)}
              color={category.color}
            >
              {category.label}
            </CheckOption>
          ))}
        </div>
        {selected.size > 0 ? (
          <button
            type="button"
            onClick={() => setSelected(new Set())}
            className="mt-3 text-xs font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            Clear domains
          </button>
        ) : null}
      </fieldset>
    );
  }

  function scaleFilters() {
    return (
      <fieldset>
        <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-[var(--muted)]">
          Response scale
        </legend>
        <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
          Matches any selected scale.
        </p>
        <div className="mt-2 space-y-0.5">
          {presentScales.map((scale) => (
            <CheckOption
              key={scale}
              checked={scaleSel.has(scale)}
              onChange={() => toggleScale(scale)}
            >
              {scale}
            </CheckOption>
          ))}
        </div>
        {scaleSel.size > 0 ? (
          <button
            type="button"
            onClick={() => setScaleSel(new Set())}
            className="mt-3 text-xs font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            Clear scales
          </button>
        ) : null}
      </fieldset>
    );
  }

  function evidenceFilters() {
    return (
      <fieldset>
        <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-[var(--muted)]">
          Evidence
        </legend>
        <div className="mt-2 space-y-0.5">
          <CheckOption
            checked={onlyItemResponses}
            onChange={() => setOnlyItemResponses((value) => !value)}
          >
            Item-level responses only
          </CheckOption>
        </div>
        <div
          role="group"
          aria-labelledby={controlsId + "-saturation-label"}
          aria-describedby={controlsId + "-saturation-help"}
          className="mt-3 border-t border-[var(--line)] pt-3"
        >
          <div
            id={controlsId + "-saturation-label"}
            className="text-xs font-semibold text-[var(--ink)]"
          >
            Saturation
          </div>
          <p
            id={controlsId + "-saturation-help"}
            className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]"
          >
            Selecting both includes analyzed benchmarks only.
          </p>
          <div className="mt-1.5 space-y-0.5">
            {SATURATION_OPTIONS.map((option) => (
              <CheckOption
                key={option.key}
                checked={satSel.has(option.key)}
                onChange={() => toggleSaturation(option.key)}
              >
                {option.label}
              </CheckOption>
            ))}
          </div>
        </div>
        {evidenceCount > 0 ? (
          <button
            type="button"
            onClick={() => {
              setOnlyItemResponses(false);
              setSatSel(new Set());
            }}
            className="mt-3 text-xs font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            Clear evidence filters
          </button>
        ) : null}
      </fieldset>
    );
  }

  function releasedFilters() {
    return (
      <fieldset>
        <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-[var(--muted)]">
          Released
        </legend>
        <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
          Setting a bound excludes undated benchmarks.
        </p>
        <div className="mt-2 grid grid-cols-2 gap-2">
          <label className="grid gap-1 text-xs text-[var(--muted)]">
            From year
            <select
              value={minYear ?? ""}
              onChange={(event) =>
                setMinYear(
                  event.target.value ? Number(event.target.value) : null,
                )
              }
              className="min-h-11 rounded-md border border-[var(--line)] bg-white px-2 text-xs font-medium text-[var(--ink)] xl:min-h-10"
            >
              <option value="">Any</option>
              {years.map((year) => (
                <option
                  key={year}
                  value={year}
                  disabled={maxYear !== null && year > maxYear}
                >
                  {year}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs text-[var(--muted)]">
            Through year
            <select
              value={maxYear ?? ""}
              onChange={(event) =>
                setMaxYear(
                  event.target.value ? Number(event.target.value) : null,
                )
              }
              className="min-h-11 rounded-md border border-[var(--line)] bg-white px-2 text-xs font-medium text-[var(--ink)] xl:min-h-10"
            >
              <option value="">Any</option>
              {years.map((year) => (
                <option
                  key={year}
                  value={year}
                  disabled={minYear !== null && year < minYear}
                >
                  {year}
                </option>
              ))}
            </select>
          </label>
        </div>
        {minYear !== null || maxYear !== null ? (
          <button
            type="button"
            onClick={() => {
              setMinYear(null);
              setMaxYear(null);
            }}
            className="mt-3 text-xs font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            Clear release range
          </button>
        ) : null}
      </fieldset>
    );
  }

  function moreFilters() {
    return (
      <fieldset>
        <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-[var(--muted)]">
          Minimum counts
        </legend>
        <div className="mt-2 space-y-2">
          <label className="flex items-center justify-between gap-3 text-xs text-[var(--muted)]">
            AI subjects
            {numInput(minSubjects, setMinSubjects)}
          </label>
          <label className="flex items-center justify-between gap-3 text-xs text-[var(--muted)]">
            Items
            {numInput(minItems, setMinItems)}
          </label>
          {hasAttacks ? (
            <label className="flex items-center justify-between gap-3 text-xs text-[var(--muted)]">
              Attacks
              {numInput(minAttacks, setMinAttacks)}
            </label>
          ) : null}
        </div>
        {moreCount > 0 ? (
          <button
            type="button"
            onClick={() => {
              setMinSubjects(null);
              setMinItems(null);
              setMinAttacks(null);
            }}
            className="mt-3 text-xs font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            Clear minimums
          </button>
        ) : null}
      </fieldset>
    );
  }

  function sortSelect(compact = false) {
    return (
      <select
        value={sortChoice}
        onChange={(event) => changeSort(event.target.value)}
        aria-label={`Sort ${searchScopeLabel}`}
        className={`rounded-md border border-[var(--line)] bg-white px-2.5 text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--muted)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] ${
          compact ? "min-h-11 min-w-0 flex-1" : "min-h-10"
        }`}
      >
        {SORT_CHOICES.map((choice) => (
          <option key={choice.value} value={choice.value}>
            {choice.label}
          </option>
        ))}
      </select>
    );
  }

  return (
    <div>
      {/* Compact controls: discrete popovers on wide screens, one in-flow
          filter panel on tablets and phones. */}
      <div
        ref={controlsRef}
        data-gallery-controls
        className="relative z-30 mb-8"
      >
        <div className="rounded-xl border border-[var(--line)] bg-white/80 p-2.5 shadow-sm">
          <BenchmarkSearchClient
            localDocuments={searchDocuments}
            categoryLabels={searchCategoryLabels}
            scopeLabel={searchScopeLabel}
            presentation="catalog"
            query={searchQuery}
            onQueryChange={onSearchQueryChange}
            trailingControls={
              <>
                <div className="flex min-w-0 gap-2 xl:hidden">
                  <button
                    id={controlsId + "-mobile-trigger"}
                    data-filter-trigger="mobile"
                    type="button"
                    aria-expanded={openPanel === "mobile"}
                    aria-controls={controlsId + "-mobile-panel"}
                    onClick={() => togglePanel("mobile")}
                    className={
                      "inline-flex min-h-11 shrink-0 items-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] " +
                      (openPanel === "mobile" || filterGroupCount > 0
                        ? "border-[var(--ink)] bg-white text-[var(--ink)]"
                        : "border-[var(--line)] bg-white text-[var(--muted)] hover:border-[var(--muted)] hover:text-[var(--ink)]")
                    }
                  >
                    Filters
                    {filterGroupCount > 0 ? (
                      <span className="rounded-full bg-[var(--fog-light)] px-1.5 py-0.5 font-mono text-[0.625rem] text-[var(--ink)]">
                        {filterGroupCount}
                      </span>
                    ) : null}
                    <CaretDown
                      size={12}
                      weight="bold"
                      aria-hidden="true"
                      className={
                        "transition-transform " +
                        (openPanel === "mobile" ? "rotate-180" : "")
                      }
                    />
                  </button>
                  {sortSelect(true)}
                </div>

                <div className="hidden items-center gap-2 xl:flex">
                  <FilterDisclosure
                    controlId={controlsId + "-domain"}
                    panelKey="domain"
                    label="Domain"
                    summary={
                      selected.size > 0 ? String(selected.size) : undefined
                    }
                    open={openPanel === "domain"}
                    onToggle={() => togglePanel("domain")}
                  >
                    {domainFilters()}
                  </FilterDisclosure>

                  {presentScales.length > 1 ? (
                    <FilterDisclosure
                      controlId={controlsId + "-scale"}
                      panelKey="scale"
                      label="Scale"
                      summary={
                        scaleSel.size > 0 ? String(scaleSel.size) : undefined
                      }
                      open={openPanel === "scale"}
                      onToggle={() => togglePanel("scale")}
                      panelClassName="w-56"
                    >
                      {scaleFilters()}
                    </FilterDisclosure>
                  ) : null}

                  <FilterDisclosure
                    controlId={controlsId + "-evidence"}
                    panelKey="evidence"
                    label="Evidence"
                    summary={
                      evidenceCount > 0 ? String(evidenceCount) : undefined
                    }
                    open={openPanel === "evidence"}
                    onToggle={() => togglePanel("evidence")}
                  >
                    {evidenceFilters()}
                  </FilterDisclosure>

                  <FilterDisclosure
                    controlId={controlsId + "-released"}
                    panelKey="released"
                    label="Released"
                    summary={releasedSummary}
                    open={openPanel === "released"}
                    onToggle={() => togglePanel("released")}
                    panelClassName="w-64"
                  >
                    {releasedFilters()}
                  </FilterDisclosure>

                  <FilterDisclosure
                    controlId={controlsId + "-more"}
                    panelKey="more"
                    label="More"
                    summary={moreCount > 0 ? String(moreCount) : undefined}
                    open={openPanel === "more"}
                    onToggle={() => togglePanel("more")}
                    panelClassName="w-56"
                  >
                    {moreFilters()}
                  </FilterDisclosure>

                  {sortSelect()}

                  {showLegend ? (
                    <FilterDisclosure
                      controlId={controlsId + "-guide"}
                      panelKey="guide"
                      label={
                        <span className="inline-flex items-center">
                          <Info size={15} aria-hidden="true" />
                          <span className="sr-only">
                            About filters and counts
                          </span>
                        </span>
                      }
                      open={openPanel === "guide"}
                      onToggle={() => togglePanel("guide")}
                      align="right"
                      panelClassName="w-96"
                    >
                      <FilterGuide />
                    </FilterDisclosure>
                  ) : null}
                </div>
              </>
            }
          />

          {openPanel === "mobile" ? (
            <div
              id={controlsId + "-mobile-panel"}
              data-filter-panel="mobile"
              role="region"
              aria-labelledby={controlsId + "-mobile-trigger"}
              className="mt-2 rounded-lg border border-[var(--line)] bg-white p-4 xl:hidden"
            >
              <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                <div>{domainFilters()}</div>
                {presentScales.length > 1 ? <div>{scaleFilters()}</div> : null}
                <div>{evidenceFilters()}</div>
                <div>{releasedFilters()}</div>
                <div>{moreFilters()}</div>
              </div>
              {showLegend ? (
                <details className="group/guide mt-5 border-t border-[var(--line)] pt-4">
                  <summary className="flex min-h-10 cursor-pointer list-none items-center gap-2 text-xs font-medium text-[var(--muted)] transition-colors hover:text-[var(--ink)] [&::-webkit-details-marker]:hidden">
                    <Info size={15} aria-hidden="true" />
                    About filters and counts
                    <CaretDown
                      size={12}
                      weight="bold"
                      aria-hidden="true"
                      className="ml-auto transition-transform group-open/guide:rotate-180"
                    />
                  </summary>
                  <div className="pt-3">
                    <FilterGuide />
                  </div>
                </details>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="mt-2 flex min-h-6 flex-wrap items-start gap-x-4 gap-y-2 px-1 text-xs text-[var(--muted)]">
          <span aria-live="polite">
            {filtered.length} of {items.length} {searchScopeLabel}
          </span>
          <div className="ml-auto flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
            {filtersActive ? (
              <button
                type="button"
                onClick={resetFilters}
                className="inline-flex min-h-11 items-center font-medium text-[var(--lagunita)] underline decoration-transparent underline-offset-2 transition-colors hover:decoration-current focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)]"
              >
                Clear filters
              </button>
            ) : null}
            {metaAnalysisId ? (
              <a
                href={`#${metaAnalysisId}`}
                className="rd-mono inline-flex min-h-11 items-center gap-1.5 text-[var(--ink)] opacity-60 transition-opacity hover:opacity-100 focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)]"
              >
                Meta-analyses <span aria-hidden>↓</span>
              </a>
            ) : null}
          </div>
        </div>
      </div>

      {/* Results */}
      {filtered.length === 0 ? (
        <p className="py-12 text-center text-sm text-[var(--muted)]">
          No {searchScopeLabel} match the current filters.
        </p>
      ) : grouped ? (
        <div className="space-y-14">
          {groups.map(({ key, label, items: groupItems }) => (
            <section key={key} aria-label={label}>
              <YearHeader label={label} count={groupItems.length} />
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {groupItems.map((b) => (
                  <BenchmarkCard
                    key={b.slug}
                    benchmark={b}
                    color={colorFor(b)}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {sorted.map((b) => (
            <BenchmarkCard key={b.slug} benchmark={b} color={colorFor(b)} />
          ))}
        </div>
      )}
    </div>
  );
}
