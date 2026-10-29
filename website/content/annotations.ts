// ---------------------------------------------------------------------------
//  Benchmark annotations — the private-benchmark verifiability survey.
//
//  WHAT THESE SCORES MEAN. The instrument codes *public verifiability*: how
//  strongly available evidence permits outsiders to assess a benchmark's claims
//  and procedures. Per the codebook's own decision rule, the scores "describe
//  strength of verifiability evidence, not the benchmark's actual quality,
//  fairness, or validity." A low code means no supporting evidence was located
//  after an adequate search — not that the practice is absent in reality. Copy
//  on these pages must not drift into reading them as a quality ranking.
//
//  The framework is organized around three trust claims (validity, integrity,
//  neutrality). Durability is deliberately NOT a fourth claim: it is the
//  constraint that motivates withholding information in the first place, and so
//  determines the disclosure limits under which the three claims must be
//  evidenced. Verifiability is also distinct from verification — making a claim
//  checkable does not mean anyone checked it.
//
//  This is decoupled from content/benchmark-details.ts and the generated card
//  catalog, which are derived from canonical parquet tables on Hugging Face.
//  None of the benchmarks below is in that catalog, and an annotation is about
//  a benchmark's documentation rather than its response data.
//
//  Source: "Trustworthy Private AI Benchmarks Require External Verification"
//  (codebook v4) and the Part 1 annotation form responses. The context fields
//  below are transcribed from coded annotations; the per-claim assessments are
//  placeholders until the scored items (V1–V5, I1–I5, N1–N4) are exported.
// ---------------------------------------------------------------------------

/** How far a benchmark's annotation has progressed. */
export type AnnotationStatus = "pending" | "context-coded" | "reviewed";

export const annotationStatusLabel: Record<AnnotationStatus, string> = {
  pending: "Not yet annotated",
  "context-coded": "Context coded",
  reviewed: "Reviewed",
};

export type ClaimId =
  | "validity"
  | "integrity"
  | "neutrality"
  | "verification";

/** The three trust claims. `question` is the paper's own one-line framing of
 *  each claim and doubles as the hover definition. */
export const claims: readonly {
  id: ClaimId;
  label: string;
  question: string;
  definition: string;
}[] = [
  {
    id: "validity",
    label: "Validity",
    question: "What does the score support?",
    definition:
      "Whether public evidence shows what capability or task distribution the score characterizes, who it is meant to inform, and what conclusions it does and does not support.",
  },
  {
    id: "integrity",
    label: "Integrity",
    question: "Was the score produced as claimed?",
    definition:
      "Whether a reported score can be shown to follow the stated procedure — item pool, sampling, configuration, scoring, exclusions, reruns and reporting decisions.",
  },
  {
    id: "neutrality",
    label: "Neutrality",
    question: "Was the evaluation administered evenhandedly?",
    definition:
      "Whether participants received comparable access and treatment, and whether funding, ownership or other conflicts that could affect results are disclosed.",
  },
  {
    // Not one of the paper's three trust claims: the other three ask whether a
    // claim *could* be checked, this one asks whether anyone did.
    id: "verification",
    label: "Verification",
    question: "Did anyone outside actually check?",
    definition:
      "Whether a party independent of both the operator and the evaluated providers has examined the benchmark's claims or procedures. Making a claim checkable is not the same as someone having checked it.",
  },
];

/** One claim's cells in the summary table. Empty arrays render as "not yet
 *  coded" rather than as an absence of findings.
 *
 *  `channels` is deliberately not called "fixes": a verifiability gap is not
 *  automatically a defect to be repaired by disclosing more. Durability is a
 *  legitimate reason to withhold, and a claim can be made checkable through
 *  controlled inspection or non-disclosing verification instead of publication.
 *  The cell records what would make the claim checkable, not what the operator
 *  ought to release. */
export type ClaimAssessment = {
  strengths: readonly string[];
  concerns: readonly string[];
  channels: readonly string[];
};

const uncoded: ClaimAssessment = { strengths: [], concerns: [], channels: [] };

// ---------------------------------------------------------------------------
//  Item-level codes (the tree)
// ---------------------------------------------------------------------------

/** One leaf's coded state. `desc` carries a descriptive answer with no status:
 *  the Durability band is not scored, so its leaves report what was coded
 *  rather than how strong the evidence is. */
export type LeafStatus = "ok" | "part" | "gap" | "risk" | "desc";

export const leafStatusLabel: Record<LeafStatus, string> = {
  ok: "Documented",
  part: "Partial",
  gap: "Not located",
  risk: "Disclosed risk",
  desc: "Descriptive",
};

/** One response option and how many annotators selected it. `code` is verbatim
 *  from the form's selection string; `k` of `n` annotators chose it. Counting
 *  is case-insensitive (coders typed "items/tasks" and "Items/tasks"), and the
 *  most common spelling is kept.
 *
 *  Note this is a SELECTION TALLY, not the majority share used in the paper's
 *  reliability section. They agree only for codes a majority selected: a code
 *  nobody picked has majority share 1.0 (all three agreed to leave it) but a
 *  tally of 0/n. The tally is what a reader can interpret directly. */
export type Vote = { code: string; k: number; n: number };

export type Leaf = {
  /** Short label for the compact diagram. */
  name: string;
  /** The form's question, verbatim including its item prefix. */
  question: string;
  /** The coded answer, shown under the name. */
  sub: string;
  /** One sentence of detail, revealed on hover/focus. */
  detail: string;
  status: LeafStatus;
  /** Per-option selection tallies, verbatim from the form. */
  votes?: readonly Vote[];
  /** Question annotator agreement: mean majority share over this question's
   *  options (multi-select) or the majority share of the chosen option
   *  (single-select). */
  qaa?: number;
  /** The chance baseline for this question. For multi-select it is always
   *  0.75 — each checkbox is an independent y/n with expected majority share
   *  (2 unanimous + 6 split x 2/3) / 8. For single-select with N options it is
   *  (N^2 + 3N - 1) / 3N^2, so it falls as the option list grows. */
  qraa?: number;
  kind?: "multi" | "single";
  /** Number of options the form offered for this question. */
  options?: number;
};

/** Agreement across the whole instrument, kept separate for multi- and
 *  single-select because their chance baselines differ. */
export type Agreement = {
  multiBoxes: number;
  multiObserved: number;
  multiRandom: number;
  singleCount: number;
  singleObserved: number;
  singleRandom: number;
};

/** One part of the codebook. `scored` is false for Durability, which the
 *  codebook explicitly excludes from any scale score. */
export type Band = {
  id: string;
  label: string;
  scored: boolean;
  leaves: readonly Leaf[];
};

export type AnnotationLink = { label: string; href: string };

/** E1 — whether a party outside the operator and the evaluated provider
 *  actually performed a relevant check. `sources` is empty when none was
 *  located; otherwise each entry links the verifier's own page. */
export type ExternalVerification = {
  status: string;
  sources: readonly AnnotationLink[];
};

export type Annotation = {
  /** URL segment under /annotations. */
  slug: string;
  name: string;
  status: AnnotationStatus;
  /** Eyebrow line: operator · domain · version. */
  operator: string | null;
  domain: string | null;
  version: string | null;
  /** C1 — consequential components intentionally withheld. */
  hidden: readonly string[];
  /** C3 — how evaluation proceeds while material stays withheld. */
  runBy: string | null;
  /** C4 — secrecy lifecycle: when protected material becomes accessible. */
  release: string | null;
  /** E1 — null while the verification items are uncoded. */
  externalVerification: ExternalVerification | null;
  summary: string | null;
  /** Guidance on reading the benchmark's published scores. */
  readingScores: string | null;
  /** The five codebook parts and their coded items. Empty until coded. */
  bands: readonly Band[];
  claims: Record<ClaimId, ClaimAssessment>;
  links: readonly AnnotationLink[];
  annotators: readonly string[];
  /** Frozen corpus cutoff — every code is tied to evidence available by then. */
  cutoff: string | null;
  /** Instrument-wide annotator agreement, shown under the root node. */
  agreement?: Agreement;
  /** WORKING NOTE — remove before publication. Open issues where the tree, the
   *  summary table and the coded tallies do not yet agree. Rendered under the
   *  tree so they stay in view while being resolved; an empty or absent list
   *  renders nothing. */
  reconcile?: readonly string[];
};

const uncodedClaims: Record<ClaimId, ClaimAssessment> = {
  validity: uncoded,
  integrity: uncoded,
  neutrality: uncoded,
  verification: uncoded,
};

// Context fields hold the majority code across the three annotators; `cutoff`
// is the latest frozen corpus date among them. Where coders split, the value
// taken by two of three is recorded and the disagreement stays in the form.
export const annotations: readonly Annotation[] = [
  {
    slug: "frontiermath",
    name: "FrontierMath",
    status: "context-coded",
    operator: "Epoch AI",
    domain: "Research-level mathematics",
    version: "Tiers 1–3, v2",
    hidden: ["Problems", "Reference answers"],
    runBy: "Epoch runs the model",
    release: "None planned",
    externalVerification: { status: "None located", sources: [] },
    summary:
      "Its scoring code, dated versions and run records are public, so on paper FrontierMath is unusually checkable. Its weak point is independence: OpenAI funded the problems and has every one of them, plus most solutions.",
    readingScores:
      "Don't compare v1 with v2 scores. v2 corrected 42% of problems.",
    bands: [
      {
        id: "durability",
        label: "Durability",
        scored: false,
        leaves: [
          {
            name: "Why it's hidden",
            question:
              "Privacy rationale — Why does the benchmark documentation say information is withheld?",
            sub: "Leakage, gaming, legal, fairness",
            detail:
              "Contamination, adaptive gaming, confidentiality and fair evaluation.",
            status: "desc",
            qaa: 0.875,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "adaptive gaming/overfitting", k: 3, n: 3 },
              { code: "contamination/leakage", k: 3, n: 3 },
              { code: "confidentiality/legal", k: 2, n: 3 },
              { code: "fair or blind evaluation", k: 2, n: 3 },
              { code: "proprietary/IP", k: 1, n: 3 },
            ],
          },
          {
            name: "What's hidden",
            question:
              "Protected component — Which consequential parts of the evaluation are intentionally withheld?",
            sub: "Problems, answers",
            detail: "The problems and their reference answers.",
            status: "desc",
            qaa: 0.9583,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "items/tasks", k: 3, n: 3 },
              { code: "labels/reference answers", k: 3, n: 3 },
              { code: "run logs/configuration", k: 1, n: 3 },
            ],
          },
          {
            name: "Hidden from",
            question:
              "Restricted audience — Who lacks ordinary access to the protected component?",
            sub: "Developers, public, clients",
            detail: "Evaluated developers, the public and customers.",
            status: "desc",
            qaa: 0.8571,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "Evaluated provider/team", k: 3, n: 3 },
              { code: "general public", k: 3, n: 3 },
              { code: "customer/client", k: 2, n: 3 },
              { code: "benchmark participant/user", k: 1, n: 3 },
              { code: "independent researcher", k: 1, n: 3 },
            ],
          },
          {
            name: "Release plan",
            question:
              "Secrecy lifecycle — When, if ever, is the protected material intended to become accessible?",
            sub: "None planned",
            detail: "No release of the hidden set is planned.",
            status: "desc",
            qaa: 0.6667,
            qraa: 0.4907,
            kind: "single",
            options: 6,
            votes: [
              { code: "Indefinite/no planned release", k: 2, n: 3 },
              { code: "not reported", k: 1, n: 3 },
            ],
          },
          {
            name: "How models run",
            question:
              "Evaluation access — How is evaluation performed while the protected material remains withheld?",
            sub: "Epoch runs them",
            detail: "Epoch runs every model itself.",
            status: "desc",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "evaluator runs the model/system", k: 3, n: 3 },
            ],
          },
          {
            name: "Durability controls",
            question:
              "Durability controls — What mechanisms are used to preserve benchmark usefulness across repeated evaluation?",
            sub: "Access limits, leak checks",
            detail: "Access controls and contamination monitoring.",
            status: "desc",
            qaa: 0.9524,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "access controls", k: 3, n: 3 },
              { code: "contamination/overlap monitoring", k: 3, n: 3 },
              { code: "retirement/replacement", k: 1, n: 3 },
            ],
          },
        ],
      },
      {
        id: "validity",
        label: "Validity",
        scored: true,
        leaves: [
          {
            name: "Claims & limits",
            question:
              "Claim framing — Which elements needed to interpret the benchmark claim are explicitly stated?",
            sub: "Documented",
            detail: "Excludes proofs and open research.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 5,
            votes: [
              { code: "intended use or decision purpose", k: 3, n: 3 },
              { code: "limits or out-of-scope interpretation", k: 3, n: 3 },
              { code: "Measured capability/construct", k: 3, n: 3 },
              { code: "target domain/population", k: 3, n: 3 },
            ],
          },
          {
            name: "Sample tasks",
            question:
              "Task transparency — What lets an outsider understand the task without seeing the live private set?",
            sub: "Documented",
            detail: "12 public sample problems.",
            status: "ok",
            qaa: 0.8889,
            qraa: 0.75,
            kind: "multi",
            options: 3,
            votes: [
              { code: "Representative public items/dev set", k: 3, n: 3 },
              { code: "detailed task blueprint/schema", k: 2, n: 3 },
            ],
          },
          {
            name: "Hidden-set coverage",
            question:
              "Hidden-set coverage evidence — What evidence supports assessing whether the hidden set covers the stated task space?",
            sub: "Partial",
            detail:
              "Expert-reviewed, but never compared with the public problems.",
            status: "part",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "documented expert or independent review of hidden set", k: 3, n: 3 },
              { code: "Sampling/source method documented", k: 3, n: 3 },
            ],
          },
          {
            name: "Scoring method",
            question:
              "Scoring transparency — What lets an outsider understand how model outputs become reported scores?",
            sub: "Documented",
            detail: "Answer-checking code is public.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 6,
            votes: [
              { code: "aggregation/threshold rules public", k: 3, n: 3 },
              { code: "judge/scorer details public", k: 3, n: 3 },
              { code: "Metric/formula public", k: 3, n: 3 },
            ],
          },
        ],
      },
      {
        id: "integrity",
        label: "Integrity",
        scored: true,
        leaves: [
          {
            name: "Run rules",
            question:
              "Protocol rules — Which consequential evaluation rules are documented?",
            sub: "Documented",
            detail: "Token limits, forced submission, zero for no answer.",
            status: "ok",
            qaa: 0.8889,
            qraa: 0.75,
            kind: "multi",
            options: 6,
            votes: [
              { code: "Model/system eligibility or configuration", k: 3, n: 3 },
              { code: "result reporting/publication rules", k: 3, n: 3 },
              { code: "task selection/randomization", k: 3, n: 3 },
              { code: "failure/exclusion handling", k: 2, n: 3 },
              { code: "submission/retry/rerun rules", k: 2, n: 3 },
            ],
          },
          {
            name: "Fixed in advance",
            question:
              "Ex-ante fixation — What evidence indicates consequential rules or content were fixed before the relevant results?",
            sub: "Partial",
            detail: "Dated versions, but no hash or preregistration.",
            status: "part",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "Dated/versioned protocol", k: 3, n: 3 },
            ],
          },
          {
            name: "Traceable to a run",
            question:
              "Run provenance — Which details identify the evaluation run that produced a reported score?",
            sub: "Documented",
            detail: "Version, model, settings and date. No scorer version.",
            status: "ok",
            qaa: 0.9048,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "Benchmark version/snapshot", k: 3, n: 3 },
              { code: "inference/system configuration", k: 3, n: 3 },
              { code: "model version/checkpoint", k: 3, n: 3 },
              { code: "task subset/seed/session", k: 3, n: 3 },
              { code: "run date", k: 2, n: 3 },
              { code: "evaluator/scorer version", k: 1, n: 3 },
            ],
          },
          {
            name: "Run records",
            question:
              "Run audit evidence — What retained evidence could be used to check whether the reported run followed the stated procedure?",
            sub: "Partial",
            detail: "Logs only for the 10 public problems.",
            status: "part",
            qaa: 0.75,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "controlled-access run records", k: 2, n: 3 },
              { code: "evaluator or third-party replay possible", k: 2, n: 3 },
              { code: "Public outputs/logs/run record", k: 2, n: 3 },
            ],
          },
        ],
      },
      {
        id: "neutrality",
        label: "Neutrality",
        scored: true,
        leaves: [
          {
            name: "Entry rules",
            question:
              "Participation-rule visibility — Who can inspect the rules governing eligibility and access to the benchmark?",
            sub: "Documented",
            detail: "Epoch reports only models it can run itself.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "Public", k: 3, n: 3 },
            ],
          },
          {
            name: "Unequal treatment",
            question:
              "Differential-treatment disclosure — Does the documentation state whether participants/providers may receive different access or evaluation treatment?",
            sub: "Disclosed risk",
            detail: "OpenAI has exclusive access to some problems.",
            status: "risk",
            qaa: 1.0,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "differential/privileged treatment disclosed or permitted", k: 3, n: 3 },
            ],
          },
          {
            name: "Ties to developers",
            question:
              "Evaluator–provider ties — What material relationships between the evaluator/operator and evaluated providers are documented?",
            sub: "Disclosed risk",
            detail: "OpenAI funded and commissioned the problems.",
            status: "risk",
            qaa: 0.9583,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "customer/vendor relationship", k: 3, n: 3 },
              { code: "funding/sponsorship", k: 3, n: 3 },
              { code: "privileged model/data/API access", k: 3, n: 3 },
              { code: "research/technical partnership", k: 3, n: 3 },
              { code: "other material tie", k: 1, n: 3 },
            ],
          },
        ],
      },
      {
        id: "verification",
        label: "Verification",
        scored: true,
        leaves: [
          {
            name: "Independent check",
            question:
              "External verification status — What evidence shows that a party outside the benchmark operator and evaluated provider actually performed a relevant check?",
            sub: "Not located",
            detail:
              "Reviews so far were commissioned by Epoch or done by OpenAI.",
            status: "gap",
            qaa: 0.6667,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "no external verification located", k: 2, n: 3 },
              { code: "operator reports external verification only", k: 1, n: 3 },
            ],
          },
        ],
      },
    ],
    claims: {
      validity: {
        strengths: [
          "Scope and limits are documented, including the exclusion of proof-based problems.",
          "Public sample problems and the answer format are published.",
        ],
        concerns: [
          "The hidden set has never been compared with the public problems.",
        ],
        channels: [
          "A published comparison of the v2 public and hidden distributions.",
        ],
      },
      integrity: {
        strengths: [
          "The scoring harness is public; per-item verifiers are not.",
          "Every version is dated, and corrections are logged.",
          "Submission, rerun, exclusion and reporting rules are all documented.",
        ],
        concerns: ["Scores on hidden problems can't be checked item by item."],
        channels: [
          "A hash of the held-out set and the scorer version with every score.",
        ],
      },
      neutrality: {
        strengths: [
          "OpenAI's funding and access are stated on the benchmark page.",
        ],
        concerns: [
          "OpenAI has all 290 problems and solutions to 237 of them.",
        ],
        channels: [],
      },
      verification: {
        strengths: [],
        concerns: ["No independent party has checked the hidden set."],
        channels: [
          "An audit of the held-out problems by a party with no OpenAI tie.",
        ],
      },
    },
    links: [
      {
        label: "Epoch AI — Transparency",
        href: "https://epoch.ai/about/transparency",
      },
      {
        label: "Epoch AI Benchmarks — About",
        href: "https://epoch.ai/benchmarks/about",
      },
      {
        label: "FrontierMath Tiers 1–3 (unversioned benchmark page)",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3",
      },
      {
        label: "FrontierMath Tiers 1–3 v1",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1",
      },
      {
        label: "FrontierMath Tiers 1–3 v2",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2",
      },
      {
        label: "GPQA Diamond benchmark graph",
        href: "https://epoch.ai/benchmarks/gpqa-diamond?view=graph&tab=release-date",
      },
      {
        label: "Epoch AI Benchmarks — Use this data",
        href: "https://epoch.ai/benchmarks/use-this-data",
      },
      {
        label: "Benchmark reviews documentation — Included benchmarks",
        href: "https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks",
      },
      {
        label: "Epoch benchmark data archive",
        href: "https://epoch.ai/data/benchmark_data.zip",
      },
      {
        label: "FrontierMath — The benchmark",
        href: "https://epoch.ai/frontiermath/the-benchmark",
      },
      {
        label: "FrontierMath Tiers 1–4",
        href: "https://epoch.ai/frontiermath/tiers-1-4",
      },
      {
        label: "FrontierMath Tiers 1–4 — About",
        href: "https://epoch.ai/frontiermath/tiers-1-4/about",
      },
      {
        label: "FrontierMath Tiers 1–4 — Benchmark problems",
        href: "https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems",
      },
      {
        label: "Less than 70% of FrontierMath is within reach for today's models",
        href: "https://epoch.ai/gradient-updates/less-than-70-percent-of-frontiermath-is-within-reach-for-todays-models",
      },
      {
        label: "Benchmarking hub update",
        href: "https://epoch.ai/latest/benchmarking-hub-update",
      },
      {
        label: "OpenAI and FrontierMath",
        href: "https://epoch.ai/latest/openai-and-frontiermath",
      },
      {
        label: "What is Epoch?",
        href: "https://epoch.ai/latest/what-is-epoch",
      },
      {
        label: "Deep Think Math",
        href: "https://epoch.ai/publications/deep-think-math",
      },
      {
        label: "Grok 4 Math",
        href: "https://epoch.ai/publications/grok-4-math",
      },
      {
        label: "GPT-5.4 set a new record on FrontierMath",
        href: "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath",
      },
      {
        label: "The Epoch Brief — June 12, 2026",
        href: "https://epochai.substack.com/p/the-epoch-brief-june-12-2026",
      },
      {
        label: "Public Inspect evaluation log",
        href: "https://logs.epoch.ai/inspect-viewer/c79c08da/viewer.html?log_file=https%3A%2F%2Flogs.epoch.ai%2Finspect_ai_logs%2FE9AfnKpQrDsFWKmXMhBbJS.eval",
      },
      {
        label: "Epoch AI LinkedIn post announcing FrontierMath Tiers 1–4 v2",
        href: "https://www.linkedin.com/posts/epochai_frontiermath-tiers-14-v2-is-live-we-activity-7471253846296711168-bRlG",
      },
      {
        label: "Epoch AI Research post on X",
        href: "https://x.com/EpochAIResearch/status/2065488154086568445",
      },
      {
        label: "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI (arXiv:2411.04872)",
        href: "https://arxiv.org/abs/2411.04872",
      },
      {
        label: "arXiv:2602.05192 (PDF, cited as a search-result snippet)",
        href: "https://arxiv.org/pdf/2602.05192",
      },
      {
        label: "FrontierMath scoring/evaluation implementation gist",
        href: "https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb",
      },
      {
        label: "Revision history for the scoring/evaluation gist",
        href: "https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb/revisions",
      },
      {
        label: "AI Wiki — FrontierMath",
        href: "https://aiwiki.ai/wiki/frontiermath",
      },
      {
        label: "What I wish I knew about FrontierMath (Lemmata)",
        href: "https://lemmata.substack.com/p/what-i-wish-i-knew-about-frontiermath",
      },
      {
        label: "Our World in Data — AI performance on FrontierMath over time",
        href: "https://ourworldindata.org/grapher/ai-frontiermath-over-time",
      },
      {
        label: "TechCrunch — AI benchmarking organization criticized for waiting to disclose funding from OpenAI",
        href: "https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai",
      },
      {
        label: "InfoQ — Epoch AI introduces FrontierMath benchmark",
        href: "https://www.infoq.com/news/2024/11/epochai-frontiermath-benchmark",
      },
      {
        label: "Some lessons from the OpenAI–FrontierMath debacle (LessWrong)",
        href: "https://www.lesswrong.com/posts/8ZgLYwBmB3vLavjKE/some-lessons-from-the-openai-frontiermath-debacle",
      },
      {
        label: "Meemi's shortform (LessWrong)",
        href: "https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform",
      },
      {
        label: "Specific comment on Meemi's shortform",
        href: "https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz",
      },
      {
        label: "How far behind are open models? (LessWrong)",
        href: "https://www.lesswrong.com/posts/rJcCrXyEsJKmmDpWG/how-far-behind-are-open-models",
      },
      {
        label: "Search Engine Journal — OpenAI secretly funded FrontierMath benchmarking dataset",
        href: "https://www.searchenginejournal.com/openai-secretly-funded-frontiermath-benchmarking-dataset/537760/",
      },
    ],
    annotators: ["Max Zhang", "Ameen Patel", "Nhi Truong"],
    cutoff: "2026-10-18",
    agreement: {
      multiBoxes: 92,
      multiObserved: 0.9348,
      multiRandom: 0.75,
      singleCount: 4,
      singleObserved: 0.8333,
      singleRandom: 0.5949,
    },
    reconcile: [
      "Run records: each annotator picked a different pair of options (2/3 on all three). The leaf reads “Partial — Logs only for the 10 public problems”, which hides a three-way split about what run evidence exists at all.",
      "Ties to developers: unanimous on four relationship types — funding, customer/vendor, research partnership and privileged model/API access — but the Neutrality summary bullet names only funding.",
      "Scoring method: the leaf detail still says “Answer-checking code is public”. The summary bullet was tightened to note that per-item verifiers are withheld; the leaf should match.",
      "Options nobody selected are invisible here, because the tally is built from what was chosen. A 0/3 line would show what was considered and rejected — for example, no public–private comparability evidence under Hidden-set coverage.",
      "Verification is 2–1, not settled. One annotator coded “operator reports external verification only”, citing Epoch-commissioned blind reviews and mathematician interviews; the other two excluded both as construction-stage and operator-commissioned.",
      "Band order differs between views: the tree runs Durability first, the summary table starts at Validity. Pick one order.",
      "Box universe: the 92 count includes the “other” and “unclear” options, which are free text — agreement on a free-text box is not well defined, and the Part 1 working sheet leaves Restricted audience’s “Other” blank and averages over the remaining five. Decide whether they are scored or excluded, and make both views match.",
      "Verification scope is counted in the instrument-wide BAA but has no leaf in the tree, because it is conditional on an external check having occurred. For FrontierMath two annotators left it blank, so its eight boxes score as near-unanimous agreement about a question that did not apply. Decide whether conditional questions enter the denominator at all.",
      "BAA does not reconcile with the working sheet: 92 boxes agreed, but the share totals differ (86.0 here against 85.3 there, giving 0.9348 against 0.92717). 85.3 cannot be produced by majority shares of 1 and 2/3 over 92 binary boxes — it would need 20.1 split boxes — so one side is treating about two boxes differently, or rounding mid-calculation. Worth tracing before either number is published.",
    ],
  },
  {
    slug: "swebench_pro",
    name: "SWE-Bench Pro",
    status: "context-coded",
    operator: "Scale AI",
    domain: "Repository-level software engineering",
    version: "Commercial set, 276 tasks",
    hidden: [
      "Tasks",
      "Reference answers",
      "Rubric and scorer",
      "Environment and tooling",
    ],
    runBy: "Scale AI runs the model",
    release: "None planned",
    externalVerification: {
      status: "None located for the commercial set",
      sources: [],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      {
        label: "Benchmark page",
        href: "https://labs.scale.com/leaderboard/swe_bench_pro?tab=private",
      },
      { label: "Paper", href: "https://arxiv.org/abs/2509.16941" },
    ],
    annotators: ["Max Zhang", "Ameen Patel", "Nhi Truong"],
    cutoff: "2026-10-17",
  },
  {
    slug: "hcast",
    name: "METR Time Horizon / HCAST",
    status: "context-coded",
    operator: "METR",
    domain: "Autonomous software and research tasks",
    version: "Time Horizon 1.1",
    hidden: [
      "Tasks",
      "Reference answers",
      "Rubric and scorer",
      "Environment and tooling",
    ],
    runBy: "METR runs the model",
    release: "None planned",
    externalVerification: {
      status: "Located",
      sources: [
        {
          label: "Nguyen & Fithian, arXiv:2610.12466",
          href: "https://arxiv.org/abs/2610.12466",
        },
        {
          label: "METR time-horizon reanalysis",
          href: "https://github.com/ofir-reich/metr-time-horizon-reanalysis",
        },
      ],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      {
        label: "Benchmark page",
        href: "https://metr.org/blog/2026-01-29-time-horizon-1-1/",
      },
      { label: "HCAST paper", href: "https://metr.org/hcast.pdf" },
    ],
    annotators: ["Max Zhang", "Ameen Patel", "Nhi Truong"],
    cutoff: "2026-10-26",
  },
  {
    slug: "gdpval",
    name: "GDPval",
    status: "context-coded",
    operator: "OpenAI",
    domain: "Economically valuable occupational tasks",
    version: "v2, hidden split of 1,100 of 1,320 tasks",
    hidden: ["Tasks", "Reference answers", "Run logs and configuration"],
    runBy: "OpenAI runs the model",
    release: "None planned",
    externalVerification: {
      status: "Located",
      sources: [
        {
          label: "Inspect Evals: GDPval (UK AISI)",
          href: "https://ukgovernmentbeis.github.io/inspect_evals/evals/gdpval/",
        },
        {
          label: "Epoch AI on economic-value benchmarks",
          href: "https://epoch.ai/publications/what-do-economic-value-benchmarks-tell-us",
        },
      ],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      { label: "Benchmark page", href: "https://openai.com/index/gdpval/" },
      { label: "Paper", href: "https://arxiv.org/abs/2510.04374" },
      {
        label: "Dataset",
        href: "https://huggingface.co/datasets/openai/gdpval",
      },
    ],
    annotators: ["Max Zhang", "Ameen Patel", "Nhi Truong"],
    cutoff: "2026-10-26",
  },
];

export function getAnnotation(slug: string): Annotation | undefined {
  return annotations.find((annotation) => annotation.slug === slug);
}
