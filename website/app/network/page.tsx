import type { Metadata } from "next";
import Link from "next/link";
import { CoverageLoader } from "@/components/coverage-network/loader";

export const metadata: Metadata = {
  title: "Coverage network | Measurement Data Bank",
  description:
    "Explore which AI subjects have been evaluated on which benchmarks in the Measurement Data Bank.",
  alternates: { canonical: "/measurement-db/network" },
};

export default function NetworkPage() {
  return (
    <main id="main-content" className="rd-white-band pb-16 pt-28">
      <div className="rd-container">
        <Link href="/" className="text-sm underline underline-offset-4">
          ← Measurement Data Bank
        </Link>
        <div className="mb-7 mt-7 max-w-3xl">
          <p className="rd-mono text-[var(--palo-alto)]">
            Explore the data bank
          </p>
          <h1 className="mt-3 text-4xl font-light tracking-tight sm:text-6xl">
            Coverage network
          </h1>
          <p className="mt-4 text-base leading-relaxed text-[var(--muted)] sm:text-lg">
            Which subjects have been evaluated together? Each line connects an
            AI subject to a benchmark with recorded evaluations. Select a node
            to explore its connections, or use the matrix to compare coverage.
          </p>
        </div>
        <CoverageLoader />
      </div>
    </main>
  );
}
