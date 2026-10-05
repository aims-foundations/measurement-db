import { readFile } from "node:fs/promises";
import path from "node:path";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { ScrollReveal } from "@/components/scroll-reveal";

const schemaPath = path.resolve(process.cwd(), "..", "schema.md");

function extractSchemaSections(schemaMarkdown: string) {
  const headings = [...schemaMarkdown.matchAll(/^(#{1,6})[ \t]+\S.*\r?$/gm)];
  const firstHeading = headings[0];

  if (!firstHeading) {
    throw new Error(
      "schema.md must contain Markdown section headings beginning with #",
    );
  }

  const sectionDepth = firstHeading[1].length;
  const sectionHeadings = headings.filter(
    (heading) => heading[1].length === sectionDepth,
  );

  if (sectionHeadings.length !== 2) {
    throw new Error(
      "schema.md must contain exactly two top-level Markdown sections",
    );
  }

  const definitionStart = sectionHeadings[0].index;
  const implementationStart = sectionHeadings[1].index;
  const normalizeHeadingDepth = (sectionSource: string) =>
    sectionSource.replace(/^#{1,6}(?=[ \t])/, "##");
  const definitionSource = normalizeHeadingDepth(
    schemaMarkdown.slice(definitionStart, implementationStart),
  );
  const implementationSource = normalizeHeadingDepth(
    schemaMarkdown.slice(implementationStart),
  );

  return { definitionSource, implementationSource };
}

const sharedMarkdownComponents: Components = {
  p: ({ children }) => <p className="rd-schema-copy">{children}</p>,
  pre: ({ children }) => <pre className="rd-schema-pre">{children}</pre>,
  code: ({ children, className }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="rd-schema-inline-code">{children}</code>
    ),
};

const definitionMarkdownComponents: Components = {
  ...sharedMarkdownComponents,
  h2: ({ children }) => (
    <h2 id="measurement-schema-definition" className="rd-h2 scroll-mt-24">
      {children}
    </h2>
  ),
  ul: ({ children }) => (
    <ul className="rd-schema-list" role="list">
      {children}
    </ul>
  ),
  li: ({ children }) => <li>{children}</li>,
};

const implementationMarkdownComponents: Components = {
  ...sharedMarkdownComponents,
  h2: ({ children }) => (
    <h2
      id="measurement-schema-implementation"
      className="rd-schema-implementation-heading rd-h2 scroll-mt-24"
    >
      {children}
    </h2>
  ),
  ul: ({ children }) => (
    <ul className="rd-schema-list" role="list">
      {children}
    </ul>
  ),
  li: ({ children }) => <li>{children}</li>,
};

export async function MeasurementSchemaSection() {
  const schemaMarkdown = await readFile(schemaPath, "utf8");
  const { definitionSource, implementationSource } =
    extractSchemaSections(schemaMarkdown);

  return (
    <section
      id="schema"
      className="rd-white-band rd-band scroll-mt-16"
      aria-labelledby="measurement-schema-definition"
    >
      <div className="rd-container">
        <ScrollReveal>
          <article className="rd-schema-article">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={definitionMarkdownComponents}
            >
              {definitionSource}
            </ReactMarkdown>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={implementationMarkdownComponents}
            >
              {implementationSource}
            </ReactMarkdown>
          </article>
        </ScrollReveal>
      </div>
    </section>
  );
}
