import katex from "katex";
import "katex/dist/katex.min.css";

/** Authored mathematical labels, with KaTeX typography and accessible MathML. */
export function InlineMath({
  tex,
  className,
}: {
  tex: string;
  className?: string;
}) {
  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(tex, {
          trust: false,
          throwOnError: false,
        }),
      }}
    />
  );
}
