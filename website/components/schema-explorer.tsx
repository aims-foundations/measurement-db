"use client";

import { useEffect, useRef, useState } from "react";
import { withBase } from "@/lib/base-path";

/** Isolate the interactive diagram's layout and styles from the article. */
export function SchemaExplorer() {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(1100);

  useEffect(() => {
    function resize(event: MessageEvent) {
      // The sandbox has an opaque origin. Accept only this iframe's messages.
      if (event.source !== frame.current?.contentWindow) return;
      const data = event.data;
      if (
        data?.type !== "measurement-db:schema-height" ||
        typeof data.height !== "number" ||
        !Number.isFinite(data.height) ||
        data.height < 100 ||
        data.height > 15000
      )
        return;
      setHeight(Math.ceil(data.height));
    }
    window.addEventListener("message", resize);
    // The lazy frame may finish loading before React hydrates this section.
    frame.current?.contentWindow?.postMessage(
      { type: "measurement-db:schema-measure" },
      "*",
    );
    return () => window.removeEventListener("message", resize);
  }, []);

  return (
    <figure className="my-8" aria-labelledby="normalized-schema-title">
      <figcaption className="mb-4">
        <h3 id="normalized-schema-title" className="rd-h3">
          Normalized schema design
        </h3>
        <p className="rd-schema-copy">
          This design separates task content, grading checks, and executions
          across benchmarks. Select a table to follow its relationships, or
          expand it to inspect all fields. Existing exports may use earlier
          table layouts.
        </p>
        <a
          href={withBase("/schema-explorer/index.html")}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm underline underline-offset-4"
        >
          Open diagram in a new tab
        </a>
      </figcaption>
      <iframe
        ref={frame}
        src={withBase("/schema-explorer/index.html")}
        title="Interactive normalized data schema"
        loading="lazy"
        sandbox="allow-scripts"
        className="block w-full border-0"
        style={{ height }}
      />
    </figure>
  );
}
