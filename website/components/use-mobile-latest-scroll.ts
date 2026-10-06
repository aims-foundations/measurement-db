"use client";

import { useEffect, useRef } from "react";

/**
 * Timelines run oldest to newest for reading and accessibility. On phones and
 * tablets, position the narrow viewport at the newest end once a closed
 * accordion has actually acquired width; readers can then swipe back through
 * earlier years.
 */
export function useMobileLatestScroll(itemCount: number) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const initialNode = ref.current;
    if (!initialNode) return;

    const narrowScreen = window.matchMedia("(max-width: 1023px)");
    let positioned = false;
    let frame = 0;

    function positionAtLatest() {
      const node = ref.current;
      if (!node) return;
      if (positioned || !narrowScreen.matches || node.clientWidth === 0) return;
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const currentNode = ref.current;
        if (!currentNode) return;
        if (!narrowScreen.matches || currentNode.clientWidth === 0) return;
        currentNode.scrollLeft = Math.max(
          0,
          currentNode.scrollWidth - currentNode.clientWidth,
        );
        positioned = true;
      });
    }

    const observer = new ResizeObserver(positionAtLatest);
    observer.observe(initialNode);
    narrowScreen.addEventListener("change", positionAtLatest);
    positionAtLatest();

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      narrowScreen.removeEventListener("change", positionAtLatest);
    };
  }, [itemCount]);

  return ref;
}
