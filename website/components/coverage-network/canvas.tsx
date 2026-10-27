"use client";

import {
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type Ref,
} from "react";
import type Sigma from "sigma";
import type { CoverageNetwork } from "@/lib/coverage-network/model";
import styles from "./network.module.css";

export const NODE_COLORS = { benchmark: "#8c1515", subject: "#167b80" };
export type NetworkControls = {
  zoom: (factor: number) => void;
  reset: () => void;
};

export function CoverageCanvas({
  data,
  visible,
  selected,
  onSelect,
  controlsRef,
}: {
  data: CoverageNetwork;
  visible: Set<string>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  controlsRef: Ref<NetworkControls>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [renderer, setRenderer] = useState<Sigma | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const duration = () =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 250;

  useEffect(() => {
    let cancelled = false;
    let instance: Sigma | undefined;
    let observer: ResizeObserver | undefined;
    // Sigma touches WebGL at import time, so load only after mounting.
    Promise.all([import("sigma"), import("graphology")])
      .then(([{ default: Sigma }, { default: Graph }]) => {
        if (cancelled || !container.current) return;
        const graph = new Graph({ type: "undirected" });
        for (const node of data.nodes)
          graph.addNode(node.id, {
            ...node,
            color: NODE_COLORS[node.kind],
            size: 3 + Math.min(6, Math.log2(node.degree + 1) * 0.8),
          });
        for (const edge of data.edges)
          graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
            size: 0.5,
            color: "#d9dfdd",
          });
        instance = new Sigma(graph, container.current, {
          labelFont: "Arial, sans-serif",
          labelSize: 12,
          labelColor: { color: "#263b38" },
          labelDensity: 0.12,
          labelGridCellSize: 120,
          labelRenderedSizeThreshold: 8,
          stagePadding: 35,
          minCameraRatio: 0.05,
          maxCameraRatio: 2.5,
          zIndex: true,
          hideEdgesOnMove: true,
        });
        instance.on("clickNode", ({ node }) => onSelect(node));
        instance.on("clickStage", () => onSelect(null));
        instance.on("enterNode", ({ node }) => {
          setHovered(node);
          if (container.current) container.current.style.cursor = "pointer";
        });
        instance.on("leaveNode", () => {
          setHovered(null);
          if (container.current) container.current.style.cursor = "grab";
        });
        observer = new ResizeObserver(() => {
          instance?.resize();
          instance?.refresh();
        });
        observer.observe(container.current);
        setRenderer(instance);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
      observer?.disconnect();
      instance?.kill();
    };
  }, [data, onSelect]);

  useImperativeHandle(
    controlsRef,
    () => ({
      zoom(factor) {
        if (!renderer) return;
        const camera = renderer.getCamera();
        void camera.animate(
          { ratio: Math.max(0.05, Math.min(2.5, camera.ratio * factor)) },
          { duration: duration() },
        );
      },
      reset() {
        void renderer
          ?.getCamera()
          .animate(
            { x: 0.5, y: 0.5, ratio: 1, angle: 0 },
            { duration: duration() },
          );
      },
    }),
    [renderer],
  );

  useEffect(() => {
    if (!renderer) return;
    const graph = renderer.getGraph();
    const active =
      selected || (hovered && visible.has(hovered) ? hovered : null);
    const neighbors = active
      ? new Set([active, ...graph.neighbors(active)])
      : null;
    renderer.setSettings({
      nodeReducer(id, attributes) {
        const emphasized = neighbors?.has(id);
        return {
          ...attributes,
          hidden: !visible.has(id),
          color: neighbors && !emphasized ? "#e8e6e2" : attributes.color,
          label: neighbors && !emphasized ? null : attributes.label,
          forceLabel: id === active,
          zIndex: id === active ? 2 : emphasized ? 1 : 0,
          size:
            (attributes.size + (id === active ? 2 : 0)) *
            Math.min(1, renderer.getDimensions().width / 700),
        };
      },
      edgeReducer(id, attributes) {
        const [source, target] = graph.extremities(id);
        const incident = source === active || target === active;
        return {
          ...attributes,
          hidden:
            !visible.has(source) ||
            !visible.has(target) ||
            (!!active && !incident),
          color: incident ? "#739b95" : attributes.color,
          size: incident ? 1 : attributes.size,
        };
      },
    });
  }, [renderer, visible, selected, hovered]);

  useEffect(() => {
    if (!renderer || !selected) return;
    const graph = renderer.getGraph();
    // Reducer updates briefly leave the display cache in raw graph coordinates.
    // Convert canonical positions explicitly instead of reading that cache.
    const framedPosition = (id: string) =>
      renderer.viewportToFramedGraph(
        renderer.graphToViewport(
          graph.getNodeAttributes(id) as { x: number; y: number },
        ),
      );
    const point = framedPosition(selected);
    const neighbors = renderer
      .getGraph()
      .neighbors(selected)
      .filter((id) => visible.has(id));
    const points = [point, ...neighbors.map(framedPosition)];
    const xs = points.map((p) => p.x),
      ys = points.map((p) => p.y);
    const minX = Math.min(...xs),
      maxX = Math.max(...xs),
      minY = Math.min(...ys),
      maxY = Math.max(...ys);
    void renderer.getCamera().animate(
      {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        ratio: Math.max(
          0.18,
          Math.min(1, Math.max(maxX - minX, maxY - minY) * 1.5),
        ),
      },
      { duration: duration() },
    );
  }, [renderer, selected, visible]);

  return (
    <div className={styles.graphFrame}>
      <div
        ref={container}
        className={styles.canvas}
        data-ready={!!renderer}
        role="img"
        aria-label="Benchmark and subject network. Use the search and connection list to explore with a keyboard."
        tabIndex={0}
        onKeyDown={(event) => {
          if (!renderer) return;
          const camera = renderer.getCamera();
          const step = camera.ratio * 0.08;
          const movement: Record<
            string,
            { x?: number; y?: number; ratio?: number }
          > = {
            ArrowLeft: { x: camera.x - step },
            ArrowRight: { x: camera.x + step },
            ArrowUp: { y: camera.y + step },
            ArrowDown: { y: camera.y - step },
            "+": { ratio: camera.ratio / 1.3 },
            "=": { ratio: camera.ratio / 1.3 },
            "-": { ratio: camera.ratio * 1.3 },
          };
          if (event.key === "Escape") onSelect(null);
          if (movement[event.key]) {
            event.preventDefault();
            camera.setState(movement[event.key]);
          }
        }}
      />
      {!renderer && !error ? (
        <p className={styles.canvasMessage} role="status">
          Loading the network…
        </p>
      ) : null}
      {error ? (
        <p className={styles.canvasMessage} role="status">
          The graph could not start in this browser. You can still explore every
          node using the search and connection list.
        </p>
      ) : null}
      <div className={styles.graphHint}>
        Drag to pan · Scroll or pinch to zoom · Click to select
      </div>
    </div>
  );
}
