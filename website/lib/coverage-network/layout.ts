import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import type { CoverageNetwork } from "./model";
import { nodeRadius } from "./appearance";

/** Computed on the server, never in the browser's rendering loop. A deterministic
 * seed and fixed iteration count keep positions stable between page visits.
 */
export function positionCoverageNetwork(
  data: CoverageNetwork,
): CoverageNetwork {
  const graph = new Graph({ type: "undirected" });
  const connected = data.nodes.filter((n) => n.degree > 0);
  connected.forEach((node, index) => {
    const angle = index * Math.PI * (3 - Math.sqrt(5));
    const radius = Math.sqrt(index + 1);
    graph.addNode(node.id, {
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
      size: nodeRadius(node.degree),
    });
  });
  for (const edge of data.edges) graph.addEdge(edge.source, edge.target);
  if (graph.order > 1) {
    forceAtlas2.assign(graph, {
      iterations: 1200,
      settings: {
        ...forceAtlas2.inferSettings(graph),
        barnesHutOptimize: true,
        gravity: 0.08,
        scalingRatio: 10,
        slowDown: 2,
      },
    });
  }
  const positions = connected.map((n) => graph.getNodeAttributes(n.id));
  const xs = positions.map((p) => p.x as number),
    ys = positions.map((p) => p.y as number);
  const minX = Math.min(0, ...xs),
    maxX = Math.max(1, ...xs);
  const minY = Math.min(0, ...ys),
    maxY = Math.max(1, ...ys);
  const scale = Math.max(maxX - minX, maxY - minY) / 2;
  // Resolve crowded neighborhoods at the graph's typical display size.
  // This is a bounded build-time pass, not a simulation on visitors' phones.
  if (graph.order > 1)
    noverlap.assign(graph, {
      maxIterations: 120,
      inputReducer: (_id, attributes) => ({
        x: ((attributes.x - (minX + maxX) / 2) / scale) * 260,
        y: ((attributes.y - (minY + maxY) / 2) / scale) * 260,
        size: attributes.size,
      }),
      settings: { margin: 1.5, ratio: 1.2 },
    });
  return {
    ...data,
    nodes: data.nodes.map((node) => {
      if (node.degree) {
        const p = graph.getNodeAttributes(node.id);
        return {
          ...node,
          x: p.x,
          y: p.y,
        };
      }
      // Unlinked benchmarks live in a labeled list, not arbitrary map positions.
      return { ...node, x: 0, y: 0 };
    }),
  };
}
