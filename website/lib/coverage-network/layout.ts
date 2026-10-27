import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import type { CoverageNetwork } from "./model";

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
    });
  });
  for (const edge of data.edges) graph.addEdge(edge.source, edge.target);
  if (graph.order > 1) {
    forceAtlas2.assign(graph, {
      iterations: 400,
      settings: {
        ...forceAtlas2.inferSettings(graph),
        barnesHutOptimize: true,
        gravity: 1,
        scalingRatio: 4,
        slowDown: 5,
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
  let isolatedIndex = 0;
  const isolatedCount = data.nodes.length - connected.length;
  const columns = Math.max(1, Math.ceil(Math.sqrt(isolatedCount)));
  return {
    ...data,
    nodes: data.nodes.map((node) => {
      if (node.degree) {
        const p = graph.getNodeAttributes(node.id);
        return {
          ...node,
          x: (p.x - (minX + maxX) / 2) / scale,
          y: (p.y - (minY + maxY) / 2) / scale,
        };
      }
      const i = isolatedIndex++;
      return {
        ...node,
        x: 1.3 + (i % columns) * 0.065,
        y: 0.5 - Math.floor(i / columns) * 0.065,
      };
    }),
  };
}
