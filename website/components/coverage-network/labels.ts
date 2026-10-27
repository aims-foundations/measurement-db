import type { NodeLabelDrawingFunction } from "sigma/rendering";

type Box = { x: number; y: number; width: number; height: number };

/** Sigma chooses candidates; measured text bounds keep neighboring labels apart. */
export function createNetworkLabels() {
  const boxes: Box[] = [];
  const draw: NodeLabelDrawingFunction = (context, node, settings) => {
    if (!node.label) return;
    const size = settings.labelSize;
    context.font = `${settings.labelWeight} ${size}px ${settings.labelFont}`;
    const width = context.measureText(node.label).width + 8;
    const height = size + 8;
    const viewport = context.canvas.clientWidth;
    const x = Math.max(
      3,
      Math.min(node.x + node.size + 3, viewport - width - 3),
    );
    const y = Math.max(3, node.y - height / 2);
    const box = { x, y, width, height };
    if (
      !node.forceLabel &&
      boxes.some(
        (other) =>
          x < other.x + other.width + 3 &&
          x + width + 3 > other.x &&
          y < other.y + other.height + 3 &&
          y + height + 3 > other.y,
      )
    )
      return;
    boxes.push(box);
    context.fillStyle = "rgba(252, 253, 250, 0.93)";
    context.fillRect(x, y, width, height);
    context.fillStyle = "#263b38";
    context.fillText(node.label, x + 4, y + size + 1);
  };
  return {
    draw,
    reset: () => {
      boxes.length = 0;
    },
  };
}
