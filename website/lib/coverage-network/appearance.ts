export const NODE_COLORS = { benchmark: "#8c1515", subject: "#167b80" };

// Compress the degree range so highly evaluated subjects do not obscure neighbors.
export const nodeRadius = (degree: number) =>
  2 + Math.sqrt(Math.log2(degree + 1));
