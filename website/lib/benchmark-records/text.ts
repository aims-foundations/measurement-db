/** Normalize math outside code spans. Currency and the original source remain intact. */
export function normalizeMath(text: string): string {
  const math =
    /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<![\\$\w])\$([^\s$](?:[^$]*?[^\s$])?)\$(?![\d$])/g;
  return text
    .split(/(```[\s\S]*?```|`[^`]*`)/g)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(
            math,
            (
              _match,
              display: string | undefined,
              inline: string | undefined,
              dollars: string | undefined,
            ) =>
              display !== undefined
                ? `\n$$\n${display}\n$$\n`
                : `$$${inline ?? dollars}$$`,
          ),
    )
    .join("");
}
