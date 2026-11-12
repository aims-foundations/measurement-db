/** Explicit LaTeX only. Currency, code and the original source remain intact. */
const MATH =
  /\$\$[\s\S]*?\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<![\\$\w])\$([^\s$](?:[^$]*?[^\s$])?)\$(?![\d$])/g;
const ARGUMENTS: Record<string, number> = {
  boxed: 1,
  frac: 2,
  dfrac: 2,
  tfrac: 2,
  sqrt: 1,
  binom: 2,
  mathbb: 1,
  mathbf: 1,
  mathrm: 1,
  mathcal: 1,
};

/** Some prompts use bare commands such as \boxed{A}, without dollar delimiters. */
function commands(text: string): string {
  let result = "",
    start = 0;
  const pattern =
    /(?<![\\\w])\\(boxed|[dt]?frac|sqrt|binom|mathbb|mathbf|mathrm|mathcal)(?=\{)/g;
  for (const match of text.matchAll(pattern)) {
    if (match.index < start) continue;
    let end = match.index + match[0].length,
      valid = true;
    for (let argument = 0; argument < ARGUMENTS[match[1]]; argument++) {
      if (text[end] !== "{") {
        valid = false;
        break;
      }
      let depth = 0;
      do {
        if (text[end] === "{" && text[end - 1] !== "\\") depth++;
        if (text[end] === "}" && text[end - 1] !== "\\") depth--;
        end++;
      } while (depth && end < text.length);
      if (depth) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;
    result +=
      text.slice(start, match.index) + `$$${text.slice(match.index, end)}$$`;
    start = end;
  }
  return result + text.slice(start);
}

function prose(text: string): string {
  let result = "",
    start = 0;
  for (const match of text.matchAll(MATH)) {
    result += commands(text.slice(start, match.index));
    result += match[0].startsWith("$$")
      ? match[0]
      : match[1] !== undefined
        ? `\n$$\n${match[1]}\n$$\n`
        : `$$${match[2] ?? match[3]}$$`;
    start = match.index + match[0].length;
  }
  return result + commands(text.slice(start));
}

function inlineCode(text: string): string {
  let result = "",
    start = 0;
  for (const match of text.matchAll(/(?<!`)(`+)[\s\S]*?\1(?!`)/g)) {
    result += prose(text.slice(start, match.index)) + match[0];
    start = match.index + match[0].length;
  }
  return result + prose(text.slice(start));
}

export function normalizeMath(text: string): string {
  let result = "",
    pending = "",
    fence = "";
  for (const line of text.match(/[^\n]*(?:\n|$)/g) ?? []) {
    if (fence) {
      result += line;
      const close = /^ {0,3}(`+|~+)[ \t]*\r?\n?$/.exec(line)?.[1];
      if (close && close[0] === fence[0] && close.length >= fence.length)
        fence = "";
    } else {
      const open = /^ {0,3}(`{3,}|~{3,})/.exec(line)?.[1];
      if (open) {
        result += inlineCode(pending) + line;
        pending = "";
        fence = open;
      } else pending += line;
    }
  }
  return result + inlineCode(pending);
}
