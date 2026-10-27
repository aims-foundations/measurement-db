import { existsSync, readFileSync } from "node:fs";
import { registerHooks, stripTypeScriptTypes } from "node:module";

// Run pure frontend modules in Node without a browser or a second test bundler.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
      const base = new URL(specifier, context.parentURL);
      for (const suffix of [".ts", "/index.ts"])
        if (existsSync(new URL(base.href + suffix)))
          return nextResolve(base.href + suffix, context);
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.startsWith("file:") && url.endsWith(".ts"))
      return {
        format: "module",
        source: stripTypeScriptTypes(readFileSync(new URL(url), "utf8"), {
          mode: "transform",
        }),
        shortCircuit: true,
      };
    return nextLoad(url, context);
  },
});
