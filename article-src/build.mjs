// Build script: concatenates src/sections/*.html, pre-renders TeX math with KaTeX,
// auto-links §-references, generates the TOC, and emits a fully static index.html.
//
//   node build.mjs
//
// Math delimiters in section sources:
//   $$ ... $$   -> display math (katex displayMode)
//   $ ... $     -> inline math
// Rendered output requires no JavaScript for math; only assets/katex/katex.min.css.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformMath, autoLinkSectionRefs, buildToc, GUARDS } from "./src/lib.mjs";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(ROOT, "src");
const SECTIONS_DIR = path.join(SRC, "sections");

const sectionFiles = fs
  .readdirSync(SECTIONS_DIR)
  .filter((f) => f.endsWith(".html"))
  .sort();

console.log("Sections:", sectionFiles.join(", "));

let body = sectionFiles
  .map((f) => fs.readFileSync(path.join(SECTIONS_DIR, f), "utf8"))
  .join("\n\n");

body = transformMath(body);
body = autoLinkSectionRefs(body);
const toc = buildToc(body);

const css = fs.readFileSync(path.join(SRC, "style.css"), "utf8");
const template = fs.readFileSync(path.join(SRC, "template.html"), "utf8");

const out = template
  .replace("<!--STYLE-->", () => `<style>\n${css}\n</style>`)
  .replace("<!--TOC-->", () => toc)
  .replace("<!--BODY-->", () => body);

fs.writeFileSync(path.join(ROOT, "index.html"), out);

const leftoverDisplay = (out.match(/\$\$/g) || []).length;
const leftoverGuards = out.includes(GUARDS.GUARD_OPEN) ? "YES" : "no";
console.log(`index.html written (${(out.length / 1024).toFixed(0)} KB). Leftover $$: ${leftoverDisplay}, leftover guards: ${leftoverGuards}`);
