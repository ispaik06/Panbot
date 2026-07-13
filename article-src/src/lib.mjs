// Shared rendering helpers for the article builds (standalone page + blog port).
import katex from "katex";

// Explicit escapes — private-use-area chars that cannot appear in prose.
const GUARD_OPEN = "\uE000";
const GUARD_CLOSE = "\uE001";

export function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function renderMath(tex, displayMode) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: true,
      strict: "ignore",
      trust: false,
    });
  } catch (err) {
    console.error(`\nKaTeX error (${displayMode ? "display" : "inline"}):\n  ${tex}\n  ${err.message}`);
    process.exitCode = 1;
    return `<code class="katex-error">${escapeHtml(tex)}</code>`;
  }
}

// Replace math outside of <pre>/<code> blocks.
// Guard sentinels are Unicode private-use chars, so they cannot collide with prose.
export function transformMath(html) {
  const guards = [];
  html = html.replace(/<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>/g, (m) => {
    guards.push(m);
    return GUARD_OPEN + (guards.length - 1) + GUARD_CLOSE;
  });

  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    return `<div class="eq">${renderMath(tex.trim(), true)}</div>`;
  });

  html = html.replace(/\$([^$\n]+?)\$/g, (_, tex) => renderMath(tex.trim(), false));

  html = html.replace(new RegExp(`${GUARD_OPEN}(\\d+)${GUARD_CLOSE}`, "g"), (_, i) => guards[Number(i)]);
  return html;
}

// ---------------------------------------------------------------------------
// Section-reference auto-linking: turns plain-text "§5" / "§4.3" into anchors.
// Already-linked references (inside <a>...</a>) are left untouched.
// ---------------------------------------------------------------------------

// Maps are derived from the headings themselves, so this works for any article
// whose h2 headings carry <span class="secno">N</span> and whose h3 headings
// start with "N.M" (e.g. <h3 id="cop">5.2&ensp;...).
function deriveSectionMaps(html) {
  const sections = {};
  const subsections = {};
  const h2re = /<h2\s+id="([^"]+)"[^>]*>\s*<span class="secno">(\d+)<\/span>/g;
  const h3re = /<h3\s+id="([^"]+)"[^>]*>\s*(\d+)\.(\d+)/g;
  let m;
  while ((m = h2re.exec(html)) !== null) sections[Number(m[2])] = m[1];
  while ((m = h3re.exec(html)) !== null) subsections[`${m[2]}.${m[3]}`] = m[1];
  return { sections, subsections };
}

export function autoLinkSectionRefs(html) {
  const { sections, subsections } = deriveSectionMaps(html);
  // Split on existing anchors so we never nest a link inside a link.
  const parts = html.split(/(<a\b[\s\S]*?<\/a>)/g);
  let linked = 0;
  const out = parts.map((part, i) => {
    if (i % 2 === 1) return part; // an existing <a>...</a> chunk
    return part.replace(/§(\d+)(?:\.(\d+))?/g, (m, sec, sub) => {
      const id = sub != null ? (subsections[`${sec}.${sub}`] || sections[Number(sec)])
                             : sections[Number(sec)];
      if (!id) return m;
      linked += 1;
      return `<a href="#${id}">${m}</a>`;
    });
  });
  console.log(`auto-linked ${linked} plain section reference(s)`);
  return out.join("");
}

// Collect h2 headings -> TOC. Headings carry explicit ids in the sources.
export function buildToc(html) {
  const re = /<h2\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g;
  const items = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    const text = m[2].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    items.push({ id: m[1], text });
  }
  let toc = `<nav class="toc" id="toc" aria-label="Table of contents">\n<div class="toc-title">Contents</div>\n<ol>\n`;
  for (const it of items) {
    toc += `<li><a href="#${it.id}">${it.text}</a></li>\n`;
  }
  toc += "</ol>\n</nav>";
  return toc;
}

export const GUARDS = { GUARD_OPEN, GUARD_CLOSE };
