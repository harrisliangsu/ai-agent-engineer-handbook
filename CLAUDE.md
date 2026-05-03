# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo nature

Pure Chinese-language documentation handbook — no source code, no build system, no tests. All content is Markdown (`.md`) plus a hand-maintained rendered HTML mirror under `*/html/`. There is nothing to compile, lint, or test. Edits land directly in the Markdown files.

Default to Chinese (zh-CN) for all prose edits and user-facing replies. Code blocks, command names, file paths, and source URLs stay in their original language.

## Layout (two parallel sections)

```
engineering-foundations/   ← knowledge: 4-layer LLM application engineering
  prompt-engineering.md    ← layer 1
  context-engineering.md   ← layer 2
  agent-engineering.md     ← layer 3
  harness-engineering.md   ← layer 4
  overview.md              ← v1 merged historical version (kept on purpose)
  sources.jsonl            ← citation registry (one JSON object per line)
  run_manifest.json        ← deep-research run metadata
  html/                    ← rendered HTML mirror + style.css

interview-prep/            ← job hunting: JD research + Q&A bank
  jd-requirements.md       ← real JD excerpts from CN/US/EU companies
  interview-questions.md   ← 86 questions in 10 categories (A–J)
  html/                    ← rendered HTML mirror + style.css
```

`engineering-foundations/` has a 4-layer abstraction story: prompt → context → agent → harness. Each layer is an independent discipline, but production systems need all four. When editing one chapter, keep cross-references to neighboring layers consistent.

## Editing rules specific to this repo

- **`overview.md` is the v1 merged version, frozen on purpose.** Do not try to reconcile it with the v2/v3 split chapters or "fix" content drift between them. README v3 changelog explains this.
- **`harness-engineering.md` deliberately rejects the older "seven components" framing in favor of "two laws (strong constraints + self-healing loop) + four pillars."** That correction is part of the project's voice — don't reintroduce the seven-components frame.
- **Citation discipline:** Every non-trivial technical claim should have ≥2 independent first-party sources with URLs. New facts → register the source in `engineering-foundations/sources.jsonl` (append a JSON line with the next `S##` id) and cite it inline in the chapter. JD excerpts in `interview-prep/jd-requirements.md` need company name + role + source URL + capture date.
- **Honest limitation notes belong inline.** The repo's stance is "research with limitation disclosures." If a JD was only partially scraped, or a benchmark number is contested, say so where the claim appears.
- **HTML mirror must stay in sync with the Markdown.** The original HTML files were generated from the Markdown by pandoc (see `<meta name="generator" content="pandoc">` in each file). For *subsequent* content edits, do NOT regenerate via pandoc — **port the diff directly into the HTML by hand**, matching the existing pandoc-style markup (`<h2 id="...">`, `<table>`, `<pre><code>`, `<div class="sourceCode">`, etc.). Whenever you edit a `.md`, propagate the same change to its `.html` in the same commit. The shared CSS lives at `*/html/style.css`.
- **`run_manifest.json` is a snapshot of the original deep-research run.** Treat it as historical metadata; don't update it on routine content edits.

## Previewing HTML locally

```bash
cd engineering-foundations/html && python3 -m http.server 8000
# or
cd interview-prep/html && python3 -m http.server 8000
```

The HTML pages have a left-floating TOC + chapter switcher + scroll-spy highlight; the CSS is shared per directory.

## Style conventions to preserve

- Question difficulty in `interview-questions.md` uses ⭐ / ⭐⭐ / ⭐⭐⭐. Each question is 200–500 字 with a primary-source URL at the end.
- Tables in chapter front-matter use the `章节 | 内容要点 | 字数` style — keep the columns consistent when adding rows.
- Dates are absolute and ISO-style (`2026-05-02`), not relative.
