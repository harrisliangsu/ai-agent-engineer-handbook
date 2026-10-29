# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Agent instructions in this file stay English. The handbook itself is bilingual. Chinese is the default edition. English is a parallel set of files, not a second site tree and not a replacement.

## Repo nature

Documentation handbook — no application source, no build system, no tests. Content is Markdown (`.md`) plus a hand-maintained HTML mirror under `*/html/` and at the repository root. Nothing to compile, lint, or test. Edits land directly in the Markdown files, then the matching HTML is updated by hand in the same change.

When the task is prose for this handbook, edit the Chinese file and its English parallel. Do not leave the two editions semantically apart. Code blocks, command names, file paths, and source URLs stay in their original language. Chinese job-description quotations stay quoted in Chinese inside the English edition, with an English framing sentence.

## Language convention

Sibling files, same directory. Do not introduce an `en/` mirror tree.

| Chinese (default) | English |
|---|---|
| `chapter.md` | `chapter.en.md` |
| `html/chapter.html` | `html/chapter.en.html` |
| `index.html` | `index.en.html` |
| `knowledge-map.md` / `knowledge-map.html` | `knowledge-map.en.md` / `knowledge-map.en.html` |
| `README.md`, `CONTRIBUTING.md` | `README.en.md`, `CONTRIBUTING.en.md` |
| `404.html` | bilingual (both languages on the one Pages 404) |

Every English page links to its Chinese twin and the other way around. Relative links inside an English chapter point at other `*.en.md` / `*.en.html` files. `translations/` are Chinese translations of English originals: do not invent an English copy; keep the link to the original. `sources.jsonl`, `run_manifest.json`, and `LICENSE` are not translated.

## Layout

```
engineering-foundations/   ← knowledge: 4-layer LLM application engineering
  prompt-engineering.md    ← layer 1 (+ prompt-engineering.en.md)
  context-engineering.md   ← layer 2
  agent-engineering.md     ← layer 3
  harness-engineering.md   ← layer 4
  overview.md              ← v1 merged historical version (kept on purpose; overview.en.md too)
  sources.jsonl            ← citation registry (one JSON object per line)
  run_manifest.json        ← deep-research run metadata
  html/                    ← rendered HTML mirror + style.css (+ *.en.html)

interview-prep/            ← job hunting: JD research + Q&A bank
  jd-requirements.md       ← real JD excerpts from CN/US/EU companies
  interview-questions.md   ← 92 questions in 11 categories (A–K)
  html/                    ← rendered HTML mirror + style.css (+ *.en.html)

translations/              ← Chinese study notes of English essays (no *.en.md)
knowledge-map.md           ← review map, not the portal
index.html                 ← Chinese portal (default)
index.en.html              ← English portal
```

`engineering-foundations/` has a 4-layer abstraction story: prompt → context → agent → harness. Each layer is an independent discipline, but production systems need all four. When editing one chapter, keep cross-references to neighboring layers consistent in both languages.

## Editing rules specific to this repo

- **`overview.md` is the v1 merged version, frozen on purpose.** So is `overview.en.md`. Do not try to reconcile either with the v2/v3 split chapters or "fix" content drift between them. README v3 changelog explains this. English may fix a mistranslation of the frozen Chinese text only.
- **`harness-engineering.md` deliberately rejects the older "seven components" framing in favor of "two laws (strong constraints + self-healing loop) + four pillars."** That correction is part of the project's voice — don't reintroduce the seven-components frame in either language.
- **Citation discipline:** Every non-trivial technical claim should have ≥2 independent first-party sources with URLs. New facts → register the source in `engineering-foundations/sources.jsonl` (append a JSON line with the next `S##` id) and cite it inline in the chapter (both languages). JD excerpts in `interview-prep/jd-requirements.md` need company name + role + source URL + capture date.
- **Honest limitation notes belong inline.** The repo's stance is "research with limitation disclosures." If a JD was only partially scraped, or a benchmark number is contested, say so where the claim appears, in both editions.
- **HTML mirror must stay in sync with the Markdown.** The original HTML files were generated from the Markdown by pandoc (see `<meta name="generator" content="pandoc">` in each file). For *subsequent* content edits, do NOT regenerate via pandoc — **port the diff directly into the HTML by hand**, matching the existing pandoc-style markup (`<h2 id="...">`, `<table>`, `<pre><code>`, `<div class="sourceCode">`, etc.). Whenever you edit a `.md` or `.en.md`, propagate the same change to its `.html` / `.en.html` in the same commit. The shared CSS lives at `*/html/style.css`. English pages set `lang="en"`; the TOC label comes from `html[lang="en"] nav#TOC::before`.
- **`run_manifest.json` is a snapshot of the original deep-research run.** Treat it as historical metadata; don't update it on routine content edits.

## Previewing HTML locally

```bash
python3 -m http.server 8000
# http://localhost:8000                  Chinese portal
# http://localhost:8000/index.en.html    English portal

# or one chapter directory
cd engineering-foundations/html && python3 -m http.server 8000
cd interview-prep/html && python3 -m http.server 8000
```

The HTML pages have a left-floating TOC + chapter switcher + scroll-spy highlight; the CSS is shared per directory. English chapter files use the `*.en.html` suffix in the same directory so `style.css` resolves unchanged.

## Style conventions to preserve

- Question difficulty in `interview-questions.md` uses ⭐ / ⭐⭐ / ⭐⭐⭐. Each Chinese answer is about 200–500 字 with a primary-source URL at the end. The English twin keeps the same claims, stars, and URL; it is a full answer, not a summary.
- Tables in chapter front-matter use the `章节 | 内容要点 | 字数` style in Chinese and `Chapter | What it covers | Size` in English — keep the columns consistent when adding rows.
- Dates are absolute and ISO-style (`2026-05-02`), not relative.
