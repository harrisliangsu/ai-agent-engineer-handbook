# Contributing

[中文](./CONTRIBUTING.md)

Corrections, sources, interview questions, and job-description excerpts are welcome. The chapters are a handbook for readers. Keep changes small and checkable.

Chinese is the default language. English is a parallel edition, not a replacement. Behavior for both languages is defined here and in the [Chinese guide](./CONTRIBUTING.md).

## Bilingual files

| Chinese (default) | English parallel |
|---|---|
| `README.md` | `README.en.md` |
| `SOME-CHAPTER.md` next to it | `SOME-CHAPTER.en.md` |
| `*/html/SOME-CHAPTER.html` | `*/html/SOME-CHAPTER.en.html` |
| `index.html` | `index.en.html` |
| `knowledge-map.md` / `knowledge-map.html` | `knowledge-map.en.md` / `knowledge-map.en.html` |
| `404.html` | bilingual page (Chinese and English on the same 404) |

Do not add an `en/` tree. Sibling names keep Chinese URLs stable and make the language switch a filename change.

`translations/` are Chinese translations of English originals. Do not write an English copy of those notes. Link to the original URL instead.

When a substantive sentence changes, update the Chinese source and the English parallel in the same pull request, then port both into their HTML mirrors. A Chinese-only edit leaves the English page stale, and the reverse leaves the default page stale.

`engineering-foundations/overview.md` and `overview.en.md` are the frozen v1 merge. Fix a translation error if the English does not match the frozen Chinese. Do not "update" either file to agree with the current four chapters.

## Where to talk

| Situation | Use this |
|---|---|
| A paragraph is wrong, stale, or the link is dead, and you can name the file | [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues/new/choose) or a pull request |
| You are adding a question or a job excerpt and the format below is ready | Pull request (an issue first is fine if you want to check scope) |
| It is not a paragraph yet: how to read, job-search progress, a concept to discuss | [Discussions](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| A new section or question type, with no draft yet | Discussions → **Ideas** |
| How to read or use the handbook | Discussions → **Q&A** |
| Introductions, progress, general chat | Discussions → **General**. The welcome thread is [discussions/4](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions/4) |

Discussions are for conversations that are not a diff yet. When a specific paragraph needs to change, open an issue or a pull request so the note is tied to a file and a line. **Announcements** are for maintainers. Polls use **Polls**.

## Changes that help

- Correct or add a fact, with a primary-source URL (paper, vendor doc, standard, or a job post with a capture date)
- Add an interview question: difficulty (⭐ / ⭐⭐ / ⭐⭐⭐) + an answer of about the same depth as the existing ones (the Chinese edition targets 200–500 characters) + a source URL. Add the English wording in the `*.en.md` twin in the same change
- Add a job excerpt: company + role + source URL + capture date. If you only captured a summary, say so in the text. Chinese excerpts stay quoted in Chinese; add an English framing sentence in the English file
- Fix a broken link, a typo, or a table-of-contents anchor. Fix it in both languages when both pages have the link
- Fix a reading bug on the web (TOC, narrow screens, a broken chapter switcher)

New facts in the four engineering chapters need a new line in [`engineering-foundations/sources.jsonl`](./engineering-foundations/sources.jsonl) with the next `S##` id, cited from the chapter. The registry stays language-neutral.

## Please do not

- Rewrite or shorten a whole chapter in `engineering-foundations/` or `interview-prep/`. Correct the local passage
- Treat [`engineering-foundations/overview.md`](./engineering-foundations/overview.md) as a draft to reconcile with the current four chapters. It is the frozen v1 merge, including [`overview.en.md`](./engineering-foundations/overview.en.md)
- Put the "seven components" frame back into harness engineering. That chapter uses two laws (strong constraints + a self-healing loop) plus four pillars
- Edit [`engineering-foundations/run_manifest.json`](./engineering-foundations/run_manifest.json) during ordinary content changes. It is the historical snapshot of that research run
- Submit numbers, salaries, or model rankings with no source

## Markdown and HTML move together

`*/html/` is a hand-maintained mirror, not a fresh pandoc export on every edit. If you change the substance of a `.md` or `.en.md`, port the same change into the matching `.html` or `.en.html`. Keep the existing markup (`<h2 id="...">`, `<table>`, `<pre><code>`, the site nav, the TOC). A Markdown-only edit leaves the [site](https://harrisliangsu.github.io/ai-agent-engineer-handbook/) and the local pages behind the source.

The English HTML mirrors were first built to match that pandoc-style markup and the existing nav, TOC, glossary, and scripts. Later edits are hand-ported the same way as the Chinese pages. Do not regenerate a Chinese page with pandoc and drop the nav.

## Local preview

```bash
python3 -m http.server 8000
# http://localhost:8000                  Chinese portal
# http://localhost:8000/index.en.html    English portal
# http://localhost:8000/knowledge-map.html
# http://localhost:8000/knowledge-map.en.html
```

To read one group of chapters, run the same command inside `engineering-foundations/html` or `interview-prep/html`. Pages use relative links, so they still resolve when the root is published with GitHub Pages. Each English page links to its Chinese twin, and each Chinese page links back.

## Pull request

- One pull request does one thing. The title should say which section changed
- A factual change includes the source URL in the description
- Tick the template checklist, especially "HTML is in sync" for both languages you touched
- Do not restyle unrelated paragraphs on the way through

Maintainers may edit wording directly or ask for a source. A factual claim with no source may not be merged.
