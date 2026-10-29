# AI Agent Engineer Handbook

[中文](./README.md) · [English portal](./index.en.html) · [Read in the browser](https://harrisliangsu.github.io/ai-agent-engineer-handbook/index.en.html)

A field guide for engineers who design, ship, or interview for production LLM agent systems. Chinese is the default edition. This file is the full English parallel of [README.md](./README.md): the same four engineering layers, the same job-description research, and the same interview bank.

> **Suggested GitHub About:** Field guide for production LLM agent systems — prompt, context, agent, and harness engineering, plus real job descriptions and an interview bank. Chinese is the default; a full English parallel edition ships beside it.
>
> Topics can stay as they are, including `chinese`. The description should not say the handbook is Chinese-only.

[![Stars](https://img.shields.io/github/stars/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/stargazers)
[![Forks](https://img.shields.io/github/forks/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/forks)
[![License: MIT](https://img.shields.io/github/license/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](./LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/commits/main)
[![Read online](https://img.shields.io/badge/read-GitHub%20Pages-b85c00?style=flat-square)](https://harrisliangsu.github.io/ai-agent-engineer-handbook/index.en.html)

**Who it's for:** engineers job-hunting, switching into, or leading AI agent / LLM application work; hiring managers calibrating a role; interviewers who want questions organized by layer.

**How to read:** the browser portal, Markdown in the repo, or local HTML. Steps are in [How to read](#how-to-read).

**Updated:** substantive handbook text 2026-05-31 (one gateway reference added 2026-09-21) · repo metadata / portal 2026-10-23 · **Author:** [harrisliangsu](https://github.com/harrisliangsu) · **License:** [MIT](./LICENSE)

**Sponsor:** [Afdian](https://afdian.com/a/sharkchiligaga)

Keywords: AI Agent, LLM, RAG, prompt engineering, context engineering, harness engineering, MCP, interview, handbook. Chinese remains the default language; English is the parallel edition.

GitHub Topics (already set on the repo; listed here only as an index): `agent` · `ai-agent` · `anthropic` · `awesome` · `career` · `chinese` · `context-engineering` · `handbook` · `interview` · `interview-prep` · `llm` · `mcp` · `openai` · `prompt-engineering` · `rag`

## Contents

- [Read online](#read-online)
- [Repository layout](#repository-layout)
- [Part 1: four engineering layers](#part-1-four-layers-of-llm-application-engineering)
- [Part 2: hiring practice](#part-2-hiring-practice-job-descriptions--interview-questions)
- [Suggested paths](#suggested-paths)
- [How to read](#how-to-read)
- [Translations and the knowledge map](#translations-and-the-knowledge-map)
- [How the research was made](#how-the-research-was-made)
- [Community](#community)
- [License](#license)

---

## Read online

No clone required. Open the handbook site in a browser:

**https://harrisliangsu.github.io/ai-agent-engineer-handbook/**

The Chinese portal is `index.html`. The English portal is `index.en.html`. In GitHub **Settings → Pages**, choose branch `main` and folder `/ (root)`. `.nojekyll` publishes the existing HTML as-is, without Jekyll.

| Start here | Page |
|---|---|
| Portal (four layers + interviews + translations) | [English home](https://harrisliangsu.github.io/ai-agent-engineer-handbook/index.en.html) · [中文](https://harrisliangsu.github.io/ai-agent-engineer-handbook/) |
| Prompt engineering | [prompt-engineering.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/prompt-engineering.en.html) |
| Context engineering / RAG | [context-engineering.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/context-engineering.en.html) |
| Agent engineering | [agent-engineering.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/agent-engineering.en.html) |
| Harness engineering | [harness-engineering.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/harness-engineering.en.html) |
| Job descriptions | [jd-requirements.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/interview-prep/html/jd-requirements.en.html) |
| Interview questions | [interview-questions.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/interview-prep/html/interview-questions.en.html) |
| Knowledge map (review map, not the portal) | [knowledge-map.en.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/knowledge-map.en.html) |

The home page is the site portal. The knowledge map is a separate review map that threads the four layers, the question bank, the translations, and the glossary into one picture. It does not replace the portal.

The site is published from the repository **root**. Chapter HTML stays in the existing `html/` directories. Local double-click or `python3 -m http.server` works the same way. Root `.nojekyll` makes Pages serve these hand-maintained HTML files without Jekyll.

---

## Repository layout

English is a sibling of each Chinese source, not a second tree. `prompt-engineering.md` stays Chinese. `prompt-engineering.en.md` is the English chapter. Rendered pages follow the same rule: `html/prompt-engineering.html` and `html/prompt-engineering.en.html`.

```
ai-agent-engineer-handbook/
├── engineering-foundations/   ← knowledge: four layers of LLM application engineering
│   ├── prompt-engineering.md      # Prompt engineering (Chinese)
│   ├── prompt-engineering.en.md   # English parallel
│   ├── context-engineering.md     # Context engineering
│   ├── agent-engineering.md       # Agent engineering
│   ├── harness-engineering.md     # Harness engineering
│   ├── overview.md                # Survey (v1 merged edition, kept as history)
│   ├── README.md                  # Series reading guide
│   ├── sources.jsonl              # First-party source registry (not translated)
│   └── html/                      # Rendered pages (floating TOC + chapter switcher)
│                                  # English mirrors are *.en.html in the same directory
│
├── interview-prep/            ← practice: job descriptions + interview questions
│   ├── jd-requirements.md         # Real CN / US / EU job-description research
│   ├── interview-questions.md     # Questions and answers (A–K)
│   └── html/
│
├── translations/              ← unofficial Chinese translations of English originals
│   ├── building-effective-agents.md
│   ├── multi-agent-research-system.md
│   ├── effective-harnesses-for-long-running-agents.md
│   ├── openai-harness-engineering.md
│   └── html/                      # Chinese study notes; each page links to the English original
│
├── knowledge-map.md           ← review map (also knowledge-map.html / knowledge-map.en.html)
├── index.html                 ← GitHub Pages portal (Chinese default)
├── index.en.html              ← English portal
└── README.md                  ← Chinese home for this file
```

`translations/` are Chinese translations of English essays. There is no extra English copy of those files. Follow the original URL on each page.

---

## Part 1: four layers of LLM application engineering

Organized as the abstraction stack prompt → context → agent → harness. Each layer is a long essay. The survey is the frozen v1 merge.

| Chapter | What it covers | Size |
|---|---|---|
| **01 Prompt engineering** | Transformer / training stack / decoding / CoT / ReAct / automatic prompt optimization / evaluation / OWASP Top 10 | ~64 KB |
| **02 Context engineering** | Full-stack RAG / what RULER says about long context / memory systems (MemGPT / mem0 / Letta / Zep) / MCP | ~81 KB |
| **03 Agent engineering** | How the agent pattern evolved / Anthropic's 6 patterns / 10-framework comparison / 5 collaboration modes / real incident write-ups / permissions | ~90 KB |
| **04 Harness engineering** | Hashimoto's name for the layer / two laws (strong constraints + self-healing loop) + four pillars / maturity model L0–L4 / entropy | ~134 KB |
| **overview** | v1 merged edition, kept as a historical reference | ~73 KB |

**Start:** [`engineering-foundations/README.en.md`](./engineering-foundations/README.en.md). The HTML edition opens [`engineering-foundations/html/prompt-engineering.en.html`](./engineering-foundations/html/prompt-engineering.en.html) (floating TOC on the left, chapter switcher at the top).

### What is distinctive

- **Dense first-party sourcing.** Non-trivial claims aim for at least two independent sources. The bibliography has 50+ core references.
- **Limitations are stated.** Chapter 04 publicly corrected the synthetic "seven components" label and uses the sourced frame "two laws + four pillars."
- **Harness engineering is the emphasis.** It quotes Hashimoto / OpenAI / Anthropic / Augment Code from the primary texts.

---

## Part 2: hiring practice (job descriptions + interview questions)

### 02-1 Job-description research ([`jd-requirements.en.md`](./interview-prep/jd-requirements.en.md))

- **China:** ByteDance Doubao / Alibaba Tongyi / Tencent Hunyuan / Moonshot Kimi / Zhipu GLM / DeepSeek / MiniMax / StepFun / Ant Group AI Force, and others. 14 real excerpts kept verbatim in Chinese, with English framing.
- **North America / Europe:** Anthropic / OpenAI / Google DeepMind / Glean / Vercel / Replit / Sierra / Cognition (Devin) / Cursor / Mistral / Harvey / Perplexity. 8 full job descriptions plus pay bands. Text that was already English stays English.
- **Top 20 recurring requirements**, **7 + 6 role variants**, and a **China vs overseas comparison**.
- **Pay:** from new-graduate packages to senior architect bands, split by location and level.
- **Sources:** BOSS Zhipin / Lagou / 51job / Zhaopin / Nowcoder / V2EX / Liepin / Greenhouse / Ashby / Glassdoor / Levels.fyi / Hacker News / LangChain ecosystem analysis, and 45+ others.

### 02-2 Interview questions and answers ([`interview-questions.en.md`](./interview-prep/interview-questions.en.md))

**92 questions, 11 categories (A–K).** Each item has a ⭐ difficulty, a technical answer, and a primary paper or official-doc URL. The difficulty counts below follow the stars in the question titles. A11 and category K were added later, so the bank is no longer the early 86 questions / 10 categories.

| Category | Count | Difficulty | Focus |
|---|---|---|---|
| **A. LLM fundamentals** | 11 | 2⭐ 7⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / long context / inference optimization / prefill-decode |
| **B. Training / fine-tuning** | 10 | 2⭐ 6⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 emergence |
| **C. Prompt engineering** | 9 | 2⭐ 6⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / automatic prompt optimization |
| **D. RAG** | 10 | 1⭐ 6⭐⭐ 3⭐⭐⭐ | Hybrid / reranker / GraphRAG / Self-RAG / RAGAS |
| **E. Agent** | 12 | 0⭐ 3⭐⭐ 9⭐⭐⭐ | **Core:** 6 patterns / multi-agent / failure modes / harness |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / mem0 / progress files / memory corruption |
| **G. System design** | 8 | 0⭐ 2⭐⭐ 6⭐⭐⭐ | Support / coding / search / LLM gateway / concurrency / cost / sandbox |
| **H. Evaluation** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | Judge bias / pairwise / benchmark hacking / drift |
| **I. Safety / guardrails** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | OWASP / prompt injection / NeMo Guardrails / EU AI Act |
| **J. Behavioral / practice** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto harness loop / zero-to-one rollout |
| **K. Agent evaluation** | 5 | 0⭐ 2⭐⭐ 3⭐⭐⭐ | Benchmark landscape / trajectory / tool-use / launch SLOs / business eval |

The Chinese README counts stars from an earlier pass of the titles. The English chapter follows the stars printed on each question. Where the two tables disagree, the question heading wins.

---

## Suggested paths

| You are | Read in this order |
|---|---|
| **New graduate / student** (interview prep) | `interview-prep/jd-requirements.en.md` for the market → `interview-questions.en.md` categories A/B/C/D → back to the matching chapter in `engineering-foundations/` |
| **Mid-level, 1–3 years** (moving into AI) | `context-engineering.en.md` → `agent-engineering.en.md` → interview sections D/E/F → whiteboard 1–2 questions from G |
| **Senior / staff, 3–5 years** | Read `harness-engineering.en.md` straight through → interview E/G/H/I/J/K (the staff gap is system design, behavior, and the eval loop) |
| **Architect / technical decision-maker** | "Two laws + four pillars" and maturity L0–L4 in `harness-engineering.en.md`, plus every system-design question in G |
| **Hiring manager / interviewer** | `jd-requirements.en.md` to calibrate the role → `interview-questions.en.md` to pick by category and difficulty |

---

## How to read

**Online:** open the [English portal](https://harrisliangsu.github.io/ai-agent-engineer-handbook/index.en.html) or the [Chinese portal](https://harrisliangsu.github.io/ai-agent-engineer-handbook/). Chapters have a floating TOC, a chapter switcher, and scroll-spy highlighting.

**Markdown** (on GitHub, or in VS Code / Obsidian / Typora). English chapters are the `*.en.md` siblings.

**Local HTML:**

```bash
# Repository root: portal + every chapter
python3 -m http.server 8000
# Chinese portal: http://localhost:8000
# English portal: http://localhost:8000/index.en.html

# Or serve one chapter directory
cd engineering-foundations/html && python3 -m http.server 8000
cd interview-prep/html && python3 -m http.server 8000
```

You can also open `engineering-foundations/html/prompt-engineering.en.html` directly.

---

## Translations and the knowledge map

Four unofficial Chinese translations live in [`translations/`](./translations/). Technical terms keep the English gloss in parentheses. The English original is the authority. This edition does not re-translate them into English.

| Chinese study note | Original |
|---|---|
| [Building Effective Agents](./translations/building-effective-agents.md) | [Anthropic · Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) |
| [Multi-agent research system](./translations/multi-agent-research-system.md) | [Anthropic · How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) |
| [Harnesses for long-running agents](./translations/effective-harnesses-for-long-running-agents.md) | [Anthropic · Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) |
| [Harness engineering](./translations/openai-harness-engineering.md) | [OpenAI · Harness engineering](https://openai.com/index/harness-engineering/) |

HTML notes are in [`translations/html/`](./translations/html/). Review map: [`knowledge-map.en.md`](./knowledge-map.en.md) / [`knowledge-map.en.html`](./knowledge-map.en.html). The map connects the system. The site entrance is still the portal, [`index.en.html`](./index.en.html).

---

## How the research was made

The author assembled this repository with [Claude Code](https://claude.com/claude-code) and a personal deep-research workflow, cross-checking primary papers, vendor docs, and real job posts. The stance is **honest research with limitation disclosures**. Deep research was a local authoring skill, not a public GitHub repository. The run record is [`engineering-foundations/run_manifest.json`](./engineering-foundations/run_manifest.json) (a historical snapshot; routine edits do not touch it). `sources.jsonl` is the citation registry and is not translated.

- Claims that cite a primary source include the URL
- Numbers, prices, and model versions move; check them against the publication date
- Some job posts could only be captured as summaries because the board required login or client-side rendering. Those are marked with an evidence grade in the methodology appendix

---

## Community

How to contribute: [CONTRIBUTING.en.md](./CONTRIBUTING.en.md) · [中文](./CONTRIBUTING.md).

| You want to | Go here |
|---|---|
| Point at a specific error, a broken link, or send a change | [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues) / [Pull request](https://github.com/harrisliangsu/ai-agent-engineer-handbook/pulls) |
| Ask how to read, compare job-search notes, or float an idea that is not a diff yet | [Discussions](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| Propose a new question type or chapter | Discussions → **Ideas** |
| Ask how to use or read the handbook | [GitHub Discussions (Q&A)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| Say hello | Discussions → **General** |

Welcome thread: [introductions / job-search progress / corrections](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions/4). Questions about use and reading go to [GitHub Discussions (Q&A)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions).

Corrections are welcome. For a fact, a dead link, or a stale sentence, open an [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues/new/choose) so it can be tied to a paragraph.

---

## License

[MIT](./LICENSE). Papers and vendor documents quoted here remain with their authors. This repository organizes them and, in the Chinese edition, retells them in Chinese. The translations are unofficial study notes.

---

## Star History

<a href="https://star-history.com/#harrisliangsu/ai-agent-engineer-handbook&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=harrisliangsu/ai-agent-engineer-handbook&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=harrisliangsu/ai-agent-engineer-handbook&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=harrisliangsu/ai-agent-engineer-handbook&type=Date" />
 </picture>
</a>
