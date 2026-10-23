# AI Agent Engineer Handbook

[中文主页](./README.md) · [Read in the browser](https://harrisliangsu.github.io/ai-agent-engineer-handbook/)

A Chinese-primary handbook for engineers who design, ship, or interview for production LLM applications. The long chapters stay in Chinese. This page is only a map.

**Updated:** 2026-05-31 · **License:** [MIT](./LICENSE) · **Author:** [harrisliangsu](https://github.com/harrisliangsu)

It is for working engineers changing into AI agent roles, candidates preparing interviews, and interviewers who want a shared outline. It is not a framework, and it is not a vendor tutorial.

## Read online

The HTML mirror is published from the repository root with GitHub Pages. Local files stay where they are, so cloning and opening `html/` still works.

**https://harrisliangsu.github.io/ai-agent-engineer-handbook/**

## What is inside

| Section | Question it answers | Start here |
|---|---|---|
| Prompt engineering | How do you make one LLM call reliable? | [Chapter](./engineering-foundations/prompt-engineering.md) · [HTML](./engineering-foundations/html/prompt-engineering.html) |
| Context engineering | What should the model see this turn? RAG, long context, memory, MCP | [Chapter](./engineering-foundations/context-engineering.md) · [HTML](./engineering-foundations/html/context-engineering.html) |
| Agent engineering | How does a model finish a multi-step task? | [Chapter](./engineering-foundations/agent-engineering.md) · [HTML](./engineering-foundations/html/agent-engineering.html) |
| Harness engineering | What environment, constraints, and feedback loop keep it from repeating mistakes? | [Chapter](./engineering-foundations/harness-engineering.md) · [HTML](./engineering-foundations/html/harness-engineering.html) |
| Job descriptions | What do real CN / US / EU roles ask for? | [JD notes](./interview-prep/jd-requirements.md) · [HTML](./interview-prep/html/jd-requirements.html) |
| Interview bank | 92 questions, categories A–K, each with a source URL | [Questions](./interview-prep/interview-questions.md) · [HTML](./interview-prep/html/interview-questions.html) |
| Knowledge map | How the layers, questions, and translations connect | [Map](./knowledge-map.md) · [HTML](./knowledge-map.html) |

`engineering-foundations/overview.md` is the frozen v1 combined essay. The split chapters are the current text.

Translations (unofficial study notes; the English originals win):

- [Building Effective Agents](./translations/building-effective-agents.md) — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Multi-agent research system](./translations/multi-agent-research-system.md) — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Harnesses for long-running agents](./translations/effective-harnesses-for-long-running-agents.md) — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Harness engineering](./translations/openai-harness-engineering.md) — [OpenAI](https://openai.com/index/harness-engineering/)

## How the research is framed

Claims in the essays are meant to rest on primary sources (papers, vendor docs, or a captured job post). Numbers, prices, and model versions move. Where a job post could only be partially captured, the chapter says so. The deep-research step was a local authoring workflow, not a public repository; the run snapshot is [`engineering-foundations/run_manifest.json`](./engineering-foundations/run_manifest.json).

## Topics

Repository topics already set on GitHub: `agent`, `ai-agent`, `anthropic`, `chinese`, `handbook`, `interview`, `llm`, `openai`, `prompt-engineering`, `rag`.

## Contribute

Factual corrections and new questions belong in an issue or pull request, with a source URL. Open-ended reading and job-search talk belongs in [Discussions](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) (Q&A, Ideas, or General). See [CONTRIBUTING.md](./CONTRIBUTING.md).

Local preview from the repo root:

```bash
python3 -m http.server 8000
```
