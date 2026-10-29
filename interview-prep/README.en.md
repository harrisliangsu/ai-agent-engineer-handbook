# 02 · Hiring practice: job descriptions + interview questions

> **English** · [中文](./README.md)

Back to the handbook home: [../README.en.md](../README.en.md) · [中文](../README.md)

---

## The two studies in this directory

### [jd-requirements.en.md](./jd-requirements.en.md) — what AI agent job posts actually ask for

**Research date:** 2026-05-02

- **China:** 14 real excerpts, quoted verbatim in Chinese, with English framing
  - Large-lab algorithm roles: ByteDance Doubao / Alibaba Tongyi / Tencent Hunyuan / Moonshot Kimi / Zhipu GLM / DeepSeek / MiniMax / StepFun
  - Agent application roles: Tencent Hunyuan Agent / Ant Group AI Force / ByteDance agent software architect
  - Mid-size and startup: Yiwu, Zhejiang / Linze, Hunan / five AI-native roles at a listed company / Shanghai video AIGC
  - Intern / prompt engineer / inference optimization / AIGC
- **North America / Europe:** 8 full job descriptions plus pay bands
  - Anthropic FDE / Applied AI Engineer (US / London)
  - Glean / Vercel / Replit / Sierra / Cognition (Devin) / Cursor / Mistral
- **Top 20 recurring requirements** across both markets
- **Role variants:** 7 kinds in China and 6 in North America, including pay bands
- **China vs overseas observations**

### [interview-questions.en.md](./interview-questions.en.md) — 92 questions and answers

**11 categories (A–K).** Each question has a ⭐ difficulty, a technical answer, and a primary paper or official-doc URL:

| Category | Count | Focus |
|---|---|---|
| A. LLM fundamentals | 11 | Transformer / RoPE / KV cache / long context / inference optimization / prefill-decode |
| B. Training / fine-tuning | 10 | SFT / DPO / GRPO / LoRA / R1 emergence |
| C. Prompt engineering | 9 | CoT / Self-Consistency / ReAct / automatic prompt optimization |
| D. RAG | 10 | Hybrid / reranker / GraphRAG / Self-RAG / RAGAS |
| E. Agent | 12 | **Core:** 6 patterns / multi-agent / failure modes / harness |
| F. Memory | 8 | MemGPT / mem0 / progress files / memory corruption |
| G. System design | 8 | Support / coding / search / LLM gateway / concurrency / cost / sandbox |
| H. Evaluation | 6 | Judge bias / pairwise / benchmark hacking / drift |
| I. Safety / guardrails | 6 | OWASP / prompt injection / NeMo Guardrails / EU AI Act |
| J. Behavioral / practice | 7 | Hashimoto harness loop / zero-to-one rollout |
| K. Agent evaluation | 5 | Benchmark landscape / trajectory / tool-use / launch SLOs / business eval |

---

## HTML

`html/` has the rendered pages (floating TOC, switcher between the two documents, scroll highlighting). English pages are the `*.en.html` siblings.

```bash
cd interview-prep/html
python3 -m http.server 8000
# http://localhost:8000/jd-requirements.en.html
# http://localhost:8000/interview-questions.en.html
```

---

## Read with

- For the "why" behind a technical answer: [../engineering-foundations/README.en.md](../engineering-foundations/README.en.md) is the four-layer series
- Handbook home: [../README.en.md](../README.en.md)
