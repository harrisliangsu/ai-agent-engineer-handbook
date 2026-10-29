# Four layers of LLM application engineering · series guide

> **English** · [中文](./README.md)

**Research date:** 2026-05-01 · **Revision:** 2026-05-01 (v3: each chapter deepened with formulas, code, production configs, and a GitHub reading path)

## What this is

A technical survey of how to run an LLM in production, organized from lower engineering abstractions to higher ones. Each layer is its own discipline. A production system needs all four.

## Reading map

```
prompt-engineering.en.md      → how to write one LLM call
              ↓
context-engineering.en.md     → what the model can see at the moment it runs
              ↓
agent-engineering.en.md       → how an LLM finishes a multi-step task on its own (goal → loop → memory)
              ↓
harness-engineering.en.md     → what environment the agent runs in, and how a mistake is prevented from happening again
```

## Files

| File | Contents | Length |
|---|---|---|
| `README.en.md` | This file: contents and how to read | short |
| `prompt-engineering.en.md` | Transformer / attention / training stack (pretraining / SFT / DPO / GRPO / RLAIF / RLVR / LoRA / QLoRA / distillation) / decoding / CoT / ReAct / ToT / Self-Refine / automatic prompts / evaluation and failure modes | ~10k Chinese characters in the source |
| `context-engineering.en.md` | Definition and origin / physics of long context / Lost-in-the-Middle / RULER / full-stack RAG choices (parse → chunk → embed → vector store → retrieve → rerank → rewrite → GraphRAG / Self-RAG / CRAG / Agentic RAG / multimodal) / four memory systems compared / context-management tactics / MCP / A2A | ~13k Chinese characters in the source |
| `agent-engineering.en.md` | Definition and coordinates / how the pattern evolved / four-layer architecture / type taxonomy / matrix of 10 frameworks / 5 multi-agent collaboration modes / benchmarks (SWE-bench / GAIA / WebArena / OSWorld 2026 ranks) / product cases (Claude Code / Cursor / Devin / Manus / Computer Use) / failure modes and real incidents (Amazon Kiro / Replit / Loop of Death) / permission model | ~12k Chinese characters in the source |
| `harness-engineering.en.md` | Origin (Hashimoto named it in 2026-02) / two laws (strong constraints + self-healing loop) / four pillars (context architecture / agent specialization / persistent memory / structured execution) / §3.3.1–§3.3.3 direct implementations of strong constraints (MCP / sandbox / guardrails) / production-stack diagram / maturity L0–L4 / synthesis / three scenario checklists / limits / full references / methodology | ~13k Chinese characters in the source |
| `overview.en.md` | v1 merged edition (the draft before the split; kept as a reference; the file was once named `article.md`) | ~30k Chinese characters in the source |
| `run_manifest.json` | Research metadata: query / mode / search paths / claim-verification table. Not translated | short |
| `sources.jsonl` | Citation registry (45 entries). Not translated | short |

## Suggested order

**Straight through:** 01 → 02 → 03 → 04 (abstractions from low to high; the four layers stack).

**By role:**

- **AI product manager:** 03 (agent types, product cases) → 02 (RAG choices) → 04 (production stack)
- **Backend engineer moving into AI:** 01 (foundations) → 02 (RAG) → 03 (frameworks) → 04 (observability / eval)
- **Research / ML engineer:** 01.2 training stack → all of 02 → 03 benchmarks → 04.3.4 inference servers
- **DevOps / platform engineer:** all of 04 → 03.10 permissions → 02.7 MCP
- **Founder choosing a stack:** the "startup MVP" recommendations in 04 → framework choice in 03 → memory-system choice in 02

## Version history

| Version | Date | Change |
|---|---|---|
| **v1** | 2026-05-01 17:30 | Original merged draft of 3 parts (`article.md`), about 30k Chinese characters |
| **v2** | 2026-05-01 19:00 | Split into 4 documents, plus a new agent-engineering chapter (real incidents, benchmarks, permissions), about 48k Chinese characters |
| **v3** | 2026-05-01, later | **Each chapter deepened** after feedback that the series was not systematic enough: formulas, code, production configs, GitHub tutorials, about 75k Chinese characters |

### What v3 added on top of v2

| Chapter | Added in v3 |
|---|---|
| **01 Prompt engineering** | Full attention math (`softmax(QK^T/√d_k)V` + multi-head formula + PyTorch) / library analogy for Q-K-V / full transformer-block diagram (Pre-Norm + RMSNorm + RoPE + SwiGLU + GQA + MoE) / RLHF objective (4-model table) / full DPO loss derivation / GRPO + RLVR flow and code / DPO variants (IPO/KTO/SimPO/CPO) / three-algorithm comparison / **GitHub tutorial section** (karpathy nn-zero-to-hero / build-nanogpt / Sebastian Raschka LLMs-from-scratch / HuggingFace TRL / unsloth, 13 items + a paper list) |
| **02 Context engineering** | ~100-line Python skeleton of a RAG pipeline (parse → chunk → embed → hybrid retrieve → rewrite → rerank → generate) / **GitHub RAG section** (NirDiamant RAG_Techniques / RAGFlow / LightRAG / Awesome-RAG, 8 items) |
| **03 Agent engineering** | **Anthropic's 6 composable patterns in full** (Augmented LLM / Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer / Agents, each with a figure, a setting, and when to pick it) + a pattern decision tree / Hello World comparison for LangGraph, CrewAI, and the OpenAI Agents SDK, plus a philosophy table / **~100-line end-to-end agent** (RAG + memory + 4 Loop-of-Death gates + checkpoint + observability) / **GitHub agent section** (microsoft AI agents for beginners / agent-framework / hello-agents / Anthropic essays / 8 frameworks, 12 items + papers) |
| **04 Harness engineering** | **Two laws + four pillars** + direct strong-constraint implementations (MCP / sandbox / guardrails) + the §3.9 appendix of 4 related infrastructure kinds (LLM gateway / inference / observability / evaluation, explicitly marked as not the harness) + **config set**: LiteLLM `config.yaml` (routing + fallback + caching) / vLLM vs SGLang launch commands / three ways to attach Langfuse / Promptfoo `promptfooconfig.yaml` (CI gating) / RAGAS in Python / NeMo Guardrails Colang / E2B sandbox / minimal MCP server / pre-commit hook (strong constraint) / evaluator-optimizer (self-healing loop) / harness maturity L0–L4 / **GitHub harness section** (7 infrastructure categories, 30+ items) |

### How this differs from v1 (the merged edition)

| Change | v1 (now `overview.md`, once `article.md`) | v2 / v3 (4 files) |
|---|---|---|
| Chapters | 3 parts | 4 parts |
| Agent material | Scattered | Its own chapter |
| Real incidents | Not covered | Added (Amazon Kiro / Replit / Loop of Death) |
| Formulas and code | Description only | v3 has full formulas and runnable code |
| Production configs | Description only | v3 has yaml/python per component |
| GitHub resources | Very few | v3 has 50+ classified links |
| Length | ~30k Chinese characters | v2 ~48k → v3 ~75k |

## Selected primary sources

- Karpathy: *Software Is Changing (Again)* (where "context engineering" is named)
- Anthropic: *Effective Context Engineering for AI Agents*
- Mitchell Hashimoto / Augment Code: *Harness Engineering for AI Coding Agents* (where "harness engineering" is named)
- Vaswani et al. (2017): *Attention Is All You Need*
- Liu et al. (2024): *Lost in the Middle* (TACL)
- Asai et al. (2023): *Self-RAG*
- Yao et al. (2022): *ReAct*
- Madaan et al. (2023): *Self-Refine* (NeurIPS)
- Packer et al. (2023): *MemGPT*
- Sebastian Raschka: *State of LLMs 2025*
- llm-stats: *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond*
- Berkeley RDI: *How We Broke Top AI Agent Benchmarks* (a credibility audit of benchmarks)

The full set of 45 is in each chapter's bibliography and in `sources.jsonl`.

## Reproduce the counts

```bash
# The public copy is this directory. The original deep-research output path lived only on the author's machine and is not in the repo.
wc -m prompt-engineering.md context-engineering.md agent-engineering.md harness-engineering.md overview.md
```
