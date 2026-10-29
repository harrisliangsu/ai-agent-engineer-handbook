# 🧠 AI Agent / LLM application engineer: panoramic knowledge map

> **English** · [中文](./knowledge-map.md)

> This repository's mind map / review map. It threads together the four engineering layers, 92 interview questions, 4 classic translations, and an 83-entry glossary, and it gives coordinates for the whole knowledge system.

**How to use:**

- When reviewing, cross-drill four lenses in this order: four engineering layers → interview categories → classic translations → glossary
- Hit an unfamiliar term: jump to the glossary and open it
- Hit an unfamiliar question: jump to the matching section in [`interview-prep/`](interview-prep/)
- Hit an unfamiliar concept: jump to the matching subsection in [`engineering-foundations/`](engineering-foundations/)
- The reading entrance for the site is the root portal: [`index.en.html`](index.en.html) is the English portal, and [`index.html`](index.html) is the Chinese portal. This file is the review map, not a substitute page for the portal

---

## 1. Repository overview

```
ai-agent-engineer-handbook/
├── engineering-foundations/     four-layer engineering abstraction (core knowledge system)
│   ├── prompt-engineering.md     L1 prompts
│   ├── context-engineering.md    L2 context
│   ├── agent-engineering.md      L3 Agent
│   ├── harness-engineering.md    L4 Harness
│   ├── overview.md               v1 historical version (merged essay)
│   └── html/                     rendered pages
├── interview-prep/              interview questions and job-search research
│   ├── jd-requirements.md        JD research (30+ roles in China and abroad)
│   ├── interview-questions.md    92 questions × 11 categories
│   └── html/                     rendered pages (interactive glossary popovers)
├── translations/                Chinese translations of classic essays
│   ├── building-effective-agents.md
│   ├── effective-harnesses-for-long-running-agents.md
│   ├── multi-agent-research-system.md
│   ├── openai-harness-engineering.md
│   └── html/
└── knowledge-map.md / .html     this file (mind map)
```

Chinese filenames above are the default edition. Each chapter also has a `.en.md` sibling in the same directory, and a `.en.html` page under `html/`. Links in the rest of this page point at those English siblings.

---

## 2. The four engineering layers (the core skeleton)

> The most important structure in the repository: **prompt → context → agent → harness** is a four-layer stack of abstractions. Each layer is its own discipline, and a production system needs all four.

### 🎯 Layer 1: Prompt engineering

**Core question:** How do you get an LLM to produce controllable, high-quality output in a single call?

| Topic | Key concepts | Chapter / questions |
|---|---|---|
| **Transformer foundations** | Attention / Multi-head / Position encoding | [L1 §1.1](engineering-foundations/prompt-engineering.en.md#s1-1), [A1–A3](interview-prep/interview-questions.en.md#a1) |
| **Training stack** | Pretrain → Post-train → SFT → RLHF/DPO/GRPO/RLVR | [L1 §1.2](engineering-foundations/prompt-engineering.en.md#s1-2), [B1–B10](interview-prep/interview-questions.en.md#b1) |
| **Inference and decoding** | Greedy / Top-k / Top-p / Temperature / Beam | [L1 §1.3](engineering-foundations/prompt-engineering.en.md#s1-3) |
| **Classic prompting techniques** | Zero-shot / Few-shot / **CoT** / **Self-Consistency** / **ToT** / **ReAct** / **Self-Refine** / **Reflexion** / Long CoT | [L1 §1.4](engineering-foundations/prompt-engineering.en.md#s1-4), [C1–C9](interview-prep/interview-questions.en.md#cat-c) |
| **Automatic optimization** | DSPy / OPRO / APE / prompt versioning | [L1 §1.5](engineering-foundations/prompt-engineering.en.md#s1-5), [C8](interview-prep/interview-questions.en.md#c8) |
| **Evaluation and red teaming** | LLM-as-judge / Prompt Injection / Jailbreak | [L1 §1.6](engineering-foundations/prompt-engineering.en.md#s1-6), [C7](interview-prep/interview-questions.en.md#c7), [I1–I3](interview-prep/interview-questions.en.md#cat-i) |

**Classic papers / essays:**

- Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Wei et al. 2022 [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- Wang et al. 2022 [Self-Consistency](https://arxiv.org/abs/2203.11171)
- Yao et al. 2022 [ReAct](https://arxiv.org/abs/2210.03629)
- Yao et al. 2023 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- Madaan et al. 2023 [Self-Refine](https://arxiv.org/abs/2303.17651)
- Shinn et al. 2023 [Reflexion](https://arxiv.org/abs/2303.11366)
- Khattab et al. 2023 [DSPy](https://arxiv.org/abs/2310.03714)

---

### 🧱 Layer 2: Context engineering

**Core question:** How do you fit the most relevant information into a limited context window?

| Topic | Key concepts | Chapter / questions |
|---|---|---|
| **Physics of long context** | KV cache / Attention sink / Lost in the Middle / RULER | [L2 §2.2](engineering-foundations/context-engineering.en.md#s2-2)–[§2.3](engineering-foundations/context-engineering.en.md#s2-3), [A5](interview-prep/interview-questions.en.md#a5), [A10](interview-prep/interview-questions.en.md#a10), [D8](interview-prep/interview-questions.en.md#d8) |
| **Inference acceleration** | PagedAttention / FlashAttention / vLLM / Continuous batching / Speculative decoding / Prefix caching | [A4](interview-prep/interview-questions.en.md#a4), [A6–A9](interview-prep/interview-questions.en.md#a6), [A11 prefill/decode](interview-prep/interview-questions.en.md#a11) |
| **Full-stack RAG** | Chunking / Embedding / **Hybrid Search (BM25 + Dense)** / **Reranker** / Vector DB / **RAGAS** | [L2 §2.4](engineering-foundations/context-engineering.en.md#s2-4), [D1–D10](interview-prep/interview-questions.en.md#cat-d) |
| **RAG variants** | GraphRAG / Self-RAG / CRAG / Agentic RAG / Contextual Retrieval | [D6](interview-prep/interview-questions.en.md#d6), [D2](interview-prep/interview-questions.en.md#d2) |
| **Embedding model selection** | BGE-M3 / Cohere v4 / Voyage-3 / OpenAI text-3 / Qwen3-Embedding | [D3](interview-prep/interview-questions.en.md#d3), [D9](interview-prep/interview-questions.en.md#d9) |
| **Memory systems** | MemGPT / Letta / Mem0 / Zep / working vs episodic vs semantic memory | [L2 §2.5](engineering-foundations/context-engineering.en.md#s2-5), [F1–F8](interview-prep/interview-questions.en.md#cat-f) |
| **Context management** | sliding window / summarization / sub-agent isolation / progress files / memory-corruption defense | [L2 §2.6](engineering-foundations/context-engineering.en.md#s2-6), [F5–F7](interview-prep/interview-questions.en.md#f5), [J3](interview-prep/interview-questions.en.md#j3) |
| **MCP / A2A protocols** | Resources / Tools / Prompts / cross-vendor collaboration | [L2 §2.7](engineering-foundations/context-engineering.en.md#s2-7), [E5](interview-prep/interview-questions.en.md#e5) |

**Classic papers / essays:**

- Kwon et al. 2023 [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- Dao et al. 2022 [FlashAttention](https://arxiv.org/abs/2205.14135)
- Liu et al. 2023 [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Hsieh et al. 2024 [RULER](https://arxiv.org/abs/2404.06654)
- Anthropic [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Packer et al. 2023 [MemGPT](https://arxiv.org/abs/2310.08560)
- Mem0 Team 2025 [Mem0](https://arxiv.org/abs/2504.19413)

---

### 🤖 Layer 3: Agent engineering

**Core question:** How do you let an LLM decide on its own, call tools, and finish a task across many steps?

| Topic | Key concepts | Chapter / questions |
|---|---|---|
| **Defining an agent** | LLM + tools + memory + loop / the Workflow vs Agent distinction | [L3 §1](engineering-foundations/agent-engineering.en.md#s1)–[§2](engineering-foundations/agent-engineering.en.md#s2), [E1](interview-prep/interview-questions.en.md#e1) |
| **Anthropic's 6 patterns** | Augmented LLM / Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer | [L3 §3](engineering-foundations/agent-engineering.en.md#s3), [E2](interview-prep/interview-questions.en.md#e2) |
| **How agent paradigms evolved** | ReAct → Plan-and-Execute → Reflexion → long-horizon autonomy | [L3 §2](engineering-foundations/agent-engineering.en.md#s2), [E3](interview-prep/interview-questions.en.md#e3) |
| **Agent frameworks** | LangChain / LangGraph / LlamaIndex / Haystack / Anthropic SDK | [L3 §5](engineering-foundations/agent-engineering.en.md#s5) |
| **Multi-agent collaboration** | Supervisor (recommended) / Hierarchical / Swarm (anti-pattern) | [L3 §6](engineering-foundations/agent-engineering.en.md#s6), [E4](interview-prep/interview-questions.en.md#e4), [J5](interview-prep/interview-questions.en.md#j5) |
| **Tool use / function calling** | schema design / granularity / deny-first / error as prompt / BFCL | [E8](interview-prep/interview-questions.en.md#e8) |
| **Computer Use / Browser Agent** | Anthropic Computer Use / Browser Use / DOM augmentation | [E9](interview-prep/interview-questions.en.md#e9) |
| **Four major agent failure modes** | one-shot / declaring victory too early / marking features complete too early / environment hard to start | [E6](interview-prep/interview-questions.en.md#e6) |
| **Loop of Death** | retry loops / goal drift / mutual contradiction / format hallucination | [E10](interview-prep/interview-questions.en.md#e10) |
| **Benchmark hacking** | SWE-bench / WebArena / GAIA / tau-bench can all be gamed for score | [E11](interview-prep/interview-questions.en.md#e11), [K1](interview-prep/interview-questions.en.md#k1) |

**Classic papers / essays:**

- Anthropic 2024 [Building Effective Agents](translations/building-effective-agents.md) ([original](https://www.anthropic.com/research/building-effective-agents))
- Anthropic 2025 [Multi-Agent Research System](translations/multi-agent-research-system.md) ([original](https://www.anthropic.com/engineering/built-multi-agent-research-system))
- Yao et al. 2022 [ReAct](https://arxiv.org/abs/2210.03629)
- Shinn et al. 2023 [Reflexion](https://arxiv.org/abs/2303.11366)

---

### 🛡️ Layer 4: Harness engineering

**Core question:** How do you engineer the environment so the agent is physically unable to repeat a mistake?

**Hashimoto, in his own words:** *"anytime you find an agent makes a mistake, take the time to engineer a solution such that the agent never makes that mistake again."*

| Topic | Key concepts | Chapter / questions |
|---|---|---|
| **Naming the harness, and where it sits** | Hashimoto named it in 2026-02 / the boundary with the model + the agent + context | [L4 §3.1](engineering-foundations/harness-engineering.en.md#s3-1)–[§3.2](engineering-foundations/harness-engineering.en.md#s3-2), [E7](interview-prep/interview-questions.en.md#e7) |
| **Two core laws** | **Strong constraints** (lint / types / sandbox / CI hard-fail) + **self-healing loop** (verifier + checkpoint) | [L4 §3.3](engineering-foundations/harness-engineering.en.md#s3-3), [J4](interview-prep/interview-questions.en.md#j4) |
| **Four pillars** | context architecture / agent specialization / persistent memory / structured execution | [E12](interview-prep/interview-questions.en.md#e12), [L4 §3.3](engineering-foundations/harness-engineering.en.md#s3-3) |
| **Supporting infrastructure** | MCP / Sandbox (E2B / Firecracker / gVisor) / Guardrails (NeMo) | [L4 §3.3](engineering-foundations/harness-engineering.en.md#s3-3), [G7](interview-prep/interview-questions.en.md#g7), [I4](interview-prep/interview-questions.en.md#i4) |
| **Production-stack reference architecture** | full 8-layer stack + role matrix + tool-permission isolation | [L4 §3.4–§3.4.5](engineering-foundations/harness-engineering.en.md#s3-4) |
| **Maturity model L0–L4** | diagnose for yourself which level your harness is on | [L4 §3.4.6](engineering-foundations/harness-engineering.en.md#s3-4-6) |
| **2026 side-by-side of mainstream LLMs** | cost / capability / context across generations of Claude / GPT / Gemini / DeepSeek / Llama | [L4 §3.5.5](engineering-foundations/harness-engineering.en.md#s3-5-5) |
| **Eight cost-optimization levers** | Prompt caching / model routing / semantic cache / output control / ... | [L4 §3.5.6](engineering-foundations/harness-engineering.en.md#s3-5-6), [G6](interview-prep/interview-questions.en.md#g6) |
| **Scenario selection checklist** | selection routes for internal tools / customer support / coding / multi-agent business systems | [L4 §3.6](engineering-foundations/harness-engineering.en.md#s3-6) |

**Classic papers / essays:**

- Hashimoto [My AI Adoption Journey · Step 5](https://mitchellh.com/writing/my-ai-adoption-journey)
- OpenAI [Harness Engineering · Leveraging Codex in an Agent-First World](translations/openai-harness-engineering.md) ([original](https://openai.com/index/harness-engineering/))
- Anthropic 2025 [Effective Harnesses for Long-Running Agents](translations/effective-harnesses-for-long-running-agents.md) ([original](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents))
- Augment Code [Harness Engineering for AI Coding Agents](https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents)

---

## 3. Interview questions: 92 questions / 11 categories

| Category | Count | Difficulty | Core coverage | Entry |
|---|---|---|---|---|
| **A. LLM fundamentals** | 11 | 1⭐ 8⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / long context / inference optimization / prefill·decode | [→ A](interview-prep/interview-questions.en.md#cat-a) |
| **B. Training / fine-tuning** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 emergence | [→ B](interview-prep/interview-questions.en.md#cat-b) |
| **C. Prompt engineering** | 9 | 1⭐ 7⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / automatic prompt optimization | [→ C](interview-prep/interview-questions.en.md#cat-c) |
| **D. RAG** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS | [→ D](interview-prep/interview-questions.en.md#cat-d) |
| **E. Agent** | 12 | 1⭐ 6⭐⭐ 5⭐⭐⭐ | **Core:** 6 patterns / multi-agent / failure modes / harness | [→ E](interview-prep/interview-questions.en.md#cat-e) |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / Letta / Mem0 / progress files / corruption | [→ F](interview-prep/interview-questions.en.md#cat-f) |
| **G. System design** | 8 | 0⭐ 1⭐⭐ 7⭐⭐⭐ | customer support / coding / search / gateway / high concurrency / cost / sandbox / closed loop | [→ G](interview-prep/interview-questions.en.md#cat-g) |
| **H. Evaluation** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | judge bias / pairwise / benchmark hacking / drift | [→ H](interview-prep/interview-questions.en.md#cat-h) |
| **I. Safety / guardrails** | 6 | 0⭐ 1⭐⭐ 5⭐⭐⭐ | OWASP / Prompt Injection / NeMo / EU AI Act | [→ I](interview-prep/interview-questions.en.md#cat-i) |
| **J. Behavioral / practice** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto closed loop / 0-to-1 rollout path / debug hallucination | [→ J](interview-prep/interview-questions.en.md#cat-j) |
| **K. Agent evaluation** | 5 | 0⭐ 2⭐⭐ 3⭐⭐⭐ | general benchmarks / three-layer trajectory / tool-use / production monitoring / **business domains** | [→ K](interview-prep/interview-questions.en.md#cat-k) |

### Review paths by difficulty

**⭐ Introductory (11 questions):** A1, A7, B1, B7, C1, C6, D1, E1 (work through them in this order to get familiar with the terms)

**⭐⭐ Intermediate (50 questions):** the intermediate questions in each category. Suggested order: A → B → C → D → E, threading the concepts into one system

**⭐⭐⭐ Hard (30 questions):** staff / architect questions. G/J/K are the main arena — be able to answer from a system-design point of view

---

## 4. Chinese translations of classic essays

| Essay | Source | Core idea | Entry |
|---|---|---|---|
| **Building Effective Agents** | Anthropic 2024-12 | Workflow vs Agent / 6 patterns / simplicity first | [→ Chinese](translations/building-effective-agents.md) · [original](https://www.anthropic.com/research/building-effective-agents) |
| **Effective Harnesses for Long-Running Agents** | Anthropic 2025 | 4 failure modes and countermeasures / progress files / 5-step startup protocol | [→ Chinese](translations/effective-harnesses-for-long-running-agents.md) · [original](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) |
| **How we built our multi-agent research system** | Anthropic 2025 | Orchestrator-Workers / sub-agent isolation / verifier | [→ Chinese](translations/multi-agent-research-system.md) · [original](https://www.anthropic.com/engineering/built-multi-agent-research-system) |
| **Harness Engineering · Leveraging Codex in an Agent-First World** | OpenAI 2026 | Agent = Model + Harness / Humans steer, Agents execute | [→ Chinese](translations/openai-harness-engineering.md) · [original](https://openai.com/index/harness-engineering/) |

---

## 5. Glossary (83 entries × 10 topics)

> Every HTML page embeds the full glossary in the lower-right corner. Click an orange marked term in the text and a popover opens without leaving the page.

### Software engineering basics (10)

- [lint](interview-prep/interview-questions.en.md#g-lint)
- [type check](interview-prep/interview-questions.en.md#g-type-check)
- [CI/CD](interview-prep/interview-questions.en.md#g-cicd)
- [sandbox](interview-prep/interview-questions.en.md#g-sandbox)
- [canary release](interview-prep/interview-questions.en.md#g-canary)
- [A/B test](interview-prep/interview-questions.en.md#g-abtest)
- [HITL / human-in-the-loop](interview-prep/interview-questions.en.md#g-hitl)
- [SLO / SLI / SLA / Error Budget](interview-prep/interview-questions.en.md#g-slo)
- [TTL](interview-prep/interview-questions.en.md#g-ttl)
- [GMV / UV / CSAT / NPS / conversion rate](interview-prep/interview-questions.en.md#g-biz-metrics)

### LLM fundamentals (9)

- [token](interview-prep/interview-questions.en.md#g-token)
- [embedding](interview-prep/interview-questions.en.md#g-embedding)
- [context window](interview-prep/interview-questions.en.md#g-context-window)
- [prefill / decode](interview-prep/interview-questions.en.md#g-prefill-decode)
- [KV cache](interview-prep/interview-questions.en.md#g-kv-cache)
- [autoregressive](interview-prep/interview-questions.en.md#g-autoregressive)
- [logit](interview-prep/interview-questions.en.md#g-logit)
- [temperature](interview-prep/interview-questions.en.md#g-temperature)
- [attention sink](interview-prep/interview-questions.en.md#g-attention-sink)

### Transformer architecture (5)

- [LayerNorm / RMSNorm](interview-prep/interview-questions.en.md#g-layernorm-rmsnorm)
- [RoPE](interview-prep/interview-questions.en.md#g-rope)
- [YaRN / NTK](interview-prep/interview-questions.en.md#g-yarn)
- [GQA / MQA](interview-prep/interview-questions.en.md#g-gqa-mqa)
- [MoE](interview-prep/interview-questions.en.md#g-moe)
- [multi-head attention](interview-prep/interview-questions.en.md#g-multi-head-attention)

### Inference acceleration (8)

- [FlashAttention](interview-prep/interview-questions.en.md#g-flash-attention)
- [PagedAttention](interview-prep/interview-questions.en.md#g-paged-attention)
- [vLLM](interview-prep/interview-questions.en.md#g-vllm)
- [continuous batching](interview-prep/interview-questions.en.md#g-continuous-batching)
- [speculative decoding](interview-prep/interview-questions.en.md#g-spec-decode)
- [prefix caching](interview-prep/interview-questions.en.md#g-prefix-caching)
- [TTFT](interview-prep/interview-questions.en.md#g-ttft)
- [TPOT / ITL](interview-prep/interview-questions.en.md#g-tpot)

### Quantization (4)

- [FP16 / BF16 / FP8](interview-prep/interview-questions.en.md#g-fp16-bf16-fp8)
- [INT4 / INT8 / NF4](interview-prep/interview-questions.en.md#g-int4-int8)
- [AWQ](interview-prep/interview-questions.en.md#g-awq)
- [GPTQ](interview-prep/interview-questions.en.md#g-gptq)

### Fine-tuning and alignment (10)

- [SFT](interview-prep/interview-questions.en.md#g-sft)
- [LoRA](interview-prep/interview-questions.en.md#g-lora)
- [QLoRA](interview-prep/interview-questions.en.md#g-qlora)
- [RLHF](interview-prep/interview-questions.en.md#g-rlhf)
- [RM / Reward Model](interview-prep/interview-questions.en.md#g-rm)
- [DPO](interview-prep/interview-questions.en.md#g-dpo)
- [PPO](interview-prep/interview-questions.en.md#g-ppo)
- [GRPO](interview-prep/interview-questions.en.md#g-grpo)
- [RLVR](interview-prep/interview-questions.en.md#g-rlvr)
- [catastrophic forgetting](interview-prep/interview-questions.en.md#g-catastrophic-forgetting)

### Prompting and reasoning paradigms (7)

- [CoT](interview-prep/interview-questions.en.md#g-cot)
- [ToT](interview-prep/interview-questions.en.md#g-tot)
- [ReAct](interview-prep/interview-questions.en.md#g-react)
- [Reflexion](interview-prep/interview-questions.en.md#g-reflexion)
- [Self-Refine](interview-prep/interview-questions.en.md#g-self-refine)
- [self-consistency](interview-prep/interview-questions.en.md#g-self-consistency)
- [few-shot / zero-shot](interview-prep/interview-questions.en.md#g-few-shot)

### RAG (7)

- [RAG](interview-prep/interview-questions.en.md#g-rag)
- [chunking](interview-prep/interview-questions.en.md#g-chunking)
- [BM25](interview-prep/interview-questions.en.md#g-bm25)
- [hybrid search](interview-prep/interview-questions.en.md#g-hybrid-search)
- [reranker / cross-encoder](interview-prep/interview-questions.en.md#g-reranker)
- [vector DB](interview-prep/interview-questions.en.md#g-vector-db)
- [mainstream embedding-model selection](interview-prep/interview-questions.en.md#g-embed-models)
- [RAG variants (GraphRAG / Self-RAG / CRAG / Agentic RAG)](interview-prep/interview-questions.en.md#g-rag-variants)
- [RAGAS](interview-prep/interview-questions.en.md#g-ragas)

### Agent engineering (10)

- [harness engineering](interview-prep/interview-questions.en.md#g-harness)
- [MCP / Model Context Protocol](interview-prep/interview-questions.en.md#g-mcp)
- [tool use / function calling](interview-prep/interview-questions.en.md#g-tool-use)
- [Computer Use / Browser Agent](interview-prep/interview-questions.en.md#g-computer-use)
- [sub-agent](interview-prep/interview-questions.en.md#g-sub-agent)
- [AGENTS.md / CLAUDE.md](interview-prep/interview-questions.en.md#g-agents-md)
- [deny-first](interview-prep/interview-questions.en.md#g-deny-first)
- [progress file](interview-prep/interview-questions.en.md#g-progress-file)
- [Loop of Death](interview-prep/interview-questions.en.md#g-loop-of-death)
- [Memory Corruption](interview-prep/interview-questions.en.md#g-memory-corruption)
- [LangChain / LangGraph and related frameworks](interview-prep/interview-questions.en.md#g-langchain)

### Evaluation and safety (7)

- [Eval / Evaluation](interview-prep/interview-questions.en.md#g-eval)
- [LLM-as-judge](interview-prep/interview-questions.en.md#g-llm-judge)
- [mainstream LLM eval benchmarks](interview-prep/interview-questions.en.md#g-llm-benchmarks)
- [Trace + Eval tool stack](interview-prep/interview-questions.en.md#g-tracing-tools)
- [Lost in the Middle](interview-prep/interview-questions.en.md#g-lost-in-middle)
- [prompt injection](interview-prep/interview-questions.en.md#g-prompt-injection)
- [hallucination](interview-prep/interview-questions.en.md#g-hallucination)
- [guardrails](interview-prep/interview-questions.en.md#g-guardrails)
- [jailbreak](interview-prep/interview-questions.en.md#g-jailbreak)

---

## 6. Suggested review paths

### Path A: New graduate / beginner (2–4 weeks)

```
Week 1: Engineering foundation concepts
  └─ L1 §1.1–§1.3 + glossary (LLM basics + Transformer architecture) + A1–A11

Week 2: Prompting + RAG introduction
  └─ L1 §1.4 + L2 §2.4 + C1–C9 + D1–D5

Week 3: Agent concepts
  └─ L3 §1–§3 + E1, E2, E3, E5, E8 + translation "Building Effective Agents"

Week 4: Safety + practice
  └─ L1 §1.6 + I1, I2 + J7 (0-to-1 rollout path)
```

### Path B: Mid-level (3–6 years, 1–2 months)

```
Foundations (already in place) → deepen:
  - All of categories A–F
  - Read L1–L4 straight through
  - Parts of G/H (system design + evaluation)
  - Read all 4 translations
  - Prepare category J behavioral answers from your own projects
```

### Path C: Senior / staff (depth + breadth)

```
- All 11 categories (A–K)
- L4 §3.4 production-stack architecture + §3.4.6 maturity-model self-check
- §3.5.6 eight cost-optimization levers
- §3.6 scenario selection checklist
- Memorize: Hashimoto's and Anthropic's own wording on harnesses
- Have ready: 1–2 quantified project cases (GMV / conversion rate / accuracy, with numbers)
```

### Path D: Scenario-driven (tailored to the target role)

| Target role | Focus chapters |
|---|---|
| **LLM application SaaS (customer support / document QA)** | L2 RAG + L3 Agent + G1 (customer support) + I (safety) + K5 (business evaluation) |
| **AI coding tools (Cursor / Devin / Augment class)** | All of L3 + L4 + G2 (Coding Agent) + E7 (harness) + translations of *Effective Harnesses* and *OpenAI Harness Engineering* |
| **Multi-agent business systems (e-commerce / travel / finance)** | L3 §6 multi-agent + E4/E10 + K5 (business-domain evaluation) + G1/G3 + J1 STAR + prepare numbers from your own projects |
| **Infrastructure / inference / training** | A4–A9 + A11 inference acceleration + B1–B10 training stack + G4/G5 (gateway / high concurrency) |
| **Security / red team** | C7 + I1–I6 + L1 §1.6 + L4 three-layer defense |

---

## 7. Version and maintenance

- **Research date:** 2026-05-02
- **Category K addition:** 2026-05-14 (business-domain agent evaluation)
- **Glossary version:** 83 entries × 10 topics
- **This file:** knowledge-map.en.md (kept in sync with the HTML). Chinese source: knowledge-map.md
- **Update guidance:** when a new model, paper, or tool appears, add it to the matching section without breaking the existing structure

---

*This knowledge map threads together everything in the repository. The rendered HTML ([knowledge-map.en.html](knowledge-map.en.html); Chinese [knowledge-map.html](knowledge-map.html)) supports collapsing, cross-page jumps, and glossary popovers.*
