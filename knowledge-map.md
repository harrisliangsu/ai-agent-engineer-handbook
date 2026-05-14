# 🧠 AI Agent / LLM 应用工程师 全景知识图谱

> 本仓库的"思维导图 / 复习地图"——把工程四层、91 道面试题、4 篇经典翻译、82 条术语速查表全部串起来，提供完整的知识体系坐标。

**怎么用**：

- 复习时按"工程四层 → 面试题分类 → 经典翻译 → 术语速查"四个 lens 交叉刷
- 看到不熟悉的术语跳到术语速查表展开
- 看到不熟的题跳到 [`interview-prep/`](interview-prep/) 对应章节
- 看到陌生的概念跳到 [`engineering-foundations/`](engineering-foundations/) 对应小节

---

## 一、仓库结构总览

```
ai-agent-engineer-handbook/
├── engineering-foundations/     工程四层抽象（核心知识体系）
│   ├── prompt-engineering.md     L1 提示词
│   ├── context-engineering.md    L2 上下文
│   ├── agent-engineering.md      L3 Agent
│   ├── harness-engineering.md    L4 Harness
│   ├── overview.md               v1 历史版本（整合篇）
│   └── html/                     渲染版
├── interview-prep/              面试题与求职研究
│   ├── jd-requirements.md        JD 调研（30+ 国内外岗位）
│   ├── interview-questions.md    91 题 × 11 大类
│   └── html/                     渲染版（含交互式术语弹窗）
├── translations/                经典文章中译
│   ├── building-effective-agents.md
│   ├── effective-harnesses-for-long-running-agents.md
│   ├── multi-agent-research-system.md
│   ├── openai-harness-engineering.md
│   └── html/
└── knowledge-map.md / .html     本文件（思维导图）
```

---

## 二、工程四层抽象（核心知识骨架）

> 这是整个仓库最重要的知识结构：**提示词 → 上下文 → Agent → Harness** 是四层递进抽象，每层独立成科但生产系统四层都需要。

### 🎯 Layer 1: 提示词工程 / Prompt Engineering

**核心问题**：如何让 LLM 在单次调用中产出可控的高质量输出？

| 主题 | 关键概念 | 对应章节 / 题目 |
|---|---|---|
| **Transformer 基础** | Attention / Multi-head / Position encoding | [L1 §1.1](engineering-foundations/prompt-engineering.md#11-起点transformer-与注意力机制)、[A1-A3](interview-prep/interview-questions.md#a1-为什么-transformer-用-layer-norm-而不是-batch-norm) |
| **训练栈** | Pretrain → Post-train → SFT → RLHF/DPO/GRPO/RLVR | [L1 §1.2](engineering-foundations/prompt-engineering.md#12-llm-训练栈预训练--后训练--微调)、[B1-B10](interview-prep/interview-questions.md#b1-预训练后训练任务微调三者怎么拆分) |
| **推理与解码** | Greedy / Top-k / Top-p / Temperature / Beam | [L1 §1.3](engineering-foundations/prompt-engineering.md#13-推理栈与解码策略) |
| **经典提示词技术** | Zero-shot / Few-shot / **CoT** / **Self-Consistency** / **ToT** / **ReAct** / **Self-Refine** / **Reflexion** / Long CoT | [L1 §1.4](engineering-foundations/prompt-engineering.md#14-提示词工程的经典技术)、[C1-C9](interview-prep/interview-questions.md#c-提示词工程9-题) |
| **自动化优化** | DSPy / OPRO / APE / Prompt 版本化 | [L1 §1.5](engineering-foundations/prompt-engineering.md#15-自动化与高级提示词)、[C8](interview-prep/interview-questions.md#c8-dspy--opro--ape-自动-prompt-优化各是什么思路) |
| **评估与红队** | LLM-as-judge / Prompt Injection / Jailbreak | [L1 §1.6](engineering-foundations/prompt-engineering.md#16-评估红队与提示词工程的黄昏)、[C7](interview-prep/interview-questions.md#c7-prompt-injection-有哪些攻击面防御纵深怎么做)、[I1-I3](interview-prep/interview-questions.md#i-安全--护栏6-题) |

**经典论文 / 文章**：

- Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Wei et al. 2022 [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- Wang et al. 2022 [Self-Consistency](https://arxiv.org/abs/2203.11171)
- Yao et al. 2022 [ReAct](https://arxiv.org/abs/2210.03629)
- Yao et al. 2023 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- Madaan et al. 2023 [Self-Refine](https://arxiv.org/abs/2303.17651)
- Shinn et al. 2023 [Reflexion](https://arxiv.org/abs/2303.11366)
- Khattab et al. 2023 [DSPy](https://arxiv.org/abs/2310.03714)

---

### 🧱 Layer 2: 上下文工程 / Context Engineering

**核心问题**：如何在有限的 context window 里塞进最相关的信息？

| 主题 | 关键概念 | 对应章节 / 题目 |
|---|---|---|
| **长上下文物理学** | KV cache / Attention sink / Lost in the Middle / RULER | [L2 §2.2-§2.3](engineering-foundations/context-engineering.md#22-长上下文的物理学)、[A5](interview-prep/interview-questions.md#a5-lost-in-the-middle-是什么工程上怎么缓解)、[A10](interview-prep/interview-questions.md#a10-上下文越长效果越好吗工程上-128k-真能用吗)、[D8](interview-prep/interview-questions.md#d8-长上下文1mvs-rag到底怎么选) |
| **推理加速** | PagedAttention / FlashAttention / vLLM / Continuous batching / Speculative decoding / Prefix caching | [A4](interview-prep/interview-questions.md#a4-vllm-的-pagedattention-解决了什么问题为什么吞吐能涨-24-倍)、[A6-A9](interview-prep/interview-questions.md#a6-speculative-decoding-原理什么场景收益最大) |
| **RAG 全栈** | Chunking / Embedding / **Hybrid Search (BM25 + Dense)** / **Reranker** / Vector DB / **RAGAS** | [L2 §2.4](engineering-foundations/context-engineering.md#24-rag-全栈选型)、[D1-D10](interview-prep/interview-questions.md#d-rag10-题) |
| **RAG 变体** | GraphRAG / Self-RAG / CRAG / Agentic RAG / Contextual Retrieval | [D6](interview-prep/interview-questions.md#d6-graphrag--self-rag--crag--agentic-rag-各解决什么问题)、[D2](interview-prep/interview-questions.md#d2-切分策略对比fixed--semantic--late-chunking-怎么选) |
| **Embedding 模型选型** | BGE-M3 / Cohere v4 / Voyage-3 / OpenAI text-3 / Qwen3-Embedding | [D3](interview-prep/interview-questions.md#d3-2026-年-embedding-模型怎么选bge-m3--cohere-v4--voyage-3--openai-text-3-取舍)、[D9](interview-prep/interview-questions.md#d9-中文-rag-有什么特殊性) |
| **Memory 系统** | MemGPT / Letta / Mem0 / Zep / 工作 vs 情景 vs 语义记忆 | [L2 §2.5](engineering-foundations/context-engineering.md#25-记忆系统完整工程指南)、[F1-F8](interview-prep/interview-questions.md#f-memory8-题) |
| **Context 管理** | 滑窗 / 摘要 / Sub-agent 隔离 / Progress 文件 / Memory Corruption 防御 | [L2 §2.6](engineering-foundations/context-engineering.md#26-上下文管理策略把-context-window-当作稀缺资源)、[F5-F7](interview-prep/interview-questions.md#f5-progress-文件--git-作为-memory-bridge-是什么为什么对长任务-agent-关键)、[J3](interview-prep/interview-questions.md#j3-agent-跑爆-context-怎么办) |
| **MCP / A2A 协议** | Resources / Tools / Prompts / 跨厂商协作 | [L2 §2.7](engineering-foundations/context-engineering.md#27-mcp--a2a-协议生态)、[E5](interview-prep/interview-questions.md#e5-mcpmodel-context-protocol是什么为什么-2026-年增速这么快) |

**经典论文 / 文章**：

- Kwon et al. 2023 [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- Dao et al. 2022 [FlashAttention](https://arxiv.org/abs/2205.14135)
- Liu et al. 2023 [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Hsieh et al. 2024 [RULER](https://arxiv.org/abs/2404.06654)
- Anthropic [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Packer et al. 2023 [MemGPT](https://arxiv.org/abs/2310.08560)
- Mem0 Team 2025 [Mem0](https://arxiv.org/abs/2504.19413)

---

### 🤖 Layer 3: Agent 工程 / Agent Engineering

**核心问题**：如何让 LLM 自主决策、调工具、跨多步完成任务？

| 主题 | 关键概念 | 对应章节 / 题目 |
|---|---|---|
| **Agent 定义** | LLM + 工具 + 记忆 + 循环 / Workflow vs Agent 区分 | [L3 §1-§2](engineering-foundations/agent-engineering.md#1-agent-的定义与坐标系)、[E1](interview-prep/interview-questions.md#e1-anthropic-building-effective-agents-把-agent-模式分成-workflow-和-agent-两类区别是什么为什么这个区分重要) |
| **Anthropic 6 大 Pattern** | Augmented LLM / Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer | [L3 §3](engineering-foundations/agent-engineering.md#3-agent-架构一种常见的-4-层拆分)、[E2](interview-prep/interview-questions.md#e2-anthropic-6-大-agentic-patterns-分别是什么什么场景用哪个) |
| **Agent 范式演进** | ReAct → Plan-and-Execute → Reflexion → 长程自治 | [L3 §2](engineering-foundations/agent-engineering.md#2-agent-范式的演进)、[E3](interview-prep/interview-questions.md#e3-react-loop--plan-and-execute--reflexion-区别) |
| **Agent 框架** | LangChain / LangGraph / LlamaIndex / Haystack / Anthropic SDK | [L3 §5](engineering-foundations/agent-engineering.md#5-主流-agent-框架对比2026) |
| **多 Agent 协作** | Supervisor (推荐) / Hierarchical / Swarm (反模式) | [L3 §6](engineering-foundations/agent-engineering.md#6-多-agent-协作模式)、[E4](interview-prep/interview-questions.md#e4-多-agent-协作模式-supervisor--hierarchical--swarm-怎么选)、[J5](interview-prep/interview-questions.md#j5-多-agent-协作冲突怎么处理) |
| **Tool Use / Function Calling** | Schema 设计 / 粒度 / Deny-first / Error as prompt / BFCL | [E8](interview-prep/interview-questions.md#e8-function-calling--tool-use-怎么设计才能让-agent-用得对) |
| **Computer Use / Browser Agent** | Anthropic Computer Use / Browser Use / DOM 增强 | [E9](interview-prep/interview-questions.md#e9-computer-use--browser-agent-怎么工作什么场景实用) |
| **Agent 4 大失败模式** | 一步到位 / 过早胜利 / 过早完成 / 环境启动困难 | [E6](interview-prep/interview-questions.md#e6-agent-4-种典型失败模式anthropic-总结及对策) |
| **Loop of Death** | 重试循环 / 目标漂移 / 互相否定 / 格式幻觉 | [E10](interview-prep/interview-questions.md#e10-90-的-loop-of-death-是怎么发生的怎么防) |
| **Benchmark hack** | SWE-bench / WebArena / GAIA / tau-bench 都可能被刷分 | [E11](interview-prep/interview-questions.md#e11-agent-benchmark-swe-bench--gaia--webarena-是怎么被-hack-的)、[K1](interview-prep/interview-questions.md#k1-agent-benchmark-全景tau-bench--swe-bench-verified--osworld--webarena--gaia--theagentcompany-各侧重什么) |

**经典论文 / 文章**：

- Anthropic 2024 [Building Effective Agents](translations/building-effective-agents.md)（[原文](https://www.anthropic.com/research/building-effective-agents)）
- Anthropic 2025 [Multi-Agent Research System](translations/multi-agent-research-system.md)
- Yao et al. 2022 [ReAct](https://arxiv.org/abs/2210.03629)
- Shinn et al. 2023 [Reflexion](https://arxiv.org/abs/2303.11366)

---

### 🛡️ Layer 4: Harness 工程 / Harness Engineering

**核心问题**：如何工程化环境，让 agent 物理上无法重复犯错？

**Hashimoto 原话**：*"anytime you find an agent makes a mistake, take the time to engineer a solution such that the agent never makes that mistake again."*

| 主题 | 关键概念 | 对应章节 / 题目 |
|---|---|---|
| **Harness 的命名与定位** | Hashimoto 2026-02 命名 / 与模型 + Agent + 上下文的边界 | [L4 §3.1-§3.2](engineering-foundations/harness-engineering.md#31-起源被命名的那一层东西)、[E7](interview-prep/interview-questions.md#e7-hashimoto-的-harness-engineering-核心思想是什么怎么落地) |
| **两条核心法则** | **强约束**（lint / type / sandbox / CI 硬失败）+ **自愈循环**（verifier + checkpoint） | [L4 §3.3](engineering-foundations/harness-engineering.md#33-驾驭层核心方法论--支撑设施)、[J4](interview-prep/interview-questions.md#j4-怎么让-agent-不再犯同样的错harness-engineering) |
| **四大支柱** | 上下文架构 / Agent 专业化 / 持久化记忆 / 结构化执行 | [E12](interview-prep/interview-questions.md#e12-agent-harness-包括哪些核心组件设计原则是什么)、[L4 §3.3](engineering-foundations/harness-engineering.md#33-驾驭层核心方法论--支撑设施) |
| **支撑设施** | MCP / Sandbox (E2B / Firecracker / gVisor) / Guardrails (NeMo) | [L4 §3.3](engineering-foundations/harness-engineering.md#33-驾驭层核心方法论--支撑设施)、[G7](interview-prep/interview-questions.md#g7-沙箱设计e2b--daytona--microvm--gvisor-怎么选)、[I4](interview-prep/interview-questions.md#i4-nemo-guardrails-5-类-rails) |
| **生产栈参考架构** | 8 层完整栈 + 角色矩阵 + 工具权限隔离 | [L4 §3.4-§3.4.5](engineering-foundations/harness-engineering.md#34-完整生产栈参考架构) |
| **成熟度模型 L0-L4** | 自我诊断你的 harness 在哪一级 | [L4 §3.4.6](engineering-foundations/harness-engineering.md#346-harness-成熟度模型-l0l4自我诊断你在哪一级) |
| **2026 主流 LLM 横评** | Claude / GPT / Gemini / DeepSeek / Llama 各代成本 / 能力 / 上下文 | [L4 §3.5.5](engineering-foundations/harness-engineering.md#355-2026-主流-llm-横向对照表) |
| **Cost 优化 8 杠杆** | Prompt caching / Model routing / Semantic cache / Output 控制 / ... | [L4 §3.5.6](engineering-foundations/harness-engineering.md#356-cost-优化系统化8-个杠杆按-roi-排序)、[G6](interview-prep/interview-questions.md#g6-成本优化prompt-caching--model-routing--semantic-cache-怎么省-80) |
| **场景选型清单** | 内部工具 / 客服 / 编码 / 多 agent 业务系统 选型路线 | [L4 §3.6](engineering-foundations/harness-engineering.md#36-recommendations三类典型场景的选型清单) |

**经典论文 / 文章**：

- Hashimoto [My AI Adoption Journey · Step 5](https://mitchellh.com/writing/my-ai-adoption-journey)
- OpenAI [Harness Engineering · Leveraging Codex in an Agent-First World](translations/openai-harness-engineering.md)（[原文](https://openai.com/index/harness-engineering/)）
- Anthropic 2025 [Effective Harnesses for Long-Running Agents](translations/effective-harnesses-for-long-running-agents.md)（[原文](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)）
- Augment Code [Harness Engineering for AI Coding Agents](https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents)

---

## 三、面试题 91 题 / 11 大类

| 分类 | 题数 | 难度 | 核心覆盖 | 入口 |
|---|---|---|---|---|
| **A. LLM 基础** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / 长上下文 / 推理优化 | [→ A](interview-prep/interview-questions.md#a-llm-基础10-题) |
| **B. 训练 / 微调** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 涌现 | [→ B](interview-prep/interview-questions.md#b-训练--微调10-题) |
| **C. 提示词工程** | 9 | 1⭐ 7⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / 自动 prompt 优化 | [→ C](interview-prep/interview-questions.md#c-提示词工程9-题) |
| **D. RAG** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS | [→ D](interview-prep/interview-questions.md#d-rag10-题) |
| **E. Agent** | 12 | 1⭐ 6⭐⭐ 5⭐⭐⭐ | **核心**：6 patterns / 多 agent / 失败模式 / harness | [→ E](interview-prep/interview-questions.md#e-agent12-题核心分类) |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / Letta / Mem0 / progress 文件 / corruption | [→ F](interview-prep/interview-questions.md#f-memory8-题) |
| **G. 系统设计** | 8 | 0⭐ 1⭐⭐ 7⭐⭐⭐ | 客服 / coding / 搜索 / 网关 / 高并发 / 成本 / 沙箱 / 闭环 | [→ G](interview-prep/interview-questions.md#g-系统设计8-题) |
| **H. 评估** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | judge bias / pairwise / benchmark hack / 漂移 | [→ H](interview-prep/interview-questions.md#h-评估6-题) |
| **I. 安全 / 护栏** | 6 | 0⭐ 1⭐⭐ 5⭐⭐⭐ | OWASP / Prompt Injection / NeMo / EU AI Act | [→ I](interview-prep/interview-questions.md#i-安全--护栏6-题) |
| **J. 行为题 / 实战** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto 闭环 / 0-1 落地路线 / debug hallucination | [→ J](interview-prep/interview-questions.md#j-系统行为题--实战题7-题) |
| **K. Agent 评测** | 5 | 0⭐ 2⭐⭐ 3⭐⭐⭐ | 通用 benchmark / trajectory 三层 / tool-use / 生产监控 / **业务领域** | [→ K](interview-prep/interview-questions.md#k-agent-评测5-题) |

### 按难度分类的复习路径

**⭐ 入门（11 题）**：A1, A7, B1, B7, C1, C6, D1, E1（按这个顺序刷一遍熟悉术语）

**⭐⭐ 中等（50 题）**：每一类的中等题，建议按 A → B → C → D → E 顺序，把概念串成体系

**⭐⭐⭐ 困难（30 题）**：staff / 架构师题，G/J/K 是主战场——能从系统设计角度回答

---

## 四、经典文章中文翻译

| 文章 | 来源 | 核心观点 | 入口 |
|---|---|---|---|
| **Building Effective Agents** | Anthropic 2024-12 | Workflow vs Agent / 6 大 pattern / 简单优先 | [→ 中文](translations/building-effective-agents.md) · [原文](https://www.anthropic.com/research/building-effective-agents) |
| **Effective Harnesses for Long-Running Agents** | Anthropic 2025 | 4 大失败模式 + 对策 / progress 文件 / 5 步开机协议 | [→ 中文](translations/effective-harnesses-for-long-running-agents.md) · [原文](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) |
| **How we built our multi-agent research system** | Anthropic 2025 | Orchestrator-Workers / Sub-agent 隔离 / Verifier | [→ 中文](translations/multi-agent-research-system.md) · [原文](https://www.anthropic.com/engineering/built-multi-agent-research-system) |
| **Harness Engineering · Leveraging Codex in an Agent-First World** | OpenAI 2026 | Agent = Model + Harness / Humans steer, Agents execute | [→ 中文](translations/openai-harness-engineering.md) · [原文](https://openai.com/index/harness-engineering/) |

---

## 五、术语速查表（82 词条 × 10 主题）

> 所有 HTML 页面右下角都内嵌完整术语库，点击文中橙色标记词条即弹浮窗，不离开当前位置。

### 软件工程基础（10）

- [lint](interview-prep/html/interview-questions.html#g-lint)
- [type check 类型检查](interview-prep/html/interview-questions.html#g-type-check)
- [CI/CD](interview-prep/html/interview-questions.html#g-cicd)
- [sandbox 沙箱](interview-prep/html/interview-questions.html#g-sandbox)
- [canary 金丝雀发布](interview-prep/html/interview-questions.html#g-canary)
- [A/B test](interview-prep/html/interview-questions.html#g-abtest)
- [HITL / human-in-the-loop](interview-prep/html/interview-questions.html#g-hitl)
- [SLO / SLI / SLA / Error Budget](interview-prep/html/interview-questions.html#g-slo)
- [TTL](interview-prep/html/interview-questions.html#g-ttl)
- [GMV / UV / CSAT / NPS / 转化率](interview-prep/html/interview-questions.html#g-biz-metrics)

### LLM 基础概念（9）

- [token](interview-prep/html/interview-questions.html#g-token)
- [embedding](interview-prep/html/interview-questions.html#g-embedding)
- [context window 上下文窗口](interview-prep/html/interview-questions.html#g-context-window)
- [prefill / decode](interview-prep/html/interview-questions.html#g-prefill-decode)
- [KV cache](interview-prep/html/interview-questions.html#g-kv-cache)
- [autoregressive 自回归](interview-prep/html/interview-questions.html#g-autoregressive)
- [logit](interview-prep/html/interview-questions.html#g-logit)
- [temperature](interview-prep/html/interview-questions.html#g-temperature)
- [attention sink](interview-prep/html/interview-questions.html#g-attention-sink)

### Transformer 架构（5）

- [LayerNorm / RMSNorm](interview-prep/html/interview-questions.html#g-layernorm-rmsnorm)
- [RoPE](interview-prep/html/interview-questions.html#g-rope)
- [YaRN / NTK](interview-prep/html/interview-questions.html#g-yarn)
- [GQA / MQA](interview-prep/html/interview-questions.html#g-gqa-mqa)
- [MoE](interview-prep/html/interview-questions.html#g-moe)
- [multi-head attention](interview-prep/html/interview-questions.html#g-multi-head-attention)

### 推理加速（7）

- [FlashAttention](interview-prep/html/interview-questions.html#g-flash-attention)
- [PagedAttention](interview-prep/html/interview-questions.html#g-paged-attention)
- [vLLM](interview-prep/html/interview-questions.html#g-vllm)
- [continuous batching](interview-prep/html/interview-questions.html#g-continuous-batching)
- [speculative decoding](interview-prep/html/interview-questions.html#g-spec-decode)
- [prefix caching](interview-prep/html/interview-questions.html#g-prefix-caching)
- [TTFT](interview-prep/html/interview-questions.html#g-ttft)

### 量化（4）

- [FP16 / BF16 / FP8](interview-prep/html/interview-questions.html#g-fp16-bf16-fp8)
- [INT4 / INT8 / NF4](interview-prep/html/interview-questions.html#g-int4-int8)
- [AWQ](interview-prep/html/interview-questions.html#g-awq)
- [GPTQ](interview-prep/html/interview-questions.html#g-gptq)

### 微调与对齐（10）

- [SFT](interview-prep/html/interview-questions.html#g-sft)
- [LoRA](interview-prep/html/interview-questions.html#g-lora)
- [QLoRA](interview-prep/html/interview-questions.html#g-qlora)
- [RLHF](interview-prep/html/interview-questions.html#g-rlhf)
- [RM / Reward Model](interview-prep/html/interview-questions.html#g-rm)
- [DPO](interview-prep/html/interview-questions.html#g-dpo)
- [PPO](interview-prep/html/interview-questions.html#g-ppo)
- [GRPO](interview-prep/html/interview-questions.html#g-grpo)
- [RLVR](interview-prep/html/interview-questions.html#g-rlvr)
- [catastrophic forgetting 灾难性遗忘](interview-prep/html/interview-questions.html#g-catastrophic-forgetting)

### 提示词与推理范式（7）

- [CoT](interview-prep/html/interview-questions.html#g-cot)
- [ToT](interview-prep/html/interview-questions.html#g-tot)
- [ReAct](interview-prep/html/interview-questions.html#g-react)
- [Reflexion](interview-prep/html/interview-questions.html#g-reflexion)
- [Self-Refine](interview-prep/html/interview-questions.html#g-self-refine)
- [self-consistency](interview-prep/html/interview-questions.html#g-self-consistency)
- [few-shot / zero-shot](interview-prep/html/interview-questions.html#g-few-shot)

### RAG（7）

- [RAG](interview-prep/html/interview-questions.html#g-rag)
- [chunking](interview-prep/html/interview-questions.html#g-chunking)
- [BM25](interview-prep/html/interview-questions.html#g-bm25)
- [hybrid search](interview-prep/html/interview-questions.html#g-hybrid-search)
- [reranker / cross-encoder](interview-prep/html/interview-questions.html#g-reranker)
- [vector DB / 向量数据库](interview-prep/html/interview-questions.html#g-vector-db)
- [主流 embedding 模型选型](interview-prep/html/interview-questions.html#g-embed-models)
- [RAG variants (GraphRAG / Self-RAG / CRAG / Agentic RAG)](interview-prep/html/interview-questions.html#g-rag-variants)
- [RAGAS](interview-prep/html/interview-questions.html#g-ragas)

### Agent 工程（10）

- [harness engineering](interview-prep/html/interview-questions.html#g-harness)
- [MCP / Model Context Protocol](interview-prep/html/interview-questions.html#g-mcp)
- [tool use / function calling](interview-prep/html/interview-questions.html#g-tool-use)
- [Computer Use / Browser Agent](interview-prep/html/interview-questions.html#g-computer-use)
- [sub-agent](interview-prep/html/interview-questions.html#g-sub-agent)
- [AGENTS.md / CLAUDE.md](interview-prep/html/interview-questions.html#g-agents-md)
- [deny-first](interview-prep/html/interview-questions.html#g-deny-first)
- [progress 文件](interview-prep/html/interview-questions.html#g-progress-file)
- [Loop of Death](interview-prep/html/interview-questions.html#g-loop-of-death)
- [Memory Corruption](interview-prep/html/interview-questions.html#g-memory-corruption)
- [LangChain / LangGraph 等框架](interview-prep/html/interview-questions.html#g-langchain)

### 评估与安全（7）

- [Eval / Evaluation](interview-prep/html/interview-questions.html#g-eval)
- [LLM-as-judge](interview-prep/html/interview-questions.html#g-llm-judge)
- [主流 LLM eval benchmark](interview-prep/html/interview-questions.html#g-llm-benchmarks)
- [Trace + Eval 工具栈](interview-prep/html/interview-questions.html#g-tracing-tools)
- [Lost in the Middle](interview-prep/html/interview-questions.html#g-lost-in-middle)
- [prompt injection](interview-prep/html/interview-questions.html#g-prompt-injection)
- [hallucination 幻觉](interview-prep/html/interview-questions.html#g-hallucination)
- [guardrails](interview-prep/html/interview-questions.html#g-guardrails)
- [jailbreak 越狱](interview-prep/html/interview-questions.html#g-jailbreak)

---

## 六、复习路径建议

### 路径 A：应届 / 入门（2-4 周）

```
Week 1: 工程基础概念
  └─ L1 §1.1-§1.3 + 术语速查（LLM 基础 + Transformer 架构）+ A1-A10

Week 2: 提示词 + RAG 入门
  └─ L1 §1.4 + L2 §2.4 + C1-C9 + D1-D5

Week 3: Agent 概念
  └─ L3 §1-§3 + E1, E2, E3, E5, E8 + 翻译《Building Effective Agents》

Week 4: 安全 + 实战
  └─ L1 §1.6 + I1, I2 + J7（0-1 落地路线）
```

### 路径 B：中级（3-6 年，1-2 月）

```
基础（已具备）→ 重点深化：
  - 全部 A-F 章节
  - L1-L4 完整过一遍
  - G/H 部分（系统设计 + 评估）
  - 翻译 4 篇全读
  - 自己项目对应 J 行为题准备
```

### 路径 C：资深 / staff（深度 + 广度）

```
- 全部 11 类（A-K）
- L4 §3.4 生产栈架构 + §3.4.6 成熟度模型自评
- §3.5.6 Cost 优化 8 杠杆
- §3.6 场景选型清单
- 必背：Hashimoto + Anthropic harness 原话
- 必备：1-2 个量化项目案例（GMV / 转化率 / 准确率有数字）
```

### 路径 D：场景驱动（按目标岗位定制）

| 目标岗位 | 重点章节 |
|---|---|
| **LLM 应用 SaaS（客服 / 文档 QA）** | L2 RAG + L3 Agent + G1 (客服) + I (安全) + K5 (业务评测) |
| **AI Coding 工具（Cursor / Devin / Augment 类）** | L3 + L4 全部 + G2 (Coding Agent) + E7 (harness) + 翻译《Effective Harnesses》《OpenAI Harness Engineering》 |
| **多智能体业务系统（电商 / 旅行 / 金融）** | L3 §6 多 Agent + E4/E10 + K5 (业务领域评测) + G1/G3 + J1 STAR + 准备自己项目数据 |
| **基础设施 / 推理 / 训练** | A4-A9 推理加速 + B1-B10 训练栈 + G4/G5 (网关 / 高并发) |
| **安全 / Red Team** | C7 + I1-I6 + L1 §1.6 + L4 三层防御 |

---

## 七、版本与维护

- **调研日期**：2026-05-02
- **K 章节增补**：2026-05-14（业务领域 Agent 评测）
- **术语库版本**：82 条 × 10 主题
- **本文件**：knowledge-map.md（与 .html 同步）
- **更新建议**：新模型 / 新论文 / 新工具出现时增补到对应章节，不破坏现有结构

---

*本知识图谱串联仓库全部内容。HTML 渲染版（[knowledge-map.html](knowledge-map.html)）支持折叠 + 跨页跳转 + 术语弹窗。*
