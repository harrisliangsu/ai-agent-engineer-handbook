# LLM 应用工程的四层演进 · 系列文档

**调研日期**：2026-05-01 · **修订**：2026-05-01（v3，每章节深化：公式 + 代码 + 实战配置 + GitHub 系统教程）

## 这是什么

一组围绕"如何把 LLM 真正在生产环境跑起来"的系统性技术综述，按工程抽象层从低到高组织。每一层是独立学科，但生产系统四层缺一不可。

## 阅读地图

```
01_prompt_engineering.md      → 单次 LLM 调用怎么写
              ↓
02_context_engineering.md     → 模型推理这一刻能看到什么
              ↓
03_agent_engineering.md       → 如何让 LLM 自主完成多步任务（goal → loop → memory）
              ↓
04_harness_engineering.md     → agent 在什么环境里运行、出错时如何不再犯同样的错
```

## 文件清单

| 文件 | 内容 | 篇幅 |
|---|---|---|
| `README.md` | 本文件，目录与阅读建议 | 小 |
| `01_prompt_engineering.md` | Transformer / 注意力 / 训练栈（预训练 / SFT / DPO / GRPO / RLAIF / RLVR / LoRA / QLoRA / 蒸馏）/ 解码策略 / CoT / ReAct / ToT / Self-Refine / 自动化 prompt / 评估与失败模式 | ~10 K 中文字 |
| `02_context_engineering.md` | 定义起源 / 长上下文物理学 / Lost-in-the-Middle / RULER / RAG 全栈选型（解析→切分→嵌入→向量库→检索→重排→改写→GraphRAG / Self-RAG / CRAG / Agentic RAG / 多模态）/ 记忆系统四强对比 / 上下文管理策略 / MCP / A2A | ~13 K 中文字 |
| `03_agent_engineering.md` ⭐ | Agent 定义与坐标系 / 范式演进 / 四层架构 / 类型分类 / 10 个主流框架矩阵 / 多 agent 协作 5 种模式 / 评估基准（SWE-bench / GAIA / WebArena / OSWorld 2026 排名）/ 真实产品案例（Claude Code / Cursor / Devin / Manus / Computer Use）/ 失败模式与真实事故（Amazon Kiro / Replit / Loop of Death）/ 安全权限模型 | ~12 K 中文字 |
| `04_harness_engineering.md` | 起源（Hashimoto 2026-02 命名）/ 两条核心法则（强约束 + 自愈循环）/ 四大支柱（上下文架构 / Agent 专业化 / 持久化记忆 / 结构化执行）/ §3.3.1-§3.3.3 harness 强约束的直接实现（MCP / 沙箱 / 护栏）/ 完整生产栈架构图 / 成熟度模型 L0-L4 / 综合演进叙事 / 三类场景选型清单 / 限制 / 参考文献全集 / 方法论 | ~13 K 中文字 |
| `article.md` | v1 合并版（4 文件未拆分前的原稿，保留为参考） | ~30 K 中文字 |
| `run_manifest.json` | 调研元数据：query / mode / search 路数 / claim 验证表 | 小 |
| `sources.jsonl` | 引用源注册表（45 条） | 小 |

## 推荐阅读顺序

**完整通读**：01 → 02 → 03 → 04（按抽象层从低到高，4 层叠加）

**按角色挑选**：
- **AI 产品经理**：03（Agent 类型、产品案例）→ 02（RAG 选型）→ 04（生产栈）
- **后端工程师准备转 AI**：01（基础）→ 02（RAG）→ 03（框架）→ 04（observability/eval）
- **算法工程师**：01.2 训练栈 → 02 上下文工程全章 → 03 评估基准 → 04.3.4 推理服务器
- **DevOps / 平台工程师**：04 全章 → 03.10 安全权限 → 02.7 MCP 协议
- **创业者评估技术栈**：04 的"创业 MVP" recommendations → 03 的框架选型 → 02 的记忆系统选型

## 版本演进

| 版本 | 日期 | 变化 |
|---|---|---|
| **v1** | 2026-05-01 17:30 | 原稿 3 大篇合并版（`article.md`），约 30 K 中文字 |
| **v2** | 2026-05-01 19:00 | 按章节拆分为 4 个独立文档 + 新增 Agent 工程章节（含真实事故、benchmark、安全权限），约 48 K 中文字 |
| **v3** | 2026-05-01 后续 | **每章节深化**（用户反馈"系统化不够"）：加公式、代码、实战配置、GitHub 系统教程，约 75 K 中文字 |

### v3 在 v2 基础上的关键深化

| 章节 | v3 新增 |
|---|---|
| **01 提示词工程** | Attention 完整数学公式（`softmax(QK^T/√d_k)V` + Multi-Head 公式 + PyTorch 代码）/ Q-K-V 图书馆类比 / Transformer Block 完整流程图（Pre-Norm + RMSNorm + RoPE + SwiGLU + GQA + MoE）/ RLHF 目标函数（含 4 模型表）/ DPO 损失函数完整推导 / GRPO + RLVR 流程与代码 / DPO 变体（IPO/KTO/SimPO/CPO）/ 三算法横向对比表 / **新 GitHub 系统教程小节**（karpathy nn-zero-to-hero / build-nanogpt / Sebastian Raschka LLMs-from-scratch / HuggingFace TRL / unsloth 等 13 项 + 论文必读清单） |
| **02 上下文工程** | RAG 完整 pipeline 100 行 Python 代码骨架（解析→切分→嵌入→混合检索→改写→重排→生成）/ **新 GitHub RAG 资源小节**（NirDiamant RAG_Techniques / RAGFlow / LightRAG / Awesome-RAG 等 8 项） |
| **03 Agent 工程** | **Anthropic 6 大 composable patterns 完整详解**（Augmented LLM / Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer / Agents，每个含图示 + 场景 + 选型）+ patterns 选型决策树 / 三大主流框架（LangGraph + CrewAI + OpenAI Agents SDK）"Hello World" 代码对比 + 哲学对比表 / **完整 100 行端到端 Agent 案例**（带 RAG + 记忆 + 4 道 Loop of Death 闸门 + checkpoint + observability）/ **新 GitHub Agent 资源小节**（microsoft AI agents for beginners / agent-framework / hello-agents 42K⭐ / Anthropic 文章 / 8 个主流框架等 12 项 + 论文必读） |
| **04 驾驭工程** | **两条法则 + 四大支柱方法论** + harness 强约束直接实现（MCP / 沙箱 / 护栏）+ §3.9 附录的 4 类相关基础设施（LLM 网关 / 推理 / 观测 / 评估，明确标注非 harness）+ **配置示例集**：LiteLLM `config.yaml`（routing + fallback + caching） / vLLM 与 SGLang 启动命令对比 / Langfuse 三种接入姿势 / Promptfoo `promptfooconfig.yaml`（CI gating） / RAGAS Python 评测 / NeMo Guardrails Colang / E2B 沙箱 / MCP Server 最小实现 / Pre-commit hook（强约束）/ Evaluator-Optimizer（自愈循环）/ Harness 成熟度模型 L0-L4 / **新 GitHub 驾驭层资源小节**（按支撑设施 7 类分类，30+ 项） |

### 与 v1（合并版）的差异

| 变化 | v1（`article.md`） | v2 / v3（4 文件） |
|---|---|---|
| 章节数 | 3 大篇 | 4 大篇 |
| Agent 内容 | 散落在各章 | 独立成章 |
| 真实事故案例 | 未涵盖 | 新增（Amazon Kiro / Replit / Loop of Death） |
| 公式与代码 | 仅描述 | v3 完整公式 + 可运行代码 |
| 实战配置 | 仅描述 | v3 每组件 yaml/python 示例 |
| GitHub 资源 | 极少 | v3 50+ 个分类资源 |
| 总字数 | ~30 K | v2 ~48 K → v3 ~75 K |

## 核心引用源（精选）

- Karpathy：*Software Is Changing (Again)*（context engineering 词源）
- Anthropic：*Effective Context Engineering for AI Agents*
- Mitchell Hashimoto / Augment Code：*Harness Engineering for AI Coding Agents*（harness engineering 命名）
- Vaswani 等 (2017)：*Attention Is All You Need*
- Liu 等 (2024)：*Lost in the Middle*（TACL）
- Asai 等 (2023)：*Self-RAG*
- Yao 等 (2022)：*ReAct*
- Madaan 等 (2023)：*Self-Refine*（NeurIPS）
- Packer 等 (2023)：*MemGPT*
- Sebastian Raschka：*State of LLMs 2025*
- llm-stats：*Post-Training in 2026: GRPO, DAPO, RLVR & Beyond*
- Berkeley RDI：*How We Broke Top AI Agent Benchmarks*（benchmark 可信度审计）

完整 45 条见各章节末尾的 Bibliography 与 `sources.jsonl`。

## 复现命令

```bash
ls -la ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/
wc -m ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/0[1-4]_*.md
```
