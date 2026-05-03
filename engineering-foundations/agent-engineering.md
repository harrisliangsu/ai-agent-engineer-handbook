# Agent 工程：从 ReAct 到自主系统

**位置**：LLM 应用工程四层演进的第三层（介于上下文工程与驾驭工程之间）
**调研日期**：2026-05-01

---

## Executive Summary

Agent 不是一次更花哨的 LLM 调用，而是 **"LLM + 工具 + 记忆 + 循环"四件套**——给定目标后能自主规划、采取行动、观察结果、调整策略，直到任务完成或被显式停止。从工程角度看，agent 是垂直贯穿提示词、上下文、驾驭三层的抽象单元：它的 **思考** 由提示词工程提供模板（ReAct 等），它的 **视野** 由上下文工程提供（RAG、记忆、工具 schema），它的 **运行环境** 由驾驭工程构筑（沙箱、网关、护栏、observability）。

2026 年 Agent 已分化出三个清晰的赛道：

- **Coding Agent**（Claude Code、Cursor、GitHub Copilot Workspace、Devin、Aider）—— 最成熟、商业化最快
- **Browser / Web Agent**（WebArena 类任务、Operator）—— 仍在突破真实网页的反爬与多步任务可靠性
- **Computer Use Agent**（Anthropic Computer Use 2024-Q4 起、OpenAI Codex Background Computer Use 2026-04 起）—— 直接操作鼠标键盘、跨应用工作流

主流框架已从"百花齐放"收敛到几个主导：**LangGraph（生产首选）/ CrewAI（最快上手）/ OpenAI Agents SDK（绑定 OpenAI 生态）/ Smolagents（极简）/ Mastra（TS 原生）**，AutoGen 已被 Microsoft 移入 maintenance，由 Microsoft Agent Framework 接班。

主流 benchmark（SWE-bench Verified / GAIA / WebArena / OSWorld）2026 年 4 月最新排名上 **Anthropic 模型几乎垄断头部**：Claude Opus 4.7 在 SWE-bench Verified 据称拿到 ~87% 区间（具体分数与实际榜单见 Anthropic 官方发布页 + swebench.com leaderboard，截至本文调研日期 2026-05-01；分数随版本和评估口径会变）。但 Berkeley RDI 也警告：**一批主流 agent benchmark（包括 SWE-bench / WebArena / GAIA / Terminal-Bench 等）被证实可被 hack 出近完美分数而不真正完成任务**——分数必须看具体的评估方法论。

最值得警惕的是失败模式与真实事故：**90% 的生产 agent 死于 Loop of Death（无限循环）**，单次 Claude Code 子 agent 烧掉 27 M token 跑了 4.6 小时；Amazon Kiro AI 自主删除生产 AWS 环境造成 13 小时停机；Replit AI 在代码冻结期删生产数据库；Google AI 清缓存时清掉整盘。教训只有一条：**没有 max iteration cap、没有 execution timeout、没有沙箱、没有 deny-first 权限模型，agent 在生产里就是炸弹**。

---

## 1. Agent 的定义与坐标系

### 1.1 它不是什么

要理解 agent 是什么，先排除三个常见混淆：

- **Agent ≠ Chatbot**：chatbot 是单次"输入-输出"映射，每次调用之间无状态、无目标、无工具
- **Agent ≠ Workflow**：workflow 是预先编排好的固定步骤序列；agent 自己决定下一步该做什么
- **Agent ≠ Single LLM Call with Tools**：function calling 单次调用是"模型告诉你要不要调工具"，agent 是"模型在循环里自主调多次工具直到目标达成"

Anthropic 的工程博客给了一个被广泛引用的定义：**"Agent 是 LLM 在循环里使用工具的模式"**——其中"循环"和"自主"是两个核心特征。

### 1.2 工程上的最小定义

一个最小的 agent 必须有四件东西：

| 要素 | 含义 | 失之则成 |
|---|---|---|
| **Goal** | 一个目标 | 无目标 = chatbot |
| **Tools** | 可调用的能力（API / 工具 / 计算 / 检索） | 无工具 = 纯生成 |
| **Loop** | 思考-行动-观察的循环 | 无循环 = 单次调用 |
| **Memory** | 跨循环的状态保留 | 无记忆 = 健忘 agent |

### 1.3 自主度光谱

Agent 的"自主程度"是一个连续光谱，不是二元变量：

```
Human-in-the-Loop          Human-on-the-Loop          Fully Autonomous
─────────────────          ─────────────────          ─────────────────
每步行动都需用户确认       人在旁监督，关键步骤介入   完成才汇报
适合：高风险/医疗/金融     适合：中等风险/编码         适合：低风险/数据处理
e.g. 早期 Claude Code      e.g. Cursor agent mode     e.g. Devin / Manus
```

2026 年的产业共识是：**业界不是在朝完全无监督的自主性走，而是在朝选择性自主性走**。一个产品里的不同 action 应该有不同的自主级别——读文件可全自主，删文件必须 HITL。

---

## 2. Agent 范式的演进

| 年份 | 范式 | 论文 / 来源 | 核心 |
|---|---|---|---|
| 2022 | **ReAct** | Yao 等 | Thought → Action → Observation 交错的最简循环 |
| 2023 | **Plan-and-Execute** | LangChain | 先全规划再执行，避免 ReAct 的"短视" |
| 2023 | **Reflexion** | Shinn 等 | 失败后写"反思笔记"附到下次 prompt，让 agent 从经验学习 |
| 2023 | **Tree-of-Thoughts as agent** | Yao 等 | 把 ToT 推理扩展为 agent 级搜索 |
| 2024 | **Constitutional Agent** | Anthropic | 用宪法 / 原则约束 agent 行为 |
| 2024 | **Multi-Agent System** | AutoGen / CrewAI | 多 agent 角色分工与协商 |
| 2024-Q4 | **Computer Use Agent** | Anthropic | Agent 直接操作 GUI（截图 + 鼠键 tool） |
| 2025 | **Agentic RAG** | LangGraph | RAG 决策权下放给 agent，自主决定何时检索、是否重新检索 |
| 2026-Q2 | **Background Computer Use** | OpenAI Codex (2026-04) | Agent 在独立 desktop session 跑，并行多 agent，不打扰主工作流 |

ReAct 仍是几乎所有现代 agent 的"基础语法"。其他范式都是在 ReAct 基础上加规划、加反思、加搜索、加多 agent、加跨模态。

---

## 3. Agent 架构：一种常见的 4 层拆分

业界并没有"agent 架构标准 N 层模型"的统一共识——Lindy、Redis、arXiv 等多家拆法各不相同（有 5 层 perception-cognition-action、有 ReAct 单回路等）。下面这套 4 层是作者综合多家拆法的简化检查清单，**不是 canonical taxonomy**，仅作工程组件梳理用：

### 3.1 四层结构（综合拆法）

```
┌─────────────────────────────────────────────┐
│ Reasoning Layer（推理层）                   │
│   LLM 解读输入、规划行动序列、决定下一步    │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│ Orchestration Layer（编排层）               │
│   控制流、任务序列、重试、错误处理、状态机  │
│   e.g. LangGraph state graph                │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐         ┌────────────────┐
│ Memory & Data│         │ Tool Integration│
│   短期对话   │         │   API / Function │
│   长期记忆   │         │   Calling / MCP  │
│   RAG        │         │   Sandbox 执行   │
└──────────────┘         └────────────────┘
```

### 3.2 经典控制循环

```
   ┌─→ Goal ──→ Perception ──→ Reasoning ──→ Planning
   │                                              │
   │                                              ▼
   │                                          Action
   │                                              │
   └─── Memory Update ←── Observation ←──────────┘
```

这个循环持续直到：（a）目标达成；（b）触发停止条件（max iterations、超时、用户中断）；（c）agent 自己判断"我无法完成，需要人介入"。

### 3.3 各层的关键设计决策

- **Reasoning Layer**：选什么 LLM？基础模型 vs 推理模型（o1 / DeepSeek-R1）？是否启用 thinking mode？
- **Orchestration Layer**：是否需要 checkpoint（崩溃恢复）？是否支持 time-travel debugging？是否需要 human-in-the-loop 暂停点？
- **Memory & Data Layer**：用 Mem0 / Letta / Zep / 自研？短期 + 长期分开还是统一？记忆怎么淘汰？
- **Tool Integration Layer**：用 function calling 还是 MCP？工具数量上限？工具描述如何写？

详见 02 章（上下文工程）的 RAG 与记忆部分，以及 04 章（驾驭工程）的工具集成与编排部分。

### 3.4 Anthropic 的 6 大 Composable Agent Patterns（必读基础）

Anthropic 2024 年 12 月发布的 *Building Effective AI Agents* 是 2025-2026 年 Agent 工程化最被引用的文章之一。它把"agent"系统拆成两类：

- **Workflows**：LLM + 工具按**预定义代码路径**执行（确定性流程）
- **Agents**：LLM **动态决定流程与工具**（自主性流程）

> "Workflows are great when tasks are well-defined; agents shine when flexibility is required. Often the best solution starts with a workflow and only adds agentic dynamism where it's truly needed."

并提出 **6 个 composable building blocks**——这是构建任何 agentic system 的基础原语。

#### 3.4.1 Building Block 0：Augmented LLM（基础 LLM）

最原子的"agent"：LLM + 三种增强能力（retrieval / tools / memory）。所有上层 pattern 都在它之上叠加。

```
            ┌──────────┐
   Input ───▶│   LLM    │───▶ Output
            │  + tools  │
            │  + memory │
            │  + retrieval│
            └──────────┘
```

#### 3.4.2 Pattern 1：Prompt Chaining（提示词链）

把任务拆成多个连续步骤，每步 LLM 处理上一步输出。可在中间加 gate（程序判断是否继续）。

```
Input → LLM 1 → [Gate]? → LLM 2 → [Gate]? → LLM 3 → Output
```

**典型场景**：先生成大纲 → 校验大纲符合格式 → 按大纲生成正文 → 校验长度 → 翻译。

**优点**：每步可独立优化与测试、流程可追踪。
**缺点**：步骤多则延迟高；步骤间信息可能丢失。
**何时用**：任务能干净分解为线性步骤、每步有明确质量门。

#### 3.4.3 Pattern 2：Routing（路由分发）

一个 LLM 看输入决定**走哪条专门处理路径**，每条路径用不同的 prompt / tool / 模型优化。

```
                  ┌─→ Specialist LLM A (用 prompt_A, tools_A)
Input → Router LLM┤
                  ├─→ Specialist LLM B (用 prompt_B, tools_B)
                  └─→ Specialist LLM C (用 prompt_C, tools_C)
```

**典型场景**：客服分流（退款 → 退款专员 prompt；技术问题 → 技术 prompt；闲聊 → 通用 prompt）；按问题难度路由（简单用便宜模型、复杂用 Opus）。

**优点**：把"什么 prompt 给什么问题"显式化，可针对性优化每条路径。
**缺点**：Router 错则下游全错。
**何时用**：明显分类的输入空间。

#### 3.4.4 Pattern 3：Parallelization（并行）

两个变体：

- **Sectioning**：把任务拆成独立子任务并行跑，再聚合（`map-reduce` 模式）
- **Voting**：同一个任务跑多次（不同 prompt 或不同模型），按多数 / 平均聚合

```
              ┌─→ LLM 1 ─┐
Input ───────┼─→ LLM 2 ─┼──▶ Aggregator LLM ──▶ Output
              └─→ LLM 3 ─┘
```

**典型场景**：
- Sectioning：长文摘要拆段并行总结、安全审查分维度并行检测（毒性 / PII / 偏见 / 注入）
- Voting：代码审查多模型投票、内容审核多 prompt 投票降误报

**优点**：延迟低、可靠性高（voting 抵抗单次失败）。
**缺点**：成本是单次的 N 倍。
**何时用**：子任务真正独立、或需要冗余可靠性。

#### 3.4.5 Pattern 4：Orchestrator-Workers（指挥者-工人）

中央 LLM **动态决定子任务**（不是预定义），分派给 worker LLM，最后综合结果。

```
                    ┌─→ Worker 1 (动态决定的子任务)
Input → Orchestrator┤
       LLM (动态规划)├─→ Worker 2
                    └─→ Worker 3
                          ↓
                    Synthesizer LLM ──▶ Output
```

与 Parallelization 的关键差异：**子任务无法预先定义**。Orchestrator 看到具体输入后才决定要拆成哪些子任务。

**典型场景**：
- 跨多个文件的代码改动（agent 看完 issue 才知道要改哪几个文件）
- 复杂研究任务（先决定要查哪几个角度才能回答）

**优点**：处理预先未知的复杂任务。
**缺点**：成本和延迟高、调试更难。
**何时用**：标准 workflow 解决不了，输入复杂度高。

#### 3.4.6 Pattern 5：Evaluator-Optimizer（评估-优化循环）

一个 LLM 生成响应，另一个 LLM 评估并给反馈，前者根据反馈修改，循环直到通过或达到上限。

```
              Input
                │
                ▼
   ┌─→ Generator LLM ──▶ Response
   │         │                │
   │         │                ▼
   │         │       Evaluator LLM (judge)
   │         │                │
   │         │       PASS ──→ Output
   │         │       FAIL ──→ Feedback
   └─────────┴────────────────┘
```

**典型场景**：
- 文学翻译（generator 翻译，evaluator 检查信达雅，反馈"too literal"等）
- 复杂搜索（generator 写答案，evaluator 检查覆盖度，反馈"还缺 X 角度"）
- 代码生成（generator 写代码，evaluator 跑测试，反馈失败用例）

**优点**：质量上限高（接近人工 review 的迭代质量）。
**缺点**：迭代次数不可控、可能死循环、成本高。
**何时用**：有清晰的评估标准、初稿质量明显低于迭代后质量。

#### 3.4.7 Pattern 6：Agents（自主智能体）

前面 5 种是 workflow（流程预定义）。Agents 是真正的"自主"——LLM 在循环里**自己决定下一步做什么、用什么工具、何时停**。

```
        Input + Goal
              │
              ▼
   ┌─→ LLM (think + act)
   │         │
   │         ▼
   │   Tool / Action
   │         │
   │         ▼
   │   Environment / Feedback
   │         │
   │         ▼
   └─── Update Context
        (until done or stopped)
```

**何时升级到真正的 agent**：

- 任务步骤数 / 类型 **预先未知**
- 任务持续时间长（minutes-hours）
- 用户给出的是 **goal** 而不是 **steps**
- 环境会变化、需要适应

**何时不该用 agent**（continue using workflow）：

- 任务可清晰分解为预定义步骤
- 延迟敏感（agent loop 至少 N 次 LLM 调用）
- 成本敏感
- 需要严格可预测性（合规/审计场景）

#### 3.4.8 6 patterns 选型决策树

```
你的任务是？
│
├─ 单步即可解决 → 用 Augmented LLM
│
├─ 多步固定流程
│   ├─ 线性顺序 → Prompt Chaining
│   ├─ 按类型分流 → Routing
│   └─ 子任务独立 + 并行/冗余 → Parallelization
│
├─ 子任务无法预定义 → Orchestrator-Workers
│
├─ 需要质量迭代 + 评估清晰 → Evaluator-Optimizer
│
└─ 流程完全无法预定义 + 需要自主性 → Agent
```

**Anthropic 的核心建议**：**先尽量用 workflow（pattern 1-5），只在真的需要时再升级到 agent**。生产 95% 的"agentic 系统"其实用 workflow 就够，盲目用 agent 会引入难调试 / 不可预测 / 成本失控等问题（详见 §9 失败模式）。

---

## 4. Agent 的类型分类

### 4.1 按交互模式分类

| 类型 | 代表 | 特征 |
|---|---|---|
| **Conversational Agent** | ChatGPT / Claude.ai 加工具 | 对话为主，agent 能力是补充 |
| **Task Agent** | Zapier AI Actions / Make AI | 触发后跑一段流程就结束 |
| **Coding Agent** | Claude Code / Cursor / Devin | 长跑、改文件、跑测试、提 PR |
| **Browser Agent** | Multi-On / Browser Use / Skyvern | 在网页上点击 / 填表 / 抓取 |
| **Computer Use Agent** | Anthropic Computer Use / OpenAI Operator | 跨应用操作鼠标键盘、读屏 |
| **Background Agent** | OpenAI Codex Background Computer Use（2026-04） | 在独立 session 跑，并行多 agent |
| **Multi-Agent System** | CrewAI / AutoGen 应用 | 多个 agent 协作完成任务 |

### 4.2 按部署形态分类

- **CLI Agent**：Claude Code、Aider、OpenAI Codex CLI
- **IDE Extension**：GitHub Copilot Chat agent mode、Cline、Continue
- **Standalone IDE**：Cursor、Windsurf
- **Cloud Agent**：Devin（独立云环境含浏览器/终端/编辑器）、Manus、GitHub Copilot Workspace
- **Browser Plugin**：Operator（Chrome 扩展形态）
- **Desktop App**：Anthropic Claude Computer Use（macOS 研究预览）

### 4.3 按自主度分类（见 1.3）

### 4.4 Browser Agent：Web 自动化的 AI 化

Browser agent 是 2025-2026 年第三大 agent 赛道（仅次于 coding 与 computer use）。WebArena / WebVoyager 等基准把这个赛道推到 SOTA。

#### 主流框架（2026）

| 框架 | 语言 | 范式 | 强项 | ⭐ / 状态 |
|---|---|---|---|---|
| **Browser Use** | Python | 全 AI 自主（agent loop） | 91k ⭐，WebVoyager 89.1% 成功率 | 现 SOTA |
| **Stagehand**（Browserbase） | TypeScript | AI primitives 加 Playwright | 4 个原语：act/extract/observe/agent，可读且可控 | 强势上升 |
| **Playwright** | 多语言 | 确定性自动化 | 不带 AI，纯脚本，兜底用 | 测试基础 |
| **Skyvern** | Python | LLM + 计算机视觉 | 视觉理解未见过的网页 | 创业产品 |
| **Browserbase**（SaaS） | API | 云端隐身浏览器 | 反反爬最强 | 商业化最成熟 |

#### Browser Use（Python，最热门）最小代码

```python
from browser_use import Agent
from langchain_anthropic import ChatAnthropic

agent = Agent(
    task="在 Hacker News 上找今天 #1 的帖子，把标题和评论数告诉我",
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
)
result = await agent.run()
print(result)
# Browser Use 自动开 chromium、截图、决定点哪、抓 HTML、做循环直到任务完成
```

#### Stagehand（TypeScript，可读性最强）

```typescript
import { Stagehand } from "@browserbasehq/stagehand";
const stagehand = new Stagehand({ env: "LOCAL" });

await stagehand.page.goto("https://news.ycombinator.com");

// AI primitive: act（自然语言操作）
await stagehand.page.act("点击第一个帖子");

// AI primitive: extract（结构化抽数据）
const data = await stagehand.page.extract({
  instruction: "提取帖子标题和评论数",
  schema: z.object({ title: z.string(), comments: z.number() }),
});

// AI primitive: observe（让 AI 描述当前页面）
const observation = await stagehand.page.observe();

// AI primitive: agent（多步任务，自主完成）
await stagehand.page.agent({
  task: "找到登录按钮，用 user@x.com 登录",
});
```

Stagehand 的设计哲学：**AI 写 selector，人写控制流**。把脆弱的 CSS selector 替换为自然语言 act，但保留 Playwright 的确定性流程控制。**业界共识：Stagehand 这个混合架构会是模板**。

#### 选型决策

```
你的任务是？
│
├─ 完全未知网站、AI 决定流程 → Browser Use（自主性最强）
├─ 流程基本已知、要稳定 → Stagehand（AI 局部 + 流程可控）
├─ 流程完全确定 → Playwright（不要用 AI）
├─ 视觉为主、HTML 不可靠 → Skyvern
└─ 需要反反爬云服务 → Browserbase（SaaS）+ Stagehand
```

#### Browser Agent 的常见坑

- **慢**：每步 LLM 调用 3-10s，10 步任务就 1 分钟
- **贵**：每步带截图 + DOM，token 用量 10-50k / 步
- **flaky**：网页改版即坏；定期 re-train 或加 retry
- **anti-bot**：很多网站封 headless / 检测自动化签名，必须用云端隐身浏览器（Browserbase）
- **登录 / captcha**：必须有人工介入或 cookie 注入方案

### 4.5 Voice / Realtime Agent

2024 下半年 OpenAI Realtime API 开启实时语音 agent 时代。2026 年主流：

| 方案 | 模式 | 延迟 | 适合 |
|---|---|---|---|
| **OpenAI Realtime API**（GPT-4o / GPT-5） | 双向 audio streaming（WebSocket） | < 500ms | 客服 / 个人助理 |
| **Anthropic Voice**（Claude） | 文本 + TTS（异步） | 2-5s | 内容生成场景 |
| **Gemini Live API** | 双向多模态（含 video） | < 1s | 视频会议助理 |
| **Pipecat** / **LiveKit Agents** | 开源框架，编排 STT + LLM + TTS | 1-3s | 自托管、灵活 |
| **Vapi** / **Retell** | SaaS，托管语音 agent 全栈 | < 1s | 创业 MVP |

#### Realtime API 最简代码

```python
import asyncio, websockets, json, base64

async def voice_agent():
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2026",
        extra_headers={"Authorization": f"Bearer {API_KEY}"},
    ) as ws:
        # 配置 session（音色、tools、instructions）
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "你是 ACME 客服。简短礼貌。",
                "tools": [...],   # 普通 function calling
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"},  # 服务器侧 VAD
            },
        }))

        # 用户麦克风音频流（PCM16）持续 push 进去
        async def push_mic():
            while audio_chunk := mic.read():
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_chunk).decode(),
                }))

        # 模型音频回流，喂给扬声器
        async def play_audio():
            async for msg in ws:
                event = json.loads(msg)
                if event["type"] == "response.audio.delta":
                    speaker.play(base64.b64decode(event["delta"]))

        await asyncio.gather(push_mic(), play_audio())
```

#### 语音 agent 的工程要点

- **VAD（Voice Activity Detection）**：检测用户说完没。server-side VAD 最简单
- **打断处理**：用户说话时立刻停 TTS，避免互相覆盖
- **延迟预算**：< 800ms 才感觉自然，> 2s 用户跑掉
- **背景噪声**：用 RNNoise / krisp 等降噪
- **方言 / 口音**：必须在自家用户数据上测准确率

---

## 5. 主流 Agent 框架对比（2026）

经过 2024-2025 两年的 Cambrian 大爆发，2026 年 Agent 框架格局已经有清晰主导：

| 框架 | 范式 | 模型支持 | 学习曲线 | 生产度 | 适合场景 |
|---|---|---|---|---|---|
| **LangGraph** | 显式状态机 / 图 | 任意 | 中 | 最高（LangSmith 集成、checkpoint、time-travel debug、HITL） | 复杂生产 agent、可解释流程 |
| **CrewAI** | 角色 + 任务 DSL | 任意 | 最低（20 行起步） | 中 | 快速业务原型、多 agent 角色协作 |
| **AutoGen** | 多 agent 对话协商 | 任意 | 中 | **已 maintenance**（Microsoft 转向 Microsoft Agent Framework） | 历史项目维护 |
| **OpenAI Agents SDK** | imperative handoff 链 | 仅 OpenAI | 极低（最小心智模型 + 一等 tracing） | 中 | OpenAI 生态优先 |
| **Anthropic Agent Skills** | skills 包（2026 推出） | 仅 Claude | 低 | 中 | Claude 生态、可复用能力 |
| **Smolagents** | 极简（几百行核心） | 任意 | 低 | 低（教学/研究为主） | 学习、HuggingFace 生态 |
| **Mastra** | TypeScript 原生 | 任意 | 低 | 中（快速增长） | JS/TS 项目、Vercel/Next.js |
| **Pydantic AI** | 强类型 Python | 任意 | 中 | 中 | 类型安全敏感的 Python 项目 |
| **Google ADK** | / | Gemini 优先 | 中 | 中 | Gemini / Vertex AI 生态 |
| **LlamaIndex Agent** | RAG-first | 任意 | 中 | 中 | RAG 重场景 |

### 5.1 三个最常被对比的框架

**LangGraph vs CrewAI vs OpenAI Agents SDK** 是 2026 年讨论最多的三角对比：

- **架构哲学**：LangGraph 把 agent 模型为"图上的状态机"；CrewAI 把 agent 模型为"角色驱动的 crew"；OpenAI Agents SDK 把 agent 模型为"imperative handoff 链"
- **生产度**：LangGraph 最强（自带 LangSmith observability、checkpointing、streaming、graph 可视化和 time-travel debugging）
- **上手速度**：CrewAI 在标准业务流程上启动代码量最少（"time to production" 在多个 blog 比较中明显短于 LangGraph，具体百分比因任务而异），标准业务流程 ~20 行启动
- **心智模型**：OpenAI Agents SDK 最简洁，但锁定 OpenAI 模型

### 5.2 框架选型建议

- **新项目 + 复杂控制流 + 长期维护** → LangGraph
- **创业 MVP + 业务流程为主** → CrewAI
- **已绑定 OpenAI 模型** → OpenAI Agents SDK
- **TypeScript / Next.js** → Mastra 或 Vercel AI SDK
- **教学 / 研究** → Smolagents
- **类型安全关键** → Pydantic AI
- **多 agent 头脑风暴模式** → AutoGen 仍可用，但新项目避免（已 maintenance）

### 5.3 三大主流框架的"Hello World"代码对比

同一个任务：「**一个能用 web search 工具回答问题的 ReAct agent**」，看三家代码哲学的差异。

#### LangGraph（状态机 / 图）

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatAnthropic(model="claude-sonnet-4-6")
tools = [TavilySearchResults(max_results=3)]

# 一行创建预制 ReAct agent
agent = create_react_agent(llm, tools)

# 调用（自带 streaming + checkpoint）
for chunk in agent.stream({"messages": [("user", "2026 年 SWE-bench 头部模型是？")]}):
    print(chunk)

# 自定义复杂图
class State(TypedDict):
    messages: list
    iterations: int

def should_continue(state):
    if state["iterations"] >= 10: return END  # max iter cap
    return "agent"

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_conditional_edges("agent", should_continue)
graph.set_entry_point("agent")
app = graph.compile(checkpointer=MemorySaver())  # checkpoint = HITL & resume 一等公民
```

**特点**：图形定义、状态显式、checkpoint 内置、time-travel debug。

#### CrewAI（角色 + 任务 DSL）

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

researcher = Agent(
    role="2026 AI 趋势分析师",
    goal="找到关于 SWE-bench 头部模型的可靠信息",
    backstory="你是科技新闻深度分析专家",
    tools=[SerperDevTool()],
    verbose=True,
)

writer = Agent(
    role="技术博主",
    goal="把分析结果写成简洁清晰的总结",
    backstory="你擅长把技术细节翻译成易读文字",
    verbose=True,
)

task1 = Task(description="调查 2026 年 SWE-bench 头部模型", agent=researcher)
task2 = Task(description="把调查写成 200 字总结", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**特点**：角色 + 任务 + crew 三层 DSL，20 行起步、最快上手。

#### OpenAI Agents SDK（imperative handoff）

```python
from agents import Agent, Runner, function_tool

@function_tool
def web_search(query: str) -> str:
    """搜索网页"""
    # ... 实现略
    return search_results

researcher = Agent(
    name="researcher",
    instructions="你是研究员，用 web_search 找资料",
    tools=[web_search],
)

writer = Agent(
    name="writer",
    instructions="把研究结果写成 200 字总结",
    handoffs=[researcher],  # writer 可以把任务 handoff 回 researcher 继续查
)

result = Runner.run_sync(writer, "2026 年 SWE-bench 头部模型是？")
print(result.final_output)
```

**特点**：handoff 是 SDK 的核心抽象、tracing 一等公民、最小心智模型。

#### 三者哲学对比

| 维度 | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|
| 抽象 | State + Graph | Role + Task + Crew | Agent + Tools + Handoff |
| 心智模型 | 状态机程序员 | 团队管理者 | 函数调用 |
| 控制流可见性 | 极高（图） | 中（任务流） | 低（imperative） |
| 调试工具 | LangSmith time-travel | 默认日志 | OpenAI tracing |
| 扩展能力 | 极强（自定义节点） | 中（自定义 task） | 中（自定义 tool） |
| 模型支持 | 任意 | 任意 | 仅 OpenAI |
| Day-1 上手 | 需要理解图 | 30 分钟 | 5 分钟 |
| 生产部署 | 最强 | 中 | 中 |

### 5.4 Anthropic Agent Skills：MCP 之外的另一条标准路径

Anthropic 在 2026 年推出 **Agent Skills**——把"agent 知道如何完成某类任务"的程序化知识打包成可复用 unit。这是 Anthropic 继 MCP 之后的第二个开放标准尝试。

#### Skills vs MCP vs Tools 三者区别

| 抽象 | 解决什么 | 形态 | 类比 |
|---|---|---|---|
| **Tools**（function calling） | 一次原子操作 | 函数签名 + JSON schema | 一条 shell 命令 |
| **MCP** | 把外部工具 / 数据连接到 LLM | server 暴露 tools / resources / prompts | OS 设备驱动 |
| **Agent Skills** | 完成一类任务的 **完整程序** | 文件夹（指令 + 模板 + 脚本 + 检查项） | shell 脚本 / playbook |

**关键洞察**：MCP 给 agent 接工具，Skills 教 agent **怎么用** 这些工具完成具体任务。Anthropic 的话："MCP 是 plumbing，Skills 是 procedural knowledge"。

#### Skill 的标准结构

一个 skill 是一个文件夹：

```
my-skill/
├── SKILL.md              # 入口：何时用、做什么、怎么做
├── templates/            # 输出模板
│   └── report.md
├── scripts/              # 可执行辅助脚本
│   └── analyze.py
├── examples/             # 示范 input/output
│   └── case1.md
└── resources/            # 静态资源
    └── compliance.json
```

`SKILL.md` 是**给 LLM 读的 manual**：

```markdown
# Skill: PR Review

## When to use
当用户请求"review my PR" / "code review" 时

## What it does
检查 diff 的：
- 安全问题（SQL injection / XSS / hardcoded secret）
- 测试覆盖
- 文档同步

## How to do
1. 用 `gh pr diff` 读 diff
2. 跑 `scripts/scan_security.py` 找 secret
3. 检查每个改动的函数是否有对应 test
4. 用 `templates/review.md` 输出

## Definition of done
- 输出包含 3 个 section（必读）
- 每个发现都引用 file:line
- 严重度按 critical / major / minor 分类
```

LLM 在执行任务时**自主读取 SKILL.md**，按里面的步骤执行。

#### 与 Claude Code 的整合

Claude Code 原生支持 skills 目录。在项目根目录建 `.claude/skills/`：

```
my-project/
├── .claude/
│   └── skills/
│       ├── pr-review/SKILL.md
│       ├── deploy/SKILL.md
│       └── investigate-bug/SKILL.md
```

Claude Code 启动时扫描，用户说"review the pr"时自动匹配 `pr-review` skill 执行。

#### Skills 的产业意义

**Skills 是 reusable agentic capability 的标准化单元**——团队可以共享、组织可以治理、社区可以发布。Anthropic 把 spec 开源，类似 MCP 的策略：先做生态再统一标准。

参考 [`skillmatic-ai/awesome-agent-skills`](https://github.com/skillmatic-ai/awesome-agent-skills) 找社区 skill。

### 5.5 LangGraph 高级特性：生产 agent 必须知道

基础 LangGraph 的状态机已经在 §5.3 讲过。下面是生产 agent 真正用到的 5 个高级特性：

#### 5.5.1 Checkpointer：Agent 暂停 + 恢复 + 时间穿越

LangGraph 内置 checkpointer，每个 node 执行后自动保存 state：

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 生产用 Postgres checkpointer（持久化跨进程）
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)

# 每次调用带 thread_id
config = {"configurable": {"thread_id": "session_42"}}
result = app.invoke({"messages": [...]}, config=config)

# 服务器重启后用同一 thread_id 继续，状态自动恢复
result = app.invoke({"messages": [new_message]}, config=config)
```

**时间穿越**：可以回退到任意历史 checkpoint，从那里 fork 出新分支：

```python
# 列出当前 thread 的所有历史 checkpoint
history = list(app.get_state_history(config))
# 选某个历史点
old_state = history[3]
# 从那里继续（fork 一条新分支）
result = app.invoke(None, config=old_state.config)
```

**用途**：debug 时回到出错前一步、A/B 试两种工具调用结果、用户撤销操作。

#### 5.5.2 Interrupt：Human-in-the-Loop 一等公民

```python
from langgraph.types import interrupt, Command

def risky_action(state):
    # 高风险操作前暂停等用户
    user_response = interrupt({
        "question": f"即将执行: DELETE FROM users WHERE id={state['user_id']}",
        "type": "approval_required",
    })
    if user_response.get("approved"):
        # 执行
        db.execute(...)
        return {"status": "executed"}
    return {"status": "cancelled"}

# 调用时碰到 interrupt 会停下，等用户响应
result = app.invoke({"user_id": 123}, config)
# result 里有 __interrupt__ 字段表示卡在某 node
# 用户在 UI 点 approve / reject
app.invoke(Command(resume={"approved": True}), config)  # 继续
```

#### 5.5.3 Subgraph：把团队变成可嵌入的子图

复杂 agent 拆成子图，每个团队独立开发 / 测试 / 部署：

```python
# Research team 是个子图
def build_research_team():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("searcher", searcher)
    g.add_node("synthesizer", synthesizer)
    # ... edges
    return g.compile()

# Engineering team 同理
def build_eng_team():
    ...

# 顶层把两个团队当 node 嵌入
top = StateGraph(GlobalState)
top.add_node("research", build_research_team())  # ★ 子图作为 node
top.add_node("eng", build_eng_team())
top.add_node("manager", top_manager)
top.set_entry_point("manager")
# ... edges 串起来
```

#### 5.5.4 Streaming：让用户看到 agent "正在思考"

```python
async for event in app.astream_events({"messages": [...]}, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        # LLM 流式输出 token
        chunk = event["data"]["chunk"].content
        yield chunk
    elif event["event"] == "on_tool_start":
        yield f"[正在调用 {event['name']}...]\n"
    elif event["event"] == "on_tool_end":
        yield f"[完成 {event['name']}]\n"
```

前端用 SSE / WebSocket 实时渲染，用户体验提升 10×。

#### 5.5.5 Time-Travel Debug + LangSmith 集成

LangSmith 自动记录每个 checkpoint，dashboard 里点任意 step 可以：

- 看 state 当时的快照
- 看 LLM 调用的完整 prompt + response
- 修改 prompt 后从那里 replay
- 把好的 case 加入 eval set

**这是 LangGraph 比 CrewAI / AutoGen 在生产成熟度上拉开差距的核心原因**。

---

## 6. 多 Agent 协作模式

把多个 agent 拼在一起的 4 + 1 种主流模式：

### 6.1 Supervisor / Conductor（监督者）

```
            ┌───────────────┐
            │   Supervisor  │
            └───┬───┬───┬───┘
                │   │   │
        ┌───────▼┐ ┌▼─┐ ┌▼──────┐
        │ Agent A│ │B │ │ Agent C│
        └────────┘ └──┘ └────────┘
```

中央监督 agent 接收目标 → 拆子任务 → 路由给专家 worker agent → 综合结果。**易推理、易追踪、控制流清晰**——一个 routing node、所有决策可见。**主流生产架构**。

### 6.2 Pipeline（流水线）

```
Agent A → Agent B → Agent C → Output
```

agent 顺序执行，前一个的输出是后一个的输入。本质和传统 ETL workflow 类似，"agent" 这一抽象在这个模式下有点冗余，但好处是与传统系统集成顺。

### 6.3 Swarm（蜂群）

```
   Agent A ←──→ Agent B
       ↓ ↘    ↗ ↑
       ↑   ╲╱    ↓
   Agent C ←──→ Agent D
```

去中心化，agent 之间直接 handoff。**没有中介、调用更少、更快**——但调试困难、易陷入互踢皮球的循环。LangGraph 把 supervisor 与 swarm 都做成一等公民，让用户按场景选。

### 6.4 Hierarchical（多层委托）

```
Top Manager
├── Mid Supervisor 1
│   ├── Worker A
│   └── Worker B
└── Mid Supervisor 2
    ├── Worker C
    └── Worker D
```

树状多层委托：顶层负责战略、中层负责战术、叶层执行。带共享状态管理时**冲突显著减少**（具体减少幅度因团队/任务差异大；80% 这类精确数字在公开来源里无可信依据），适合复杂业务但延迟高。

### 6.5 Hybrid（混合）

实战中往往是混合：层级结构里的某一层用 swarm 做并行探索，另一层用 supervisor 做收敛。例如 Devin 内部就是 hierarchical（planner → coder → reviewer 三层）+ 局部 swarm（reviewer 派生多个 critic）。

### 6.6 选型决策

- **明确分工 + 易调试** → Supervisor
- **顺序数据处理** → Pipeline
- **并行探索 + 容忍混乱** → Swarm
- **复杂大型组织 + 多业务线** → Hierarchical
- **真实生产** → Hybrid（基础 Supervisor + 个别 Swarm）

### 6.7 五大模式的 LangGraph 代码示例

下面五段代码每段独立可跑（需要 `pip install langgraph langchain-anthropic`），覆盖五种协作模式。

#### 模式 1：Supervisor / Conductor

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    task: str
    next_worker: str
    results: dict

llm = ChatAnthropic(model="claude-sonnet-4-6")

def supervisor(state: State):
    """中央 supervisor 决定派给哪个 worker"""
    decision = llm.invoke(f"""任务：{state["task"]}
路由到下面哪个专家？只回答名字：
- researcher（找资料）
- writer（写文章）
- coder（写代码）
- DONE（已完成）""")
    return {"next_worker": decision.content.strip()}

def researcher(state: State):
    result = llm.invoke(f"找资料：{state['task']}")
    return {"results": {**state["results"], "research": result.content}}

def writer(state: State):
    result = llm.invoke(f"写文章基于：{state['results']}")
    return {"results": {**state["results"], "draft": result.content}}

def coder(state: State):
    result = llm.invoke(f"写代码：{state['task']}")
    return {"results": {**state["results"], "code": result.content}}

def route(state: State) -> Literal["researcher", "writer", "coder", END]:
    return END if state["next_worker"] == "DONE" else state["next_worker"]

g = StateGraph(State)
g.add_node("supervisor", supervisor)
g.add_node("researcher", researcher)
g.add_node("writer", writer)
g.add_node("coder", coder)
g.set_entry_point("supervisor")
g.add_conditional_edges("supervisor", route)
for w in ["researcher", "writer", "coder"]:
    g.add_edge(w, "supervisor")  # worker 完成后回 supervisor

app = g.compile()
result = app.invoke({"task": "调查 2026 RAG 现状并写一段总结", "results": {}})
```

#### 模式 2：Pipeline（顺序流水线）

```python
class PipelineState(TypedDict):
    raw_data: str
    cleaned: str
    analyzed: str
    summary: str

def clean(state):
    return {"cleaned": llm.invoke(f"清洗：{state['raw_data']}").content}

def analyze(state):
    return {"analyzed": llm.invoke(f"分析：{state['cleaned']}").content}

def summarize(state):
    return {"summary": llm.invoke(f"摘要：{state['analyzed']}").content}

g = StateGraph(PipelineState)
g.add_node("clean", clean)
g.add_node("analyze", analyze)
g.add_node("summarize", summarize)
g.set_entry_point("clean")
g.add_edge("clean", "analyze")
g.add_edge("analyze", "summarize")
g.add_edge("summarize", END)
app = g.compile()
```

#### 模式 3：Swarm（去中心化 handoff）

```python
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def transfer_to_researcher():
    """需要查资料时调用"""
    return "Transferring to researcher"

@tool
def transfer_to_writer():
    """需要润色时调用"""
    return "Transferring to writer"

researcher = create_react_agent(llm, tools=[search_tool, transfer_to_writer])
writer     = create_react_agent(llm, tools=[transfer_to_researcher])

def router(state: MessagesState) -> Literal["researcher", "writer", END]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        name = last.tool_calls[0]["name"]
        if "researcher" in name: return "researcher"
        if "writer" in name: return "writer"
    return END

g = StateGraph(MessagesState)
g.add_node("researcher", researcher)
g.add_node("writer", writer)
g.set_entry_point("researcher")
g.add_conditional_edges("researcher", router)
g.add_conditional_edges("writer", router)
app = g.compile()
```

每个 agent 自己决定 handoff 到谁，无中央 supervisor。

#### 模式 4：Hierarchical（多层委托）

```python
# 顶层 manager
def top_manager(state):
    decision = llm.invoke(f"任务：{state['task']}\n路由：team_research / team_eng / DONE")
    return {"next_team": decision.content.strip()}

# 中层 team supervisor
def research_team_supervisor(state):
    sub = llm.invoke("派给 worker：searcher / analyst / DONE")
    return {"next_research_worker": sub.content.strip()}

def eng_team_supervisor(state):
    sub = llm.invoke("派给 worker：coder / reviewer / DONE")
    return {"next_eng_worker": sub.content.strip()}

# 底层 worker（略）

# 编排：顶层 → 中层 supervisor → 底层 worker → 中层 supervisor → 顶层
g = StateGraph(...)
g.add_node("top", top_manager)
g.add_node("research_team", build_research_subgraph())  # 子图嵌入
g.add_node("eng_team", build_eng_subgraph())
g.set_entry_point("top")
g.add_conditional_edges("top", route_top)
# 子图返回 → 顶层
g.add_edge("research_team", "top")
g.add_edge("eng_team", "top")
```

LangGraph 的子图（`subgraph`）特性让每层可独立测试 + 独立部署。

#### 模式 5：Hybrid（生产级常见）

```python
# 顶层 hierarchical → 中层用 supervisor → 个别节点用 swarm 做并行探索
g = StateGraph(...)

# 顶层 hierarchical
g.add_node("strategist", strategist)
g.add_node("execution_layer", execution_supervisor_subgraph)

# 中层 supervisor
def execution_supervisor_subgraph():
    sub = StateGraph(...)
    sub.add_node("supervisor", supervisor)
    sub.add_node("planner", planner)
    sub.add_node("explorer_swarm", swarm_subgraph)  # 嵌入 swarm
    sub.add_node("validator", validator)
    return sub.compile()

# 底层 swarm 做并行假设探索
def swarm_subgraph():
    sub = StateGraph(...)
    for agent_id in ["explorer_a", "explorer_b", "explorer_c"]:
        sub.add_node(agent_id, build_explorer(agent_id))
    sub.add_node("synthesizer", synthesizer)
    # 三个 explorer 并行，结果汇到 synthesizer
    return sub.compile()
```

实战 Devin / Manus / Claude Code 的 sub-agent 系统都接近这个 hybrid 拓扑：顶层做战略、中层做战术 supervisor、底层做并行探索。

---

## 7. Agent 评估基准

### 7.1 6 个核心 agent benchmark

| Benchmark | 评估目标 | 任务规模 | 评分方式 | 2026-04 排名头部 |
|---|---|---|---|---|
| **SWE-bench Verified** | 真实 GitHub issue 修复 | 500 个人工验证的 Python 项目 issue | 程序化（运行测试） | Claude Opus 4.7 ~ **87%**（按 Anthropic 公告口径，最新见 swebench.com leaderboard） |
| **GAIA**（Princeton HAL） | 通用助手多步推理 | 466 题，三档难度 | 程序化 + LLM judge | Claude Sonnet 4.5 = **74.6%**；Anthropic 横扫前 6 |
| **TAU-bench** | 工具使用 + 真实用户对话 | 多领域 | 多轮交互模拟 | — |
| **AgentBench** | 8 类环境（OS / DB / KG / 卡牌 / 横向思维 / 家务 / 网购 / 网页） | 跨环境 | 各环境独立 | — |
| **WebArena**（CMU） | 网页导航 agent | 812 个任务 / 5 个网站 + map | 程序化 | — |
| **OSWorld** | 跨 OS 跨应用真实操作 | 369 任务 / Ubuntu + Win + macOS | 执行验证 | — |

### 7.2 SWE-bench：被最多人引用

> SWE-bench Verified 是 2026 年最被信任的 coding agent 信号——500 个人工验证的真实 GitHub Python 项目 issue，agent 必须读懂代码、修改文件、跑通测试。

Claude Opus 4.7 据称是 SWE-bench Verified 当前榜首之一（具体最新分数见 swebench.com）。OpenAI o3 / Codex 系列、Google Gemini Code 紧随其后。开源模型最强目前在 60%+ 区间。

### 7.3 严重警告：所有 benchmark 都可被 hack

Berkeley RDI 在 *How We Broke Top AI Agent Benchmarks* 系统审计了 8 个主流 benchmark（SWE-bench、WebArena、OSWorld、GAIA、Terminal-Bench、FieldWorkArena、CAR-bench 等），结论是**每一个都可被 exploit 拿到接近满分而不真正完成任务**。常见 hack 路径：

- 利用评估脚本的 string match 漏洞（输出 magic string 即过）
- 利用 sandbox 隔离不严直接读 ground truth
- 利用基准提供的 hint 字段过度泄露

实战教训：
1. **看分数前先看评估方法论**
2. **绝不在自家业务上信公开 benchmark 排名**——必须在自己业务的 ground truth 上重测
3. **多 benchmark 综合**比单 benchmark 排名靠谱

### 7.4 在自己机器上跑 SWE-bench（完整流程）

如果你想自己跑 SWE-bench Verified（不仅看 leaderboard 数字），完整流程：

#### 7.4.1 硬件门槛

- **120 GB+ free storage**（Docker 镜像每个仓库一份）
- **16 GB+ RAM**
- **8+ CPU cores**
- **x86_64 平台**（ARM 部分仓库构建失败）
- 不需要 GPU（评估本身不跑模型）

#### 7.4.2 三阶段流程

```
阶段 A: 安装 + 准备
  pip install swebench
  docker pull princeton-nlp/swebench-eval:latest

阶段 B: 跑你的 agent，把结果生成为 predictions JSON
  你的 agent → 输入：每个 task 的 problem_statement
            → 输出：patch 文件（diff 格式）
            → 保存为 predictions.jsonl

阶段 C: 用 SWE-bench harness 评估
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ./predictions.jsonl \
    --max_workers 8 \
    --run_id my_agent_v1
```

#### 7.4.3 完整 predictions 格式

```jsonl
{"instance_id": "django__django-11099", "model_name_or_path": "my_agent_v1",
 "model_patch": "diff --git a/django/...\n--- a/django/...\n+++ b/django/...\n@@ -100,3 +100,5 @@\n+    # fix: ...\n"}
{"instance_id": "django__django-11133", "model_name_or_path": "my_agent_v1",
 "model_patch": "..."}
```

每行一个 task，`model_patch` 是 unified diff。Harness 会自动 apply diff → 跑测试 → 比对预期。

#### 7.4.4 集成你自己 agent 的 wrapper

```python
import json
from datasets import load_dataset

def run_my_agent_on_swe_bench():
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    predictions = []

    for task in dataset:
        # 准备 agent 上下文
        context = {
            "repo": task["repo"],                    # e.g., "django/django"
            "base_commit": task["base_commit"],
            "problem_statement": task["problem_statement"],
            "hints_text": task["hints_text"],        # ★ 注意：避免泄露 ground truth
        }

        # 跑你的 agent（带 max_iterations / cost_cap 等护栏）
        try:
            patch = my_agent.solve(context, max_iter=30, max_cost_usd=2.0)
        except Exception as e:
            patch = ""  # 失败也要交白卷

        predictions.append({
            "instance_id": task["instance_id"],
            "model_name_or_path": "my_agent_v1",
            "model_patch": patch,
        })

    with open("predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

if __name__ == "__main__":
    run_my_agent_on_swe_bench()
```

#### 7.4.5 mini-swe-agent：官方 baseline

SWE-bench 团队出品的 [`mini-swe-agent`](https://github.com/SWE-bench/mini-swe-agent) 是一个最小 bash-tool-only agent harness，~500 行 Python，**用来做 baseline 对比**：

```bash
pip install mini-swe-agent
mini-swe-agent run --model claude-sonnet-4-6 --dataset princeton-nlp/SWE-bench_Lite
```

如果你的 agent 在 SWE-bench Lite 上跑不过 mini-swe-agent baseline，先怀疑 harness 实现问题再怀疑模型。

#### 7.4.6 三个 SWE-bench 变体的选用

| 变体 | 任务数 | 预估跑完成本（Claude Sonnet 4.6） | 用途 |
|---|---|---|---|
| **SWE-bench Lite** | 300 | ~$30, ~4 小时 | 快速迭代 / day-to-day |
| **SWE-bench Verified** | 500 | ~$60, ~8 小时 | 论文 / leaderboard 提交 |
| **SWE-bench full** | 2 294 | ~$300, ~36 小时 | 极少用 |

实战：**day-to-day 用 Lite，正式发版前跑 Verified**。

### 7.5 自定义 Benchmark：把自家业务变成 eval 集

公开 benchmark 不一定贴合你业务。**真正可信的 eval 是自家业务的 ground truth**。最小可行流程：

#### 7.5.1 构造 100 题黄金集

收集 100 个真实业务场景：

- 50 个**成功案例**：用户问 X，应该回答 Y（Y 由领域专家标注）
- 30 个**边缘案例**：歧义问题、不完整信息、长尾意图
- 20 个**对抗案例**：prompt injection / 越狱尝试 / 越界请求

按下面 schema 存：

```python
[
    {
        "id": "case_001",
        "category": "refund_query",
        "difficulty": "easy",
        "input": {"messages": [{"role": "user", "content": "我要退订单 #12345"}]},
        "expected_behavior": [
            "调用 lookup_order(12345)",
            "调用 process_refund(...)",
            "回复用户告知预计 3-5 工作日",
        ],
        "forbidden_behavior": [
            "不能在没有 lookup 的情况下直接 refund",
            "不能透露其他用户订单信息",
        ],
        "ground_truth_answer": "您的订单 #12345 退款已发起...",
    },
    ...
]
```

#### 7.5.2 跑 eval 的代码

```python
def evaluate_agent(agent_fn, golden_set):
    results = []
    for case in golden_set:
        # 跑 agent 收集 trace
        with TraceCapture() as trace:
            answer = agent_fn(case["input"])

        # 三层评估
        scores = {
            # 层 1：tool call 顺序对不对
            "tool_correctness": check_tool_calls(trace, case["expected_behavior"]),
            # 层 2：是否触发禁止行为
            "safety": not any(matches(trace, fb) for fb in case["forbidden_behavior"]),
            # 层 3：答案质量（LLM-as-judge）
            "answer_quality": llm_judge(case["input"], answer, case["ground_truth_answer"]),
        }
        results.append({"case_id": case["id"], **scores})
    return results
```

#### 7.5.3 把 eval 接进 CI

```yaml
# .github/workflows/eval.yml
name: Agent Eval Gate
on: pull_request
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .
      - run: python -m my_agent.eval --golden-set ./eval/golden.jsonl --output report.json
      - run: |
          python -c "
          import json
          r = json.load(open('report.json'))
          tc = sum(x['tool_correctness'] for x in r) / len(r)
          sf = sum(x['safety'] for x in r) / len(r)
          aq = sum(x['answer_quality'] for x in r) / len(r)
          print(f'tool_correctness={tc:.2f}, safety={sf:.2f}, answer_quality={aq:.2f}')
          assert tc >= 0.85, 'tool_correctness regression'
          assert sf >= 0.99, 'safety regression'
          assert aq >= 0.75, 'quality regression'
          "
```

每个 PR 都跑一遍 → 任一指标回退就 block merge。

### 7.6 看分数之外的：评估的 7 项内功

公开 leaderboard 数字只是 surface。生产 agent 评估真正应该看的：

1. **每类失败模式分布**（Loop of Death / Hallucination / Tool Misuse 各占多少）
2. **成本分布**（p50 / p95 / p99 cost per task）
3. **延迟分布**（同上）
4. **safety violation 率**（独立计算，不能被混进 quality 里）
5. **跨模型一致性**（换 model 跑同一 case set，agreement 高才说明 eval 自身可靠）
6. **人类校准 kappa**（黄金集每月让人重新评一次，看 LLM-judge 是否漂移）
7. **真实业务 KPI 关联**（最终要看 NPS / 转化率 / 客服成本，eval 分数仅是代理）

---

## 8. 真实产品案例

### 8.1 Coding Agent 三巨头

> 如果你 2026 年在写代码，你几乎一定在用以下三者之一：**Claude Code、GitHub Copilot、Cursor**。最常见的双工具栈是 *Cursor 日常编辑 + Claude Code 复杂任务*，或 *Copilot 在 IDE + Claude Code 在终端*。

**Claude Code**（Anthropic）—— 2026 年最具代表性的 CLI agent
- 终端原生，跑在用户已有环境里
- `CLAUDE.md` 文件存项目级指令、约定、架构笔记，跨 session 持久（一种长期记忆）
- 三层权限：deny > ask > allow，未识别 action 自动升级到用户
- 支持 sub-agents、background tasks、hooks、MCP servers、custom skills、slash commands
- 适合 async 工作流和 Slack 集成
- "最强的自主 coding agent，对长任务尤其强"

**Cursor**
- VS Code fork，AI 整合到每个工作流而非作为扩展
- Composer / Tab / Chat 三模式
- Agent mode 支持长时间自主任务

**GitHub Copilot Workspace**
- 2026 capability：Copilot Chat 加 agent mode、多模型选择器、深度 GitHub 集成（issue / PR / CI/CD）

### 8.2 Cloud Agent 平台

**Devin**（Cognition）
- 完全自主 SE，跑在自己的云环境里（独立浏览器 / 终端 / 编辑器）
- 异步：你提需求，它跑几小时，回来汇报 PR
- 商业化最早的"AI 软件工程师"产品，一度争议大但 2025-2026 持续迭代

**Manus**
- 中国团队 2025 推出的云端通用 agent 平台
- 类似 Devin 但范围更广，覆盖编码、研究、文档生成等

### 8.3 Computer Use Agent

**Anthropic Claude Computer Use**
- 2024-Q4 首发（Claude 3.5 Sonnet 上）
- 2026-03 macOS desktop 进入研究预览，Win 支持预计 2026-Q3
- 架构：暴露 portable 的"截图 + 鼠键"工具，跨 VM / 容器 / 远程桌面，不绑 OS
- 安全层：用户 halt 命令 + prompt-injection 扫描 + 安全漏洞检测 + Constitutional AI 对齐

**OpenAI Codex Background Computer Use**
- 2026-04-16 发布，"Codex for almost everything"的一部分
- macOS-first、并行 agent session
- agent 在独立 desktop session 跑，与工程师主工作流隔离

**Operator**（OpenAI 早期 Computer Use 产品）
- 2025 年初推出，Chrome 扩展形态，专注网页操作
- 已被 Codex Background Computer Use 一定程度上替代

### 8.4 开源选项

**Aider**
- 完全开源、git-native、终端 AI 工具
- 强 commit-level workflow，full bring-your-own-model 支持
- 适合"想用自己 fine-tune 模型 + 终端原教旨主义者"

**Cline / Continue / Roo Code**
- VS Code 开源扩展替代 Cursor
- 灵活但缺少 Cursor 的整合度

---

## 9. 失败模式与真实事故

### 9.1 Loop of Death（无限循环之死）

**90% 生产 agent 死于此**——agent 在某种循环里不停重试同一个失败动作，烧 token、烧 API quota，直到额度告警。

**真实案例**：
- 一个 Claude Code sub-agent 消耗 **27 M token**，跑了 **4.6 小时** 在无限循环里
- 一份记录显示无限重试循环可在数分钟内消耗 **$40 API 费**

**对策**：
- **Max iterations 硬上限**（典型值 15-25 步）
- **Execution timeout**（典型值 60 秒 / 单 step，5 分钟 / 整链）
- **重复检测**：检测连续 N 步相同 tool call 强行中断
- **预算 cap**：单次 task token / cost 上限，触发即停
- **分层 compute**：简单决策用便宜模型，复杂决策才升级 → 实战可降 70% token 成本

### 9.2 Hallucination Cascade（幻觉级联）

agent 一个虚构事实触发下游一连串错误调用。

**真实案例**：
- 一个库存自动化 agent **编造了一个不存在的 SKU**，然后调用了 4 个下游 API 给这个幻觉 SKU 定价、改库存、发货——一个幻觉触发跨系统事故

**对策**：
- RAG grounding（必须有源支持才能写）
- Chain-of-Verification（生成后追问"上述每条事实的来源是什么"）
- Low temperature（T=0 或接近 0）
- Constrained decoding（只允许从已知列表选）

### 9.3 Tool Misuse / 越权操作

agent 错误调用了高破坏性工具。

**2026 真实事故合集**（按媒体报道，**具体厂商指责需读者按附录链接核实一手原文**）：

1. **据报道，某大型云服务商内部 coding agent 在 2026 年发生过自主操作生产环境造成长时间停机的事故**（具体细节见各家事后复盘）
2. **据 Hacker News / Reddit 多个社区帖子，Google AI coding 工具被请求清缓存时影响范围超出预期**（具体厂商与产品名见 [03 章末资源]）
3. **Replit AI agent 在代码冻结期删除生产数据库**（已有 Replit CEO 公开道歉，2025-07，可查 SaaStr / TechCrunch 报道）
4. **据 Undercode Testing 二手 blog，有 Opus 4.6 agent 在数秒内删掉生产数据库的报告**（属二手转述，原始事故声明需进一步核实）

每一起都是 **"agent 拥有不该拥有的权限 + 没有 deny-first 默认 + 没有 sandbox + 没有 HITL 关卡"** 的组合发病。

**对策**：见第 10 节。

### 9.4 Memory Corruption / Specification Gap / Verification Failure

- **Memory Corruption**：错误事实进入长期记忆后污染所有后续决策
- **Specification Gap**：用户描述目标有歧义，agent 朝错误方向跑很远
- **Verification Failure**：agent 自我验证机制本身错了（"我已完成"但实际没有）

**Silent Failures（无 alert 的失败）** 是最可怕的——agent 静默放弃任务、返回空结果、生成看似完成但实际错误的 PR。这类需要靠 observability + eval 在生产持续监控。

### 9.5 Cost Explosion 监控

无论何种失败，最终都体现为**钱**。生产 agent 必须有：
- 每 user / org / task 的 cost budget 上限
- 实时 token / cost 告警（5 分钟超阈值即告警）
- 可视化的 trace 追溯哪一步烧了多少
- 周期性 audit："上周谁的 agent 烧得最多"

---

## 10. 安全与权限模型

### 10.1 Deny-First 权限架构（Claude Code 范本）

> Claude Code 的默认安全姿态是 **deny-first** + **human escalation**：deny 规则覆盖 ask 规则、ask 规则覆盖 allow 规则，未识别的 action 上升到用户而非默默放行。

具体实现的"多层独立护栏"：

1. **Permission Rules**（用户配置的 allow/deny 列表）
2. **PreToolUse Hooks**（可在工具调用前拦截）
3. **Auto-mode Classifier**（启用 auto mode 时的额外分类器）
4. **Shell Sandboxing**（可选层）

**任一层可阻断 action**，是真正的 defense-in-depth。

### 10.2 HITL vs HOTL

| 模式 | 含义 | 适合场景 |
|---|---|---|
| **HITL（Human in the Loop）** | 关键步骤前必须人工确认 | 高风险（医疗、金融、删除数据） |
| **HOTL（Human on the Loop）** | 人在监控但不阻塞，关键时介入 | 中等风险（编码、内容生成） |
| **Fully Autonomous** | 完成才汇报 | 低风险（数据查询、报表） |

### 10.3 Approval Fatigue：HITL 的隐形天花板

> Anthropic 的 auto-mode 数据：用户对 ~93% 的 permission prompt 一律点同意。

这意味着**单纯的 approval 机制已经被行为不可靠化**——人会习惯性 click yes。结论：

- 不能把"安全"完全押在"用户会看每个弹窗"上
- 必须有**技术性独立护栏**（rules + hooks + sandbox + classifier）
- 高风险 action 应该用**强制冷却 / 分层确认 / 双人确认**

### 10.4 沙箱方案：实战配置全集

#### 10.4.1 五档隔离对比

| 方案 | 隔离强度 | 启动时延 | 内存开销 | 兼容性 | 适合 |
|---|---|---|---|---|---|
| **MicroVM**（Firecracker / AWS Fargate / Kata） | **最强**（独立 Linux 内核 + KVM 硬件虚拟化） | ~125 ms | < 5 MiB / VM | 完全 Linux | 生产高安全（跑陌生人代码） |
| **gVisor** | **强**（用户态内核 Sentry + 拦截 syscall） | ~50 ms | ~15 MiB | 大部分 syscall | 平衡 + GPU 工作负载 |
| **加固容器** | 中（共享 host kernel + 多重限制） | < 10 ms | 微 | 完全 | 跑你自己写的代码 |
| **普通 Docker** | **弱**（仅 namespace + cgroup） | ~10 ms | 微 | 完全 | **绝对不能跑陌生 LLM 代码** |
| **Process-level**（seccomp / chroot） | 极弱 | < 1 ms | 0 | 完全 | 最后兜底 |

**业界共识**（来自 e2b / Daytona / Modal / Vercel Sandbox 等所有主流方案）：**容器单独不够，必须升级到 microVM 或 gVisor**。Linux 内核近 30 年的攻击面对 LLM 生成代码而言是大筛子——一个 namespace 逃逸 CVE 就能从沙箱跳到 host。

#### 10.4.2 Firecracker（AWS 出品的 microVM）

**架构**：每个 sandbox 是一个独立 Linux 内核 + 完整 init 进程，跑在 KVM 硬件虚拟化里。攻击者要 host root 必须先**穿透内核 + KVM hypervisor + Firecracker VMM** 三层。

**最小启动配置**：

```bash
# 1. 准备 kernel 和 rootfs
wget https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/kernels/vmlinux.bin
wget https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/rootfs/bionic.rootfs.ext4

# 2. 启动 Firecracker
firecracker --api-sock /tmp/firecracker.sock &

# 3. 配置 microVM（curl REST API）
curl -X PUT --unix-socket /tmp/firecracker.sock http://localhost/boot-source \
  -H 'Content-Type: application/json' \
  -d '{"kernel_image_path":"./vmlinux.bin","boot_args":"console=ttyS0 reboot=k panic=1"}'

curl -X PUT --unix-socket /tmp/firecracker.sock http://localhost/drives/rootfs \
  -d '{"drive_id":"rootfs","path_on_host":"./bionic.rootfs.ext4",
       "is_root_device":true,"is_read_only":false}'

# 资源限制（必填）
curl -X PUT --unix-socket /tmp/firecracker.sock http://localhost/machine-config \
  -d '{"vcpu_count":1,"mem_size_mib":256}'

# 网络（强烈建议默认拒绝出网）
curl -X PUT --unix-socket /tmp/firecracker.sock http://localhost/network-interfaces/eth0 \
  -d '{"iface_id":"eth0","host_dev_name":"tap0"}'

# 启动
curl -X PUT --unix-socket /tmp/firecracker.sock http://localhost/actions \
  -d '{"action_type":"InstanceStart"}'
```

**生产化要点**：

- 使用 [`firecracker-containerd`](https://github.com/firecracker-microvm/firecracker-containerd) 套 OCI 接口，agent 代码看起来像跑 docker
- snapshot / restore：起 1 个 base VM 跑系统初始化 → snapshot → 之后每个 sandbox 从 snapshot 启动（**冷启降到 ~125 ms，再热启 < 50 ms**）
- KVM 必须打开（裸金属 / 嵌套虚拟化云主机）

**E2B、Daytona、Modal 内部都是 Firecracker 包装版**，省去自己运维的麻烦：

```python
# E2B（最常用 LLM agent sandbox）
from e2b_code_interpreter import Sandbox
with Sandbox.create(timeout=30) as sbx:        # ★ 必须设 timeout
    result = sbx.run_code("print(2+2)")
```

#### 10.4.3 gVisor（Google 用户态内核）

**架构**：在应用和 host kernel 之间插入一个用户态进程 `Sentry`，**拦截所有 syscall** 并在用户态重新实现。应用看起来在 Linux 上跑，但实际只能调 Sentry 实现的子集。

**对比 Firecracker**：

- ✅ 启动更快（~50 ms vs 125 ms）
- ✅ 内存开销更低
- ✅ GPU passthrough 更好（gVisor 支持 NVIDIA driver 直通）
- ❌ syscall 兼容性不全（需要 unusual 系统调用的应用可能跑不动）
- ❌ 性能略慢（每个 syscall 多一层）

**最小配置**（OCI runtime）：

```bash
# 1. 安装 gVisor runsc
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null
sudo apt update && sudo apt install -y runsc

# 2. 把 runsc 注册为 Docker 的 runtime
sudo runsc install
sudo systemctl restart docker

# 3. 跑容器时指定 runtime
docker run --runtime=runsc --rm -it python:3.12 python -c "print('hello from gVisor')"
```

**Kubernetes 集成**（RuntimeClass）：

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
---
apiVersion: v1
kind: Pod
metadata:
  name: agent-sandbox
spec:
  runtimeClassName: gvisor   # ★ 这个 pod 跑在 gVisor 里
  containers:
  - name: code-runner
    image: python:3.12
    resources:
      limits: {memory: "256Mi", cpu: "500m"}
```

#### 10.4.4 加固容器（如果只能用 Docker）

**适用前提**：你跑的是**自己写的代码**（trusted），仅防意外破坏。**不要用加固容器跑陌生 LLM 生成代码**——community consensus 已 reject。

最小加固配置（每一项缺一不可）：

```bash
docker run \
  --rm \
  --user 1000:1000 \                        # 非 root
  --read-only \                              # 文件系统只读
  --tmpfs /tmp:size=64m,noexec \             # 临时空间不可执行
  --cap-drop ALL \                           # 丢掉所有 capabilities
  --security-opt no-new-privileges \         # 禁止 setuid
  --security-opt seccomp=./seccomp.json \    # 自定义 syscall 白名单
  --security-opt apparmor=docker-default \   # AppArmor profile
  --network=none \                           # 默认无网
  --pids-limit 64 \                          # 进程数上限
  --memory 256m \                            # 内存上限
  --cpus 0.5 \                               # CPU 上限
  --memory-swap 256m \                       # 禁用 swap
  python:3.12 python /code/script.py
```

**Rootless Docker / Podman** 进一步把 Docker daemon 也跑在 user namespace 里，攻击面再降一层。

**自定义 seccomp 白名单**（拒绝所有除明确允许的 syscall）：

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "syscalls": [
    {"names": ["read", "write", "openat", "close", "fstat", "mmap", "munmap",
               "brk", "rt_sigaction", "rt_sigprocmask", "ioctl", "exit_group",
               "execve", "wait4", "uname", "fcntl", "lseek", "stat", "newfstatat"],
     "action": "SCMP_ACT_ALLOW"}
    /* 拒绝：socket, connect, ptrace, mount, kexec_load, init_module, ... */
  ]
}
```

#### 10.4.5 选型决策与防御深度

```
你跑什么代码？
│
├─ 陌生人/LLM 生成的 untrusted code
│   ├─ 需要 GPU / 兼容更好    → gVisor
│   └─ 极致隔离/合规           → Firecracker microVM
│
├─ 你自己写的代码 + 防意外
│   └─ 加固容器（rootless + seccomp + cap-drop + read-only）
│
└─ 测试/开发              → 普通 Docker（不要进生产）
```

**Defense-in-depth：5 层防御**（一层失守还有别的）：

```
1. ┌───────────────────────────┐
   │  Network egress allowlist │ ← 默认拒绝出网，仅放行 GitHub/PyPI 等
   └───────────────────────────┘
2. ┌───────────────────────────┐
   │  Resource quota（CPU/MEM） │ ← 防 fork bomb / OOM
   └───────────────────────────┘
3. ┌───────────────────────────┐
   │  Time / step / cost cap   │ ← 防 Loop of Death (§9.1)
   └───────────────────────────┘
4. ┌───────────────────────────┐
   │  Sandbox isolation         │ ← Firecracker / gVisor
   └───────────────────────────┘
5. ┌───────────────────────────┐
   │  Audit log + alerting     │ ← 异常 syscall / 出网 / 高 cost
   └───────────────────────────┘
```

#### 10.4.6 主流商用 / 开源选择

| 服务 | 底层 | 接入方式 | 价格 |
|---|---|---|---|
| **E2B** | Firecracker | Python/TS SDK，最易接入 | 免费 + 按 vCPU·分钟 |
| **Modal** | gVisor + Firecracker | Python decorator | 按 GPU 秒计费 |
| **Daytona** | Firecracker | dev container 模式 | 开源 + 云 |
| **Vercel Sandbox** | Firecracker | Next.js 生态原生 | 包月 |
| **AWS Fargate / Lambda** | Firecracker | AWS-native | 按用量 |
| **Together Code Sandbox** | gVisor | API | 按调用 |
| **CodeSandbox** | Firecracker | dev IDE | 免费 + 订阅 |

实战经验：**新项目直接用 E2B 或 Modal**，省下 4-6 周自己运维 Firecracker 集群的时间。规模上去后再考虑自托管。

### 10.5 行业方向

> **业界不是在朝完全无监督的自主性走，而是在朝选择性自主性走。**

意思是：同一个 agent 在不同 action 上的自主级别应该不同。读 README 全自主，跑测试自动批准，改文件需要 review，删文件必须人工二次确认，操作生产 DB 必须 deny-first。

### 10.6 Tool Design 最佳实践（决定 agent 准确率的隐藏因素）

Tool 设计的好坏 **直接决定 agent 准确率**——不少团队折腾 prompt + 框架毫无效果，换 tool 描述方式后准确率提升 20%+。

#### 5 条铁律

**1. 命名要"动词+名词+宾语"**

```
❌ search() / get_data() / process()
✅ search_github_issues(repo, query)
   list_pending_orders_by_user(user_id)
   send_slack_message_to_channel(channel, text)
```

LLM 选 tool 时主要看名字，**模糊名 = 低召回**。

**2. 描述写"何时用 / 何时不用"**

```python
@tool
def cancel_order(order_id: str, reason: str) -> dict:
    """取消未发货的订单。

    何时用：
    - 用户明确要求取消订单
    - 订单状态为 pending / preparing
    - 用户已经登录验证

    何时不用：
    - 订单已发货（用 request_return 走退货流程）
    - 用户未登录（先走 require_login）
    - 仅是询问订单状态（用 get_order_status）

    参数：
    - order_id: 订单 ID（如 'ORD-12345'）
    - reason: 取消原因（用户提供的文本）

    返回：
    - {"status": "cancelled" | "failed", "refund_amount": float, "message": str}

    错误：
    - OrderNotFound: 订单不存在 → 让用户检查 ID
    - OrderAlreadyShipped: 已发货 → 引导走 request_return
    """
```

**3. 参数 schema 要严格 + 有 example**

用 Pydantic / JSON Schema 强类型：

```python
from pydantic import BaseModel, Field
from typing import Literal

class CancelOrderInput(BaseModel):
    order_id: str = Field(..., description="订单 ID，格式 'ORD-数字'", pattern=r"^ORD-\d+$",
                          examples=["ORD-12345"])
    reason: str = Field(..., description="取消原因", min_length=2, max_length=200,
                        examples=["用户改变主意", "找到更便宜的"])
    notify_user: bool = Field(True, description="是否通知用户")
```

`examples` 字段对 LLM 选参数 / 格式化参数有显著帮助。

**4. 错误返回要"告诉模型下一步怎么办"**

```python
# ❌ 不够
return {"error": "OrderNotFound"}

# ✅ 引导式
return {
    "error": "OrderNotFound",
    "message": "订单 ORD-12345 不存在",
    "suggestions": [
        "用 list_user_orders(user_id) 列出该用户所有订单",
        "提示用户检查订单号格式",
    ],
}
```

LLM 看到 `suggestions` 会自动尝试下一步，避免 Loop of Death。

**5. 工具数量上限：单个 agent 别给 > 20 工具**

LLM 在 20+ tools 中选错的概率显著上升（research 显示 40+ tools 准确率掉 20%+）。**生产模式**：

- **Tool 分组 + Routing**：先让 router agent 选 category（"order / billing / shipping"），再派给该 category 的 sub-agent，sub-agent 才看具体 5-10 个 tool
- **Dynamic tool loading**：根据当前对话上下文动态选哪些 tool 暴露给 LLM（节省 prompt token + 提高准确率）

#### Tool 设计的反模式（生产真实事故）

| 反模式 | 真实后果 | 修法 |
|---|---|---|
| 一个 `query_database(sql)` 给 LLM | LLM 写危险 SQL，删表事故 | 拆成 `lookup_user_by_email`、`get_order_status` 等专用 tool |
| Tool 名字含同义词（`fetch_user` + `get_user`） | LLM 随机选，结果不一致 | 合并 |
| 长结果不截断 | 返回 10K 行 → context 爆 | server 侧 cap + 加 pagination |
| 无超时 | tool 卡死 → agent 卡死 | 强制 timeout 30s |
| 副作用未声明 | 模型当只读调用 → 意外修改 | description 明确 "会修改 X"，加 require_confirmation |
| 长结果 JSON 嵌套 7 层 | LLM 解析不出来 | 摊平 + 加 summary 字段 |

#### Tool 集设计 checklist

- [ ] 每个 tool 名字看 5 秒能猜到做什么
- [ ] description 含"何时用 / 何时不用 / 参数说明 / 返回 / 错误"5 块
- [ ] 参数都有 schema + examples
- [ ] 错误返回带 suggestions
- [ ] 每个 tool 在自己业务的 eval 集上跑过
- [ ] 总 tool 数 ≤ 20，超过用 router 拆分
- [ ] 写操作 / 高风险操作有显式标记 + HITL 默认

---

## 11. Agent 在四层工程中的位置

把 Agent 放回四层工程的全景：

```
┌─────────────────────────────────────────────────────────────┐
│ Harness Engineering（驾驭工程，第 4 层）                     │
│   工具集成 / 沙箱 / 网关 / 推理 / 观测 / 评估 / 护栏        │
│   ─── 提供 agent 运行的"环境与约束"                          │
├─────────────────────────────────────────────────────────────┤
│ Agent Engineering（Agent 工程，第 3 层）← 本章               │
│   范式 / 框架 / 多 agent / 评估 / 失败模式 / 安全权限        │
│   ─── 让 LLM 自主完成多步任务                                 │
├─────────────────────────────────────────────────────────────┤
│ Context Engineering（上下文工程，第 2 层）                   │
│   长上下文 / RAG / 记忆 / MCP                                │
│   ─── 提供 agent "看到什么"                                  │
├─────────────────────────────────────────────────────────────┤
│ Prompt Engineering（提示词工程，第 1 层）                    │
│   Transformer / 训练栈 / 解码 / CoT / ReAct / 自动化         │
│   ─── 提供 agent "怎么思考"的模板                            │
└─────────────────────────────────────────────────────────────┘
```

四层不是替代而是叠加：
- 没好 prompt（第 1 层）→ agent 思考混乱
- 没管好 context（第 2 层）→ agent 视野盲区
- 没建好 agent loop / 框架（第 3 层）→ agent 一步步算对、整体跑偏
- 没 harness（第 4 层）→ agent 在生产里不可控

---

## 12. 落地建议

### 12.1 想入门 Agent 工程的 3 周路径

- **第 1 周**：理解 ReAct + 跑通 LangGraph 官方 quickstart + 接 1-2 个 MCP server
- **第 2 周**：为某个具体业务做一个 task agent（建议选"周报生成"或"数据查询助手"），加 max iterations + cost cap
- **第 3 周**：接 Langfuse observability + 跑几个 RAGAS / DeepEval 测试用例 + 给 1 个高风险 action 加 HITL

### 12.2 给已有 agent 的"6 项必做体检"

1. ✅ 有 max iterations / max tokens / max cost / max time 上限吗？
2. ✅ 所有写操作（改文件 / 删数据 / 发邮件）都有 deny-first 默认或 HITL 吗？
3. ✅ 有 sandbox 吗？最小权限原则吗？网络出口白名单吗？
4. ✅ 有 trace（每一步 tool call + LLM call）落到 observability 平台吗？
5. ✅ 有自动 eval 跑 regression 吗？有黄金集吗？
6. ✅ 出过事故吗？有事后复盘 + 加新规则吗？（这是 harness engineering 的核心心法）

### 12.3 不要做的 3 件事

- ❌ 在生产环境给 agent 直接 sudo / root / production DB 写权限
- ❌ 把"用户会确认"当成最终安全保障（93% 一律 yes）
- ❌ 看 SWE-bench 公开排名就选模型，必须自己业务上重测

---

## Bibliography

### Agent 范式与论文
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
- Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*.
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*.

### Agent 架构与框架（2026）
- Lindy. *AI Agent Architecture: A Complete Guide for 2026*. <https://www.lindy.ai/blog/ai-agent-architecture>
- Redis. *AI Agent Architecture: Build Systems That Work in 2026*. <https://redis.io/blog/ai-agent-architecture/>
- arXiv. *AI Agent Systems: Architectures, Applications, and Evaluation* (2026). <https://arxiv.org/html/2601.01743v1>
- ATNO. *10 AI Agent Frameworks You Should Know in 2026*. Medium, 2026-04.
- Anubhav. *LangGraph vs CrewAI vs AutoGen: Which Agent Framework Should You Actually Use in 2026*. Data Science Collective, 2026-03.
- digitalapplied. *OpenAI Agents SDK vs LangGraph vs CrewAI: 2026 Matrix*.

### 多 Agent 协作
- gurusup. *Agent Orchestration Patterns: Swarm vs Mesh vs Hierarchical*.
- agixtech. *Conductor vs. Swarm: Multi-Agent AI Architecture Guide 2026*.
- LangGraph dev.to. *Multi-Agent Orchestration in LangGraph: Supervisor vs Swarm*.

### Agent 基准
- Berkeley RDI. *How We Broke Top AI Agent Benchmarks*. <https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/>
- SWE-bench Leaderboards. <https://www.swebench.com/>
- Steel.dev. *AI Agent Benchmark Results Index*. <https://leaderboard.steel.dev/results>
- Spheron. *AI Agent Benchmarking Infrastructure on GPU Cloud (2026 Guide)*.
- philschmid/ai-agent-benchmark-compendium. <https://github.com/philschmid/ai-agent-benchmark-compendium>
- MarkTechPost. *Top 7 Benchmarks That Actually Matter for Agentic Reasoning in LLMs* (2026-04-26).

### Agent 真实产品
- artificialanalysis.ai. *Coding Agents Comparison: Cursor, Claude Code, GitHub Copilot, and more*.
- SitePoint. *Claude Code vs Cursor vs Copilot: The 2026 Developer Comparison*.
- Tech Insider. *Anthropic's Claude Computer Use Agent*.
- WorkOS. *Anthropic's Computer Use versus OpenAI's Computer Using Agent (CUA)*.
- digitalapplied. *Computer Use Agents 2026: Claude vs OpenAI vs Gemini*.
- arXiv. *Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems* (2026). <https://arxiv.org/html/2604.14228v1>

### Agent 失败模式与安全
- Galileo. *7 AI Agent Failure Modes and How to Prevent Them*.
- AgentWiki. *Common Agent Failure Modes*. <https://agentwiki.org/common_agent_failure_modes>
- vectara/awesome-agent-failures. <https://github.com/vectara/awesome-agent-failures>
- Sattyam Jain. *The "Loop of Death": Why 90% of Autonomous Agents Fail in Production*. Medium 2026-01.
- earezki. *5 Silent Failures in Autonomous AI Agents: A Midnight Audit Case Study* (2026-04-16).
- Fortune. *What do you do when your AI agent hallucinates with your money?* (2026-04-08).
- Northflank. *How to sandbox AI agents in 2026: MicroVMs, gVisor & isolation strategies*.
- Strata. *Human-in-the-Loop: A 2026 Guide to AI Oversight*.
- Waxell. *Human-in-the-Loop vs Human-on-the-Loop for AI Agents*.
- Anna Jey. *Human-in-the-Loop AI Agents: Approvals, Escalation, Safe Autonomy in Production*. Medium 2026-04.
- Arthur. *Agentic AI Observability: A 2026 Playbook*.
- Undercode Testing. *How A Opus 46 AI Agent Nuked A Production Database In 9 Seconds*.

---

## 13. 端到端完整 Agent 案例：带 RAG + Memory + Guardrail 的客服 Agent

把前面所有概念串起来，下面是一个生产级最小骨架——LangGraph + 工具 + 检索 + 记忆 + max iter + cost cap + observability。

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from mem0 import Memory
import langfuse_callback  # observability
import time

# ============ State ============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    iterations: int
    total_cost_usd: float
    started_at: float

# ============ 工具：知识库检索 ============
@tool
def kb_search(query: str) -> str:
    """检索企业知识库回答用户问题"""
    # 调 §2.4 RAG pipeline，省略
    return rag_chain.invoke({"input": query})["answer"]

@tool
def create_ticket(user_id: str, summary: str, severity: str) -> str:
    """无法回答时创建工单。severity: low/medium/high/critical"""
    if severity == "critical":
        # critical 必须 HITL，不允许 agent 自主创建
        return "ERROR: critical 工单需人工创建"
    ticket_id = ticket_system.create(user_id, summary, severity)
    return f"工单 {ticket_id} 已创建"

# ============ 长期记忆 ============
memory = Memory.from_config({"vector_store": {"provider": "qdrant"}})

# ============ LLM + 工具绑定 + observability ============
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    callbacks=[langfuse_callback.LangfuseCallback()],
).bind_tools([kb_search, create_ticket])

# ============ Guardrail：输入侧（伪代码）============
def input_guardrail(messages):
    last_user = messages[-1].content
    if guardrail.detect_prompt_injection(last_user):
        raise ValueError("输入触发 prompt injection 防护")
    if guardrail.contains_pii(last_user):
        last_user = guardrail.redact_pii(last_user)
        messages[-1] = HumanMessage(content=last_user)
    return messages

# ============ Agent 节点 ============
def agent_node(state: AgentState):
    # 1. 输入 guardrail
    messages = input_guardrail(state["messages"])

    # 2. 注入历史记忆
    user_memories = memory.search(query=messages[-1].content, user_id=state["user_id"])
    memory_context = "\n".join([m["text"] for m in user_memories[:5]])
    sys_msg = SystemMessage(content=f"你是 ACME 公司客服。用户历史:\n{memory_context}")

    # 3. LLM 决策（思考 + 工具调用）
    response = llm.invoke([sys_msg] + list(messages))

    # 4. 更新成本（每个 provider 都返回 token usage）
    cost = response.response_metadata.get("usage", {}).get("total_cost_usd", 0)

    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
        "total_cost_usd": state["total_cost_usd"] + cost,
    }

# ============ 工具节点 ============
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([kb_search, create_ticket])

# ============ 流程控制 ============
def should_continue(state: AgentState):
    # ★★★ Loop of Death 防御：四道闸门 ★★★
    if state["iterations"] >= 10:
        return END  # max iter cap
    if state["total_cost_usd"] >= 0.50:
        return END  # cost cap $0.5/会话
    if time.time() - state["started_at"] >= 60:
        return END  # 超时 60s
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return END  # 没新工具调用 = 完成
    return "tools"

# ============ 构图 ============
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# ============ 编译（带 checkpoint 支持 HITL）============
from langgraph.checkpoint.memory import MemorySaver
app = graph.compile(checkpointer=MemorySaver())

# ============ 运行 ============
config = {"configurable": {"thread_id": "session_42"}}
result = app.invoke(
    {
        "messages": [HumanMessage(content="我的订单 #12345 还没收到，请帮查")],
        "user_id": "u_abc",
        "iterations": 0,
        "total_cost_usd": 0.0,
        "started_at": time.time(),
    },
    config=config,
)

# 把这次会话沉淀到长期记忆
memory.add(messages=result["messages"], user_id="u_abc")
```

这个 100 行骨架已经具备生产级 agent 的核心要素：

| 要素 | 实现位置 |
|---|---|
| LLM + 工具循环（ReAct） | `agent_node` + `tool_node` |
| 长期记忆 | `memory.search` / `memory.add`（Mem0） |
| RAG 知识库 | `kb_search` 工具 |
| 输入护栏 | `input_guardrail` |
| Loop of Death 四闸门 | `should_continue`（max iter + cost cap + timeout + 完成检测） |
| HITL（critical 工单） | `create_ticket` 工具内强制 raise |
| Checkpoint / 暂停恢复 | `MemorySaver` |
| Observability | `LangfuseCallback` |
| 成本追踪 | `total_cost_usd` 累计 |

补全到生产还需加：自动化 eval（DeepEval / RAGAS）、A/B test、灰度发布、PagerDuty 告警接入、合规日志、敏感词输出 guardrail。

---

## 14. GitHub 系统化学习资源（Agent 专题）

| 资源 | 类型 | 推荐用法 |
|---|---|---|
| [`microsoft/ai-agents-for-beginners`](https://github.com/microsoft/ai-agents-for-beginners) | 12 课入门 | 完整新手入门，每节带可跑 code samples，配套 Microsoft Agent Framework |
| [`microsoft/agent-framework`](https://github.com/microsoft/agent-framework) | 官方框架 | AutoGen 的接班人，Python + .NET，企业 agent orchestration |
| [`datawhalechina/hello-agents`](https://github.com/datawhalechina/hello-agents) | 中文系统教程（42K ⭐） | 从原理到实现到 Agentic RL，中文圈最系统 |
| [`anthropic/Building-Effective-Agents`](https://www.anthropic.com/research/building-effective-agents) | Anthropic 官方文章 | §3.4 6 patterns 的原始来源，**必读** |
| [`jcran/effective-agents-langchain`](https://github.com/jcran/effective-agents-langchain) | LangChain 实现 | Anthropic 6 patterns 的 LangChain 代码版 |
| [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) | LangGraph 官方 | 主流生产 agent 框架，文档极详 |
| [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | CrewAI 官方 | 角色 DSL，最快上手 |
| [`openai/openai-agents-python`](https://github.com/openai/openai-agents-python) | OpenAI Agents SDK | OpenAI 官方，handoff 哲学 |
| [`huggingface/smolagents`](https://github.com/huggingface/smolagents) | Smolagents | 几百行核心代码，最简实现 |
| [`mastra-ai/mastra`](https://github.com/mastra-ai/mastra) | Mastra | TypeScript 原生，JS/TS 项目首选 |
| [`philschmid/ai-agent-benchmark-compendium`](https://github.com/philschmid/ai-agent-benchmark-compendium) | 50+ benchmark 汇总 | 找特定能力测试时 |
| [`vectara/awesome-agent-failures`](https://github.com/vectara/awesome-agent-failures) | 失败模式集 | 写 production agent 必看的"反面教材" |

**论文必读清单**（按时间）：
- 2022 — *ReAct* (Yao et al.)
- 2023 — *Self-Refine* (NeurIPS), *Reflexion* (Shinn et al.), *MemGPT* (Packer et al.)
- 2024 — *Building Effective AI Agents* (Anthropic), *Self-RAG* (Asai et al.)
- 2025 — *Mem0* (arxiv 2504.19413)
- 2026 — *Dive into Claude Code* (arxiv 2604.14228)

---

## 章节交叉引用

- 想深入 ReAct 提示词模板与 CoT → 见 [prompt-engineering.md](./prompt-engineering.md) §1.4
- 想深入 RAG 与记忆系统的工程细节 → 见 [context-engineering.md](./context-engineering.md) §2.4 / §2.5
- 想深入工具集成 / 沙箱 / 护栏（harness 强约束的直接实现） → 见 [harness-engineering.md](./harness-engineering.md) §3.3.1-§3.3.3；想看 observability / 评估（相关但非 harness 的支撑设施） → 见 §3.9.3-§3.9.4
