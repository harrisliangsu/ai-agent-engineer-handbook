# Agent engineering: from ReAct to autonomous systems

> **English** · [中文](./agent-engineering.md)

**Position:** The third layer of the four-layer progression of LLM application engineering (between context engineering and harness engineering)
**Research date:** 2026-05-01

---

<a id="executive-summary"></a>

## Executive Summary

An agent is not a fancier LLM call. It is the quartet **"LLM + tools + memory + loop"** — given a goal, it plans on its own, takes action, observes the result, and adjusts its strategy until the task is done or it is explicitly stopped. From an engineering standpoint, an agent is the unit of abstraction that cuts vertically through the prompt, context, and harness layers: its **thinking** is given templates by prompt engineering (ReAct and others), its **field of view** is supplied by context engineering (RAG, memory, tool schemas), and its **runtime** is built by harness engineering (sandbox, gateway, guardrails, observability).

In 2026, agents have split into three clear tracks:

- **Coding Agent** (Claude Code, Cursor, GitHub Copilot Workspace, Devin, Aider) — the most mature, and the fastest to commercialize
- **Browser / Web Agent** (WebArena-class tasks, Operator) — still short of a breakthrough on anti-bot defenses for real websites, and on reliability for multi-step tasks
- **Computer Use Agent** (Anthropic Computer Use from 2024-Q4, OpenAI Codex Background Computer Use from 2026-04) — operates the mouse and keyboard directly, across application workflows

Mainstream frameworks have converged from a wide proliferation to a few that dominate: **LangGraph (the production default) / CrewAI (the fastest to pick up) / OpenAI Agents SDK (tied to the OpenAI ecosystem) / Smolagents (minimal) / Mastra (TypeScript-native)**. Microsoft has moved AutoGen into maintenance, and Microsoft Agent Framework has taken over.

On the latest rankings as of April 2026 for the mainstream benchmarks (SWE-bench Verified / GAIA / WebArena / OSWorld), **Anthropic models nearly monopolize the top**. Claude Opus 4.7 is reported to land in the ~87% range on SWE-bench Verified (for the exact score and the live board, see Anthropic's official release page and the swebench.com leaderboard, as of this survey's date 2026-05-01; scores change with version and with evaluation criteria). Berkeley RDI also warns: **a set of mainstream agent benchmarks (including SWE-bench / WebArena / GAIA / Terminal-Bench and others) have been shown to be hackable to near-perfect scores without the task actually being completed** — a score has to be read against the specific evaluation methodology.

Failure modes and real incidents are what most deserve caution: **90% of production agents die in a Loop of Death (an infinite loop)**; a single Claude Code sub-agent burned 27 M tokens and ran for 4.6 hours; Amazon Kiro AI autonomously deleted a production AWS environment and caused a 13-hour outage; Replit AI deleted a production database during a code freeze; Google AI wiped an entire disk while clearing a cache. There is one lesson: **without a max iteration cap, without an execution timeout, without a sandbox, and without a deny-first permission model, an agent in production is a bomb**.

---

<a id="s1"></a>

## 1. Definition of an agent, and its coordinates

<a id="s1-1"></a>

### 1.1 What it is not

To understand what an agent is, first rule out three common confusions:

- **Agent ≠ Chatbot**: a chatbot is a one-shot input-to-output map. Between calls there is no state, no goal, and no tools
- **Agent ≠ Workflow**: a workflow is a fixed sequence of steps laid out in advance. An agent decides for itself what the next step should be
- **Agent ≠ Single LLM Call with Tools**: a single function-calling invocation is "the model tells you whether to call a tool." An agent is "the model calls tools on its own, many times, inside a loop, until the goal is reached"

Anthropic's engineering blog gives a widely cited definition: **"An agent is the pattern of an LLM using tools in a loop"** — "loop" and "autonomy" are the two core features.

<a id="s1-2"></a>

### 1.2 The engineering minimum

A minimal agent has to have four things:

| Element | Meaning | If missing, it becomes |
|---|---|---|
| **Goal** | A goal | No goal = chatbot |
| **Tools** | Callable capabilities (API / tool / compute / retrieval) | No tools = pure generation |
| **Loop** | A think-act-observe loop | No loop = a single call |
| **Memory** | State kept across iterations of the loop | No memory = an amnesiac agent |

<a id="s1-3"></a>

### 1.3 The autonomy spectrum

An agent's "degree of autonomy" is a continuous spectrum, not a binary variable:

```
Human-in-the-Loop          Human-on-the-Loop          Fully Autonomous
─────────────────          ─────────────────          ─────────────────
每步行动都需用户确认       人在旁监督，关键步骤介入   完成才汇报
适合：高风险/医疗/金融     适合：中等风险/编码         适合：低风险/数据处理
e.g. 早期 Claude Code      e.g. Cursor agent mode     e.g. Devin / Manus
```

The 2026 industry consensus is this: **the industry is not moving toward fully unsupervised autonomy; it is moving toward selective autonomy**. Different actions inside one product should have different autonomy levels — reading a file can be fully autonomous; deleting a file must be HITL.

---

<a id="s2"></a>

## 2. How agent paradigms evolved

| Year | Paradigm | Paper / source | Core |
|---|---|---|---|
| 2022 | **ReAct** | Yao et al. | The minimal loop that interleaves Thought → Action → Observation |
| 2023 | **Plan-and-Execute** | LangChain | Plan the whole thing first, then execute, avoiding ReAct's short-sightedness |
| 2023 | **Reflexion** | Shinn et al. | After a failure, write a "reflection note" and attach it to the next prompt, so the agent learns from experience |
| 2023 | **Tree-of-Thoughts as agent** | Yao et al. | Extend ToT reasoning into search at the agent level |
| 2024 | **Constitutional Agent** | Anthropic | Constrain agent behavior with a constitution / principles |
| 2024 | **Multi-Agent System** | AutoGen / CrewAI | Role division and negotiation among multiple agents |
| 2024-Q4 | **Computer Use Agent** | Anthropic | The agent operates a GUI directly (screenshot + mouse-and-keyboard tool) |
| 2025 | **Agentic RAG** | LangGraph | Delegate the RAG decision to the agent, which decides on its own when to retrieve and whether to retrieve again |
| 2026-Q2 | **Background Computer Use** | OpenAI Codex (2026-04) | The agent runs in its own desktop session, with multiple agents in parallel, without disturbing the main workflow |

ReAct is still the "base grammar" of almost every modern agent. The other paradigms add planning, reflection, search, multiple agents, and cross-modality on top of ReAct.

---

<a id="s3"></a>

## 3. Agent architecture: a common 4-layer split

There is no shared industry consensus on a "standard N-layer model of agent architecture" — Lindy, Redis, arXiv, and others each split it differently (some use 5 layers of perception-cognition-action, some a single ReAct loop, and so on). The 4 layers below are the author's simplified checklist, synthesized from several of those splits. **They are not a canonical taxonomy.** They are only a way to inventory the engineering components:

<a id="s3-1"></a>

### 3.1 Four-layer structure (a synthesized split)

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

<a id="s3-2"></a>

### 3.2 The classic control loop

```
   ┌─→ Goal ──→ Perception ──→ Reasoning ──→ Planning
   │                                              │
   │                                              ▼
   │                                          Action
   │                                              │
   └─── Memory Update ←── Observation ←──────────┘
```

The loop continues until: (a) the goal is reached; (b) a stop condition fires (max iterations, timeout, user interrupt); (c) the agent itself judges "I cannot finish this; a human needs to step in."

<a id="s3-3"></a>

### 3.3 Key design decisions at each layer

- **Reasoning Layer**: Which LLM? A base model or a reasoning model (o1 / DeepSeek-R1)? Turn on thinking mode?
- **Orchestration Layer**: Do you need checkpoints (crash recovery)? Time-travel debugging? Human-in-the-loop pause points?
- **Memory & Data Layer**: Mem0 / Letta / Zep / in-house? Keep short-term and long-term separate, or unify them? How is memory evicted?
- **Tool Integration Layer**: Function calling or MCP? What is the cap on the number of tools? How are tool descriptions written?

See chapter 02 (context engineering) for the RAG and memory sections, and chapter 04 (harness engineering) for tool integration and orchestration.

<a id="s3-4"></a>

### 3.4 Anthropic's 6 composable agent patterns (required foundational reading)

*Building Effective AI Agents*, published by Anthropic in December 2024, is one of the most cited essays on productionizing agents in 2025-2026. It splits "agent" systems into two kinds:

- **Workflows**: an LLM and tools run along a **predefined code path** (a deterministic process)
- **Agents**: the LLM **dynamically decides the process and the tools** (an autonomous process)

> "Workflows are great when tasks are well-defined; agents shine when flexibility is required. Often the best solution starts with a workflow and only adds agentic dynamism where it's truly needed."

It also proposes **6 composable building blocks** — the base primitives for building any agentic system.

<a id="s3-4-1"></a>

#### 3.4.1 Building Block 0: Augmented LLM (the base LLM)

The most atomic "agent": an LLM plus three augmentations (retrieval / tools / memory). Every pattern above it is stacked on top of this one.

```
            ┌──────────┐
   Input ───▶│   LLM    │───▶ Output
            │  + tools  │
            │  + memory │
            │  + retrieval│
            └──────────┘
```

<a id="s3-4-2"></a>

#### 3.4.2 Pattern 1: Prompt Chaining (a chain of prompts)

Split the task into a sequence of steps. Each step's LLM processes the previous step's output. A gate (a program that decides whether to continue) can sit between steps.

```
Input → LLM 1 → [Gate]? → LLM 2 → [Gate]? → LLM 3 → Output
```

**Typical setting**: generate an outline → check that the outline matches the format → generate the body from the outline → check the length → translate.

**Strength**: each step can be optimized and tested on its own, and the process can be traced.
**Weakness**: more steps means higher latency; information can be lost between steps.
**When to use**: the task decomposes cleanly into linear steps, and each step has a clear quality gate.

<a id="s3-4-3"></a>

#### 3.4.3 Pattern 2: Routing (dispatch)

One LLM looks at the input and decides **which specialized path to take**. Each path is optimized with its own prompt / tools / model.

```
                  ┌─→ Specialist LLM A (用 prompt_A, tools_A)
Input → Router LLM┤
                  ├─→ Specialist LLM B (用 prompt_B, tools_B)
                  └─→ Specialist LLM C (用 prompt_C, tools_C)
```

**Typical setting**: customer-support triage (a refund → a refund-specialist prompt; a technical issue → a technical prompt; chitchat → a general prompt); route by difficulty (a cheap model for an easy question, Opus for a hard one).

**Strength**: it makes "which prompt for which question" explicit, so each path can be optimized on purpose.
**Weakness**: if the router is wrong, everything downstream is wrong.
**When to use**: an input space with obvious categories.

<a id="s3-4-4"></a>

#### 3.4.4 Pattern 3: Parallelization (parallel execution)

Two variants:

- **Sectioning**: split the task into independent subtasks, run them in parallel, then aggregate (the `map-reduce` pattern)
- **Voting**: run the same task several times (different prompts or different models) and aggregate by majority or by average

```
              ┌─→ LLM 1 ─┐
Input ───────┼─→ LLM 2 ─┼──▶ Aggregator LLM ──▶ Output
              └─→ LLM 3 ─┘
```

**Typical settings**:
- Sectioning: split a long document into sections and summarize them in parallel; split a safety review across dimensions and check them in parallel (toxicity / PII / bias / injection)
- Voting: multi-model voting on a code review; multi-prompt voting on content moderation, to cut false positives

**Strength**: latency is low and reliability is high (voting resists a single failure).
**Weakness**: the cost is N times a single run.
**When to use**: the subtasks are truly independent, or you need redundant reliability.

<a id="s3-4-5"></a>

#### 3.4.5 Pattern 4: Orchestrator-Workers (conductor and workers)

A central LLM **dynamically decides the subtasks** (they are not predefined), dispatches them to worker LLMs, and synthesizes the result at the end.

```
                    ┌─→ Worker 1 (动态决定的子任务)
Input → Orchestrator┤
       LLM (动态规划)├─→ Worker 2
                    └─→ Worker 3
                          ↓
                    Synthesizer LLM ──▶ Output
```

The key difference from Parallelization: **the subtasks cannot be defined in advance**. The orchestrator sees the concrete input, and only then decides which subtasks to split it into.

**Typical settings**:
- A code change across several files (only after reading the issue does the agent know which files to change)
- A complex research task (it first has to decide which angles to look up before it can answer)

**Strength**: it handles complex tasks that are not known in advance.
**Weakness**: cost and latency are high, and debugging is harder.
**When to use**: a standard workflow cannot solve it, and the input is highly complex.

<a id="s3-4-6"></a>

#### 3.4.6 Pattern 5: Evaluator-Optimizer (an evaluate-and-optimize loop)

One LLM generates a response. Another LLM evaluates it and gives feedback. The first revises from that feedback. The loop continues until the response passes, or a cap is hit.

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

**Typical settings**:
- Literary translation (the generator translates; the evaluator checks faithfulness, expressiveness, and elegance, and feeds back "too literal" and the like)
- Complex search (the generator writes an answer; the evaluator checks coverage and feeds back "the X angle is still missing")
- Code generation (the generator writes code; the evaluator runs the tests and feeds back the failing cases)

**Strength**: a high ceiling on quality (iterative quality close to a human review).
**Weakness**: the iteration count is not controllable, it can loop forever, and the cost is high.
**When to use**: the evaluation criteria are clear, and the first draft is clearly worse than the result after iteration.

<a id="s3-4-7"></a>

#### 3.4.7 Pattern 6: Agents (autonomous agents)

The previous five are workflows (the process is predefined). Agents are genuinely autonomous — inside the loop, the LLM **decides for itself what to do next, which tool to use, and when to stop**.

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

**When to upgrade to a real agent**:

- The number and kind of steps are **not known in advance**
- The task runs for a long time (minutes-hours)
- What the user gives is a **goal**, not **steps**
- The environment changes, and the system has to adapt

**When not to use an agent** (continue using workflow):

- The task decomposes cleanly into predefined steps
- Latency-sensitive (an agent loop is at least N LLM calls)
- Cost-sensitive
- Strict predictability is required (compliance / audit settings)

<a id="s3-4-8"></a>

#### 3.4.8 Decision tree for choosing among the 6 patterns

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

**Anthropic's core advice**: **stay with a workflow (patterns 1-5) for as long as you can, and upgrade to an agent only when you truly need one**. In production, 95% of "agentic systems" are served well enough by a workflow. Using an agent blindly brings in behavior that is hard to debug, unpredictable, and prone to runaway cost (see §9, failure modes).

---

<a id="s4"></a>

## 4. A taxonomy of agent types

<a id="s4-1"></a>

### 4.1 Classified by interaction mode

| Type | Examples | Traits |
|---|---|---|
| **Conversational Agent** | ChatGPT / Claude.ai with tools | Conversation first; agent capability is supplementary |
| **Task Agent** | Zapier AI Actions / Make AI | After a trigger, it runs one process and stops |
| **Coding Agent** | Claude Code / Cursor / Devin | Long-running; edits files, runs tests, opens PRs |
| **Browser Agent** | Multi-On / Browser Use / Skyvern | Clicks, fills forms, and scrapes on web pages |
| **Computer Use Agent** | Anthropic Computer Use / OpenAI Operator | Operates mouse and keyboard across applications; reads the screen |
| **Background Agent** | OpenAI Codex Background Computer Use (2026-04) | Runs in its own session; multiple agents in parallel |
| **Multi-Agent System** | CrewAI / AutoGen applications | Several agents collaborate to finish a task |

<a id="s4-2"></a>

### 4.2 Classified by deployment form

- **CLI Agent**: Claude Code, Aider, OpenAI Codex CLI
- **IDE Extension**: GitHub Copilot Chat agent mode, Cline, Continue
- **Standalone IDE**: Cursor, Windsurf
- **Cloud Agent**: Devin (a separate cloud environment with a browser, a terminal, and an editor), Manus, GitHub Copilot Workspace
- **Browser Plugin**: Operator (in the form of a Chrome extension)
- **Desktop App**: Anthropic Claude Computer Use (macOS research preview)

<a id="s4-3"></a>

### 4.3 Classified by degree of autonomy (see 1.3)

<a id="s4-4"></a>

### 4.4 Browser agents: AI applied to web automation

The browser agent is the third-largest agent track of 2025-2026 (behind only coding and computer use). Benchmarks such as WebArena and WebVoyager have pushed this track to the SOTA.

#### Mainstream frameworks (2026)

| Framework | Language | Paradigm | Strength | ⭐ / status |
|---|---|---|---|---|
| **Browser Use** | Python | Fully autonomous AI (agent loop) | 91k ⭐, 89.1% success rate on WebVoyager | Current SOTA |
| **Stagehand** (Browserbase) | TypeScript | AI primitives on top of Playwright | 4 primitives: act / extract / observe / agent; readable and controllable | Rising strongly |
| **Playwright** | Multi-language | Deterministic automation | No AI; pure scripts; the fallback | Testing baseline |
| **Skyvern** | Python | LLM + computer vision | Visual understanding of pages it has not seen before | A startup product |
| **Browserbase** (SaaS) | API | Cloud stealth browser | Strongest against anti-bot defenses | The most commercially mature |

#### Browser Use (Python, the most popular): minimal code

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

#### Stagehand (TypeScript, the most readable)

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

Stagehand's design philosophy: **the AI writes the selectors; the human writes the control flow**. Brittle CSS selectors are replaced by natural-language `act`, while Playwright's deterministic control of the flow stays. **The industry consensus: Stagehand's hybrid architecture will be the template.**

#### How to choose

```
你的任务是？
│
├─ 完全未知网站、AI 决定流程 → Browser Use（自主性最强）
├─ 流程基本已知、要稳定 → Stagehand（AI 局部 + 流程可控）
├─ 流程完全确定 → Playwright（不要用 AI）
├─ 视觉为主、HTML 不可靠 → Skyvern
└─ 需要反反爬云服务 → Browserbase（SaaS）+ Stagehand
```

#### Common pitfalls of browser agents

- **Slow**: each LLM call takes 3-10s; a 10-step task is already a minute
- **Expensive**: each step carries a screenshot plus the DOM; token use is 10-50k per step
- **Flaky**: a site redesign breaks it; re-train on a schedule, or add retries
- **Anti-bot**: many sites block headless browsers or detect automation signatures; a cloud stealth browser (Browserbase) is required
- **Login / captcha**: a human has to step in, or you need a cookie-injection scheme

<a id="s4-5"></a>

### 4.5 Voice / Realtime Agent

In the second half of 2024, the OpenAI Realtime API opened the era of realtime voice agents. The 2026 mainstream:

| Approach | Mode | Latency | Fit |
|---|---|---|---|
| **OpenAI Realtime API** (GPT-4o / GPT-5) | Bidirectional audio streaming (WebSocket) | < 500ms | Customer support / personal assistant |
| **Anthropic Voice** (Claude) | Text + TTS (asynchronous) | 2-5s | Content-generation settings |
| **Gemini Live API** | Bidirectional multimodal (including video) | < 1s | Meeting assistant |
| **Pipecat** / **LiveKit Agents** | Open-source frameworks that orchestrate STT + LLM + TTS | 1-3s | Self-hosted; flexible |
| **Vapi** / **Retell** | SaaS; a hosted full stack for voice agents | < 1s | Startup MVP |

#### Minimal Realtime API code

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

#### Engineering points for voice agents

- **VAD (Voice Activity Detection)**: detect whether the user has finished speaking. Server-side VAD is the simplest
- **Interruption handling**: the moment the user speaks, stop TTS, so the two sides do not talk over each other
- **Latency budget**: under 800ms it feels natural; over 2s, the user leaves
- **Background noise**: denoise with RNNoise / krisp and the like
- **Dialects / accents**: accuracy has to be measured on your own users' data

---

<a id="s5"></a>

## 5. Mainstream agent frameworks compared (2026)

After the two-year Cambrian explosion of 2024-2025, the 2026 agent-framework landscape has clear leaders:

| Framework | Paradigm | Model support | Learning curve | Production readiness | Fit |
|---|---|---|---|---|---|
| **LangGraph** | Explicit state machine / graph | Any | Medium | Highest (LangSmith integration, checkpoint, time-travel debug, HITL) | Complex production agents; explainable flows |
| **CrewAI** | Role + task DSL | Any | Lowest (starts at 20 lines) | Medium | Fast business prototypes; multi-agent role collaboration |
| **AutoGen** | Multi-agent conversational negotiation | Any | Medium | **In maintenance** (Microsoft has shifted to Microsoft Agent Framework) | Maintaining legacy projects |
| **OpenAI Agents SDK** | Imperative handoff chain | OpenAI only | Extremely low (minimal mental model + first-class tracing) | Medium | OpenAI-ecosystem first |
| **Anthropic Agent Skills** | Skill packages (released in 2026) | Claude only | Low | Medium | The Claude ecosystem; reusable capabilities |
| **Smolagents** | Minimal (a core of a few hundred lines) | Any | Low | Low (mainly teaching / research) | Learning; the HuggingFace ecosystem |
| **Mastra** | TypeScript-native | Any | Low | Medium (growing fast) | JS/TS projects; Vercel / Next.js |
| **Pydantic AI** | Strongly typed Python | Any | Medium | Medium | Python projects where type safety matters |
| **Google ADK** | / | Gemini-first | Medium | Medium | The Gemini / Vertex AI ecosystem |
| **LlamaIndex Agent** | RAG-first | Any | Medium | Medium | RAG-heavy settings |

<a id="s5-1"></a>

### 5.1 The three frameworks compared most often

**LangGraph vs CrewAI vs OpenAI Agents SDK** is the three-way comparison discussed most in 2026:

- **Architecture philosophy**: LangGraph models the agent as a state machine on a graph; CrewAI models the agent as a role-driven crew; the OpenAI Agents SDK models the agent as an imperative handoff chain
- **Production readiness**: LangGraph is the strongest (LangSmith observability, checkpointing, streaming, graph visualization, and time-travel debugging come with it)
- **Speed of getting started**: on a standard business process, CrewAI needs the least startup code ("time to production" is clearly shorter than LangGraph in several blog comparisons; the exact percentage varies by task). A standard business process starts in ~20 lines
- **Mental model**: the OpenAI Agents SDK is the most compact, but it locks you to OpenAI models

<a id="s5-2"></a>

### 5.2 How to choose a framework

- **A new project, with complex control flow, maintained for a long time** → LangGraph
- **A startup MVP, built around a business process** → CrewAI
- **Already bound to OpenAI models** → OpenAI Agents SDK
- **TypeScript / Next.js** → Mastra or the Vercel AI SDK
- **Teaching / research** → Smolagents
- **Type safety is critical** → Pydantic AI
- **A multi-agent brainstorming mode** → AutoGen still works, but avoid it for new projects (already in maintenance)

<a id="s5-3"></a>

### 5.3 "Hello World" compared across the three mainstream frameworks

The same task: "**a ReAct agent that can answer questions with a web-search tool**". Watch how the three code philosophies differ.

#### LangGraph (state machine / graph)

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

**Traits**: defined as a graph, state explicit, checkpoint built in, time-travel debug.

#### CrewAI (role + task DSL)

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

**Traits**: a three-layer DSL of role + task + crew; starts at 20 lines; the fastest to pick up.

#### OpenAI Agents SDK (imperative handoff)

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

**Traits**: handoff is the SDK's core abstraction; tracing is first-class; the mental model is minimal.

#### How the three philosophies compare

| Dimension | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|
| Abstraction | State + Graph | Role + Task + Crew | Agent + Tools + Handoff |
| Mental model | State-machine programmer | Team manager | Function calls |
| Control-flow visibility | Very high (the graph) | Medium (the task flow) | Low (imperative) |
| Debugging tools | LangSmith time-travel | Default logs | OpenAI tracing |
| Extensibility | Very strong (custom nodes) | Medium (custom tasks) | Medium (custom tools) |
| Model support | Any | Any | OpenAI only |
| Day-1 ramp-up | Requires understanding graphs | 30 minutes | 5 minutes |
| Production deployment | Strongest | Medium | Medium |

<a id="s5-4"></a>

### 5.4 Anthropic Agent Skills: a second standards path, beside MCP

In 2026 Anthropic introduced **Agent Skills** — packaging the procedural knowledge of "how an agent completes a class of task" into a reusable unit. This is Anthropic's second attempt at an open standard, after MCP.

#### How Skills, MCP, and Tools differ

| Abstraction | What it solves | Form | Analogy |
|---|---|---|---|
| **Tools** (function calling) | One atomic operation | A function signature + a JSON schema | One shell command |
| **MCP** | Connect external tools and data to an LLM | A server exposes tools / resources / prompts | An OS device driver |
| **Agent Skills** | A **complete procedure** for finishing a class of task | A folder (instructions + templates + scripts + checklist items) | A shell script / playbook |

**The key insight**: MCP connects tools to the agent; Skills teach the agent **how to use** those tools to finish a concrete task. In Anthropic's words: "MCP is plumbing, Skills is procedural knowledge."

#### Standard layout of a skill

A skill is a folder:

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

`SKILL.md` is a **manual for the LLM to read**:

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

While it is carrying out a task, the LLM **reads `SKILL.md` on its own** and follows the steps in it.

#### Integration with Claude Code

Claude Code supports a skills directory natively. Create `.claude/skills/` at the project root:

```
my-project/
├── .claude/
│   └── skills/
│       ├── pr-review/SKILL.md
│       ├── deploy/SKILL.md
│       └── investigate-bug/SKILL.md
```

Claude Code scans it at startup. When the user says "review the pr", it matches the `pr-review` skill and runs it.

#### What Skills mean for the industry

**Skills are the standardized unit of reusable agentic capability** — a team can share them, an organization can govern them, and the community can publish them. Anthropic open-sourced the spec, the same strategy as with MCP: build the ecosystem first, then unify the standard.

For community skills, see [`skillmatic-ai/awesome-agent-skills`](https://github.com/skillmatic-ai/awesome-agent-skills).

<a id="s5-5"></a>

### 5.5 Advanced LangGraph features a production agent must know

The state machine of basic LangGraph was covered in §5.3. Below are five advanced features that production agents actually use:

<a id="s5-5-1"></a>

#### 5.5.1 Checkpointer: pause the agent, resume it, and travel in time

LangGraph has a built-in checkpointer. After each node runs, state is saved automatically:

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

**Time-travel**: you can roll back to any historical checkpoint and fork a new branch from there:

```python
# 列出当前 thread 的所有历史 checkpoint
history = list(app.get_state_history(config))
# 选某个历史点
old_state = history[3]
# 从那里继续（fork 一条新分支）
result = app.invoke(None, config=old_state.config)
```

**Uses**: during debugging, go back to the step before the error; A/B-test two tool-call results; let the user undo an action.

<a id="s5-5-2"></a>

#### 5.5.2 Interrupt: human-in-the-loop as a first-class citizen

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

<a id="s5-5-3"></a>

#### 5.5.3 Subgraph: turn a team into an embeddable subgraph

Split a complex agent into subgraphs. Each team develops, tests, and deploys on its own:

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

<a id="s5-5-4"></a>

#### 5.5.4 Streaming: let the user see the agent "thinking"

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

Render it live on the front end over SSE or WebSocket. The user experience improves 10×.

<a id="s5-5-5"></a>

#### 5.5.5 Time-travel debug, integrated with LangSmith

LangSmith records every checkpoint automatically. In the dashboard, click any step and you can:

- See the snapshot of state at that moment
- See the full prompt and response of the LLM call
- Edit the prompt and replay from there
- Add a good case to the eval set

**This is the core reason LangGraph pulls ahead of CrewAI and AutoGen on production maturity.**

---

<a id="s6"></a>

## 6. Multi-agent collaboration patterns

The 4 + 1 mainstream patterns for putting several agents together:

<a id="s6-1"></a>

### 6.1 Supervisor / Conductor

```
            ┌───────────────┐
            │   Supervisor  │
            └───┬───┬───┬───┘
                │   │   │
        ┌───────▼┐ ┌▼─┐ ┌▼──────┐
        │ Agent A│ │B │ │ Agent C│
        └────────┘ └──┘ └────────┘
```

A central supervisor agent receives the goal → splits it into subtasks → routes them to specialist worker agents → synthesizes the result. **Easy to reason about, easy to trace, and the control flow is clear** — one routing node, and every decision is visible. **The mainstream production architecture.**

<a id="s6-2"></a>

### 6.2 Pipeline

```
Agent A → Agent B → Agent C → Output
```

Agents run in order. The output of one is the input of the next. In essence this is like a traditional ETL workflow. The "agent" abstraction is a bit redundant in this pattern; the benefit is that it integrates smoothly with traditional systems.

<a id="s6-3"></a>

### 6.3 Swarm

```
   Agent A ←──→ Agent B
       ↓ ↘    ↗ ↑
       ↑   ╲╱    ↓
   Agent C ←──→ Agent D
```

Decentralized: agents hand off directly to one another. **No intermediary, fewer calls, faster** — but hard to debug, and easy to fall into a loop of passing the buck. LangGraph makes both supervisor and swarm first-class, so you choose by scenario.

<a id="s6-4"></a>

### 6.4 Hierarchical (multi-level delegation)

```
Top Manager
├── Mid Supervisor 1
│   ├── Worker A
│   └── Worker B
└── Mid Supervisor 2
    ├── Worker C
    └── Worker D
```

Tree-shaped, multi-level delegation: the top handles strategy, the middle handles tactics, and the leaves execute. With shared state management, **conflicts drop by a large margin** (how large varies widely by team and by task; precise figures of the "80%" kind have no credible basis in public sources). It fits complex business, and the latency is high.

<a id="s6-5"></a>

### 6.5 Hybrid (a mix)

In practice it is often a hybrid: one layer of a hierarchy uses a swarm for parallel exploration, and another layer uses a supervisor to converge. Devin's internals, for example, are hierarchical (planner → coder → reviewer, three layers) plus a local swarm (the reviewer spawns several critics).

<a id="s6-6"></a>

### 6.6 How to choose

- **A clear division of labor, and easy debugging** → Supervisor
- **Sequential data processing** → Pipeline
- **Parallel exploration, with tolerance for mess** → Swarm
- **A large, complex organization, with several business lines** → Hierarchical
- **Real production** → Hybrid (a Supervisor as the base, plus the occasional Swarm)

<a id="s6-7"></a>

### 6.7 LangGraph code samples for the five patterns

Each of the five code samples below runs on its own (you need `pip install langgraph langchain-anthropic`) and covers one of the five collaboration patterns.

#### Pattern 1: Supervisor / Conductor

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

#### Pattern 2: Pipeline (a sequential pipeline)

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

#### Pattern 3: Swarm (decentralized handoff)

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

Each agent decides for itself whom to hand off to. There is no central supervisor.

#### Pattern 4: Hierarchical (multi-level delegation)

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

LangGraph's subgraph (`subgraph`) feature lets each layer be tested on its own and deployed on its own.

#### Pattern 5: Hybrid (common in production-grade systems)

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

In practice, the sub-agent systems of Devin, Manus, and Claude Code are all close to this hybrid topology: strategy at the top, a tactical supervisor in the middle, and parallel exploration at the bottom.

---

<a id="s7"></a>

## 7. Agent evaluation benchmarks

<a id="s7-1"></a>

### 7.1 Six core agent benchmarks

| Benchmark | What it evaluates | Task scale | Scoring | Head of the 2026-04 ranking |
|---|---|---|---|---|
| **SWE-bench Verified** | Repairing real GitHub issues | 500 human-verified issues from Python projects | Programmatic (run the tests) | Claude Opus 4.7 ~ **87%** (Anthropic's announced figure; the latest numbers are on the swebench.com leaderboard) |
| **GAIA** (Princeton HAL) | Multi-step reasoning for a general assistant | 466 questions, three difficulty tiers | Programmatic + LLM judge | Claude Sonnet 4.5 = **74.6%**; Anthropic swept the top 6 |
| **TAU-bench** | Tool use + dialogue with a real user | Multiple domains | Simulated multi-turn interaction | — |
| **AgentBench** | 8 environment classes (OS / DB / KG / card games / lateral thinking / household chores / online shopping / web) | Cross-environment | Scored per environment | — |
| **WebArena** (CMU) | Web-navigation agents | 812 tasks / 5 websites + a map | Programmatic | — |
| **OSWorld** | Real operation across operating systems and applications | 369 tasks / Ubuntu + Windows + macOS | Verified by execution | — |

<a id="s7-2"></a>

### 7.2 SWE-bench: the benchmark people cite most

> SWE-bench Verified is the coding-agent signal most trusted in 2026: 500 human-verified issues from real GitHub Python projects. The agent has to read the code, edit files, and make the tests pass.

Claude Opus 4.7 is reported as one of the current leaders on SWE-bench Verified (the latest score is on swebench.com). OpenAI o3 / the Codex series and Google Gemini Code follow. The strongest open-source models currently sit in the 60%+ band.

<a id="s7-3"></a>

### 7.3 A hard warning: every benchmark can be hacked

Berkeley RDI's *How We Broke Top AI Agent Benchmarks* systematically audited 8 mainstream benchmarks (SWE-bench, WebArena, OSWorld, GAIA, Terminal-Bench, FieldWorkArena, CAR-bench, and others). The conclusion: **every one of them can be exploited to a near-perfect score without actually completing the task**. Common hack paths:

- A string-match hole in the evaluation script (emit a magic string and you pass)
- A sandbox that is not isolated tightly enough, so the agent reads the ground truth directly
- Hint fields shipped with the benchmark that leak too much

What this means in practice:

1. **Read the evaluation methodology before you read the score**
2. **Do not trust a public benchmark ranking for your own business.** Re-test on your own ground truth
3. **A composite of several benchmarks** is more reliable than a single benchmark ranking

<a id="s7-4"></a>

### 7.4 Running SWE-bench on your own machine (the full procedure)

If you want to run SWE-bench Verified yourself, and not only read the leaderboard number, the full procedure is below.

<a id="s7-4-1"></a>

#### 7.4.1 Hardware floor

- **120 GB+ free storage** (one Docker image per repository)
- **16 GB+ RAM**
- **8+ CPU cores**
- **x86_64** (some repositories fail to build on ARM)
- No GPU is required (the evaluation itself does not run a model)

<a id="s7-4-2"></a>

#### 7.4.2 Three-stage procedure

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

<a id="s7-4-3"></a>

#### 7.4.3 Full predictions format

```jsonl
{"instance_id": "django__django-11099", "model_name_or_path": "my_agent_v1",
 "model_patch": "diff --git a/django/...\n--- a/django/...\n+++ b/django/...\n@@ -100,3 +100,5 @@\n+    # fix: ...\n"}
{"instance_id": "django__django-11133", "model_name_or_path": "my_agent_v1",
 "model_patch": "..."}
```

Each line is one task. `model_patch` is a unified diff. The harness applies the diff, runs the tests, and compares the result with what was expected.

<a id="s7-4-4"></a>

#### 7.4.4 A wrapper that plugs in your own agent

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

<a id="s7-4-5"></a>

#### 7.4.5 mini-swe-agent: the official baseline

The SWE-bench team's [`mini-swe-agent`](https://github.com/SWE-bench/mini-swe-agent) is a minimal agent harness whose only tool is bash, about 500 lines of Python, **meant as a baseline to compare against**:

```bash
pip install mini-swe-agent
mini-swe-agent run --model claude-sonnet-4-6 --dataset princeton-nlp/SWE-bench_Lite
```

If your agent cannot beat the mini-swe-agent baseline on SWE-bench Lite, suspect the harness implementation before you suspect the model.

<a id="s7-4-6"></a>

#### 7.4.6 Choosing among the three SWE-bench variants

| Variant | Tasks | Estimated cost to finish (Claude Sonnet 4.6) | Use |
|---|---|---|---|
| **SWE-bench Lite** | 300 | ~$30, ~4 hours | Fast iteration / day-to-day |
| **SWE-bench Verified** | 500 | ~$60, ~8 hours | Papers / leaderboard submission |
| **SWE-bench full** | 2,294 | ~$300, ~36 hours | Rarely used |

In practice: **use Lite day to day, and run Verified before a formal release**.

<a id="s7-5"></a>

### 7.5 A custom benchmark: turn your own business into an eval set

A public benchmark does not necessarily fit your business. **The eval you can actually trust is ground truth from your own business.** The minimum viable procedure:

<a id="s7-5-1"></a>

#### 7.5.1 Build a golden set of 100 cases

Collect 100 real business scenarios:

- 50 **success cases**: the user asks X, and the answer should be Y (Y labeled by a domain expert)
- 30 **edge cases**: ambiguous questions, incomplete information, long-tail intent
- 20 **adversarial cases**: prompt injection / jailbreak attempts / out-of-scope requests

Store them with the schema below:

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

<a id="s7-5-2"></a>

#### 7.5.2 Code that runs the eval

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

<a id="s7-5-3"></a>

#### 7.5.3 Wire the eval into CI

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

Every pull request runs the suite. A regression on any metric blocks the merge.

<a id="s7-6"></a>

### 7.6 Beyond the score: seven disciplines of evaluation

A public leaderboard number is only the surface. What a production agent evaluation should actually look at:

1. **The distribution across failure modes** (what share is Loop of Death / hallucination / tool misuse)
2. **The cost distribution** (p50 / p95 / p99 cost per task)
3. **The latency distribution** (the same)
4. **The safety-violation rate** (computed on its own; it must not be folded into quality)
5. **Cross-model consistency** (run the same case set on another model; high agreement is what shows that the eval itself is reliable)
6. **Human-calibration kappa** (have people re-grade the golden set every month, and watch whether the LLM judge drifts)
7. **Correlation with real business KPIs** (what you eventually need is NPS / conversion / support cost; the eval score is only a proxy)

---

<a id="s8"></a>

## 8. Real product cases

<a id="s8-1"></a>

### 8.1 The big three coding agents

> If you are writing code in 2026, you are almost certainly using one of these three: **Claude Code, GitHub Copilot, Cursor**. The most common two-tool stack is *Cursor for daily editing + Claude Code for hard tasks*, or *Copilot in the IDE + Claude Code in the terminal*.

**Claude Code** (Anthropic) — the most representative CLI agent of 2026

- Terminal-native. It runs in the environment the user already has
- A `CLAUDE.md` file stores project-level instructions, conventions, and architecture notes, and persists across sessions (one form of long-term memory)
- Three permission layers: deny > ask > allow. An unrecognized action escalates to the user
- Supports sub-agents, background tasks, hooks, MCP servers, custom skills, and slash commands
- Fits asynchronous workflows and Slack integration
- "The strongest autonomous coding agent, and especially strong on long tasks"

**Cursor**

- A VS Code fork. AI is built into each workflow, rather than attached as an extension
- Three modes: Composer / Tab / Chat
- Agent mode supports long autonomous tasks

**GitHub Copilot Workspace**

- 2026 capability: Copilot Chat plus agent mode, a multi-model picker, and deep GitHub integration (issues / PRs / CI/CD)

<a id="s8-2"></a>

### 8.2 Cloud agent platforms

**Devin** (Cognition)

- A fully autonomous software engineer, running in its own cloud environment (its own browser / terminal / editor)
- Asynchronous: you file the request, it runs for hours, and it comes back with a PR
- The earliest commercialized "AI software engineer." It drew a lot of controversy at one point, and it kept iterating through 2025–2026

**Manus**

- A cloud general-purpose agent platform launched by a Chinese team in 2025
- Similar to Devin, with a wider scope: coding, research, document generation, and more

<a id="s8-3"></a>

### 8.3 Computer-use agents

**Anthropic Claude Computer Use**

- First shipped in 2024-Q4, on Claude 3.5 Sonnet
- The macOS desktop app entered research preview in 2026-03. Windows support is expected in 2026-Q3
- Architecture: a portable "screenshot + mouse and keyboard" tool surface, across VMs, containers, and remote desktops, not bound to one OS
- Safety layer: a user halt command + prompt-injection scanning + vulnerability detection + Constitutional AI alignment

**OpenAI Codex Background Computer Use**

- Released 2026-04-16, as part of "Codex for almost everything"
- macOS-first, with parallel agent sessions
- The agent runs in a separate desktop session, isolated from the engineer's main workflow

**Operator** (OpenAI's earlier computer-use product)

- Launched in early 2025 as a Chrome extension, focused on operating the web
- Partly superseded by Codex Background Computer Use

#### How computer use works (one shared architecture)

Anthropic Computer Use, Operator, OpenAI CUA, and Manus all run the same **perceive–decide–act** loop:

```
┌─────────────────────────────────────────────────────────┐
│  ① 截图（screenshot tool）                              │
│     resolution: 1280x800（Anthropic 推荐）              │
│     格式: PNG → base64 → vision LLM input                │
└─────────────────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│  ② VLM 推理（Vision LLM）                                │
│     - 解析当前界面（按钮 / 输入框 / 文本 / 图标）        │
│     - 决定下一步动作（"点击右上角的『下一步』按钮"）     │
│     - 输出结构化 action：{type, x, y, text, key}         │
└─────────────────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│  ③ 执行（在 sandbox VM / 远程桌面 / Chrome extension）   │
│     - mouse_move(x, y) / click / drag                    │
│     - type_text("...") / key_press("ctrl+c")             │
│     - 100-500ms 后截下一张图                             │
└─────────────────────────────┬───────────────────────────┘
                              │  循环
                              └──────────────────────────▶
```

**The hard engineering problems:**

| Problem | What you see | What the field does about it |
|---|---|---|
| **Pixel coordinates miss** | The LLM says "click (450, 320)" and the real target is at (445, 318). Five pixels, and the wrong button | Anthropic strengthens the grounding loss at training time. At runtime, allow a tolerance and verify with a second screenshot |
| **Dynamic / asynchronous UI** | The page is still loading after the click, so the next screenshot is an intermediate state | Introduce `wait_for_idle`, plus the DOM / accessibility tree as a second signal (pixels alone are not enough) |
| **CAPTCHA / anti-bot** | Almost every CAPTCHA stops the agent | The field has given this up: either HITL, so the user solves it, or switch to an API path |
| **Multiple windows / focus drift** | A notification pops up, or the app switches, and the agent loses context | Capture the full screen and keep a `window_id`. OpenAI Operator isolates a dedicated browser |
| **PII and privacy leakage** | The screenshot sends passwords, mail, and messages that are on screen to the LLM | Mask sensitive regions on the client; filter with local OCR; use a dedicated computer-use VM |
| **Latency and cost** | 1–3 seconds per step, and 1–2K vision tokens per screenshot. Ten minutes easily runs past $5 | Screenshot on an interval, lower the resolution, cache UI-element detection, and prefer the DOM over vision |

**Mainstream reliability numbers (OSWorld, 2026-04):**

- **Claude Sonnet 4.6 + Computer Use**: ~38% task-completion rate
- **OpenAI CUA**: ~32%
- **Human baseline**: 72%

So at the end of 2026, computer use is still **research and assistance**, not a production replacement. Before it sits on a critical product path, there has to be a HITL fallback.

<a id="s8-4"></a>

### 8.4 Open-source options

**Aider**

- Fully open source, git-native, a terminal AI tool
- A strong commit-level workflow, and full bring-your-own-model support
- Fits someone who wants their own fine-tuned model and wants the terminal

**Cline / Continue / Roo Code**

- Open-source VS Code extensions that stand in for Cursor
- Flexible, with less of Cursor's integration

---

<a id="s9"></a>

## 9. Failure modes and real incidents

<a id="s9-1"></a>

### 9.1 Loop of Death

**90% of production agents die here.** The agent keeps retrying the same failed action inside some loop, burning tokens and API quota until the budget alarm fires.

**Real cases:**

- One Claude Code sub-agent consumed **27 M tokens** and ran for **4.6 hours** inside an infinite loop
- One recorded infinite-retry loop burned **$40 in API fees** within a few minutes

#### Five typical shapes of a Loop of Death (ordered by how often they fire)

| Shape | How it triggers | Example | What makes it fatal |
|---|---|---|---|
| **Tool Retry Loop** | The tool keeps returning the same error, and the agent keeps trying the same arguments | `read_file('config.yaml')` → `FileNotFoundError`, and the agent retries 50 times | The LLM has no strong signal that it already tried this; the error message carries no diagnosis |
| **Test-Fail-Fix-Break** | A change at A makes test B pass, and the new change breaks test C. Around it goes | `npm test` is red → edit → a different test goes red → edit again → ... | There is no dashboard of the whole test state. Each step sees only a local slice |
| **Hallucinated Tool / API Loop** | The agent calls a tool or API that does not exist, then nudges the name and tries again | `search_web` → missing → `web_search` → `do_search` → ... | The tool list has no enforced schema, and the error does not return the list of available tools |
| **Plan-Replan Spiral** | After every failure the agent rewrites the whole plan, and the record of the old failure is lost | "Replan" makes the LLM redesign the same failing steps | There is no episodic memory of "X was already tried and did not work" |
| **Approval Wait Deadlock** | The agent waits for HITL approval, and nobody is looking at the UI | `request_human_approval` fired, nobody responded, and the agent sat for hours | There is no approval timeout and no escalation |

#### Anatomy of a single step (the "Anthropic 90% case")

> An internal Anthropic analysis reports that **about 90% of Loops of Death are the cycle "thinking → tool call → the same error → the same thinking → the same tool call."** The LLM has no explicit memory of "I just tried this," so it keeps recommending the same action.

Four defenses. All four are required:

```python
# 1. 硬上限：iterations / timeout / token / cost
MAX_STEPS = 25
MAX_DURATION = 300  # seconds
MAX_TOKENS = 200_000
MAX_COST_USD = 5.0

# 2. 重复检测：相同 tool + 相同参数 ≥ N 次 → 强制 break
def detect_repeat(history, tool_call, n=3):
    same = [h for h in history[-10:] if h.tool == tool_call.tool
            and h.args == tool_call.args and h.error == tool_call.error]
    return len(same) >= n

# 3. 错误信号注入：把"已失败 N 次"用强语言注入下一轮 prompt
if detect_repeat(history, tool_call, n=2):
    inject = f"⚠️ You have called {tool_call.tool}({tool_call.args}) "
             f"and got the same error '{tool_call.error}' {n} times. "
             f"Do NOT repeat. Choose a different approach OR stop and report."
    next_messages.append({"role": "system", "content": inject})

# 4. Escape hatch：明确给 agent "放弃 / 报告" 工具
tools.append({
    "name": "give_up_and_report",
    "description": "Use when stuck. Reports what you tried and asks user.",
    "parameters": {"reason": "string", "attempted": "list"}
})
```

**Engineering tradeoffs:**

- **Hard caps are tiered by task class.** A long refactor and a one-shot question differ by about an order of magnitude in what a reasonable cap is. One global value either cuts a real task off too early or might as well not be there. In production, set a threshold set per task type and per tool risk, and treat "approaching the cap" as an alert rather than a silent truncation.
- **The repeat-detection window N should not be set greedily small.** N=2 has a very high false-positive rate on legitimate polling, pagination, and repeated `read_file`. In practice the fingerprint has to include "the same error," and read-only tools need a looser threshold. Otherwise the escape hatch fires by mistake, and the agent gives up on a task it could have finished.
- **Tiered compute is the highest-ROI piece of this defense.** Leave planning and reflection on a strong model, and push tool-argument filling, formatting, and the retry decision down to a cheap model. Community and vendor cookbooks commonly report token cost **down ~70%**. The loop also burns less money, so the hard cap's remaining budget lasts longer.
- **This defense does not stop a dead loop at the specification level.** When the goal itself is unreachable (code that will not compile, a request the remote API rejects), the agent can change method on every step and still spin among semantically isomorphic attempts. The backstop has to be an upstream verifier or a HITL gate. See [§9.4](#s9-4) and [§10](#s10).

<a id="s9-2"></a>

### 9.2 Hallucination cascade

One invented fact triggers a chain of bad downstream calls.

**A real case:**

- An inventory-automation agent **invented a SKU that does not exist**, then called 4 downstream APIs to price that hallucinated SKU, change its stock, and ship it. One hallucination became a cross-system incident

**Countermeasures:**

- RAG grounding (a write requires a source behind it)
- Chain-of-Verification (after generation, ask "what is the source of each fact above?")
- Low temperature (T=0, or near 0)
- Constrained decoding (choose only from a known list)

<a id="s9-3"></a>

### 9.3 Tool misuse / over-privileged actions

The agent calls a highly destructive tool by mistake.

**A 2026 collection of real incidents** (as reported in the press; **readers should check any specific accusation against a vendor against the primary source linked in the bibliography**):

1. **A large cloud provider's internal coding agent was reported, in 2026, to have operated on a production environment on its own and caused a long outage** (the details are in each vendor's postmortem)
2. **Multiple Hacker News and Reddit threads report that a Google AI coding tool, asked to clear a cache, affected more than was intended** (for the specific vendor and product name, see the [resources at the end of chapter 03](#s14))
3. **The Replit AI agent deleted a production database during a code freeze** (the Replit CEO publicly apologized, 2025-07; covered by SaaStr and TechCrunch)
4. **A second-hand Undercode Testing blog reports an Opus 4.6 agent deleting a production database within a few seconds** (a second-hand retelling; the original incident statement still needs verification)

Every one of them is the same combination failing at once: **"the agent held permissions it should not have held + no deny-first default + no sandbox + no HITL gate."**

**Countermeasures:** see [§10](#s10).

<a id="s9-4"></a>

### 9.4 Memory corruption / specification gap / verification failure

- **Memory corruption:** a wrong fact enters long-term memory and contaminates every later decision
- **Specification gap:** the user's description of the goal is ambiguous, and the agent runs a long way in the wrong direction
- **Verification failure:** the agent's own check is wrong ("I am done" when it is not)

**Silent failures (failures with no alert)** are the worst kind. The agent quietly abandons the task, returns an empty result, or opens a PR that looks finished and is wrong. This class has to be watched continuously in production, with observability plus eval.

<a id="s9-5"></a>

### 9.5 Cost-explosion monitoring

Whatever the failure, it shows up as **money**. A production agent needs:

- A cost-budget cap per user / org / task
- Real-time token and cost alerts (alarm as soon as a 5-minute window crosses the threshold)
- A visual trace of which step burned how much
- A periodic audit: "whose agent burned the most last week"

<a id="s9-6"></a>

### 9.6 The four failure modes Anthropic summarizes, and the fixes that go with them

Anthropic's *Building Effective AI Agents*, plus later engineering posts, collapses production-agent failures into 4 classes. They overlap [§9.1](#s9-1)–[§9.5](#s9-5) above, from a different angle: organized by the fix.

| Failure mode | How it shows up | Root cause | The fix that goes with it |
|---|---|---|---|
| **① Tool misuse** | The wrong tool / the right tool with illegal arguments / a highly destructive tool | The tool description is vague, the parameter schema is loose, and there is no pre-check | (1) A strict JSON schema, plus a second check on the server; (2) `requires_confirmation: true` on highly destructive tools; (3) a PreToolUse hook blocks before execution; (4) the tool description states when **not** to use it |
| **② Hallucinated tool / hallucinated API** | Calling a tool that does not exist at all, or inventing an API | The model guesses the tool list instead of reading it; the error message does not steer back to the right tool | (1) Attach an `available_tools` list when returning an error; (2) constrained decoding, so the tool name is chosen from an allowlist; (3) deny-by-default: an unregistered tool is rejected immediately, with no retry |
| **③ Context explosion** | Tool results, history, and intermediate thinking blow the context up | No windowing; tool results are not trimmed; old steps are not summarized | (1) Graded trimming of tool results (see [context-engineering.en.md §2.6.3](./context-engineering.en.md#s2-6-3)); (2) periodic compaction of old steps; (3) sub-agent isolation, so a long-running task uses a child context window |
| **④ Reward Hacking / Goal Misgen** | The agent meets the goal in form and violates the intent (delete tests so CI passes, or write a mock and pretend it is implemented) | The verifier signal is incomplete or gameable; there is no structured feedback | (1) The verifier has to be multi-signal (test pass + diff size + lint + behavior diff); (2) a second look with the Evaluator-Optimizer pattern; (3) HITL spot checks + a long-running audit |

**What Anthropic's engineering experience adds up to** (from production):

1. **≤ 20 tools is the sweet spot.** Past that, the rate at which the LLM picks the wrong tool rises sharply. Above 20, use a routing pattern and split the work into sub-agents
2. **The error message is a prompt.** Write a tool error as an instruction to the LLM ("this failed because X. To fix, try Y"). Do not return a raw traceback
3. **Adding an evaluator** has a higher ROI than making the agent cleverer. The evaluator can be a cheaper model, and it can catch 80% of silent failures
4. **Never skip the sandbox.** Even when the agent "looks well behaved," a shell means a sandbox. One hallucination is one incident

---

<a id="s10"></a>

## 10. Security and the permission model

<a id="s10-1"></a>

### 10.1 A deny-first permission architecture (the Claude Code pattern)

> Claude Code's default security posture is **deny-first** plus **human escalation**: deny rules override ask rules, ask rules override allow rules, and an unrecognized action escalates to the user rather than being let through quietly.

The concrete "independent layers of guardrails":

1. **Permission rules** (the user's configured allow/deny lists)
2. **PreToolUse hooks** (can intercept before a tool call)
3. **Auto-mode classifier** (an extra classifier when auto mode is on)
4. **Shell sandboxing** (an optional layer)

**Any one layer can block an action.** That is real defense in depth.

<a id="s10-2"></a>

### 10.2 HITL vs HOTL

| Mode | Meaning | Where it fits |
|---|---|---|
| **HITL (human in the loop)** | A human must confirm before the critical step | High risk (medical, finance, deleting data) |
| **HOTL (human on the loop)** | A human is watching and does not block the run; they step in at the critical moment | Medium risk (coding, content generation) |
| **Fully autonomous** | Report only when the work is finished | Low risk (data lookup, reports) |

<a id="s10-3"></a>

### 10.3 Approval fatigue: HITL's invisible ceiling

> Anthropic's auto-mode data: users click approve on ~93% of permission prompts, without distinction.

So **approval by itself has been made behaviorally unreliable.** People habitually click yes. The conclusion:

- Safety cannot rest entirely on "the user will read every dialog"
- There have to be **technically independent guardrails** (rules + hooks + sandbox + classifier)
- High-risk actions should use **a mandatory cooldown, layered confirmation, or two-person confirmation**

<a id="s10-4"></a>

### 10.4 Sandboxes: a full set of production configs

<a id="s10-4-1"></a>

#### 10.4.1 Five isolation tiers, compared

| Option | Isolation strength | Startup latency | Memory overhead | Compatibility | Fits |
|---|---|---|---|---|---|
| **MicroVM** (Firecracker / AWS Fargate / Kata) | **Strongest** (its own Linux kernel + KVM hardware virtualization) | ~125 ms | < 5 MiB / VM | Full Linux | High-security production (running a stranger's code) |
| **gVisor** | **Strong** (the user-space Sentry kernel + syscall interception) | ~50 ms | ~15 MiB | Most syscalls | The balance, plus GPU workloads |
| **Hardened container** | Medium (a shared host kernel + several restrictions) | < 10 ms | Negligible | Full | Code you wrote yourself |
| **Plain Docker** | **Weak** (namespaces + cgroups only) | ~10 ms | Negligible | Full | **Must not run untrusted LLM-generated code** |
| **Process-level** (seccomp / chroot) | Very weak | < 1 ms | 0 | Full | The last-resort backstop |

**The industry consensus** (from e2b, Daytona, Modal, Vercel Sandbox, and the other mainstream options): **a container alone is not enough. Move up to a microVM or to gVisor.** Nearly 30 years of Linux kernel attack surface is a sieve against LLM-generated code. One namespace-escape CVE jumps from the sandbox to the host.

<a id="s10-4-2"></a>

#### 10.4.2 Firecracker (the microVM from AWS)

**Architecture:** each sandbox is its own Linux kernel plus a full init process, running under KVM hardware virtualization. To get host root, an attacker first has to **break through the kernel, the KVM hypervisor, and the Firecracker VMM** — three layers.

**A minimal boot config:**

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

**What it takes to run this in production:**

- Use [`firecracker-containerd`](https://github.com/firecracker-microvm/firecracker-containerd) to put an OCI interface on it, so agent code looks like it is running Docker
- Snapshot / restore: boot one base VM through system init, snapshot it, and start every later sandbox from that snapshot (**cold start down to ~125 ms, and a warm start under 50 ms**)
- KVM has to be on (bare metal, or a cloud host with nested virtualization)

**E2B, Daytona, and Modal are all Firecracker wrappers inside**, which saves you from operating the cluster yourself:

```python
# E2B（最常用 LLM agent sandbox）
from e2b_code_interpreter import Sandbox
with Sandbox.create(timeout=30) as sbx:        # ★ 必须设 timeout
    result = sbx.run_code("print(2+2)")
```

<a id="s10-4-3"></a>

#### 10.4.3 gVisor (Google's user-space kernel)

**Architecture:** a user-space process, `Sentry`, sits between the application and the host kernel, **intercepts every syscall**, and reimplements it in user space. The application looks like it is running on Linux. It can only call the subset Sentry implements.

**Compared with Firecracker:**

- ✅ Faster startup (~50 ms vs 125 ms)
- ✅ Lower memory overhead
- ✅ Better GPU passthrough (gVisor supports NVIDIA driver passthrough)
- ❌ Incomplete syscall compatibility (an application that needs an unusual syscall may not run)
- ❌ Slightly slower (one extra layer on every syscall)

**A minimal config** (OCI runtime):

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

**Kubernetes integration** (RuntimeClass):

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

<a id="s10-4-4"></a>

#### 10.4.4 A hardened container (when Docker is all you have)

**When this applies:** you are running **code you wrote** (trusted), and you only need to stop accidental damage. **Do not run untrusted LLM-generated code in a hardened container.** Community consensus has already rejected that.

The minimum hardening config (every item is required):

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

**Rootless Docker / Podman** also runs the Docker daemon inside a user namespace, which takes one more layer off the attack surface.

**A custom seccomp allowlist** (deny every syscall except those explicitly allowed):

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

<a id="s10-4-5"></a>

#### 10.4.5 How to choose, and how deep the defense goes

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

**Defense in depth: 5 layers** (if one gives way, another is still there):

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

<a id="s10-4-6"></a>

#### 10.4.6 Mainstream commercial and open-source choices

| Service | Underneath | How you attach | Price |
|---|---|---|---|
| **E2B** | Firecracker | Python/TS SDK, the easiest to attach | Free, plus per vCPU-minute |
| **Modal** | gVisor + Firecracker | A Python decorator | Billed per GPU-second |
| **Daytona** | Firecracker | Dev-container style | Open source + cloud |
| **Vercel Sandbox** | Firecracker | Native to the Next.js ecosystem | Monthly |
| **AWS Fargate / Lambda** | Firecracker | AWS-native | Usage-based |
| **Together Code Sandbox** | gVisor | API | Per call |
| **CodeSandbox** | Firecracker | A dev IDE | Free + a subscription |

Practical experience: **start a new project on E2B or Modal**, and skip the 4–6 weeks of operating your own Firecracker cluster. Revisit self-hosting once you are at scale.

<a id="s10-5"></a>

### 10.5 Where the industry is heading

> **The industry is moving toward selective autonomy, not fully unsupervised autonomy.**

What that means: the same agent should have a different autonomy level on different actions. Reading a README can be fully autonomous. Running tests can be auto-approved. Editing a file needs review. Deleting a file needs a second human confirmation. Operating a production database stays deny-first.

<a id="s10-6"></a>

### 10.6 Tool-design practice (the hidden factor in agent accuracy)

How well the tools are designed **directly sets agent accuracy.** Plenty of teams grind on the prompt and the framework and get nowhere; rewriting how the tools are described lifts accuracy by 20%+.

#### Five hard rules

**1. Name it "verb + noun + object"**

```
❌ search() / get_data() / process()
✅ search_github_issues(repo, query)
   list_pending_orders_by_user(user_id)
   send_slack_message_to_channel(channel, text)
```

When the LLM picks a tool, it mostly looks at the name. **A vague name means low recall.**

**2. Write the description as "when to use it / when not to use it"**

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

**3. The parameter schema is strict, and it has examples**

Use Pydantic / JSON Schema, with strong types:

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

The `examples` field materially helps the LLM choose arguments and format them.

**4. An error return should tell the model what to do next**

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

When the LLM sees `suggestions`, it tries the next step on its own, which avoids a Loop of Death.

**5. A cap on tool count: do not give a single agent more than 20 tools**

The probability that an LLM picks the wrong tool rises sharply once there are more than 20 tools (research shows accuracy dropping 20%+ at 40+ tools). **The production pattern:**

- **Group the tools, then route.** A router agent first picks a category ("order / billing / shipping"), then hands off to that category's sub-agent. Only the sub-agent sees the specific 5–10 tools
- **Dynamic tool loading.** From the current conversation, choose which tools to expose to the LLM. That saves prompt tokens and raises accuracy

#### Tool-design anti-patterns (real production incidents)

| Anti-pattern | What actually happened | The fix |
|---|---|---|
| One `query_database(sql)` handed to the LLM | The LLM writes dangerous SQL, and a table gets dropped | Split it into dedicated tools such as `lookup_user_by_email` and `get_order_status` |
| Tool names that are synonyms (`fetch_user` + `get_user`) | The LLM picks at random, and the results are inconsistent | Merge them |
| Long results are not truncated | 10K rows come back, and the context blows up | Cap on the server, and add pagination |
| No timeout | The tool hangs, so the agent hangs | Force a 30s timeout |
| Side effects are undeclared | The model calls it as if it were read-only, and something gets modified | The description states "this modifies X", and add `require_confirmation` |
| A long JSON result nested 7 levels deep | The LLM cannot parse it | Flatten it, and add a summary field |

#### Tool-set design checklist

- [ ] You can tell what each tool does within 5 seconds, from the name
- [ ] The description has all 5 blocks: when to use / when not to use / parameters / return / errors
- [ ] Every parameter has a schema and examples
- [ ] Error returns carry suggestions
- [ ] Each tool has been run on your own business eval set
- [ ] The total tool count is ≤ 20; above that, split with a router
- [ ] Writes and high-risk operations are explicitly marked, and HITL is the default

---

<a id="s11"></a>

## 11. Where the agent sits in the four engineering layers

Put the agent back into the picture of the four engineering layers:

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

The four layers stack. Each one is still required:

- Without a good prompt (layer 1), the agent's thinking is muddled
- With context left unmanaged (layer 2), the agent has blind spots
- Without a solid agent loop and framework (layer 3), the agent can be right on each step and still drift overall
- Without a harness (layer 4), the agent is not controllable in production

---

<a id="s12"></a>

## 12. What to do when you ship

<a id="s12-1"></a>

### 12.1 A three-week path into agent engineering

- **Week 1:** understand ReAct, get the official LangGraph quickstart running, and attach 1–2 MCP servers
- **Week 2:** build a task agent for one concrete business (a weekly-report generator, or a data-query assistant, is a good pick), and add a max-iteration cap plus a cost cap
- **Week 3:** attach Langfuse observability, run a few RAGAS / DeepEval cases, and put HITL on one high-risk action

<a id="s12-2"></a>

### 12.2 A six-item checkup for an agent you already run

1. ✅ Are there caps on max iterations, max tokens, max cost, and max time?
2. ✅ Does every write (edit a file, delete data, send email) default to deny-first or to HITL?
3. ✅ Is there a sandbox? Least privilege? An egress allowlist?
4. ✅ Does a trace (every tool call and every LLM call) land on an observability platform?
5. ✅ Does an automatic eval run regression? Is there a golden set?
6. ✅ Has there been an incident? Was there a postmortem, and were new rules added? (This is the core practice of harness engineering.)

<a id="s12-3"></a>

### 12.3 Three things not to do

- ❌ Give an agent direct sudo, root, or production-database write access in a production environment
- ❌ Treat "the user will confirm" as the last line of safety (93% click yes regardless)
- ❌ Pick a model from the public SWE-bench ranking. Re-test on your own business

---

<a id="bibliography"></a>

## Bibliography

### Agent paradigms and papers

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
- Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*.
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*.

### Agent architecture and frameworks (2026)

- Lindy. *AI Agent Architecture: A Complete Guide for 2026*. <https://www.lindy.ai/blog/ai-agent-architecture>
- Redis. *AI Agent Architecture: Build Systems That Work in 2026*. <https://redis.io/blog/ai-agent-architecture/>
- arXiv. *AI Agent Systems: Architectures, Applications, and Evaluation* (2026). <https://arxiv.org/html/2601.01743v1>
- ATNO. *10 AI Agent Frameworks You Should Know in 2026*. Medium, 2026-04.
- Anubhav. *LangGraph vs CrewAI vs AutoGen: Which Agent Framework Should You Actually Use in 2026*. Data Science Collective, 2026-03.
- digitalapplied. *OpenAI Agents SDK vs LangGraph vs CrewAI: 2026 Matrix*.

### Multi-agent collaboration

- gurusup. *Agent Orchestration Patterns: Swarm vs Mesh vs Hierarchical*.
- agixtech. *Conductor vs. Swarm: Multi-Agent AI Architecture Guide 2026*.
- LangGraph dev.to. *Multi-Agent Orchestration in LangGraph: Supervisor vs Swarm*.

### Agent benchmarks

- Berkeley RDI. *How We Broke Top AI Agent Benchmarks*. <https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/>
- SWE-bench Leaderboards. <https://www.swebench.com/>
- Steel.dev. *AI Agent Benchmark Results Index*. <https://leaderboard.steel.dev/results>
- Spheron. *AI Agent Benchmarking Infrastructure on GPU Cloud (2026 Guide)*.
- philschmid/ai-agent-benchmark-compendium. <https://github.com/philschmid/ai-agent-benchmark-compendium>
- MarkTechPost. *Top 7 Benchmarks That Actually Matter for Agentic Reasoning in LLMs* (2026-04-26).

### Real agent products

- artificialanalysis.ai. *Coding Agents Comparison: Cursor, Claude Code, GitHub Copilot, and more*.
- SitePoint. *Claude Code vs Cursor vs Copilot: The 2026 Developer Comparison*.
- Tech Insider. *Anthropic's Claude Computer Use Agent*.
- WorkOS. *Anthropic's Computer Use versus OpenAI's Computer Using Agent (CUA)*.
- digitalapplied. *Computer Use Agents 2026: Claude vs OpenAI vs Gemini*.
- arXiv. *Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems* (2026). <https://arxiv.org/html/2604.14228v1>

### Agent failure modes and security

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

<a id="s13"></a>

## 13. A complete end-to-end agent: customer support with RAG, memory, and guardrails

The concepts above, wired together. Below is a production-grade minimal skeleton: LangGraph + tools + retrieval + memory + a max-iteration cap + a cost cap + observability.

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

This 100-line skeleton already has the core pieces of a production agent:

| Piece | Where it is implemented |
|---|---|
| The LLM + tool loop (ReAct) | `agent_node` + `tool_node` |
| Long-term memory | `memory.search` / `memory.add` (Mem0) |
| A RAG knowledge base | the `kb_search` tool |
| An input guardrail | `input_guardrail` |
| Four Loop of Death gates | `should_continue` (max iterations + cost cap + timeout + a completion check) |
| HITL (critical tickets) | a forced raise inside the `create_ticket` tool |
| Checkpoint / pause and resume | `MemorySaver` |
| Observability | `LangfuseCallback` |
| Cost tracking | a running total in `total_cost_usd` |

What you still add before this is production: automated eval (DeepEval / RAGAS), an A/B test, a staged rollout, a PagerDuty alert hookup, compliance logs, and an output guardrail for sensitive terms.

---

<a id="s14"></a>

## 14. A systematic GitHub reading list (the agent track)

| Resource | Kind | How to use it |
|---|---|---|
| [`microsoft/ai-agents-for-beginners`](https://github.com/microsoft/ai-agents-for-beginners) | A 12-lesson introduction | A complete start for newcomers. Each lesson has runnable code samples, paired with the Microsoft Agent Framework |
| [`microsoft/agent-framework`](https://github.com/microsoft/agent-framework) | The official framework | The successor to AutoGen. Python + .NET, for enterprise agent orchestration |
| [`datawhalechina/hello-agents`](https://github.com/datawhalechina/hello-agents) | A systematic Chinese tutorial (42K ⭐) | From principles, to implementation, to agentic RL. The most systematic treatment in the Chinese-language community |
| [`anthropic/Building-Effective-Agents`](https://www.anthropic.com/research/building-effective-agents) | Anthropic's official essay | The original source of the six patterns in [§3.4](#s3-4). **Required reading** |
| [`jcran/effective-agents-langchain`](https://github.com/jcran/effective-agents-langchain) | A LangChain implementation | The LangChain code version of Anthropic's six patterns |
| [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) | Official LangGraph | The mainstream production agent framework. The documentation is very thorough |
| [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | Official CrewAI | A role DSL. The fastest to get running |
| [`openai/openai-agents-python`](https://github.com/openai/openai-agents-python) | The OpenAI Agents SDK | Official OpenAI. The handoff philosophy |
| [`huggingface/smolagents`](https://github.com/huggingface/smolagents) | Smolagents | A few hundred lines of core code. The most minimal implementation |
| [`mastra-ai/mastra`](https://github.com/mastra-ai/mastra) | Mastra | TypeScript-native. The first choice for a JS/TS project |
| [`philschmid/ai-agent-benchmark-compendium`](https://github.com/philschmid/ai-agent-benchmark-compendium) | A compendium of 50+ benchmarks | When you are looking for a test of one specific capability |
| [`vectara/awesome-agent-failures`](https://github.com/vectara/awesome-agent-failures) | A collection of failure modes | The negative examples to read before you write a production agent |

**Papers to read** (in time order):

- 2022 — *ReAct* (Yao et al.)
- 2023 — *Self-Refine* (NeurIPS), *Reflexion* (Shinn et al.), *MemGPT* (Packer et al.)
- 2024 — *Building Effective AI Agents* (Anthropic), *Self-RAG* (Asai et al.)
- 2025 — *Mem0* (arxiv 2504.19413)
- 2026 — *Dive into Claude Code* (arxiv 2604.14228)

---

<a id="cross-references"></a>

## Cross-references

- For ReAct prompt templates and chain-of-thought, see [prompt-engineering.en.md](./prompt-engineering.en.md#s1-4) §1.4
- For the engineering detail of RAG and memory systems, see [context-engineering.en.md](./context-engineering.en.md#s2-4) §2.4 / [§2.5](./context-engineering.en.md#s2-5)
- For tool integration, sandboxes, and guardrails (the direct implementation of the harness strong constraints), see [harness-engineering.en.md](./harness-engineering.en.md#s3-3-1) §3.3.1–§3.3.3. For observability and evaluation (related supporting infrastructure, and not the harness itself), see [§3.9.3–§3.9.4](./harness-engineering.en.md#s3-9-3)
