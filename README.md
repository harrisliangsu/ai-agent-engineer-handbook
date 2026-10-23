# AI Agent Engineer Handbook · AI Agent 工程师手册

中文 AI Agent / LLM 应用工程师的一站式手册：先建立 **Prompt → Context → Agent → Harness** 四层工程坐标，再落到真实 JD 与面试题库。

> **English:** A Chinese-primary handbook for engineers who design, ship, or interview for production LLM agent systems — four engineering layers, real job descriptions, and a question bank. [Short English guide](./README.en.md)

[![Stars](https://img.shields.io/github/stars/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/stargazers)
[![Forks](https://img.shields.io/github/forks/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/forks)
[![License: MIT](https://img.shields.io/github/license/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](./LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/harrisliangsu/ai-agent-engineer-handbook?style=flat-square)](https://github.com/harrisliangsu/ai-agent-engineer-handbook/commits/main)
[![Read online](https://img.shields.io/badge/read-GitHub%20Pages-b85c00?style=flat-square)](https://harrisliangsu.github.io/ai-agent-engineer-handbook/)

**适合谁**：在 AI Agent / LLM 应用方向求职、转型或带队的工程师；招聘方校准岗位；面试官按层出题。

**怎么读**：在线门户、仓库里的 Markdown，或本地 HTML。步骤见 [如何阅读](#如何阅读)。

**最近更新**：正文最近实质性更新：2026-05-31（此后仅 2026-09-21 增补一处网关参考） · 仓库元数据/门户：2026-10-23 · **作者**：[harrisliangsu](https://github.com/harrisliangsu) · **License**：[MIT](./LICENSE)

**支持 / Sponsor**：[爱发电](https://afdian.com/a/sharkchiligaga)

关键词：AI Agent、LLM、RAG、prompt engineering、context engineering、harness engineering、MCP、面试、handbook、中文。

GitHub Topics（仓库已设置，此处只作索引）：`agent` · `ai-agent` · `anthropic` · `awesome` · `career` · `chinese` · `context-engineering` · `handbook` · `interview` · `interview-prep` · `llm` · `mcp` · `openai` · `prompt-engineering` · `rag`

## 目录

- [在线阅读](#在线阅读)
- [仓库结构](#仓库结构)
- [第一部分：四层工程](#第一部分llm-应用工程的四层演进知识体系)
- [第二部分：求职实战](#第二部分求职实战jd--面试题)
- [推荐学习路径](#推荐学习路径)
- [如何阅读](#如何阅读)
- [译文与知识图谱](#译文与知识图谱)
- [工作流声明](#工作流声明)
- [社区](#社区)
- [License](#license)

---

## 在线阅读

不用 clone，用浏览器打开手册站点：

**https://harrisliangsu.github.io/ai-agent-engineer-handbook/**

门户文件在仓库根目录（`index.html`）。在 GitHub **Settings → Pages** 里选择分支 `main`、目录 `/ (root)` 后，这个地址就会生效。`.nojekyll` 让现有 HTML 原样发布，不经 Jekyll。

| 想先看什么 | 网页 |
|---|---|
| 门户（四层 + 面试 + 译文入口） | [首页](https://harrisliangsu.github.io/ai-agent-engineer-handbook/) |
| 提示词工程 | [prompt-engineering.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/prompt-engineering.html) |
| 上下文工程 / RAG | [context-engineering.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/context-engineering.html) |
| Agent 工程 | [agent-engineering.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/agent-engineering.html) |
| 驾驭工程 | [harness-engineering.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/engineering-foundations/html/harness-engineering.html) |
| JD 拆解 | [jd-requirements.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/interview-prep/html/jd-requirements.html) |
| 面试题 | [interview-questions.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/interview-prep/html/interview-questions.html) |
| 全景知识图谱（复习地图） | [knowledge-map.html](https://harrisliangsu.github.io/ai-agent-engineer-handbook/knowledge-map.html) |

首页是站点门户，用来进入各章。全景知识图谱是另一页复习地图，把四层、题库、译文和术语串成一张图，不是门户的替代页。

站点从仓库**根目录**发布。章节 HTML 仍在原来的 `html/` 目录，本地双击或 `python3 -m http.server` 的读法不变。根目录的 `.nojekyll` 让 Pages 按原文件提供这些手写 HTML，而不再走 Jekyll。

---

## 仓库结构

```
ai-agent-engineer-handbook/
├── engineering-foundations/   ← 知识：LLM 应用工程的四层演进
│   ├── prompt-engineering.md     # 提示词工程
│   ├── context-engineering.md    # 上下文工程
│   ├── agent-engineering.md      # Agent 工程
│   ├── harness-engineering.md    # 驾驭工程
│   ├── overview.md               # 综述（v1 合并版，历史保留）
│   ├── README.md                 # 系列阅读指南
│   ├── sources.jsonl             # 全部一手来源清单
│   └── html/                     # 渲染后的网页版（含左侧 TOC + 章节切换）
│
├── interview-prep/            ← 实战：求职 JD + 面试题
│   ├── jd-requirements.md        # AI Agent 岗位 JD 要求研究（中外大厂真实 JD 拆解）
│   ├── interview-questions.md    # 面试题与答案（A–K）
│   └── html/                     # 渲染后的网页版
│
├── translations/              ← 经典文章中译（非官方，以原文为准）
│   ├── building-effective-agents.md
│   ├── multi-agent-research-system.md
│   ├── effective-harnesses-for-long-running-agents.md
│   ├── openai-harness-engineering.md
│   └── html/
│
├── knowledge-map.md           ← 全景复习地图（另有 knowledge-map.html）
├── index.html                 ← GitHub Pages 门户
└── README.en.md               ← 英文短导览
```

---

## 第一部分：LLM 应用工程的四层演进（知识体系）

按"prompt → context → agent → harness"四层抽象组织，每层一篇深度长文，配综述。

| 章节 | 内容要点 | 字数 |
|---|---|---|
| **01 提示词工程** | Transformer / 训练栈 / 解码 / CoT / ReAct / 自动化 prompt 优化 / 评估 / OWASP Top 10 | ~64 KB |
| **02 上下文工程** | RAG 全栈 / RULER 长上下文真相 / 记忆系统（MemGPT / mem0 / Letta / Zep）/ MCP 协议 | ~81 KB |
| **03 Agent 工程** | Agent 范式演进 / Anthropic 6 大 patterns / 10 框架对比 / 5 协作模式 / 真实事故合集 / 安全权限 | ~90 KB |
| **04 驾驭工程** | Hashimoto harness 工程命名 / 两条法则（强约束 + 自愈循环）+ 四大支柱 / 成熟度模型 L0-L4 / 熵管理 | ~134 KB |
| **overview** | v1 合并版（保留作历史参考） | ~73 KB |

**入口**：从 [`engineering-foundations/README.md`](./engineering-foundations/README.md) 开始；HTML 版本在 [`engineering-foundations/html/`](./engineering-foundations/html/) 下打开 `prompt-engineering.html` 即可（左侧浮动 TOC + 顶部章节切换）。

### 系列特点

- **一手来源密度高**：每个论断尽量带 ≥2 个独立来源，bibliography 50+ 条核心参考
- **承认局限**：04 章公开订正过"七大组件"这种合成框架命名，改用业界有出处的"两条法则 + 四大支柱"
- **驾驭工程是重点**：完整复刻 Hashimoto / OpenAI / Anthropic / Augment Code 等一手原文 verbatim 引用

---

## 第二部分：求职实战（JD + 面试题）

### 02-1 AI Agent 岗位 JD 要求研究（[`jd-requirements.md`](./interview-prep/jd-requirements.md)）

- **中国市场**：字节豆包 / 阿里通义 / 腾讯混元 / 月之暗面 Kimi / 智谱 GLM / DeepSeek / MiniMax / 阶跃星辰 / 蚂蚁 AI Force ……14 条带原文 verbatim 的真实 JD 摘录
- **北美 / 欧洲市场**：Anthropic / OpenAI / Google DeepMind / Glean / Vercel / Replit / Sierra / Cognition (Devin) / Cursor / Mistral / Harvey / Perplexity ……8 条完整 JD + 薪资带
- **高频考察点 Top 20** + **岗位变种 7+6 类区分** + **海外/中国对比观察**
- **薪资数据**：从应届 SSP 到资深架构师，按地点和级别分级
- **来源**：BOSS 直聘 / 拉勾 / 51job / 智联 / 牛客 / V2EX / 猎聘 / Greenhouse / Ashby / Glassdoor / Levels.fyi / Hacker News / LangChain ecosystem analysis 等 45+ 来源

### 02-2 面试题与答案（[`interview-questions.md`](./interview-prep/interview-questions.md)）

**92 道题，11 大分类（A–K）**。每题：⭐ 难度分级 + 200–500 字技术答案 + 一手论文或官方文档 URL。下表难度按题目标题里的星级统计（A11 与 K 章是后加入的，总题数已不是早期的 86 / 10 类）。

| 分类 | 题数 | 难度分布 | 重点 |
|---|---|---|---|
| **A. LLM 基础** | 11 | 2⭐ 7⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / 长上下文 / 推理优化 / Prefill-Decode |
| **B. 训练 / 微调** | 10 | 2⭐ 6⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 涌现 |
| **C. 提示词工程** | 9 | 2⭐ 6⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / 自动 prompt 优化 |
| **D. RAG** | 10 | 1⭐ 6⭐⭐ 3⭐⭐⭐ | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS |
| **E. Agent** | 12 | 0⭐ 3⭐⭐ 9⭐⭐⭐ | **核心**：6 patterns / 多 agent / 失败模式 / harness |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / mem0 / progress 文件 / memory corruption |
| **G. 系统设计** | 8 | 0⭐ 2⭐⭐ 6⭐⭐⭐ | 客服 / coding / 搜索 / LLM 网关 / 高并发 / 成本 / 沙箱 |
| **H. 评估** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | judge bias / pairwise / benchmark hack / 漂移 |
| **I. 安全 / 护栏** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | OWASP / Prompt Injection / NeMo Guardrails / EU AI Act |
| **J. 行为题 / 实战** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto harness 闭环 / 0-1 落地路线 |
| **K. Agent 评测** | 5 | 0⭐ 2⭐⭐ 3⭐⭐⭐ | benchmark 全景 / trajectory / tool-use / 上线 SLO / 业务评测 |

---

## 推荐学习路径

| 你是谁 | 推荐顺序 |
|---|---|
| **应届生 / 在校生**（求职准备） | `interview-prep/jd-requirements` 看市场 → `interview-prep/interview-questions` 分类 A/B/C/D 重点刷 → 反向去 `engineering-foundations/` 对应章节深化 |
| **1-3 年中级工程师**（转 AI 方向） | `engineering-foundations/context-engineering` 起读 → `agent-engineering` → `interview-prep/interview-questions` D/E/F 全刷 → G 选 1-2 题白板练 |
| **3-5 年资深 / Staff** | `engineering-foundations/harness-engineering` 通读 → `interview-prep/interview-questions` E/G/H/I/J/K 重点（Staff 的差异在系统设计、行为题和评测闭环）|
| **架构师 / 技术决策者** | `engineering-foundations/harness-engineering` 的"两条法则 + 四大支柱"+ 成熟度模型 L0-L4 + `interview-prep/interview-questions` G 系统设计全部 |
| **招聘方 / 面试官** | `interview-prep/jd-requirements` 校准岗位标准 → `interview-prep/interview-questions` 按分类难度出题 |

---

## 如何阅读

**在线**：打开 [手册站点](https://harrisliangsu.github.io/ai-agent-engineer-handbook/)。各章带左侧浮动 TOC、章节切换和滚动高亮。

**Markdown**（GitHub 上直接看，或用 VS Code / Obsidian / Typora 打开）

**本地 HTML**：

```bash
# 仓库根目录：门户 + 全部章节
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000

# 只看某一组章节时也可以进对应 html 目录
cd engineering-foundations/html && python3 -m http.server 8000
cd interview-prep/html && python3 -m http.server 8000
```

也可以直接双击 `engineering-foundations/html/prompt-engineering.html`。

---

## 译文与知识图谱

四篇经典文章的非官方中译在 [`translations/`](./translations/)，技术名词以括注英文为准，解释以原文为准：

| 中译 | 原文 |
|---|---|
| [构建高效的 Agent](./translations/building-effective-agents.md) | [Anthropic · Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) |
| [多 Agent 研究系统](./translations/multi-agent-research-system.md) | [Anthropic · How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) |
| [长时程 Agent 的 Harness](./translations/effective-harnesses-for-long-running-agents.md) | [Anthropic · Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) |
| [Harness 工程](./translations/openai-harness-engineering.md) | [OpenAI · Harness engineering](https://openai.com/index/harness-engineering/) |

网页版在 [`translations/html/`](./translations/html/)。全景复习地图：[`knowledge-map.md`](./knowledge-map.md) / [`knowledge-map.html`](./knowledge-map.html)。地图用来把体系串成一张图；站点入口仍是根目录门户 [`index.html`](./index.html)。

---

## 工作流声明

本仓库内容由作者使用 [Claude Code](https://claude.com/claude-code) 与自用 deep-research 工作流整理，结合一手论文、厂商官方文档和真实招聘 JD 交叉核实，**带局限性声明的诚实研究**。deep-research 是作者本地的调研技能，没有对应的公开 GitHub 仓库；当时的运行记录在 [`engineering-foundations/run_manifest.json`](./engineering-foundations/run_manifest.json)（历史快照，日常改动不改这份文件）。

- 引用一手来源处都给了 URL，便于读者深挖
- 数字 / 价格 / 模型版本可能随时间变化，请按发布日期查最新
- 部分 JD 因招聘网站登录限制或动态渲染只能拿到摘要，已在 Methodology Appendix 标注证据等级

---

## 社区

怎么参与，见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

| 你想做的事 | 去哪里 |
|---|---|
| 指出具体错误、断链，或提交一处改动 | [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues) / [Pull request](https://github.com/harrisliangsu/ai-agent-engineer-handbook/pulls) |
| 问怎么读、交流求职进度、还没落到文件上的想法 | [Discussions](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| 提议新的题型或章节 | Discussions 的 **Ideas** |
| 用法和阅读问题 | [GitHub Discussions（Q&A）](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| 随便聊聊 | Discussions 的 **General** |

欢迎帖：[自我介绍 / 求职进度 / 纠错建议](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions/4)。用法和阅读上的提问，请到 [GitHub Discussions（Q&A）](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions)。

欢迎纠错：事实、断链或过时表述，请开 [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues/new/choose)，方便对到具体段落。

---

## License

[MIT](./LICENSE)。文章引用的论文与厂商官方文档归原作者所有，本仓库只做整理与中文转述。译文是非官方学习稿。
