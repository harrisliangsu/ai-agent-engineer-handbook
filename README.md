# AI Agent Engineer Handbook · AI Agent 工程师面试与学习手册

> 给中文 AI Agent / LLM 应用工程师的**一站式知识库** —— 从底层原理（Prompt → Context → Agent → Harness 四层工程）到求职实战（真实大厂 JD 拆解 + 86 道高质量面试题与答案）。

**整理日期**：2026-05-02
**作者**：[harrisliangsu](https://github.com/harrisliangsu)
**License**：[MIT](./LICENSE)
**适合谁**：在 AI Agent / LLM 应用方向求职、转型、深化的工程师；招聘方做岗位标准；面试官出题参考

---

## 仓库结构

```
ai-agent-engineer-handbook/
├── engineering-foundations/   ← 知识：LLM 应用工程的四层演进
│   ├── prompt-engineering.md     # 提示词工程
│   ├── context-engineering.md    # 上下文工程
│   ├── agent-engineering.md      # Agent 工程
│   ├── harness-engineering.md    # 驾驭工程
│   ├── overview.md                  # 综述（v1 合并版）
│   ├── README.md                    # 系列阅读指南
│   ├── sources.jsonl                # 全部一手来源清单
│   └── html/                        # 渲染后的网页版（含左侧 TOC + 章节切换）
│
└── interview-prep/               ← 实战：求职 JD + 面试题
    ├── jd-requirements.md        # AI Agent 岗位 JD 要求研究（中外大厂真实 JD 拆解）
    ├── interview-questions.md    # 86 道高质量面试题与答案（10 大分类）
    └── html/                        # 渲染后的网页版
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

**86 道题，10 大分类**，每题：⭐ 难度分级 + 200-500 字技术答案 + 一手论文/官方文档来源 URL：

| 分类 | 题数 | 难度分布 | 重点 |
|---|---|---|---|
| **A. LLM 基础** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / 长上下文 / 推理优化 |
| **B. 训练 / 微调** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 涌现 |
| **C. 提示词工程** | 9 | 1⭐ 7⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / 自动 prompt 优化 |
| **D. RAG** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS |
| **E. Agent** | 12 | 1⭐ 6⭐⭐ 5⭐⭐⭐ | **核心**：6 patterns / 多 agent / 失败模式 / harness |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / mem0 / progress 文件 / memory corruption |
| **G. 系统设计** | 8 | 0⭐ 1⭐⭐ 7⭐⭐⭐ | 客服 / coding / 搜索 / LLM 网关 / 高并发 / 成本 / 沙箱 |
| **H. 评估** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | judge bias / pairwise / benchmark hack / 漂移 |
| **I. 安全 / 护栏** | 6 | 0⭐ 1⭐⭐ 5⭐⭐⭐ | OWASP / Prompt Injection / NeMo Guardrails / EU AI Act |
| **J. 行为题 / 实战** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto harness 闭环 / 0-1 落地路线 |

---

## 推荐学习路径

| 你是谁 | 推荐顺序 |
|---|---|
| **应届生 / 在校生**（求职准备） | `interview-prep/jd-requirements` 看市场 → `interview-prep/interview-questions` 分类 A/B/C/D 重点刷 → 反向去 `engineering-foundations/` 对应章节深化 |
| **1-3 年中级工程师**（转 AI 方向） | `engineering-foundations/context-engineering` 起读 → `agent-engineering` → `interview-prep/interview-questions` D/E/F 全刷 → G 选 1-2 题白板练 |
| **3-5 年资深 / Staff** | `engineering-foundations/harness-engineering` 通读 → `interview-prep/interview-questions` E/G/H/I/J 重点（Staff 的差异在系统设计 + 行为题）|
| **架构师 / 技术决策者** | `engineering-foundations/harness-engineering` 的"两条法则 + 四大支柱"+ 成熟度模型 L0-L4 + `interview-prep/interview-questions` G 系统设计全部 |
| **招聘方 / 面试官** | `interview-prep/jd-requirements` 校准岗位标准 → `interview-prep/interview-questions` 按分类难度出题 |

---

## 如何阅读

**Markdown 版**（推荐用 GitHub 直接看，或本地 VSCode / Obsidian / Typora 打开）

**HTML 版**（带左侧浮动 TOC + 章节切换 + 滚动高亮）：
```bash
# 在本地 clone 后用 Python 起个静态服务器
cd ai-agent-engineer-handbook/engineering-foundations/html
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000
```

或者直接双击 `html/prompt-engineering.html` 用浏览器打开。

---

## 工作流声明

本仓库内容由作者使用 [Claude Code](https://claude.com/claude-code) + [deep-research](https://github.com/) 工作流整理，结合大量一手论文 / 厂商官方文档 / 真实招聘 JD 二手核实，**带局限性声明的诚实研究**：

- 引用一手来源处都给了 URL，便于读者深挖
- 数字 / 价格 / 模型版本可能随时间变化，请按发布日期查最新
- 部分 JD 因招聘网站登录限制或动态渲染只能拿到摘要，已在 Methodology Appendix 标注证据等级

如发现错误或过时内容，欢迎开 issue / PR。

---

## 贡献

欢迎 issue / PR：
- 校正 / 补充事实（请附一手来源 URL）
- 新增面试题（请按现有格式：难度 + 200-500 字答案 + 来源）
- 补充 JD 摘录（请附公司名 + 岗位名 + 来源 URL + 抓取日期）

---

## License

[MIT](./LICENSE)。文章引用的论文与厂商官方文档归原作者所有，本仓库只做整理与中文转述。
