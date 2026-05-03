# 02 · 求职实战：JD + 面试题

返回主目录：[../README.md](../README.md)

---

## 本目录两份核心研究

### [01-jd-requirements.md](./01-jd-requirements.md) — AI Agent 岗位 JD 要求研究

**调研日期**：2026-05-02

- **中国市场**：14 条带原文 verbatim 的真实 JD 摘录
  - 大厂算法岗：字节豆包 / 阿里通义 / 腾讯混元 / 月之暗面 Kimi / 智谱 GLM / DeepSeek / MiniMax / 阶跃星辰
  - Agent 应用岗：腾讯混元 Agent / 蚂蚁 AI Force / 字节 Agent 软件架构师
  - 中型厂 / 创业：浙江义乌 / 湖南林泽 / 上市公司 AI Native 5 岗 / 上海 Video AIGC
  - 实习 / Prompt 工程师 / 推理优化 / AIGC
- **北美 / 欧洲市场**：8 条完整 JD + 薪资带
  - Anthropic FDE / Applied AI Engineer (US/London)
  - Glean / Vercel / Replit / Sierra / Cognition (Devin) / Cursor / Mistral
- **高频考察点 Top 20**（综合中外 JD）
- **岗位变种**：中国 7 类 + 北美 6 类区分（含薪资带）
- **海外 / 中国对比观察**

### [02-interview-questions.md](./02-interview-questions.md) — 86 道面试题与答案

**10 大分类**，每题 ⭐ 难度分级 + 200-500 字技术答案 + 一手论文/官方文档 URL：

| 分类 | 题数 | 重点 |
|---|---|---|
| A. LLM 基础 | 10 | Transformer / RoPE / KV cache / 长上下文 / 推理优化 |
| B. 训练 / 微调 | 10 | SFT / DPO / GRPO / LoRA / R1 涌现 |
| C. 提示词工程 | 9 | CoT / Self-Consistency / ReAct / 自动 prompt 优化 |
| D. RAG | 10 | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS |
| E. Agent | 12 | **核心**：6 patterns / 多 agent / 失败模式 / harness |
| F. Memory | 8 | MemGPT / mem0 / progress 文件 / memory corruption |
| G. 系统设计 | 8 | 客服 / coding / 搜索 / LLM 网关 / 高并发 / 成本 / 沙箱 |
| H. 评估 | 6 | judge bias / pairwise / benchmark hack / 漂移 |
| I. 安全 / 护栏 | 6 | OWASP / Prompt Injection / NeMo Guardrails / EU AI Act |
| J. 行为题 / 实战 | 7 | Hashimoto harness 闭环 / 0-1 落地路线 |

---

## HTML 版

`html/` 目录有渲染后的网页版（带左侧浮动 TOC + 文档间切换 + 滚动高亮）。本地浏览：

```bash
cd 02-job-search/html
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000/01_jd_requirements.html
```

---

## 配套阅读

- 想深化技术答案的"为什么"：[../01-llm-engineering-series/](../01-llm-engineering-series/) 是配套的四层工程深度长文系列
- 顶层导览：[../README.md](../README.md)
