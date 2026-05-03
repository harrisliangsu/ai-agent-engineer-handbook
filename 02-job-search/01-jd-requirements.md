# AI Agent 岗位面试 JD 要求研究报告

**调研日期**：2026-05-02
**覆盖范围**：中国大陆 + 北美 / 欧洲市场，BOSS 直聘 / 拉勾 / 51job / 智联 / V2EX / 猎聘 / Hacker News / LinkedIn / Greenhouse / Ashby / Glassdoor / Levels.fyi
**目标读者**：求职者、招聘方、转型 AI Agent 方向的工程师

---

## Executive Summary

2026 年 5 月，"AI Agent 工程师"已经从 2023 年的"概念词"转化为有清晰边界的工程岗位。中国大陆和北美/欧洲市场呈现**两极化与岗位收敛**的特征：

- **中国市场**：大厂算法岗（字节豆包 / 阿里通义 / 腾讯混元 / 百度文心 / 月之暗面 Kimi / 智谱 GLM / DeepSeek / MiniMax / 阶跃星辰 / 百川 / 零一万物）继续抢博士与顶会一作；中型厂和传统 IT/SaaS 把"AI Agent 工程师 / LLM 应用开发工程师"当成新工种放出，硬性要求高度趋同——**Python + LangChain/LangGraph + RAG + Tool Calling + 向量库 + Docker/K8s**。3-5 年经验普遍开 25-60K·14/15 薪。Prompt 工程师独立岗位明显萎缩（2025 年比 2023 年减少约 70%），正被"Agent 应用工程师"和"上下文工程"吸收。
- **北美/欧洲市场**：6 类岗位已收敛——**Forward Deployed Engineer (FDE)**、**Applied AI Engineer**、**Agent Engineer / Software Engineer Agent**、**Research Engineer**、**Deployed/Field Engineer**、**AI Platform/Infra Engineer**。Anthropic、Glean、Vercel、Replit、Sierra、Cognition (Devin)、Cursor、Harvey、Perplexity 是 mid-stage 主力雇主。Senior IC 在 SF / NYC / Seattle 集中在 base $200K-$320K（含 equity 通常 2-4 倍），London 折合 £225K-£240K，欧洲 €80K-€160K。

跨地区共同信号：**生产 LLM 经验、Python（含 TypeScript 作为常见 #2）、agent 框架（LangGraph 领先，Claude Agent SDK / OpenAI Agents SDK 上升）、评估基础设施、MCP（2026 年增速最快的单一技能）、Prompt/Context Engineering**。

---

## 1. 中国市场：完整 JD 摘录与硬指标

### 1.1 编程语言 / 框架 / 工具栈高频要求

下面均带原文 + URL，非二手转述：

1. **「精通 LangChain、LlamaIndex、AutoGen、Dify、CrewAI 等至少一种主流 Agent 开发框架」「能驾驭 ClaudeCode、Cursor 进行意图级编程」**
   — 浙江义乌 AI Agent 工程师（V2EX 招聘帖，25-35K/月）— <https://www.v2ex.com/t/1200716>

2. **「熟悉 Langchain、LangGraph、AutoGen 等 Agent 相关框架」「熟悉 LLM、LVM、推理等相关大模型知识」**
   — 腾讯混元大模型 Agent 开发工程师（深圳，3 年+，2025-11-28）— <https://jobs.niuqizp.com/job-vm85LaLa5.html>

3. **「精通 Python，熟悉 Java/Go/C++ 之一」「掌握 Prompt 设计、多轮对话管理、Tool/Function Calling」「熟悉 RAG 技术与向量数据库」**
   — 湖南林泽科技 LLM 应用工程师（长沙，1.5-2.5 万，3-5 年）— <https://www.zhaopin.com/jobdetail/CC455023520J40918852615.htm>

4. **「熟练掌握 C/C++、Python」「熟悉 GPU/AI 芯片编程（CUDA、OpenCL、Ascend C 等）」「熟悉主流大模型推理框架（vllm、sglang、tensorrt-llm、FasterTransformer）」**
   — 高级推理工程师通用 JD（牛客网集合）

5. **「掌握 Python、Java 和 TensorFlow、PyTorch 框架」**
   — 蚂蚁集团 AI Force AI-Agent 算法工程师（金融场景，社招 13-15 级）— <https://blog.csdn.net/m0_59162248/article/details/149967708>

6. **「熟悉 Coze、Spring AI Alibaba 等 AI Agent 开发工具」「熟悉 OpenAI、阿里通义、豆包、DeepSeek 等 API」**
   — 字节跳动 Agent 软件架构师-移动 OS（北京海淀，70-100K·15 薪，3-5 年）— <https://www.liepin.com/job/1978252587.shtml>

7. **「熟练使用 vibe coding，能独立完成实验与结果分析」**
   — 月之暗面 Kimi Agent Team 实习（北京，6-7K/月）— <https://www.superlinear.academy/c/collaborate/kimi-agent-team>

8. **「熟悉 ChatGLM 或 Llama 等模型的使用或训练经验者优先」**
   — 智谱 AI Agent 算法工程师 / 实习生（北京）— <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/137574967>

### 1.2 领域知识要求

9. **「精通 SFT、RLHF、RAG 等技术，有 1 年以上落地经验」** — 蚂蚁 AI Force 社招
10. **「研究 Prompt 工程、指令微调、RAG、Agent 系统等技术」** — 一线大厂大模型算法岗（V2EX 高薪贴 6-15 万+/月）— <https://www.v2ex.com/t/1186466>
11. **「AI Agent 架构设计、多代理系统、RAG、长期记忆等技术研究」** — 同上
12. **「整合 ASR、NLP、TTS 技术实现多轮对话和风险感知」** — 蚂蚁金融场景 AI-Agent 算法岗
13. **「探索大模型在 Ranking 技术中的应用范式，包括 LLM-based 向量召回、相关性模型、Ranking 模型等」** — 字节豆包大模型搜索增强算法工程师 — <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/145331986>
14. **「构建覆盖文生文、多模态理解与生成的全模态评测基准，设计多维度指标并开发自动化评测工具链」** — 腾讯混元大模型评测算法研究员 — <https://www.monimianshi.com/job/51362>
15. **「设计和开发多 Agent 协同交互框架，支持使用一套标准协议接入内外部开发的 Agent」** — 字节 Agent 软件架构师-移动 OS
16. **「跟踪并复现前沿 RL & Agent 论文，快速验证创新 idea」** — 月之暗面 Kimi Agent Team

### 1.3 工程能力 / 系统能力

17. **「具有复杂系统开发经验」「会微信小程序或 app 开发」** — 湖南林泽 LLM 应用工程师
18. **「系统设计、云原生、容器化（Docker/K8s）、API 开发」** — 浙江义乌 AI Agent 工程师 JD
19. **「主导客户端整体架构设计与重构，包括模块拆分、跨端协同、数据流管理、多模态融合等核心层面」** — 字节 Agent 软件架构师-移动 OS
20. **「评测/标注平台、AI 辅标、自动化质检及评测能力，持续提升各场景的准确性和效率」** — 腾讯混元 Agent 开发工程师
21. **「PaaS 智能体平台研发：多 Agent 编排、智能体设计、多智能体协作、任务规划、工具调用」** — 蚂蚁 PaaS 智能体平台研发专家

### 1.4 学历 / 经验门槛

| 公司类型 | 学历要求 | 经验要求 | 备注 |
|---|---|---|---|
| 头部大模型算法岗（DeepSeek / Kimi / 智谱 / 阶跃 / 百川 / MiniMax） | 硕博为主 | 顶会论文 / Kaggle / ACM 金牌优先 | DeepSeek 官方说"不设学历门槛"，实操高度筛选发文与竞赛 |
| 大厂 Agent 应用 / 落地工程岗 | 本科即可 | 3-5 年是主流 | 腾讯混元 Agent：本科+3 年；字节 Agent 软件架构师：本科+3-5 年 |
| 大模型推理 / Infra 工程师 | 硕士优先 | 3-7 年 | C++/CUDA 强背景；vLLM/SGLang 实操经验 |
| 实习生 | 在读 | 6 个月连续可投入 | 强调 vibe coding 能力和复现论文 |

### 1.5 加分项 / Nice-to-have（按出现频率降序）

1. **顶会一作**（NeurIPS / ICML / ACL / CVPR / ICLR / KDD）— 几乎所有大模型公司算法岗
2. **ACM / ICPC、NOI / IOI、Top Coder、Kaggle 获奖** — 字节豆包、阶跃星辰、DeepSeek、华为天才少年通道
3. **开源贡献**（vLLM、SGLang、LangChain、HuggingFace 模型仓库）— 腾讯混元、阿里通义、推理工程师 JD
4. **行业垂域经验**：金融（蚂蚁、智谱）、医疗（百川）、法律 NLP、电商搜索（字节、阿里）、智能客服 / 外呼（蚂蚁）
5. **顶尖高校博士 + "天才少年"通道**：字节年薪近亿挖入"天才少年"陈冠英（21 经济网，2026-04-17）；华为盘古多模态算法组保留组内"天才少年名额"

### 1.6 薪资水平（2026-05 抓取数据）

| 层级 | 岗位 / 公司 | 月薪 / 总包 | 地点 | 来源 |
|---|---|---|---|---|
| 应届 SSP | 字节 AIGC 算法（卓越） | 33-36K×15 + 5W 签字 + 10W 股票 ≈ 55W+ | 北京 / 上海 | CSDN 156735755 |
| 应届 SSP | 百度 大模型算法（卓越） | 40K+×16 + 10W 股票 ≈ 70W+ | 北京 | 同上 |
| 应届 SSP | 拼多多 搜推广算法（卓越） | 40K+×16 + 5W 签字 + 15W 股票 ≈ 80W+ | 上海 | 同上 |
| 应届 SP | 腾讯 算法（优先） | 30-32K×15 + 3W 签字 ≈ 48-52W | 深圳 | 同上 |
| 顶尖博士 | 大厂 AI 架构师 | 200 万+/年，传闻"近亿元"挖人个例 | 北京 | 21 经济网 |
| 资深 Agent 架构师 | 字节 Agent 软件架构师-移动 OS | 70-100K·15 薪 | 北京海淀 | 猎聘 1978252587 |
| 一线大厂 Agent | 预训练 / Agent / 多模态 / RL | 6-15W+/月 | 北京 / 上海 | V2EX 1186466 |
| 高级 Agent | AI Agent 专家 / 架构师 | 50-80K·18 薪 | 北京 / 上海 | 火山引擎 ADG |
| 中级 Agent | LLM 应用开发工程师（AI Agent 方向） | 40-70K·18 薪 | 上海 | 猎聘 |
| 中级 Agent | AI Agent 开发工程师 | 18-28K·14 薪 | — | 火山引擎 ADG |
| 中型厂 | 浙江义乌 AI Agent 工程师 | 25-35K | 浙江义乌 | V2EX 1200716 |
| 中型厂 | 湖南林泽 LLM 应用工程师 | 15-25K | 长沙 | 智联 |
| Prompt 工程师 | 北京 1959085873 | 15-30K·13 薪，1-3 年 | 北京 | 猎聘（已下线）|
| 实习 | 月之暗面 Kimi Agent | 6-7K/月，要求连续 6 月 | 北京 | Superlinear |
| 资深博士 | DeepSeek 深度学习研究员-AGI | 8-11W/月，14 薪，年包 154W | 北京 | 财新基金 |

行业宏观数据：AI 岗月薪下限均值 4.7 万、上限均值 7.8 万（时代周报 2025-09，仍是 2026 年招聘市场基准引用源）；大模型算法岗月薪应届中位 24.8K，90 分位 5.2 万。

### 1.7 不同级别岗位差异

- **应届**：算法岗看顶会 / 竞赛 / 实习；应用岗看 GitHub Agent 项目（LangChain / LlamaIndex demo + Cursor 重构经历）
- **1-3 年**：腾讯混元 Agent、湖南林泽这类岗位定位为"主力执行"，要 RAG、向量库、Tool Calling、Prompt 设计落地经验
- **3-5 年**：开始要"主导客户端架构设计与重构"、"设计多 Agent 协同框架"（字节 Agent 软件架构师），薪资跳到 70K+
- **5-10 年**：架构师 / 专家，要求"已在生产带过 Agent 系统上线" + "客户 / 业务层咨询能力"
- **资深博士 / 科研**：DeepSeek 给"深度学习研究员-AGI" 8-11 万月薪、14 薪，年包 154 万

---

## 2. 北美 / 欧洲市场：完整 JD 摘录与硬指标

### 2.1 8 条真实完整 JD 摘录（原文 + URL，2026-05-02 抓取）

#### JD-EN-1：Anthropic — Forward Deployed Engineer, Applied AI

URL：<https://job-boards.greenhouse.io/anthropic/jobs/4985877008>
地点：Boston / Chicago / NYC / Seattle / SF / Washington DC
薪资：**$200,000 – $300,000**

> "Embed directly with our most strategic customers to drive transformational AI adoption."
> "Work within customer systems to build production applications with Claude models."
> "Deliver technical artifacts for customers like MCP servers, sub-agents, and agent skills."
> "Provide white glove deployment support for Anthropic products in enterprise environments."
> "3+ years of experience in a technical, customer facing role."
> "Production experience with LLMs including advanced prompt engineering, agent development, evaluation frameworks."
> "Strong programming skills with proficiency in Python (and ideally in one or more additional languages)."

差旅 ~25%；25% in-office minimum；提供签证支持。

#### JD-EN-2：Anthropic — Applied AI Engineer, Enterprise Tech

URL：<https://job-boards.greenhouse.io/anthropic/jobs/5057647008>
地点：SF / NYC / Seattle，每周 3 天 in office
薪资：**$200,000 – $320,000**

> "Serve as a specialist technical advisor to Anthropic customers as they deploy new products & workflows."
> "Influence technical architecture decisions and customer product strategy by developing customized pilots, prototypes, and evaluation suites."
> "Lead hands-on technical workshops and code reviews with customer engineering teams."
> "Create scalable public and internal assets, documenting the latest LLM prompting, evaling, agentic, and architecture techniques."

要求：**"Production experience with LLMs including advanced prompt engineering, agent development and frameworks, evaluation frameworks, transcript analysis, MCP, and deployment at scale."** "Strong programming skills with proficiency in Python or TypeScript." 4+ YOE。

#### JD-EN-3：Anthropic — Applied AI Engineer (London)

URL：<https://job-boards.greenhouse.io/anthropic/jobs/5116274008>
薪资：**£225,000 – £240,000 GBP**

> "Production experience building LLM-powered applications, including prompting, context engineering, agent architectures."

4+ YOE。

#### JD-EN-4：Glean — Machine Learning Engineer, AI Assistant & Autonomous AI Agents

URL：<https://job-boards.greenhouse.io/gleanwork/jobs/4605215005>
地点：SF Bay Area，hybrid 3-4 天/周
薪资：**$240,000 – $300,000 base**

> "Build frameworks for LLM-powered agents to use tools and knowledge sources effectively."
> "Invent new agentic architectures and signals to improve reasoning, planning, and personalization."
> "Design and optimize reinforcement learning and fine-tuning approaches."
> "Lead development of scalable evaluation, benchmarking, and optimization loops for agents."

要求：2+ years as Staff/Principal + 5+ years AI/ML industry；Python / Go / Java / C++。

#### JD-EN-5：Replit — Senior Software Engineer, Agent Platform

URL：<https://jobs.reachcapital.com/companies/replit-2/jobs/65431609-agent-platform-engineer>
地点：Foster City, CA hybrid (M/W/F in office)
薪资：**$180K – $260K + equity + bonus**

> "Bridging the gap between the AI team (working on the core Agent logic) and the UX team (crafting delightful Agent experiences)."
> "Tackling complex challenges across the full stack, from browser-based interfaces to high-performance backends to Linux systems engineering."

要求：5+ YOE backend services；"experience building AI-powered tooling or products"；"experience with collaborative editing technologies such as OT, CRDTs, or Git"；"experience with event sourcing systems"。

#### JD-EN-6：Cognition — Deployed Engineer (Devin)

URL：<https://www.glassdoor.com/job-listing/deployed-engineer-cognition-JV_KO0,17_KE18,27.htm>
地点 / 薪资：SF $98K-$165K / Emeryville $120K-$180K / Austin $82K-$137K / Federal SF $152K-$241K

> "Deployed Engineers (DEs) are customer-facing technical experts who help customers maximize the value they get out of Devin/Windsurf."
> "Work directly with engineers to understand their workflows, identify high-impact use cases, and deploy Devin/Windsurf into real production environments, leading demos and pilots, supporting integrations, and driving adoption of the Cognition platform."

#### JD-EN-7：Vercel — AI Engineer

URL：<https://vercel.com/careers/ai-engineer-5517523004>
SF base：**$192,000 – $288,000**

> "Sit at the intersection of machine learning, product development, and frontend engineering, working closely with product designers, ML scientists, and full-stack engineers to prototype and deploy innovative systems."

要求：5+ YOE；strong programming in Python and JavaScript/TypeScript；experience working with or fine-tuning LLMs (OpenAI, Hugging Face)；solid understanding of modern frontend (React, Tailwind)。

#### JD-EN-8：Sierra — Software Engineer, Agent + APX (New Grad)

URL：<https://jobs.ashbyhq.com/Sierra/b7d1dbcd-ca72-472f-b15e-5b4b0f886be0>，APX：<https://jobs.ashbyhq.com/sierra/6a75b530-b7bb-4439-bb67-37b4f2b75b96>

> "Agent Engineers at Sierra design and deliver production-grade AI agents that are central, mission-critical and drive revenue directly to Sierra's growth."
> "APX is an 18-month rotational program that gives new graduates in computer science or a related field the opportunity to work closely with their customers, building AI agents as both an Agent Engineer and an Agent Product Manager."

### 2.2 LangChain Ecosystem 频率统计（2026-04 数据）

来源：<https://agentic-engineering-jobs.com/langchain-job-market-2026>

| 技能 | 出现频率 | 备注 |
|---|---|---|
| Python | **93.4%** | 通用默认 |
| LangChain | 34.3% | 全部 agentic 岗位 |
| AWS | 34.7% | 云平台 #1 |
| Docker | 32.4% | DevOps baseline |
| GCP | 29.1% | 云平台 #2 |
| Kubernetes | 27.2% | 容器编排 |
| Azure | 24.4% | 云平台 #3 |
| LangGraph | 22.1% | Agent orchestration |
| Pinecone | 18.8% | 向量库 #1 |
| FastAPI | 17.4% | Python 后端 |
| TypeScript | 17.4% | 前端 / fullstack |
| **MCP** | **16.9%** | **2026 增速最快的单一技能** |
| Weaviate | 16.0% | 向量库 #2 |
| LangSmith | 8.5% | 评估观测 |

### 2.3 北美 / 欧洲薪资对照（2026-05）

| 公司 | 岗位 | base | 来源 |
|---|---|---|---|
| Anthropic | FDE Applied AI (US) | $200K-$300K | greenhouse.io/anthropic/jobs/4985877008 |
| Anthropic | Applied AI Engineer Enterprise Tech (SF/NYC/Seattle) | $200K-$320K | greenhouse.io/anthropic/jobs/5057647008 |
| Anthropic | Applied AI Engineer (London) | £225K-£240K | greenhouse.io/anthropic/jobs/5116274008 |
| OpenAI | Software Engineer (median TC) | $580K（top reported $1.27M） | levels.fyi/companies/openai |
| Google DeepMind | Research Engineer Gemini Post-Training (NYC) | $215K-$250K base + bonus + equity | startup.jobs/research-engineer-gemini-post-training-nyc-deepmind-5964742 |
| Glean | MLE AI Assistant + Autonomous Agents | $240K-$300K | greenhouse.io/gleanwork/jobs/4605215005 |
| Vercel | AI Engineer (SF) | $192K-$288K | vercel.com/careers/ai-engineer-5517523004 |
| Replit | Senior SWE Agent Platform | $180K-$260K + equity + bonus | reachcapital.com/companies/replit-2 |
| Cognition | Deployed Engineer SF | $98K-$165K | Glassdoor |
| Cognition | Deployed Engineer Federal SF | $152K-$241K | Glassdoor |
| Mistral | Research Scientist | €80K-€160K + equity | eujobs.co/mistral |
| Industry | Prompt Engineer (entry → senior) | $90K → $250K+ | pecollective.com/blog/prompt-engineering-salary-guide |

LangChain ecosystem：framework-agnostic agent roles 中位 max **$290K**；LangChain-only 中位 **$210K**；Senior+ 平均 top **$264K**。

---

## 3. 高频考察点 Top 20（综合中外 JD）

| # | 技能 / 知识点 | 在哪些公司 / JD 见过 | 解释 |
|---|---|---|---|
| 1 | **Python 精通**（含 PyTorch / FastAPI） | 几乎全部 JD | 通用默认 |
| 2 | **生产 LLM 应用经验**（"shipped LLM apps"） | Anthropic / OpenAI / Glean / HN listings | 有 production track record |
| 3 | **LangChain / LangGraph** | 腾讯混元、字节、义乌 JD、上市公司 AI Native、湖南林泽、Anthropic 生态 | 主流 agent 框架 |
| 4 | **RAG（向量库 + 检索增强）** | 蚂蚁、字节、腾讯混元、智谱、湖南林泽、义乌、Kimi、Glean、Perplexity、Harvey | 长上下文之外的知识扩展 |
| 5 | **Tool Calling / Function Calling / 多轮对话管理** | 湖南林泽、腾讯混元、字节 Agent 架构师 | Agent 核心能力 |
| 6 | **Prompt 工程 / 指令微调** | 一线大厂高薪贴、智谱、Prompt 工程师独立岗、Anthropic | 从 prompt 到 context engineering |
| 7 | **SFT / RLHF / DPO** | 蚂蚁、月之暗面、阶跃星辰、DeepSeek、字节豆包、百度、OpenAI Applied | 训练栈核心 |
| 8 | **Multi-Agent / 任务规划 / 工具调用编排** | 字节 Agent 架构师、腾讯混元 Agent、蚂蚁 PaaS、Sierra、Glean | 复杂业务必备 |
| 9 | **顶会论文一作**（NeurIPS / ICML / ACL / CVPR / ICLR / KDD） | 字节、腾讯、阿里、阶跃、DeepSeek、智谱、Kimi、DeepMind、Mistral | 算法岗硬筛 |
| 10 | **ACM / ICPC、NOI / IOI、Kaggle 竞赛奖** | 字节、阶跃、DeepSeek、华为天才少年 | 算法岗加分 |
| 11 | **Docker / K8s / 云原生 / 微服务** | 义乌 JD、上市公司 AI Native、上海 Video AIGC、Microsoft、Replit | 部署基础 |
| 12 | **向量数据库**（Faiss / Milvus / Chroma / Pinecone / Weaviate / Qdrant） | 湖南林泽、义乌、Glean、RAG-heavy 岗 | 检索基础设施 |
| 13 | **AutoGen / CrewAI / Dify / Coze 等 Agent 框架** | 义乌、字节移动 OS、腾讯混元 Agent | 国内生态 |
| 14 | **推理优化 vLLM / SGLang / TensorRT-LLM** | 推理工程师、MiniMax 推理优化、HN listings | 自部署 |
| 15 | **模型评测 / Eval / A/B 测试** | 腾讯混元评测、智谱、Glean、Anthropic、LangSmith team | CI/CD gating |
| 16 | **MCP（Model Context Protocol）** | Anthropic 显式、Cognition 暗示、国内 V2EX 讨论 | **2026 年最快增长的单一技能** |
| 17 | **AI Coding 工具：Cursor / Claude Code / vibe coding** | 月之暗面、上市公司 AI Native、义乌 JD | 工程效率工具 |
| 18 | **行业垂域**：金融 / 智能客服 / 外呼 / 智能家居 / IoT | 蚂蚁 AI Force、湖南林泽、Harvey（法律） | 业务交付 |
| 19 | **C++ / CUDA / GPU 编程** | 推理工程师 JD、华为大模型算法、DeepMind | infra 岗 |
| 20 | **英文论文阅读与复现能力** | 智谱、Kimi、阶跃星辰、字节豆包、几乎所有研究岗 | 算法岗硬要求 |

---

## 4. 岗位变种区分

### 4.1 中国市场 7 类

| 岗位变种 | 核心交付物 | 主要要求 | 薪资带（北京 / 上海） | 典型公司 |
|---|---|---|---|---|
| 大模型算法工程师（预训练 / 后训练） | 模型本身效果 | 顶会一作 + PyTorch + RLHF/SFT + 千卡训练经验 | 应届 SSP 50-80W；社招 100-300W | DeepSeek、字节豆包 Seed、阿里通义、Kimi、智谱、阶跃 |
| Agent 应用算法工程师 | 业务场景内的 Agent 流程效果 | RAG + Tool Use + 多 Agent 编排 + 评测 | 35-60K·15-18 薪 | 腾讯混元、蚂蚁 AI Force、智谱、阿里 Qwen-Agent |
| LLM 应用 / 后端开发工程师 | 上线服务、并发、可用性 | Python/Go + LangChain + 向量库 + Docker/K8s | 15-35K·13/14 薪 | 美团、京东、湖南林泽、义乌中型厂 |
| Prompt 工程师（独立岗位） | Prompt 模板 + AIGC 产物 | ChatGPT/Claude 使用经验 + Excel/SQL/Python | 15-30K·13 薪 | 中小型 AIGC 公司，**正在快速萎缩** |
| 大模型推理优化 / Infra 工程师 | 推理延迟 / 吞吐 | vLLM/SGLang + CUDA + 算子开发 | 30-60K·15 薪 | MiniMax、华为昇腾、字节、阿里 AI Infra |
| AI Coding Agent 产品工程师 | 内部"用 Cursor/Claude Code 写代码"产线 | 全栈 + Cursor/Claude Code 实操 + 0-1 产品力 | 25-60K，部分上市公司"3-5 倍效率"作硬指标 | 上市公司 AI Native 创新中心、Cursor/Coze 类创业 |
| AIGC（视觉 / 视频）工程师 | 图 / 视频生成产品效果 | Diffusion + ControlNet + LoRA + ComfyUI 工作流 | 7-25K（中小厂）；大厂 30K+ | 腾讯混元 AIGC、ComfyUI 工程师、Video AIGC 创业 |

### 4.2 北美 / 欧洲 6 类

| 岗位 | 定位 | 典型雇主 | 薪资 |
|---|---|---|---|
| **Research Engineer (RE)** | 训模型 / 算法研究 | DeepMind、OpenAI、Anthropic、Mistral | $215K-$300K base + equity |
| **Applied AI Engineer** | 在模型之上构建应用与参考架构 | Anthropic、Glean、Vercel | $200K-$320K base |
| **Forward Deployed Engineer (FDE)** | 嵌入战略客户做 white-glove 部署 | Anthropic、Cognition、Palantir | $200K-$300K base |
| **Deployed / Field Engineer** | sales-attached 客户实施 | Cognition (Devin)、Cursor、LangChain | $98K-$241K |
| **MLE / Agent Platform Engineer** | 构建 agent 框架 / 平台 / 评估基础设施 | Glean、Replit、Sierra | $180K-$300K base |
| **AI Infra Engineer** | vLLM / 推理 / 服务基础设施 | Modal、Together AI、Replicate | 与公司估值挂钩 |

---

## 5. 海外 / 中国对比观察

1. **薪资差**：北美 Anthropic / OpenAI Agent 工程师 base + equity 折合人民币年包普遍 200-400 万，中国大厂同等级 70-200 万；但中国应届 SSP（拼多多 80W、百度 70W、字节 55W+）在头部已与海外同年级新人接近。

2. **技术栈差异**：
   - 海外 JD 提 **Anthropic SDK / OpenAI SDK / MCP / Claude Code** 频率高
   - 国内更多写 **LangChain / LangGraph / Coze / Dify / 通义千问 / 文心 / DeepSeek API**
   - **MCP** 在国内 JD 中刚开始零星出现，主要在 V2EX、知乎工程师讨论中

3. **学历偏好**：
   - 海外更看重产品落地（GitHub、Show HN、Demo）
   - 中国头部公司仍把"顶会论文 + 大厂实习"作为算法岗硬筛
   - 应用工程岗在两边趋同

4. **工作模式**：
   - 国内多数 JD 默认 Onsite / 大小周或 9-9-6（智谱明确"双休、不卷"是亮点）
   - 海外 hybrid 3 天 in office 是主流（Microsoft AI 2026-01 起改为 4 天/周）
   - 远程岗仅在小型创业公司和 Dify、Corust 等出现

5. **出海 / 转岗趋势**：Manus 总部迁新加坡后 80 人国内裁员、40 人留新加坡；这是 2026 年中国 AI Agent 创业公司"出海保命"的代表性事件，对求职者意味着海外英文沟通能力开始重要。

6. **地理薪资**：欧洲 vs 美国 base 差距 50-100%，但欧洲福利（医保、法定 PTO、育儿假）部分弥补；签证支持率仅 3.3%（LangChain ecosystem 数据）。

---

## 6. 面向求职者的关键建议

1. **MVP 项目**：一份能跑通的 LangChain/LangGraph + RAG + Tool Calling 项目（带 GitHub commits）已经是中型厂 AI Agent 工程师的最低入场门槛。义乌 JD 明确要求"提供 GitHub ID，展示 contributions"。

2. **算法岗 vs 应用岗分流**：算法岗仍是顶会论文 + Kaggle/ACM 双重赛道；走应用岗的话，把 Cursor / Claude Code 写代码的"3-5 倍效率"证据放进简历。

3. **薪资谈判**：参照 V2EX 1186466 一线大厂高薪贴的 6-15W+/月 范围对资深岗喊价，参照 CSDN 156735755 校招分级数据对应届 SP/SSP 喊价。

4. **Prompt 工程师转型**：Prompt 工程师独立岗位正在消失，建议把 title 改成"LLM 应用工程师 / Agent 工程师"，并补 Python + 向量库 + Docker。

5. **MCP 提前学**：MCP 是 2026 年增速最快的单一技能，提前掌握能在 6-12 个月内吃到红利。Anthropic Applied AI Engineer 已经把 MCP 列为 production 必备。

6. **Context Engineering 趋势**：关注岗位变化中的"上下文工程 / Context Engineering"——目前还在阿里云开发者社区与知乎讨论阶段，大概率会在 2026 下半年成为 JD 中的标准词。

7. **海外路径**：如果有英文沟通能力，Anthropic FDE / Applied AI Engineer 是当前性价比最高的 senior 岗（$200K-$320K base + equity，且公开招聘）；Cognition / Glean / Sierra / Cursor / Replit 是 mid-stage 不错的选择。

---

## Limitations & Caveats

1. **BOSS 直聘、智联、拉勾**的 JD 详情页大多需要登录或动态渲染，本次 WebFetch 仅能拿到部分（湖南林泽、Prompt 工程师等），其余通过搜索引擎缓存的摘要交叉验证。
2. **阿里通义实验室、百度文心、华为盘古**的官方招聘页需要登录或滑动验证，本报告中相关 JD 引自校招公开转载页和 CSDN/知乎二手汇编，可信度低于腾讯/字节那几条原文。
3. **京东 TGT、美团北斗计划、字节 TopSeed** 等"顶尖人才项目"具体年薪在公开 JD 中均不明示，仅在新闻稿（财联社、券商中国、虎嗅）中以"百万年薪 / 200 万 / 300 万"出现，请理解为报道而非 JD。
4. **OpenAI / HN April 2026 / Sierra Ashby** 三处页面对 WebFetch 返回 403 或空 body，相关内容来自搜索摘要二手转述，应当作为参考而非一手原文。
5. **LangChain ecosystem 频率数据**来自单一聚合源（agentic-engineering-jobs.com），可能过度代表 LangChain/LangGraph 相对于全局 agent 岗位的份额，作为方向性参考。
6. **Manus、零一万物**当前已极度收缩或迁出，相关 JD 不在本次有效抓取范围。

---

## Bibliography

### 中国市场（按引用顺序）

1. <https://jobs.bytedance.com/experienced/position/7563248386344847621/detail> — 字节豆包大语言模型 Agent 应用架构工程师
2. <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/145331986> — 字节豆包大模型搜索增强算法工程师 JD
3. <https://www.liepin.com/job/1978252587.shtml> — 字节 Agent 软件架构师-移动 OS（70-100K·15 薪）
4. <https://jobs.niuqizp.com/job-vm85LaLa5.html> — 腾讯混元大模型 Agent 开发工程师（深圳）
5. <https://www.monimianshi.com/job/51362> — 腾讯混元大模型评测算法研究员（北京）
6. <https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/135053896> — 腾讯混元 AIGC 算法研究员
7. <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/137574967> — 智谱 AI Agent 算法工程师 / 实习生
8. <https://www.superlinear.academy/c/collaborate/kimi-agent-team> — 月之暗面 Kimi Agent Team
9. <https://blog.csdn.net/m0_59162248/article/details/149967708> — 蚂蚁 AI Force AI-Agent 算法工程师
10. <https://career.nankai.edu.cn/correcruit/content/id/99919.html> — 阶跃星辰 Agent 算法工程师
11. <https://www.chnfund.com/article/AR4bb44e4e-4975-129f-48ed-3a17e29fe9bb> — DeepSeek 招聘 52 个岗位
12. <https://www.zhaopin.com/jobdetail/CC455023520J40918852615.htm> — 湖南林泽 LLM 应用工程师
13. <https://www.v2ex.com/t/1194601> — 上市公司 AI Native 前沿创新中心 5 岗
14. <https://www.v2ex.com/t/1200716> — 浙江义乌 AI Agent 工程师
15. <https://www.v2ex.com/t/1186466> — 一线大厂高薪 预训练/Agent/多模态/RL
16. <https://www.liepin.com/job/1959085873.shtml> — Prompt 工程师（北京 15-30K）
17. <https://blog.csdn.net/2401_85343303/article/details/156735755> — 2025 互联网大厂校招薪酬全解析
18. <https://www.21jingji.com/article/20260417/...> — 字节"年薪近亿"挖天才少年
19. <https://www.stcn.com/article/detail/2602152.html> — Manus 全球化困于合规与代码
20. <https://xz.chsi.com.cn/xz/zyts/202505/...> — Prompt 工程师为何遇冷（2025 岗位下降 70%）

### 北美 / 欧洲市场

21. <https://job-boards.greenhouse.io/anthropic/jobs/4985877008> — Anthropic FDE Applied AI
22. <https://job-boards.greenhouse.io/anthropic/jobs/5057647008> — Anthropic Applied AI Engineer Enterprise Tech
23. <https://job-boards.greenhouse.io/anthropic/jobs/5116274008> — Anthropic Applied AI Engineer London
24. <https://openai.com/careers/research-engineer-applied-ai-engineering-san-francisco/> — OpenAI Research Engineer Applied AI
25. <https://www.levels.fyi/companies/openai/salaries/software-engineer> — OpenAI SWE 薪资
26. <https://job-boards.greenhouse.io/gleanwork/jobs/4605215005> — Glean MLE
27. <https://deepmind.google/careers/> — DeepMind 招聘主页
28. <https://startup.jobs/research-engineer-gemini-post-training-nyc-deepmind-5964742> — DeepMind Gemini Post-Training NYC
29. <https://jobs.lever.co/mistral> — Mistral 招聘
30. <https://sierra.ai/careers> — Sierra 招聘主页
31. <https://jobs.ashbyhq.com/Sierra/b7d1dbcd-ca72-472f-b15e-5b4b0f886be0> — Sierra Software Engineer Agent
32. <https://cursor.com/careers> — Cursor 招聘主页
33. <https://cognition.ai/> — Cognition (Devin) 招聘主页
34. <https://www.glassdoor.com/job-listing/deployed-engineer-cognition-JV_KO0,17_KE18,27.htm> — Cognition Deployed Engineer
35. <https://replit.com/careers> — Replit 招聘
36. <https://jobs.reachcapital.com/companies/replit-2/jobs/65431609-agent-platform-engineer> — Replit Senior SWE Agent Platform
37. <https://vercel.com/careers/ai-engineer-5517523004> — Vercel AI Engineer
38. <https://www.langchain.com/careers> — LangChain 招聘
39. <https://agentic-engineering-jobs.com/langchain-job-market-2026> — LangChain ecosystem 分析
40. <https://news.ycombinator.com/item?id=47601859> — HN "Who is Hiring? April 2026"
41. <https://www.kore1.com/ai-engineer-salary-guide/> — AI Engineer 薪资 2026
42. <https://www.secondtalent.com/resources/most-in-demand-ai-engineering-skills-and-salary-ranges/> — Top 10 in-demand AI 技能
43. <https://www.eujobs.co/career-guides/mistral-ai-career-guide> — Mistral career guide
44. <https://pecollective.com/blog/prompt-engineering-salary-guide/> — Prompt Engineer 薪资 2026
45. <https://www.devopsschool.com/blog/agent-reliability-engineer-role-blueprint-...> — Agent Reliability Engineer blueprint

---

## Methodology Appendix

- **调研日期**：2026-05-02
- **检索路数**：26 路（中文 14 + 英文 12）
- **核心覆盖**：BOSS 直聘 / 拉勾 / 51job / 智联 / 牛客 / V2EX / 猎聘 / 知乎 / CSDN / 字节官方招聘页 / Greenhouse / Ashby / Glassdoor / Levels.fyi / LangChain ecosystem analysis / Hacker News
- **派遣 agent**：2 个并行 web-search-agent（中文 1 个 + 英文 1 个），每个调用 40-70 次 WebSearch / WebFetch
- **证据等级标记**：
  - **A 级（一手原文 verbatim）**：Anthropic Greenhouse、腾讯混元、湖南林泽、Glean、Vercel、Replit Reach Capital
  - **B 级（缓存摘要二手）**：BOSS 直聘 / 智联（需登录）、阿里通义、百度文心、华为盘古
  - **C 级（媒体报道转述）**：京东 TGT、美团北斗、字节"近亿元"挖人
  - **D 级（聚合统计）**：LangChain ecosystem analysis（agentic-engineering-jobs.com）
