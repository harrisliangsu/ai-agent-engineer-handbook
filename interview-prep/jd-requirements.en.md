# Research Report on JD Requirements for AI Agent Roles

> **English** · [中文](./jd-requirements.md)
>
> Chinese job excerpts are quoted in the original Chinese. US and EU excerpts that were published in English stay in English.

**Research date:** 2026-05-02
**Coverage:** Mainland China plus North America / Europe. BOSS Zhipin / Lagou / 51job / Zhaopin / V2EX / Liepin / Hacker News / LinkedIn / Greenhouse / Ashby / Glassdoor / Levels.fyi
**Intended readers:** Candidates, hiring managers, and engineers moving into AI Agent work

---

<a id="executive-summary"></a>

## Executive Summary

As of May 2026, "AI Agent engineer" has moved from a 2023 buzzword to an engineering role with a clear boundary. Mainland China and the North America / Europe markets show **polarization and role convergence**:

- **China market:** Big-tech algorithm roles (ByteDance Doubao / Alibaba Tongyi / Tencent Hunyuan / Baidu Wenxin / Moonshot Kimi / Zhipu GLM / DeepSeek / MiniMax / StepFun / Baichuan / 01.AI) are still competing for PhDs and first authors at top conferences. Mid-size firms and traditional IT/SaaS are opening "AI Agent engineer / LLM application engineer" as a new job category, and the hard requirements have converged tightly on **Python + LangChain/LangGraph + RAG + tool calling + a vector database + Docker/K8s**. Roles asking for 3-5 years commonly offer 25-60K RMB on a 14- or 15-month pay scheme. Standalone prompt-engineer openings have shrunk sharply (about 70% fewer in 2025 than in 2023) and are being absorbed into "Agent application engineer" and "context engineering."
- **North America / Europe:** Six role types have converged: **Forward Deployed Engineer (FDE)**, **Applied AI Engineer**, **Agent Engineer / Software Engineer, Agent**, **Research Engineer**, **Deployed/Field Engineer**, and **AI Platform/Infra Engineer**. Anthropic, Glean, Vercel, Replit, Sierra, Cognition (Devin), Cursor, Harvey, and Perplexity are the main mid-stage employers. Senior ICs in SF / NYC / Seattle cluster at a base of $200K-$320K (total compensation with equity is typically 2-4x base). London is £225K-£240K, and Europe is €80K-€160K.

Shared signals across regions: **production LLM experience, Python (with TypeScript as a common #2), agent frameworks (LangGraph leads; Claude Agent SDK / OpenAI Agents SDK are rising), evaluation infrastructure, MCP (the single fastest-growing skill of 2026), and prompt/context engineering.**

---

<a id="s1"></a>

## 1. China market: full JD excerpts and hard requirements

<a id="s1-1"></a>

### 1.1 High-frequency programming languages / frameworks / toolchains

Each item below carries the original wording plus a URL. These are not second-hand paraphrases:

1. Yiwu, Zhejiang AI Agent Engineer (V2EX job post, 25-35K RMB/month) requires mastery of at least one mainstream agent framework and intent-level programming with Claude Code and Cursor.
   **「精通 LangChain、LlamaIndex、AutoGen、Dify、CrewAI 等至少一种主流 Agent 开发框架」「能驾驭 ClaudeCode、Cursor 进行意图级编程」**
   — <https://www.v2ex.com/t/1200716>

2. Tencent Hunyuan LLM Agent development engineer (Shenzhen, 3+ years, 2025-11-28) asks for familiarity with agent frameworks such as LangChain, LangGraph, and AutoGen, and with LLM, LVM, inference, and related large-model knowledge.
   **「熟悉 Langchain、LangGraph、AutoGen 等 Agent 相关框架」「熟悉 LLM、LVM、推理等相关大模型知识」**
   — <https://jobs.niuqizp.com/job-vm85LaLa5.html>

3. Hunan Linze Technology LLM application engineer (Changsha, 15-25K RMB, 3-5 years) asks for strong Python plus one of Java/Go/C++, prompt design, multi-turn dialogue management, tool/function calling, and RAG with vector databases.
   **「精通 Python，熟悉 Java/Go/C++ 之一」「掌握 Prompt 设计、多轮对话管理、Tool/Function Calling」「熟悉 RAG 技术与向量数据库」**
   — <https://www.zhaopin.com/jobdetail/CC455023520J40918852615.htm>

4. A generic senior inference-engineer JD (Nowcoder compilation) asks for strong C/C++ and Python, GPU/AI-chip programming (CUDA, OpenCL, Ascend C, and similar), and mainstream LLM inference frameworks (vLLM, SGLang, TensorRT-LLM, FasterTransformer).
   **「熟练掌握 C/C++、Python」「熟悉 GPU/AI 芯片编程（CUDA、OpenCL、Ascend C 等）」「熟悉主流大模型推理框架（vllm、sglang、tensorrt-llm、FasterTransformer）」**

5. Ant Group AI Force AI-Agent algorithm engineer (financial scenarios, experienced hire, levels 13-15) asks for Python, Java, and the TensorFlow and PyTorch frameworks.
   **「掌握 Python、Java 和 TensorFlow、PyTorch 框架」**
   — <https://blog.csdn.net/m0_59162248/article/details/149967708>

6. ByteDance Agent software architect, mobile OS (Haidian, Beijing, 70-100K RMB x 15 months, 3-5 years) asks for familiarity with AI Agent tools such as Coze and Spring AI Alibaba, and with APIs from OpenAI, Alibaba Tongyi, Doubao, and DeepSeek.
   **「熟悉 Coze、Spring AI Alibaba 等 AI Agent 开发工具」「熟悉 OpenAI、阿里通义、豆包、DeepSeek 等 API」**
   — <https://www.liepin.com/job/1978252587.shtml>

7. Moonshot Kimi Agent Team internship (Beijing, 6-7K RMB/month) asks for fluent vibe coding and the ability to run experiments and analyze results independently.
   **「熟练使用 vibe coding，能独立完成实验与结果分析」**
   — <https://www.superlinear.academy/c/collaborate/kimi-agent-team>

8. Zhipu AI Agent algorithm engineer / intern (Beijing) prefers candidates with experience using or training models such as ChatGLM or Llama.
   **「熟悉 ChatGLM 或 Llama 等模型的使用或训练经验者优先」**
   — <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/137574967>

<a id="s1-2"></a>

### 1.2 Domain-knowledge requirements

9. Ant Group AI Force experienced hire claims mastery of SFT, RLHF, and RAG, with more than one year of production deployment.
   **「精通 SFT、RLHF、RAG 等技术，有 1 年以上落地经验」**
10. A frontline big-tech LLM algorithm role (V2EX high-pay post, 60-150K+ RMB/month) covers research on prompt engineering, instruction tuning, RAG, and agent systems.
    **「研究 Prompt 工程、指令微调、RAG、Agent 系统等技术」**
    — <https://www.v2ex.com/t/1186466>
11. The same posting covers research on AI Agent architecture, multi-agent systems, RAG, and long-term memory.
    **「AI Agent 架构设计、多代理系统、RAG、长期记忆等技术研究」**
12. Ant Group's financial-scenario AI-Agent algorithm role asks for integrating ASR, NLP, and TTS so the system can hold multi-turn conversations and sense risk.
    **「整合 ASR、NLP、TTS 技术实现多轮对话和风险感知」**
13. ByteDance Doubao LLM search-augmentation algorithm engineer asks for exploring how large models are applied in ranking, including LLM-based vector recall, relevance models, and ranking models.
    **「探索大模型在 Ranking 技术中的应用范式，包括 LLM-based 向量召回、相关性模型、Ranking 模型等」**
    — <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/145331986>
14. Tencent Hunyuan LLM evaluation algorithm researcher asks for a full-modality evaluation benchmark covering text-to-text, multimodal understanding, and generation, plus multi-dimensional metrics and an automated evaluation toolchain.
    **「构建覆盖文生文、多模态理解与生成的全模态评测基准，设计多维度指标并开发自动化评测工具链」**
    — <https://www.monimianshi.com/job/51362>
15. ByteDance Agent software architect, mobile OS, asks for a multi-agent collaboration framework that connects internally and externally built agents through one standard protocol.
    **「设计和开发多 Agent 协同交互框架，支持使用一套标准协议接入内外部开发的 Agent」**
16. Moonshot Kimi Agent Team asks candidates to track and reproduce frontier RL and agent papers and to validate new ideas quickly.
    **「跟踪并复现前沿 RL & Agent 论文，快速验证创新 idea」**

<a id="s1-3"></a>

### 1.3 Engineering ability / systems ability

17. Hunan Linze LLM application engineer asks for complex-systems development experience and WeChat mini-program or app development.
    **「具有复杂系统开发经验」「会微信小程序或 app 开发」**
18. The Yiwu, Zhejiang AI Agent Engineer JD asks for systems design, cloud native, containerization (Docker/K8s), and API development.
    **「系统设计、云原生、容器化（Docker/K8s）、API 开发」**
19. ByteDance Agent software architect, mobile OS, asks the hire to lead overall client architecture design and refactoring, including module splits, cross-client coordination, data-flow management, and multimodal fusion.
    **「主导客户端整体架构设计与重构，包括模块拆分、跨端协同、数据流管理、多模态融合等核心层面」**
20. Tencent Hunyuan Agent development engineer covers evaluation and labeling platforms, AI-assisted labeling, automated quality checks, and evaluation capability, continuously raising accuracy and efficiency in each scenario.
    **「评测/标注平台、AI 辅标、自动化质检及评测能力，持续提升各场景的准确性和效率」**
21. Ant Group PaaS agent-platform R&D expert covers PaaS agent-platform development: multi-agent orchestration, agent design, multi-agent collaboration, task planning, and tool calling.
    **「PaaS 智能体平台研发：多 Agent 编排、智能体设计、多智能体协作、任务规划、工具调用」**

<a id="s1-4"></a>

### 1.4 Education / experience thresholds

| Company type | Education | Experience | Notes |
|---|---|---|---|
| Top LLM algorithm roles (DeepSeek / Kimi / Zhipu / StepFun / Baichuan / MiniMax) | Master's or PhD is the norm | Top-conference papers / Kaggle / ACM gold preferred | DeepSeek officially says it sets no degree threshold; in practice hiring screens heavily on publications and contests |
| Big-tech Agent application / production engineering | A bachelor's degree is enough | 3-5 years is the mainstream | Tencent Hunyuan Agent: bachelor's + 3 years; ByteDance Agent software architect: bachelor's + 3-5 years |
| LLM inference / infra engineer | Master's preferred | 3-7 years | Strong C++/CUDA background; hands-on vLLM/SGLang |
| Intern | Currently enrolled | 6 consecutive months available | Emphasizes vibe-coding ability and paper reproduction |

<a id="s1-5"></a>

### 1.5 Nice-to-haves (descending by how often they appear)

1. **First-author papers at top conferences** (NeurIPS / ICML / ACL / CVPR / ICLR / KDD). Nearly every algorithm role at major LLM companies.
2. **ACM / ICPC, NOI / IOI, TopCoder, and Kaggle awards.** ByteDance Doubao, StepFun, DeepSeek, and the Huawei Genius Youth track.
3. **Open-source contributions** (vLLM, SGLang, LangChain, Hugging Face model repositories). Tencent Hunyuan, Alibaba Tongyi, and inference-engineer JDs.
4. **Vertical-industry experience:** finance (Ant Group, Zhipu), healthcare (Baichuan), legal NLP, e-commerce search (ByteDance, Alibaba), intelligent customer service / outbound calling (Ant Group).
5. **A PhD from a top university plus a "Genius Youth" track:** ByteDance offered a package of nearly 100 million RMB a year to recruit Genius Youth Chen Guanying (陈冠英) (21st Century Business Herald, 2026-04-17). Huawei's Pangu multimodal algorithm group keeps an internal Genius Youth quota.

<a id="s1-6"></a>

### 1.6 Pay levels (data captured 2026-05)

| Level | Role / company | Monthly pay / package (RMB) | Location | Source |
|---|---|---|---|---|
| New-grad SSP | ByteDance AIGC algorithms (outstanding) | 33-36K x 15 + 50K signing bonus + 100K stock, about 550K+ | Beijing / Shanghai | CSDN 156735755 |
| New-grad SSP | Baidu LLM algorithms (outstanding) | 40K+ x 16 + 100K stock, about 700K+ | Beijing | Same as above |
| New-grad SSP | Pinduoduo search / ads / recommendation algorithms (outstanding) | 40K+ x 16 + 50K signing bonus + 150K stock, about 800K+ | Shanghai | Same as above |
| New-grad SP | Tencent algorithms (priority) | 30-32K x 15 + 30K signing bonus, about 480-520K | Shenzhen | Same as above |
| Top PhD | Big-tech AI architect | 2M+ per year; a reported one-off poach near 100 million RMB | Beijing | 21st Century Business Herald |
| Senior Agent architect | ByteDance Agent software architect, mobile OS | 70-100K x 15 months | Haidian, Beijing | Liepin 1978252587 |
| Frontline big-tech Agent | Pretraining / Agent / multimodal / RL | 60-150K+ per month | Beijing / Shanghai | V2EX 1186466 |
| Senior Agent | AI Agent expert / architect | 50-80K x 18 months | Beijing / Shanghai | Volcengine ADG |
| Mid-level Agent | LLM application engineer (AI Agent track) | 40-70K x 18 months | Shanghai | Liepin |
| Mid-level Agent | AI Agent development engineer | 18-28K x 14 months | — | Volcengine ADG |
| Mid-size company | Yiwu, Zhejiang AI Agent Engineer | 25-35K | Yiwu, Zhejiang | V2EX 1200716 |
| Mid-size company | Hunan Linze LLM application engineer | 15-25K | Changsha | Zhaopin |
| Prompt engineer | Beijing 1959085873 | 15-30K x 13 months, 1-3 years | Beijing | Liepin (posting removed) |
| Intern | Moonshot Kimi Agent | 6-7K/month, 6 consecutive months required | Beijing | Superlinear |
| Senior PhD | DeepSeek deep-learning researcher, AGI | 80-110K/month, 14 months, annual package 1.54M | Beijing | Caixin Fund |

Industry-level figures: the average monthly floor for AI roles is 47K RMB and the average ceiling is 78K RMB (Time Weekly, 2025-09; still cited as a baseline for the 2026 hiring market). For LLM algorithm roles, the new-grad median monthly pay is 24.8K RMB and the 90th percentile is 52K RMB.

<a id="s1-7"></a>

### 1.7 How roles differ by seniority

- **New graduates:** Algorithm roles look at top conferences, contests, and internships. Application roles look at GitHub agent projects (a LangChain / LlamaIndex demo plus experience refactoring with Cursor).
- **1-3 years:** Roles such as Tencent Hunyuan Agent and Hunan Linze are positioned as the people who execute the work. They want shipped experience with RAG, vector databases, tool calling, and prompt design.
- **3-5 years:** The bar moves to leading client architecture design and refactoring, and to designing multi-agent collaboration frameworks (ByteDance Agent software architect). Pay jumps above 70K RMB.
- **5-10 years:** Architect / expert. The requirement is having taken an agent system live in production, plus consulting ability at the customer and business layer.
- **Senior PhD / research:** DeepSeek pays the Deep Learning Researcher, AGI role 80-110K RMB per month on a 14-month scheme, an annual package of 1.54 million RMB.

---

<a id="s2"></a>

## 2. North America / Europe: full JD excerpts and hard requirements

<a id="s2-1"></a>

### 2.1 Eight real, complete JD excerpts (original text + URL, captured 2026-05-02)

#### JD-EN-1: Anthropic — Forward Deployed Engineer, Applied AI

URL: <https://job-boards.greenhouse.io/anthropic/jobs/4985877008>
Location: Boston / Chicago / NYC / Seattle / SF / Washington DC
Pay: **$200,000 – $300,000**

> "Embed directly with our most strategic customers to drive transformational AI adoption."
> "Work within customer systems to build production applications with Claude models."
> "Deliver technical artifacts for customers like MCP servers, sub-agents, and agent skills."
> "Provide white glove deployment support for Anthropic products in enterprise environments."
> "3+ years of experience in a technical, customer facing role."
> "Production experience with LLMs including advanced prompt engineering, agent development, evaluation frameworks."
> "Strong programming skills with proficiency in Python (and ideally in one or more additional languages)."

Travel is about 25%. There is a 25% in-office minimum. Visa sponsorship is offered.

#### JD-EN-2: Anthropic — Applied AI Engineer, Enterprise Tech

URL: <https://job-boards.greenhouse.io/anthropic/jobs/5057647008>
Location: SF / NYC / Seattle, 3 days in office per week
Pay: **$200,000 – $320,000**

> "Serve as a specialist technical advisor to Anthropic customers as they deploy new products & workflows."
> "Influence technical architecture decisions and customer product strategy by developing customized pilots, prototypes, and evaluation suites."
> "Lead hands-on technical workshops and code reviews with customer engineering teams."
> "Create scalable public and internal assets, documenting the latest LLM prompting, evaling, agentic, and architecture techniques."

Requirements: **"Production experience with LLMs including advanced prompt engineering, agent development and frameworks, evaluation frameworks, transcript analysis, MCP, and deployment at scale."** "Strong programming skills with proficiency in Python or TypeScript." 4+ years of experience.

#### JD-EN-3: Anthropic — Applied AI Engineer (London)

URL: <https://job-boards.greenhouse.io/anthropic/jobs/5116274008>
Pay: **£225,000 – £240,000 GBP**

> "Production experience building LLM-powered applications, including prompting, context engineering, agent architectures."

4+ years of experience.

#### JD-EN-4: Glean — Machine Learning Engineer, AI Assistant & Autonomous AI Agents

URL: <https://job-boards.greenhouse.io/gleanwork/jobs/4605215005>
Location: SF Bay Area, hybrid 3-4 days/week
Pay: **$240,000 – $300,000 base**

> "Build frameworks for LLM-powered agents to use tools and knowledge sources effectively."
> "Invent new agentic architectures and signals to improve reasoning, planning, and personalization."
> "Design and optimize reinforcement learning and fine-tuning approaches."
> "Lead development of scalable evaluation, benchmarking, and optimization loops for agents."

Requirements: 2+ years as Staff/Principal plus 5+ years in the AI/ML industry; Python / Go / Java / C++.

#### JD-EN-5: Replit — Senior Software Engineer, Agent Platform

URL: <https://jobs.reachcapital.com/companies/replit-2/jobs/65431609-agent-platform-engineer>
Location: Foster City, CA hybrid (M/W/F in office)
Pay: **$180K – $260K + equity + bonus**

> "Bridging the gap between the AI team (working on the core Agent logic) and the UX team (crafting delightful Agent experiences)."
> "Tackling complex challenges across the full stack, from browser-based interfaces to high-performance backends to Linux systems engineering."

Requirements: 5+ years of experience with backend services; "experience building AI-powered tooling or products"; "experience with collaborative editing technologies such as OT, CRDTs, or Git"; "experience with event sourcing systems".

#### JD-EN-6: Cognition — Deployed Engineer (Devin)

URL: <https://www.glassdoor.com/job-listing/deployed-engineer-cognition-JV_KO0,17_KE18,27.htm>
Location / pay: SF $98K-$165K / Emeryville $120K-$180K / Austin $82K-$137K / Federal SF $152K-$241K

> "Deployed Engineers (DEs) are customer-facing technical experts who help customers maximize the value they get out of Devin/Windsurf."
> "Work directly with engineers to understand their workflows, identify high-impact use cases, and deploy Devin/Windsurf into real production environments, leading demos and pilots, supporting integrations, and driving adoption of the Cognition platform."

#### JD-EN-7: Vercel — AI Engineer

URL: <https://vercel.com/careers/ai-engineer-5517523004>
SF base: **$192,000 – $288,000**

> "Sit at the intersection of machine learning, product development, and frontend engineering, working closely with product designers, ML scientists, and full-stack engineers to prototype and deploy innovative systems."

Requirements: 5+ years of experience; strong programming in Python and JavaScript/TypeScript; experience working with or fine-tuning LLMs (OpenAI, Hugging Face); solid understanding of modern frontend (React, Tailwind).

#### JD-EN-8: Sierra — Software Engineer, Agent + APX (New Grad)

URL: <https://jobs.ashbyhq.com/Sierra/b7d1dbcd-ca72-472f-b15e-5b4b0f886be0>. APX: <https://jobs.ashbyhq.com/sierra/6a75b530-b7bb-4439-bb67-37b4f2b75b96>

> "Agent Engineers at Sierra design and deliver production-grade AI agents that are central, mission-critical and drive revenue directly to Sierra's growth."
> "APX is an 18-month rotational program that gives new graduates in computer science or a related field the opportunity to work closely with their customers, building AI agents as both an Agent Engineer and an Agent Product Manager."

<a id="s2-2"></a>

### 2.2 LangChain ecosystem frequency counts (2026-04 data)

Source: <https://agentic-engineering-jobs.com/langchain-job-market-2026>

| Skill | Frequency | Notes |
|---|---|---|
| Python | **93.4%** | Default across the board |
| LangChain | 34.3% | All agentic roles |
| AWS | 34.7% | Cloud platform #1 |
| Docker | 32.4% | DevOps baseline |
| GCP | 29.1% | Cloud platform #2 |
| Kubernetes | 27.2% | Container orchestration |
| Azure | 24.4% | Cloud platform #3 |
| LangGraph | 22.1% | Agent orchestration |
| Pinecone | 18.8% | Vector database #1 |
| FastAPI | 17.4% | Python backend |
| TypeScript | 17.4% | Frontend / fullstack |
| **MCP** | **16.9%** | **The single fastest-growing skill of 2026** |
| Weaviate | 16.0% | Vector database #2 |
| LangSmith | 8.5% | Evaluation and observability |

<a id="s2-3"></a>

### 2.3 North America / Europe pay comparison (2026-05)

| Company | Role | Base | Source |
|---|---|---|---|
| Anthropic | FDE Applied AI (US) | $200K-$300K | greenhouse.io/anthropic/jobs/4985877008 |
| Anthropic | Applied AI Engineer, Enterprise Tech (SF/NYC/Seattle) | $200K-$320K | greenhouse.io/anthropic/jobs/5057647008 |
| Anthropic | Applied AI Engineer (London) | £225K-£240K | greenhouse.io/anthropic/jobs/5116274008 |
| OpenAI | Software Engineer (median TC) | $580K (top reported $1.27M) | levels.fyi/companies/openai |
| Google DeepMind | Research Engineer, Gemini Post-Training (NYC) | $215K-$250K base + bonus + equity | startup.jobs/research-engineer-gemini-post-training-nyc-deepmind-5964742 |
| Glean | MLE, AI Assistant + Autonomous Agents | $240K-$300K | greenhouse.io/gleanwork/jobs/4605215005 |
| Vercel | AI Engineer (SF) | $192K-$288K | vercel.com/careers/ai-engineer-5517523004 |
| Replit | Senior SWE, Agent Platform | $180K-$260K + equity + bonus | reachcapital.com/companies/replit-2 |
| Cognition | Deployed Engineer, SF | $98K-$165K | Glassdoor |
| Cognition | Deployed Engineer, Federal SF | $152K-$241K | Glassdoor |
| Mistral | Research Scientist | €80K-€160K + equity | eujobs.co/mistral |
| Industry | Prompt Engineer (entry to senior) | $90K to $250K+ | pecollective.com/blog/prompt-engineering-salary-guide |

LangChain ecosystem: the median maximum for framework-agnostic agent roles is **$290K**; the LangChain-only median is **$210K**; the Senior+ average top is **$264K**.

---

<a id="s3"></a>

## 3. Top 20 recurring topics (China and overseas JDs combined)

| # | Skill / topic | Where it shows up | What it means here |
|---|---|---|---|
| 1 | **Strong Python** (including PyTorch / FastAPI) | Nearly every JD | Default across the board |
| 2 | **Production LLM application experience** ("shipped LLM apps") | Anthropic / OpenAI / Glean / HN listings | A production track record |
| 3 | **LangChain / LangGraph** | Tencent Hunyuan, ByteDance, the Yiwu JD, a listed company's AI Native team, Hunan Linze, the Anthropic ecosystem | Mainstream agent frameworks |
| 4 | **RAG (vector database + retrieval augmentation)** | Ant Group, ByteDance, Tencent Hunyuan, Zhipu, Hunan Linze, Yiwu, Kimi, Glean, Perplexity, Harvey | Extending knowledge beyond the long context window |
| 5 | **Tool calling / function calling / multi-turn dialogue management** | Hunan Linze, Tencent Hunyuan, ByteDance Agent architect | A core agent capability |
| 6 | **Prompt engineering / instruction tuning** | Frontline big-tech high-pay posts, Zhipu, standalone prompt-engineer roles, Anthropic | From prompt engineering to context engineering |
| 7 | **SFT / RLHF / DPO** | Ant Group, Moonshot, StepFun, DeepSeek, ByteDance Doubao, Baidu, OpenAI Applied | Core of the training stack |
| 8 | **Multi-agent / task planning / tool-call orchestration** | ByteDance Agent architect, Tencent Hunyuan Agent, Ant Group PaaS, Sierra, Glean | Required for complex business workflows |
| 9 | **First-author top-conference papers** (NeurIPS / ICML / ACL / CVPR / ICLR / KDD) | ByteDance, Tencent, Alibaba, StepFun, DeepSeek, Zhipu, Kimi, DeepMind, Mistral | A hard screen for algorithm roles |
| 10 | **ACM / ICPC, NOI / IOI, Kaggle contest awards** | ByteDance, StepFun, DeepSeek, Huawei Genius Youth | A plus for algorithm roles |
| 11 | **Docker / K8s / cloud native / microservices** | The Yiwu JD, a listed company's AI Native team, Shanghai video AIGC, Microsoft, Replit | The deployment baseline |
| 12 | **Vector databases** (Faiss / Milvus / Chroma / Pinecone / Weaviate / Qdrant) | Hunan Linze, Yiwu, Glean, RAG-heavy roles | Retrieval infrastructure |
| 13 | **Agent frameworks such as AutoGen / CrewAI / Dify / Coze** | Yiwu, ByteDance mobile OS, Tencent Hunyuan Agent | The domestic ecosystem |
| 14 | **Inference optimization: vLLM / SGLang / TensorRT-LLM** | Inference engineers, MiniMax inference optimization, HN listings | Self-hosted deployment |
| 15 | **Model evaluation / eval / A/B testing** | Tencent Hunyuan evaluation, Zhipu, Glean, Anthropic, the LangSmith team | CI/CD gating |
| 16 | **MCP (Model Context Protocol)** | Named explicitly at Anthropic, implied at Cognition, discussed on Chinese V2EX | **The single fastest-growing skill of 2026** |
| 17 | **AI coding tools: Cursor / Claude Code / vibe coding** | Moonshot, a listed company's AI Native team, the Yiwu JD | Engineering productivity tools |
| 18 | **Vertical domains:** finance / intelligent customer service / outbound calling / smart home / IoT | Ant Group AI Force, Hunan Linze, Harvey (legal) | Delivering into a business |
| 19 | **C++ / CUDA / GPU programming** | Inference-engineer JDs, Huawei LLM algorithms, DeepMind | Infra roles |
| 20 | **Ability to read and reproduce English papers** | Zhipu, Kimi, StepFun, ByteDance Doubao, nearly every research role | A hard requirement for algorithm roles |

---

<a id="s4"></a>

## 4. How role variants differ

<a id="s4-1"></a>

### 4.1 Seven kinds in the China market

| Role variant | Core deliverable | Main requirements | Pay band (Beijing / Shanghai, RMB) | Typical companies |
|---|---|---|---|---|
| LLM algorithm engineer (pretraining / post-training) | The model's own quality | First-author top-conference paper + PyTorch + RLHF/SFT + experience training on a thousand GPUs | New-grad SSP 500-800K annual; experienced hire 1-3M annual | DeepSeek, ByteDance Doubao Seed, Alibaba Tongyi, Kimi, Zhipu, StepFun |
| Agent application algorithm engineer | Agent-workflow quality inside a business scenario | RAG + tool use + multi-agent orchestration + evaluation | 35-60K x 15-18 months | Tencent Hunyuan, Ant Group AI Force, Zhipu, Alibaba Qwen-Agent |
| LLM application / backend engineer | A live service, concurrency, and availability | Python/Go + LangChain + a vector database + Docker/K8s | 15-35K x 13/14 months | Meituan, JD.com, Hunan Linze, Yiwu mid-size firms |
| Prompt engineer (standalone role) | Prompt templates + AIGC outputs | ChatGPT/Claude experience + Excel/SQL/Python | 15-30K x 13 months | Small and mid-size AIGC companies; **this role is shrinking fast** |
| LLM inference optimization / infra engineer | Inference latency / throughput | vLLM/SGLang + CUDA + operator development | 30-60K x 15 months | MiniMax, Huawei Ascend, ByteDance, Alibaba AI Infra |
| AI coding-agent product engineer | An internal line that writes code with Cursor / Claude Code | Full stack + hands-on Cursor/Claude Code + 0-to-1 product sense | 25-60K; some listed companies treat 3-5x efficiency as a hard metric | Listed-company AI Native innovation centers; Cursor/Coze-style startups |
| AIGC (vision / video) engineer | Image / video generation quality | Diffusion + ControlNet + LoRA + ComfyUI workflows | 7-25K at smaller firms; 30K+ at big tech | Tencent Hunyuan AIGC, ComfyUI engineers, video AIGC startups |

<a id="s4-2"></a>

### 4.2 Six kinds in North America / Europe

| Role | Positioning | Typical employers | Pay |
|---|---|---|---|
| **Research Engineer (RE)** | Train models / algorithm research | DeepMind, OpenAI, Anthropic, Mistral | $215K-$300K base + equity |
| **Applied AI Engineer** | Build applications and reference architectures on top of the model | Anthropic, Glean, Vercel | $200K-$320K base |
| **Forward Deployed Engineer (FDE)** | Embed with strategic customers for white-glove deployment | Anthropic, Cognition, Palantir | $200K-$300K base |
| **Deployed / Field Engineer** | Sales-attached customer implementation | Cognition (Devin), Cursor, LangChain | $98K-$241K |
| **MLE / Agent Platform Engineer** | Build agent frameworks, platforms, and evaluation infrastructure | Glean, Replit, Sierra | $180K-$300K base |
| **AI Infra Engineer** | vLLM / inference / serving infrastructure | Modal, Together AI, Replicate | Tied to the company valuation |

---

<a id="s5"></a>

## 5. Overseas vs China

1. **Pay gap:** Base plus equity for an Anthropic or OpenAI Agent engineer in North America commonly converts to an annual package of 2-4 million RMB. The same level at a Chinese big-tech company is 700K-2 million RMB. China's new-grad SSP offers (Pinduoduo 800K, Baidu 700K, ByteDance 550K+) already approach same-year new-grad packages overseas at the top end.

2. **Stack differences:**
   - Overseas JDs mention **Anthropic SDK / OpenAI SDK / MCP / Claude Code** more often.
   - Domestic posts more often name **LangChain / LangGraph / Coze / Dify / Tongyi Qianwen / Wenxin / the DeepSeek API**.
   - **MCP** is only beginning to show up sporadically in domestic JDs, mainly in engineer discussions on V2EX and Zhihu.

3. **Degree preference:**
   - Overseas hiring weighs shipped products more heavily (GitHub, Show HN, demos).
   - Top Chinese companies still use "a top-conference paper plus a big-tech internship" as a hard screen for algorithm roles.
   - Application-engineering roles are converging on both sides.

4. **Work pattern:**
   - Most domestic JDs assume onsite work, an alternating five-day and six-day week (大小周), or a 996 schedule. Zhipu explicitly advertising two-day weekends and a non-crunch culture (双休、不卷) is treated as a selling point.
   - Overseas, hybrid with 3 days in the office is the mainstream. Microsoft AI moved to 4 days a week starting 2026-01.
   - Remote roles show up only at small startups and at companies such as Dify and Corust.

5. **Going overseas / switching tracks:** After Manus moved its headquarters to Singapore, 80 people were laid off in China and 40 stayed in Singapore. This is a representative 2026 case of a Chinese AI Agent startup going overseas in order to survive, and it means English communication is starting to matter for candidates.

6. **Geography and pay:** The gap in base pay between Europe and the United States is 50-100%, but European benefits (healthcare, statutory PTO, parental leave) offset part of it. The visa-sponsorship rate is only 3.3% (LangChain ecosystem data).

---

<a id="s6"></a>

## 6. Practical advice for candidates

1. **An MVP project.** A working LangChain/LangGraph + RAG + tool-calling project, with GitHub commits, is already the minimum bar for an AI Agent engineer at a mid-size company. The Yiwu JD explicitly requires 「提供 GitHub ID，展示 contributions」 (provide a GitHub ID and show contributions).

2. **Algorithm track vs application track.** Algorithm roles are still a dual track of top-conference papers plus Kaggle/ACM. On the application track, put evidence of 3-5x efficiency from writing code with Cursor / Claude Code on the résumé.

3. **Pay negotiation.** For senior roles, anchor the ask to the 60-150K+ RMB/month range in the frontline big-tech high-pay post on V2EX 1186466. For new-grad SP/SSP, anchor the ask to the campus-recruiting tier data in CSDN 156735755.

4. **Leaving the prompt-engineer title.** Standalone prompt-engineer roles are disappearing. Change the title to "LLM application engineer / Agent engineer" and add Python, a vector database, and Docker.

5. **Learn MCP early.** MCP is the single fastest-growing skill of 2026. Getting ahead of it can capture the upside within 6-12 months. Anthropic's Applied AI Engineer posting already lists MCP as a production requirement.

6. **The context-engineering trend.** Watch 上下文工程 / context engineering in how roles are worded. It is still at the discussion stage on the Alibaba Cloud developer community and on Zhihu, and it is likely to become a standard phrase in JDs in the second half of 2026.

7. **An overseas path.** With English communication ability, Anthropic FDE / Applied AI Engineer is currently the strongest senior opening on value: a public posting at $200K-$320K base plus equity. Cognition, Glean, Sierra, Cursor, and Replit are solid mid-stage options.

---

## Limitations & Caveats

1. **BOSS Zhipin, Zhaopin, and Lagou** job-detail pages mostly require login or are rendered dynamically. This WebFetch pass could retrieve only some of them (Hunan Linze, the prompt engineer, and similar postings). The rest were cross-checked against snippets cached by search engines.
2. **Alibaba Tongyi Lab, Baidu Wenxin, and Huawei Pangu** official career pages require login or a slide-to-verify challenge. Related JDs in this report are taken from publicly reposted campus-recruiting pages and from second-hand compilations on CSDN and Zhihu. They are less reliable than the Tencent and ByteDance items that quote the original text.
3. **Top-talent programs such as JD.com TGT, Meituan Beidou, and ByteDance TopSeed** do not state a specific annual salary in the public JD. They appear in the press (Cailian Press, Securities China, Huxiu) only as "a million-RMB annual salary / 2 million / 3 million." Read those figures as reporting, not as JD text.
4. **OpenAI, the April 2026 Hacker News thread, and Sierra's Ashby pages** returned 403 or an empty body to WebFetch. Related content comes from second-hand search snippets and should be treated as reference material, not as first-party original text.
5. **LangChain ecosystem frequency data** comes from a single aggregator (agentic-engineering-jobs.com). It may over-represent the share of LangChain/LangGraph relative to all agent roles. Use it as a directional reference.
6. **Manus and 01.AI** have already contracted sharply or moved out. Their JDs were outside the effective capture range of this pass.

---

## Bibliography

### China market (in citation order)

1. <https://jobs.bytedance.com/experienced/position/7563248386344847621/detail> — ByteDance Doubao large-language-model Agent application architect
2. <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/145331986> — ByteDance Doubao LLM search-augmentation algorithm engineer JD
3. <https://www.liepin.com/job/1978252587.shtml> — ByteDance Agent software architect, mobile OS (70-100K x 15 months)
4. <https://jobs.niuqizp.com/job-vm85LaLa5.html> — Tencent Hunyuan LLM Agent development engineer (Shenzhen)
5. <https://www.monimianshi.com/job/51362> — Tencent Hunyuan LLM evaluation algorithm researcher (Beijing)
6. <https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/135053896> — Tencent Hunyuan AIGC algorithm researcher
7. <https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/137574967> — Zhipu AI Agent algorithm engineer / intern
8. <https://www.superlinear.academy/c/collaborate/kimi-agent-team> — Moonshot Kimi Agent Team
9. <https://blog.csdn.net/m0_59162248/article/details/149967708> — Ant Group AI Force AI-Agent algorithm engineer
10. <https://career.nankai.edu.cn/correcruit/content/id/99919.html> — StepFun Agent algorithm engineer
11. <https://www.chnfund.com/article/AR4bb44e4e-4975-129f-48ed-3a17e29fe9bb> — DeepSeek hiring for 52 roles
12. <https://www.zhaopin.com/jobdetail/CC455023520J40918852615.htm> — Hunan Linze LLM application engineer
13. <https://www.v2ex.com/t/1194601> — A listed company's AI Native frontier innovation center, 5 openings
14. <https://www.v2ex.com/t/1200716> — Yiwu, Zhejiang AI Agent Engineer
15. <https://www.v2ex.com/t/1186466> — Frontline big-tech high pay: pretraining / Agent / multimodal / RL
16. <https://www.liepin.com/job/1959085873.shtml> — Prompt engineer (Beijing, 15-30K)
17. <https://blog.csdn.net/2401_85343303/article/details/156735755> — A full breakdown of 2025 big-tech campus-recruiting pay
18. <https://www.21jingji.com/article/20260417/...> — ByteDance offering nearly 100 million RMB a year to poach a Genius Youth
19. <https://www.stcn.com/article/detail/2602152.html> — Manus's globalization constrained by compliance and code
20. <https://xz.chsi.com.cn/xz/zyts/202505/...> — Why prompt engineers went cold (openings down 70% in 2025)

### North America / Europe

21. <https://job-boards.greenhouse.io/anthropic/jobs/4985877008> — Anthropic FDE, Applied AI
22. <https://job-boards.greenhouse.io/anthropic/jobs/5057647008> — Anthropic Applied AI Engineer, Enterprise Tech
23. <https://job-boards.greenhouse.io/anthropic/jobs/5116274008> — Anthropic Applied AI Engineer, London
24. <https://openai.com/careers/research-engineer-applied-ai-engineering-san-francisco/> — OpenAI Research Engineer, Applied AI
25. <https://www.levels.fyi/companies/openai/salaries/software-engineer> — OpenAI software-engineer pay
26. <https://job-boards.greenhouse.io/gleanwork/jobs/4605215005> — Glean MLE
27. <https://deepmind.google/careers/> — DeepMind careers homepage
28. <https://startup.jobs/research-engineer-gemini-post-training-nyc-deepmind-5964742> — DeepMind Gemini Post-Training, NYC
29. <https://jobs.lever.co/mistral> — Mistral careers
30. <https://sierra.ai/careers> — Sierra careers homepage
31. <https://jobs.ashbyhq.com/Sierra/b7d1dbcd-ca72-472f-b15e-5b4b0f886be0> — Sierra Software Engineer, Agent
32. <https://cursor.com/careers> — Cursor careers homepage
33. <https://cognition.ai/> — Cognition (Devin) careers homepage
34. <https://www.glassdoor.com/job-listing/deployed-engineer-cognition-JV_KO0,17_KE18,27.htm> — Cognition Deployed Engineer
35. <https://replit.com/careers> — Replit careers
36. <https://jobs.reachcapital.com/companies/replit-2/jobs/65431609-agent-platform-engineer> — Replit Senior SWE, Agent Platform
37. <https://vercel.com/careers/ai-engineer-5517523004> — Vercel AI Engineer
38. <https://www.langchain.com/careers> — LangChain careers
39. <https://agentic-engineering-jobs.com/langchain-job-market-2026> — LangChain ecosystem analysis
40. <https://news.ycombinator.com/item?id=47601859> — HN "Who is Hiring? April 2026"
41. <https://www.kore1.com/ai-engineer-salary-guide/> — AI Engineer pay, 2026
42. <https://www.secondtalent.com/resources/most-in-demand-ai-engineering-skills-and-salary-ranges/> — Top 10 in-demand AI skills
43. <https://www.eujobs.co/career-guides/mistral-ai-career-guide> — Mistral career guide
44. <https://pecollective.com/blog/prompt-engineering-salary-guide/> — Prompt engineer pay, 2026
45. <https://www.devopsschool.com/blog/agent-reliability-engineer-role-blueprint-...> — Agent Reliability Engineer blueprint

---

<a id="methodology"></a>

## Methodology Appendix

- **Research date:** 2026-05-02
- **Search tracks:** 26 (14 in Chinese + 12 in English)
- **Core coverage:** BOSS Zhipin / Lagou / 51job / Zhaopin / Nowcoder / V2EX / Liepin / Zhihu / CSDN / ByteDance official careers pages / Greenhouse / Ashby / Glassdoor / Levels.fyi / LangChain ecosystem analysis / Hacker News
- **Agents dispatched:** 2 parallel web-search agents (1 Chinese + 1 English), each making 40-70 WebSearch / WebFetch calls
- **Evidence grades:**
  - **Grade A (first-party original text, verbatim):** Anthropic Greenhouse, Tencent Hunyuan, Hunan Linze, Glean, Vercel, Replit on Reach Capital
  - **Grade B (cached snippets, second-hand):** BOSS Zhipin / Zhaopin (login required), Alibaba Tongyi, Baidu Wenxin, Huawei Pangu
  - **Grade C (press reports, paraphrased):** JD.com TGT, Meituan Beidou, the ByteDance poach reported near 100 million RMB
  - **Grade D (aggregated statistics):** LangChain ecosystem analysis (agentic-engineering-jobs.com)
