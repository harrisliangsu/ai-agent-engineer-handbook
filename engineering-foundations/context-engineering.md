# 上下文工程：含记忆系统与 RAG 全栈选型

**位置**：LLM 应用工程四层演进的第二层
**调研日期**：2026-05-01

---

## Executive Summary

**Context Engineering（上下文工程）** 在 2024 下半年由 Karpathy 在 YC AI School 演讲中正式命名，被 Anthropic [6] 定位为"提示词工程的自然演进"。其核心观察是：在 100K-1M 上下文窗口和 Agent 时代，决定模型表现的不再是"那一句 prompt"，而是 **模型在推理这一刻能看到的全部信息**——system prompt、对话历史、工具定义、检索文档、记忆、规划状态。

本章覆盖：

- **§2.1** 定义与起源（Karpathy + Anthropic + LangChain 三方共识）
- **§2.2** 长上下文物理学：O(N²) 成本、KV cache、Lost-in-the-Middle U 形召回曲线
- **§2.3** RULER 基准与 1M 上下文真相——Gemini 1.5 Pro 单 needle 99.7% 但多事实召回 ~60%；长上下文比 RAG 显著更慢、更贵（按单源 [41] 估算约 30-60× 慢、~1000× 贵；具体倍率随模型和场景差异大）
- **§2.4** RAG 全栈选型——从演进四代到 10 个子节（解析→切分→嵌入→向量库→检索→重排→改写→GraphRAG→Self-RAG/CRAG/Agentic RAG→多模态）
- **§2.5** 记忆系统四强对比：Mem0 / Letta（前 MemGPT）/ Zep（temporal KG）/ Anthropic Memory Tool
- **§2.6** 上下文管理策略（优先级、淘汰、摘要、`/compact`）
- **§2.7** MCP / A2A 协议生态（MCP 月下载量 9700 万）

关键转折：**生产级 LLM 应用的核心难度从"写 prompt"迁移到了"管 context"**——这是 Agent（[03](./agent-engineering.md)）和驾驭工程（[04](./harness-engineering.md)）的前提。

---

## 2.1 定义与起源

"Context Engineering"（上下文工程）这个词在 2024 年下半年开始在社区流传，普遍归功于 Karpathy 在 YC AI School 的 *Software Is Changing (Again)* 演讲 [5]。他给的定义后来被反复引用：

> "In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step."
> （在每一个工业级 LLM 应用里，上下文工程是把恰好正确的信息填进上下文窗口、为下一步推理做准备的精细艺术与科学。）

Anthropic 在 2025 年发布的官方文章 *Effective Context Engineering for AI Agents* [6] 进一步把它定位为"提示词工程的自然演进"：

> 「prompt engineering 关心的是你给模型说的那句话；context engineering 关心的是模型在推理那一刻能看到的所有信息——system prompt、对话历史、工具定义、检索到的文档、记忆、规划状态、约束与策略。」

LangChain 团队在 *Context Engineering for Agents* 博客 [7] 里给了一个常被引用的工程定义：

> "Context Engineering 是设计**动态系统**的学科，这些系统在**正确的时间、以正确的格式，向 LLM 提供正确的信息和工具**，让模型有完成任务所需的一切。"

为什么"提示词工程"不够了？三个根本变化：

1. **窗口扩张**：从 GPT-3.5 的 4K 一路涨到 Gemini 2.5 Pro 的 1M、Claude Opus 4.7 的 1M。能塞进窗口的信息从"几段文字"变成了"一整本书"。决定塞什么变得比写好那一句话重要得多
2. **Agent 兴起**：单轮 prompt 是孤立交易，agent 是持续状态机。每一步推理的 context 都来自上一步的输出 + 工具返回 + 检索 + 记忆，"prompt"已经不是用户写的那句话，而是 agent 框架动态拼装的产物
3. **多模态 + 工具调用**：context 不再只是文本，还有图像、表格、代码、tool schema、retrieved chunks、structured memory、scratchpad。把这些异构信息组织好是工程问题

到 2026 年初，业界共识是：**生产级 LLM 应用的核心难度从"写 prompt"迁移到了"管 context"**。

---

## 2.2 长上下文的物理学

要理解为什么"塞满窗口"是个坏主意，必须先理解上下文的三个物理约束：

### 2.2.1 O(N²) 成本

回到 [prompt-engineering.md](./prompt-engineering.md) §1.1：标准 attention 是 N² 复杂度。上下文从 32K 翻到 128K，注意力计算量是 16 倍而非 4 倍。对推理服务来说这意味着两件事：

- 首 token 延迟（TTFT）上升，因为要先 prefill 整个 prompt
- 显存被 KV cache 占满，并发数下降

### 2.2.2 KV Cache 显存

每个 attention 层在推理时要维护 K 和 V 的缓存（避免重复计算）。**一个 70B 模型在 128K 上下文时，KV cache 占用可达 40+ GB，比模型权重本身还大**。这就是为什么 vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 这类"管 KV cache"的技术成为推理服务器的核心竞争力（详见 [harness-engineering.md](./harness-engineering.md) §3.9.2）。

### 2.2.3 Lost-in-the-Middle

Liu 等人 2024 TACL 论文 *Lost in the Middle: How Language Models Use Long Contexts* [2] 系统验证了一个反直觉现象：把关键信息放在 prompt 开头或结尾，模型能很好地利用；放在中间，模型常常视而不见。**召回率随位置呈 U 型曲线**——开头有 primacy bias、结尾有 recency bias、中间是黑洞。

这条曲线在 GPT-3.5 / GPT-4 / Claude 上都验证过。即便 2025 年的 Gemini 1.5 Pro 在简单的 needle-in-a-haystack 测试上能拿到 99.7% 召回，但当任务变成"在 1M token 里找出 8 个相关事实并综合"，平均召回率掉到 ~60% [41]。

---

## 2.3 RULER 与 1M 上下文的真实能力

NVIDIA 等机构 2024 年发布的 **RULER 基准** [12]（*What's the Real Context Size of Your Long-Context Language Models?*）把"长上下文能力"从 needle-in-a-haystack 的玩具测试升级为"折磨套件"：

- **多种类多数量的 needle**：不只是埋一个事实，是埋多个不同类型
- **多跳推理**：找到 A 才能找到 B
- **聚合任务**：在长文里数总数 / 求平均
- **变量追踪**：跟踪一个变量在 100K token 范围内的多次重新赋值

RULER 暴露出的真相：**几乎所有声称 100K+ 上下文的模型，在 RULER 上的"有效上下文"远小于宣称值**。比如某模型宣称支持 200K，但 RULER 上 50K 后准确率就掉到 80% 以下，128K 时已经接近 random。

这把行业拉回了现实：**long context 不是 RAG 的替代，而是补充**。

### 长上下文 vs RAG 决策框架

Tianpan 在 *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool* [41] 给了一个 2026 主流的决策框架：

| 维度 | 1M 上下文模型 | RAG |
|---|---|---|
| **延迟** | 30-60 倍慢（必须 prefill 整个窗口） | 检索 + 生成两阶段，但生成阶段输入小 |
| **每查询成本** | ~1250 倍贵 | 检索几乎免费，生成成本随相关 chunk 数 |
| **数据更新** | 每次查询都要重传 | 索引更新一次，所有查询受益 |
| **可解释性** | 黑箱 | 检索结果可审计、可引用 |
| **多事实召回** | ~60%（实战） | 配合 reranker 可达 90%+ |
| **真正适合长上下文的场景** | 单次深度分析（"读完这篇 200 页的 PDF 总结洞察"） | 知识库问答、客服、文档检索 |

**实战指引**：90% 的"知识检索"场景应该用 RAG，10% 的"深度分析单文档"才用长上下文。

---

## 2.4 RAG 全栈选型

RAG（Retrieval-Augmented Generation）是上下文工程最成熟的一支，已演进出完整的子学科。2025-2026 年的状态可以用四代演进概括 [43]：

- **第一代：Naive RAG**（2020-2022）—— 检索 → 拼接 → 生成
- **第二代：Advanced RAG**（2023-2024）—— 加 query rewriting / reranking / hybrid search / metadata filter
- **第三代：Modular RAG**（2024）—— indexing / pre-retrieval / retrieval / post-retrieval / generation 各自可独立替换
- **第四代：Agentic RAG**（2025-2026）—— Agent 自主决定何时检索、检索什么、如何评判

下面按 RAG 流水线顺序拆解 2026 年的选型空间。

### 2.4.1 文档解析（Parsing & Layout）

现实世界的文档是 PDF、PPT、扫描件、Excel、HTML 而不是纯文本。这一步常被忽略但影响极大：

- **基础**：`pypdf` / `pdfplumber`（速度快，但表格 / 多栏排版易错）
- **进阶**：`unstructured` / `LlamaParse`（带 layout 模型，能识别标题 / 段落 / 表格 / 列表）
- **OCR**：`PaddleOCR` / `Tesseract` / `EasyOCR`（扫描件、图片）
- **多模态原生**：用 GPT-4o / Gemini / Claude 直接读 PDF 页（vision），适合排版极复杂的文档但成本高
- **专用商用**：Azure Document Intelligence、AWS Textract、Microsoft MarkItDown

最佳实践：**至少跑两套 parser 对比**，再决定用哪个。表格 / 公式必须特别处理（转 markdown / LaTeX）。

### 2.4.2 切分策略（Chunking）

切分策略对召回质量影响远大于嵌入模型的选择：

- **Fixed Size Chunking**：固定 N token，最简单。问题：容易切碎语义单元
- **Recursive Character Splitting**：按段落 → 句子 → 词层级递归切。LangChain 默认。性价比好的 baseline
- **Semantic Chunking**：用 embedding 计算相邻句子相似度，相似度骤降处切分
- **Document-Specific**：Markdown 按 heading 切、code 按 function 切、HTML 按 DOM 切
- **Agentic Chunking**：让 LLM 阅读文档后自己决定怎么切。最贵但最准
- **Late Chunking**（Jina 2024）：先对整文档 embed 得到 token 级 embedding，再切分聚合。在长上下文场景比传统切分召回高 10-15%

通用建议：**chunk size 256-512 token、overlap 50-100 token**，是大多数任务的 sweet spot。

### 2.4.3 嵌入模型（Embedding Models）

2026 年 4 月 MTEB（Massive Text Embedding Benchmark）榜单 [13] 主要选手：

| 模型 | MTEB | 维度 | 价格 / 1M tok | 特点 |
|---|---|---|---|---|
| **OpenAI text-embedding-3-large** | 64.6 | 3072（可截短） | $0.13 | 老牌稳定 |
| **Cohere embed-v4** | 65.2 | 1536 | $0.10 | 当前 MTEB 第一 |
| **Voyage 3 large** | ~66 | 1024/2048 | $0.18 | 平均比 OpenAI 高 ~10% |
| **Google Gemini Embedding 2** | 64+ | 3072 | $0.006 | 多模态原生（文本/图/视频/音频/PDF），100+ 语言 |
| **BGE-M3**（BAAI） | 64+ | 1024 | 免费（开源） | 同模型支持 dense / sparse / multi-vector 三种检索 |
| **Jina v5-text-small** | 71.7（v2 榜） | 512 | 免费（Apache 2.0） | 677M 参数，最佳质量 / 体积比 |

2026 年的两个关键趋势：

1. **Matryoshka Representation Learning（套娃表示）**：同一个 embedding 在不同截断长度（128 / 256 / 512 / 1024 / 3072）都能用，索引时存全长，检索时按需截短。Gemini 2 / Voyage 4 / Cohere v4 / OpenAI text-3-* / Jina v5 / Microsoft Harrier / Nomic v1.5 都支持。可以让"先粗筛后精排"在 embedding 层就实现
2. **多模态原生**：Gemini Embedding 2 把图像 / 视频 / 音频 / PDF 编进同一个语义空间，多模态 RAG 的索引复杂度从"分模态建索引 + 跨模态对齐"降到"统一索引"

中文场景实测：**BGE-M3 是免费第一**（开源 + 多语言强 + 同模型支持稀疏/稠密），**Voyage 3 large** 是付费第一（中文召回普遍比 OpenAI 高 10%+）。

### 2.4.4 向量数据库（Vector Database）

2026 年生产部署的五大主流 [14]：**pgvector / Qdrant / Weaviate / Milvus / LanceDB**。

性能基准（2026，128 维 embedding，1M 向量量级）[14]：

| 数据库 | p99 延迟 | QPS（单节点） | 备注 |
|---|---|---|---|
| Qdrant | 2 ms | 12,000 | Rust 实现，元数据过滤最强 |
| FAISS（in-memory） | 3 ms | 15,000 | 单机性能王，无服务化能力 |
| Milvus | 5 ms | 8,000 | 分布式 + 高可用，2.5 版本混合检索 30x 快于 ES |
| Pinecone | 8 ms | 5,000 | 全托管，运维零负担 |
| Weaviate | 10 ms | 4,000 | 内置 GraphQL 与知识图能力 |

选型决策：

- **已经在用 Postgres** → `pgvector` 0.9 版本性能跟上了，避免运营第二个数据库的复杂度
- **需要复杂元数据过滤**（按时间 / 类目 / 权限） → Qdrant（ACORN 算法解决了 filtered HNSW 的老大难问题）
- **要做向量 + 知识图** → Weaviate
- **企业级 + 海量 + 高可用** → Milvus
- **想要嵌入式（不需要服务）** → LanceDB / Chroma
- **极简快速 prototype** → Chroma
- **想用 Elasticsearch / OpenSearch 已有栈** → 它们自己也支持向量

### 2.4.5 检索：稀疏 + 稠密 + 后期交互

向量检索（稠密）擅长语义相似度但不擅长精确关键词匹配（产品 ID、专有名词、错别字容忍）。**生产级 RAG 几乎都是 hybrid search**：

- **稀疏**：BM25 / SPLADE / BGE-sparse（关键词 + 词频）
- **稠密**：embedding-based vector search（语义）
- **后期交互（late interaction）**：ColBERT / ColBERT-v2，每个 token 单独 embedding，检索时按 token 级 max-similarity 聚合。比单向量 dense 召回高，但索引大 100x。Vespa、Qdrant 已支持

混合策略：用 RRF（Reciprocal Rank Fusion）合并稀疏与稠密的结果列表。简单且无需调参。

**RRF 公式**：

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

其中 `R` 是各路检索器，`r(d)` 是文档 `d` 在该路检索器里的排名（1, 2, 3...），`k` 是平滑常数（默认 60，Cormack 等 2009）。直觉：排名越靠前贡献越大，但因为是倒数，差距收敛快，避免某一路霸榜。

**RRF 调权**（生产实践）：默认 RRF 把所有检索器等权对待。如果想给稠密 / 稀疏不同权重：

```python
def weighted_rrf(results_dict, weights, k=60):
    # results_dict = {"dense": [doc1, doc2, ...], "sparse": [docA, docB, ...]}
    # weights = {"dense": 0.7, "sparse": 0.3}
    scores = defaultdict(float)
    for retriever, docs in results_dict.items():
        w = weights[retriever]
        for rank, doc in enumerate(docs, start=1):
            scores[doc.id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**RRF vs 其他融合**：

| 方法 | 是否需调参 | 跨检索器分数对齐 | 何时用 |
|---|---|---|---|
| **RRF** | 否（k=60 几乎万能） | 用 rank 不用 score，天然对齐 | 默认首选 |
| **CombSUM / Linear** | 是（要归一化分数） | 难（BM25 vs cosine 量纲不同） | 不推荐，除非已有校准 |
| **WAND（Weighted AND）** | 中 | 在 sparse 内部用，不跨融合 | Lucene / Elasticsearch 加速倒排 |
| **Convex Combination + ML** | 高（要 learning-to-rank） | 训练对齐 | 大规模、有点击数据时最优 |

实证：**hybrid search + reranker 比 semantic-only 提升 +9.3 个百分点 MRR**，配合 Cohere Rerank 准确率可以到 90%+。

### 2.4.6 重排（Reranker）

第一阶段检索（embedding / BM25）召回 top-100 后，用更贵但更准的 cross-encoder 模型对 query-doc 对重排，取 top-5 给 LLM。这一步的 ROI 极高：

| Reranker | 类型 | ELO | 备注 |
|---|---|---|---|
| **Zerank 2** | API | 1638 | 当前榜首 |
| **Cohere Rerank v4 Pro** | API | 1629 | 老牌 |
| **Cohere Rerank 3 Nimble** | API | — | 速度优化版本 |
| **BGE-reranker-v2-m3** | 自托管 | — | 开源，效果接近 Cohere |
| **FlashRank** | 自托管 | — | 极轻量，毫秒级 |
| **ColBERT v2** | 自托管 | — | 后期交互架构 |
| **RankLLM**（用 LLM 直接 rerank） | LLM | — | 最贵但解释性最好 |

**top rerankers 比纯 embedding 提升 15-40% 精确率**。如果只允许加一项 RAG 优化，加 reranker 通常 ROI 第一。

### 2.4.7 查询改写（Query Rewriting / Expansion）

用户问的"今天天气咋样"和文档里写的"current weather conditions"语义相同但词汇不重叠。Query rewriting 在检索前先对 query 做处理：

- **HyDE（Hypothetical Document Embeddings）**：先让 LLM 假想"如果有一篇答案文档会写什么"，用这段假想文本去检索（不是用原 query）
- **Query2Doc**（Microsoft 2023）：HyDE 的变体，让 LLM 直接生成"包含答案的伪文档段落"作为查询扩展
- **Multi-Query / Query Expansion**：让 LLM 把原 query 改写成 3-5 个不同表达，分别检索后合并
- **Step-Back Prompting**（Zheng 等 2024）：先把 query 抽象成更宏观的问题（"Python 3.12 的 GIL 改了什么" → "GIL 是什么、它如何工作"），用抽象问题的检索结果做铺垫
- **Sub-Query Decomposition**：把复合问题拆成多个子问题分别检索（"对比 A 和 B" → ["A 的特点"、"B 的特点"]）
- **HyDE + Multi-Query 组合** 在 BEIR 等检索基准上是当前最强 baseline 之一

**怎么选**（按 query 形态决策）：

| query 形态 | 推荐方法 | 原因 |
|---|---|---|
| 短查询 / 关键词缺失 | HyDE / Query2Doc | 短 query 与文档 chunk 长度不匹配，扩成假文档可拉近 embedding 距离 |
| 多义词 / 同义词问题 | Multi-Query | 改写覆盖不同措辞，提升召回 |
| 涉及背景知识的窄问题 | Step-Back | 先检索宏观知识再聚焦细节 |
| 复合 / 比较 / 多跳 | Sub-Query Decomposition | 单次检索无法覆盖多个子主题 |
| 通用对话 + 代码 / 数字 | 不要重写 | LLM 重写可能改变关键术语 / 数字，反而召回掉 |

**陷阱**：
1. Query rewriting 增加 1 次额外 LLM 调用（latency +500-1500 ms），高 QPS 场景需评估
2. 如果原 query 已经精确（"OpenAI 2025 年 Q3 收入"），重写可能引入噪声，建议加一个 "is_specific" 分类先判断
3. HyDE 在 **冷门 / 长尾 / 专业领域** 反而效果差 —— LLM 假想的伪文档与真实文档措辞偏差大

### 2.4.8 GraphRAG

Microsoft GraphRAG（2024 开源）和 LightRAG / Nano-GraphRAG 等社区变体把检索从"向量相似"扩展到"知识图遍历"：

- **构建阶段**：用 LLM 抽取文档里的实体和关系，建一张知识图
- **检索阶段**：把 query 映射到图上的实体，做局部子图查询 + summary
- **优势**：能回答"主题级"问题（"我们所有供应商合同里的合规风险有哪些"），传统 RAG 因为这种问题没有局部相似 chunk 而失败
- **代价**：图构建非常贵（LLM 把每段文档抽实体），适合静态高价值知识库

GraphRAG 在某些场景把检索精度推到 99%（金融合规、法律、医疗等关系密集的领域）[43]。

### 2.4.9 Self-RAG / CRAG / Agentic RAG

- **Self-RAG**（Asai 2023）：让模型在生成时自己决定**何时**检索、**评估**检索结果是否相关、**批判**自己的输出是否有依据
- **CRAG（Corrective RAG）**：检索后对 chunk 打"高/中/低置信度"标签，低置信度时触发 web search 兜底
- **Agentic RAG**：把上面所有能力包进一个 agent loop，让 agent 在"想 → 检索 → 评判 → 重写 query → 重新检索 → 综合"里循环直到自信为止。LangGraph 是实现 Agentic RAG 的主流框架

**三者横向对比**：

| 维度 | Self-RAG | CRAG | Agentic RAG |
|---|---|---|---|
| **决定何时检索** | 模型自己（特殊 token `[Retrieve]`） | 始终检索 | Agent 控制器决定 |
| **检索结果评判** | 模型自己（`[IsRel]` 反思 token） | 独立 retrieval evaluator（小模型，分高/中/低） | LLM judge 或工具 |
| **失败兜底** | 不再检索 + 提示不确定 | 触发 web search 重检索 | 改写 query / 换数据源 / 多轮迭代 |
| **训练成本** | 高（需对反思 token 做 SFT） | 低（plug-and-play） | 0（运行时编排） |
| **延迟** | 中（一次推理多输出 token） | 高（始终检索 + 评估 + 可能 web） | 最高（可能多轮 loop） |
| **适用** | 自带知识充足的通用问答 | 知识库时新性差、需要 web 兜底 | 多跳推理、研究型问答、Deep Research |
| **2026 主流落地** | OpenAI gpt-4o web search 模式 | Perplexity / You.com 内部架构 | Claude / Cursor / Devin 的研究 agent |

实证：Self-RAG 在生产部署中能减少 25-40% 的"不必要检索"——因为模型学会了对自己已有知识的事情不去检索。Agentic RAG 在多跳问答（HotpotQA / MuSiQue）准确率比 vanilla RAG +15-25 个点，但 token 成本翻 3-10 倍。

详见 [agent-engineering.md](./agent-engineering.md) §2 范式演进、§5 框架对比。

### 2.4.10 多模态 RAG

图像 + 文本混合的索引方案：

- **Late fusion**：分别对文本和图像 embed，检索时分别检索后合并
- **Early fusion**：用 CLIP / Gemini Embedding 等多模态 embedding，把图文嵌入同一空间
- **VLM-as-extractor**：用 GPT-4o / Claude 3.5 Sonnet 把图片转成详细描述，再走文本 RAG（成本最高但召回最稳）
- **ColPali / ColQwen2**（2024）：直接对**整页 PDF 截图**做 late-interaction embedding，跳过 OCR / 解析，在表格 / 图表 / 多栏排版的文档检索准确率比传统 OCR + 文本 RAG +20-40%

**Layout-aware chunking**（关键工程点，被低估）：

传统 RecursiveCharacterTextSplitter 对 PDF / Word 这类有版式的文档效果差 —— 表格被劈成两半、figure caption 与图分离、目录与正文混合。生产级方案：

| 工具 | 路线 | 强项 |
|---|---|---|
| **Unstructured.io** | 规则 + ML 元素分类 | 通用，HTML/PDF/PPT/Email 全覆盖 |
| **LlamaParse** | LlamaIndex 自家 LLM 解析 | Markdown 输出，表格保真度高 |
| **PyMuPDF + pdfplumber** | 底层 PDF SDK | 自定义 layout 规则，保留 bbox 坐标 |
| **Docling**（IBM 2024） | 开源结构化文档解析 | 表格 / 公式 / 图像 caption 关联 |
| **MinerU**（OpenDataLab 2024） | 中文 PDF 友好 | 中文学术论文 / 教材效果好 |

**多模态 RAG 选型决策**：
- 文档以 **图表 / 截图 / 扫描件** 为主 → ColPali / ColQwen2，跳过 OCR
- 文档以 **表格密集型 PDF**（财报 / 规范）为主 → Docling / LlamaParse + Markdown chunking
- 文档以 **混排 PPT / 营销材料** 为主 → VLM-as-extractor（让 Claude / GPT-4o 描述每页）
- 输入是 **用户上传图 + 文本提问** → Gemini Embedding 2 / Cohere Embed v4 多模态原生

**陷阱**：多模态 embedding 在跨模态检索（用文字找图）效果远好于跨模态对齐（图与文字到底说的是不是一件事），生成阶段仍需 VLM 二次确认相关性。

### 2.4.11 完整 RAG pipeline 代码骨架

把上面 10 个子节串起来，一个生产级 RAG pipeline 的最小可运行骨架：

```python
# 1. 文档解析
from llama_parse import LlamaParse
parser = LlamaParse(result_type="markdown")
docs = parser.load_data("./knowledge_base/")

# 2. 切分
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80,
    separators=["\n\n", "\n", "。", "!", "?", ".", " "]
)
chunks = splitter.split_documents(docs)

# 3. 嵌入 + 入向量库
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
vectorstore = QdrantVectorStore.from_documents(
    chunks, embeddings, url="http://localhost:6333", collection_name="kb"
)

# 4. 混合检索（dense + sparse + RRF）
from langchain.retrievers import EnsembleRetriever, BM25Retriever
bm25_retriever = BM25Retriever.from_documents(chunks); bm25_retriever.k = 20
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6]
)

# 5. 查询改写（multi-query + step-back）
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
multi_q_retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=llm)

# 6. 重排（Cohere Rerank 或 BGE-reranker）
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
reranker = CohereRerank(model="rerank-v3.5", top_n=5)
final_retriever = ContextualCompressionRetriever(
    base_retriever=multi_q_retriever, base_compressor=reranker
)

# 7. 生成（带 citation）
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的助手。仅根据给定 context 回答，每条事实必须标注 [doc_id]。"),
    ("human", "Context:\n{context}\n\n问题: {input}"),
])
combine_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(final_retriever, combine_chain)

answer = rag_chain.invoke({"input": "Transformer 的 attention 复杂度是多少？"})
```

这个骨架覆盖了 RAG 的 7 个核心步骤。生产化还需要加：

- **eval 闭环**：RAGAS 跑 faithfulness / context_precision / answer_relevancy
- **观测**：Langfuse 接入捕获每次检索 + 生成的 trace
- **缓存**：embedding cache + LLM 响应 cache
- **降级**：检索失败 fallback、重排失败 fallback、LLM 失败 fallback

### 2.4.12 GitHub 系统化学习资源（RAG 专题）

| 资源 | 类型 | 推荐用法 |
|---|---|---|
| [`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques) | 高级技术 notebook 集 | 30+ 种 RAG 高级技术每个都有可跑 notebook，最全 |
| [`infiniflow/ragflow`](https://github.com/infiniflow/ragflow) | 开源 RAG 引擎 | 想要"开箱即用"的企业级 RAG 平台 |
| [`HKUDS/LightRAG`](https://github.com/HKUDS/LightRAG) | EMNLP 2025 论文 + 实现 | 简洁高效的 RAG 范式参考 |
| [`Danielskry/Awesome-RAG`](https://github.com/Danielskry/Awesome-RAG) | awesome list | 找特定主题资源时的入口 |
| [`lehoanglong95/rag-all-in-one`](https://github.com/lehoanglong95/rag-all-in-one) | 全栈指南 | 想要按组件对照工具时 |
| [`pguso/rag-from-scratch`](https://github.com/pguso/rag-from-scratch) | 从零实现 | 想搞清"黑盒里到底发生了什么" |
| [`microsoft/graphrag`](https://github.com/microsoft/graphrag) | GraphRAG 官方 | 实体关系图 RAG 的参考实现 |
| [`abhinav-kimothi/A-Simple-Guide-to-RAG`](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG) | 入门书配套 | 完全新手入门 |

---

## 2.5 记忆系统：完整工程指南

如果 RAG 解决"**外部知识库**怎么塞进 context"，记忆系统解决"**agent 自身经验**怎么塞进 context"。这两件事在工程上有相似的部分（都是检索 + 注入 prompt），但有本质差异：

| 维度 | RAG | 记忆系统 |
|---|---|---|
| 数据来源 | 外部静态文档（公司 wiki、PDF） | 历史对话、agent 行为、用户偏好 |
| 写入触发 | 索引时一次性 | 每次交互都可能触发增量写入 |
| 数据生命周期 | 跟着源文档走 | 有独立的"过期 / 淘汰 / 摘要"逻辑 |
| 隔离粒度 | 全局或部门级 | 必须 per-user / per-agent / per-session |
| 时效性 | 通常分钟到天 | 可能秒级（"用户刚才说的"） |
| 一致性 | 强（一份 ground truth） | 弱（事实会变、有矛盾） |

### 2.5.1 为什么仅靠 context window 不够

"上下文窗口已经 1 M 了，还需要记忆系统吗？" 是个真实问题。三个理由：

1. **成本与延迟**：1M context 一次推理比同等任务的 RAG 显著更慢更贵（按 [41] 单一来源估算约 30-60× 慢、~1000× 贵，**具体倍率因模型和任务差异很大，建议在自家场景实测**）
2. **Lost-in-the-Middle**：把所有历史塞进 context，关键事实被中间黑洞吞掉（U 形召回）
3. **跨 session 持久化**：context window 是单次推理生命周期，session 结束即销毁；记忆系统要跨天、跨周、跨设备

记忆系统本质上是 **"把过去的全部经验变成一个可检索的小数据库，每次推理只检索最相关的几条注入 context"**——把"全部输入到模型"换成"模型按需检索"。

### 2.5.2 记忆类型的工程映射

借自认知心理学的分类，每一类对应不同的工程实现：

| 记忆类型 | 认知含义 | 工程实现 | 存储后端 | 写入时机 | 检索时机 |
|---|---|---|---|---|---|
| **Working Memory（工作）** | 当前任务的"草稿纸" | 当前 context window | LLM 内部 KV cache | 每个 token | 每个 token |
| **Short-term（短期）** | 最近 N 轮对话 | 滚动 message buffer | 内存 / Redis | 每轮 | 每轮全注入 |
| **Episodic（情景）** | "什么时候发生了什么" | 带时间戳的对话片段 | 向量库 + 时间索引 | 会话结束 / 重要事件 | 按主题 + 时间窗 |
| **Semantic（语义）** | 关于世界 / 用户的稳定事实 | KV pair 或图节点 | 图库 / KV 库 / SQL | 抽取出新事实时 | 按实体名 |
| **Procedural（程序）** | "这类任务怎么做" | 可执行 skill / 流程脚本 | 文件系统 / skill registry | 任务成功后总结 | 按任务类型 |
| **Long-term（长期）** | 跨 session 的所有上述类型 | 上面几种的合集 | 通常多种后端组合 | 异步 / 后台 worker | 按需 |

**关键洞察**：一个生产级 agent 的"记忆"通常**不是单一系统**，而是上面 6 类的不同子集组合。Letta（MemGPT）把短期 + 语义放 in-context（Core Memory），把情景放向量库（Recall Memory），把外部知识放另一个向量库（Archival Memory）——这是 OS-inspired 设计的经典三层。

### 2.5.3 记忆系统的核心架构组件

任何记忆系统拆开都是这 5 个组件：

```
                    ┌──────────────────────────┐
                    │  Conversation / Action   │
                    │  (raw events)            │
                    └────────────┬─────────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  ① Extractor（抽取器）                    │
           │  LLM 把原始事件抽成结构化"记忆事实"       │
           │  e.g. "用户偏好暗色主题"                  │
           └─────────────────────┬──────────────────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  ② Writer（写入器）                       │
           │  去重 / merge / conflict resolution        │
           │  embed + 入库                             │
           └─────────────────────┬──────────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Storage Backend  │
                       │  vector / graph / │
                       │  KV / SQL / file  │
                       └─────────┬─────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  ③ Retriever（检索器）                   │
           │  query → top-K 相关记忆                  │
           │  hybrid: semantic + keyword + graph       │
           └─────────────────────┬──────────────────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  ④ Forgetter（遗忘器）                   │
           │  按 LRU / 重要性 / 时间衰减 淘汰          │
           └─────────────────────┬──────────────────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  ⑤ Compressor（压缩器）                  │
           │  老记忆摘要、冲突合并、知识图重建          │
           └────────────────────────────────────────────┘
```

每个组件都有独立的工程决策点。下面 §2.5.4-2.5.8 逐个展开。

### 2.5.4 存储后端选型

记忆系统的存储后端不是"选一个"而是"选哪几个组合"。常见选项：

| 后端 | 强项 | 弱项 | 典型用法 |
|---|---|---|---|
| **向量库**（Qdrant / Weaviate / Pinecone / pgvector） | 语义相似检索 | 不擅长精确匹配 / 关系查询 | 情景记忆主存储 |
| **图库**（Neo4j / FalkorDB / Memgraph） | 实体关系 / 多跳查询 / 时序 | 部署复杂 / 检索慢 | 语义记忆 + 关系 + 时序 |
| **KV 库**（Redis / DynamoDB） | 极低延迟 / 简单 | 无语义检索 | 短期 working memory + 缓存 |
| **关系型**（Postgres + pgvector / Tiger Data） | 事务性 / SQL filter / 一站式 | 大规模向量索引调优麻烦 | 中小规模一体化 |
| **文件系统**（Anthropic Memory Tool） | 简单 / 可审计 / agent 自管 | 检索能力依赖 agent 自己 grep | Claude 生态原生 |
| **专用记忆 SaaS**（Mem0 Cloud / Zep Cloud / Supermemory） | 零运维 | 锁定 / 数据出域 | 创业 MVP |

#### 选型决策树

```
你的记忆主体是什么？
│
├─ 主要是用户偏好 / 实体事实 → 语义记忆为主
│   ├─ 关系密集（"A 是 B 的同事"、"X 公司 2024 收购 Y"）→ Zep（Graphiti 图）
│   └─ 关系简单 → Mem0（vector + 可选 graph）
│
├─ 主要是对话历史 + 长跑任务 → 情景记忆为主 + agent 自管
│   └─ → Letta（Core + Recall + Archival 三层）
│
├─ 主要是 agent 行为可审计 / 可回滚
│   └─ → Anthropic Memory Tool（filesystem，可 git 管理）
│
└─ 已用 Redis / Postgres，不想加新栈
    ├─ 用 Redis Agent Memory Server
    └─ 用 Postgres + pgvector + Tiger
```

#### 实测性能（2026）

- **pgvectorscale**: 50 M 向量上 471 QPS @ 99% recall（来自 Tiger Data 基准）
- **Redis vector**: p99 < 5 ms（in-memory 优势）
- **Qdrant**: p99 ~2 ms，元数据 filter 强
- **Neo4j 图遍历**: 多跳查询毫秒级，但全量语义检索不如纯 vector

### 2.5.5 写入流程：抽取 + 去重 + 关联

记忆**不是简单地把对话原文存下来**，而是经过 LLM 抽取的"事实"。Mem0 v2 的写入算法（论文 [19]）拆解：

```
1. Retrieve 相关记忆（top-10 semantically similar）
2. LLM 单次调用抽取新事实（Add-only, 不 Update / Delete）
   └─ Prompt: "Given new conversation + existing memories,
              extract NEW facts that should be remembered.
              Return JSON list."
3. Embed + MD5 hash 去重
4. 实体抽取 + 链接（用于 graph memory）
5. 批量写入 vector store + entity collection
```

**关键设计选择**：

- **Add-only vs Update-in-place**：Mem0 选 Add-only（新事实和老事实并存，靠检索器过滤过期的）；Zep 选 invalidate（老事实标记为 invalid，保留历史）；Letta 让 agent 自己 decide（用 `core_memory_replace` 工具）
- **抽取触发时机**：每轮 / N 轮一次 / 会话结束 / 重要事件触发。Mem0 默认每轮，但可配
- **去重粒度**：完全字符串一致用 MD5（Mem0），语义近似用 embedding 距离（Letta）

写入伪代码（自实现版）：

```python
def write_memory(user_id, conversation_chunk):
    # 1. 检索相关历史
    relevant = vector_store.search(
        embed(conversation_chunk), filter={"user_id": user_id}, k=10
    )

    # 2. LLM 抽取
    extracted = llm.extract_facts(
        prompt=f"对话：{conversation_chunk}\n现有记忆：{relevant}\n新事实？",
        schema={"facts": [{"text": str, "category": str, "confidence": float}]},
    )

    # 3. 过滤低置信度
    facts = [f for f in extracted["facts"] if f["confidence"] > 0.7]

    # 4. 去重
    new_facts = []
    for f in facts:
        h = hashlib.md5(f["text"].encode()).hexdigest()
        if not vector_store.exists(hash=h, user_id=user_id):
            new_facts.append({**f, "hash": h, "user_id": user_id, "ts": now()})

    # 5. 写入
    vector_store.add_batch(new_facts)
```

### 2.5.6 检索流程：多信号融合

仅用 embedding 检索召回会漏，2026 年生产级记忆系统都用 **multi-signal fusion**——和 RAG 的混合检索是同一思路。Mem0 v2 的检索：

```
Query → preprocess（lemmatize + entity 抽取）
      ↓
   ┌──┴──────────────────────────┬────────────────────────┐
   ▼                             ▼                        ▼
Semantic（vector cosine）    BM25 keyword             Entity match
   │                             │                        │
   └──────────────┬──────────────┴────────────────────────┘
                  ▼
            Score Fusion (RRF or weighted)
                  ▼
           Top-K Memories → 注入 system prompt
```

**进阶技术**：

- **个性化重排**：把"被检索后是否被 LLM 实际引用"作为反馈信号训练 reranker
- **时间衰减**：检索分数 × `exp(-λ·age)`，新记忆优先
- **重要性加权**：写入时让 LLM 给每条记忆打 importance 分（0-1），检索时加权
- **冷启动 fallback**：用户刚开始用、记忆为空时返回空而不是噪声

### 2.5.7 淘汰策略：6 种 Forgetting Policies

记忆爆炸是真实问题。MaRS 架构（arXiv 2603.07670）形式化了 6 种主流淘汰策略：

| 策略 | 公式 / 规则 | 适用 |
|---|---|---|
| **FIFO** | 按插入时间 | 最简单，几乎不用（删了重要老事实） |
| **LRU** | 按最后访问时间 | desktop agent 默认起点 |
| **Priority Decay** | `score = w₁·prior + w₂·recency + w₃·frequency` | 通用最佳 baseline |
| **Reflection-Summary** | 老的 N 条 → LLM 摘要成 1 条 | 配合 LRU 用，长跑 agent 必备 |
| **Random-Drop** | 随机删 | 仅作 baseline 对比 |
| **Hybrid** | LRU + Priority Decay + Summary 联用 | 生产推荐 |

**实战配方**（Letta 与 Mem0 都接近这个思路）：

```python
def evict_if_needed(memories, max_count=10000):
    if len(memories) <= max_count:
        return memories

    # 1. 计算每条记忆的"保留分数"
    for m in memories:
        m.keep_score = (
            m.importance * 0.4              # 写入时打的重要性
            + recency_score(m.last_access) * 0.3   # 越近越高
            + access_frequency(m) * 0.2     # 被引用次数
            - age_penalty(m.created) * 0.1  # 越老越低
        )

    # 2. 按分数排序，淘汰末尾 N 条
    memories.sort(key=lambda m: m.keep_score, reverse=True)
    to_evict = memories[max_count:]

    # 3. 老记忆不直接删，先尝试摘要
    summary_chunks = group_by_topic(to_evict, n=10)  # 10 条一组
    for chunk in summary_chunks:
        summary = llm.summarize(chunk)
        memories.append(Memory(
            text=summary, importance=avg([m.importance for m in chunk]),
            ts=min([m.created for m in chunk]),
            sources=[m.id for m in chunk],
        ))

    # 4. 真正删除原始 N 条
    delete_batch([m.id for m in to_evict])

    return memories
```

**经验法则**：

- desktop agent / 简单 chatbot → 纯 LRU 起步
- 长跑 agent / 多用户 SaaS → Priority Decay
- 需要追溯 / 法务场景 → Hybrid + 摘要不删原始（archive 到冷存储）

### 2.5.8 四大主流系统深度剖析

#### A. Mem0 — 通用记忆层（最易接入）

**架构**：5 个支柱（论文 [19] 与 Mem0 docs 总结）：

1. **LLM-Powered Fact Extraction**：单 LLM 调用把对话转结构化事实
2. **Vector Storage**：默认 Qdrant，可换 24+ 后端（Chroma / Pinecone / pgvector / MongoDB / Weaviate / OpenSearch / 等）
3. **Graph Storage**（可选）：Neo4j / Neptune / FalkorDB / Memgraph，存实体关系
4. **Hybrid Retrieval**：semantic + BM25 + entity matching 并行打分 + RRF 融合
5. **Production Infrastructure**：批处理、MD5 去重、用户/会话/agent 三层 hierarchy

> 注：上述 5 项是按工程职责整理的检查清单，不是 Mem0 / Letta / Zep 任何一家的官方架构图。Mem0 论文实际描述的是 Add/Update/Delete/NoOp 四种操作 + LLM 抽取 pipeline；Letta 用 Core/Recall/Archival 三层；Zep 用 Episodic/Entity/Edge 三类节点。

**性能**（v2 算法，按 Mem0 论文 arXiv:2504.19413 报告口径）：在 LoCoMo / LongMemEval 等长对话记忆 benchmark 上显著优于 RAG / full-context baseline（具体提升幅度按子任务和指标维度差异较大，详见原论文表 4-5）；抽取延迟降一半。

**完整代码示例**：

```python
from mem0 import Memory

# 初始化（用 Qdrant + Neo4j + OpenAI）
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
    "graph_store": {  # 可选
        "provider": "neo4j",
        "config": {"url": "bolt://localhost:7687",
                   "username": "neo4j", "password": "..."},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
}
m = Memory.from_config(config)

# 写入：从对话自动抽事实
m.add(
    messages=[
        {"role": "user", "content": "我是 Liang，住北京，做 AI 工程师，我喜欢冷咖啡"},
        {"role": "assistant", "content": "记下了"},
    ],
    user_id="liang_001",
)
# Mem0 自动：提取 ["用户名是 Liang", "住北京", "职业 AI 工程师", "偏好冷咖啡"]

# 检索
memories = m.search(query="他喝什么咖啡？", user_id="liang_001", limit=3)
# [{"memory": "用户偏好冷咖啡", "score": 0.89}, ...]

# 注入下次 prompt
context = "\n".join([f"- {m['memory']}" for m in memories])
prompt = f"已知用户信息：\n{context}\n\n用户：今天给我推荐什么饮料？"
```

**AWS 推荐生产架构**：Mem0 + ElastiCache for Valkey（短期 + 缓存）+ Neptune Analytics（图）+ OpenSearch（向量）。

**适合**：通用 chatbot、客服、个人助手——"5 行代码加记忆"的场景。

#### B. Letta（前 MemGPT）— OS-Inspired，Agent 自管

**架构**：把 LLM 当 OS，记忆分三层（核心创新）：

| 层 | 名称 | 位置 | 容量 | 谁能改 |
|---|---|---|---|---|
| L1 | **Core Memory**（核心） | **In-context**（system prompt 里） | 固定（如 4 KB） | agent 自己用 tool |
| L2 | **Recall Memory**（回忆） | 外部 DB | 完整对话历史 | 自动 |
| L3 | **Archival Memory**（档案） | 向量库 | 无限 | agent 自己用 tool |

**Core Memory** 是关键创新——它**不是数据库**而是**system prompt 的一部分**。Agent 可以用 `core_memory_append` / `core_memory_replace` 工具像编辑文件一样改它，下一次推理这些内容自动出现在 context 头部。这等价于让 agent **"持续编辑自己的人设"**。

**6 个内置工具**（agent 自管内存）：

| 工具 | 作用 |
|---|---|
| `send_message` | 给用户发消息 |
| `core_memory_append` | 往 in-context 内存追加新 fact |
| `core_memory_replace` | 替换 in-context 内存（修正错误 / 更新偏好） |
| `conversation_search` | 搜历史对话（recall memory） |
| `archival_memory_insert` | 写入长期档案 |
| `archival_memory_search` | 检索长期档案 |

**完整代码**：

```python
from letta_client import Letta

client = Letta(base_url="http://localhost:8283")  # 自托管 server

# 创建 agent，定义 core memory blocks
agent = client.agents.create(
    name="my_assistant",
    memory_blocks=[
        {"label": "human", "value": "用户的名字、偏好、当前正在做的事"},
        {"label": "persona", "value": "你是一个简洁、礼貌的助理"},
    ],
    model="anthropic/claude-sonnet-4-6",
    embedding="openai/text-embedding-3-small",
)

# 对话（多轮，每次 Letta 自动管理 context）
response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "我叫 Liang，下周要面试"}],
)
# Letta 内部：agent 调用 core_memory_append("human", "用户叫 Liang，下周面试")

# 关掉、重启、明天继续——agent 还记得 Liang 与他的面试
response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "帮我准备一下"}],
)
# agent 看 core memory → 知道 Liang 要面试 → 自动反馈
```

**适合**：长跑 agent（数天 / 数周）、需要"持久身份"的产品（个性化助理 / AI 教练）、Anthropic Memory Tool 之外的 OSS 替代。

#### C. Zep（Graphiti）— Temporal Knowledge Graph

**核心创新**：每条事实带 **双时间轴**：

- **Event Time T**：这条事实在现实世界何时为真（"Alice 2023-01 加入 TechCorp"）
- **Ingestion Time T'**：Zep 何时学到这条事实（对话发生在 2024-03）

**架构**：3 类节点 + 边

```
┌────────────────┐  rel  ┌────────────────┐
│  Entity Node   │◀─────▶│  Entity Node   │
│  Alice         │  edge │  TechCorp      │
└───────┬────────┘       └────────────────┘
        │ source                    ▲
        ▼                           │ joined
┌────────────────┐                  │
│ Episodic Node  │──────────────────┘
│ "对话片段"     │  derives
│ T=2024-03-15   │
└────────────────┘
```

**关键能力**：当事实变化时（"Alice 2024-06 离职"），Zep **不删除**老事实，而是给老 edge 标 `valid_to=2024-06`，新 edge 标 `valid_from=2024-06`。查询时可问：

- "Alice 现在在哪家公司？" → 查 `valid_to=NULL` 的 edge
- "Alice 2024-04 在哪？" → 查 `valid_from <= 2024-04 <= valid_to` 的 edge
- "TechCorp 历史所有员工？" → 不限时间窗的 edge

**Hybrid 检索**：semantic embedding + BM25 + 直接图遍历——**无需 LLM 调用**，纯索引和图查询，延迟 100-500 ms。

**最小代码**（Graphiti SDK）：

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

g = Graphiti(uri="bolt://localhost:7687", user="neo4j", password="...")
await g.build_indices_and_constraints()

# 写入（episode = 一段对话或事件）
await g.add_episode(
    name="meeting_2026_03",
    episode_body="Alice 离开 TechCorp，加入 Anthropic 做 research engineer",
    source=EpisodeType.message,
    reference_time=datetime.now(),
)
# Graphiti 自动：抽 entities → 建 nodes → 抽 relations → 建 edges → invalid 旧 edges

# 查询
results = await g.search(query="Alice 现在在哪工作？", limit=5)
# [{"fact": "Alice 现在在 Anthropic", "valid_from": "2026-03", "valid_to": None}, ...]
```

**适合**：用户状态会变的场景——CRM、客服、医疗病史、合规审计、法律事实库。

#### D. Anthropic Memory Tool — Filesystem-as-Memory

**2026-04-23 公测发布**。设计哲学：**把记忆当文件**，让 Claude 用熟悉的 `bash` + `code execution` 工具读写。

**架构**：

```
Claude Agent
    │
    ├─ tool: bash (read/write/grep files)
    ├─ tool: code execution
    └─ tool: memory (mount filesystem)
                │
                ▼
        ┌───────────────────┐
        │  /memories/       │
        │   ├ liang.md      │ ← user profile
        │   ├ tasks/        │
        │   │   ├ today.md  │
        │   │   └ done.md   │
        │   └ playbooks/    │
        │       └ deploy.md │ ← procedural
        └───────────────────┘
            (any backend:
             local fs, S3,
             encrypted, ...)
```

**完整代码**（Python SDK，public beta）：

```python
from anthropic import Anthropic
from anthropic.lib.tools import BetaAbstractMemoryTool

# 选项 1：默认本地 filesystem 后端
client = Anthropic()
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[
        {"type": "memory_2026_04_23", "name": "memory"},
        {"type": "bash_2026_04_23", "name": "bash"},
    ],
    messages=[{"role": "user", "content": "记一下：我下周三要去机场，记得提醒我提前 3 小时出发"}],
    extra_headers={"anthropic-beta": "memory-2026-04-23"},
)
# Claude 自动：bash > echo "trip_2026_03_20 ..." > /memories/calendar.md

# 选项 2：自定义后端（S3 / 加密 / 数据库）
class S3MemoryTool(BetaAbstractMemoryTool):
    def __init__(self, bucket):
        self.bucket = bucket
    async def list_files(self, path: str) -> list[str]:
        return list_s3_keys(self.bucket, path)
    async def read_file(self, path: str) -> str:
        return s3_get(self.bucket, path)
    async def write_file(self, path: str, content: str):
        s3_put(self.bucket, path, content)
    # ... delete / move / search 等
```

**早期客户案例**（Anthropic 官方公布）：

- **Netflix**：内容元数据校验，**首遍错误率降 97%**
- **Wisedocs**：医保文档审核，**速度提升 30%**
- **Rakuten / Ando**：客户支持自动化

**适合**：已绑 Claude 生态、需要"agent 像程序员一样管自己的笔记本"、需要审计/版本控制（memory 文件可入 git）的场景。

#### E. 其他值得关注

- **Cognee**（[github.com/topoteretes/cognee](https://github.com/topoteretes/cognee)）：图原生记忆框架，强 ontology
- **Supermemory**（[supermemory.ai](https://supermemory.ai)）：Anthropic Memory Tool 的兼容层，跨厂商
- **Redis Agent Memory Server**：Redis 出品，semantic + keyword + hybrid 一站式
- **AgentCore Long-Term Memory**：AWS Bedrock 内置

#### 横向对比矩阵（2026）

| 维度 | Mem0 | Letta | Zep / Graphiti | Anthropic Memory Tool |
|---|---|---|---|---|
| 数据模型 | facts (vector + graph) | core / recall / archival | temporal KG | filesystem |
| Agent 主动性 | passive（auto extract） | **agent 自管 tool** | passive | **agent 自管 tool** |
| 多模型支持 | 16+ LLM | 任意 | 任意 | 仅 Claude |
| 后端选择 | 24+ vector + graph | Postgres / 内置 | Neo4j / FalkorDB | filesystem 任意 |
| 时序追踪 | 弱（基于 timestamp） | 弱 | **强（双时间轴）** | 文件 mtime |
| 关系建模 | 中（可选 graph） | 弱 | **强（图原生）** | 弱 |
| 上手成本 | **5 行** | 50 行 | 30 行 | 10 行 |
| 开源 | ✓ | ✓ | ✓（Graphiti） | × |
| 中文支持 | 配中文 embedder 即可 | 同 | 同 | 原生 |
| 论文 / 数字 | LoCoMo +20pt | MemGPT 论文 | arXiv 2501.13956 | Netflix 错误率 -97% |

### 2.5.9 多租户隔离：生产必修课

**绝大多数 SaaS 必踩的坑**：记忆系统跨用户泄露。两层防御：

#### 层 1：写入与查询都强制 user_id

```python
# 错误示范（会跨用户混）
m.add(messages=msgs)  # ❌ 没 user_id

# 正确：强制 user_id 必填
m.add(messages=msgs, user_id=request.user.id)
m.search(query=q, user_id=request.user.id, ...)  # 检索也必须带

# 强制约束（包一层）：
class TenantSafeMemory:
    def __init__(self, m, user_id):
        self.m = m; self.uid = user_id
    def add(self, messages, **kw):
        return self.m.add(messages=messages, user_id=self.uid, **kw)
    def search(self, query, **kw):
        return self.m.search(query=query, user_id=self.uid, **kw)
```

#### 层 2：存储后端层 namespace / collection / DB 隔离

| 方案 | 隔离强度 | 性能 | 适合规模 |
|---|---|---|---|
| 共享 collection + `user_id` filter | 弱（依赖应用层） | 好 | < 1k 用户 |
| 每用户一个 collection | 强 | 中（collection 数有上限） | 1k - 10k |
| 每用户一个 namespace（Pinecone / Turbopuffer） | **强** | **好** | 10k - 1M+ |
| 每租户一个独立 DB instance | 最强（物理隔离） | 中 | 大客户 / 合规 |

**Turbopuffer** 等专门为多租户设计的向量库可以撑**百万 namespace**，单租户内部数据严格隔离。

#### 层 3：审计 + GDPR 合规

```python
# 审计：所有跨用户的访问立刻报警
@audit_log(level="critical")
def admin_search_all(query):
    return vector_store.search(query=query, no_user_filter=True)

# GDPR "被遗忘权"
def forget_user(user_id):
    vector_store.delete_all(user_id=user_id)
    graph_store.delete_subgraph(user_id=user_id)
    redis.delete_pattern(f"user:{user_id}:*")
    audit_log(f"user {user_id} memory wiped at {now()}")
```

### 2.5.10 记忆 vs RAG：边界与协同

实战中**记忆和 RAG 经常并存**，不冲突：

```
User Query
    │
    ├──▶ RAG（公司文档 / 产品手册 / 政策） → 通用知识
    │
    └──▶ Memory（用户偏好 / 历史交互 / agent 经验） → 个性化

       两路检索结果合并 → 注入 system prompt → LLM
```

**典型生产 prompt 结构**：

```
你是 ACME 公司客服。

# 用户档案（来自 Memory）
- 姓名：Liang
- 历史订单：3 单（2 单已完成）
- 偏好：电邮通知优先

# 相关知识（来自 RAG）
[政策片段] 30 天内可全额退款
[FAQ 片段] 退款流程通常 3-5 工作日

# 当前对话
用户：我想退订单 #12345
```

### 2.5.10.5 Progress File：长跑 Agent 的"git memory"

vector memory / KV memory 解决"事实"，但**长跑 coding agent / research agent** 还需要一种结构化、人类可读、可 diff 的"工作进度"记忆 —— 这就是 **progress file**（Augment Code、Claude Code、Devin 都在用的事实标准）。

**典型 schema**（Markdown，常见命名 `PROGRESS.md` / `CONTEXT.md` / `.agent/state.md`）：

```markdown
---
task_id: refactor-auth-2026-05-03
started: 2026-05-03T09:12Z
last_update: 2026-05-03T11:48Z
status: in_progress
parent_branch: main
worktree: /Users/x/dev/repo/.worktrees/refactor-auth
---

## Goal
迁移 session-based auth 到 JWT，保持向前兼容 30 天

## Decisions made
- [2026-05-03 09:30] 采用 RS256 而非 HS256（多服务部署需要公钥分发）— 见 ADR-007
- [2026-05-03 10:15] 旧 session cookie 保留 30 天，双写策略

## Files touched
- `src/auth/jwt.ts` (new)
- `src/auth/middleware.ts` (modified, +120/-45)
- `src/auth/session.ts` (deprecated, kept for 30d)

## Open questions
- [ ] refresh token 是否需要黑名单？需问 @security-team
- [ ] mobile 客户端最低支持版本是否覆盖 JWT 解析？

## Next steps
1. 写 jwt.test.ts 单元测试覆盖签 / 验 / 过期 / tampering
2. 在 staging 跑 24h 双写，看 metric
3. 文档 docs/auth-migration.md
```

**为什么这种格式而不是 vector memory**：
- **可被人 review**：PR 里直接 diff，不需要打开 vector DB
- **可被 git 管理**：每次 agent 决策落 commit，回滚天然支持
- **可被 LLM 直接读 + 改**：不需要 embedding / retrieval 中间层，避免召回失败
- **跨 session 重启友好**：agent 拉 git → 读 `PROGRESS.md` → 立刻知道上次到哪

**关键工程点**：
1. **写入时机**：每个"决策点"或"工具调用结果"后追加，不要等到 session 结束（断电会丢）
2. **Merge 冲突**：多 agent 并发写要用 append-only sections + 时间戳；或拆成 per-agent 子文件
3. **OOM / token 爆炸**：定期 compaction —— 把超过 N 天的 decision 折叠成摘要保留在文件头
4. **与 vector memory 协作**：progress file 当 source of truth，vector memory 只索引指向 progress file 的 chunk
5. **失败恢复 protocol**：agent 启动时先 `git status + git log -5 + cat PROGRESS.md`，三件套是恢复"我是谁、上次到哪、下一步"的最低成本方式

参考：Anthropic *Memory Tool*、Augment Code 工程博客、Claude Code 的 `.claude/` 目录、Cursor 的 `.cursorrules`、Codename Goose 的 `goose hints`。

### 2.5.11 失败模式与对策

| 失败模式 | 表象 | 对策 |
|---|---|---|
| **记忆爆炸** | 记 1000 条后检索全是噪声 | LRU + 摘要（§2.5.7） |
| **过期记忆** | "我喜欢冷咖啡"过期但仍生效 | Zep 时序 / 写入时检测冲突 / 用户主动覆盖 |
| **幻觉记忆** | LLM 把没说过的事抽成事实 | 抽取时加 confidence 阈值 + 用户可审计 |
| **跨用户泄露** | A 的偏好被注入 B 的 prompt | 强制 user_id（§2.5.9） |
| **冷启动** | 新用户记忆为空，agent 显得"陌生" | 显式 onboarding + 通用画像兜底 |
| **隐私 / 合规** | GDPR / CCPA 要求删除用户数据 | `forget_user` API 全栈级联删除 |
| **抽取成本** | 每轮调 LLM 抽事实，token 成本翻倍 | importance gate 跳过无关对话 + 批处理 N 轮 |
| **检索召回低** | LLM 看不到关键事实 | 加 BM25 + entity match（§2.5.6 multi-signal） |
| **记忆腐坏（memory corruption）** | 见下面专栏 | 定期一致性检查 + ground truth 校准 + ttl |

#### Memory Corruption 深度剖析

"记忆腐坏"是 stateful agent 部署超过几周后**必然出现**的失败模式，但很少被早期工程团队预防。Anthropic / Letta / Mem0 团队公开过的 4 类典型 corruption：

| 腐坏类型 | 例子 | 根因 | 防御 |
|---|---|---|---|
| **过期事实（Stale fact）** | "用户在杭州" → 三个月后已搬到上海，但旧记忆仍在 top-K | 没有 TTL、没有"事实新鲜度"的写入时检测 | Zep 风格 valid_from / valid_to；写入时检测同实体的旧事实并标 invalid |
| **矛盾事实（Contradiction）** | 一条说"用户素食"，另一条说"用户喜欢牛排" | 多 session 抽取没有跨 session 一致性检查 | 写入前跑"该用户已有事实 + 新事实"的矛盾检测 LLM；或定期 reconciliation job |
| **幻觉事实（Hallucinated fact）** | LLM 抽取时从对话里"推理"出用户没明说的偏好 | 抽取 prompt 没限定"只抽显式陈述" | 抽取时返回 evidence span（原对话引用）；无 evidence 不落库 |
| **人格漂移（Persona drift）** | 长期 self-update memory 导致 agent 自我描述与初始 system prompt 严重不一致 | agent 自管 memory + 没有 freeze section | core memory 区分 immutable / mutable 段；mutable 段定期 reset 到 baseline |

**防御工程实践**（按 ROI 排序）：
1. **TTL + 时序字段**：每条记忆带 `created_at` / `valid_from` / `valid_to`，检索时按当前时间过滤
2. **写入时矛盾检测**：新事实 vs 同实体已有 top-3 事实，跑一次 LLM"是否矛盾"判断，矛盾时人工或时间戳决胜
3. **Evidence-required 抽取**：抽取 prompt 强制返回 `(fact, evidence_quote, confidence)`，confidence < 0.7 不落库
4. **每周 reconciliation job**：随机抽样 1% 记忆跑 LLM 对比真实对话，发现腐坏率超阈值就触发清理
5. **可解释 + 可审计**：用户可以看到"agent 关于我的记忆"原文 + 来源，并一键删错

### 2.5.12 GitHub 实战资源（记忆系统专题）

| 资源 | 类型 | 推荐用法 |
|---|---|---|
| [`mem0ai/mem0`](https://github.com/mem0ai/mem0) | 官方仓库 | 通用记忆层，5 行接入 |
| [`letta-ai/letta`](https://github.com/letta-ai/letta) | 官方仓库 | OS-inspired stateful agent |
| [`getzep/graphiti`](https://github.com/getzep/graphiti) | 官方仓库 | Temporal KG 引擎（Zep 内核） |
| [`getzep/zep`](https://github.com/getzep/zep) | 官方仓库 | 完整 Zep 服务（基于 Graphiti） |
| [`redis/agent-memory-server`](https://github.com/redis/agent-memory-server) | 官方仓库 | Redis 一站式 agent memory |
| [`topoteretes/cognee`](https://github.com/topoteretes/cognee) | 官方仓库 | 图原生 + ontology |
| [`supermemoryai/supermemory`](https://github.com/supermemoryai/supermemory) | 官方仓库 | 跨厂商记忆 SDK |
| Anthropic Memory Tool 文档 | 官方文档 | `https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool` |

**论文必读**：

- [`Mem0 论文`](https://arxiv.org/abs/2504.19413) — Production-Ready AI Agents with Scalable Long-Term Memory
- [`MemGPT 论文`](https://arxiv.org/abs/2310.08560) — Towards LLMs as Operating Systems
- [`Zep 论文`](https://arxiv.org/abs/2501.13956) — A Temporal Knowledge Graph Architecture for Agent Memory
- [`MaRS 论文`](https://arxiv.org/abs/2603.07670) — Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers（2026 综述）
- [`Forgetful but Faithful`](https://arxiv.org/abs/2512.12856) — Cognitive Memory Architecture and Benchmark

---

## 2.6 上下文管理策略：把 context window 当作稀缺资源

### 2.6.1 为什么需要"管理"

Context window 即使到了 1M tokens，仍然是**稀缺资源**——三个原因：

1. **成本**：1 M token prompt 一次 ~$1（Claude Opus），一天调几千次就破产
2. **延迟**：1 M token prefill ~30 秒，用户体验崩塌
3. **质量**：Lost-in-the-Middle，多事实场景召回降到 ~60%（§2.2.3）

所以生产级 agent 不是"塞满 1 M"，而是**精打细算**：每个 token 都该有理由出现在 prompt 里。

### 2.6.2 Context 的"分区"模型

把 context window 看作有逻辑分区的固定空间，每个分区有不同优先级和淘汰策略：

```
┌─────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────┐ │
│ │ [固定区] System Prompt + Persona                │ │  ← 永不淘汰
│ │   - 人设、约束、风格、角色                       │ │     ~500-2000 tokens
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [可变高优] Memory（注入的用户/agent 记忆）       │ │  ← 检索 + 重排
│ │   - 用户档案、偏好、近期事件                     │ │     ~500-3000 tokens
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [可变高优] RAG 检索结果                          │ │  ← top-K + reranker
│ │   - 公司知识、文档片段                           │ │     ~2000-8000 tokens
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [可变中优] 工具定义 schema                       │ │  ← 仅放当前可能用到的
│ │   - tool descriptions + signatures               │ │     ~500-3000 tokens
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [可变中优] 历史对话                              │ │  ← 滑窗 + 摘要
│ │   - 最近 N 轮 + 早期摘要                         │ │     可变
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [低优 / 易淘汰] 工具调用结果                     │ │  ← 旧的截断或摘要
│ │   - tool outputs，往往最长                       │ │     可变
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [尾部] 当前 user message                         │ │  ← 永远是最后
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
   头部                                          尾部
   ↑ primacy bias                       recency bias ↑
                  Lost-in-the-Middle 黑洞
```

**配额分配建议**（按业务类型）：

| 业务类型 | System | Memory | RAG | Tools | 历史 | 工具结果 | User |
|---|---|---|---|---|---|---|---|
| 客服 chatbot | 10% | 15% | 30% | 5% | 20% | 10% | 10% |
| Coding agent | 5% | 5% | 40% | 10% | 10% | 25% | 5% |
| 研究 agent | 5% | 10% | 30% | 5% | 10% | 35% | 5% |
| 个人助理 | 15% | 30% | 10% | 10% | 25% | 5% | 5% |

实战经验：**任何分区超过 50% 都是设计有问题的信号**。

### 2.6.3 上下文管理的 5 类常见动作（综合 Anthropic / LangChain 实践）

> 业界两个权威来源给的拆法都不是"5"——Anthropic *Effective Context Engineering* [6] 给 **compaction / structured note-taking / sub-agent architectures** 三类策略；LangChain *Context Engineering for Agents* [7] 给 **write / select / compress / isolate** 四类。下面 5 个动作是把两家的拆法合并展开，便于工程实施时逐项检查；不是业界标准 taxonomy。

#### 动作 1：优先级排序（Priority Ranking）

利用 primacy + recency bias，把最重要的放头尾，次要的放中间：

```python
def assemble_context(system, memories, rag_chunks, tools, history, current_user_msg):
    return [
        system,                              # 头部：永远在前
        *high_priority(memories),            # 头部 2：用户身份与偏好
        *high_priority(rag_chunks, top=3),   # 头部 3：最相关的 3 个片段
        *medium_priority(history),           # 中部：历史对话（可被中部黑洞稀释）
        *low_priority(rag_chunks, rest=True),# 中部：剩余 RAG 片段
        *truncate(tool_results),             # 尾部 -2：截断的工具结果
        tools_definitions,                    # 尾部 -1：tool schema
        current_user_msg,                    # 尾部：当前消息
    ]
```

#### 动作 2：动态裁剪（Trimming）

定时检查 token 使用率，超阈值触发：

```python
def trim_if_needed(context, max_tokens=180_000, target=140_000):
    current = count_tokens(context)
    if current < max_tokens: return context

    # 优先淘汰：旧的工具结果
    context = truncate_old_tool_results(context, keep_recent=5)
    if count_tokens(context) <= target: return context

    # 其次：早期对话压缩成摘要
    context = summarize_old_history(context, summary_length=500)
    if count_tokens(context) <= target: return context

    # 再次：删除非 top-K 的 RAG 片段
    context = drop_low_score_rag(context, threshold=0.6)

    return context
```

#### 动作 3：摘要压缩（Summarization）—— `/compact` 实现

Claude Code 的 `/compact` 命令把当前长对话压成摘要 + 关键 facts，回收 token：

```python
def compact(messages, llm):
    # 1. 抽取关键事实（不能丢）
    facts = llm.invoke(f"""
    Extract critical facts from this conversation that MUST be preserved:
    {messages}
    Return as bullet list. Include: user requirements, decisions made, code locations,
    pending tasks, open questions.
    """)

    # 2. 写一段精简摘要
    summary = llm.invoke(f"""
    Summarize this conversation in <500 words. Focus on:
    - What problem we're solving
    - What approach we took
    - Current progress
    {messages}
    """)

    # 3. 重置 context
    new_context = [
        SystemMessage(original_system),
        SystemMessage(f"# Compressed conversation context\n\n{summary}\n\n# Key facts\n{facts}"),
        # 保留最近 3 轮原文（细节在新对话里有用）
        *messages[-3:],
    ]
    return new_context
```

**关键点**：摘要必须 LLM 生成（rule-based 抽取会丢上下文），但 LLM 摘要本身要 prompt 明确要求"保留代码位置/数字/决定"等关键信息。

#### 动作 4：Scratchpad 分离

ReAct 等推理 agent 在思考时会产生大量 thought / observation，不应该污染对外 context：

```python
class AgentMemory:
    def __init__(self):
        self.public_messages = []   # 用户和 agent 互发
        self.scratchpad = []        # agent 内部草稿（thought / 中间 tool result）

    def get_llm_context(self):
        # 给 LLM 看的：public + 最近 5 步 scratchpad（防止失忆）
        return self.public_messages + self.scratchpad[-5:]

    def get_user_facing(self):
        # 给用户看的：仅 public
        return self.public_messages
```

LangGraph 的 state 机制 / Letta 的 core vs recall 都是这个思路的产品化。

#### 动作 5：工具结果裁剪

工具返回的 JSON 经常爆长（比如一次 SQL 查询 1000 行）。两层过滤：

```python
def clip_tool_result(tool_name, result, current_step, total_steps):
    # 1. 大结果直接截断
    s = str(result)
    if len(s) > 4000:
        s = s[:2000] + f"\n... [truncated {len(s)-2000} chars] ...\n" + s[-500:]

    # 2. 老的 tool result（>3 步前）摘要化
    if total_steps - current_step > 3:
        s = summarize_tool_result(s, max_length=300)

    return s
```

### 2.6.4 三种"压缩"路径对比

| 方法 | 压缩比 | 信息损失 | 延迟 | 适用 |
|---|---|---|---|---|
| **简单截断** | 中 | 高 | 0 | 工具结果 / 兜底 |
| **rule-based 抽取**（regex / structured） | 中 | 低 | 0 | 已知格式（JSON / 表格） |
| **LLM 摘要**（`/compact`） | **高** | **低** | 1-3 秒 | 长对话 / 高价值历史 |
| **Anchored Iterative Summary**（2026 SOTA） | **高** | **最低** | 3-10 秒 | 极长任务（数小时 agent） |

#### 何时触发 compaction（决策信号）

Compaction 不是"context 满了再压"，否则 LLM 已经被噪声污染。生产实践用多信号触发：

| 触发信号 | 阈值（参考） | 适用场景 |
|---|---|---|
| **Token 占用率** | > 70% max_tokens（如 180K / 256K = 70%） | 通用兜底，最常用 |
| **历史轮次** | > 50 轮对话 | 长对话客服 / 陪伴产品 |
| **工具结果累积** | 单步 tool result > 4000 tokens 或累积 > 30K | coding agent / data agent |
| **质量信号下滑** | 最近 5 轮 user 满意度 / verifier 通过率下降 | 有 reward signal 的产品 |
| **任务边界** | 一个子任务完成（agent 自己 declare） | 长跑 multi-step agent |
| **时间衰减** | 历史消息 > 30 分钟无引用 | 多任务并行 agent |
| **Anthropic Auto-Compact 信号** | Claude API `usage.context_window_remaining` < 20% | 用 Claude 的产品直接读官方信号 |

**关键原则（Anthropic *Effective Context Engineering* 2026）**：
1. Compaction 是**有损**的，能避免就避免（先做 §2.6.3 的动作 1-5）
2. 永远保留 **最后 N 轮原始对话**（N=3-5），LLM 需要近端 raw 信号判断"刚刚发生了什么"
3. 摘要要保留 **decision、open question、未完成子任务、代码 / 文件路径** 这些"指针"，而不是讲故事
4. 摘要后立刻 evaluator 跑一遍，确认关键事实没丢；丢了就回滚到 raw + 升级到更强压缩策略
5. **Compaction 本身要被 trace** —— 哪些被丢、压缩前后的 token diff、是否触发回滚，都进 observability

### 2.6.5 实战配方：完整 Context Manager 类

```python
class ContextManager:
    def __init__(self, max_tokens=180_000, target_tokens=140_000):
        self.max = max_tokens
        self.target = target_tokens

    def assemble(self, system, memories, rag, tools, history, user_msg):
        ctx = self._assemble_by_priority(system, memories, rag, tools, history, user_msg)
        if count_tokens(ctx) > self.max:
            ctx = self._compress(ctx)
        return ctx

    def _assemble_by_priority(self, ...):
        # 见 §2.6.3 动作 1

    def _compress(self, ctx):
        # 顺序尝试 5 种压缩，越早越轻：
        for fn in [
            lambda c: drop_low_score_rag(c, threshold=0.6),
            lambda c: clip_tool_results(c, keep_recent=5),
            lambda c: summarize_old_history(c, length=500),
            lambda c: drop_low_priority_memory(c, keep=10),
            lambda c: hard_truncate_history(c, keep_recent=3),
        ]:
            ctx = fn(ctx)
            if count_tokens(ctx) <= self.target:
                return ctx
        return ctx  # 实在压不下，让 LLM 自己拒绝

    def metrics(self, ctx):
        # observability：每次返回 dict 让 dashboard 跟踪每分区占比
        return {
            "total": count_tokens(ctx),
            "system_pct": count_tokens(ctx[0]) / count_tokens(ctx),
            "rag_pct": ...,
            "history_pct": ...,
            "tool_results_pct": ...,
        }
```

把 `metrics()` 接进 Langfuse / LangSmith，能在 dashboard 看到"哪类内容在吃 context 配额"——这是诊断 agent 漂移最实用的工具。

### 2.6.6 Prompt / Context Caching：2026 年最大 cost 杠杆

**所有现代 LLM provider 都支持 prompt caching**——把"prompt 前缀"在 KV cache 层缓存起来，下次相同前缀复用。**这是 2025-2026 年单次最大的 cost 优化机会，最高 90% 折扣**。

#### 三大 provider 实现对比

| Provider | 触发方式 | 命中折扣 | 最小命中长度 | TTL | 写入 |
|---|---|---|---|---|---|
| **Anthropic** | 显式（`cache_control` 标记） | **90%**（cached input $0.30 vs base $3.00 / Sonnet） | 1024 tokens（Sonnet）/ 2048（Haiku） | 5 min（默认）/ 1h（付费） | 1.25× base |
| **OpenAI** | 自动（≥1024 token prefix 自动缓存） | **50-90%**（视模型） | 1024 tokens | 5-10 min | 0（免费写） |
| **Google Gemini** | 隐式（2.5+ 默认开）+ 显式 API | 隐式约 50% / 显式约 75% | 4096 tokens | 默认 1h | 按存储付费 |
| **AWS Bedrock** | 全 provider 转译 | 同各 provider | 同 | 同 | 同 |

实证（ProjectDiscovery 公开数据）：把 prompt caching 接好后，**LLM 月账单降 59-70%**。学术研究（arXiv 2601.06007）跑 long-horizon agent 任务，**API cost 降 41-80%、TTFT 降 13-31%**。

#### 什么 prompt 适合缓存

按"复用频率 × 长度"打分：

| 内容类型 | 适合缓存？ |
|---|---|
| **System prompt + persona** | ★★★ 强烈推荐 |
| **大量 few-shot examples** | ★★★ |
| **工具定义 schema（10+ tools）** | ★★★ |
| **RAG 检索的固定文档（如 SLA / 政策手册）** | ★★ |
| **会变的对话历史** | ✗ 别缓存（缓存击穿） |
| **每次都不同的 user query** | ✗ |

**经验法则**：**prefix 重复率 > 50% 的就缓存，否则别**。

#### Anthropic 显式缓存代码

```python
from anthropic import Anthropic

client = Anthropic()

# system + tools + 大段 RAG 都加 cache_control
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是 ACME 公司客服，遵循以下规则...",  # 长 system
            "cache_control": {"type": "ephemeral"},          # ★ 标记为可缓存
        },
        {
            "type": "text",
            "text": KB_KNOWLEDGE_BASE,  # 100K token 的知识库
            "cache_control": {"type": "ephemeral"},
        },
    ],
    tools=[
        {"name": "search", "description": "...", "input_schema": {...},
         "cache_control": {"type": "ephemeral"}},
    ],
    messages=[
        {"role": "user", "content": user_query},  # 这部分不缓存
    ],
)

# 看 cache 命中
print(f"Cache read: {response.usage.cache_read_input_tokens}")
print(f"Cache write: {response.usage.cache_creation_input_tokens}")
print(f"Regular input: {response.usage.input_tokens}")

# Cost 计算
cost = (
    response.usage.cache_read_input_tokens * 0.30 / 1e6        # 命中超便宜
    + response.usage.cache_creation_input_tokens * 3.75 / 1e6   # 写入贵 1.25×
    + response.usage.input_tokens * 3.00 / 1e6                  # 普通输入
    + response.usage.output_tokens * 15.00 / 1e6                # 输出
)
```

#### Cache 命中的几个关键约束

1. **必须严格前缀匹配**：从开头到 `cache_control` 标记位置必须**完全一致**——多一个空格都 miss
2. **顺序敏感**：把固定的（system / tools / RAG）放前，可变的（user message）放后
3. **TTL 短**：默认 5 min，意味着 5 分钟内没第二次调用就白缓存了——适合"高 QPS 同一系统 prompt"场景
4. **Cache write 也付费**：第一次写入 1.25×，所以**至少要被命中 2 次才划算**

#### 显式 vs 自动 caching 选型

- **OpenAI 自动**：零改动接入，但你不知道哪些命中、不可控
- **Anthropic 显式**：要改 SDK 调用，但**完全可控**——可以精确知道节省多少
- **生产推荐 Anthropic**：把 system / tools / 大段 RAG 都标 cache_control，配合 observability 看 cache hit 率

#### "Don't Break the Cache"：agent 长任务的关键

agent 跑 long-horizon 任务时，prompt 是"前缀（系统）+ 不断追加的对话历史"。**追加新内容不破坏前缀缓存**，所以前缀部分一直命中：

```
Round 1: [system] [user1]                          → write cache（system）
Round 2: [system] [user1] [bot1] [user2]           → cache HIT 前缀，付钱仅 [bot1][user2]
Round 3: [system] [user1] [bot1] [user2] [bot2] [user3] → cache HIT 更长前缀
...
```

**反面**：如果你在中间插入 / 修改 / 删除任何内容（如 RAG 重新检索把不同 chunk 塞进 system），cache 直接失效，下一次又要全付费。

agent 设计的隐藏 best practice：**把"会变的内容"放尾部，前缀永远只追加不修改**。

#### 监控 cache 效果

```python
# Langfuse 把 cache metrics 也记下来
@observe(metadata={"cache_strategy": "anthropic_explicit"})
def my_agent_call(query):
    response = client.messages.create(...)
    langfuse_context.update_current_observation(
        usage={
            "cache_read": response.usage.cache_read_input_tokens,
            "cache_write": response.usage.cache_creation_input_tokens,
            "regular_input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )
    return response
```

dashboard 关注：

- **Cache hit rate**（cache_read / total_input）—— 目标 > 60%
- **Cost per request 对比**（启用前 vs 启用后）—— 目标降 50%+
- **TTFT 对比**——目标降 20%+

---

## 2.7 MCP / A2A 协议生态

### 2.7.1 为什么需要 MCP：解决"M×N 集成噩梦"

把工具和数据源接入 LLM 长期是脏活：

- 每个 LLM provider 的 function calling schema 都不一样（OpenAI / Anthropic / Google）
- 每个工具都要手写 wrapper、维护多份
- 工具升级 = 所有 agent 都要重新对接
- 跨 LLM 切换 = 全套 wrapper 重写

数学模型：M 个 LLM × N 个工具 = M·N 个适配器。10 个 LLM × 50 个工具 = 500 个集成点。

**MCP（Model Context Protocol）** [9] 是 Anthropic 2024 年 11 月推出的开放标准，把 M·N 拆成 M+N：

```
  没有 MCP                       有 MCP
  ┌───┬───┬───┐                ┌───┬───┬───┐
  │GPT│Cld│Gem│                │GPT│Cld│Gem│
  └─┬─┴─┬─┴─┬─┘                └─┬─┴─┬─┴─┬─┘
    │   │   │ × N tools          └───┼───┘
   各连各的 (M·N)                    │  MCP protocol (JSON-RPC 2.0)
                                     │
                              ┌──────┼──────┐
                              │      │      │
                            ┌─┴─┬──┬─┴─┬──┬─┴─┐
                            │GH │DB│Fs │Sl│...│  ← N tools
                            └───┴──┴───┴──┴───┘
```

**类比**：USB-C 之前每个手机一根线（M·N）；USB-C 之后任意主机接任意外设（M+N）。MCP 是 AI 的 USB-C。

### 2.7.2 MCP 协议规格

#### 三类原语（Primitives）

MCP server 暴露给 client 的能力分三类：

| 原语 | 含义 | 例子 | 由谁触发 |
|---|---|---|---|
| **Tools** | 可执行函数，有副作用 | `create_issue`、`run_sql` | **模型决定调用** |
| **Resources** | 只读数据，URI 寻址 | `file:///doc.md`、`db://table/users` | **应用预加载** |
| **Prompts** | 用户可调用的模板 | `/explain-code`、`/summarize` | **用户触发** |

这三类对应 LLM 应用的三种数据流：模型主动调用、应用注入、用户调用。

#### 三种 Transport

| Transport | 适用 | 部署 |
|---|---|---|
| **stdio** | 本地进程间通信（Claude Desktop / Cursor / Cline 默认） | 进程子进程，配置在 `mcp_servers.json` |
| **SSE**（Server-Sent Events） | 远程服务，单向流 | HTTP server，长连接 |
| **Streamable HTTP**（2025 替代 SSE） | 远程服务，双向流，更标准 | HTTP server，事实标准 |

### 2.7.3 完整 MCP Server 实现示例

#### Python（FastMCP）

```python
# server.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("my-toolkit", version="1.0.0")

# === Tools（有副作用的函数）===
@mcp.tool()
def create_github_issue(repo: str, title: str, body: str) -> dict:
    """在指定 GitHub repo 创建 issue。
    Args:
        repo: owner/name 格式
        title: issue 标题
        body: issue 正文（markdown）
    Returns:
        {"id": int, "url": str}
    """
    # 实际调用 GitHub API
    response = github.create_issue(repo, title, body)
    return {"id": response.id, "url": response.html_url}

@mcp.tool()
async def run_sql(query: str, params: list = None) -> list[dict]:
    """对只读副本执行 SQL 查询。"""
    # 安全：用 parameterized query
    result = await db.fetch_all(query, params or [])
    return [dict(r) for r in result[:1000]]  # 上限 1000 行

# === Resources（只读数据）===
@mcp.resource("docs://{topic}")
def get_docs(topic: str) -> str:
    """获取某主题的内部文档。"""
    return knowledge_base.get(topic)

@mcp.resource("user://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """获取用户档案（仅 PII redacted 版）。"""
    return user_db.get_redacted(user_id)

# === Prompts（用户模板）===
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """让 AI 做代码审查。"""
    return f"""请对以下 {language} 代码做 code review：
- 找出潜在 bug
- 评估可读性
- 建议优化

```{language}
{code}
```
"""

if __name__ == "__main__":
    mcp.run(transport="stdio")  # 或 sse / streamable-http
```

#### TypeScript（@modelcontextprotocol/sdk）

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "my-toolkit", version: "1.0.0" });

// Tool
server.tool(
  "create_issue",
  { repo: z.string(), title: z.string(), body: z.string() },
  async ({ repo, title, body }) => {
    const result = await github.createIssue(repo, title, body);
    return { content: [{ type: "text", text: JSON.stringify(result) }] };
  }
);

// Resource
server.resource(
  "user-profile",
  "user://{user_id}/profile",
  async (uri, { user_id }) => ({
    contents: [{ uri: uri.href, mimeType: "application/json",
                 text: JSON.stringify(await userDb.get(user_id)) }],
  })
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 2.7.4 MCP Client 接入

#### Claude Desktop / Cursor / Cline 配置

`~/.config/claude/mcp_servers.json`（Mac/Linux）或 Cursor / Cline 类似配置：

```json
{
  "mcpServers": {
    "my-toolkit": {
      "command": "python",
      "args": ["/abs/path/to/server.py"],
      "env": {"GITHUB_TOKEN": "ghp_..."}
    },
    "github-official": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/Users/liang/projects"]
    }
  }
}
```

重启 client，所有 tools / resources / prompts 自动可用。

#### 在 LangChain / LangGraph 里用

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# 一次连多个 MCP server
async with MultiServerMCPClient({
    "my-toolkit": {"command": "python", "args": ["./server.py"], "transport": "stdio"},
    "github":     {"url": "http://localhost:3000/sse", "transport": "sse"},
}) as client:
    tools = client.get_tools()  # 自动转成 LangChain Tool 列表
    agent = create_react_agent("anthropic:claude-sonnet-4-6", tools)
    result = await agent.ainvoke({"messages": [("user", "在 acme/api repo 开个 bug issue")]})
```

### 2.7.5 主流 MCP Server 生态（2026）

按类别选用：

| 类别 | 仓库 | 用途 |
|---|---|---|
| **官方参考实现** | [`modelcontextprotocol/servers`](https://github.com/modelcontextprotocol/servers) | 几十个开箱即用：filesystem / git / github / postgres / sqlite / slack / google-drive / brave-search / sequential-thinking |
| **Awesome 列表** | [`punkpeye/awesome-mcp-servers`](https://github.com/punkpeye/awesome-mcp-servers) | 社区 server 索引（数百个） |
| **数据库** | postgres / mysql / sqlite / mongodb / redis 都有官方或社区 MCP |
| **云服务** | aws-mcp / gcp-mcp / azure-mcp |
| **DevOps** | kubernetes / terraform / docker MCP |
| **生产力** | notion / linear / jira / confluence MCP |
| **浏览器** | playwright-mcp / browserbase-mcp / Puppeteer MCP |
| **企业商用** | Zapier MCP（5000+ 集成） |

### 2.7.6 MCP 时间线与采用度

| 时间 | 事件 |
|---|---|
| 2024-11 | Anthropic 发布 MCP 协议 |
| 2025-Q1 | Cursor / Cline 等 IDE 接入 |
| 2025-Q2 | OpenAI 宣布支持 |
| 2025-Q3 | Google / Microsoft / Amazon 跟进 |
| 2025-Q4 | MCP 治理转入独立 **MCP Steering Committee**（Anthropic + Block + OpenAI + Microsoft 等共同治理）|
| 2026-02 | Python + TypeScript SDK 月下载量破 **9700 万** |
| 2026-Q1 | 几乎所有主流 LLM 客户端原生支持 |

### 2.7.7 A2A：Agent 之间的协议族

MCP 解决"agent 与工具的沟通"，**A2A（Agent-to-Agent）** 解决"agent 之间的沟通"。这块仍在多协议竞争：

| 协议 / 框架 | 出处 | 范式 | 状态 |
|---|---|---|---|
| **Google A2A** | Google 2025 | HTTP-based agent discovery + task delegation | 早期 |
| **Microsoft Agent Framework Protocol** | Microsoft 2026 | AutoGen 接班，多 agent 对话标准 | 推广中 |
| **Anthropic Agent Skills** | Anthropic 2026 | Skill 包：可复用 agent 能力单元 | 早期 |
| **MCP for A2A** | 社区 | 把"另一个 agent"当作 MCP server | 实践中 |
| **Letta multi-agent** | Letta | 通过 message passing | 框架内 |

**MCP vs A2A 类比**：

> "**MCP 给 agent 一双手**（连工具/数据），**A2A 让多个 agent 组队干活**。MCP 是垂直整合（单 agent ↔ 工具栈），A2A 是水平协作（agent ↔ agent 网络）。"

2026 春的现状：MCP 已是事实标准，A2A 还在 Cambrian 大爆发期。**实战建议**：现在做单 agent → 全押 MCP；做多 agent → 用具体框架（LangGraph / Microsoft Agent Framework）的内置协议，等 A2A 标准明朗。

### 2.7.8 常见 MCP 设计陷阱

| 陷阱 | 表现 | 对策 |
|---|---|---|
| **Tool 太多** | 50+ tool 让 LLM 决策困难 | 分组 + 路由（先选类目再选 tool） |
| **Tool description 太短** | 模型不知道何时用 | 写"何时用 / 何时不用"+ 1-2 个 example |
| **副作用未声明** | 模型当只读调用，造成意外修改 | description 明确"会修改 X"+ 在 client 侧加 deny-first |
| **超长返回** | 一次返回 10K rows 撑爆 context | server 侧 cap + 分页 + 摘要 |
| **同步阻塞** | 长任务把 client 卡死 | 用 async transport（streamable-http） |
| **认证混乱** | 每个 tool 自己处理 token | server 侧统一从 env 读，client 侧用 OAuth flow |

工具集成在 Agent 实现层面的细节 → [agent-engineering.md](./agent-engineering.md) §3.1 与 §5；
LLM 网关 / observability / 推理服务器层面的基础设施 → [harness-engineering.md](./harness-engineering.md) §3.9 附录（注：04 章把这 4 类显式标记为"相关但非 harness 工程"）。

---

## Bibliography

[2] Liu, N. F. et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL.
[5] Karpathy, A. *Software Is Changing (Again)*. YC AI School talk.
[6] Anthropic. *Effective Context Engineering for AI Agents*. <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
[7] LangChain. *Context Engineering for Agents*. <https://www.langchain.com/blog/context-engineering-for-agents>
[9] Wikipedia. *Model Context Protocol*. <https://en.wikipedia.org/wiki/Model_Context_Protocol>
[12] Hsieh, C. et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* <https://arxiv.org/abs/2404.06654>
[13] Cheney Zhang. *Embedding Models Benchmark 2026*.
[14] CallSphere. *Vector Database Benchmarks 2026: pgvector / Qdrant / Weaviate / Milvus / LanceDB*.
[18] Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*.
[19] Mem0 Team (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. <https://arxiv.org/abs/2504.19413>
[21] Anthropic Memory Tool docs. <https://docs.anthropic.com>
[41] TianPan. *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool*.
[43] Squirro. *RAG in 2026: Bridging Knowledge and Generative AI*.

补充：
- Asai et al. (2023). *Self-RAG*.
- Microsoft GraphRAG. <https://github.com/microsoft/graphrag>
- LightRAG. <https://github.com/HKUDS/LightRAG>
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
- BGE-M3. <https://huggingface.co/BAAI/bge-m3>

---

## 章节交叉引用

- 想理解长上下文背后的 Transformer 物理 → [prompt-engineering.md](./prompt-engineering.md) §1.1
- 想理解 Agentic RAG / 记忆 在 Agent 里怎么用 → [agent-engineering.md](./agent-engineering.md) §3 + §4
- 想理解 KV cache / prefix caching 推理服务器层面 → [harness-engineering.md](./harness-engineering.md) §3.9.2

