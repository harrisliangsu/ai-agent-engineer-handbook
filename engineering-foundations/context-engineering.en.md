# Context Engineering: Memory Systems and Full-Stack RAG Choices

> **English** · [中文](./context-engineering.md)

**Position:** Layer 2 of the four-layer evolution of LLM application engineering
**Research date:** 2026-05-01

---

<a id="executive-summary"></a>

## Executive Summary

**Context engineering** was formally named in the second half of 2024 by Karpathy, in his YC AI School talk, and Anthropic [6] positions it as "the natural evolution of prompt engineering." The core observation: in the era of 100K–1M context windows and agents, what determines how the model performs is no longer "that one prompt," but **everything the model can see at the moment of inference**—the system prompt, conversation history, tool definitions, retrieved documents, memory, and planning state.

This chapter covers:

- **§2.1** Definition and origin (the three-way consensus of Karpathy, Anthropic, and LangChain)
- **§2.2** The physics of long context: O(N²) cost, the KV cache, and the U-shaped recall curve of Lost-in-the-Middle
- **§2.3** The RULER benchmark and the reality of 1M context—Gemini 1.5 Pro scores 99.7% on a single needle, but multi-fact recall is ~60%; long context is substantially slower and more expensive than RAG (a single source [41] estimates roughly 30–60× slower and ~1000× more expensive; the exact multiple varies widely by model and scenario)
- **§2.4** Full-stack RAG choices—from four generations of evolution to 10 subsections (parsing → chunking → embedding → vector store → retrieval → reranking → rewriting → GraphRAG → Self-RAG/CRAG/Agentic RAG → multimodal)
- **§2.5** Four leading memory systems compared: Mem0 / Letta (formerly MemGPT) / Zep (temporal KG) / Anthropic Memory Tool
- **§2.6** Context-management tactics (priority, eviction, summarization, `/compact`)
- **§2.7** The MCP / A2A protocol ecosystem (MCP at 97 million monthly downloads)

The pivotal shift: **the core difficulty of a production LLM application has moved from "writing the prompt" to "managing context"**—and that is the prerequisite for agents ([03](./agent-engineering.en.md)) and harness engineering ([04](./harness-engineering.en.md)).

---

<a id="s2-1"></a>

## 2.1 Definition and origin

The term "context engineering" began circulating in the community in the second half of 2024, and is generally credited to Karpathy's YC AI School talk *Software Is Changing (Again)* [5]. The definition he gave has been quoted repeatedly since:

> "In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step."
> (In every industrial-grade LLM application, context engineering is the delicate art and science of filling the context window with exactly the right information, in preparation for the next step of inference.)

Anthropic's official 2025 article *Effective Context Engineering for AI Agents* [6] goes further and positions it as "the natural evolution of prompt engineering":

> "What prompt engineering cares about is the sentence you say to the model; what context engineering cares about is all the information the model can see at the moment of inference—the system prompt, conversation history, tool definitions, retrieved documents, memory, planning state, constraints, and policies."

The LangChain team, in the blog post *Context Engineering for Agents* [7], gives an engineering definition that is often cited:

> "Context engineering is the discipline of designing **dynamic systems** that, **at the right time and in the right format, supply the LLM with the right information and tools**, so the model has everything it needs to complete the task."

Why is "prompt engineering" no longer enough? Three fundamental changes:

1. **Window expansion:** from GPT-3.5's 4K all the way up to Gemini 2.5 Pro's 1M and Claude Opus 4.7's 1M. What can be put in the window has gone from "a few paragraphs" to "an entire book." Deciding what to put in has become far more important than writing that one sentence well.
2. **The rise of agents:** a single-turn prompt is an isolated transaction; an agent is a continuing state machine. The context for each step of inference comes from the previous step's output + tool returns + retrieval + memory. The "prompt" is no longer the sentence the user wrote; it is something the agent framework assembles dynamically.
3. **Multimodal input + tool calling:** context is no longer text alone. It also includes images, tables, code, tool schemas, retrieved chunks, structured memory, and a scratchpad. Organizing this heterogeneous information is an engineering problem.

By early 2026, the industry consensus is: **the core difficulty of a production LLM application has moved from "writing the prompt" to "managing context."**

---

<a id="s2-2"></a>

## 2.2 The physics of long context

To understand why "filling the window" is a bad idea, you first have to understand three physical constraints on context:

<a id="s2-2-1"></a>

### 2.2.1 O(N²) cost

Back to [prompt-engineering.en.md](./prompt-engineering.en.md) §1.1: standard attention is N² complexity. When context goes from 32K to 128K, attention compute is 16×, not 4×. For an inference service that means two things:

- Time to first token (TTFT) rises, because the entire prompt must be prefilled first
- GPU memory is filled by the KV cache, and concurrency drops

<a id="s2-2-2"></a>

### 2.2.2 KV cache GPU memory

Each attention layer maintains caches of K and V at inference time (to avoid recomputation). **For a 70B model at 128K context, the KV cache can occupy 40+ GB, larger than the model weights themselves.** That is why techniques that "manage the KV cache," such as vLLM's PagedAttention and SGLang's RadixAttention, have become a core competitive advantage of inference servers (see [harness-engineering.en.md](./harness-engineering.en.md) §3.9.2).

<a id="s2-2-3"></a>

### 2.2.3 Lost-in-the-Middle

Liu et al.'s 2024 TACL paper *Lost in the Middle: How Language Models Use Long Contexts* [2] systematically verified a counterintuitive phenomenon: put the critical information at the start or the end of the prompt and the model uses it well; put it in the middle and the model often fails to see it. **Recall as a function of position follows a U-shaped curve**—primacy bias at the start, recency bias at the end, and a black hole in the middle.

The curve has been verified on GPT-3.5 / GPT-4 / Claude. Even Gemini 1.5 Pro in 2025 reaches 99.7% recall on a simple needle-in-a-haystack test, but when the task becomes "find 8 relevant facts inside 1M tokens and synthesize them," average recall drops to ~60% [41].

---

<a id="s2-3"></a>

## 2.3 RULER and the real capability of 1M context

The **RULER benchmark** [12] (*What's the Real Context Size of Your Long-Context Language Models?*), published in 2024 by NVIDIA and other institutions, upgrades "long-context capability" from the toy needle-in-a-haystack test to a torture suite:

- **Needles of many kinds and many counts:** not one buried fact, but several facts of different types
- **Multi-hop reasoning:** you can find B only after finding A
- **Aggregation tasks:** count a total or take an average over a long document
- **Variable tracking:** follow a variable through multiple reassignments across a span of 100K tokens

What RULER exposes: **almost every model that claims a context of 100K+ has an "effective context" on RULER far smaller than the advertised figure.** For example, a model may claim to support 200K, but on RULER accuracy falls below 80% past 50K, and at 128K it is already close to random.

That pulls the industry back to reality: **long context is not a replacement for RAG; it is a complement.**

### Long context vs. RAG: a decision framework

In *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool* [41], Tianpan gives a decision framework that was mainstream in 2026:

| Dimension | 1M-context model | RAG |
|---|---|---|
| **Latency** | 30–60× slower (the entire window must be prefilled) | Two stages, retrieval + generation, but the generation-stage input is small |
| **Cost per query** | ~1250× more expensive | Retrieval is nearly free; generation cost scales with the number of relevant chunks |
| **Data updates** | Retransmitted on every query | The index is updated once, and every query benefits |
| **Explainability** | Black box | Retrieved results can be audited and cited |
| **Multi-fact recall** | ~60% (in practice) | With a reranker, 90%+ is reachable |
| **Scenarios that truly suit long context** | A one-shot deep analysis ("read this 200-page PDF and summarize the insights") | Knowledge-base Q&A, customer support, document retrieval |

**Practical guidance:** 90% of "knowledge retrieval" scenarios should use RAG; only the 10% that are "deep analysis of a single document" should use long context.

---

<a id="s2-4"></a>

## 2.4 Full-stack RAG choices

RAG (Retrieval-Augmented Generation) is the most mature branch of context engineering, and it has already evolved into a complete sub-discipline. The state of 2025–2026 can be summarized as four generations of evolution [43]:

- **Generation 1: Naive RAG** (2020–2022) — retrieve → concatenate → generate
- **Generation 2: Advanced RAG** (2023–2024) — adds query rewriting / reranking / hybrid search / metadata filters
- **Generation 3: Modular RAG** (2024) — indexing / pre-retrieval / retrieval / post-retrieval / generation are each independently replaceable
- **Generation 4: Agentic RAG** (2025–2026) — the agent decides on its own when to retrieve, what to retrieve, and how to judge the result

The sections below unpack the 2026 design space in RAG pipeline order.

<a id="s2-4-1"></a>

### 2.4.1 Document parsing (Parsing & Layout)

Real-world documents are PDFs, slide decks, scans, Excel, and HTML, not plain text. This step is often skipped, and its impact is very large:

- **Baseline:** `pypdf` / `pdfplumber` (fast, but error-prone on tables and multi-column layout)
- **Advanced:** `unstructured` / `LlamaParse` (with a layout model that can recognize headings, paragraphs, tables, and lists)
- **OCR:** `PaddleOCR` / `Tesseract` / `EasyOCR` (scans and images)
- **Native multimodal:** have GPT-4o / Gemini / Claude read PDF pages directly (vision). Suited to documents with extremely complex layout, but costly
- **Dedicated commercial products:** Azure Document Intelligence, AWS Textract, Microsoft MarkItDown

Best practice: **run at least two parsers and compare them**, then decide which to use. Tables and formulas must be handled specially (convert to Markdown / LaTeX).

<a id="s2-4-2"></a>

### 2.4.2 Chunking strategies (Chunking)

The chunking strategy affects recall quality far more than the choice of embedding model:

- **Fixed Size Chunking:** a fixed N tokens; the simplest. Problem: it easily cuts semantic units apart
- **Recursive Character Splitting:** recursively split at the paragraph → sentence → word levels. LangChain's default. A baseline with a good cost/quality tradeoff
- **Semantic Chunking:** use embeddings to compute similarity between adjacent sentences, and split where similarity drops sharply
- **Document-Specific:** split Markdown on headings, code on functions, and HTML on the DOM
- **Agentic Chunking:** let an LLM read the document and decide how to split it. The most expensive, and the most accurate
- **Late Chunking** (Jina, 2024): embed the whole document first to obtain token-level embeddings, then chunk and aggregate. In long-context settings, recall is 10–15% higher than traditional chunking

General recommendation: **a chunk size of 256–512 tokens and an overlap of 50–100 tokens** is the sweet spot for most tasks.

<a id="s2-4-3"></a>

### 2.4.3 Embedding models (Embedding Models)

The main contenders on the April 2026 MTEB (Massive Text Embedding Benchmark) leaderboard [13]:

| Model | MTEB | Dimensions | Price / 1M tok | Notes |
|---|---|---|---|---|
| **OpenAI text-embedding-3-large** | 64.6 | 3072 (truncatable) | $0.13 | Established and stable |
| **Cohere embed-v4** | 65.2 | 1536 | $0.10 | Current MTEB #1 |
| **Voyage 3 large** | ~66 | 1024/2048 | $0.18 | On average ~10% higher than OpenAI |
| **Google Gemini Embedding 2** | 64+ | 3072 | $0.006 | Native multimodal (text/image/video/audio/PDF), 100+ languages |
| **BGE-M3** (BAAI) | 64+ | 1024 | Free (open source) | The same model supports dense / sparse / multi-vector retrieval |
| **Jina v5-text-small** | 71.7 (v2 leaderboard) | 512 | Free (Apache 2.0) | 677M parameters; best quality-to-size ratio |

Two key trends in 2026:

1. **Matryoshka Representation Learning (nested representations):** the same embedding is usable at different truncation lengths (128 / 256 / 512 / 1024 / 3072). Store the full length when indexing, and truncate on demand when retrieving. Gemini 2 / Voyage 4 / Cohere v4 / OpenAI text-3-* / Jina v5 / Microsoft Harrier / Nomic v1.5 all support it. "Coarse filter, then fine rank" can be implemented at the embedding layer itself.
2. **Native multimodal:** Gemini Embedding 2 encodes images / video / audio / PDFs into the same semantic space. The indexing complexity of multimodal RAG drops from "a separate index per modality + cross-modal alignment" to "a unified index."

Measured on Chinese workloads: **BGE-M3 is the best free option** (open source + strong multilingual performance + the same model supports sparse/dense), and **Voyage 3 large** is the best paid option (Chinese recall is generally 10%+ higher than OpenAI).

<a id="s2-4-4"></a>

### 2.4.4 Vector databases (Vector Database)

The five mainstream systems in 2026 production deployments [14]: **pgvector / Qdrant / Weaviate / Milvus / LanceDB**.

Performance benchmark (2026, 128-dimensional embeddings, on the order of 1M vectors) [14]:

| Database | p99 latency | QPS (single node) | Notes |
|---|---|---|---|
| Qdrant | 2 ms | 12,000 | Rust implementation; strongest metadata filtering |
| FAISS (in-memory) | 3 ms | 15,000 | Single-machine performance leader; no service-deployment capability |
| Milvus | 5 ms | 8,000 | Distributed + high availability; version 2.5 hybrid search is 30x faster than ES |
| Pinecone | 8 ms | 5,000 | Fully managed; zero operational burden |
| Weaviate | 10 ms | 4,000 | Built-in GraphQL and knowledge-graph capability |

Selection decisions:

- **Already on Postgres** → `pgvector` 0.9 has caught up in performance, avoiding the complexity of operating a second database
- **Need complex metadata filters** (by time / category / permission) → Qdrant (the ACORN algorithm solved the long-standing problem of filtered HNSW)
- **Want vectors + a knowledge graph** → Weaviate
- **Enterprise-grade + massive scale + high availability** → Milvus
- **Want embedded (no service required)** → LanceDB / Chroma
- **Minimal, fast prototype** → Chroma
- **Want to stay on an existing Elasticsearch / OpenSearch stack** → they support vectors themselves

<a id="s2-4-5"></a>

### 2.4.5 Retrieval: sparse + dense + late interaction

Vector retrieval (dense) is good at semantic similarity and weak at exact keyword matching (product IDs, proper nouns, typo tolerance). **Production-grade RAG is almost always hybrid search:**

- **Sparse:** BM25 / SPLADE / BGE-sparse (keywords + term frequency)
- **Dense:** embedding-based vector search (semantics)
- **Late interaction:** ColBERT / ColBERT-v2. Each token is embedded separately, and retrieval aggregates by token-level max-similarity. Recall is higher than single-vector dense, but the index is 100x larger. Vespa and Qdrant already support it.

Hybrid strategy: merge the sparse and dense result lists with RRF (Reciprocal Rank Fusion). Simple, and no parameter tuning.

**The RRF formula:**

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

where `R` is the set of retrievers, `r(d)` is the rank of document `d` in that retriever (1, 2, 3, ...), and `k` is a smoothing constant (default 60, Cormack et al. 2009). Intuition: a better rank contributes more, but because the term is a reciprocal the gaps converge quickly, which keeps any one retriever from dominating the list.

**Weighted RRF** (production practice): default RRF treats every retriever with equal weight. To give dense and sparse different weights:

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

**RRF vs. other fusion methods:**

| Method | Needs tuning? | Score alignment across retrievers | When to use |
|---|---|---|---|
| **RRF** | No (k=60 is nearly universal) | Uses rank, not score, so alignment is natural | Default first choice |
| **CombSUM / Linear** | Yes (scores must be normalized) | Hard (BM25 vs. cosine have different units) | Not recommended unless calibration already exists |
| **WAND (Weighted AND)** | Medium | Used inside sparse retrieval, not across fusion | Speeds up inverted indexes in Lucene / Elasticsearch |
| **Convex Combination + ML** | High (requires learning-to-rank) | Alignment is trained | Best at large scale, when click data is available |

Empirical result: **hybrid search + a reranker improves MRR by +9.3 percentage points over semantic-only**, and with Cohere Rerank accuracy can reach 90%+.

<a id="s2-4-6"></a>

### 2.4.6 Reranking (Reranker)

After first-stage retrieval (embedding / BM25) recalls the top 100, a more expensive but more accurate cross-encoder reranks the query–document pairs, and the top 5 are given to the LLM. The ROI of this step is extremely high:

| Reranker | Type | ELO | Notes |
|---|---|---|---|
| **Zerank 2** | API | 1638 | Current leader |
| **Cohere Rerank v4 Pro** | API | 1629 | Established |
| **Cohere Rerank 3 Nimble** | API | — | Speed-optimized version |
| **BGE-reranker-v2-m3** | Self-hosted | — | Open source; quality close to Cohere |
| **FlashRank** | Self-hosted | — | Extremely lightweight; millisecond-scale |
| **ColBERT v2** | Self-hosted | — | Late-interaction architecture |
| **RankLLM** (rerank directly with an LLM) | LLM | — | Most expensive, but the most explainable |

**Top rerankers improve precision by 15–40% over embeddings alone.** If you are allowed to add only one RAG optimization, adding a reranker is usually the highest ROI.

<a id="s2-4-7"></a>

### 2.4.7 Query rewriting / expansion (Query Rewriting / Expansion)

A user asking "今天天气咋样" and a document that says "current weather conditions" mean the same thing, but the vocabularies do not overlap. Query rewriting processes the query before retrieval:

- **HyDE (Hypothetical Document Embeddings):** first have the LLM imagine "what an answer document would say," and retrieve with that imagined text (not with the original query)
- **Query2Doc** (Microsoft, 2023): a variant of HyDE. Have the LLM directly generate a "pseudo-document passage that contains the answer" and use it as query expansion
- **Multi-Query / Query Expansion:** have the LLM rewrite the original query into 3–5 different phrasings, retrieve with each, then merge
- **Step-Back Prompting** (Zheng et al., 2024): first abstract the query into a higher-level question ("what did Python 3.12 change about the GIL" → "what the GIL is and how it works"), and use the retrieval results for the abstract question as the setup
- **Sub-Query Decomposition:** split a compound question into sub-questions and retrieve each separately ("compare A and B" → ["characteristics of A", "characteristics of B"])
- **The combination of HyDE + Multi-Query** is one of the strongest current baselines on retrieval benchmarks such as BEIR

**How to choose** (decide by the shape of the query):

| Query shape | Recommended method | Why |
|---|---|---|
| Short query / missing keywords | HyDE / Query2Doc | A short query does not match the length of a document chunk; expanding it into a pseudo-document pulls the embeddings closer |
| Polysemy / synonym problems | Multi-Query | Rewrites cover different phrasings and raise recall |
| A narrow question that depends on background knowledge | Step-Back | Retrieve the broad knowledge first, then focus on the detail |
| Compound / comparison / multi-hop | Sub-Query Decomposition | A single retrieval cannot cover multiple subtopics |
| Ordinary conversation + code / numbers | Do not rewrite | An LLM rewrite can change key terms or numbers, and recall drops |

**Pitfalls:**

1. Query rewriting adds 1 extra LLM call (latency +500–1500 ms). High-QPS settings need to evaluate that.
2. If the original query is already precise ("OpenAI Q3 2025 revenue"), rewriting may introduce noise. Add an "is_specific" classifier and judge first.
3. HyDE actually works poorly in **obscure / long-tail / specialized domains**—the wording of the LLM's imagined pseudo-document diverges sharply from the real documents.

<a id="s2-4-8"></a>

### 2.4.8 GraphRAG

Microsoft GraphRAG (open-sourced in 2024) and community variants such as LightRAG / Nano-GraphRAG extend retrieval from "vector similarity" to "knowledge-graph traversal":

- **Build phase:** use an LLM to extract entities and relations in the documents and build a knowledge graph
- **Retrieval phase:** map the query onto entities in the graph, then run a local subgraph query + summary
- **Advantage:** it can answer "topic-level" questions ("what compliance risks are there across all of our vendor contracts"). Traditional RAG fails on this kind of question because there is no locally similar chunk
- **Cost:** graph construction is very expensive (an LLM extracts entities from every passage). It suits static, high-value knowledge bases

In some scenarios GraphRAG pushes retrieval precision to 99% (relation-dense domains such as financial compliance, law, and medicine) [43].

<a id="s2-4-9"></a>

### 2.4.9 Self-RAG / CRAG / Agentic RAG

- **Self-RAG** (Asai, 2023): during generation, the model itself decides **when** to retrieve, **evaluates** whether the retrieved results are relevant, and **critiques** whether its own output is grounded
- **CRAG (Corrective RAG):** after retrieval, label each chunk "high / medium / low confidence"; low confidence triggers a web-search fallback
- **Agentic RAG:** wrap all of the capabilities above into an agent loop, and let the agent cycle through "think → retrieve → judge → rewrite the query → retrieve again → synthesize" until it is confident. LangGraph is the mainstream framework for implementing Agentic RAG

**Side-by-side comparison of the three:**

| Dimension | Self-RAG | CRAG | Agentic RAG |
|---|---|---|---|
| **Decides when to retrieve** | The model itself (special token `[Retrieve]`) | Always retrieves | The agent controller decides |
| **Judging retrieval results** | The model itself (`[IsRel]` reflection token) | An independent retrieval evaluator (a small model; high / medium / low) | An LLM judge or a tool |
| **Failure fallback** | Stop retrieving + signal uncertainty | Trigger a web search and retrieve again | Rewrite the query / switch data sources / multi-round iteration |
| **Training cost** | High (requires SFT on the reflection tokens) | Low (plug-and-play) | 0 (runtime orchestration) |
| **Latency** | Medium (one inference emits extra tokens) | High (always retrieve + evaluate + possibly the web) | Highest (possibly a multi-round loop) |
| **Fits** | General Q&A where the model's own knowledge is sufficient | Knowledge bases with poor freshness that need a web fallback | Multi-hop reasoning, research Q&A, Deep Research |
| **Mainstream deployments in 2026** | OpenAI gpt-4o web-search mode | The internal architecture of Perplexity / You.com | The research agents in Claude / Cursor / Devin |

Empirical results: in production deployments, Self-RAG can reduce "unnecessary retrieval" by 25–40%—because the model learns not to retrieve for things it already knows. On multi-hop QA (HotpotQA / MuSiQue), Agentic RAG is +15–25 points more accurate than vanilla RAG, but token cost multiplies by 3–10×.

See [agent-engineering.en.md](./agent-engineering.en.md) §2 on how the pattern evolved, and §5 on the framework comparison.

<a id="s2-4-10"></a>

### 2.4.10 Multimodal RAG

Indexing schemes that mix images and text:

- **Late fusion:** embed text and images separately; at retrieval time, retrieve each and then merge
- **Early fusion:** use a multimodal embedding such as CLIP / Gemini Embedding, and place images and text in the same space
- **VLM-as-extractor:** use GPT-4o / Claude 3.5 Sonnet to turn images into detailed descriptions, then run text RAG (highest cost, but the most stable recall)
- **ColPali / ColQwen2** (2024): run late-interaction embeddings directly on **full-page PDF screenshots**, skipping OCR / parsing. On documents with tables, charts, and multi-column layout, retrieval accuracy is +20–40% over traditional OCR + text RAG

**Layout-aware chunking** (a key engineering point, and underestimated):

A traditional RecursiveCharacterTextSplitter works poorly on laid-out documents such as PDF / Word—tables are split in half, figure captions are separated from the figures, and the table of contents is mixed with the body. Production options:

| Tool | Approach | Strength |
|---|---|---|
| **Unstructured.io** | Rules + ML element classification | General purpose; full coverage of HTML/PDF/PPT/email |
| **LlamaParse** | LlamaIndex's own LLM parser | Markdown output; high table fidelity |
| **PyMuPDF + pdfplumber** | Low-level PDF SDK | Custom layout rules; preserves bbox coordinates |
| **Docling** (IBM, 2024) | Open-source structured document parsing | Associates tables / formulas / image captions |
| **MinerU** (OpenDataLab, 2024) | Friendly to Chinese PDFs | Works well on Chinese academic papers and textbooks |

**Multimodal RAG selection:**

- Documents dominated by **charts / screenshots / scans** → ColPali / ColQwen2; skip OCR
- Documents dominated by **table-heavy PDFs** (financial reports / specifications) → Docling / LlamaParse + Markdown chunking
- Documents dominated by **mixed-layout slide decks / marketing material** → VLM-as-extractor (have Claude / GPT-4o describe each page)
- The input is **a user-uploaded image + a text question** → native multimodal Gemini Embedding 2 / Cohere Embed v4

**Pitfall:** multimodal embeddings are much better at cross-modal retrieval (finding an image from text) than at cross-modal alignment (whether the image and the text are actually about the same thing). The generation stage still needs a VLM to confirm relevance a second time.

<a id="s2-4-11"></a>

### 2.4.11 A complete RAG pipeline skeleton

Threading the 10 subsections above together, a minimal runnable skeleton of a production-grade RAG pipeline:

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

This skeleton covers the 7 core steps of RAG. Taking it to production still requires adding:

- **An eval closed loop:** run RAGAS for faithfulness / context_precision / answer_relevancy
- **Observability:** integrate Langfuse to capture a trace of every retrieval + generation
- **Caching:** an embedding cache + an LLM response cache
- **Degradation:** a fallback for retrieval failure, a fallback for rerank failure, and a fallback for LLM failure

<a id="s2-4-12"></a>

### 2.4.12 Systematic GitHub learning resources (RAG)

| Resource | Type | Recommended use |
|---|---|---|
| [`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques) | A collection of advanced-technique notebooks | 30+ advanced RAG techniques, each with a runnable notebook; the most complete |
| [`infiniflow/ragflow`](https://github.com/infiniflow/ragflow) | Open-source RAG engine | When you want an enterprise RAG platform that works out of the box |
| [`HKUDS/LightRAG`](https://github.com/HKUDS/LightRAG) | EMNLP 2025 paper + implementation | A reference for a concise, efficient RAG paradigm |
| [`Danielskry/Awesome-RAG`](https://github.com/Danielskry/Awesome-RAG) | awesome list | An entry point when looking for resources on a specific topic |
| [`lehoanglong95/rag-all-in-one`](https://github.com/lehoanglong95/rag-all-in-one) | Full-stack guide | When you want to compare tools by component |
| [`pguso/rag-from-scratch`](https://github.com/pguso/rag-from-scratch) | From-scratch implementation | When you want to understand what is actually happening inside the black box |
| [`microsoft/graphrag`](https://github.com/microsoft/graphrag) | Official GraphRAG | The reference implementation of entity-relation graph RAG |
| [`abhinav-kimothi/A-Simple-Guide-to-RAG`](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG) | Companion to an introductory book | An introduction for complete beginners |

---

<a id="s2-5"></a>

## 2.5 Memory systems: a complete engineering guide

If RAG solves "how to get an **external knowledge base** into context," a memory system solves "how to get the **agent's own experience** into context." The two share pieces of engineering (both are retrieval + injection into the prompt), but they differ in kind:

| Dimension | RAG | Memory system |
|---|---|---|
| Data source | External static documents (company wiki, PDFs) | Conversation history, agent behavior, user preferences |
| Write trigger | Once, at index time | Every interaction may trigger an incremental write |
| Data lifecycle | Follows the source document | Has its own "expiration / eviction / summarization" logic |
| Isolation granularity | Global or department-level | Must be per-user / per-agent / per-session |
| Freshness | Usually minutes to days | Can be seconds ("what the user just said") |
| Consistency | Strong (one ground truth) | Weak (facts change, and there are contradictions) |

<a id="s2-5-1"></a>

### 2.5.1 Why a context window alone is not enough

"The context window is already 1 M. Do we still need a memory system?" is a real question. Three reasons:

1. **Cost and latency:** one inference over a 1M context is substantially slower and more expensive than RAG on the equivalent task (a single source [41] estimates roughly 30–60× slower and ~1000× more expensive; **the exact multiple varies widely by model and task, and you should measure it on your own workload**)
2. **Lost-in-the-Middle:** stuffing all of the history into context lets the middle black hole swallow the critical facts (U-shaped recall)
3. **Persistence across sessions:** a context window lives for a single inference and is destroyed when the session ends; a memory system has to last across days, across weeks, and across devices

A memory system is, in essence, **"turning all past experience into a small retrievable database, and on each inference retrieving only the few most relevant items to inject into context"**—replacing "feed everything to the model" with "the model retrieves on demand."

<a id="s2-5-2"></a>

### 2.5.2 An engineering map of memory types

A classification borrowed from cognitive psychology. Each type corresponds to a different engineering implementation:

| Memory type | Cognitive meaning | Engineering implementation | Storage backend | When written | When retrieved |
|---|---|---|---|---|---|
| **Working memory** | The "scratch paper" of the current task | The current context window | The LLM's internal KV cache | Every token | Every token |
| **Short-term** | The most recent N turns | A rolling message buffer | RAM / Redis | Every turn | Injected in full every turn |
| **Episodic** | "What happened when" | Timestamped conversation fragments | Vector store + time index | End of session / important events | By topic + time window |
| **Semantic** | Stable facts about the world / the user | KV pairs or graph nodes | Graph store / KV store / SQL | When a new fact is extracted | By entity name |
| **Procedural** | "How to do this kind of task" | Executable skills / procedure scripts | Filesystem / skill registry | Summarized after the task succeeds | By task type |
| **Long-term** | All of the types above, across sessions | The union of the types above | Usually a combination of several backends | Async / background worker | On demand |

**Key insight:** a production agent's "memory" is usually **not a single system**, but some combination of subsets of the 6 types above. Letta (MemGPT) puts short-term + semantic memory in context (Core Memory), episodic memory in a vector store (Recall Memory), and external knowledge in another vector store (Archival Memory)—the classic three layers of an OS-inspired design.

<a id="s2-5-3"></a>

### 2.5.3 Core architectural components of a memory system

Taken apart, any memory system is these 5 components:

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

Each component has its own engineering decision points. §2.5.4–2.5.8 below walks through them one by one.

<a id="s2-5-4"></a>

### 2.5.4 Choosing a storage backend

A memory system's storage backend is not "pick one" but "pick which ones to combine." Common options:

| Backend | Strengths | Weaknesses | Typical use |
|---|---|---|---|
| **Vector store** (Qdrant / Weaviate / Pinecone / pgvector) | Semantic similarity retrieval | Weak at exact match / relationship queries | Primary store for episodic memory |
| **Graph store** (Neo4j / FalkorDB / Memgraph) | Entity relations / multi-hop queries / time | Complex to deploy / slow retrieval | Semantic memory + relations + time |
| **KV store** (Redis / DynamoDB) | Extremely low latency / simple | No semantic retrieval | Short-term working memory + cache |
| **Relational** (Postgres + pgvector / Tiger Data) | Transactional / SQL filters / one-stop | Tuning a large-scale vector index is painful | An integrated stack at small-to-medium scale |
| **Filesystem** (Anthropic Memory Tool) | Simple / auditable / agent-managed | Retrieval depends on the agent grepping for itself | Native to the Claude ecosystem |
| **Dedicated memory SaaS** (Mem0 Cloud / Zep Cloud / Supermemory) | Zero operations | Lock-in / data leaves your boundary | Startup MVP |

#### Selection decision tree

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

#### Measured performance (2026)

- **pgvectorscale:** 471 QPS at 99% recall on 50 M vectors (from the Tiger Data benchmark)
- **Redis vector:** p99 < 5 ms (the in-memory advantage)
- **Qdrant:** p99 ~2 ms; strong metadata filters
- **Neo4j graph traversal:** multi-hop queries in milliseconds, but full semantic retrieval is weaker than a pure vector store

<a id="s2-5-5"></a>

### 2.5.5 The write path: extraction + dedup + linking

Memory is **not simply storing the raw conversation**. It is "facts" extracted by an LLM. The Mem0 v2 write algorithm (paper [19]), broken down:

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

**Key design choices:**

- **Add-only vs. update-in-place:** Mem0 chooses add-only (new facts coexist with old facts; the retriever filters out what has expired); Zep chooses invalidate (old facts are marked invalid, and history is kept); Letta lets the agent decide (via the `core_memory_replace` tool)
- **When extraction fires:** every turn / once every N turns / end of session / triggered by an important event. Mem0 defaults to every turn, but it is configurable
- **Dedup granularity:** exact string match uses MD5 (Mem0); semantic near-duplicates use embedding distance (Letta)

Write pseudocode (a from-scratch version):

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

<a id="s2-5-6"></a>

### 2.5.6 The retrieval path: multi-signal fusion

Retrieval by embedding alone will miss things. Production memory systems in 2026 all use **multi-signal fusion**—the same idea as hybrid retrieval in RAG. Mem0 v2 retrieval:

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

**Advanced techniques:**

- **Personalized reranking:** train a reranker on the feedback signal "after it was retrieved, did the LLM actually cite it?"
- **Time decay:** retrieval score × `exp(-λ·age)`; newer memories are preferred
- **Importance weighting:** at write time, have the LLM assign each memory an importance score (0–1), and weight by it at retrieval
- **Cold-start fallback:** when the user has just started and memory is empty, return empty rather than noise

<a id="s2-5-7"></a>

### 2.5.7 Eviction: 6 forgetting policies

Memory explosion is a real problem. The MaRS architecture (arXiv 2603.07670) formalizes 6 mainstream eviction policies:

| Policy | Formula / rule | Fits |
|---|---|---|
| **FIFO** | By insertion time | The simplest; almost unused (it deletes important old facts) |
| **LRU** | By last-access time | The default starting point for a desktop agent |
| **Priority Decay** | `score = w₁·prior + w₂·recency + w₃·frequency` | The best general-purpose baseline |
| **Reflection-Summary** | The old N items → an LLM summarizes them into 1 item | Used together with LRU; essential for a long-running agent |
| **Random-Drop** | Delete at random | Only as a baseline for comparison |
| **Hybrid** | LRU + Priority Decay + summary used together | Recommended for production |

**A practical recipe** (Letta and Mem0 are both close to this idea):

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

**Rules of thumb:**

- Desktop agent / simple chatbot → start with pure LRU
- Long-running agent / multi-user SaaS → Priority Decay
- Need traceability / legal scenarios → Hybrid + do not delete the originals when summarizing (archive them to cold storage)

<a id="s2-5-8"></a>

### 2.5.8 A deep look at the four mainstream systems

#### A. Mem0 — a general-purpose memory layer (easiest to adopt)

**Architecture:** 5 pillars (summarized from paper [19] and the Mem0 docs):

1. **LLM-Powered Fact Extraction:** a single LLM call turns the conversation into structured facts
2. **Vector Storage:** Qdrant by default; 24+ backends can be swapped in (Chroma / Pinecone / pgvector / MongoDB / Weaviate / OpenSearch / and others)
3. **Graph Storage** (optional): Neo4j / Neptune / FalkorDB / Memgraph, storing entity relations
4. **Hybrid Retrieval:** semantic + BM25 + entity matching, scored in parallel, then fused with RRF
5. **Production Infrastructure:** batching, MD5 dedup, and a three-level hierarchy of user / session / agent

> Note: the 5 items above are a checklist arranged by engineering responsibility, not an official architecture diagram from any of Mem0 / Letta / Zep. What the Mem0 paper actually describes is four operations, Add/Update/Delete/NoOp, plus an LLM extraction pipeline; Letta uses three layers, Core/Recall/Archival; Zep uses three node kinds, Episodic/Entity/Edge.

**Performance** (the v2 algorithm, in the terms reported by the Mem0 paper arXiv:2504.19413): on long-conversation memory benchmarks such as LoCoMo / LongMemEval it is substantially better than the RAG / full-context baselines (the size of the gain varies considerably by subtask and by metric; see Tables 4–5 of the original paper); extraction latency is cut in half.

**Complete code example:**

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

**Production architecture recommended by AWS:** Mem0 + ElastiCache for Valkey (short-term + cache) + Neptune Analytics (graph) + OpenSearch (vectors).

**Fits:** general chatbots, customer support, and personal assistants—the "add memory in 5 lines of code" case.

#### B. Letta (formerly MemGPT) — OS-inspired, agent-managed

**Architecture:** treat the LLM as an OS and divide memory into three layers (the core innovation):

| Layer | Name | Where | Capacity | Who can change it |
|---|---|---|---|---|
| L1 | **Core Memory** | **In-context** (inside the system prompt) | Fixed (for example 4 KB) | The agent itself, via a tool |
| L2 | **Recall Memory** | External DB | The full conversation history | Automatic |
| L3 | **Archival Memory** | Vector store | Unlimited | The agent itself, via a tool |

**Core Memory** is the key innovation—it **is not a database** but **part of the system prompt**. The agent can change it with the `core_memory_append` / `core_memory_replace` tools the way it would edit a file, and on the next inference that content automatically appears at the head of the context. This is equivalent to letting the agent **"keep editing its own persona."**

**6 built-in tools** (the agent manages its own memory):

| Tool | Role |
|---|---|
| `send_message` | Send a message to the user |
| `core_memory_append` | Append a new fact to in-context memory |
| `core_memory_replace` | Replace in-context memory (correct an error / update a preference) |
| `conversation_search` | Search conversation history (recall memory) |
| `archival_memory_insert` | Write to the long-term archive |
| `archival_memory_search` | Retrieve from the long-term archive |

**Complete code:**

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

**Fits:** long-running agents (days / weeks), products that need a "persistent identity" (a personalized assistant / an AI coach), and an OSS alternative to the Anthropic Memory Tool.

#### C. Zep (Graphiti) — Temporal Knowledge Graph

**Core innovation:** every fact carries **two time axes**:

- **Event time T:** when this fact was true in the real world ("Alice joined TechCorp in 2023-01")
- **Ingestion time T':** when Zep learned this fact (the conversation took place in 2024-03)

**Architecture:** 3 node kinds + edges

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

**Key capability:** when a fact changes ("Alice left in 2024-06"), Zep **does not delete** the old fact. It marks the old edge `valid_to=2024-06` and the new edge `valid_from=2024-06`. A query can then ask:

- "Which company is Alice at now?" → look up edges with `valid_to=NULL`
- "Where was Alice in 2024-04?" → look up edges with `valid_from <= 2024-04 <= valid_to`
- "Every employee TechCorp has ever had?" → edges with no time-window restriction

**Hybrid retrieval:** semantic embeddings + BM25 + direct graph traversal—**no LLM call**, pure index and graph queries, latency 100–500 ms.

**Minimal code** (Graphiti SDK):

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

**Fits:** scenarios where user state changes—CRM, customer support, medical history, compliance audit, and legal fact bases.

#### D. Anthropic Memory Tool — Filesystem-as-Memory

**Released to public beta on 2026-04-23.** Design philosophy: **treat memory as files**, and let Claude read and write them with the familiar `bash` + `code execution` tools.

**Architecture:**

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

**Complete code** (Python SDK, public beta):

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

**Early customer cases** (published officially by Anthropic):

- **Netflix:** content-metadata validation; **first-pass error rate down 97%**
- **Wisedocs:** medical-insurance document review; **30% faster**
- **Rakuten / Ando:** customer-support automation

**Fits:** teams already tied to the Claude ecosystem, cases that need "an agent managing its own notebook the way a programmer would," and cases that need audit / version control (memory files can go into git).

#### E. Others worth watching

- **Cognee** ([github.com/topoteretes/cognee](https://github.com/topoteretes/cognee)): a graph-native memory framework with a strong ontology
- **Supermemory** ([supermemory.ai](https://supermemory.ai)): a compatibility layer for the Anthropic Memory Tool, across vendors
- **Redis Agent Memory Server:** from Redis; semantic + keyword + hybrid in one place
- **AgentCore Long-Term Memory:** built into AWS Bedrock

#### Side-by-side comparison matrix (2026)

| Dimension | Mem0 | Letta | Zep / Graphiti | Anthropic Memory Tool |
|---|---|---|---|---|
| Data model | facts (vector + graph) | core / recall / archival | temporal KG | filesystem |
| Agent initiative | passive (auto extract) | **agent-managed tools** | passive | **agent-managed tools** |
| Multi-model support | 16+ LLMs | Any | Any | Claude only |
| Backend choice | 24+ vector + graph | Postgres / built-in | Neo4j / FalkorDB | Any filesystem |
| Temporal tracking | Weak (timestamp-based) | Weak | **Strong (two time axes)** | File mtime |
| Relationship modeling | Medium (optional graph) | Weak | **Strong (graph-native)** | Weak |
| Cost to get started | **5 lines** | 50 lines | 30 lines | 10 lines |
| Open source | ✓ | ✓ | ✓ (Graphiti) | × |
| Chinese support | Pair it with a Chinese embedder | Same | Same | Native |
| Papers / numbers | LoCoMo +20pt | MemGPT paper | arXiv 2501.13956 | Netflix error rate -97% |

<a id="s2-5-9"></a>

### 2.5.9 Multi-tenant isolation: a required course for production

**The pitfall the vast majority of SaaS products will hit:** the memory system leaks across users. Two layers of defense:

#### Layer 1: force user_id on both writes and queries

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

#### Layer 2: namespace / collection / DB isolation at the storage backend

| Scheme | Isolation strength | Performance | Scale it fits |
|---|---|---|---|
| Shared collection + `user_id` filter | Weak (depends on the application layer) | Good | < 1k users |
| One collection per user | Strong | Medium (the number of collections has a ceiling) | 1k–10k |
| One namespace per user (Pinecone / Turbopuffer) | **Strong** | **Good** | 10k–1M+ |
| A separate DB instance per tenant | Strongest (physical isolation) | Medium | Large customers / compliance |

Vector stores designed specifically for multi-tenancy, such as **Turbopuffer**, can support **a million namespaces**, with each tenant's data strictly isolated inside.

#### Layer 3: audit + GDPR compliance

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

<a id="s2-5-10"></a>

### 2.5.10 Memory vs. RAG: the boundary and how they work together

In practice, **memory and RAG often coexist**, and they do not conflict:

```
User Query
    │
    ├──▶ RAG（公司文档 / 产品手册 / 政策） → 通用知识
    │
    └──▶ Memory（用户偏好 / 历史交互 / agent 经验） → 个性化

       两路检索结果合并 → 注入 system prompt → LLM
```

**A typical production prompt structure:**

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

<a id="s2-5-10-5"></a>

### 2.5.10.5 Progress files: "git memory" for long-running agents

Vector memory / KV memory handles "facts," but a **long-running coding agent / research agent** also needs a structured, human-readable, diffable memory of work in progress—that is the **progress file** (the de facto standard used by Augment Code, Claude Code, and Devin).

**A typical schema** (Markdown; common names are `PROGRESS.md` / `CONTEXT.md` / `.agent/state.md`):

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

**Why this format rather than vector memory:**

- **A person can review it:** diff it directly in the PR; no need to open a vector DB
- **Git can manage it:** each agent decision lands in a commit, and rollback is supported naturally
- **An LLM can read and edit it directly:** no embedding / retrieval middle layer, which avoids recall failure
- **Friendly to restart across sessions:** the agent pulls git → reads `PROGRESS.md` → and immediately knows where it left off

**Key engineering points:**

1. **When to write:** append after every "decision point" or tool-call result. Do not wait until the session ends (a power loss drops it).
2. **Merge conflicts:** concurrent writes from multiple agents should use append-only sections + timestamps, or be split into per-agent files.
3. **OOM / token explosion:** compact on a schedule—fold decisions older than N days into a summary kept at the head of the file.
4. **Working with vector memory:** the progress file is the source of truth; vector memory only indexes chunks that point at the progress file.
5. **Failure-recovery protocol:** on startup the agent first runs `git status + git log -5 + cat PROGRESS.md`. That trio is the lowest-cost way to recover "who I am, where I left off, and what is next."

References: Anthropic's *Memory Tool*, the Augment Code engineering blog, Claude Code's `.claude/` directory, Cursor's `.cursorrules`, and Codename Goose's `goose hints`.

<a id="s2-5-11"></a>

### 2.5.11 Failure modes and countermeasures

| Failure mode | Symptom | Countermeasure |
|---|---|---|
| **Memory explosion** | After 1,000 items are stored, retrieval is all noise | LRU + summarization (§2.5.7) |
| **Stale memory** | "I like cold coffee" has expired but still takes effect | Zep time axes / conflict detection at write time / the user overrides it |
| **Hallucinated memory** | The LLM extracts as fact something that was never said | A confidence threshold at extraction + the user can audit |
| **Cross-user leak** | A's preferences are injected into B's prompt | Force user_id (§2.5.9) |
| **Cold start** | A new user's memory is empty, and the agent seems "unfamiliar" | Explicit onboarding + a generic-profile fallback |
| **Privacy / compliance** | GDPR / CCPA require deletion of user data | A `forget_user` API that cascade-deletes across the stack |
| **Extraction cost** | Calling an LLM to extract facts every turn doubles token cost | An importance gate skips irrelevant conversation + batch every N turns |
| **Low retrieval recall** | The LLM does not see the critical fact | Add BM25 + entity match (§2.5.6 multi-signal) |
| **Memory corruption** | See the section below | Periodic consistency checks + ground-truth calibration + a TTL |

#### Memory corruption in depth

"Memory corruption" is a failure mode that **inevitably appears** after a stateful agent has been deployed for more than a few weeks, and early engineering teams rarely prevent it. Four typical kinds of corruption that the Anthropic / Letta / Mem0 teams have discussed in public:

| Corruption type | Example | Root cause | Defense |
|---|---|---|---|
| **Stale fact** | "The user is in Hangzhou" → three months later they have moved to Shanghai, but the old memory is still in the top-K | No TTL, and no write-time check of "fact freshness" | Zep-style valid_from / valid_to; at write time, detect old facts about the same entity and mark them invalid |
| **Contradiction** | One item says "the user is vegetarian," another says "the user likes steak" | Extraction across multiple sessions has no cross-session consistency check | Before writing, run a contradiction-detection LLM on "this user's existing facts + the new fact"; or a periodic reconciliation job |
| **Hallucinated fact** | During extraction the LLM "infers" from the conversation a preference the user never stated | The extraction prompt does not restrict itself to "explicit statements only" | At extraction time, return an evidence span (a quote from the original conversation); do not store anything without evidence |
| **Persona drift** | Long-running self-updating memory makes the agent's self-description seriously inconsistent with the initial system prompt | Agent-managed memory + no frozen section | Split core memory into immutable / mutable sections; periodically reset the mutable section to the baseline |

**Defensive engineering practice** (ordered by ROI):

1. **TTL + temporal fields:** every memory carries `created_at` / `valid_from` / `valid_to`, and retrieval filters by the current time
2. **Contradiction detection at write time:** the new fact vs. the top 3 existing facts about the same entity; run one LLM judgment of "is this a contradiction?"; when it is, a human or a timestamp decides
3. **Evidence-required extraction:** the extraction prompt must return `(fact, evidence_quote, confidence)`; confidence < 0.7 is not stored
4. **A weekly reconciliation job:** randomly sample 1% of memories and have an LLM compare them with the real conversation; if the corruption rate exceeds a threshold, trigger cleanup
5. **Explainable + auditable:** the user can see the original text and the source of "what the agent remembers about me," and delete a mistake in one click

<a id="s2-5-12"></a>

### 2.5.12 Practical GitHub resources (memory systems)

| Resource | Type | Recommended use |
|---|---|---|
| [`mem0ai/mem0`](https://github.com/mem0ai/mem0) | Official repository | A general memory layer; integrate in 5 lines |
| [`letta-ai/letta`](https://github.com/letta-ai/letta) | Official repository | OS-inspired stateful agent |
| [`getzep/graphiti`](https://github.com/getzep/graphiti) | Official repository | Temporal KG engine (Zep's core) |
| [`getzep/zep`](https://github.com/getzep/zep) | Official repository | The full Zep service (built on Graphiti) |
| [`redis/agent-memory-server`](https://github.com/redis/agent-memory-server) | Official repository | Redis one-stop agent memory |
| [`topoteretes/cognee`](https://github.com/topoteretes/cognee) | Official repository | Graph-native + ontology |
| [`supermemoryai/supermemory`](https://github.com/supermemoryai/supermemory) | Official repository | Cross-vendor memory SDK |
| Anthropic Memory Tool docs | Official documentation | `https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool` |

**Papers to read:**

- [`Mem0 paper`](https://arxiv.org/abs/2504.19413) — Production-Ready AI Agents with Scalable Long-Term Memory
- [`MemGPT paper`](https://arxiv.org/abs/2310.08560) — Towards LLMs as Operating Systems
- [`Zep paper`](https://arxiv.org/abs/2501.13956) — A Temporal Knowledge Graph Architecture for Agent Memory
- [`MaRS paper`](https://arxiv.org/abs/2603.07670) — Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers (a 2026 survey)
- [`Forgetful but Faithful`](https://arxiv.org/abs/2512.12856) — Cognitive Memory Architecture and Benchmark

---

<a id="s2-6"></a>

## 2.6 Context-management strategy: treat the context window as a scarce resource

<a id="s2-6-1"></a>

### 2.6.1 Why "management" is needed

Even at 1M tokens, the context window is still a **scarce resource** — for three reasons:

1. **Cost**: one 1 M token prompt is ~$1 (Claude Opus); a few thousand calls a day and you go broke
2. **Latency**: 1 M token prefill takes ~30 seconds, and the user experience collapses
3. **Quality**: Lost-in-the-Middle; in multi-fact settings recall drops to ~60% (§2.2.3)

So a production-grade agent does not "fill the 1 M." It **spends tokens deliberately**: every token should have a reason to appear in the prompt.

<a id="s2-6-2"></a>

### 2.6.2 A "partition" model of context

Treat the context window as a fixed space with logical partitions. Each partition has its own priority and its own eviction policy:

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

**Suggested quota allocation** (by product type):

| Product type | System | Memory | RAG | Tools | History | Tool results | User |
|---|---|---|---|---|---|---|---|
| Customer-support chatbot | 10% | 15% | 30% | 5% | 20% | 10% | 10% |
| Coding agent | 5% | 5% | 40% | 10% | 10% | 25% | 5% |
| Research agent | 5% | 10% | 30% | 5% | 10% | 35% | 5% |
| Personal assistant | 15% | 30% | 10% | 10% | 25% | 5% | 5% |

Production experience: **any partition above 50% is a signal that the design is wrong**.

<a id="s2-6-3"></a>

### 2.6.3 Five common actions in context management (synthesized from Anthropic / LangChain practice)

> The splits given by the two authoritative industry sources are not "5." Anthropic's *Effective Context Engineering* [6] gives **three** strategies: compaction / structured note-taking / sub-agent architectures. LangChain's *Context Engineering for Agents* [7] gives **four**: write / select / compress / isolate. The five actions below merge the two splits and expand them, so an implementation can check them item by item. They are not an industry-standard taxonomy.

#### Action 1: Priority ranking

Use primacy + recency bias: put the most important content at the head and the tail, and the secondary content in the middle.

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

#### Action 2: Dynamic trimming

Periodically check token utilization, and trigger when it crosses the threshold:

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

#### Action 3: Summarization — implementing `/compact`

Claude Code's `/compact` command compresses the current long conversation into a summary plus the key facts, and reclaims tokens:

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

**The key point**: the summary must be generated by an LLM (rule-based extraction loses context), but the prompt for the LLM summary itself must explicitly require that critical information such as "code locations / numbers / decisions" be preserved.

#### Action 4: Scratchpad separation

Reasoning agents such as ReAct produce a large volume of thought / observation while thinking, and that should not pollute the outward-facing context:

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

LangGraph's state mechanism / Letta's core vs recall are both productizations of this idea.

#### Action 5: Tool-result clipping

JSON returned by tools is often explosively long (for example, one SQL query of 1000 rows). Two layers of filtering:

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

<a id="s2-6-4"></a>

### 2.6.4 Three "compression" paths compared

| Method | Compression ratio | Information loss | Latency | Fit |
|---|---|---|---|---|
| **Simple truncation** | medium | high | 0 | tool results / fallback |
| **Rule-based extraction** (regex / structured) | medium | low | 0 | known formats (JSON / tables) |
| **LLM summary** (`/compact`) | **high** | **low** | 1-3 seconds | long conversations / high-value history |
| **Anchored Iterative Summary** (2026 SOTA) | **high** | **lowest** | 3-10 seconds | extremely long tasks (agents running for hours) |

#### When to trigger compaction (decision signals)

Compaction is not "compress only once the context is full." Otherwise the LLM has already been contaminated by noise. Production practice triggers on multiple signals:

| Trigger signal | Threshold (reference) | Where it applies |
|---|---|---|
| **Token occupancy** | > 70% of max_tokens (for example 180K / 256K = 70%) | general fallback, the most commonly used |
| **History turns** | > 50 conversation turns | long-conversation customer support / companion products |
| **Accumulated tool results** | a single-step tool result > 4000 tokens, or a cumulative total > 30K | coding agent / data agent |
| **Quality-signal decline** | user satisfaction / verifier pass rate over the last 5 turns drops | products that have a reward signal |
| **Task boundary** | a subtask completes (the agent declares it itself) | long-running multi-step agents |
| **Time decay** | a historical message goes > 30 minutes with no citation | agents running multiple tasks in parallel |
| **Anthropic Auto-Compact signal** | Claude API `usage.context_window_remaining` < 20% | products on Claude read the official signal directly |

**Key principles (Anthropic, *Effective Context Engineering*, 2026)**:
1. Compaction is **lossy**. Avoid it whenever you can (do actions 1-5 of §2.6.3 first).
2. Always keep the **last N turns of the original conversation** (N=3-5). The LLM needs recent raw signal to judge "what just happened."
3. The summary should keep **pointers** — decisions, open questions, unfinished subtasks, code / file paths — rather than tell a story.
4. Immediately after the summary, run an evaluator once and confirm the critical facts were not lost. If they were, roll back to raw and upgrade to a stronger compression strategy.
5. **Compaction itself has to be traced** — what was dropped, the token diff before and after compression, and whether a rollback fired all go into observability.

<a id="s2-6-5"></a>

### 2.6.5 Production recipe: a complete Context Manager class

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

Wire `metrics()` into Langfuse / LangSmith, and the dashboard shows "which kind of content is eating the context quota" — the most practical tool for diagnosing agent drift.

<a id="s2-6-6"></a>

### 2.6.6 Prompt / context caching: the largest cost lever of 2026

**Every modern LLM provider supports prompt caching** — the "prompt prefix" is cached at the KV cache layer and reused the next time the same prefix appears. **This is the single largest cost-optimization opportunity of 2025-2026, with a discount of up to 90%.**

#### How the three major providers implement it

| Provider | How it triggers | Hit discount | Minimum length for a hit | TTL | Write |
|---|---|---|---|---|---|
| **Anthropic** | explicit (`cache_control` marker) | **90%** (cached input $0.30 vs base $3.00 / Sonnet) | 1024 tokens (Sonnet) / 2048 (Haiku) | 5 min (default) / 1h (paid) | 1.25× base |
| **OpenAI** | automatic (a prefix of ≥1024 tokens is cached automatically) | **50-90%** (depends on the model) | 1024 tokens | 5-10 min | 0 (writes are free) |
| **Google Gemini** | implicit (on by default for 2.5+) + explicit API | implicit about 50% / explicit about 75% | 4096 tokens | default 1h | billed by storage |
| **AWS Bedrock** | translated for every provider | same as each provider | same | same | same |

Empirical result (public data from ProjectDiscovery): once prompt caching is wired up, **the monthly LLM bill drops 59-70%**. Academic work (arXiv 2601.06007) on long-horizon agent tasks reports **API cost down 41-80% and TTFT down 13-31%**.

#### Which prompts are a fit for caching

Score by "reuse frequency × length":

| Content type | Worth caching? |
|---|---|
| **System prompt + persona** | ★★★ strongly recommended |
| **A large set of few-shot examples** | ★★★ |
| **Tool-definition schemas (10+ tools)** | ★★★ |
| **Fixed documents retrieved by RAG (for example an SLA / policy handbook)** | ★★ |
| **Conversation history that changes** | ✗ do not cache (cache breakdown) |
| **A user query that differs every time** | ✗ |

**Rule of thumb**: **if a prefix's repeat rate is > 50%, cache it; otherwise do not.**

#### Anthropic explicit-caching code

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

#### Hard constraints on a cache hit

1. **Strict prefix match.** From the start through the `cache_control` marker, the text must be **identical** — one extra space is a miss.
2. **Order-sensitive.** Put the fixed parts (system / tools / RAG) first, and the variable parts (the user message) last.
3. **The TTL is short.** The default is 5 min, so if no second call arrives within 5 minutes the cache write was wasted — this fits a "high QPS, the same system prompt" workload.
4. **Cache writes are billed too.** The first write costs 1.25×, so **it pays off only after at least 2 hits**.

#### Choosing explicit vs. automatic caching

- **OpenAI automatic**: zero changes to adopt it, but you do not know which prefixes hit, and it is not controllable.
- **Anthropic explicit**: the SDK call has to change, but it is **fully controllable** — you can know exactly how much was saved.
- **Production recommendation: Anthropic.** Mark system / tools / large RAG blocks with `cache_control`, and watch the cache hit rate alongside observability.

#### "Don't Break the Cache": the key to a long-running agent task

When an agent runs a long-horizon task, the prompt is "a prefix (the system) + conversation history that keeps being appended." **Appending new content does not break the prefix cache**, so the prefix portion keeps hitting:

```
Round 1: [system] [user1]                          → write cache（system）
Round 2: [system] [user1] [bot1] [user2]           → cache HIT 前缀，付钱仅 [bot1][user2]
Round 3: [system] [user1] [bot1] [user2] [bot2] [user3] → cache HIT 更长前缀
...
```

**The reverse**: if you insert / modify / delete anything in the middle (for example, a fresh RAG retrieval stuffing different chunks into the system prompt), the cache is invalidated immediately, and the next call is billed in full again.

A hidden best practice of agent design: **put "content that changes" at the tail; the prefix is only ever appended to, never modified.**

#### Monitoring cache effectiveness

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

On the dashboard, watch:

- **Cache hit rate** (`cache_read` / `total_input`) — target > 60%
- **Cost per request, compared** (before enabling vs. after enabling) — target a drop of 50%+
- **TTFT, compared** — target a drop of 20%+

---

<a id="s2-7"></a>

## 2.7 The MCP / A2A protocol ecosystem

<a id="s2-7-1"></a>

### 2.7.1 Why MCP is needed: solving the "M×N integration nightmare"

Wiring tools and data sources into an LLM has long been dirty work:

- Each LLM provider's function-calling schema is different (OpenAI / Anthropic / Google)
- Every tool needs a hand-written wrapper, and several copies have to be maintained
- A tool upgrade means every agent has to be re-integrated
- Switching LLMs means rewriting the whole set of wrappers

The mathematical model: M LLMs × N tools = M·N adapters. 10 LLMs × 50 tools = 500 integration points.

**MCP (Model Context Protocol)** [9] is an open standard Anthropic released in November 2024. It splits M·N into M+N:

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

**The analogy**: before USB-C, every phone had its own cable (M·N); after USB-C, any host connects to any peripheral (M+N). MCP is the USB-C of AI.

<a id="s2-7-2"></a>

### 2.7.2 The MCP protocol specification

#### Three classes of primitives

The capabilities an MCP server exposes to a client fall into three classes:

| Primitive | Meaning | Example | Who triggers it |
|---|---|---|---|
| **Tools** | an executable function, with side effects | `create_issue`, `run_sql` | **the model decides to call it** |
| **Resources** | read-only data, addressed by URI | `file:///doc.md`, `db://table/users` | **the application preloads it** |
| **Prompts** | a template the user can invoke | `/explain-code`, `/summarize` | **the user triggers it** |

These three classes correspond to the three data flows of an LLM application: the model calls on its own initiative, the application injects, and the user invokes.

#### Three transports

| Transport | Where it fits | How it is deployed |
|---|---|---|
| **stdio** | local inter-process communication (the default for Claude Desktop / Cursor / Cline) | a child process, configured in `mcp_servers.json` |
| **SSE** (Server-Sent Events) | a remote service, one-way stream | an HTTP server, long-lived connection |
| **Streamable HTTP** (the 2025 replacement for SSE) | a remote service, bidirectional stream, more standard | an HTTP server; the de facto standard |

<a id="s2-7-3"></a>

### 2.7.3 A complete MCP server implementation example

#### Python (FastMCP)

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

#### TypeScript (`@modelcontextprotocol/sdk`)

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

<a id="s2-7-4"></a>

### 2.7.4 Connecting an MCP client

#### Claude Desktop / Cursor / Cline configuration

`~/.config/claude/mcp_servers.json` (Mac/Linux), or a similar configuration in Cursor / Cline:

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

Restart the client, and every tool / resource / prompt is available automatically.

#### Using it in LangChain / LangGraph

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

<a id="s2-7-5"></a>

### 2.7.5 The mainstream MCP server ecosystem (2026)

Pick by category:

| Category | Repository | Use |
|---|---|---|
| **Official reference implementations** | [`modelcontextprotocol/servers`](https://github.com/modelcontextprotocol/servers) | dozens that run out of the box: filesystem / git / github / postgres / sqlite / slack / google-drive / brave-search / sequential-thinking |
| **Awesome list** | [`punkpeye/awesome-mcp-servers`](https://github.com/punkpeye/awesome-mcp-servers) | a community server index (hundreds) |
| **Databases** | postgres / mysql / sqlite / mongodb / redis each have an official or community MCP |
| **Cloud services** | aws-mcp / gcp-mcp / azure-mcp |
| **DevOps** | kubernetes / terraform / docker MCP |
| **Productivity** | notion / linear / jira / confluence MCP |
| **Browsers** | playwright-mcp / browserbase-mcp / Puppeteer MCP |
| **Commercial enterprise** | Zapier MCP (5000+ integrations) |

<a id="s2-7-6"></a>

### 2.7.6 MCP timeline and adoption

| Time | Event |
|---|---|
| 2024-11 | Anthropic publishes the MCP protocol |
| 2025-Q1 | IDEs such as Cursor / Cline integrate it |
| 2025-Q2 | OpenAI announces support |
| 2025-Q3 | Google / Microsoft / Amazon follow |
| 2025-Q4 | MCP governance moves to an independent **MCP Steering Committee** (jointly governed by Anthropic + Block + OpenAI + Microsoft and others) |
| 2026-02 | Monthly downloads of the Python + TypeScript SDKs exceed **97 million** |
| 2026-Q1 | Nearly every mainstream LLM client supports it natively |

<a id="s2-7-7"></a>

### 2.7.7 A2A: the protocol family between agents

MCP solves communication between an agent and its tools. **A2A (Agent-to-Agent)** solves communication between agents. This area is still a contest among several protocols:

| Protocol / framework | Origin | Paradigm | Status |
|---|---|---|---|
| **Google A2A** | Google 2025 | HTTP-based agent discovery + task delegation | early |
| **Microsoft Agent Framework Protocol** | Microsoft 2026 | successor to AutoGen; a standard for multi-agent conversation | being promoted |
| **Anthropic Agent Skills** | Anthropic 2026 | Skill package: a reusable unit of agent capability | early |
| **MCP for A2A** | community | treat "another agent" as an MCP server | in practice |
| **Letta multi-agent** | Letta | via message passing | inside the framework |

**The analogy between MCP and A2A**:

> "**MCP gives an agent a pair of hands** (connecting tools / data). **A2A lets multiple agents team up and do the work.** MCP is vertical integration (a single agent ↔ a tool stack). A2A is horizontal collaboration (agent ↔ agent network)."

The state of affairs in spring 2026: MCP is already the de facto standard, and A2A is still in its Cambrian explosion. **Practical advice**: building a single agent now → go all-in on MCP; building multiple agents → use the built-in protocol of a concrete framework (LangGraph / Microsoft Agent Framework), and wait until the A2A standard is clear.

<a id="s2-7-8"></a>

### 2.7.8 Common MCP design traps

| Trap | Symptom | Countermeasure |
|---|---|---|
| **Too many tools** | 50+ tools make the LLM's decision hard | group + route (pick a category first, then a tool) |
| **Tool description too short** | the model does not know when to use it | write "when to use / when not to use" + 1-2 examples |
| **Side effects left undeclared** | the model treats the call as read-only and causes an unexpected modification | the description states explicitly that it "will modify X", and the client side adds deny-first |
| **Return value too long** | one return of 10K rows blows up the context | server-side cap + pagination + summary |
| **Synchronous blocking** | a long task freezes the client | use an async transport (streamable-http) |
| **Authentication is a mess** | each tool handles its own token | the server reads tokens uniformly from env; the client uses an OAuth flow |

Details of tool integration at the agent-implementation layer → [agent-engineering.en.md](./agent-engineering.en.md#s3-1) and [agent-engineering.en.md](./agent-engineering.en.md#s5);
infrastructure at the LLM gateway / observability / inference-server layer → [harness-engineering.en.md](./harness-engineering.en.md#s3-9) appendix (note: chapter 04 explicitly marks these 4 kinds as "related but not harness engineering").

---

<a id="bibliography"></a>

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

Additional:
- Asai et al. (2023). *Self-RAG*.
- Microsoft GraphRAG. <https://github.com/microsoft/graphrag>
- LightRAG. <https://github.com/HKUDS/LightRAG>
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
- BGE-M3. <https://huggingface.co/BAAI/bge-m3>

---

<a id="cross-references"></a>

## Chapter cross-references

- To understand the Transformer physics behind long context → [prompt-engineering.en.md](./prompt-engineering.en.md#s1-1)
- To understand how Agentic RAG / memory is used inside an agent → [agent-engineering.en.md](./agent-engineering.en.md#s3) + [agent-engineering.en.md](./agent-engineering.en.md#s4)
- To understand KV cache / prefix caching at the inference-server layer → [harness-engineering.en.md](./harness-engineering.en.md#s3-9-2)
