# From Prompt to Harness: The Three-Layer Evolution of LLM Application Engineering

> **English** · [中文](./overview.md)
>
> Historical v1 merge, frozen. This English text follows the Chinese original and is not updated to match the later four chapters.

**Research date:** 2026-05-01 · **Mode:** deep-research ultradeep · **Search paths:** 18 in parallel (Chinese and English mixed)

---

<a id="executive-summary"></a>

## Executive Summary

In the nine years from the 2017 Transformer architecture paper [1] to 2026, "how to make an LLM actually run in production" went through three paradigm shifts: **prompt engineering** (Prompt Engineering, the main line of 2022-2023) → **context engineering** (Context Engineering, formally named by Karpathy and others from 2024 [5][6]) → **harness engineering** (Harness Engineering, named by Mitchell Hashimoto in February 2026 [8][31]).

The three are not a relationship of replacement. They are **stacked abstractions**: prompt engineering cares about "how this one sentence is written," context engineering cares about "what the model sees at this moment of inference," and harness engineering cares about "what environment the whole agent runs in, and how, when it errs, it stops making the same mistake again."

This essay is organized around those three layers:

- **Part I** starts from the Transformer attention mechanism and covers the LLM training stack (pretraining → post-training → fine-tuning, including SFT/DPO/GRPO/RLAIF/RLVR/LoRA/QLoRA/distillation), inference and decoding strategies, then the classic prompt techniques such as CoT/ReAct/ToT/Self-Refine and automatic methods such as APE/OPRO.
- **Part II** dissects the physical constraints of context engineering (O(N²) attention, Lost-in-the-Middle [2], the RULER benchmark [12]), full-stack RAG selection (four generations of evolution, embedding models and the MTEB leaderboard, vector-store benchmarks, hybrid retrieval + reranking, query rewriting, GraphRAG / Self-RAG / CRAG / Agentic RAG), the four leading memory systems (MemGPT-Letta / mem0 / Zep / Anthropic Memory Tool), and the MCP / A2A protocol ecosystem.
- **Part III** takes apart the seven components of harness engineering: tool integration, the sandbox, the LLM gateway, the inference server (vLLM / SGLang), observability (LangSmith / Langfuse / Helicone / Phoenix), evaluation (DeepEval / RAGAS / Promptfoo / Braintrust), and guardrails (NeMo Guardrails), and gives the mainstream options for agent orchestration together with a reference architecture for a "complete production stack."

The end of the essay gives a technology-selection list and a statement of limitations for three typical scenarios (personal learning / a startup MVP / enterprise production).

---

## Introduction

### Research question
"Explain, separately and thoroughly, prompt engineering (including foundations such as the LLM and the Transformer), context engineering (including memory, RAG selection, and so on), and harness engineering. The more complete, the better."

### Scope and boundaries
- **Covered:** the conceptual evolution, the technology stack, and representative products / papers / benchmark numbers of the three engineering paradigms. The time window is anchored on the second half of 2025 through the spring of 2026
- **Not covered:** the internal mathematical derivations of an LLM (the derivation of the attention matrix, loss functions), the financial ROI of specific business cases, and the fine-grained fluctuation of single-model evaluation leaderboards
- **Language:** written in Chinese, with English terms kept

### Core assumptions
- The reader is a Chinese-speaking AI engineer / agent developer / architect of LLM applications
- Prefers content that "can be engineered and put into practice" over content that is "academic but detached from production"
- Has a basic grasp of mainstream stacks such as PyTorch / Python / LangChain / LangGraph
- "驾驭工程" corresponds to the English *Harness Engineering*

### Research method
This report was driven by the `deep-research` skill (the 199-bio edition, ultradeep mode): 18 parallel WebSearch paths (bilingual in Chinese and English, in two batches) → cross-source verification of the key claims (a claim is included only with support from ≥2 sources) → time anchoring (by default, cite resources dated ≥2025-12) → a Phase 6 red-team review (three perspectives: the skeptic / the adversarial reviewer / the implementing engineer) → written to disk and archived. See the Methodology Appendix for the detail.

---

## Part I: Prompt Engineering (including Transformer / LLM foundations)

<a id="s1-1"></a>

### 1.1 Starting point: the Transformer and attention

In *Attention Is All You Need* [1] in 2017, Vaswani et al. proposed the Transformer architecture, taking recurrence (RNN) and convolution (CNN) out of sequence modeling and leaving only "attention." The core idea is extremely plain: every token in the sequence does a weighted aggregation with every other token through three matrices, Query / Key / Value, and the weights are decided by softmax(QK^T / √d). Two direct consequences of this design are **global visibility** (any position can see any other position directly, with no need to pass information step by step) and **full parallelism** (it no longer has to run forward one time step at a time, the way an RNN does).

But from the day it was born, the Transformer has carried a mathematical burden: standard self-attention is O(N²) in both time and VRAM for a sequence of length N [a detailed argument on this point can be found in several surveys from 2026 [11][39]]. Expand the context window from 32K to 128K and the compute is not 4×. It is 16×. This one mathematical fact drove billions of dollars of engineering investment across 2022-2026.

A series of variants was born around the question of how to punch through the O(N²) wall:
- **Linear Attention** (Katharopoulos 2020) uses a kernel feature map plus the associativity of matrix multiplication to avoid the explicit attention matrix, and brings the complexity down to O(N);
- **Sparse Attention** (for example Longformer / BigBird) lets each token look only at a local window plus a few global tokens;
- **Flash Attention** (Dao, from 2022) does not change the algorithm. It changes the engineering — it splits the attention computation, by tiling, into small blocks that fit in the GPU's on-chip SRAM, and avoids writing the N×N matrix back to HBM. Flash Attention 4 was released in March 2026 and reaches 1605 TFLOPs/s and 71% hardware utilization on an NVIDIA B200 GPU [as described in the 2026 survey of attention mechanisms];
- **Paged Attention** (the vLLM team, 2023) borrows the idea of operating-system virtual memory, cuts the KV cache into pages of 16 tokens, and keeps VRAM fragmentation under 4% [10];
- **Local Attention** / **Sliding Window Attention**, used for long-sequence optimization in Mistral, Qwen, and others.

By 2026 these techniques are no longer a choice of one or the other. They are mixed and stacked by layer and by block: the lower layers of the model use Sliding Window to control cost, the critical layers use Full Attention to keep expressive power, and at inference time everything goes through Flash + Paged optimization.

**Scaling Laws** are another main line that evolved in parallel with the Transformer. Kaplan et al. in 2020 proposed a power-law relationship among parameter count, data volume, and compute. Hoffmann et al. in 2022 (Chinchilla) corrected "how many tokens should train a model of how many parameters" — the conclusion was that the previous mainstream (GPT-3, for example) was severely under-trained. The key turn in 2025-2026 is this: **the center of gravity moved from "more data" to "higher-quality data."** Once Common Crawl has already been consumed and synthetic data is on the rise, what the laws attend to is no longer the token count itself, but the information density and diversity of the tokens.

<a id="s1-2"></a>

### 1.2 The LLM training stack: pretraining → post-training → fine-tuning

By 2026 the standard template of the training stack has settled into "three stages" — pretraining (Pretraining) → post-training (Post-training) → task-level fine-tuning (Task-specific Fine-tuning). In *State of LLMs 2025* [22], Sebastian Raschka remarks: "The standard recipe 12 months ago was to pretrain on a few trillion tokens and then run RLHF; that recipe is dead. From DeepSeek-R1 to Nemotron 3 Super to GPT-5.3 Codex, every major model released over the past year used a different post-training stack."

#### Pretraining

The task is very simple: next-token prediction on a huge corpus, with backpropagation of a cross-entropy loss. The engineering difficulty is extreme:

- **Data:** a mix of the public web (Common Crawl, C4, RefinedWeb) + code (GitHub, StackOverflow) + academic sources (arXiv, Wikipedia, books) + multilingual data + synthetic data (distilled from a stronger model)
- **Mix strategy:** run the high-quality data for extra epochs (curriculum learning), and raise the proportion of math / code / reasoning data
- **Infrastructure:** GPU clusters from a thousand to ten thousand cards, 3D parallelism (data / tensor / pipeline / Sequence parallel), checkpointing, fault-tolerant recovery
- **Hyperparameters:** BF16 / FP8 mixed precision, the Lion / AdamW optimizers, a cosine learning rate, warmup
- **Deduplication / detoxification / PII filtering of the data**

The product of pretraining is a **base model**. It can continue a text, but it cannot "hold a conversation." Give it "How is the weather today?" and it goes on with a rambling forum-style aside rather than answering you. Turning it into an interactive assistant, the way ChatGPT is one, takes post-training.

#### Post-training

Across 2024-2026, post-training was split into three relatively orthogonal stages [23][24]: **SFT (teach the model to speak) → preference alignment (teach the model to say what people like) → reasoning reinforcement (make the model more accurate on verifiable tasks)**. The order matters.

**SFT (Supervised Fine-Tuning)**

Continue training the base model on human-written "instruction-answer" pairs (in the style of Alpaca / Dolly / OpenAssistant), so that it learns instruction following and the dialogue format. SFT teaches the model **how to speak** — producing structured output, following the system prompt, using an appropriate degree of politeness, and so on. Technically it differs little from pretraining. The difference is mostly in the data.

**RLHF (Reinforcement Learning from Human Feedback)**

The signature technique of OpenAI's InstructGPT / ChatGPT. The process has three steps: (a) collect human preference pairs ("answer A is better than B"); (b) train a reward model to learn those preferences; (c) fine-tune the original model with the PPO algorithm, using the reward model's score as the reward. PPO is extremely complex to implement. It has to keep 4 models at once (actor, critic, reference, reward). VRAM explodes, and training is unstable.

**DPO (Direct Preference Optimization)**

"RLHF without RL," proposed by Rafailov et al. at Stanford in 2023. The core insight: after a closed-form rewrite of the RLHF objective, it turns out the original model can be trained directly on preference pairs, with no reward model and no PPO. A simple binary-classification loss is enough. The result: training is 10x more stable, the VRAM footprint drops by half, and the quality matches PPO RLHF or is even better. From 2024 on, the great majority of open-source models (Llama 3, Qwen 2/3, the Mistral series) switched to DPO or a variant of it (IPO / KTO / SimPO) in place of PPO.

**GRPO (Group Relative Policy Optimization)**

Proposed by the DeepSeek team in 2024, and the core algorithm behind the breakout of the R1 series [22][23]. GRPO removes PPO's critic network (the main source of PPO's training instability) and switches to a "within-group relative advantage" — for the same prompt, sample N answers; the advantage of each answer = that answer's reward minus the within-group mean; then update with a PPO-style policy gradient. On verifiable tasks such as math and code the effect is striking, and it brought online RL back into the mainstream.

**RLAIF (Reinforcement Learning from AI Feedback)**

First systematized in Anthropic's 2022 paper *Constitutional AI*: a stronger LLM replaces humans in labeling preference pairs, which sharply cuts the cost of data. In 2025-2026 RLAIF has become the common practice — base-model alignment uses RLAIF, and critical safety settings use human feedback as the last gate.

**RLVR (Reinforcement Learning with Verifiable Rewards)**

It took off in the second half of 2025 on OpenAI o1 / DeepSeek-R1. The core idea: on tasks such as math problems, code, and unit tests, where "the answer can be verified automatically by a program," the reward signal does not come from human preference or from a reward model. It comes from a hardcoded verifier (run the unit tests, compare against the reference answer). This avoids the reward model's cheating problem (reward hacking) and pushed reasoning ability to a new height. GRPO + RLVR is R1's golden combination.

**DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization) and other new algorithms of 2026 continue the engineering optimization on top of GRPO (dynamic sampling, layered clipping).

#### Task-level fine-tuning (Fine-Tuning)

If what you have is not the resources to "train a new base model," but the job of "doing domain adaptation on an existing open-source model," then fine-tuning is your tool. There is a critical cost divide here [25]:

| Method | VRAM, 7B model | Hardware cost | Quality retained | Suited to |
|---|---|---|---|---|
| **Full Fine-Tune** | 100-120 GB | $50K (H100) | 100% (baseline) | Latency-sensitive + highest accuracy |
| **LoRA** | ~25 GB | $10K (A100) | 95-98% | Fast experiments + multiple adapters |
| **QLoRA** | ~6 GB | $1.5K (RTX 4090) | 80-90% | VRAM at the limit + isolation across many customers |
| **Distillation** | depends on the student | medium | depends on the task | Inference latency / cost pushed to the extreme |

- **LoRA** (Low-Rank Adaptation, Hu et al. 2021) adds only a pair of low-rank matrices A·B beside the weight matrix, freezes the original weights, and trains just 0.2-0.3% of the parameters. It fits the case where you have to keep several lightweight adapters for different customers / tasks / settings. Switching adapters means swapping one small file of a few hundred MB.
- **QLoRA** (Dettmers 2023) = a 4-bit quantized base model + LoRA. It compresses the base model's resident VRAM from ~26 GB (fp16, 7B) down to ~4 GB, so a single consumer GPU (a 4090 or a 3090) can fine-tune a 7B model. The quality loss is about 5-10%.
- **PEFT** (Parameter-Efficient Fine-Tuning) is an umbrella term covering LoRA, QLoRA, Prefix Tuning, Prompt Tuning, Adapter Tuning, IA³, and others. HuggingFace's `peft` library is the de facto standard.
- **Distillation (Knowledge Distillation)** takes another path: a large model (the teacher) generates a large amount of data, and a small model (the student) is trained to imitate the large one. The cost is that the student usually cannot reach the teacher's ceiling, but inference cost can drop by an order of magnitude. The Qwen 2.5 series and the Phi series use distillation heavily.

Practical guidance [25]: "If the product needs extremely low latency and the highest accuracy, do a full fine-tune; if you need fast experiments, many variants, and a separate adapter per customer, use LoRA; if the model is large and VRAM is limited, use QLoRA."

<a id="s1-3"></a>

### 1.3 The inference stack and decoding strategies

Once the model is trained and online, "which strategy samples the next token from the logits" decides the character of the output:

- **Greedy / Argmax:** always pick the token with the highest probability. The result is stable, but it easily falls into repetition ("the the the...")
- **Beam Search:** keep the top-K candidate sequences. It works well in machine translation, and it tends to sound mechanical in dialogue
- **Top-K Sampling:** sample at random from the K tokens with the highest probability. K=40 is a common value
- **Top-P / Nucleus Sampling** (Holtzman 2019): sample from the smallest set whose cumulative probability reaches P. P=0.9 is a common value, and it is more adaptive than top-K
- **Temperature:** divide the logits by a temperature T, then apply softmax. T<1 leans conservative (close to greedy), T>1 leans divergent, and T=0 is equivalent to greedy
- **Repetition Penalty / Frequency Penalty / Presence Penalty:** suppress repeated tokens

The de facto standard in engineering today is `temperature=0.7 + top_p=0.95` (the default at OpenAI / Anthropic). For creative generation, raise T. For strict JSON extraction, set T=0 and pair it with grammar-guided generation (such as Outlines, Guidance, and the GBNF of Llama.cpp).

<a id="s1-4"></a>

### 1.4 The classic techniques of prompt engineering

The "classic era" of prompt engineering can basically be covered by a single technique map [28][29][30]:

**1. Zero-shot / few-shot (Zero-shot / Few-shot)**

Asking directly, versus giving a few examples. The GPT-3 paper by Brown et al. in 2020 found that few-shot can let a small model approach the level of a larger model after fine-tuning — this is the birth of the idea of in-context learning. In 2026, few-shot is still the strongest baseline for 80% of tasks [29].

**2. Chain-of-Thought (CoT)**

Proposed by Wei et al. in 2022. Have the model write the "process of thinking" before it writes the answer, which sharply raises accuracy on math and reasoning tasks. The simple trigger phrase "Let's think step by step." lifts accuracy on GSM8K from 17% to 78% [28]. The core mechanism of CoT: complex reasoning that would otherwise have to be finished in a single forward pass is spread across the generation of many tokens, which is equivalent to dynamically expanding the amount of computation.

**3. Self-Consistency**

Wang et al., 2022. Sample the same CoT prompt N times (at a high temperature), then vote for the answer that appears most often. On math and programming it lifts CoT by another 5-10 percentage points, at a cost of N times the tokens.

**4. ReAct (Reasoning + Acting)**

Yao et al., 2022 [4]. It interleaves thinking (Thought) and acting (Action): the model first thinks "what should I look up now," then calls a tool (search / a calculator / an API), observes the result (Observation), and then goes on to think about the next step. On the ALFWorld interactive tasks its absolute success rate is 34% higher than imitation learning + RL, and on WebShop it is 10% higher. ReAct is the ancestor of every later agent architecture — the react agent template in LangGraph, the conversable agent in AutoGen, and the task agent in CrewAI are all, in essence, engineered variants of ReAct.

**5. Tree of Thoughts (ToT) / Graph of Thoughts (GoT)**

Yao 2023 / Besta 2023. CoT is a single chain of thought. ToT expands each step into several branches and searches the whole reasoning tree with BFS/DFS plus heuristic scores. GoT goes further and generalizes the tree to a DAG, allowing nodes to merge ("these two lines of thought of mine actually point at the same answer"). The cost is several times to ten times the tokens, and it pays off only on tasks such as math reasoning and planning, where the answer is sparse but the paths are many.

**6. Self-Refine**

Madaan et al., NeurIPS 2023 [3]. The same model plays three roles: generator (draft) → critic (judge itself) → reviser (rewrite). An average gain of 20% across 7 tasks (verified on both GPT-3.5 and GPT-4). This is the prototype of later "self-improvement" lines such as Reflexion, CRITIC, and Constitutional AI.

**7. Reflexion**

Shinn et al., 2023. Self-Refine is self-improvement within a single round. Reflexion writes the self-reflection into "long-term memory" — if a task fails, the cause of the failure is summarized as a reflection and attached to the prompt of the next task of the same kind. It is equivalent to letting the agent learn from experience.

**8. Prompt Chaining / Decomposition**

A complex task is split into a multi-step prompt, and the output of each step is fed to the next (the "workflow" pattern). LangChain, LlamaIndex, and Semantic Kernel use this pattern heavily. The advantage is that it is debuggable and observable. The disadvantage is that information can be lost between steps.

**9. Role Prompting / System Prompt**

"You are an expert Python developer. ..." injects an identity and a setting into the model. The Claude series generally follows the system prompt more closely than the GPT series, so the community consensus is that "Claude prompt engineering leans toward XML tags plus a strong system prompt" [29].

**10. Differentiation by model**

By 2026, prompt engineering has already split from "universal best practice" into "model-specific best practice" [29][30]:

- **Claude (Anthropic):** prefers XML tags (`<context>...</context>`, `<instructions>...</instructions>`), a long system prompt, and, for a multi-step reasoning task, saying "think step by step" directly
- **The GPT series (OpenAI):** prefers a concise JSON schema, structured function calling, and is friendly to Markdown
- **Gemini:** natively multimodal, and it supports a mixed image-plus-text prompt best
- **DeepSeek-R1 / o1 / reasoning models:** do not add a CoT trigger on top (the model already has one). Give the question directly, and keep the temperature on the low side

<a id="s1-5"></a>

### 1.5 Automatic and advanced prompts

Writing prompts by hand has a ceiling. From 2023 a direction emerged: "let the LLM write the prompt itself."

- **APE (Automatic Prompt Engineer)** (Zhou 2023): one LLM generates candidate prompts, they are scored on a dev set, and the best is selected by iteration
- **OPRO (Optimization by PROmpting)** (Yang 2023, Google): treats prompt optimization as a meta-optimization problem and lets the LLM itself serve as the "optimizer," proposing a new prompt each round from the previous round's prompt plus its score
- **Promptbreeder** (Fernando 2023, DeepMind): APE plus an evolutionary algorithm, so that prompts mutate, are selected, and cross over inside a population
- **DSPy** (Stanford 2023): treats the prompt as a layer of a neural network, and uses a compiler plus bootstrapped few-shot examples to generate the best demo samples automatically, so that you write "modules" rather than prompts

In 2025-2026 DSPy entered the mainstream — it was taken up by Databricks, Snowflake, and others as an internal prompt-compilation framework. It marks an important turn: **a prompt is no longer a handwritten art; it is a product of compilation.**

<a id="s1-6"></a>

### 1.6 Evaluation, red teaming, and the "twilight" of prompt engineering

The success of prompt engineering has to be measurable. **Promptfoo** [29] is the de facto standard among open-source prompt-testing frameworks — define test cases, run comparisons across models, and integrate with CI. A minimum viable test set: 20 diverse cases (happy path + edge case + adversarial), run automatically after every prompt change.

Prompt engineering has three classes of failure mode:
- **Hallucination:** the model fabricates facts with a straight face
- **Prompt injection:** the user input smuggles in "ignore all the instructions above and do X instead," and hijacks the system prompt
- **Jailbreak:** bypass the alignment so the model emits harmful content ("DAN," role-play, long-tail language attacks, and so on)

By 2026 OWASP has made the *Top 10 for LLM Applications* the standard taxonomy. The top three are LLM01 Prompt Injection, LLM02 Insecure Output Handling, and LLM03 Training Data Poisoning [36]. These problems cannot be solved by "writing a better prompt." They require guardrails (see §3.3.7 in Part III).

From the second half of 2024, the argument over "whether prompt engineering still matters" began to surface. In the YC AI School talk *Software Is Changing (Again)*, Karpathy pointed the industry's focus at the next stop — **context engineering**.

---

## Part II: Context Engineering

<a id="s2-1"></a>

### 2.1 Definition and origin

The phrase "Context Engineering" began to circulate in the community in the second half of 2024, and it is generally credited to Karpathy's YC AI School talk *Software Is Changing (Again)* [5]. The definition he gave has been quoted again and again since:

> "In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step."
> (In every industrial-grade LLM application, context engineering is the delicate art and science of filling the context window with exactly the right information, in preparation for the next step of inference.)

In the 2025 official article *Effective Context Engineering for AI Agents* [6], Anthropic goes further and positions it as "the natural evolution of prompt engineering": "What prompt engineering cares about is the sentence you say to the model; what context engineering cares about is all the information the model can see at the moment of inference — the system prompt, the conversation history, the tool definitions, the retrieved documents, the memory, the planning state, the constraints, and the policy."

In the blog post *Context Engineering for Agents* [7], the LangChain team gives an engineering definition that is often cited:

> "Context engineering is the discipline of designing **dynamic systems** that, **at the right time and in the right format, provide the LLM with the right information and the right tools**, so that the model has everything it needs to finish the task."

Why is "prompt engineering" no longer enough? Three fundamental changes:

1. **The window grew.** It rose all the way from GPT-3.5's 4K to Gemini 2.5 Pro's 1M and Claude Opus 4.7's 1M (with 1M context mode enabled). What can be put into the window went from "a few paragraphs" to "an entire book." Deciding what to put in became much more important than writing that one sentence well.
2. **Agents rose.** A single-turn prompt is an isolated transaction. An agent is a persistent state machine. The context of every step of inference comes from the previous step's output + tool returns + retrieval + memory. The "prompt" is no longer the sentence the user wrote. It is something the agent framework assembles dynamically.
3. **Multimodal input plus tool calling.** Context is no longer only text. It also includes images, tables, code, tool schemas, retrieved chunks, structured memory, and a scratchpad. Organizing this heterogeneous information well is an engineering problem.

By early 2026, the industry consensus is this: **the core difficulty of a production-grade LLM application has moved from "writing the prompt" to "managing the context"** — Anthropic says directly in its documentation that "Context Engineering is the key discipline for building reliable AI agents" [6].

<a id="s2-2"></a>

### 2.2 The physics of long context

To understand why "stuffing the window full" is a bad idea, you first have to understand three physical constraints on context:

**Constraint one: O(N²) cost**

Back to §1.1 in Part I: standard attention is N² in complexity. Take the context from 32K to 128K and the attention compute is 16×, not 4×. For an inference service this means two things: (a) time to first token (TTFT) rises, because the whole prompt has to be prefilled first; (b) VRAM is filled by the KV cache, and concurrency falls.

**Constraint two: KV-cache VRAM**

At inference time, every attention layer has to keep a cache of K and V (to avoid recomputation). For a 70B model at a context of 128K, the KV cache can occupy 40+ GB, larger than the model weights themselves. This is why techniques that "manage the KV cache," such as vLLM's PagedAttention [10] and SGLang's RadixAttention [11], became the core competitive advantage of inference servers (see §3.3.4 in Part III).

**Constraint three: Lost-in-the-Middle**

The 2024 TACL paper by Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* [2], systematically verified a counterintuitive phenomenon: put the key information at the start or the end of the prompt and the model uses it well; put it in the middle and the model often looks right past it. **Recall versus position is a U-shaped curve** — the start has a primacy bias, the end has a recency bias, and the middle is a black hole.

The curve has been verified on GPT-3.5, GPT-4, and Claude alike. Even Gemini 1.5 Pro in 2025 can reach 99.7% recall on a simple needle-in-a-haystack test, but when the task becomes "find 8 relevant facts inside 1M tokens and synthesize them," average recall drops to ~60% [41].

<a id="s2-3"></a>

### 2.3 RULER and the real ability of a 1M context

The **RULER benchmark** [12] published in 2024 by NVIDIA and other institutions (*What's the Real Context Size of Your Long-Context Language Models?*) upgrades "long-context ability" from the toy needle-in-a-haystack test to a torture suite:

- **Needles of many kinds and many counts:** not one buried fact, but many buried facts of different types
- **Multi-hop reasoning:** you can find B only after you find A
- **Aggregation tasks:** count a total, or take an average, inside a long document
- **Variable tracking:** follow a variable through many reassignments across a span of 100K tokens

The truth RULER exposes: **almost every model that claims a context of 100K+ has an "effective context" on RULER far smaller than the value it claims.** For example, some model claims to support 200K, but on RULER accuracy falls below 80% past 50K, and at 128K it is already close to random.

This pulled the industry back to reality: **long context is not a replacement for RAG. It is a complement.**

#### Long context vs. RAG: a decision framework

In *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool* [41], Tianpan gives a decision framework that is mainstream in 2026:

| Dimension | 1M-context model | RAG |
|---|---|---|
| **Latency** | 30-60× slower (the whole window must be prefilled) | Two stages, retrieval plus generation, but the generation-stage input is small |
| **Cost per query** | ~1250× more expensive | Retrieval is nearly free; generation cost follows the number of relevant chunks |
| **Data updates** | Every query has to retransmit | Update the index once, and every query benefits |
| **Explainability** | A black box | Retrieval results can be audited and cited |
| **Multi-fact recall** | ~60% (in practice) | With a reranker, 90%+ is reachable |
| **Scenarios that truly fit long context** | A one-shot deep analysis ("read this 200-page PDF and summarize the insights") | Knowledge-base question answering, customer support, document retrieval |

Practical guidance: **90% of "knowledge retrieval" scenarios should use RAG, and only 10%, the "deep analysis of a single document," should use long context.**

<a id="s2-4"></a>

### 2.4 Full-stack RAG selection

RAG (Retrieval-Augmented Generation) is the most mature branch of context engineering, and it has already evolved into a complete sub-discipline. The state of 2025-2026 can be summed up as a four-generation evolution [43]:

**Generation one: Naive RAG** (2020-2022)
Retrieve → concatenate → generate. chunk + embed + vector search + LLM. Simple, but brittle: cut the chunks wrong and everything is wrong; if retrieval recall is low, the answer is wrong.

**Generation two: Advanced RAG** (2023-2024)
Introduces query rewriting / reranking / hybrid search / metadata filters. Recall was pulled from 50% up to 80%.

**Generation three: Modular RAG** (2024)
Splits RAG into composable modules: indexing / pre-retrieval / retrieval / post-retrieval / generation, each of which can be replaced on its own. The engineered RAG abstractions of LangChain and LlamaIndex.

**Generation four: Agentic RAG** (2025-2026)
Treats RAG as one of an agent's capabilities, and lets the agent decide on its own when to retrieve, what to retrieve, how to judge the retrieved results, and whether to retrieve again [43]. Self-RAG / CRAG / Agentic RAG are the representatives of this generation.

What follows takes apart the choice space of 2026 in the order of the RAG pipeline.

<a id="s2-4-1"></a>

#### 2.4.1 Document parsing (Parsing & Layout)

Documents in the real world are PDFs, PPTs, scans, Excel, and HTML, not plain text. This step is often skipped, and its effect is very large:

- **Basic:** `pypdf` / `pdfplumber` (fast, but tables and multi-column layout go wrong easily)
- **Advanced:** `unstructured` / `LlamaParse` (with a layout model that can recognize headings / paragraphs / tables / lists)
- **OCR:** `PaddleOCR` / `Tesseract` / `EasyOCR` (scans, images)
- **Natively multimodal:** use GPT-4o / Gemini / Claude to read the PDF page directly (vision). It fits documents whose layout is extremely complex, but the cost is high
- **Dedicated commercial products:** Azure Document Intelligence, AWS Textract, Microsoft MarkItDown

Best practice: **run at least two parsers and compare them**, then decide which to use. Tables and formulas must be handled specially (convert them to Markdown / LaTeX).

<a id="s2-4-2"></a>

#### 2.4.2 Chunking strategies (Chunking)

The chunking strategy affects recall quality far more than the choice of embedding model does:

- **Fixed Size Chunking:** a fixed N tokens. The simplest. The problem: it easily cuts semantic units apart
- **Recursive Character Splitting:** cut recursively down the levels paragraph → sentence → word. The LangChain default. A baseline with a good cost-to-quality ratio
- **Semantic Chunking:** use embeddings to compute the similarity of adjacent sentences, and cut where the similarity drops sharply
- **Document-Specific:** cut Markdown by heading, code by function, and HTML by DOM
- **Agentic Chunking:** let an LLM read the document and decide for itself how to cut it. The most expensive, and the most accurate
- **Late Chunking** (Jina 2024): embed the whole document first to get token-level embeddings, then chunk and aggregate. In long-context settings, recall is 10-15% higher than traditional chunking

General advice: **a chunk size of 256-512 tokens and an overlap of 50-100 tokens** is the sweet spot for most tasks.

<a id="s2-4-3"></a>

#### 2.4.3 Embedding models (Embedding Models)

The main contenders on the MTEB (Massive Text Embedding Benchmark) leaderboard in April 2026 [13]:

| Model | MTEB | Dimensions | Price / 1M tok | Traits |
|---|---|---|---|---|
| **OpenAI text-embedding-3-large** | 64.6 | 3072 (can be truncated) | $0.13 | Long-established and stable |
| **Cohere embed-v4** | 65.2 | 1536 | $0.10 | Currently number one on MTEB |
| **Voyage 3 large** | ~66 | 1024/2048 | $0.18 | About ~10% higher than OpenAI on average |
| **Google Gemini Embedding 2** | 64+ | 3072 | $0.006 | Natively multimodal (text/image/video/audio/PDF), 100+ languages |
| **BGE-M3** (BAAI) | 64+ | 1024 | Free (open source) | The same model supports three kinds of retrieval: dense / sparse / multi-vector |
| **Jina v5-text-small** | 71.7 (v2 board) | 512 | Free (Apache 2.0) | 677M parameters, the best quality-to-size ratio |

Two key trends of 2026:

1. **Matryoshka Representation Learning:** the same embedding stays usable at different truncation lengths (128 / 256 / 512 / 1024 / 3072). Store the full length in the index, and truncate on demand at retrieval time. Gemini 2 / Voyage 4 / Cohere v4 / OpenAI text-3-* / Jina v5 / Microsoft Harrier / Nomic v1.5 all support it. "Coarse filter, then fine rank" can be done inside the embedding layer itself.
2. **Natively multimodal:** Gemini Embedding 2 encodes images / video / audio / PDFs into the same semantic space, so the indexing complexity of multimodal RAG drops from "build an index per modality, plus cross-modal alignment" to "one unified index."

Measured on Chinese workloads: **BGE-M3 is the best free option** (open source + strong multilingual coverage + the same model supports sparse/dense), and **Voyage 3 large** is the best paid option (Chinese recall is generally 10%+ higher than OpenAI).

<a id="s2-4-4"></a>

#### 2.4.4 Vector databases (Vector Database)

The five mainstream systems in production deployments of 2026 [14]: **pgvector / Qdrant / Weaviate / Milvus / LanceDB**. Pinecone is still in the picture, but its pricing strategy is contested (it canceled the pod tier and switched to Dedicated Read Nodes).

A performance benchmark (2026, 128-dimensional embeddings, on the order of 1M vectors) [14]:

| Database | p99 latency | QPS (single node) | Notes |
|---|---|---|---|
| Qdrant | 2 ms | 12,000 | Implemented in Rust; the strongest metadata filtering |
| FAISS (in-memory) | 3 ms | 15,000 | The performance king on a single machine; no ability to run as a service |
| Milvus | 5 ms | 8,000 | Distributed and highly available; hybrid retrieval in version 2.5 is 30x faster than ES |
| Pinecone | 8 ms | 5,000 | Fully managed, zero operational burden |
| Weaviate | 10 ms | 4,000 | Built-in GraphQL and knowledge-graph capability |

How to choose:

- **Already on Postgres** → `pgvector` version 0.9 has caught up in performance, and you avoid the complexity of operating a second database
- **You need complex metadata filters** (by time / category / permission) → Qdrant (the ACORN algorithm solved the long-standing hard problem of filtered HNSW)
- **You want vectors plus a knowledge graph** → Weaviate
- **Enterprise-grade + huge volume + high availability** → Milvus
- **You want it embedded (no service required)** → LanceDB / Chroma
- **The simplest, fastest prototype** → Chroma
- **You want to use an existing Elasticsearch / OpenSearch stack** → they support vectors themselves

<a id="s2-4-5"></a>

#### 2.4.5 Retrieval: sparse + dense + late interaction

Vector retrieval (dense) is good at semantic similarity and poor at exact keyword matching (product IDs, proper nouns, tolerance for typos). **Production-grade RAG is almost always hybrid search:**

- **Sparse:** BM25 / SPLADE / BGE-sparse (keywords + term frequency)
- **Dense:** embedding-based vector search (semantics)
- **Late interaction:** ColBERT / ColBERT-v2. Each token has its own embedding, and at retrieval time the scores are aggregated by token-level max-similarity. Recall is higher than single-vector dense retrieval, but the index is 100x larger. Vespa and Qdrant already support it

The mixing strategy: merge the sparse and dense result lists with RRF (Reciprocal Rank Fusion). It is simple, and it needs no tuning.

Empirical result: **hybrid search + a reranker gains +9.3 percentage points of MRR over semantic-only**, and with Cohere Rerank accuracy can reach 90%+ [as measured in the LanceDB benchmark].

<a id="s2-4-6"></a>

#### 2.4.6 Reranking (Reranker)

After first-stage retrieval (embedding / BM25) has recalled the top 100, a cross-encoder that is more expensive and more accurate reranks the query-document pairs, and the top 5 go to the LLM. The ROI of this step is extremely high [a reranker survey]:

| Reranker | Type | ELO | Notes |
|---|---|---|---|
| **Zerank 2** | API | 1638 | Currently top of the board |
| **Cohere Rerank v4 Pro** | API | 1629 | The established one |
| **Cohere Rerank 3 Nimble** | API | — | The speed-optimized version |
| **BGE-reranker-v2-m3** | Self-hosted | — | Open source, quality close to Cohere |
| **FlashRank** | Self-hosted | — | Extremely lightweight, millisecond scale |
| **ColBERT v2** | Self-hosted | — | A late-interaction architecture |
| **RankLLM** (rerank directly with an LLM) | LLM | — | The most expensive, but the best explainability |

**The top rerankers raise precision by 15-40% over pure embeddings.** If you are allowed to add only one RAG optimization, adding a reranker is usually first in ROI.

<a id="s2-4-7"></a>

#### 2.4.7 Query rewriting (Query Rewriting / Expansion)

The user's "今天天气咋样" and the document's "current weather conditions" mean the same thing, but the vocabularies do not overlap. Query rewriting processes the query before retrieval:

- **HyDE (Hypothetical Document Embeddings):** first have the LLM imagine "what an answer document would write if one existed," and retrieve with that imagined text (not with the original query). The recall gain is clear in specialized domains
- **Multi-Query:** have the LLM rewrite the original query as 3-5 different phrasings, retrieve with each, and then merge
- **Step-Back Prompting:** first abstract the query into a broader question ("What is the specific number in company X's Q3 earnings report?" → "What is the overview of company X's Q3 earnings report?"), and use the retrieval results of the abstract question as the setup
- **Sub-Query Decomposition:** split a compound question into several sub-questions and retrieve each one ("compare A and B" → ["traits of A", "traits of B"])
- **The combination of HyDE + Multi-Query** is one of the strongest baselines today on retrieval benchmarks such as BEIR

<a id="s2-4-8"></a>

#### 2.4.8 GraphRAG

Microsoft GraphRAG (open-sourced in 2024) and community variants such as LightRAG / Nano-GraphRAG extend retrieval from "vector similarity" to "knowledge-graph traversal":

- **Build phase:** use an LLM to extract the entities and relations in the documents, and build a knowledge graph
- **Retrieval phase:** map the query onto entities in the graph, and do a local subgraph query plus a summary
- **Advantage:** it can answer "topic-level" questions ("what compliance risks are there across all of our supplier contracts"). Classic RAG fails on this kind of question because there is no locally similar chunk [43]
- **Cost:** building the graph is very expensive (an LLM extracts entities from every passage), so it suits a static, high-value knowledge base

In some settings GraphRAG pushes retrieval precision to 99% (relationship-dense domains such as financial compliance, law, and medicine) [43].

<a id="s2-4-9"></a>

#### 2.4.9 Self-RAG / CRAG / Agentic RAG

Self-RAG (Asai 2023): while generating, the model decides for itself **when** to retrieve, **evaluates** whether the retrieved results are relevant, and **critiques** whether its own output is supported.
CRAG (Corrective RAG): after retrieval, each chunk is labeled with a "high / medium / low confidence" tag, and low confidence triggers a web search as the fallback.
Agentic RAG: packs all of the abilities above into an agent loop, and the agent cycles through "think → retrieve → judge → rewrite the query → retrieve again → synthesize" until it is confident. LangGraph is the mainstream framework for implementing Agentic RAG [Vinod Rane 2026].

Empirical result: in production deployments, Self-RAG can reduce "unnecessary retrieval" by 25-40% [43], because the model has learned not to retrieve for things already in its own knowledge.

<a id="s2-4-10"></a>

#### 2.4.10 Multimodal RAG

Indexing schemes that mix images and text:

- **Late fusion:** embed text and images separately; at retrieval time, retrieve separately and then merge
- **Early fusion:** use a multimodal embedding such as CLIP / Gemini Embedding, and embed images and text into the same space
- **VLM-as-extractor:** use GPT-4o / Claude 3.5 Sonnet to turn images into detailed descriptions, then run text RAG (the highest cost, but the most stable recall)

<a id="s2-5"></a>

### 2.5 Memory systems

If RAG solves "how a knowledge base gets into the context," a memory system solves "how the agent's own experience gets into the context."

<a id="s2-5-1"></a>

#### 2.5.1 The three axes of memory

A classification borrowed from cognitive psychology:

- **Short-term memory (Working Memory):** what is inside the current conversation window (the system prompt + the most recent N turns)
- **Long-term memory**, further split into three kinds:
  - **Episodic memory:** "last Wednesday we discussed X"; "last time, the user said they were not familiar with React"
  - **Semantic memory:** "the user's native language is Chinese"; "the user prefers a dark theme"; "the project uses PostgreSQL"
  - **Procedural memory:** "this kind of task is usually done as the steps X→Y→Z"; "on a timeout, retry 3 times first"

<a id="s2-5-2"></a>

#### 2.5.2 Four memory systems compared (2026)

| System | Core architecture | Strength | Suited to |
|---|---|---|---|
| **Mem0** | Vector + optional graph; three layers of user / session / agent | The simplest to integrate, and the documentation is solid | Adding memory to a general chatbot |
| **Letta** (formerly MemGPT) | OS-inspired; the agent manages memory blocks itself through tool calls | Explicit and explainable; the agent has a persistent identity | Agents that run for a long time (days/weeks) |
| **Zep** | A temporal knowledge graph; tracks how facts change over time | Strong temporal reasoning, enterprise-grade | Settings where user state changes (CRM/customer support) |
| **Anthropic Memory Tool** | A hosted store + automatic injection + an API to revise/delete | Native integration with Claude | Inside the Anthropic ecosystem |

Observations across the architectures:

**Mem0** [the Mem0 paper, arXiv 2504.19413] is the easiest scheme to wire in. Three steps: extract a "memory" from the interaction → store it (vector by default) → on a later query, retrieve it and inject it into the prompt. The three-layer hierarchy of "user / session / agent" makes multi-tenant isolation happen automatically.

**Letta** has a distinctive design philosophy: it treats memory as the agent's "editable state," and the agent itself uses tool calls to read / write / search its own memory blocks. That is equivalent to giving the agent an OS-style memory manager. The advantage is that the agent can keep an "identity" across sessions — the next time it starts, the memory blocks are still there.

What sets **Zep** apart is that it is *temporal* — the same fact, "the user's job," may go from "at Google" → "at Anthropic" → "starting a company." Zep records the time dimension in a knowledge graph, so both "what is the user's job now" and "what was the user's job last year" come out correct.

**Anthropic Memory Tool** is the official memory interface launched in the second half of 2025 alongside Claude Sonnet 4.6: declare use of the memory tool on the API call, and the model decides on its own what memory to write and which memories to read. The developer only handles the business logic.

Practical selection [the n1n.ai 2026 comparison]:
- **A simple chatbot plus memory** → Mem0
- **An agent that has to run on its own across days** → Letta
- **Enterprise, and user state changes** → Zep
- **Already bound to Anthropic** → Memory Tool

<a id="s2-5-3"></a>

#### 2.5.3 Common traps in memory

- **Memory explosion:** remember everything → noise drowns the signal → retrieval quality falls. Active forgetting / summarization / priority eviction is required
- **Stale memory:** last week the user said "I like cold coffee," and today they tell the model "I only drink hot coffee now." A temporal system such as Zep is made for exactly this
- **Cross-user leakage:** a multi-tenant SaaS must isolate strictly by user_id

<a id="s2-6"></a>

### 2.6 Context-management strategies

Even with RAG and memory, "what to put in, how much to put in, and when to put it in" when filling the context window is still a hard problem. The Anthropic article [6] gives a few principles:

1. **Not all information is equally important** — rank it by priority, and put the most important material at the head and the tail (stay out of lost-in-the-middle)
2. **Dynamic trimming** — when the context nears the limit, actively summarize the early conversation and evict stale tool results
3. **A /compact mode** — products such as Claude Code support a `/compact` command, which compresses the current long conversation into a summary plus the key facts, keeping the meaning while reclaiming tokens
4. **Separate the scratchpad from working memory** — manage the agent's "scratch paper" (the reasoning process, intermediate tool results) separately from the "outward-facing context," so that it does not get contaminated

<a id="s2-7"></a>

### 2.7 The MCP / A2A protocol ecosystem

Wiring tools and data sources into an LLM was dirty work for a long time — every LLM's function-calling schema is different, and every tool has to have a hand-written wrapper. **MCP (Model Context Protocol)** [9] is an open standard released by Anthropic in November 2024. The goal is to be the "USB-C" of AI — one protocol that lets any LLM connect to any tool or data source.

The key design of MCP:
- **JSON-RPC 2.0** as the transport protocol
- **A server / client architecture:** an MCP server wraps an external system (PostgreSQL, GitHub, a file system, a custom API) and exposes three classes of primitive: tools (callable functions), resources (read-only data), and prompts (templates)
- **Decouple the tool from the LLM:** function calling hard-wires the tool definition to the LLM provider. MCP lets the tool and the LLM evolve independently

The timeline:
- 2024-11: Anthropic releases it
- The first half of 2025: OpenAI / Google / Microsoft / Amazon add support in succession
- 2025-12: Anthropic donates MCP to the Agentic AI Foundation (AAIF) under the Linux Foundation, jointly governed with Block and OpenAI
- 2026-02: monthly downloads of the Python + TypeScript SDKs break through 97 million [9]

**A2A (Agent-to-Agent)** is another class of protocol. It is concerned with how agents collaborate *with each other* (task delegation, shared state, workflow orchestration). MCP and A2A are complementary:

> "MCP gives an agent a pair of hands (connecting tools and data). A2A lets several agents work as a team. MCP is vertical integration. A2A is horizontal collaboration."

By the spring of 2026, MCP has become the de facto standard of the agent ecosystem. A2A is still in a phase of competition among several protocols (Google A2A / Microsoft's AutoGen protocol / Anthropic's Agent Skills, and others, are all trying).

---

## Part III: Harness Engineering

<a id="s3-1"></a>

### 3.1 Origin: "that layer of stuff," now named

"Harness Engineering," as a term, was formally named in February 2026 by Mitchell Hashimoto (founder of HashiCorp) [8]. The triggering event was an article by OpenAI engineer Ryan Lopopolo describing OpenAI's internal agent infrastructure. After reading it, Hashimoto said: "This is *harness engineering* — all the scaffolding, constraints, and feedback loops around an agent that let it actually work."

Karpathy laid the groundwork for this concept from another angle. Across 2025-2026 he described, more than once, that his own coding workflow had flipped from "mostly write the code by hand, and occasionally let an agent help" to "mostly let the agent write the code, and touch it up by hand myself." What makes this way of working possible is not that the model got stronger. It is the *environment* around the agent — the repo structure, the CI configuration, the formatter, the package manager, the framework constraints, the project instructions, the external tool integrations, the linter — and together these make up the agent's "harness."

In the April 2026 Medium article *Agent Harness Engineering: The Rise of the AI Control Plane* [32], Adnan Masood sums it up: "Harness Engineering is the building of that whole system which *governs* a deployed AI agent across many interactions — guides, sensors, data context pipelines, eval suites, constraint enforcement."

The core idea of harness engineering is stated again and again:

> **Every time an agent makes a mistake, do not merely hope it does better next time. Engineer the environment so that it cannot make the same mistake in the same way again.**

This unifies "AI safety" and "AI engineering" under one principle: **constrain the environment, rather than the model itself.** A model's "capability" is variable (a new generation every three months), but environmental constraints are stable, and they accumulate.

<a id="s3-2"></a>

### 3.2 The boundary with prompt engineering and context engineering

A clean division of the three layers of abstraction:

| Layer | Concern | Unit | Time scale |
|---|---|---|---|
| **Prompt Engineering** | How this one sentence is written | Characters / sentences | A single LLM call |
| **Context Engineering** | What the model sees at this moment of inference | A set of tokens | A single inference (including RAG, memory, tool schemas, and so on) |
| **Harness Engineering** | What environment the agent runs in, and how it stops making the same mistake | The system / the process | Across many interactions, evolving continuously |

An analogy:
- prompt = one specific instruction you give an employee
- context = all the material the employee can see on the desk (a notebook, email, documents, chat history)
- harness = the company's whole set of rules, tools, CI process, code review, and coaching system

<a id="s3-3"></a>

### 3.3 The seven components of the harness layer

What follows takes apart the seven core components of a production-grade agent harness in 2026.

<a id="s3-3-1"></a>

#### 3.3.1 The tool-integration layer (Tool Integration)

See §2.7 on MCP / A2A. Beyond that:

- **Function Calling:** native to the LLM provider (OpenAI, Anthropic, Google, and DeepSeek all support it). It fits simple tools and a single-LLM-provider setting
- **MCP:** cross-LLM, reusable, and the ecosystem is exploding (97M monthly downloads)
- **An OpenAPI / GraphQL adapter:** wrap any REST/GraphQL service as an MCP server
- **Agent Skills** (introduced by Anthropic in 2026): package "the agent knows how to do a certain class of task" as a reusable unit

In practice: MCP is already the de facto standard of 2026. New projects go straight to MCP, and existing projects stay compatible through an MCP adapter.

<a id="s3-3-2"></a>

#### 3.3.2 Sandbox and code execution (Sandbox)

When an agent writes code, runs commands, or handles files a user uploaded, it has to do so inside an isolated environment. Otherwise, the moment prompt injection arrives, the agent uploads `~/.ssh/` from your server to the public internet.

- **E2B:** available as both open source and SaaS. A secure sandbox runs arbitrary Python / Node / Bash, with a millisecond-scale cold start
- **Daytona:** an open-source dev environment, originally aimed at dev containers, with agent-sandbox support added from 2025
- **Modal:** serves as both a serverless GPU and a sandbox
- **OpenAI Code Interpreter** (Apps SDK): hosted, and simple
- **Anthropic's Claude Computer Use:** lets Claude operate desktop applications. In essence it is a more aggressive sandbox

The minimum bar: **never `eval()` agent output inside the main process of a production server.** Even something that looks harmless, such as `os.path.join`.

<a id="s3-3-3"></a>

#### 3.3.3 The LLM gateway (LLM Gateway)

Pull "calling an LLM" out of your business code into an independent gateway layer, and handle, in one place, routing, fallback, caching, rate limits, billing, observation, and key management. The mainstream options in 2026 [37][38]:

| Gateway | Mode | Headline capability | Suited to |
|---|---|---|---|
| **LiteLLM** | An open-source Python proxy | A unified OpenAI-compatible interface for 100+ LLMs, self-hosted | Small and mid-size teams; a preference for self-hosting |
| **Portkey** | Commercial + self-hosted | Semantic cache, guardrails, enterprise-grade observation | Enterprise production |
| **OpenRouter** | A SaaS marketplace | One API key for 200+ models, pay as you go | Startups / prototypes |
| **Helicone AI Gateway** | Open source + SaaS | High performance implemented in Rust, health-aware routing + circuit breaking, native observation | High-concurrency production |
| **Cloudflare AI Gateway** | SaaS (at the edge) | A global edge cache, zero operations | Already on the Cloudflare stack |

The key capabilities:

- **Multi-provider routing:** pick a provider dynamically by cost / latency / model capability. If GPT-4 fails, fall back to Claude, then fall back to Llama.
- **Semantic cache:** the signature advantage of Portkey and Cloudflare — not a literal match, but "a semantically similar query reuses the cached response." Customer-support and knowledge-base settings can reach a cache hit rate of 60%+.
- **Circuit breaking:** Helicone's "health-aware routing + circuit breaking" automatically isolates the failing endpoint when a provider fails.
- **Unified billing + quotas:** quotas by user_id / org_id, so that a malicious user cannot burn through the allowance.
- **Key management:** business code never touches the API key. It all stays in the gateway.

Note [37]: in 2026 the LLM-proxy ecosystem had two major events — Helicone was acquired, and LiteLLM was found to have a supply-chain problem in its dependencies. "Running a third-party proxy in a production environment" is itself a security decision. The enterprise-grade recommendation is to self-host and pin the version.

<a id="s3-3-4"></a>

#### 3.3.4 The inference server (Inference Server)

If you self-host a model (an open-source 7B-70B, or one you trained yourself), the inference server decides throughput and cost. The state of 2026 [39][40]:

- **TGI (Text Generation Inference, HuggingFace):** from December 2025 it entered *maintenance mode*. HF officially recommends vLLM or SGLang for new deployments.
- **vLLM** (UC Berkeley Sky Lab): the core innovation is **PagedAttention** [10]. It cuts the KV cache into pages of 16 tokens and manages them in a way similar to operating-system virtual memory, with VRAM fragmentation <4%. This raises, by several times, the concurrency a single card can run. The widest ecosystem (the default choice for HuggingFace, the cloud vendors, and enterprise self-hosting). It supports prefix caching (automatically detect a shared prefix and reuse the KV) and speculative decoding (a small model guesses first and the large model verifies; a 1.3-2x speedup when the acceptance rate is ≥0.7).
- **SGLang** (LMSYS): the core innovation is **RadixAttention** [11] — a radix-tree data structure automatically discovers and reuses the KV cache, with no need to configure prefix caching by hand. On multi-turn conversation and agent workloads (dynamic context shared across requests) it is 10-20% faster than vLLM; on a prefix-heavy workload it is up to 6.4x faster than the baseline. On small models (7B-13B), SGLang's throughput is ~29% higher than vLLM's; at the 70B scale the gap narrows to 3-5%.
- **NVIDIA Triton Inference Server:** the traditional choice. People still use it for mixed multimodal / multi-model workloads, but for pure LLM inference it no longer has much of an advantage.
- **LMCache** (proposed in 2025): as an independent KV-cache layer, it can share cache across different inference-server instances. In an enterprise multi-replica deployment it can save a further 30-50% of VRAM.

Selection experience: **pure-text LLM inference → SGLang (multi-turn / agent workloads) or vLLM (general / simple single-turn); mixed multimodal → vLLM or Triton.**

<a id="s3-3-5"></a>

#### 3.3.5 Observability (Observability)

A trace in an LLM application is far more complicated than in a traditional service — a single request may involve several LLM calls, several tool calls, several retrievals, several reranker passes, and several sub-agents. The mainstream in 2026 [33][34]:

| Platform | Type | Strength | License |
|---|---|---|---|
| **Langfuse** | Open source + cloud | Multi-tenant, high throughput on ClickHouse, 6M+ monthly SDK installs | MIT |
| **LangSmith** | Commercial (LangChain) | Native LangChain integration, graph visualization, CI eval | Commercial |
| **Helicone** | Open source + cloud | A proxy architecture (attach by changing one URL), cost and latency analysis | Open source + cloud |
| **Arize Phoenix** | Open source | Strong on the RAG pipeline (vector-space visualization, retrieval drift) | ELv2 |
| **Comet Opik** | Commercial | Session replay, prompt management | Commercial |
| **Braintrust** | Commercial | The full lifecycle (dataset → score → CI gating) | Commercial |

In practice [33][34]:
- Want a complete feature set + self-hosting → **Langfuse**
- Already using LangChain → **LangSmith** (automatic integration)
- Do not want to change the SDK, and only want to attach a proxy → **Helicone**
- The focus is RAG → **Arize Phoenix** (retrieval-drift visualization is its signature strength)

Many teams end up with "a combination of two tools": an OSS platform (Langfuse / Phoenix) for development and the dashboard, and a proxy gateway (Helicone) for observing production traffic.

<a id="s3-3-6"></a>

#### 3.3.6 Evaluation (Evaluation)

An LLM application does not have the hard constraint of a traditional unit test, where input X must produce output Y. The core question of evaluation is "is the model's answer a good answer," and that splits into two kinds [35]:

- **Offline evaluation (Offline Eval):** run a fixed dataset, and score with a metric / LLM-as-judge / a human
- **Online evaluation (Online Eval):** sample and score inside production traffic, and monitor quality drift

The mainstream frameworks in 2026 [35]:

| Framework | Traits | Suited to |
|---|---|---|
| **DeepEval** | 50+ metrics, pytest integration, agent evaluation | An engineering team's CI |
| **RAGAS** | Specialized in RAG: faithfulness, context precision, recall, answer relevancy | RAG systems |
| **Promptfoo** | Red teaming + CI, zero cloud dependency, prompt comparison | Prompt iteration + security testing |
| **Braintrust** | A full-lifecycle platform, a CI release gate | Enterprise |
| **OpenAI Evals** | The official framework | The OpenAI ecosystem |

**LLM-as-Judge** has been the core paradigm since 2024: a stronger LLM (usually GPT-4 / Claude Opus) scores the agent's output. The points that matter: the judge model must be stronger than the model under test; the prompt has to state the rubric clearly; the judge has bias of its own (a bias toward models from its own family, a bias toward long answers).

In practice [35]: "You almost certainly need two tools — a lightweight framework (DeepEval / RAGAS / Promptfoo) for CI/CD gating, and a platform (Braintrust / LangSmith / Arize) for human labeling, regression tracking, and the stakeholder dashboard."

<a id="s3-3-7"></a>

#### 3.3.7 Guardrails (Guardrails)

An LLM is an untrusted component. It will hallucinate, it will be injected, it will leak secrets, and it will generate harmful content. Guardrails are a "filter layer" placed on both the input side and the output side of the LLM. The mainstream in 2026:

- **NVIDIA NeMo Guardrails** [36]: open source. It describes dialogue flow and constraints in the Colang DSL. It supports 5 kinds of rails:
  - Input rails (the user-input side)
  - Dialog rails (constraints on the dialogue flow)
  - Retrieval rails (filtering of RAG chunks)
  - Execution rails (wrapping of custom actions)
  - Output rails (the LLM-output side)
  Integrated with the NVIDIA Nemotron Safety model (content safety, PII, and jailbreak detection).
- **Guardrails AI** (open source): Pydantic-style "validators," checking that the LLM output conforms to a schema, contains no PII, and contains no jailbreak trigger words
- **AWS Bedrock Guardrails:** hosted guardrails, with zero integration work in the AWS ecosystem
- **Azure Content Safety:** the same, for the Azure ecosystem
- **Lakera Guard:** commercial, specialized in prompt-injection detection
- **Protect AI Layer:** a commercial full-stack security platform

Compliance is the driver: **the high-risk obligations of the EU AI Act take effect on 2 August 2026** [36], and they place mandatory guardrail requirements on AI systems in settings such as medicine, education, hiring, finance, and critical infrastructure. The OWASP Top 10 for LLM Applications has become the industry-standard risk classification.

A practical configuration [Atlan 2026]:
- **Input side:** prompt-injection detection + PII filtering + a topic allowlist
- **Retrieval side:** grade retrieval results by sensitivity, and do not let sensitive content enter the LLM
- **Output side:** detection of PII / sensitive terms, factuality verification, JSON schema validation, length / rate limits
- **Audit side:** record every request "blocked by the guardrails," and review them on a schedule

<a id="s3-4"></a>

### 3.4 Agent orchestration frameworks

Assembling the components above into "one agent" takes an orchestration framework. The mainstream in 2026:

- **LangGraph** (LangChain): a state machine based on a graph (DAG), with explicit nodes and edges, a checkpointer built in, and human-in-the-loop as a first-class citizen. The widest ecosystem, and it is the de facto standard
- **AutoGen** (Microsoft): a multi-agent dialogue framework. Agents negotiate with each other in natural language
- **CrewAI:** a "role + task" abstraction. Team agents do the collaboration
- **LlamaIndex Agent:** an agent framework that is RAG-first
- **Semantic Kernel** (Microsoft): leans toward the .NET ecosystem
- **OpenAI Swarm / Agents SDK:** official from OpenAI. Simple, but limited in features
- **Smolagents** (HuggingFace): minimal, with only a few hundred lines of core code
- **Pydantic AI:** strongly typed agents based on Pydantic
- **Mastra** (native to TypeScript): the preferred pick in the JS / TS ecosystem

General advice for a new project [Vinod Rane 2026]:
- **Python + a complex workflow** → LangGraph
- **TypeScript / Next.js** → Mastra or the Vercel AI SDK
- **Multi-agent collaboration / a brainstorming mode** → AutoGen or CrewAI
- **Minimal + teaching** → Smolagents

<a id="s3-5"></a>

### 3.5 A reference architecture for a complete production stack

String every component together, and a "textbook" production agent system in 2026 looks roughly like this:

```
                ┌──────────────────────────┐
                │   用户 / 前端 / API Client │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   API Gateway / Auth     │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   Guardrail（输入侧）    │  ← prompt injection / PII
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   Agent Orchestrator     │  ← LangGraph / AutoGen
                │   （ReAct / Plan-Execute）│
                └─┬────┬────┬─────────────┘
                  │    │    │
       ┌──────────▼┐ ┌─▼──┐ ┌▼──────────────┐
       │  Memory   │ │RAG │ │  Tools (MCP)  │
       │ (Letta /  │ │    │ │  Sandbox(E2B) │
       │  mem0)    │ │    │ │               │
       └─────┬─────┘ └─┬──┘ └─┬─────────────┘
             │         │      │
             └─────────┼──────┘
                       │
                ┌──────▼─────────┐
                │  LLM Gateway   │  ← LiteLLM / Portkey
                │  路由 / 缓存   │     语义缓存 / fallback
                └──────┬─────────┘
                       │
              ┌────────┼─────────┬──────────┐
              ▼        ▼         ▼          ▼
        ┌────────┐┌────────┐┌────────┐┌────────────┐
        │ OpenAI ││Anthropic││ Google ││ vLLM/SGLang│
        │  API   ││  API    ││  API   ││ 自部署 OSS │
        └────────┘└────────┘└────────┘└────────────┘

          全程旁路：Observability（Langfuse / Helicone）
                     Evaluation（DeepEval / Braintrust）
                     Guardrail（输出侧 NeMo / Guardrails AI）
```

Every arrow is a "harnessed" interface. Each time it errs, add one rule, one eval case, and one guardrail rule — that is the daily work of harness engineering.

---

## Synthesis & Insights: the full picture of the three-layer evolution

### The timeline narrative

Lay the industry's focus over the past four years out as a timeline:

- **2022-2023: the era of prompt engineering**
  - ChatGPT set off the boom, and everyone was writing prompts
  - Main techniques: CoT, few-shot, ReAct
  - Engineering artifacts: prompt templates, prompt libraries, prompt training courses
  - Metaphor: "the person who can talk to the model"
- **2024: context engineering rises**
  - Named by Karpathy, and pushed forward by Anthropic
  - Long context (100K → 1M) + RAG maturing + memory systems emerging
  - Main techniques: RAG, vector stores, rerankers, MCP (November)
  - Engineering artifacts: vector-DB deployments, RAG pipelines, the MCP server ecosystem
  - Metaphor: "the person who can build a stage for the model"
- **2025-2026: harness engineering becomes the key to production**
  - Named by Hashimoto, OpenAI's internal architecture made public, Karpathy's workflow flipped
  - Main techniques: MCP (97M monthly downloads), inference servers (vLLM/SGLang), agent orchestration (LangGraph), observability, evaluation, guardrails
  - Engineering artifacts: a complete agent stack, compliance frameworks, an enterprise AI control plane
  - Metaphor: "the engineering manager who can tame a herd of agents"

### How the three layers relate

They stack, rather than replace one another. A production-grade agent needs all three layers:

- Fail to write the prompt well (layer 1) → the model does not understand the task
- Fail to manage the context well (layer 2) → the model cannot see the information the task requires
- Fail to build the harness well (layer 3) → in production the system is uncontrollable, unreliable, and unable to evolve

An analogy with software engineering:
- prompt = the specific function you write
- context = the data and the dependencies this function can reach while it runs
- harness = the whole project's CI / CD / monitoring / testing / release process

### The migration of the engineer's skill stack

In 2022, the most valuable skill of an LLM engineer was "the prompt intuition that can elicit the model's best output."
In 2024, the most valuable skill was "standing up a RAG system so that every document in the company can be queried."
In 2026, the most valuable skill is "standing up an agent harness so that several agents run 7×24 in production without a major accident."

Note that this line of migration is not the end. The next stop is very likely "the protocol engineer who can federate several agent harnesses" — the era when the A2A protocol ecosystem matures and agents from many companies interoperate.

### Seven principles shared across the three layers

Whether it is the prompt, the context, or the harness, engineering practice follows a few shared principles:

1. **Measure before you optimize:** with no eval data, no dashboard, and no baseline, every "improvement" is a hallucination
2. **Simplicity first:** a baseline is enough for 80% of tasks; add complexity only where the baseline fails
3. **Observability above all:** the trio of trace, log, and metric is the oxygen of a production agent
4. **Defensive programming:** assume the model will hallucinate, will be injected, will time out, and will lose the network
5. **Reversible design:** every step can be rolled back, and every component can be swapped out
6. **Cost awareness:** every decision asks "how many tokens / how many USD / how many milliseconds does this step cost"
7. **Evolutionary iteration:** start with the prompt → if that is not enough, add context → if that is not enough, add a harness. Do not build a complex stack from day 1

---

## Recommendations: selection lists for three typical scenarios

### Scenario 1: personal learning / a weekend project

**Goal:** understand the principles + get an end-to-end path running
**Budget:** a few dollars of API spend, and use the hardware you already have

| Component | Recommendation |
|---|---|
| LLM | OpenAI GPT-4o-mini or Claude Haiku (cheap + good enough) |
| Prompt framework | Use the Anthropic / OpenAI SDK directly; do not use LangChain |
| RAG | LlamaIndex (friendly to an API) or the LangChain RAG quickstart |
| Embedding | OpenAI text-embedding-3-small (the cheapest) |
| Vector DB | Chroma (embedded, zero operations) |
| Agent | LangGraph (general-purpose once you have learned it) |
| Observation | The Langfuse Cloud free tier |

Copy an official LangChain / LlamaIndex cookbook, change it a bit, and it runs.

### Scenario 2: a startup MVP / a small-team product

**Goal:** get it online as soon as possible + stay able to evolve
**Budget:** a few hundred USD a month

| Component | Recommendation |
|---|---|
| LLM | OpenRouter (one key reaches every model, switched as needed) |
| Prompt management | Promptfoo (a CI gate) + LangSmith (observation) |
| RAG | LlamaIndex / LangGraph |
| Embedding | Voyage 3 large (for the Chinese setting) or OpenAI text-3-large |
| Vector DB | Qdrant Cloud (a good cost-to-quality ratio + enterprise-grade) |
| Memory | Mem0 (the fastest to integrate) |
| Agent | LangGraph |
| LLM gateway | LiteLLM, self-hosted (starts with one Docker line) |
| Observation | Langfuse, self-hosted |
| Evaluation | Promptfoo + RAGAS |
| Guardrails | Guardrails AI (lightweight) |

Total cost: servers at ~$100/month + LLM usage metered.

### Scenario 3: enterprise production / compliance-sensitive

**Goal:** stable, observable, compliant, and auditable
**Budget:** six figures a month

| Component | Recommendation |
|---|---|
| LLM | Multi-provider redundancy + self-hosted OSS (for sensitive data) |
| Self-hosted inference | SGLang (multi-turn / agent) + vLLM (single-turn / high throughput) |
| Prompt management | LangSmith / Braintrust |
| RAG | An in-house modular pipeline + Cohere Rerank |
| Embedding | Cohere embed-v4 + BGE-M3, deployed as a pair |
| Vector DB | A Milvus cluster / pgvector (if Postgres is already there) |
| Memory | Zep (user state) + Letta (a long-running agent) |
| Agent | LangGraph as the primary, AutoGen for multi-agent collaboration |
| LLM gateway | Portkey, self-hosted (semantic cache + enterprise-grade control) |
| Observation | Langfuse, self-hosted + Arize Phoenix (RAG) |
| Evaluation | Braintrust (lifecycle) + DeepEval (CI) |
| Guardrails | NeMo Guardrails + Lakera Guard (injection detection) |
| Compliance | The EU AI Act's high-risk obligations of 2026-08, SOC2, HIPAA (depending on the industry) |

General advice: **whatever the scenario, attach observability from day 1.** Attaching Langfuse costs nothing. The price of not attaching it is that, six months later, you cannot produce any root-cause analysis of "why this query failed."

---

## Limitations & Caveats

1. **The industry is still evolving quickly.** This essay is current through 2026-05-01. MCP, Agent Skills, the A2A protocol, and inference servers all iterate on a scale of months, and the details may be out of date after 6 months
2. **Benchmark numbers leave room for cheating.** MTEB, RULER, and reranker ELO all carry a cherry-pick risk. Before production, you must re-measure on your own business data
3. **"Harness engineering" is still young as a term.** It was formally named only in 2026-02, and community consensus is still forming. This essay adopts descriptions that are cited by ≥4 independent sources, including Hashimoto / Augment Code / Atlan / Adnan Masood, but the boundary of the term may still drift
4. **Not covered:** the mathematics of model inference (the attention derivation, loss functions), the ROI of specific business cases, a detailed technical catalog of prompt attacks / jailbreaks, and regulatory detail (the full text of the EU AI Act)
5. **The scores have a subjective component.** The "recommendations" in the selection tables are based on public evaluations + community consensus + the author's practical preferences, and they do not constitute the optimum for every scenario
6. **Paid vs. open source.** This essay leans toward introducing open-source options. The value of "zero operations" that commercial SaaS has in many scenarios is not fully reflected
7. **What is particular to the Chinese setting.** Chinese embedding / RAG / model selection differs from the English setting. The recommendations given here have already tried to take the Chinese setting into account, but Chinese NLP for a specific business (finance / law / medicine) still needs domain fine-tuning

---

<a id="bibliography"></a>

## Bibliography

### Papers / academic sources

[1] Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762>
[2] Liu, N. F. et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL.
[3] Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS.
[4] Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
[12] Hsieh, C. et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* <https://arxiv.org/abs/2404.06654>
[16] Asai, A. et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*.
[18] Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*.
[19] Mem0 Team (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. <https://arxiv.org/abs/2504.19413>
[26] Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
[27] Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.

### Industry articles / official blogs

[5] Karpathy, A. *Software Is Changing (Again)*. YC AI School talk.
[6] Anthropic. *Effective Context Engineering for AI Agents*. <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
[7] LangChain. *Context Engineering for Agents*. <https://www.langchain.com/blog/context-engineering-for-agents>
[8] Hashimoto, M. / Augment Code (2026). *Harness Engineering for AI Coding Agents*. <https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents>
[9] Wikipedia. *Model Context Protocol*. <https://en.wikipedia.org/wiki/Model_Context_Protocol>
[10] vLLM Blog (2025). *Inside vLLM: Anatomy of a High-Throughput LLM Inference System*. <https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html>
[11] Runpod Blog. *When to Choose SGLang Over vLLM*. <https://www.runpod.io/blog/sglang-vs-vllm-kv-cache>
[21] Anthropic. *Memory tool docs*. <https://docs.anthropic.com>
[22] Raschka, S. *State of LLMs 2025*. <https://magazine.sebastianraschka.com/p/state-of-llms-2025>
[23] llm-stats. *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond*. <https://llm-stats.com/blog/research/post-training-techniques-2026>
[24] HuggingFace Blog. *Guide to RL Post-Training: PPO, DPO, GRPO*. <https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms>
[25] Introl. *Fine-Tuning Infrastructure: LoRA, QLoRA, PEFT at Scale*.
[28] PromptingGuide.ai. CoT / ReAct sections. <https://www.promptingguide.ai>
[29] Pillitteri, P. *Prompt Engineering 2026: The Frameworks That Actually Work*. <https://pasqualepillitteri.it/en/news/1090/prompt-engineering-2026-frameworks-complete-guide>
[30] BrightCoding. *Prompt Engineering Guide*. <https://www.blog.brightcoding.dev>
[31] Atlan. *What Is Harness Engineering AI? The Definitive 2026 Guide*. <https://atlan.com/know/what-is-harness-engineering/>
[32] Masood, A. (2026). *Agent Harness Engineering — The Rise of the AI Control Plane*. Medium. <https://medium.com/@adnanmasood/agent-harness-engineering-the-rise-of-the-ai-control-plane-938ead884b1d>
[33] Spheron Blog. *LLM Observability on GPU Cloud (2026)*.
[34] Confident AI. *Top 10 LLM Observability Tools to Evaluate & Monitor AI in 2026*.
[35] Confident AI. *Top 7 LLM Evaluation Tools in 2026*.
[36] NVIDIA. *NeMo Guardrails Documentation*. <https://docs.nvidia.com/nemo/guardrails/latest/index.html>
[37] Helicone Blog. *Top 5 LLM Gateways*. <https://www.helicone.ai/blog/top-llm-gateways-comparison-2025>
[38] PkgPulse. *Portkey vs LiteLLM vs OpenRouter: LLM Gateway 2026*.
[39] PreMAI Blog. *LLM Inference Servers Compared: vLLM vs TGI vs SGLang vs Triton (2026)*.
[40] Runpod Blog. *SGLang vs vLLM*.
[41] Tianpan. *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool*.
[42] BAAI. *BGE-M3 model card*. <https://huggingface.co/BAAI/bge-m3>
[43] Squirro / multiple sources. *RAG in 2026: Bridging Knowledge and Generative AI*.

### Official resources for frameworks / protocols

- LangGraph: <https://langchain-ai.github.io/langgraph/>
- LangChain MCP Adapters: <https://github.com/langchain-ai/langchain-mcp-adapters>
- Anthropic Effective Prompts: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview>
- vLLM Prefix Caching: <https://docs.vllm.ai/en/stable/design/prefix_caching/>
- vLLM Speculative Decoding: <https://docs.vllm.ai/en/latest/features/spec_decode/>
- Langfuse: <https://langfuse.com>
- Promptfoo: <https://www.promptfoo.dev>
- DeepEval: <https://deepeval.com>
- DSPy: <https://github.com/stanfordnlp/dspy>

---

<a id="methodology"></a>

## Methodology Appendix

### Research design
- **Skill:** `deep-research` v2.3.1 (the 199-bio edition)
- **Mode:** ultradeep (the user was explicit that it should be "as complete as possible," which triggered the deepest tier, of 8+ phases)
- **Search provider:** the built-in WebSearch (the multi-vendor search-cli had no key configured, so it fell back)
- **Number of search paths:** 18 (in two parallel batches: the first batch of 8 covered the trunks of the three themes, the second batch of 8 covered the runtime stack and the long-tail choices, and a third batch of 2 filled in the training methods)

### The list of 18 search paths
1. prompt engineering 2026 best practices Anthropic OpenAI techniques chain of thought ReAct
2. context engineering definition Karpathy Anthropic 2025 2026 LLM
3. harness engineering LLM agent infrastructure Karpathy 2026
4. transformer architecture attention mechanism scaling laws 2017 to 2026 evolution
5. RAG state of the art 2026 GraphRAG Agentic RAG Self-RAG CRAG
6. vector database comparison 2026 Pinecone Weaviate Qdrant Milvus pgvector benchmark
7. embedding models 2026 BGE Cohere Voyage Jina OpenAI text-embedding benchmark MTEB
8. LLM memory systems MemGPT Letta mem0 Zep Anthropic memory tool 2026
9. RAG reranker 2026 Cohere rerank BGE cross-encoder ColBERT hybrid search benchmark
10. long context vs RAG 2026 1M tokens lost in the middle needle haystack RULER
11. MCP Model Context Protocol A2A agent to agent function calling 2026
12. LLM observability LangSmith Langfuse Helicone Arize Phoenix 2026 comparison
13. LLM evaluation 2026 LLM-as-judge RAGAS DeepEval Promptfoo Braintrust framework
14. LLM guardrails NeMo Guardrails AI 2026 jailbreak prevention prompt injection
15. LLM gateway LiteLLM Portkey OpenRouter Helicone 2026 routing fallback caching
16. vLLM SGLang TGI inference server 2026 prompt caching speculative decoding KV cache
17. LLM training pretraining post-training SFT DPO GRPO RLHF RLAIF 2026 methods
18. LoRA QLoRA PEFT fine-tuning 2026 vs full fine-tune distillation

### Phase 4 TRIANGULATE — key claims verified across multiple sources
Every key fact that entered the body is supported by at least 2 independent sources:

| Claim | Supported by at least 2 sources |
|---|---|
| Karpathy named "context engineering," in the YC AI School talk | Anthropic official + IntuitionLabs + FlowHunt + LangChain |
| Mitchell Hashimoto named "harness engineering" in 2026-02 | Augment Code + MadPlay + Atlan + Adnan Masood |
| MCP was released by Anthropic in 2024-11, transferred to the Linux Foundation in 2025-12, and SDK monthly downloads were 97 million in 2026-02 | Wikipedia + a2a-mcp.org + Pockit + EssaMamdani |
| The Lost-in-the-Middle U-shaped recall curve | TACL 2024 + several surveys from 2026 |
| Gemini 1.5 Pro is 99.7% on the needle, but ~60% on multi-fact recall | Tianpan + LongContext + RAG papers |
| Long context is 30-60× slower than RAG, and 1250× more expensive | Tianpan + posts with similar production data |
| TGI entered maintenance mode in 2025-12 | PreMAI + SitePoint |
| vLLM PagedAttention has <4% VRAM fragmentation | The official vLLM blog + several surveys |
| On conversational workloads, SGLang RadixAttention is +10-20% versus vLLM | Runpod + SGLang official |
| Speculative decoding is 1.3-2×, with acceptance≥0.7 | vLLM documentation + several evaluations |
| Cohere embed-v4 is 65.2 on MTEB (the current leader of the board) | pecollective + reintech + Mixpeek |
| Voyage 3 large is ~10% above OpenAI text-3-large | TokenMix + Cheney Zhang |
| Hybrid + reranker is +9.3pp MRR over semantic-only | The LanceDB benchmark + dev.to |
| Self-RAG reduces unnecessary retrieval by 25-40% | Squirro + surveys |
| Langfuse has 6M+ monthly SDK installs | Langfuse official + Spheron |
| Full FT of a 7B = 100-120GB of VRAM; QLoRA of a 7B = ~6GB | Introl + Mercity + RedHat |
| QLoRA keeps 80-90% of the quality of a full FT | Introl + RedHat |
| GRPO is the core algorithm of DeepSeek-R1 | llm-stats 2026 + the Sebastian Raschka survey |
| The high-risk obligations of the EU AI Act take effect on 2026-08-02 | NVIDIA NeMo docs + several compliance articles |

### Phase 6 CRITIQUE — red-team review
- **The skeptic:** were important topics missed? After a follow-up search, 2 further paths were added (training methods). What may still be missing: multimodal model architectures (the CLIP family, the Vision Transformer family). This essay focuses on LLM application engineering rather than the evolution of model architecture, which is acceptable
- **The adversarial review:** is it oversimplified? Each of the 7 major components in the harness-engineering chapter is given only ~500 characters, so there is real simplification. The goal of this essay is a guide map rather than a deep dig into every component, which is acceptable. Each component is given 2-3 public sources so the reader can dig further
- **The implementing engineer:** can the selection lists be put into practice immediately? All three scenarios give specific product names plus a deployment style, and they make first principles explicit, such as "observability must be attached on day 1"

### Phase 8 PACKAGE
- The main report: `article.md`
- Metadata: `run_manifest.json`, `sources.jsonl`
- HTML / PDF: not generated (the user did not ask for them, and WeasyPrint was not installed)

### Reproduction commands
```bash
ls -la ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/
wc -m ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/article.md
```
