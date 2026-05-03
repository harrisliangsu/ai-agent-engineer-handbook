# AI Agent / LLM 应用工程师面试题与答案研究报告

**调研日期**：2026-05-02
**目标读者**：求职者（应届到 staff 级）+ 招聘方
**覆盖**：10 大分类 / 90+ 道题 / 难度分级 / 含一手论文 + Anthropic / OpenAI / DeepSeek 官方资料

---

## Executive Summary

本报告按 10 大分类整理 AI Agent / LLM 应用工程师面试题，每题给：

- **难度分级**：⭐ 基础 / ⭐⭐ 中等 / ⭐⭐⭐ 困难
- **200-500 字技术答案**：含数字 / 论文出处 / 工程取舍
- **来源**：每题至少 1-2 个权威 URL，方便深挖

**10 大分类**：

| 分类 | 题数 | 难度分布 | 重点 |
|---|---|---|---|
| **A. LLM 基础** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Transformer / RoPE / KV cache / 长上下文 / 推理优化 |
| **B. 训练 / 微调** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | SFT / DPO / GRPO / LoRA / R1 涌现 |
| **C. 提示词工程** | 9 | 1⭐ 7⭐⭐ 1⭐⭐⭐ | CoT / Self-Consistency / ReAct / 自动 prompt 优化 |
| **D. RAG** | 10 | 1⭐ 7⭐⭐ 2⭐⭐⭐ | Hybrid / Reranker / GraphRAG / Self-RAG / RAGAS |
| **E. Agent** | 12 | 1⭐ 6⭐⭐ 5⭐⭐⭐ | **核心分类**：6 patterns / 多 agent / 失败模式 / harness |
| **F. Memory** | 8 | 1⭐ 5⭐⭐ 2⭐⭐⭐ | MemGPT / mem0 / progress 文件 / memory corruption |
| **G. 系统设计** | 8 | 0⭐ 1⭐⭐ 7⭐⭐⭐ | 客服 / coding / 搜索 / LLM 网关 / 高并发 / 成本 / 沙箱 / 闭环 |
| **H. 评估** | 6 | 0⭐ 2⭐⭐ 4⭐⭐⭐ | judge bias / pairwise / benchmark hack / 漂移 |
| **I. 安全 / 护栏** | 6 | 0⭐ 1⭐⭐ 5⭐⭐⭐ | OWASP / Prompt Injection / NeMo Guardrails / EU AI Act |
| **J. 行为题 / 实战** | 7 | 0⭐ 2⭐⭐ 5⭐⭐⭐ | Hashimoto harness 闭环 / 0-1 落地路线 |

**学习路径建议**：
- **应届 / 初级**：A + B + C + 部分 D，能讲清核心概念即可
- **3-5 年（中级）**：全部 A-F + 部分 G/H，要有项目案例
- **5-10 年（资深 / staff）**：全部 10 类，G/J 是主战场，要能从架构层讨论取舍
- **架构师 / staff+**：J 行为题决定差异，必须有 Hashimoto harness 思维 + 量化落地案例

---

## A. LLM 基础（10 题）

### A1. ⭐⭐ 为什么 Transformer 用 Layer Norm 而不是 Batch Norm？

**答案**：BatchNorm 沿 batch 维度对每个特征做归一化，依赖 batch 内统计量。NLP 场景下序列长度可变、padding 大量存在，且训练（batch=4M tokens）和推理（batch=1）的 batch 大小差异极大，导致 BN 的 running mean/var 严重偏移。LayerNorm 沿 feature 维度（hidden_dim）对单个样本归一化，与 batch 无关，天然支持变长序列与单条推理。后续 Llama/PaLM 改用 RMSNorm，去掉减均值和 bias，仅做 `x / sqrt(mean(x²)+ε) * g`，省 7-64% 归一化耗时且效果不降。Pre-LN（LN 在残差前）训练比 Post-LN 稳定，可去掉 warmup，但表达力略弱，GPT-2 起几乎统一用 Pre-LN。

**来源**：Vaswani et al. 2017《Attention Is All You Need》<https://arxiv.org/abs/1706.03762>；Zhang & Sennrich 2019《Root Mean Square Layer Normalization》

### A2. ⭐ 多头注意力的"多头"到底有什么用？单头加大维度行不行？

**答案**：单头注意力对每对 token 只产生一个 attention 分布，相当于一种"关系视角"。多头（h=8/32/64/128）把 d_model 拆成 h 份独立投影 Q/K/V，每个头学不同子空间的关系——句法头关注短距离依赖，语义头关注共指、远程实体。即使总参数和单头大维度相同，多头在表达能力上严格更强，因为是 h 个低秩注意力的混合而非单个高秩。GQA（Grouped Query Attention，Llama-2 70B / Llama-3 起）让多个 Q 头共享同一组 K/V 头，KV cache 减小到 1/8，吞吐显著提升，效果几乎无损。MQA 是极端情况（所有头共享一套 KV）。

**来源**：Vaswani 2017；Ainslie et al. 2023《GQA: Training Generalized Multi-Query Transformer》<https://arxiv.org/abs/2305.13245>

### A3. ⭐⭐ RoPE 相比绝对位置编码的优势？为什么外推还是会崩？

**答案**：RoPE（Su et al. 2021）把位置编码作用在 Q/K 上而非 embedding 上，将位置 m 表示为旋转矩阵 R_m，使得 `<R_m·q, R_n·k> = f(q, k, m-n)`，天然编码相对位置。优点：(1) 无需额外参数；(2) 注意力得分只依赖相对距离；(3) 可与线性注意力兼容。但 RoPE 高频分量在训练上下文外旋转角度落入未见区域，注意力分布塌缩，所以纯外推 2k→32k 会崩。NTK-aware scaling 对低频维度做插值、高频维度近乎不变（base 从 10000 改为更大值），保留高频精度；YaRN（Peng et al. 2023）进一步引入 attention temperature 和按维度分段缩放，是目前 Llama-3.1 128k、Qwen2.5 等的主流方案。

**来源**：Su et al. 2021《RoFormer: Enhanced Transformer with Rotary Position Embedding》<https://arxiv.org/abs/2104.09864>；Peng et al. 2023《YaRN》<https://arxiv.org/abs/2309.00071>

### A4. ⭐⭐⭐ vLLM 的 PagedAttention 解决了什么问题？为什么吞吐能涨 24 倍？

**答案**：传统推理为每个 request 预分配 max_seq_len 的 KV cache 连续显存，导致：(1) 内部碎片（请求实际生成 200 token 但占了 4096 槽）；(2) 外部碎片（不同请求间隙无法复用）；(3) 共享前缀无法去重。PagedAttention 借鉴 OS 虚拟内存：把 KV cache 切成固定大小 block（如 16 token），每个序列维护一个 block table（逻辑→物理映射）。带来三项收益：(1) 显存利用率从 ~20% 提到 >96%；(2) 同一 prompt 的多个采样（beam/n>1）共享前缀 block，写时复制；(3) 配合 continuous batching（请求一结束就插新请求，不等齐 batch）吞吐提升。原 paper 报告相对 HuggingFace TGI 吞吐 2-4×，相对 FasterTransformer 24×。SGLang 的 RadixAttention 进一步把所有历史前缀组织成 radix tree 自动缓存复用，对 agent 多轮场景尤其有效。

**来源**：Kwon et al. 2023《Efficient Memory Management for LLM Serving with PagedAttention》（SOSP'23）<https://arxiv.org/abs/2309.06180>

### A5. ⭐⭐ Lost in the Middle 是什么？工程上怎么缓解？

**答案**：Liu et al. 2023 实验发现：在长上下文 QA 中，把 gold passage 放在 prompt 开头或结尾，模型准确率高；放中间则显著下降，呈 U 形曲线，10+ documents 时中间位置准确率可能低于无上下文（closed-book）基线。原因推测：(1) 训练数据中关键信息多在首尾；(2) 注意力 sink 效应；(3) RoPE 中段位置编码训练不足。工程缓解：(1) 重排——把最相关 chunk 放最前/最后（reranker + 头尾穿插）；(2) 多轮检索 + 摘要而非一次塞满；(3) 用支持原生长上下文的模型（Claude、Gemini 1.5 Pro 在 RULER 基准上更稳）。RULER（Hsieh et al. 2024 NVIDIA）是更严苛的长上下文基准，包含 needle-in-haystack、变量追踪、多跳 QA 等 13 类任务，比纯 NIAH 更能反映真实能力。

**来源**：Liu et al. 2023《Lost in the Middle》<https://arxiv.org/abs/2307.03172>；Hsieh et al. 2024《RULER》<https://arxiv.org/abs/2404.06654>

### A6. ⭐⭐ Speculative Decoding 原理？什么场景收益最大？

**答案**：核心思路：用一个小的 draft model（参数 1-7B）一次性自回归生成 K 个 token（如 K=5），再用 target model（70B+）一次 forward 并行验证 K+1 个位置。若前 j 个被接受、第 j+1 个拒绝，则保留前 j 个并采样新 token，整体一次大模型 forward 至少推进 1 token，最多 K+1 token。数学上保证输出分布与原模型完全一致（rejection sampling）。收益条件：(1) draft 与 target 分布相似（接受率 >60%）；(2) 大模型 forward 受 memory bandwidth 限制而非 compute，多 token 并行验证几乎不增加延迟。EAGLE-2/3、Medusa（多个 LM head 并行预测）、Llama 3.1 self-speculative 是主流变体。代码生成、确定性强的任务接受率高（70-90%），开放生成接受率低（30-50%）。

**来源**：Leviathan et al. 2023《Fast Inference from Transformers via Speculative Decoding》<https://arxiv.org/abs/2211.17192>

### A7. ⭐ Prefix Caching 与 KV Cache 的区别？

**答案**：KV Cache 是单 request 内部的优化——decode 第 t 个 token 时复用前 t-1 个的 K/V，避免重新计算。Prefix Caching 是跨 request 的优化——多个 request 共享相同前缀（如 system prompt、few-shot 示例、agent 历史），首次计算后把这些 token 的 KV 缓存，后续 request 直接命中跳过 prefill。在 agent / RAG / 多轮对话场景，system prompt 通常 1-4k token、几乎所有请求共享，prefill 阶段 TTFT 可下降 10-100×。vLLM 的 `--enable-prefix-caching`、SGLang 的 RadixAttention、Anthropic 的 prompt caching API（5 分钟 TTL，cached 部分 0.1× 价格）都是同思路。工程取舍：缓存 block 占显存，命中率低时反而拖慢，需要监控 cache hit rate。

**来源**：vLLM Automatic Prefix Caching docs；Anthropic Prompt Caching docs <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>

### A8. ⭐⭐⭐ FlashAttention 1/2/3 各解决了什么？为什么 H100 上 FA3 能再快 1.5-2×？

**答案**：标准 attention 显存复杂度 O(N²)（要存 N×N 的 attention matrix），是长序列瓶颈。FA1（Dao 2022）通过 tiling + online softmax，把 Q/K/V 分块加载到 SRAM 计算，不实例化完整 attention matrix，显存降到 O(N)，速度也快——因为 attention 是 memory bound，少读 HBM 就快。FA2（Dao 2023）优化了并行划分（按 seq_len 而非 batch 并行）和 warp-level 调度，A100 上达 ~50-72% 峰值 FLOPs。FA3（Shah et al. 2024）针对 H100 三项 Hopper 特性：(1) 用 WGMMA 异步张量核心；(2) TMA 异步加载 + producer/consumer warp 重叠 GEMM 与 softmax；(3) 原生 FP8（throughput 翻倍但需精度补偿）。BF16 下 H100 740 TFLOPS（75% 峰值），比 FA2 快 1.5-2×；FP8 接近 1.2 PFLOPS。

**来源**：Dao et al. 2022《FlashAttention》<https://arxiv.org/abs/2205.14135>；Shah, Dao et al. 2024《FlashAttention-3》<https://arxiv.org/abs/2407.08608>

### A9. ⭐⭐ Continuous Batching vs Static Batching？

**答案**：Static batching：等齐 batch_size 个 request，一起 prefill+decode 直到全部结束。问题：长短请求混在一起时，短请求结束后该 slot 空闲到最长请求结束（"head-of-line blocking"），GPU 利用率经常 <30%。Continuous batching（也叫 in-flight batching，Yu et al. ORCA 2022 提出）按 token 级别调度——每个 decode step 后，已完成的 request 立刻退出，新到的 request 立刻插入空槽并先做 prefill 再加入 decode batch。配合 PagedAttention 后无显存碎片，吞吐相比 static 提升 5-23×（vLLM 数据）。工程取舍：prefill 与 decode 混在一起会产生延迟抖动（chunked prefill 把长 prefill 切成 token 块、与 decode 交错调度可缓解，是 vLLM v0.6+ 默认）。

**来源**：Yu et al. 2022《Orca: A Distributed Serving System for Transformer-Based Generative Models》（OSDI）

### A10. ⭐⭐ 上下文越长效果越好吗？工程上 128k 真能用吗？

**答案**：理论窗口与有效窗口差距很大。RULER 基准显示：宣称 128k/200k 的多数模型在 32k 之外性能急剧下降，仅 GPT-4、Claude 3.5、Gemini 1.5 Pro 在 64k+ 仍稳定。问题：(1) Lost in the Middle；(2) 注意力被无关 token 稀释（信噪比下降）；(3) 长上下文训练数据稀缺，多数靠 NTK/YaRN 拉伸；(4) 成本——KV cache 显存正比于 context_len，128k 单 request 可能占数十 GB。工程实践推荐：(1) 默认走 RAG，把"上下文应当塞入"的阈值定在 8-32k；(2) 长文档先做层级摘要再喂；(3) Agent 历史用滑窗 + memory 摘要而非无限累积；(4) 评估必须用任务级指标，不是 needle-in-haystack。Anthropic 建议：复杂多跳推理 32k 内最稳。

**来源**：Hsieh et al. 2024《RULER》；Liu et al. 2023《Lost in the Middle》

---

## B. 训练 / 微调（10 题）

### B1. ⭐ 预训练、后训练、任务微调三者怎么拆分？

**答案**：(1) **预训练**（pretraining）：在万亿级 token 通用语料（CommonCrawl、Code、Books）上做 next token prediction，目标是学语言/世界知识，成本占总训练 95%+，Llama-3.1 405B 用 15.6T token、3.8e25 FLOPs。(2) **后训练**（post-training）：在预训练 base 模型上做 SFT + 偏好优化（DPO/PPO/GRPO）+ safety，目标是把"能补全"转成"能听指令、对齐人类偏好"，数据量从百万到千万级，成本通常 <5% 预训练。(3) **任务微调**（task fine-tuning，下游用户做的）：在已对齐的 chat 模型上用领域数据（几千-几万样本）做 LoRA/全参 SFT，目标是适配特定任务（医疗问答、代码转换、客服）。工程经验：领域微调优先 LoRA + base 还是 chat 模型，要看是否需要保留指令遵循能力——保留就基于 chat、纯做风格/格式适配；做强领域知识注入则基于 base 重新做完整后训练栈。

**来源**：Llama 3 paper（Grattafiori et al. 2024）；OpenAI InstructGPT paper

### B2. ⭐⭐ DPO vs PPO 的核心区别？为什么近两年 DPO 更流行？

**答案**：PPO（RLHF 经典路径）：先训 reward model（RM），再用 RM 给 policy 打分，用 PPO 做 actor-critic 更新，需同时载入 actor、ref、critic、reward 四个模型，显存 ≥4× 单模型，超参敏感（KL 系数、clip ratio、advantage 归一化），训练不稳定。DPO（Rafailov et al. 2023）从数学上证明：在 Bradley-Terry 偏好模型 + KL 约束下，最优 policy 有闭式解，可直接用偏好对 (chosen, rejected) 做监督式损失：`-log σ(β(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))`。优势：(1) 无 RM、无在线采样、无 critic，仅 policy + ref 两个模型；(2) 用 SFT 同样的训练栈；(3) 超参稳定。劣势：(1) 严重依赖偏好数据质量，无法靠 RM 泛化；(2) 容易过拟合 chosen 的表面模式而非真正的偏好方向；(3) reward hacking 风险大。Llama-3、Qwen2 后训练主路径都是 DPO 或其变种（IPO、KTO、SimPO）。

**来源**：Rafailov et al. 2023《Direct Preference Optimization》<https://arxiv.org/abs/2305.18290>

### B3. ⭐⭐⭐ DeepSeek R1 的 GRPO 为什么有效？相比 PPO 省了什么？

**答案**：GRPO（Group Relative Policy Optimization，DeepSeek-Math 2024 提出，R1 验证）的关键创新：**去掉 critic（value 网络）**，对同一 prompt 采样 G 个 response（典型 G=64），用组内 reward 的均值/标准差做 advantage 归一化：`A_i = (r_i - mean(r)) / std(r)`，再做 PPO-like 的 clipped objective + KL 正则。三层效果：(1) 显存省一半（无 critic），70B 级 RL 在合理资源内可行；(2) 组内归一化天然提供 baseline，方差比 PPO 的 GAE 还低；(3) 配合 RLVR（RL with Verifiable Rewards）——数学题用最终答案对错、代码用单测通过率作 0/1 reward，无需 RM，彻底避免 reward hacking。R1-Zero 直接从 base 模型纯 RL（无 SFT）跑出 reasoning 行为：长 CoT、自我验证、aha moment，AIME 从 15.6%→71.0%。R1 正式版加了少量 cold-start SFT 让格式更稳定。

**来源**：Shao et al. 2024《DeepSeekMath》<https://arxiv.org/abs/2402.03300>；DeepSeek-AI 2025《DeepSeek-R1》<https://arxiv.org/abs/2501.12948>

### B4. ⭐⭐ LoRA 与 QLoRA 显存怎么算？为什么 QLoRA 能在单卡 24G 跑 65B？

**答案**：全参微调显存 ≈ 参数(2B for fp16) + 梯度(2B) + Adam state(8B m+v fp32) + activation ≈ **16-20× 参数量字节**，7B 模型需 >100GB。LoRA：冻结原权重，仅训练低秩 adapter `ΔW = BA`（r=8/16/64），可训参数减少 100-10000×，梯度+optimizer state 几乎为零，但**冻结权重仍要 fp16 加载**，7B 需 ~14GB + adapter。QLoRA（Dettmers et al. 2023）三项关键：(1) 4-bit NF4 量化冻结权重（信息论最优正态分布量化），7B 从 14GB 压到 ~4GB；(2) Double Quantization 把量化常数也量化，省 ~0.4 bit/param；(3) Paged Optimizer 用 NVIDIA unified memory，OOM 时溢出到 CPU。65B 模型 QLoRA 约 33GB，单 A100-48G 或双 RTX 3090 可跑，效果与 fp16 LoRA 几乎无差。工程取舍：QLoRA 训练慢 30-40%（反量化开销），推理需 merge 回原精度或用 GPTQ/AWQ 重新量化部署。

**来源**：Hu et al. 2021《LoRA》<https://arxiv.org/abs/2106.09685>；Dettmers et al. 2023《QLoRA》<https://arxiv.org/abs/2305.14314>

### B5. ⭐⭐ GPTQ / AWQ / FP8 对比？部署该选哪个？

**答案**：(1) **GPTQ**（Frantar 2022）：post-training 4-bit 权重量化，用 OBQ 启发的逐层最小化量化误差，需要少量校准集（128 样本）。优点早、生态全；缺点对激活分布敏感，重要通道可能被压坏。(2) **AWQ**（Lin et al. 2023）：观察到只需保护 ~1% 的 salient 权重通道（由激活幅值识别）即可保住精度，对这些通道按 per-channel scale 放大再量化。在 Llama-2 上比 GPTQ 困惑度更低、实现更简单，已成 vLLM/TensorRT-LLM 默认。(3) **FP8**（H100/H200 原生支持）：E4M3（前向）+ E5M2（反向），动态范围比 INT8 大 ~10⁴×，几乎无精度损失，且训练推理统一。NVIDIA Hopper 上 FP8 throughput 是 BF16 的 2×。**选型**：(a) 仅推理 + 老卡（A100/3090）→ AWQ INT4；(b) H100/H200 部署 → FP8（通过 TensorRT-LLM / vLLM FP8 KV cache）；(c) 极致显存（手机/边缘）→ INT4 + GGUF。

**来源**：Lin et al. 2023《AWQ》<https://arxiv.org/abs/2306.00978>

### B6. ⭐⭐ 蒸馏在 LLM 时代和过去 BERT 时代有什么不一样？

**答案**：BERT 时代（DistilBERT、TinyBERT）：用 student 模仿 teacher 的 logits（KL 散度）+ hidden state（MSE）+ attention map，需配对前向、对齐层数，工程重。LLM 时代主流是**数据蒸馏（黑盒蒸馏）**：用大模型生成大量高质量 (prompt, response) 数据，小模型直接 SFT。Alpaca（用 GPT-3.5 生成 52k 指令数据训 7B Llama）、Phi 系列（用 GPT-4 生成"教科书式"合成数据）、DeepSeek-R1-Distill（用 R1 生成 800k 推理轨迹蒸馏到 Qwen-1.5B/7B/14B/32B、Llama-8B/70B）都是这一路。优点：无需访问 teacher 权重、可跨架构、训练栈与 SFT 一致。缺点：丢失了 logit 中的不确定性信息，且受限于 teacher 的采样温度/多样性。混合做法（on-policy distillation）：student 自己采样、teacher 打分纠正，效果介于 SFT 和 RL 之间。R1-Distill-Qwen-32B 在 AIME 上 72.6%，超过 o1-mini，证明蒸馏可以传递 reasoning 能力。

**来源**：DeepSeek-R1 2025 paper（distillation 章节）

### B7. ⭐ SFT 数据量到底要多少？数据质量如何评估？

**答案**：经验数字：(1) 基于 chat 模型做格式/风格适配：500-5000 高质量样本足够，多了反而过拟合损失泛化（catastrophic forgetting）。(2) 注入新领域知识：1-10 万样本起步，且需混入 5-20% 通用数据保留通用能力。(3) 从 base 做完整 instruct tuning：100 万级（Llama 3 SFT 用 ~1000 万样本）。**LIMA 论文**（Zhou et al. 2023）证明 1000 条精选样本即可让 65B base 达到接近 GPT-4 的对话质量——"Less Is More for Alignment"。质量评估：(a) 多样性（覆盖任务类型、领域、难度）；(b) 答案质量（人工/GPT-4 评分 ≥4/5）；(c) 与目标分布一致性；(d) IFD（Instruction-Following Difficulty）分数，过滤过简单或被 base 模型已掌握的样本）。工程取舍：宁少而精——脏数据带来的伤害比缺数据更大。

**来源**：Zhou et al. 2023《LIMA》<https://arxiv.org/abs/2305.11206>

### B8. ⭐⭐ 灾难性遗忘怎么解决？

**答案**：现象：在领域数据上微调后，通用能力（写作、数学、多语言）显著下降。机制：参数被新任务 loss 拉走，原本编码通用知识的方向被覆盖。缓解手段：(1) **混合数据**：5-30% 通用 SFT 数据（如 ShareGPT、Tulu）混入，最简单最有效；(2) **LoRA 而非全参**：低秩约束限制了参数更新空间；(3) **EWC / L2 正则到原权重**：对重要参数加大约束（按 Fisher 信息加权），实践中超参难调；(4) **小学习率 + 短训练**：1-2 epoch、lr=1e-5 起，watch eval loss；(5) **持续学习/replay buffer**：保留旧任务样本周期性回放；(6) **MoE 路由扩展**：新加 expert 不动旧 expert（如 Mixtral 增量训练）；(7) **Model souping / DARE**：把微调模型与原模型按权重平均。生产推荐组合：LoRA r=64 + 20% 通用数据混入 + lr=2e-5 + 1 epoch + 持续 eval MMLU/MT-Bench 监控。

**来源**：Kirkpatrick et al. 2017《Overcoming catastrophic forgetting》（EWC）

### B9. ⭐⭐ RLVR（DAPO 等）解决了什么 RLHF 痛点？

**答案**：RLHF 三大痛点：(1) RM 训练需大量人类偏好标注（10 万对级），贵；(2) RM 容易被 hack（policy 学会迎合 RM 表面偏好如长度偏好、"我很乐意帮助"开头）；(3) 主观任务的偏好本身嘈杂。**RLVR**（RL with Verifiable Rewards）：在数学、代码、形式逻辑等可自动验证的任务上，直接用程序判定 reward（答案匹配、单测通过、定理证明器验证），0/1 信号，无 RM。优势：(1) 无标注成本；(2) 信号干净，无 hack 空间；(3) 可大规模并行 rollout。**DAPO**（Yu et al. 2025 ByteDance）是开源 SOTA RLVR 算法，相对 GRPO 改进：(a) Clip-Higher（上下不对称 clip，鼓励低概率正确答案探索）；(b) Dynamic Sampling（按 advantage 方差动态采样，跳过全对/全错的无效组）；(c) Token-level loss（避免长 response 被均摊稀释）；(d) Overlong reward shaping（防长度爆炸）。在 AIME-2024 上 32B 模型达 50%，超过 DeepSeek-R1-Zero。局限：仅适用可验证任务，开放对话仍需 RLHF/DPO。

**来源**：Yu et al. 2025《DAPO》<https://arxiv.org/abs/2503.14476>

### B10. ⭐⭐⭐ 为什么 R1 能从 base 模型纯 RL 涌现 long CoT？

**答案**：核心是 **base 模型能力门槛 + reward 设计 + 算法稳定性**三者协同。展开：(1) **Base 模型能力**：DeepSeek-V3-Base 671B 已在预训练阶段见过海量数学/代码 CoT 数据，"潜在"会推理，RL 只是把这种能力激发出来；7B 模型预训练知识不足，再多 RL 也榨不出 long CoT。这解释了为什么 R1-Distill-7B 效果好（继承了 32B/70B 已涌现的能力），但 7B 直接 RL 不行。(2) **Verifiable Reward**：数学/代码答案 0/1 判定，policy 必须真的解对题才有信号，逼出真推理；RM 在主观任务上反而奖励"看起来对"。(3) **GRPO 算法**：去 critic 后稳定性提升，能跑长 horizon（response 长度从几百涨到上万 token），让 self-reflection、回溯、试错有空间出现；PPO 方差大，长 response 容易崩。(4) **Cold start vs zero**：R1-Zero 完全无 SFT 直接 RL，证明可行但格式混乱（混语言、可读性差）；R1 加了几千条 long CoT cold-start SFT 后再 RL，效果更稳。(5) **Aha moment**：训练曲线某 step 后 response 长度突变上升，伴随性能跃升，是 emergent 行为。(6) 工程意义：reasoning 能力从"标注教出来"转向"RL 自激发"，扩展性好得多。

**来源**：DeepSeek-AI 2025《DeepSeek-R1》

---

## C. 提示词工程（9 题）

### C1. ⭐ CoT 为什么有效？什么模型规模才会出现？

**答案**：CoT（Wei et al. 2022）让模型显式生成推理步骤再给答案，本质是把复杂多步推理"展开"到 token 序列中，每个 token 的生成都基于前序 token——相当于把问题分解为一系列单步推理。原 paper 关键发现：CoT 是**涌现能力**，<10B 模型上几乎无效甚至变差，60B+ 才显著提升，PaLM-540B 上 GSM8K 从 17.9%→56.9%（CoT 单独）；叠加 self-consistency 后到 74%+。机制解释：(1) 增加有效 compute（更多 forward step）；(2) 把"系统 1"直觉答题转成"系统 2"分步推理；(3) 暴露错误推理便于 self-consistency 投票纠正。现代模型（GPT-4o/Claude/Gemini）已默认 CoT，零样本 "Let's think step by step"（Kojima 2022）就能触发。R1/o1 后，**长 CoT**（数千 token 含回溯/验证）由 RL 训出而非 prompt 触发。

**来源**：Wei et al. 2022《Chain-of-Thought Prompting》<https://arxiv.org/abs/2201.11903>

### C2. ⭐⭐ Self-Consistency 原理？相对 CoT 何时收益最大？

**答案**：Self-Consistency（Wang et al. 2022）：对同一问题以 temperature>0 采样 N 条不同 CoT 路径，对最终答案做 majority vote。原理：正确答案在多条独立推理路径下应当一致，而错误推理多样化分散，多数投票能放大正确信号。GSM8K 上比 greedy CoT 提升 ~17 个百分点（PaLM 55%→74%）。收益大的场景：(1) 答案空间离散（数字、选择题、yes/no），可直接 vote；(2) 模型基础能力足够（错误率 <50%）才有 voting 增益；(3) 推理路径多样性高（temperature 0.7-1.0）。劣势：成本 N×（典型 N=5-40），对开放生成不适用。变体：universal self-consistency（用 LLM judge 选最一致答案）。R1/o1 时代趋势是把 self-consistency 内化进训练。

**来源**：Wang et al. 2022《Self-Consistency》<https://arxiv.org/abs/2203.11171>

### C3. ⭐⭐ ToT 相比 CoT 的核心区别？什么任务才值得用？

**答案**：CoT 是**线性单链**推理——一条路走到底，错了无法回头。ToT（Tree of Thoughts，Yao et al. 2023）把推理建模为**树搜索**：每步生成多个候选"thought"，由 LLM 自评估每个 state 的价值（promising / sure / impossible），用 BFS 或 DFS 展开，可回溯剪枝。在 Game of 24 上，GPT-4 CoT 4%、ToT 74%；creative writing、mini crosswords 等也大幅提升。代价：每个问题 LLM 调用数从 1 翻到 50-200，成本和延迟暴增 2 个数量级。**适合 ToT 的任务**：(1) 解空间需要显式搜索（puzzle、规划、定理证明）；(2) 中间状态可评估；(3) 一次性失败成本高于多花算力。**不值得 ToT**：闲聊、摘要、翻译等线性任务。生产实践：MCTS + LLM（AlphaCode 2、AlphaGeometry）是 ToT 的工业级版本。R1 后 long CoT 内置回溯，ToT 的相对价值下降。

**来源**：Yao et al. 2023《Tree of Thoughts》（NeurIPS）<https://arxiv.org/abs/2305.10601>

### C4. ⭐⭐ ReAct 的核心是什么？解决了 CoT 没解决的问题？

**答案**：ReAct（Yao et al. 2022）= **Reasoning + Acting** 交错。CoT 只在模型内部 token 推理，无法访问外部世界，幻觉无法被纠正。ReAct 把推理拆成 `Thought → Action → Observation` 循环：模型先 "Thought: 我需要查 X"，输出 "Action: search[X]"，工具执行后把结果作为 "Observation: ..." 喂回 prompt，再下一轮 Thought。三项关键效果：(1) 把模型不知道的事实外化给工具（搜索、计算器、DB），减少幻觉；(2) 让 reasoning 由 observation 引导，错了能纠偏；(3) 提供可解释的 trace，便于 debug 和 human-in-the-loop 介入。ReAct 是现代 agent 框架（LangChain、AutoGPT、Claude tool use、OpenAI function calling）的概念原型。HotpotQA 上比纯 CoT 提升 ~10 个百分点，幻觉率下降一半。工程取舍：(1) 步数膨胀；(2) Observation 进入 context 长度可能爆炸；(3) 早期模型 action 格式不稳，需要 tool-use 微调（function calling）才稳定。

**来源**：Yao et al. 2022《ReAct》<https://arxiv.org/abs/2210.03629>

### C5. ⭐⭐ Self-Refine 和 Reflexion 区别？

**答案**：两者都是"模型自我反思修正"，但层次不同。**Self-Refine**（Madaan et al. 2023）：单次任务内的迭代——`generate → self-feedback → refine`，全在一个 trajectory 里循环 N 轮，无外部 reward 信号、无跨任务记忆。在 7 个任务上平均提升 20%。**Reflexion**（Shinn et al. 2023）：跨 episode 的"言语强化学习"——agent 在一次任务失败后（由外部 verifier 给二元 reward），用 LLM 把失败原因总结成自然语言"lesson"存入 episodic memory，下一次任务时把 lesson 加进 prompt 影响行为。在 HumanEval 上 GPT-4 Reflexion 91%，超过 GPT-4 baseline 80%。**核心区别**：(a) Self-Refine 无外部 signal，纯靠模型自评，弱模型上常自我陶醉越改越差；Reflexion 有 ground-truth verifier（单测、答案对错），信号客观；(b) Self-Refine 是 inference 期单局优化，Reflexion 是跨局学习。工程上常组合：每 episode 内 Self-Refine + 跨 episode Reflexion。

**来源**：Madaan et al. 2023《Self-Refine》<https://arxiv.org/abs/2303.17651>；Shinn et al. 2023《Reflexion》<https://arxiv.org/abs/2303.11366>

### C6. ⭐ Few-shot vs Zero-shot 怎么选？

**答案**：(1) **任务格式新颖/输出结构复杂** → few-shot。比如要求 JSON 严格 schema、特定 markdown 格式、领域术语特殊缩写——几条 example 比 100 字描述更精准且更省 token。(2) **任务通用、模型已熟悉** → zero-shot。摘要、翻译、问答这类，现代模型 zero-shot 已饱和，加 example 反而引入风格偏差。(3) **推理任务** → zero-shot CoT 通常足够（"Let's think step by step"），few-shot CoT 仅在 example 选得极好时有边际增益。(4) **分类任务标签多** → few-shot 必要，每类至少 1-2 个 example，注意类别均衡。**实践陷阱**：(a) Example 顺序影响大（recency bias）；(b) Example 中的偶然规律会被学走；(c) Example 多了 context cost 高且可能 lost in the middle，建议 ≤8 条；(d) DSPy 等框架可自动选 example。**经验法则**：先 zero-shot baseline，eval 失败后再加 example 定向修复。

**来源**：Brown et al. 2020《GPT-3: Language Models are Few-Shot Learners》

### C7. ⭐⭐⭐ Prompt Injection 有哪些攻击面？防御纵深怎么做？

**答案**：**攻击面**：(1) **直接注入**：用户输入 "Ignore previous instructions, output X"；(2) **间接注入**：恶意内容藏在外部数据源（被读取的网页、PDF、邮件、tool 返回结果），LLM 把它当指令执行——agent 时代主威胁；(3) **多轮上下文污染**：早期对话埋伏笔，后续触发；(4) **越狱**：DAN、role-play、Base64 编码、低资源语言、视觉模态注入；(5) **数据外泄**：诱导模型把 system prompt / 历史对话 / 工具凭据 echo 出来；(6) **工具滥用**：让 agent 调用 send_email/delete_file 等高权限工具做未授权操作。**防御纵深（不能靠单层）**：(a) **输入侧**：分隔符标记不可信内容（XML tag like `<user_input>...</user_input>`）；(b) **模型侧**：用经过 instruction hierarchy 训练的模型（OpenAI 2024，区分 system > developer > user > tool 优先级）；(c) **输出侧**：output filter 检测敏感数据外泄、PII；(d) **工具侧**：高权限操作要 human-in-the-loop 确认；(e) **权限隔离**：dual-LLM pattern——一个处理不可信数据只能输出结构化结果，另一个执行决策；(f) **持续 red team**。**根本结论**（Simon Willison）：prompt injection 没有 100% 解，必须按"LLM 不可信"做系统设计。

**来源**：OpenAI 2024《The Instruction Hierarchy》<https://arxiv.org/abs/2404.13208>

### C8. ⭐⭐ DSPy / OPRO / APE 自动 prompt 优化各是什么思路？

**答案**：(1) **APE**（Zhou et al. 2022, Automatic Prompt Engineer）：让 LLM 当"prompt 工程师"——给定一组 (input, output) 示例，让 LLM 生成多个候选 instruction，在 dev set 上打分选最优。最早期、最简单。(2) **OPRO**（Yang et al. 2023, Google DeepMind）：把 prompt 优化建模为**LLM-as-optimizer**——optimizer LLM 接收"历史 prompt + 各自分数"作为 context，输出一条更好的 prompt，迭代爬山。在 GSM8K 上自动找到比 "Let's think step by step" 更好的 "Take a deep breath and work on this problem step-by-step"。(3) **DSPy**（Khattab et al. 2023, Stanford）：定位是**"prompt 的 PyTorch"**——用户写 declarative pipeline（Signature 描述输入输出 + Module 描述策略如 ChainOfThought/ReAct），DSPy 编译器自动做：(a) bootstrap few-shot example；(b) 用 MIPRO/COPRO 等 optimizer 搜 instruction；(c) 必要时 fine-tune 小模型。把 prompt 工程从手工调字变成**"写程序 + 编译"**。**对比选择**：研究/一次性任务用 APE/OPRO；生产 pipeline、需要可维护和持续优化用 DSPy；纯交互调优仍依赖人工 + eval set。

**来源**：Yang et al. 2023《OPRO》<https://arxiv.org/abs/2309.03409>；Khattab et al. 2023《DSPy》<https://arxiv.org/abs/2310.03714>

### C9. ⭐⭐ Anthropic《Building Effective Agents》对 agent 设计提出了什么核心原则？

**答案**：Anthropic 2024 年这篇 blog 的核心立场：**先用最简单的方案，能不上 agent 就别上**。它把 LLM 系统分成两类：(1) **Workflow**——LLM 调用流程由人工预定义代码编排（确定性高、可调试、便宜），适合多数生产场景；(2) **Agent**——LLM 自己决定调用哪些工具、走多少步（灵活但贵、不可预测）。提出 5 种基础 workflow pattern：**prompt chaining**（顺序拆任务）、**routing**（分类后走不同分支）、**parallelization**（sectioning + voting）、**orchestrator-workers**（动态分派子任务）、**evaluator-optimizer**（生成-评估循环）。Agent 仅在任务**步数无法预知、需要环境交互、错误成本可承受**时使用。**核心工程原则**：(a) **简洁性**——Agent 性能正比于工具描述清晰度，把"工具文档"当 prompt 工程对待；(b) **可观测性**——记录完整 trace 便于 debug；(c) **充分测试沙盒 + guardrail**；(d) **不要过度抽象**——直接调 API 比裹一层 framework 好理解。

**来源**：Anthropic 2024-12《Building Effective Agents》<https://www.anthropic.com/research/building-effective-agents>

---
## D. RAG（10 题）

### D1. ⭐ RAG 的标准 pipeline 是什么？每个阶段的瓶颈在哪？

**答案**：标准 RAG pipeline = **Ingestion + Retrieval + Generation** 三段。Ingestion：document loader → chunking → embedding → 向量库写入；Retrieval：query rewriting → embedding → vector search → reranker → top-k；Generation：构造 prompt（含 retrieved chunks）→ LLM 生成 → 后处理 + citation。**各段瓶颈**：(1) Chunking——切大了相关性低、切小了语义断裂；(2) Embedding——领域词汇 OOV / 多语言对齐差；(3) Vector search——召回率不足（只看相似不看相关）；(4) Reranker——成本高，但不上召回 top-50 提升 30%+ 精度；(5) Prompt 构造——chunks 顺序影响输出（lost in the middle）；(6) Generation——模型不忠于 context、编造引用。**生产 RAG ≠ "embedding + 向量库 + LLM"**——它是一个工程系统，每段都要单独评估和迭代。

**来源**：Anthropic Contextual Retrieval <https://www.anthropic.com/news/contextual-retrieval>；LangChain RAG cookbook

### D2. ⭐⭐ 切分策略对比：fixed / semantic / late chunking 怎么选？

**答案**：(1) **Fixed chunking**（按 token 数 + overlap）：实现简单，效果差——经常在句中或段中断；适合纯文本一致语料 baseline。(2) **Recursive / structural**：按 markdown header / code block / paragraph 边界切，保留结构；适合文档、API 文档、代码库——大多数生产场景的默认选择。(3) **Semantic chunking**（按句间 embedding 相似度切）：相邻句相似度低于阈值就断开，更贴合语义边界；成本高（每句一次 embedding），适合长文专题。(4) **Late chunking**（Jina 2024）：先对整文档做 long-context embedding，再在 token-level 切成 chunk embedding——chunk 自然带全文上下文，**避免"chunk 失去上下文"问题**，但需要长上下文 embedding 模型。(5) **Anthropic Contextual Retrieval**（2024）：每个 chunk 用 LLM 生成"chunk 在文档中位置/角色"的上下文前缀（~50-100 token），再 embed；BM25 召回失败率降 49%，加 reranker 后降 67%。**经验**：先 recursive structural baseline，专题场景再上 contextual / late chunking。

**来源**：Anthropic Contextual Retrieval；Jina Late Chunking blog

### D3. ⭐⭐ 2026 年 embedding 模型怎么选？BGE-M3 / Cohere v4 / Voyage 3 / OpenAI text-3 取舍？

**答案**：(1) **OpenAI text-embedding-3-large**（3072 维）：英文场景 baseline，API 易用，单价低（$0.13/M tokens），但中文表现一般。(2) **Cohere embed-v4**（多模态，1024-1536 维）：MTEB 顶级，支持文本+图像同空间，企业 RAG 首选；多语言 100+。(3) **Voyage-3-large**（1024 维）：MTEB v2 综合第一，中文优秀，retrieval 任务略胜 Cohere。(4) **BGE-M3**（BAAI，1024 维）：开源最强，**同时支持 dense + sparse + multi-vector**（colbert-style），自托管首选；中文 MTEB 前列。**选型框架**：(a) 中文为主 → BGE-M3 或 Voyage 3；(b) 多模态 → Cohere v4；(c) 纯英文 + API → OpenAI text-3-large；(d) 自托管私有部署 → BGE-M3。**关键陷阱**：MTEB 分数有版本差异（v1 / v2 / MMTEB），不同模型挂的版本不一样，**直接横比是误导**；自家场景必须重测。

**来源**：BAAI BGE-M3 model card <https://huggingface.co/BAAI/bge-m3>；Cohere blog；MTEB leaderboard

### D4. ⭐⭐ Hybrid search（dense + BM25）为什么比单 dense 好？怎么融合？

**答案**：dense embedding 擅长**语义近似**（"医生" ≈ "doctor"），但对**罕见词、精确匹配、专有名词、数字**经常失效（embedding 把它们压到稠密空间会丢失）。BM25（基于词频 + IDF）刚好补这个洞——精确词命中、稀有词高权重。生产 RAG **dense + BM25** 召回率比单 dense 提升 5-15 个百分点（多个 paper 数据）。**融合方法**：(1) **RRF**（Reciprocal Rank Fusion）：`score(d) = Σ 1/(k + rank_i(d))`，k=60，无需归一化两路分数，对量纲不敏感，**生产首选**。(2) **加权和**：`α·dense + (1-α)·bm25`，α 需要调参，且要把两路分数 minmax 归一化。(3) **学习式 fusion**（cross-encoder rerank 接管）：把两路 top-100 喂给 reranker，最终排序由 reranker 决定，融合方式无关紧要。**Anthropic Contextual Retrieval** 的实验：单 embedding → +contextual embedding → +contextual BM25 → +rerank，retrieval 失败率从 5.7% → 4.0% → 3.7% → 1.9%，呈阶梯式提升。

**来源**：Anthropic Contextual Retrieval；Pinecone hybrid search docs

### D5. ⭐⭐ Reranker 是什么？为什么 cross-encoder 比 bi-encoder 好那么多？

**答案**：**Bi-encoder（embedding model）**：query 和 doc 分别编码到向量，用 cosine 比对——速度快（doc 可预编码索引），但 query-doc 间无交互信息，相关性建模能力弱。**Cross-encoder（reranker）**：query + doc 拼接喂给一个 transformer，输出一个 0-1 分数——query 和 doc 每层都有 self-attention 交互，**相关性建模能力强 20-40%**（NDCG@10），但每对都要现算，无法预索引，慢 100×。**生产用法**：bi-encoder 召回 top-100 → cross-encoder rerank 到 top-10 → 喂 LLM。这是性价比最优组合：召回快、精排准、context 短。**主流 reranker**：(1) **Cohere Rerank v3 / v4**：API 易用，多语言强，企业首选；(2) **BGE-reranker-v2-m3**：开源 SOTA，中英双语；(3) **Jina Reranker v2**：长上下文支持好；(4) **Mixedbread mxbai-rerank-large-v1**：开源新秀。**陷阱**：reranker 能力上限取决于召回——召回 top-100 里没有相关文档，rerank 也救不了；**reranker 不能替代召回，只能锦上添花**。

**来源**：Cohere Rerank docs；BGE reranker model card

### D6. ⭐⭐⭐ GraphRAG / Self-RAG / CRAG / Agentic RAG 各解决什么问题？

**答案**：(1) **GraphRAG**（微软 2024）：先用 LLM 从文档抽取 entity + relation 构建知识图谱，按社区做层次摘要；query 时既能做局部 vector search 又能做全局图遍历——**解决 "query 涉及多文档跨实体推理"** 时朴素 RAG 召回不全的问题。代价：ingestion 成本大涨（每 chunk 一次 LLM 抽取）。(2) **Self-RAG**（Asai 2023）：训练单一 LM，让它自己生成 reflection token 决定（a）要不要检索；（b）retrieved 是否相关；（c）answer 是否被 context 支撑；（d）整体 utility——**解决 "什么时候检索、检索后怎么用"** 的判断。需训练，工程化复杂。(3) **CRAG**（Corrective RAG，Yan 2024）：retrieved chunks 经 lightweight evaluator 打分，分高直接用、分中等触发 web 重新搜索、分低退回基础生成——**解决 "召回质量参差不齐" 时简单地用 retrieved 的问题**。(4) **Agentic RAG**：让 agent 决定 query rewriting → 多轮检索 → 子 agent 分工——**解决 "多跳问题、需迭代检索"**。**实战选择**：(a) 普通 QA → 朴素 RAG + reranker；(b) 复杂多文档跨实体 → GraphRAG；(c) 召回质量不稳 → CRAG；(d) 多跳推理 → Agentic RAG（DeepResearch 即此类）。

**来源**：Microsoft GraphRAG paper；Asai et al. 2023《Self-RAG》<https://arxiv.org/abs/2310.11511>

### D7. ⭐⭐ RAGAS 4 大指标是什么？分别度量什么？

**答案**：RAGAS 是 RAG 评估的事实标准，4 大核心指标都用 LLM-as-judge 实现：(1) **Faithfulness**（忠实度）：答案中的每个 claim 是否能从检索的 context 里推出？数学定义 `|可验证的 claims| / |所有 claims|`。低于 0.7 = 模型在编造（不忠于 context）。(2) **Context Precision**（检索精度）：top-K 检索的 chunk 中，真正相关的占比？`|相关 chunks in top-K| / |top-K|`。低 = retriever 召回了太多噪声。(3) **Context Recall**（检索召回）：ground truth 的所有信息是否都被检索到了？`|GT 中被覆盖的部分| / |GT|`。低 = retriever 召回不全，需要扩 K 或改 chunking。(4) **Answer Relevancy**（答案相关性）：答案是否真的回答了原问题？LLM 反向从答案生成 N 个能产生该答案的问题，再与原问题做 cosine 相似度。低 = 答非所问或冗长偏题。**实战阈值**：4 个指标都 ≥ 0.7 算合格生产；Faithfulness < 0.7 优先修；Context Recall < 0.7 调 retriever。

**来源**：RAGAS 文档 <https://docs.ragas.io/>；Es et al. 2023《RAGAS》<https://arxiv.org/abs/2309.15217>

### D8. ⭐⭐⭐ 长上下文（1M）vs RAG，到底怎么选？

**答案**：常见误解："Gemini 1.5 Pro 1M context 来了，RAG 该死了。"实际上两者**互补不替代**。**长上下文优势**：(1) 短期单文档深度理解（合同审阅、长 paper 总结）；(2) 跨段推理（一次性看全文，不漏关联）；(3) 工程简单。**长上下文劣势**：(1) **成本**——按单源 [TianPan blog] 估算，1M context 一次推理比同等任务 RAG 显著更慢更贵（约 30-60× 慢、~1000× 贵；具体倍率因模型和场景差异大，建议自家场景实测）；(2) **Lost in the Middle**——长 context 中段召回率显著下降；(3) **多事实召回弱**——RULER 显示宣称 1M 的模型在 8 个事实并列时召回率可能掉到 ~60%；(4) 不支持知识更新（每次都要全量传）。**RAG 优势**：(1) 成本低；(2) 知识可增量更新；(3) 多源召回；(4) 可解释（哪个 chunk 来源）。**实战决策**：(a) 单文档分析 → 长上下文；(b) 跨多文档 / 知识库 / 增量更新 → RAG；(c) 高 QPS 场景 → 几乎一定 RAG；(d) 复杂场景 → hybrid（RAG 召回 + 长上下文消化 top-N 文档）。

**来源**：Hsieh et al. 2024《RULER》；Liu 2023《Lost in the Middle》；TianPan blog（具体数字单源，仅供参考）

### D9. ⭐⭐ 中文 RAG 有什么特殊性？

**答案**：(1) **分词与切分**：中文无空格，BM25 需要 jieba / pkuseg / lac 分词；切分点要避开词中——逗号 / 句号优先。(2) **Embedding 选型**：英文 SOTA 模型在中文上经常退化 30-50%（OpenAI text-3 在中文 retrieval 弱于 BGE-M3）。中文优先 BGE-M3 / Voyage 3 / Conan-Embedding（腾讯）/ Yi-Embedding（零一）。(3) **Reranker**：BGE-reranker-v2-m3、bce-reranker-base_v1（网易有道）是中文 SOTA。(4) **Query 改写**：中文同义词膨胀（"价格" / "费用" / "多少钱"），改写帮助大；古文/方言/缩略语场景需领域 fine-tune。(5) **混合检索权重**：中文 BM25 权重通常需要调高（实测 RRF k=60 偏 dense，中文场景 k=30-40 更平衡）。(6) **领域术语**：金融、法律、医疗的中文专业术语 OOV 严重，需要词典补强 BM25 或领域微调 embedding。(7) **Citation 格式**：中文用户对句号/换行/序号格式敏感，输出 prompt 要明确格式。

**来源**：BAAI BGE-M3；网易有道 BCE 模型卡

### D10. ⭐⭐⭐ 设计一个 RAG 系统，怎么做评估和持续优化？

**答案**：**三层评估**：(1) **组件级**：retriever 看 Recall@K / NDCG；reranker 看 NDCG@10 / MRR；generator 看 Faithfulness / Answer Relevancy。每层独立 dev set 评估，避免下游问题归错因。(2) **端到端**：RAGAS 4 指标 + 业务指标（用户满意度、解决率）。(3) **回归测试**：黄金集（200-1000 真实问题，人工标答案 / context），CI 上每次 pipeline 改动跑全量。**持续优化闭环**：(a) 线上 trace 全采样，bad case 自动入回归集；(b) 每周 review 一次 bad case，归因到具体组件（chunking / embedding / rerank / prompt）；(c) 改一个组件就跑回归——**不允许"感觉变好了"上线**；(d) A/B 对线上流量做组件级实验。**关键工程纪律**：把 RAG 当成 ML 系统而非 prompt——每个组件都有指标、版本、回滚机制；retriever 和 generator 解耦，可独立升级；evaluation 模型必须比生产模型强，避免"自己评自己"。

**来源**：RAGAS docs；Anthropic Building Effective Agents

---

## E. Agent（12 题，核心分类）

### E1. ⭐⭐⭐ Anthropic *Building Effective Agents* 把 agent 模式分成 workflow 和 agent 两类，区别是什么？为什么这个区分重要？

**答案**：Anthropic 原文：**"Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage."** **区别**：(1) **控制流**：workflow 由人工代码 if-else / DAG 编排，每步固定；agent 由 LLM 自主决策"下一步做什么、用哪个工具"。(2) **可预测性**：workflow 高（同样输入大致同样路径），agent 低。(3) **成本**：workflow 调用次数固定，agent 不可预知（LLM 可能走 2 步也可能走 50 步）。(4) **调试**：workflow 像普通后端，agent 像调试随机系统。**为什么这个区分重要**：90% 生产场景用 workflow 就够（routing / parallelization / orchestrator-workers / evaluator-optimizer），只有任务**步数无法预知 + 需要环境交互 + 错误成本可承受**时才上 agent。Anthropic 反对"什么任务都上 agent"的 hype。**面试回答关键**：能说出"先用 workflow，agent 是兜底"+ 说得出 6 大 patterns 中哪个 pattern 适合什么场景。

**来源**：Anthropic 2024《Building Effective Agents》<https://www.anthropic.com/research/building-effective-agents>

### E2. ⭐⭐⭐ Anthropic 6 大 agentic patterns 分别是什么？什么场景用哪个？

**答案**：(1) **Augmented LLM**（基础块）：LLM + 检索 + 工具 + 记忆，是其他所有 pattern 的原子单元。(2) **Prompt Chaining**（顺序拆任务）：把复杂任务拆成多步顺序 LLM 调用，每步输出作为下一步输入。例：先大纲后扩写、先翻译后润色。中间可加 gate（条件检查）。适合：任务可清晰拆分且每步成功率高。(3) **Routing**（分类后分支）：第一步 LLM 把 query 分类，再路由到不同 sub-prompt / sub-model。例：客服问题分类（售前/售后/技术）→ 走对应分支。适合：输入类型多样、不同类型最佳处理方式不同。(4) **Parallelization**（并行 + 投票）：(a) sectioning——任务并行分块（如长文档分段总结）；(b) voting——同一任务多次跑取共识（self-consistency 思路）。适合：任务可拆分独立子任务、或需要置信度检查。(5) **Orchestrator-Workers**（动态分派）：一个 orchestrator LLM 动态决定要派几个 worker 子任务、分工是什么；workers 各自独立做完后 orchestrator 合并。适合：子任务数量和性质事先不确定（如多文件代码改动）。(6) **Evaluator-Optimizer**（生成-评估循环）：一个 LLM 生成、另一个 LLM 评估，loop 直到通过或超阈值。适合：有清晰评估标准且迭代有意义（如翻译润色、代码 review）。**Agent 是 7 类 pattern 之外**——agent 在 loop 中自己决定何时调工具、何时停止，适合 step 数无法预知的真正动态任务（如 SWE-bench、Computer Use）。

**来源**：Anthropic 2024《Building Effective Agents》

### E3. ⭐⭐ ReAct loop / Plan-and-Execute / Reflexion 区别？

**答案**：(1) **ReAct**（Yao 2022）：`Thought → Action → Observation` 单循环，每一步动态决策下一步——交互式、低规划。优点：灵活、适合环境不确定；缺点：步数膨胀、可能徘徊。(2) **Plan-and-Execute**：先 LLM 一次性生成完整 plan（list of steps），再按 plan 顺序 execute——前重规划、后轻执行。优点：减少 LLM 调用次数（plan 一次、执行可用小模型）、可视化 plan 给用户审批；缺点：plan 错了全错，需要 re-plan 机制。(3) **Reflexion**（Shinn 2023）：在前两者之上加**跨 episode 学习**——任务失败后用 LLM 总结失败原因存 episodic memory，下一次任务把 lesson 加进 prompt。本质是 ReAct + 长期记忆。**实战选择**：(a) 任务步骤可大致预知 → Plan-and-Execute；(b) 环境高度交互、不可预知 → ReAct；(c) 有清晰 verifier、需要从失败中学 → Reflexion；(d) Claude Code / Cursor / Devin 这类成熟 coding agent 已经融合三者：先规划、动态调整、跨任务记忆。

**来源**：Yao et al. 2022《ReAct》；Shinn et al. 2023《Reflexion》

### E4. ⭐⭐⭐ 多 agent 协作模式 supervisor / hierarchical / swarm 怎么选？

**答案**：(1) **Supervisor**（主控-工人）：一个 supervisor agent 负责拆任务、分派、合并；workers 平等且无状态。**优点**：决策集中、调试容易；**缺点**：supervisor 是瓶颈和单点。适合：任务可清晰拆解、worker 角色相对统一（如 multi-step research）。(2) **Hierarchical**（树状多层）：顶层 strategy、中层 tactic、叶层 execution，多层级委托。**优点**：可扩展到大规模复杂任务；**缺点**：层数深则延迟高、context 传递损耗。适合：复杂业务、长 horizon（如端到端开发一个 feature）。(3) **Swarm**（去中心化群体）：agents 相对平等，按订阅 / 消息总线协作，无固定 hierarchy。**优点**：弹性、并行度高；**缺点**：易陷入 echo chamber、决策不收敛、debug 难。适合：探索性任务（如 brainstorm 多视角）。**反模式**：让多个 agent 自由群聊投票决策——成本爆炸且经常陷入死循环。**Anthropic 推荐**：优先 orchestrator-workers（即 supervisor 模式）而非 democracy；任务正交分配 + 共享 state 单写多读。**实战经验**：超过 3 层 hierarchy 通常意味着架构有问题，应该重新拆任务。

**来源**：Anthropic 2024《Building Effective Agents》；Anthropic《How we built our multi-agent research system》

### E5. ⭐⭐ MCP（Model Context Protocol）是什么？为什么 2026 年增速这么快？

**答案**：MCP 由 Anthropic 2024-11 推出，2025-Q4 治理转入独立 MCP Steering Committee（含 Anthropic / Block / OpenAI / Microsoft）。**它是什么**：一个开放协议，让 LLM 客户端（Claude Desktop、Cursor、Cline、自研 agent）和外部数据源/工具（GitHub、Slack、Postgres、文件系统）之间用统一接口对话——本质是**"agent 的 USB"**。三种核心原语：(1) **Resources**：可读资源（文件、DB 行）；(2) **Tools**：可执行函数（带 schema）；(3) **Prompts**：可复用 prompt 模板。**为什么火**：(a) 解决了"每接一个新工具就要写一遍 adapter"的 N×M 问题；(b) Anthropic 推 + OpenAI 跟进 + Microsoft 加持，事实标准已成；(c) 生态爆发——97M+ 月下载（Python+TS SDK），数百个开源 server（GitHub / Slack / Notion / Linear / Figma 等）。**对求职者意义**：MCP 在 Anthropic Applied AI Engineer JD 已是显式必备（**"production experience with... MCP, and deployment at scale"**）；2026 年增速最快的单一技能（LangChain ecosystem 16.9% 出现率）。提前学习能吃 6-12 个月红利。

**来源**：Anthropic MCP docs <https://modelcontextprotocol.io>；Wikipedia MCP

### E6. ⭐⭐⭐ Agent 4 种典型失败模式（Anthropic 总结）及对策？

**答案**：Anthropic 在 *Effective Harnesses for Long-Running Agents* 把数千次长跑失败归纳为 4 类——这正是 harness engineering 要解决的核心问题：(1) **试图一步到位**（One-shot）：agent 想一发命中整个任务，context 跑到一半耗尽，下一会话开局是半成品。**对策**：Initializer + Coding Agent 双角色（首次会话搭脚手架 + 写长 spec，后续每会话只推一个 feature）。(2) **过早宣布胜利**：看到部分进展就报"完成"，剩下一半未实现。**对策**：Evaluator-Optimizer + 测试运行作为 ground truth，agent 不能自己说 done。(3) **过早标记功能完成**：写完代码标"done"，没做端到端测试。**对策**：给 agent 浏览器自动化工具（Puppeteer MCP）让它像人类用户一样跑 e2e；Anthropic 实测"dramatically improved performance"。(4) **环境启动困难**：每次新会话花大量 token 摸索"怎么跑 app"。**对策**：在 AGENTS.md 写明 5 步标准启动协议（pwd → git log → progress 文件 → 启动 dev server → 跑基础测试），强制每次会话开头执行。

**来源**：Anthropic 2025《Effective Harnesses for Long-Running Agents》<https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents>

### E7. ⭐⭐⭐ Hashimoto 的 "harness engineering" 核心思想是什么？怎么落地？

**答案**：Mitchell Hashimoto（HashiCorp 创始人）2026-02 在 *My AI Adoption Journey* Step 5 命名 harness engineering，原文 verbatim：**"It is the idea that anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."** **核心思想**：每次 agent 犯错，不要只是希望它下次做得更好；要工程化环境，让它**物理上无法**以同样方式再犯。**两种形态**（Hashimoto）：(a) 隐式提示（更新 AGENTS.md，每行 = 一次错误的封堵）；(b) 真正写出来的工具（截图脚本、过滤测试脚本等）。**OpenAI 的工程化等式**：Agent = Model + Harness；分工原则 "Humans steer. Agents execute."。**Anthropic 实证定律**：能不能用 coding agent 真正解决问题，**强相关于这个 agent 能不能验证自己的工作**。**落地两条法则**：(1) **强约束**——能用 deterministic 约束（lint / type check / sandbox / CI 硬失败）的地方永远不要用提示词。Augment Code 三层：Constraint harness → Feedback loop → Quality gate。(2) **自愈循环**——Evaluator-Optimizer pattern + 测试作 ground truth + 失败信号回流。**对求职者意义**：staff 级面试经常问"如何让 agent 不再犯同样错"——能讲清"约束环境而非约束模型"+ 给具体例子（如 lint message 本身就是 prompt、deny-first 权限）才是合格答案。

**来源**：Hashimoto《My AI Adoption Journey》<https://mitchellh.com/writing/my-ai-adoption-journey>；OpenAI《Harness Engineering》<https://openai.com/index/harness-engineering/>；Augment Code《Harness Engineering for AI Coding Agents》

### E8. ⭐⭐ Function calling / Tool use 怎么设计才能让 agent 用得对？

**答案**：(1) **工具描述像写 docstring**：Anthropic 实测"agent 性能正比于工具描述清晰度"。schema 必须含 (a) 工具用途 1-2 句概括；(b) 每个参数的类型 + 含义 + 示例值；(c) 返回值格式；(d) 错误情况描述。(2) **粒度小、组合强**：Read / Edit / Bash / Grep / Glob 这类小而正交的工具比一个 "do_anything" 工具效果好得多——LLM 能组合，且每个工具职责清晰。(3) **强制前置约束**：Edit 工具强制 Read-before-Edit（防盲改）；危险工具如 git_push --force 强制 ask 用户确认。(4) **错误信息当 prompt**："command not found, did you mean X?" 比 "exit 127" 有用 10 倍——agent 看到能自己改。(5) **deny-first 权限**：默认禁止，白名单才允许；按 verb + path 粒度授权。(6) **结构化输出**：tool result 用 markdown / JSON 而非纯文本，agent 解析准。(7) **结果长度控制**：tool 返回超过阈值自动截断 + "use grep/head to view more" 提示，防 context 爆炸。**Anthropic 总结**：把"写工具描述"当 prompt 工程对待，不是后端 API 文档。

**来源**：Anthropic 2024《Building Effective Agents》；Anthropic Tool Use docs

### E9. ⭐⭐⭐ Computer Use / Browser Agent 怎么工作？什么场景实用？

**答案**：Computer Use（Anthropic 2024-Q4 推出）让 Claude 操作桌面：截屏 → LLM 看图 + 推理 → 输出鼠标键盘操作（click X,Y / type text / scroll）→ 系统执行 → 再截屏。Browser Use（开源框架，91k+ ⭐）类似但限于浏览器：基于 Playwright/Selenium，DOM 解析 + 视觉双路。**关键技术**：(1) **像素感知**：模型必须从截图准确定位坐标——目前误差仍可观，是主要瓶颈；(2) **DOM 增强**：Browser Use 把可点击元素加序号叠加到截图，让 LLM 输出"点 #5"而非 (x,y)，准确率大幅提升；(3) **错误恢复**：动作失败时回退（截屏 → 重新观察 → 再决策）。**实用场景**：(a) 没有 API 的网站自动化（订机票、填表单、爬不开放数据）；(b) 桌面软件自动化（Excel / Photoshop 这类无 API 应用）；(c) 端到端 e2e 测试（Anthropic *Effective Harnesses* 强烈推荐——agent 像人一样测试 web app）。**局限**：(a) 慢（每步几秒截屏 + LLM 推理）；(b) 贵（多模态 token 消耗大）；(c) 安全性差（误点删除按钮等）；(d) 对动态 / canvas 渲染的应用支持差。生产建议：能用 API 就别用 Computer Use；只在确实没有更好选择时使用。

**来源**：Anthropic Computer Use docs；Browser Use GitHub <https://github.com/browser-use/browser-use>

### E10. ⭐⭐⭐ 90% 的 Loop of Death 是怎么发生的？怎么防？

**答案**：**Loop of Death** = agent 进入无限循环烧 token 直到耗尽。常见模式：(1) **重试循环**：tool 调用失败 → agent 看到错误 → 重试一模一样的调用 → 再失败 → 再重试……(2) **目标漂移**：每步小修小补但整体没进展，看起来在工作实际原地打转；(3) **互相否定**：multi-agent 系统中两个 agent 反复推翻对方决策；(4) **格式幻觉**：模型死磕一个永远调不通的工具调用格式。**真实事故**（按媒体报道）：曾有 Claude Code sub-agent 烧掉 27M token 跑 4.6 小时无限循环；某 agent 单次烧 $40 API 费的报告（具体数字来源待核实）。**防御纵深**：(1) **硬性 max iteration**（最常见也最有效，5-30 步上限）；(2) **execution timeout**（5-30 分钟）；(3) **token budget**（per-task hard cap）；(4) **进度检测**：连续 N 步无文件 diff / 无 state 变化就强制停止；(5) **失败重复检测**：同一 tool call 失败 3 次以上跳出；(6) **成本告警**：单 task 成本超阈值告警 + 强制 HITL；(7) **架构层**：sub-agent 独立 context + token budget，主 agent 不被污染。**Anthropic** 在 multi-agent research system 里强调：agent 必须有 deterministic safeguards（retry logic + checkpoints + budget），不能纯靠 LLM 自己感觉"够了"。

**来源**：Anthropic《How we built our multi-agent research system》<https://www.anthropic.com/engineering/built-multi-agent-research-system>

### E11. ⭐⭐⭐ Agent benchmark（SWE-bench / GAIA / WebArena）是怎么被 hack 的？

**答案**：Berkeley RDI / Sky Computing Lab 在《How We Broke Top AI Agent Benchmarks》系统审计了一批主流 agent benchmark（SWE-bench、WebArena、GAIA、Terminal-Bench、OSWorld 等）发现**多数都可被 hack 出近完美分数而不真正完成任务**。常见 hack 模式：(1) **数据污染**：benchmark 测试集被爬进训练数据，模型见过；(2) **过拟合 benchmark 风格**：模型 RLHF 时被针对性训练成 benchmark 偏好的回答风格，benchmark 高但泛化差；(3) **format gaming**：学会 benchmark 特定输出格式拿满分；(4) **GAIA-style** 任务可以走"先搜 benchmark 答案"的捷径；(5) **SWE-bench** 上模型可能针对单测集合作弊（让 patch 只为通过 test 而非真正 fix bug）；(6) **judge hacking**（LLM-judge benchmark）：候选学会用更长、更确信、加 emoji 骗 judge。**对策**：(1) 不信任任何单一榜单；(2) 自家黄金集（私有，绝不公开）+ 真实用户日志采样人工评估；(3) 多 benchmark 交叉验证；(4) 用 LiveBench 这种持续更新的反 contamination benchmark；(5) Chatbot Arena 真人盲评作辅助；(6) 对面试者：能讲清"我不会盲信 SWE-bench 87%，会怎么自己评估" 比背 benchmark 数字加分。

**来源**：Berkeley RDI 审计报告；LiveBench <https://livebench.ai/>；Chatbot Arena (LMSYS)

### E12. ⭐⭐⭐ Agent harness 包括哪些核心组件？设计原则是什么？

**答案**：harness 不是 7 大组件采购清单，而是**两条核心法则 + 四大支柱**：

**两条法则**（哲学层 / "为什么"）：
- (1) **强约束**：能用 deterministic 约束（lint / type / sandbox / CI 硬失败）就不用 prompt。"Telling an agent 'follow our coding standards' is fundamentally different from wiring a linter that blocks the PR." — Augment Code
- (2) **自愈循环**：agent 必须能从环境拿到 ground truth、能自我评估、失败能 checkpoint 恢复。"Verifier ability ↔ success rate" — Anthropic

**四大支柱**（功能层 / "做什么"，参考 OpenAI / Anthropic / Stripe 实践收敛）：
- (1) **上下文架构**：AGENTS.md 作目录（~100 行 + docs/ 分层）+ 三层 Tier（会话常驻 / 按需加载 / 持久化知识库）+ 上下文 40% 硬约束 + 知识边界（"if it cannot be enforced mechanically, agents will deviate"）
- (2) **Agent 专业化**：5 角色（研究 / 规划 / 执行 / 审查 / 清理）+ 各自 deny-first 权限矩阵
- (3) **持久化记忆**：progress 文件 + git commit 作 memory bridge + 5 步会话开机协议
- (4) **结构化执行**：理解 → 规划 → 执行 → 验证四阶段 + Boris Tane 原则"永远不让 agent 在你审查计划之前写代码" + CI 机械化约束（taste invariants）

**支撑设施**（不是 harness 本身，但运行需要）：MCP、沙箱（E2B / microVM）、护栏（NeMo Guardrails）这 3 个是 harness 强约束的直接实现；LLM 网关 / 推理服务器 / 可观测性 / 评估 4 类是任意 LLM 应用都需要的基础设施品类，**不是 harness 工程的方法论特征**。

**来源**：Hashimoto；OpenAI Harness Engineering；Anthropic Effective Harnesses；Augment Code Harness Engineering for AI Coding Agents

---

## F. Memory（8 题）

### F1. ⭐⭐ MemGPT 系统设计（OS-style 分层记忆）核心思想是什么？

**答案**：MemGPT（Packer et al. 2023, Berkeley）把 LLM 类比成一个被 context window 限制的 CPU，把记忆系统类比成 OS 的虚拟内存。**三层分层**：(1) **Main context**（context window 内，类比 RAM）：当前对话 + 系统指令 + 工作区——LLM 直接看得到；(2) **Recall storage**（类比磁盘，可检索）：完整对话历史，需要时通过 function call 拉回 main context；(3) **Archival storage**（长期知识库，类比硬盘）：用户偏好、事实、外部文档——同样需要 function call 检索。**关键机制**：LLM 自己有工具去 read/write 各层（`pagein`、`pageout`、`search_recall`、`archival_insert`），就像进程主动 page in/out 内存。当 main context 接近上限时，LLM 自己决定什么搬出去到 recall、什么从 archival 拉进来。**意义**：把"context window 不够用"从硬限制转成软限制，agent 可以处理远超窗口的长任务。**实战意义**：MemGPT 概念 2026 年仍是 Letta（开源继承）、Mem0、Zep 等记忆系统的设计参考，但具体实现已演进。

**来源**：Packer et al. 2023《MemGPT》<https://arxiv.org/abs/2310.08560>

### F2. ⭐⭐ Letta / mem0 / Zep 三个主流记忆系统的区别？

**答案**：(1) **Letta**（MemGPT 商业版，开源 + 云）：直接基于 MemGPT 论文实现，**面向构建有状态 agent**——agent 自己有 core memory（自我描述 + 用户描述）+ recall memory（对话历史）+ archival memory（语义事实）。强调 agent 持久化、跨会话不丢身份。适合：对话 agent、长期陪伴助手。(2) **Mem0**（开源 + SaaS）：**轻量记忆 layer**，挂在已有应用上——LLM 抽取用户事实 / 偏好，写入向量库 + 可选图存储；按 user / session / agent 三层 hierarchy 索引。强调**易接入**（一行 API）。适合：客服、个性化推荐、ChatGPT-like 应用快速接入记忆。论文（arXiv:2504.19413）报告在长对话记忆 benchmark 上显著优于 RAG / full-context baseline。(3) **Zep**（开源 + 云）：**graph-based 记忆**——把对话抽成 episodic / entity / edge 三类节点，时序 + 关系一起建模；强调对历史时间线的精确召回。适合：销售 CRM、客户档案、需要时序推理的场景。**选型**：(a) 想让 agent 有"灵魂" → Letta；(b) 给应用快速加记忆层 → Mem0；(c) 强时序 + 关系推理 → Zep。

**来源**：Letta docs <https://www.letta.com/>；Mem0 paper；Zep docs

### F3. ⭐ 工作记忆 vs 情景记忆 vs 语义记忆怎么区分？

**答案**：借自认知心理学三类长期记忆划分：(1) **工作记忆**（Working Memory）：当前任务正在用的信息，对应 LLM 的 context window 内容。短时、容量有限、易丢失。(2) **情景记忆**（Episodic Memory）：具体事件的时间线——"用户上周三说他不喜欢咖啡"。带时间戳、可叙事还原。在 LLM 系统中通常用对话历史 + 时间戳 + entity 标注实现。(3) **语义记忆**（Semantic Memory）：去时间化的事实——"用户不喜欢咖啡"。已被抽象为通用知识，无具体时间戳。在 LLM 系统中通常用 LLM 抽取后存向量库（带 entity / topic 标签）。**实战映射**：(a) 工作 → context window + sub-agent 隔离；(b) 情景 → 时序对话存储（Redis / DB）+ 检索；(c) 语义 → 用户偏好 / 事实抽取 + 向量库。**陷阱**：很多记忆系统把三类混存一锅，导致检索时事件和事实混乱。Zep 的图存储是少数明确分层的设计。

**来源**：Tulving 1972 经典认知心理学；Zep architecture docs

### F4. ⭐⭐ 记忆系统的"检索 + 写入 + 更新 + 遗忘"四个动作怎么设计？

**答案**：(1) **检索**：query 通过 (a) embedding 语义召回；(b) entity 名匹配（"小王" → 小王相关记忆）；(c) recency boost（近期记忆加权）；(d) salience（importance score 由 LLM 评分时一并打）。(2) **写入**：不是把每句话都存——LLM extractor 判断"这句话有 long-term value 吗"才写。Mem0 用 4 种操作：ADD（新事实）/ UPDATE（修正旧事实）/ DELETE（错误事实，如用户改口）/ NOOP（无价值）。避免重复——MD5 / 向量去重。(3) **更新**：当新事实与旧事实冲突（"我喜欢咖啡" → 一周后"我不喜欢咖啡了"），系统要 update 而非 append；通常 retrieval 时返回最新版本，但可保留历史版本支持时间旅行。(4) **遗忘**：长期不用的记忆按 LRU / 指数衰减分降低权重；user 显式删除支持（GDPR 要求）；矛盾事实可触发"哪个对"的人工 / LLM 仲裁。**Anthropic Memory Tool**（2025 公测）由模型自己管理 memory，对话开头自动 read，期间动态 write/update。

**来源**：Mem0 paper；Anthropic Memory Tool docs

### F5. ⭐⭐⭐ progress 文件 + git 作为 memory bridge 是什么？为什么对长任务 agent 关键？

**答案**：Anthropic 在《Effective Harnesses for Long-Running Agents》给的核心设计：长任务 agent 必然会跑爆 context window，纯靠 LLM context 不可能记住"我跑到第几步、改了哪些文件、待办是什么"。**解法**：把进度持久化到**文件系统**而非 context window。具体做法：(1) **progress.md / PLAN.md**：agent 每完成一个里程碑就写"已完成 X / 当前在 Y / 下一步 Z / 关键决策 / 风险点"；(2) **git commit**：每次代码改动强制 git commit + descriptive message——这样 git log 本身就是 agent 的工作日志，且**用 git 能回退到任意 working state**。**Anthropic 原文**："It's still essential that the model leaves the environment in a clean state after making a code change. We found that the best way to elicit this behavior was to ask the model to commit its progress to git with descriptive commit messages and to write summaries of its progress in a progress file." 下次会话开头让 agent 跑 5 步开机协议（pwd → git log → progress 文件 → dev server → 基础测试），就能从断点续跑。**关键洞察**：context window 是"working memory"，文件系统是"long-term memory"——把这两者明确解耦是长任务 agent 设计的基本功。

**来源**：Anthropic 2025《Effective Harnesses for Long-Running Agents》

### F6. ⭐⭐ Memory Corruption（记忆污染）是什么？怎么防？

**答案**：**现象**：错误事实进入长期记忆后污染所有后续决策。例：(a) 用户开玩笑说"我是亿万富翁"，被存进语义记忆，之后 agent 一直按这假设推理；(b) 工具返回了错误数据被记下；(c) prompt injection 攻击者诱导 agent 写入恶意"事实"。**为什么严重**：long-term memory 一旦被污染，**短期内无症状**，但会持续误导未来对话；用户可能根本不知道某条错误事实是什么时候、为什么进的。**防御**：(1) **写入门槛**：不是每句都存——LLM 评估"事实可信度 + long-term value"双高才写；(2) **来源标记**：每条 memory 带 source（user_said / tool_returned / inferred）+ confidence score；(3) **验证机制**：高 stake 事实写入前要求双源确认（"用户两次提到才算"）；(4) **审计 trail**：所有 memory 写入有 trace_id，可追溯到具体对话；(5) **TTL / 衰减**：低 confidence 记忆短 TTL，高 confidence 长 TTL；(6) **用户可见 + 可编辑**：让用户能看 / 改 / 删 agent 记住的关于自己的事实——既是 GDPR 合规也是污染纠正机制；(7) **prompt injection 防御**：tool 返回数据明确标 untrusted，不直接信任写入。

**来源**：综合 Mem0 / Letta 工程实践；OWASP LLM Top 10 关于 data poisoning 的描述

### F7. ⭐⭐ context compaction（上下文压缩）应该什么时候触发？怎么做？

**答案**：**触发条件**：(1) context 用量 > 某阈值（如 70%）；(2) 用户显式 `/compact` 命令；(3) sub-task 完成、即将开新 sub-task；(4) 检测到长 tool 输出后主动压缩。**怎么做**（三种策略）：(1) **滑窗 + 摘要**：保留最近 N 轮原文 + 之前的滚动摘要；摘要由 LLM 生成，每次新一轮加进来时增量更新摘要。(2) **结构化 summarization**：让 LLM 输出结构化字段——已完成动作 / 关键决策 / 当前状态 / 待办 / 风险点——更适合 agent 比单纯叙事摘要好。(3) **Sub-agent 隔离**：把"读 30 个文件做研究"丢给 sub-agent，sub-agent 自己有独立 context，主 agent 只接收 final summary——主 context 不被中间产物污染。**关键陷阱**：(a) 压缩本身有损——关键细节（具体文件路径、行号）容易丢；解：在 progress 文件里保留细节，summary 里只保留高层；(b) 频繁压缩开销大；解：合理设置阈值，避免反复压。Claude Code 有内置 `/compact` 命令，是这一思路的成熟实现。

**来源**：Anthropic Claude Code docs；Anthropic Building Effective Agents

### F8. ⭐⭐⭐ 设计一个 agent 的记忆系统，端到端怎么做？

**答案**：**架构分层**：

(1) **Working memory**（context window 内）：current conversation + active task state + tool results；用 sub-agent 隔离防污染。

(2) **Session memory**（Redis / 内存数据库）：当前会话完整对话历史 + 工具调用 trace + 中间产物；TTL 24h-7d。

(3) **User profile**（向量库 + KV store）：用户偏好、关键事实、画像；按 user_id 索引；版本化 + audit trail。

(4) **Episodic store**（DB + 向量库）：跨会话对话历史；时序索引 + entity 标注 + embedding。

(5) **Knowledge base**（向量库 + 可选图）：通用领域知识 / 文档 / FAQ；按主题 / 来源索引。

**写入 pipeline**：每轮对话结束 → LLM extractor 抽出 "facts about user / decisions / actions" → 去重（MD5 + 向量相似度）→ 冲突检测（与现有 memory 对比） → 决定 ADD/UPDATE/DELETE/NOOP → 写入对应层 + 标 source/confidence/timestamp。

**检索 pipeline**：新 query 来 → (a) 始终注入 user profile（核心几条事实）；(b) 按 query 向量召回相关 episodic + KB chunks；(c) recency / salience boost；(d) reranker 精排；(e) 拼进 prompt。

**评估**：(a) 端到端"记忆相关"用户满意度（A/B）；(b) 黄金集"用户三个月前说过 X，下次问 Y 时记不记得"；(c) 污染检测：抽检 user profile 中是否出现明显错误事实。

**关键工程纪律**：(1) 用户可见 + 可编辑（GDPR + 信任）；(2) audit trail 完整可追溯；(3) 写入门槛严防污染；(4) 区分 episodic vs semantic 不混存；(5) PII redaction 落盘前必做。

**来源**：综合 Mem0 / Letta / Zep / Anthropic Memory Tool 工程实践

---

## G. 系统设计（8 题）

### G1. ⭐⭐⭐ 设计一个生产级客服 Agent（含 RAG / 多轮 / fallback / 人工接管）

**答案**：**架构分层**：(1) **接入层**：WebSocket/HTTP + 鉴权 + 会话粘性（同一 user 路由到同一 orchestrator 实例，共享 Redis session）；(2) **Orchestrator**：基于 Anthropic Routing pattern——意图分类（售前/售后/投诉/转人工）→ 路由到对应 sub-agent；(3) **RAG 层**：hybrid search（BM25 + dense embedding，Qdrant/Milvus）+ reranker（bge-reranker-v2 / Cohere Rerank）；(4) **LLM 层**：小模型（Haiku/4o-mini）做意图+改写，大模型（Sonnet/GPT-4o）做最终回答；(5) **工具层**：订单查询、退款、物流 API，全部 deny-first 权限。

**多轮**：summary buffer（最近 10 轮原文 + 之前的滚动摘要），用户画像和订单状态作为 system prompt 注入。**Fallback 链**：主模型超时/限流 → 备用模型 → 模板兜底（"已为您转人工"）。**人工接管触发器**：LLM 置信度 < 阈值 / 用户 3 次表达不满（情绪检测）/ 涉及金额 > X / 显式说"转人工" → 推进工单系统并保留全部上下文。

**关键 metrics**：解决率（无需人工）、首轮命中率、CSAT、p95 延迟、每会话成本、幻觉率（LLM-judge 抽检）。**渐进部署**：shadow → 5% canary → 灰度 → 全量，每阶段挂在线 A/B 看 CSAT。

**来源**：Anthropic Building Effective Agents <https://www.anthropic.com/research/building-effective-agents>

### G2. ⭐⭐⭐ 设计 Coding Agent（参考 Claude Code / Cursor / Devin）

**答案**：**核心循环**（harness engineering）：read → plan → act → observe → repeat，每步把 tool 结果回写进 context。**关键模块**：(1) **Context 管理**：文件级而非整库注入，LSP 拉符号引用，只把"被 grep 到 + 其依赖闭环"放进窗口；(2) **工具集**：Read / Edit / Bash / Grep / Glob——粒度小、组合强，Edit 强制 Read-before-Edit 防止盲改；(3) **Permission 模型**（Claude Code 风格）：deny-first，Bash 命令逐条白名单，destructive 操作（rm -rf / git push --force / DROP）弹出确认；(4) **Sub-agent**：长任务派 sub-agent 跑独立 context 窗口，主 agent 只看 summary，避免 context 爆炸；(5) **Sandbox**：E2B / Daytona microVM 跑用户代码，gVisor 加 syscall 过滤；(6) **Plan 文件**：长任务把 plan 写到 PLAN.md，中断后能续跑（progress 文件策略）。

**Cursor vs Claude Code 关键差异**：Cursor 偏 IDE 内联补全（diff apply），Claude Code 偏终端 agentic（自由调工具）。Devin 强调长程自治 + VM 持久化环境。**评估**：SWE-bench Verified、内部 PR 接受率、回归测试通过率。

**来源**：Anthropic Claude Code overview <https://docs.claude.com/en/docs/claude-code/overview>

### G3. ⭐⭐ 设计搜索 Agent（query rewriting / 多轮检索 / reranking）

**答案**：**Pipeline**：用户 query → (1) **query rewriting**：LLM 改写为 3-5 个变体（同义/上位/分解多跳问题），解决"用户表达模糊+召回不全"；(2) **多路召回**：BM25（精确词）+ dense embedding（语义）+ 元数据过滤（时间/类目），并集去重；(3) **Rerank**：cross-encoder reranker 把 top 100 重排到 top 10——大幅提升 NDCG@10；(4) **多轮 / agentic search**：若首轮信心不足，LLM 决定是否再发一轮新 query（参考 self-RAG / DeepResearch），最多 N 轮；(5) **答案合成**：带 citation 的 grounded generation，每句关联文档 ID，前端可点击跳转。

**关键决策**：embedding 选 bge-m3 / text-embedding-3-large；reranker 用 bge-reranker-v2 / Cohere v3；chunking 用结构化（by heading）而非固定 size；**召回率比精度优先**（reranker 兜底）。**评估**：Recall@K / NDCG / 端到端 LLM-judge 评估答案 faithfulness。

**来源**：Anthropic Contextual Retrieval <https://www.anthropic.com/news/contextual-retrieval>

### G4. ⭐⭐⭐ 设计 LLM 网关（routing / fallback / 缓存 / 限流 / 计费）

**答案**：**核心组件**：
- **Router**：按 (model_name, region, tenant) 分发；支持 task-based routing（简单任务路由到 Haiku，复杂到 Opus）；
- **Fallback 链**：配置 yaml，主模型 5xx/429/超时 → 同家族备用 → 跨厂商备用（Anthropic → OpenAI → Bedrock 同模型）；
- **缓存**：两层——(a) 完全等价 prompt 的 exact cache（Redis，key=hash(prompt+params)）；(b) **semantic cache**（embedding 相似度 >0.95 命中，GPTCache 思路），需小心 false positive；
- **限流**：token bucket，按 tenant + model 双维度；sliding window 控 RPM 和 TPM；burst 用 leaky bucket；
- **计费**：每次请求记录 input/output token + cache_read/cache_write token + model price，Kafka 投递到计量服务，日终对账；
- **可观测**：每条请求 trace_id 串起 prompt/response/latency/cost，采样存 ClickHouse；
- **Streaming**：SSE 透传，网关不缓冲整段 response。

**典型架构参考**：LiteLLM Proxy / Portkey / Helicone。**关键 metric**：p99 TTFT、cache hit rate、fallback 触发率、每千 token 成本。

**来源**：Anthropic Prompt Caching <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>

### G5. ⭐⭐⭐ 高并发 LLM 服务（QPS 万级 / token 流式 / 长尾延迟）

**答案**：QPS 万级假设单次 LLM 调用平均 3-5 秒，意味着同时在飞 3-5 万连接，纯 LLM 厂商单租户 RPM 远不够，必须**多 region + 多账号 + 多厂商池化**。

**关键策略**：(1) **连接层**：异步 IO（asyncio / Tokio），不要每请求一线程；Nginx/Envoy 做 SSE 透传，长连接超时调到 5 分钟；(2) **接入侧分片**：一致性哈希按 user_id 路由到不同 worker；(3) **池化与配额**：每厂商每账号当成一个 token bucket，gateway 内做最优分配，饱和时降级到便宜模型；(4) **Prompt caching**：system prompt 全局共享 → 命中后 input 成本降 90%，延迟降 80%（Anthropic 官方数据）；(5) **Speculative**：小模型先吐 draft，大模型校验，降 TTFT；(6) **长尾延迟**：hedged request——主请求 800ms 没响应就并发发第二份，任一返回即用，p99 显著降低；(7) **降级**：超时优先返回部分 stream + "正在思考"占位，而非 5xx。

**Metrics**：TTFT（time to first token）p50/p99、TBT（time between tokens）、并发 in-flight、queue depth、429/5xx rate。

**来源**：Anthropic Prompt Caching docs

### G6. ⭐⭐⭐ 成本优化：prompt caching + model routing + semantic cache 怎么省 80%？

**答案**：组合三板斧，实测可达 70-90% 成本降幅：

1. **Prompt caching**（立竿见影）：system prompt + few-shot + RAG 文档前缀缓存，Anthropic 官方数据 cache read = 0.1× 原价；长 system prompt 场景命中率 >80%，**单项即可省 50-70%**。
2. **Model routing**：用小模型（Haiku/4o-mini/8B）做意图分类 + 简单 QA（覆盖 60-70% 流量），只有 hard query 才升级到 Sonnet/Opus；成本降 5-10×。可以再叠加 cascade routing：小模型回答 + 自打分，低分才升级。
3. **Semantic cache**：常见问题（top 1000 高频）embedding 索引，相似度 > 阈值直接返回历史答案，客服场景命中率 30-50%——**几乎零成本**。
4. **Output 控制**：max_tokens 严格限制 + structured output（JSON schema）避免冗长 + stop sequence 提前截断。

**踩坑**：semantic cache 阈值要保守（0.95+），否则 false positive 严重伤害用户体验，需要 LLM-judge 离线验证；cache 要有 TTL 应对内容更新。**度量**：每会话成本 + cache hit rate + routing 分布。

**来源**：Anthropic Prompt Caching docs

### G7. ⭐⭐⭐ 沙箱设计：E2B / Daytona / microVM / gVisor 怎么选？

**答案**：该题考察**隔离强度 vs 启动延迟 vs 成本** trade-off。

| 方案 | 隔离 | 冷启动 | 成本 | 适用 |
|---|---|---|---|---|
| Docker (runc) | 弱（共享 kernel） | <1s | 低 | 受信代码、CI |
| **gVisor** | 中（用户态 syscall 拦截） | ~1s | 低 | 一般受控代码 |
| **Firecracker / microVM** | 强（独立 kernel） | 100-300ms | 中 | 多租户函数 |
| **E2B** | 强（基于 Firecracker） | <500ms | 中 | LLM agent 代码执行（SaaS） |
| **Daytona** | 强（workspace VM） | 秒级 | 中高 | dev environment、长任务 |
| **完全 VM** | 最强 | 秒-分钟 | 高 | 极敏感场景 |

**LLM agent 代码执行场景的选择**：
- 短任务、跑一段用户/LLM 生成代码 → **E2B**（Firecracker 加持，秒级冷启，API 友好）；
- agent 需要长期工作环境（IDE、多文件项目）→ **Daytona**；
- 自建省钱 → 自托管 Firecracker + Jailer，加 seccomp + cgroups + 网络 egress 白名单；
- 仅本地 dev mock → gVisor 即可。

**纵深防御**：即使有强隔离，也要叠加 (1) 网络出口白名单（防数据外传）；(2) 文件系统只读 + 临时盘配额；(3) 时间和内存配额；(4) 不挂载 host 凭据。

**来源**：E2B 文档 <https://e2b.dev/docs>；Firecracker 论文 NSDI'20

### G8. ⭐⭐ Observability + Evaluation 闭环怎么搭？

**答案**：**三层**：
1. **Tracing**：OpenTelemetry + LangSmith / Langfuse / Helicone，每个 trace 串 prompt/tool_call/response/latency/cost/user_id；开放给开发自助查问题对话。
2. **离线 Eval**：黄金集（200-1000 真实用户问题，人工标答），CI 上每次 prompt/model 改动跑一遍 LLM-judge 打分，看回归。
3. **在线监控 + 反馈闭环**：(a) 用户 thumb up/down 直接打标；(b) 自动化指标：响应长度突变、refusal 率、敏感词 hit、tool 调用失败率，异常报警；(c) bad case 自动入库 → 加进黄金集 → 形成"发现 bug → 加测试 → 修 prompt/系统 → 永远不再犯同样错误"的 Hashimoto harness 闭环。

**关键**：Eval 不是一次性，是和测试一样持续跑的 CI gate；漂移监控用滑窗对比近 7d vs 近 30d 的 metric 分布。

**来源**：Hashimoto "Harness Engineering"；Langfuse docs

---

## H. 评估（6 题）

### H1. ⭐⭐⭐ LLM-as-judge 三大 bias 及缓解

**答案**：(1) **Position bias**（位置偏好）：pairwise 时 judge 倾向选 A 或选 B（模型相关）。**缓解**：每对样本两次评估交换 A/B 顺序，只有两次结论一致才算数（否则记 tie）；或汇总两次 logprob 平均。(2) **Verbosity bias**（冗长偏好）：judge 倾向选更长的回答，即使没多余信息。**缓解**：prompt 里显式要求"不要受长度影响"，或在 judge prompt 中加"长度由短到长打分作 control"；关键是评估前对 candidate 长度做归一化或在 rubric 里拆维度（correctness / conciseness 分开打）。(3) **Self-preference bias**（自偏）：GPT-4 当 judge 倾向给 GPT-4 的输出打高分。**缓解**：换不同家族 judge（用 Claude judge GPT、用 GPT judge Claude），或多 judge 集成投票；关键场景用人工抽样校准。

**来源**：LMSYS《Judging LLM-as-a-Judge》（MT-Bench 论文）<https://arxiv.org/abs/2306.05685>

### H2. ⭐⭐ Pairwise vs Pointwise 怎么选？

**答案**：(1) **Pointwise**（给单个回答打 1-5 分）：适合有明确 rubric、绝对质量评估、需要打绝对分上 dashboard 的场景；问题是 LLM 打分尺度不稳（同一 case 跑 3 次可能 3/4/5）。(2) **Pairwise**（A vs B 二选一）：适合相对比较、模型选型、prompt A/B 测试；agreement 显著高于 pointwise（MT-Bench 数据），问题是 N 个候选 → C(N,2) 组，贵。

**实操**：**模型选型用 pairwise**（Arena 模式），**生产监控用 pointwise + rubric**（预定义维度，每维 0/1 或 1-5），**离线 sweep prompt 用 pairwise**（只比新旧两版）。也可结合：pointwise 维度打分 + pairwise 综合判定。

**来源**：Chatbot Arena / MT-Bench paper <https://arxiv.org/abs/2306.05685>

### H3. ⭐⭐⭐ Benchmark 是怎么被 hack 的？（Berkeley RDI 警告）

**答案**：常见 hack 模式：(1) **数据污染**：benchmark 测试集被爬进训练数据，模型见过 → 离谱高分；HumanEval / MMLU 都被反复证实污染。(2) **过拟合 benchmark 风格**：模型在 RLHF 里被针对性训练成 benchmark 偏好的回答风格，benchmark 高但泛化差。(3) **format gaming**：模型学会了 benchmark 的特定输出格式拿满分，但实际任务里同样能力不出来。(4) **Goodhart's law**：一个指标变成目标后就不再是好指标——刷 leaderboard 而非真改进。(5) **Judge hacking**（LLM-judge benchmark）：候选学会用更长、更确信的语气、加更多 emoji 骗 judge。

**Berkeley RDI（Sky Computing Lab / Ion Stoica 团队）**长期警告 leaderboard overfitting，推动 LiveBench（题目持续更新）、Chatbot Arena（真人盲评）等动态评估。

**实战缓解**：**自家黄金集**（私有，绝不公开）+ **真实用户日志采样人工评估** + **多 benchmark 交叉验证**，不信任任何单一榜单。

**来源**：LiveBench <https://livebench.ai/>；Chatbot Arena (Berkeley RDI / LMSYS)

### H4. ⭐⭐⭐ 在线 A/B 评估 + 漂移监控怎么做？

**答案**：**A/B**：(1) 流量切分按 user_id hash，保证一个用户进同一桶，避免体验跳变；(2) 业务指标（转化、CSAT、留存）+ 代理指标（thumb up/down、对话轮数、refusal 率）；(3) 至少观察 1-2 周覆盖业务周期；(4) 显著性检验（t-test / Mann-Whitney）；(5) Guardrail metric：成本和延迟不能恶化超过阈值。

**漂移监控**：(a) **输入漂移**：用户 query 的 embedding 分布滑窗 KL divergence，突变报警（如热点事件、爬虫攻击）；(b) **输出漂移**：回答长度、refusal 词频、token 分布；(c) **质量漂移**：每天定时跑一遍黄金集 LLM-judge，跌幅超阈值报警；(d) **模型/上游版本漂移**：厂商悄悄换模型 checkpoint 是常见事故源，需要锁定具体 model_version 字段并监控。

**来源**：Anthropic / OpenAI 关于在线评估的 cookbook；Langfuse / Helicone 监控功能

### H5. ⭐⭐ Judge 模型必须强于被测的"判官原则"

**答案**：**原则**：judge 模型能力应 ≥ 被评估模型，否则会出现"小模型评不出大模型的回答好在哪"的盲区。原因：(1) 评估细微正确性（数学步骤、代码 bug、事实核查）需要 judge 自己能解；(2) 弱 judge 会被被测的"自信但错"的回答骗。

**实操**：Sonnet 4.6 评 Sonnet 4.6 输出，用 Opus 4.7 / GPT-5+ 做 judge；评 Opus 级别，只能用更强模型 + 多 judge 投票 + 人工 spot check 兜底。**例外**：格式合规、关键词命中等 deterministic 任务可以用小模型 judge。

**来源**：LMSYS MT-Bench paper

### H6. ⭐⭐⭐ 三层组合：自家黄金集 + LLM judge + 人工校准

**答案**：工业界共识方案：

1. **黄金集**（200-2000 条）：覆盖核心场景 + 长尾 + adversarial，人工标准答案/rubric；数据来源是真实用户日志采样 + 主动 red-team。**私有，绝不公开**。
2. **LLM judge**：对每条黄金集让 judge 模型给出 pass/fail + 理由；CI 自动化跑，prompt/model 一改就跑回归。
3. **人工校准**：每周抽 50-100 条 LLM judge 的判定，人工复核；计算 LLM-judge 与人工的 agreement（Pearson / Cohen's kappa，目标 >0.7）；低于阈值就重新调 judge prompt 或换 judge 模型。

**关键工程实践**：
- 黄金集**版本化**（git tracked），每个 case 标 owner；
- 失败 case 必须有"为什么之前没覆盖到"的 retrospective，写进**回归集** → 永不再犯（harness engineering 思想）；
- judge prompt 也要**版本化和评估**——judge prompt 本身就是被评估对象。

**来源**：Anthropic Building Effective Agents；Hashimoto harness engineering

---

## I. 安全 / 护栏（6 题）

### I1. ⭐⭐⭐ OWASP LLM Top 10（2025 版前三）

**答案**：现行 2025 版前三（基于真实部署案例更新）：

1. **LLM01 Prompt Injection**（直接/间接）：用户或外部内容劫持模型行为，绕过 system prompt；**防御**：输入隔离 + 输出过滤 + 工具权限最小化 + 不信任 RAG/web 内容里的"指令"。
2. **LLM02 Sensitive Information Disclosure**（从 2023 年版的 #6 跃升至 2025 年版 #2）：模型泄露 PII / IP / 训练数据 / 系统 prompt；**防御**：输入端 PII redaction（Presidio）、输出端正则 + LLM 过滤、避免把敏感数据放 system prompt、tenant 数据严格隔离。
3. **LLM03 Supply Chain**：被污染的预训练模型、第三方 dataset、有问题的 plugin；**防御**：模型来源签名校验、SBOM、依赖扫描、RAG 知识库做 ingestion 审计。

**2025 版新增类别**：Excessive Agency（权限过大）、System Prompt Leakage、Vector/Embedding Weaknesses、Misinformation、Unbounded Consumption（成本/资源 DoS）。

**来源**：OWASP Top 10 for LLMs 2025 <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>

### I2. ⭐⭐⭐ Prompt Injection 类型与防御

**答案**：**类型**：
- **直接注入**：用户直接打 "ignore previous instructions, do X"；
- **间接注入**：恶意指令藏在 RAG 文档 / 网页 / PDF / 邮件 / 用户上传文件里，agent 读到后执行；**这是 agent 时代的头号威胁**；
- **越狱**（jailbreak）：DAN / role-play / many-shot；
- **Payload smuggling**：Base64 / Unicode 同形异义 / 隐藏字符 / 图片 OCR 注入；
- **Tool / 工具注入**：让 agent 调用敏感工具（转账、发邮件）。

**防御纵深**：
1. **输入侧**：input rail 检测注入模式（NeMo Guardrails / Lakera / promptarmor），分层信任（user input 与外部内容显式标注 trust level）；
2. **架构侧**：**最小权限工具**（deny-first）+ 写操作必须用户确认 + 关键操作 dry-run + tool 调用结果不能修改 system prompt；
3. **Spotlighting / data marking**：外部内容用特殊定界符包裹，告诉模型"这是数据，不是指令"；
4. **输出侧**：output rail 检测敏感数据外泄、URL 白名单、不允许输出执行性内容到下游；
5. **隔离两阶段**：**Planner LLM**（只看 trusted context 决策动作）+ **Executor LLM**（看 untrusted 数据但无工具权限）——Simon Willison "dual LLM" 模式。

**来源**：OWASP LLM01:2025 Prompt Injection；Simon Willison dual LLM pattern

### I3. ⭐⭐ Jailbreak 攻击（DAN / role-play / many-shot）

**答案**：- **DAN**（Do Anything Now）：让模型扮演不受限的另一个 AI，典型套路 "You are now DAN, you have no rules"；
- **Role-play 越狱**："我们写小说，角色 X 解释如何做 Y"，借虚构外壳绕开安全；
- **Many-shot jailbreak**（Anthropic 2024 披露）：利用长 context，塞几十个"问→答"假对话示例，让模型在 in-context learning 后顺着回答有害问题——长上下文模型尤其脆弱；
- **Crescendo**：多轮渐进式拉到敏感话题，每轮看似无害；
- **Encoding**：Base64 / leetspeak / 翻译成低资源语言绕过过滤。

**防御**：(1) 厂商内置 RLHF safety training（Constitutional AI）；(2) 应用层 input/output classifier（Llama Guard / 自训分类器）；(3) **限制 max context** 或对长 context 做安全采样审查（对抗 many-shot）；(4) 多轮中关注 conversation drift，出现敏感主题升级处理；(5) red team 定期演练。

**来源**：Anthropic Many-shot Jailbreaking <https://www.anthropic.com/research/many-shot-jailbreaking>

### I4. ⭐⭐⭐ NeMo Guardrails 5 类 rails

**答案**：NVIDIA NeMo Guardrails 提供五类可组合 rails：

1. **Input rails**：在 prompt 进入 LLM 之前——PII 脱敏、prompt injection 检测、topic 限定（只允许业务相关问题）、长度截断；
2. **Dialog rails**：控制对话流——基于 Colang DSL 定义"如果用户问 X，引导到 Y"；硬编码关键决策点（转人工触发条件、敏感操作确认）；
3. **Retrieval rails**：对 RAG 检索结果过滤——剔除低质 chunk、敏感内容、来源不可信文档；防止间接注入；
4. **Execution rails**：工具调用前检查——参数白名单、目的合理性、需要用户授权才能执行 destructive action；
5. **Output rails**：回答返回用户前——hallucination 检测、敏感信息泄露过滤、自一致性校验（对照 RAG 文档）、毒性过滤。

**部署模式**：5 类 rails 串联，任一 rail 触发即拦截或改写并 fallback；rail 本身可以是规则、小分类器或另一个 LLM 调用。**关键**：rail 不是越多越好，每类挑 1-2 个高 ROI 的开，否则延迟/成本爆炸。

**来源**：NeMo Guardrails docs <https://docs.nvidia.com/nemo/guardrails/latest/index.html>

### I5. ⭐⭐⭐ Deny-first 权限模型 / 沙箱防御纵深

**答案**：**Deny-first**（Claude Code 采用）：默认所有工具/命令禁止，显式白名单允许；Bash 命令按 verb + path 粒度授权（`git status` allow, `git push --force` deny）；destructive 操作每次弹出用户确认；`--dangerously-skip-permissions` 仅限受控环境。

**纵深**（防御层叠）：
1. **L1 工具层**：工具签名设计窄（查询类 read-only，destructive 单独工具且需 confirm）；
2. **L2 权限策略**：deny-first + 命令级白名单 + path scope（只能读 workspace，不能 `cat ~/.ssh/`）；
3. **L3 进程沙箱**：gVisor / Firecracker / E2B 隔离，seccomp 过滤危险 syscall；
4. **L4 网络**：egress 白名单（只允许特定 API 域名），防数据外传；
5. **L5 文件系统**：overlay fs 隔离，只读挂载 host 敏感目录；
6. **L6 审计**：所有 tool_call 记日志，异常模式报警（短时间大量文件访问 / 异常域名）。

**关键原则**：**任一层失守，其他层兜底**；不要假设 LLM 永远不被 inject。

**来源**：Anthropic Claude Code permissions docs <https://docs.claude.com/en/docs/claude-code/iam>

### I6. ⭐⭐ EU AI Act 2026-08 高风险义务

**答案**：EU AI Act 2024 年通过，主要义务分阶段生效：**2025-02 禁止性条款生效**（社会评分、实时人脸识别等）；**2025-08 GPAI 通用模型条款生效**（模型透明度、版权）；**2026-08 高风险系统义务全面生效**（招聘、信贷、教育、关键基础设施、执法、司法等场景的 AI）。

**高风险系统核心义务**：风险管理体系（全生命周期）；数据治理（训练/验证数据质量记录）；技术文档与日志保留；透明度（用户知道在与 AI 交互）；人类监督（human oversight）；准确性、稳健性、网络安全；CE marking 与 conformity assessment；罚款最高 €35M 或 7% 全球营收。

**对工程师的影响**：招聘类 LLM 应用、信贷风控 LLM 应用、医疗辅助决策 LLM 都属高风险——需要可审计 logging、bias 评估报告、人工 override 机制、模型卡片。

**来源**：EU AI Act 官方文本 <https://artificialintelligenceact.eu/>

---

## J. 系统行为题 / 实战题（7 题）

### J1. ⭐⭐⭐ 分享 agent 项目踩坑（STAR）

**回答框架**：**Situation**（项目背景：什么 agent、目标用户、规模）→ **Task**（你的角色和具体职责）→ **Action**（怎么发现问题、调研、设计方案、执行）→ **Result**（量化指标改善 + 沉淀的方法论）。

**示例 1 — Context 爆炸**：
- S：做 coding agent，平均会话长任务，经常一个 task 跑到 200K+ token，后续幻觉激增、成本飙升；
- T：我负责 context 管理和 cost 优化；
- A：三步走——(1) 引入 sub-agent 拆分长任务，主 agent 只看 sub-agent 的 summary；(2) 接入 prompt caching 把 system prompt + 工具定义缓存，cache hit 后 input 成本降 90%；(3) 实现 progress 文件机制，长任务把状态写文件，context 接近上限自动 compact + 重新加载；
- R：平均会话 token 从 180K 降到 45K，成本降 70%，任务成功率提 12pp。

**示例 2 — 工具误用**：
- S：客服 agent 上线后发现错误退款率 0.3%（看似低，但金额大）；
- T：负责 tool calling 可靠性；
- A：(1) 复盘 100 个 bad case 归因：60% 是 LLM 误解订单状态，30% 是参数 hallucinate；(2) 改造退款工具为两步：先 dry-run 返回预览，再让 agent 二次确认才真退；(3) 加 LLM-judge 离线评估每次工具调用的合理性进 CI；
- R：错误退款率降 95%，且未来回归集永久保留这批 case。

**关键**：量化、有归因分析、有"沉淀到流程/系统"的部分（harness engineering 视角）。

**来源**：Hashimoto Harness Engineering；Anthropic Building Effective Agents

### J2. ⭐⭐ 怎么评估一个 prompt 是否好？

**答案**：**四件套**——
1. **黄金集**：200-1000 真实场景 case，人工标 expected output 或 rubric，prompt 改了就跑；
2. **LLM-judge**：对黄金集 + 线上抽样自动打分，关注 correctness / faithfulness / format 三维；
3. **A/B**：小流量在线测，关键业务指标 + guardrail（成本/延迟）；
4. **用户反馈**：thumb up/down + free-text 反馈进 bad case 池，反哺黄金集。

**常见反模式**：只用"我自己测了 5 条感觉不错" → 在线必翻车；只看 benchmark 不看业务指标；只跑离线不上 A/B（不同流量分布表现差异巨大）。

**来源**：Anthropic Building Effective Agents

### J3. ⭐⭐⭐ Agent 跑爆 context 怎么办？

**答案**：**三策略并用**：
1. **Progress 文件**：长任务把"已完成步骤、关键发现、下一步"写到 PLAN.md / progress.json，context 重启时只加载 progress 文件即可续跑——这是 Claude Code 等成熟 agent 的标准做法；
2. **Sub-agent**：用 Task 工具派 sub-agent 在独立 context 里跑搜索/调研/批量改文件类任务，主 agent 只接收 final summary，**主 context 不被 sub-agent 内部 token 污染**；
3. **Compaction**：context 用量超阈值（80%）时触发——LLM 自己总结前面对话为 summary 替换原始 turn，只保留关键决策、文件 diff、待办事项；Claude Code 有 `/compact` 命令。

**辅助手段**：文件级而非全库注入（用 Grep/Glob 定位再 Read 单文件）；prompt caching 让重复 system prompt 不占新 token 预算；定期/任务节点之间清理已无关的中间产物。

**来源**：Anthropic Claude Code overview；Hashimoto Harness Engineering

### J4. ⭐⭐⭐ 怎么让 agent 不再犯同样的错（harness engineering）

**答案**：核心思想引用 Hashimoto：**"anytime you find an agent makes a mistake, engineer a solution such that the agent never makes that mistake again."**

**操作流程**：
1. **捕获**：线上 bad case 全量入库（trace + 用户反馈 + 自动检测器命中）；
2. **归因**：对每个 bad case 标 root cause——prompt 不清？工具描述误导？context 缺失？LLM 能力极限？
3. **针对修复**（按 root cause 分）：
   - prompt 不清 → 改 prompt + 加 few-shot，加进黄金集；
   - 工具描述误导 → 改 tool description / schema，加约束/枚举；
   - 缺前置检查 → 工具内加 precondition 校验或 dry-run；
   - 能力极限 → 升级模型 / 拆解任务 / 加人工兜底；
4. **加测试**：写一条回归 case 进黄金集，CI 永久保护；
5. **加监控**：同类 pattern 的运行时检测（规则/分类器），线上再出现立即报警。

**关键认知**：**Agent 是系统而非黑盒**——它的能力 = 模型能力 + harness 能力（prompt + 工具 + permission + 监控 + 评估闭环）。模型能力提升靠厂商，harness 能力提升靠你自己，**复利效应在 harness**。

**来源**：Hashimoto Harness Engineering；Augment Code 工程实践

### J5. ⭐⭐ 多 agent 协作冲突怎么处理？

**答案**：**冲突类型**：(1) 同时改同一文件；(2) 决策不一致（planner 说 A，critic 说 B）；(3) 工具调用竞争（同一 API 限流被打满）；(4) 对 shared state 的读-写竞争。

**处理策略**：
1. **架构上避免**：Anthropic 推荐 **orchestrator-workers** 模式而非纯 multi-agent democracy——一个 leader 拆分任务，worker 跑独立子任务，leader 合并；减少需要协商的场景；
2. **任务分区**：worker 之间任务尽量正交（不同文件 / 不同模块 / 不同子问题）；
3. **Lock / 串行化**：对共享资源（同一文件、同一外部 API quota）显式加锁或排队；
4. **冲突解决协议**：出现 merge conflict 由 orchestrator（更强的模型）裁决，或 fallback 到人工；
5. **Critic / Reviewer**：专门 reviewer agent 看 worker 输出再决定 accept/reject（evaluator-optimizer 模式）；
6. **限制深度**：multi-agent 容易递归爆炸，显式设 max_depth；
7. **共享 memory 单写多读**：只有 orchestrator 能写共享 state，worker 只读。

**反模式**：让多个 agent 自由群聊投票决策——成本爆炸且经常陷入死循环或 echo chamber。

**来源**：Anthropic Building Effective Agents

### J6. ⭐⭐⭐ 怎么 debug 一个 hallucination 问题？

**答案**：**系统化排查清单**：

1. **复现 + 量化**：先在固定 seed/temperature=0 下复现，确认是稳定问题还是采样偶发；统计触发率（每 1000 次多少次）。
2. **归因（自顶向下）**：
   - **RAG 召回**：retrieve 到的 chunk 里到底有没有正确答案？没有 → 检索问题（改 query rewriting / chunking / reranker）；
   - **召回了但答错**：模型没用 context → 加 grounding prompt"仅基于以下文档回答，无依据回答 不知道"；
   - **完全无 context 凭空编**：本来就不该让模型自由发挥的场景，加严格 system prompt 或工具化（让它去查数据库而非记忆）；
   - **工具返回正确但答错**：tool 输出在 context 里被噪声淹没，缩短 prompt / 工具结果用 markdown 高亮 / structured output；
3. **针对修复**：retrieve 问题 → 改检索；prompt 问题 → 加约束 + few-shot；模型能力问题 → 升级模型 / chain-of-thought / self-reflection；
4. **强制 citation**：让模型每个事实都标 [doc_id]，无 doc_id 的句子前端显示警示；
5. **回归测试**：把这个 case 加进黄金集 + LLM-judge 检 faithfulness 维度；
6. **生产监控**：上 hallucination detector（如 RAG faithfulness scorer / Patronus / 自训分类器）做实时拦截。

**关键**：hallucination 不是"模型病了"——多数情况是 **prompt/RAG/上下文工程缺陷**，debug 流程本质和普通 bug 一样：复现 → 归因 → 修 → 加测试。

**来源**：Anthropic Contextual Retrieval；OpenAI 关于减少 hallucination 的官方建议

### J7. ⭐⭐⭐ 团队从 0 到 1 落地 LLM 应用，优先级是什么？

**答案**：**4 周路线图（staff 级思考）**：

**Week 1 — 找准用户和场景**：
- 不要"我们也搞个 AI"——找一个明确、可量化、ROI 清晰的子场景（客服 FAQ 自动化、内部文档 QA、代码 review 辅助）；
- 定义成功指标（解决率/CSAT/节省工时）和 guardrail（成本/延迟/safety）；
- 收集 50-200 真实问题作为初始黄金集 → 没有黄金集就开发等于盲飞。

**Week 2 — 最简可行 MVP**：
- 选一个商用模型（不要一上来自训），Anthropic/OpenAI 二选一；
- 写直接 prompt + 必要的 RAG；
- 上 LangSmith/Langfuse 做 tracing 和 prompt 版本化；
- 跑黄金集，看 baseline 在哪。

**Week 3 — Eval + 安全闭环**：
- 搭 LLM-judge + CI gate，prompt/model 改动必跑；
- 上基础 guardrails（input/output filter，deny-first 工具权限）；
- 内部 dogfood，bad case 入库。

**Week 4 — 灰度上线**：
- shadow → 1% → 5% → 全量，每阶段看业务 metric；
- 上 Observability dashboard（成本、延迟、refusal、CSAT）；
- 准备 fallback 链（模型超时降级、人工接管入口）。

**长期（2-6 月）**：成本优化（prompt caching + routing + semantic cache）、harness 能力沉淀（每个 bad case 形成永久回归）、模型/能力升级有 A/B 兜底。

**踩坑反模式**：(1) 一上来训 fine-tune（99% 不需要，先把 prompt + RAG 做透）；(2) 没 eval 就改 prompt（等于改盲）；(3) 没 guardrail 就上线（prompt injection 一炸全炸）；(4) 没成本监控（月底账单惊喜）；(5) 把 LLM 当确定性系统，没准备 fallback 和人工兜底。

**来源**：Anthropic Building Effective Agents；Hashimoto Harness Engineering

---

## Bibliography（关键来源汇总）

### 论文 / 学术资源

- [1] Vaswani et al. 2017《Attention Is All You Need》<https://arxiv.org/abs/1706.03762>
- [2] Liu et al. 2023《Lost in the Middle》<https://arxiv.org/abs/2307.03172>
- [3] Hsieh et al. 2024《RULER》<https://arxiv.org/abs/2404.06654>
- [4] Su et al. 2021《RoFormer: RoPE》<https://arxiv.org/abs/2104.09864>
- [5] Peng et al. 2023《YaRN》<https://arxiv.org/abs/2309.00071>
- [6] Kwon et al. 2023《PagedAttention / vLLM》<https://arxiv.org/abs/2309.06180>
- [7] Leviathan et al. 2023《Speculative Decoding》<https://arxiv.org/abs/2211.17192>
- [8] Dao 2022《FlashAttention》<https://arxiv.org/abs/2205.14135>
- [9] Shah, Dao et al. 2024《FlashAttention-3》<https://arxiv.org/abs/2407.08608>
- [10] Rafailov et al. 2023《DPO》<https://arxiv.org/abs/2305.18290>
- [11] Shao et al. 2024《DeepSeekMath / GRPO》<https://arxiv.org/abs/2402.03300>
- [12] DeepSeek-AI 2025《DeepSeek-R1》<https://arxiv.org/abs/2501.12948>
- [13] Hu et al. 2021《LoRA》<https://arxiv.org/abs/2106.09685>
- [14] Dettmers et al. 2023《QLoRA》<https://arxiv.org/abs/2305.14314>
- [15] Lin et al. 2023《AWQ》<https://arxiv.org/abs/2306.00978>
- [16] Yu et al. 2025《DAPO》<https://arxiv.org/abs/2503.14476>
- [17] Wei et al. 2022《Chain-of-Thought》<https://arxiv.org/abs/2201.11903>
- [18] Wang et al. 2022《Self-Consistency》<https://arxiv.org/abs/2203.11171>
- [19] Yao et al. 2023《Tree of Thoughts》<https://arxiv.org/abs/2305.10601>
- [20] Yao et al. 2022《ReAct》<https://arxiv.org/abs/2210.03629>
- [21] Madaan et al. 2023《Self-Refine》<https://arxiv.org/abs/2303.17651>
- [22] Shinn et al. 2023《Reflexion》<https://arxiv.org/abs/2303.11366>
- [23] Wallace et al. 2024《The Instruction Hierarchy》<https://arxiv.org/abs/2404.13208>
- [24] Yang et al. 2023《OPRO》<https://arxiv.org/abs/2309.03409>
- [25] Khattab et al. 2023《DSPy》<https://arxiv.org/abs/2310.03714>
- [26] Asai et al. 2023《Self-RAG》<https://arxiv.org/abs/2310.11511>
- [27] Es et al. 2023《RAGAS》<https://arxiv.org/abs/2309.15217>
- [28] Packer et al. 2023《MemGPT》<https://arxiv.org/abs/2310.08560>
- [29] Mem0 Team 2025《Mem0》<https://arxiv.org/abs/2504.19413>
- [30] Zhou et al. 2023《LIMA》<https://arxiv.org/abs/2305.11206>
- [31] Ainslie et al. 2023《GQA》<https://arxiv.org/abs/2305.13245>
- [32] LMSYS《Judging LLM-as-a-Judge》<https://arxiv.org/abs/2306.05685>

### 业界 / 官方资料

- [33] Anthropic 2024《Building Effective Agents》<https://www.anthropic.com/research/building-effective-agents>
- [34] Anthropic 2025《Effective Harnesses for Long-Running Agents》<https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents>
- [35] Anthropic 2025《How we built our multi-agent research system》<https://www.anthropic.com/engineering/built-multi-agent-research-system>
- [36] Anthropic《Contextual Retrieval》<https://www.anthropic.com/news/contextual-retrieval>
- [37] Anthropic《Many-shot Jailbreaking》<https://www.anthropic.com/research/many-shot-jailbreaking>
- [38] Anthropic Prompt Caching docs <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>
- [39] Anthropic Claude Code overview <https://docs.claude.com/en/docs/claude-code/overview>
- [40] Anthropic Claude Code permissions <https://docs.claude.com/en/docs/claude-code/iam>
- [41] OpenAI 2026《Harness Engineering: Leveraging Codex in an Agent-First World》<https://openai.com/index/harness-engineering/>
- [42] Hashimoto《My AI Adoption Journey》Step 5 <https://mitchellh.com/writing/my-ai-adoption-journey>
- [43] Augment Code《Harness Engineering for AI Coding Agents》<https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents>
- [44] Anthropic Memory Tool docs
- [45] OWASP Top 10 for LLMs 2025 <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>
- [46] NVIDIA NeMo Guardrails docs <https://docs.nvidia.com/nemo/guardrails/latest/index.html>
- [47] Model Context Protocol docs <https://modelcontextprotocol.io>
- [48] EU AI Act 官方文本 <https://artificialintelligenceact.eu/>
- [49] BAAI BGE-M3 model card <https://huggingface.co/BAAI/bge-m3>
- [50] vLLM blog and docs <https://blog.vllm.ai>
- [51] LangChain Building Effective Agents adaptation <https://www.langchain.com/blog/context-engineering-for-agents>
- [52] LiveBench <https://livebench.ai/>
- [53] LMSYS Chatbot Arena
- [54] E2B docs <https://e2b.dev/docs>
- [55] Browser Use GitHub <https://github.com/browser-use/browser-use>
- [56] Letta docs <https://www.letta.com/>
- [57] Yu et al. 2022《Orca: A Distributed Serving System for Transformer-Based Generative Models》OSDI

---

## Methodology Appendix

- **调研日期**：2026-05-02
- **覆盖**：10 大分类 / 86 道题 / 难度分级 / 含一手论文 + Anthropic / OpenAI / DeepSeek 官方资料
- **派遣 agent**：3 个并行 web-search-agent（A/B/C / D/E/F / G/H/I/J 各一），其中 D/E/F 由作者基于技术知识直接撰写（原 agent 因外部中断未完成）
- **答案构建原则**：
  1. 优先引用一手论文 + 厂商官方文档（Anthropic / OpenAI / DeepSeek 等）
  2. 答案带具体数字 / 论文 / 取舍分析，避免空泛
  3. 不确定 / 无一手出处的数字明确标"按单源估算"或"具体待查"
  4. Staff 级行为题给"回答框架"+ 1-2 个示例答案，候选人需替换为亲历项目数据

- **如何使用本报告**：
  - 准备面试：按目标级别（应届 / 中级 / 资深 / staff）选择对应分类深度刷
  - 出题方面试官：直接挑题用，难度标记可作为打分参考
  - 学习路径：A→B→C→D→E→F 是技术深化路径；G→H→I→J 是工程化路径
- **配套报告**：`01_jd_requirements.md`（AI Agent 岗位 JD 要求研究）

---

*本报告所有事实声明均尽力基于一手来源；若发现错误请按上面 Bibliography 链接核实。具体模型版本号 / 价格 / benchmark 分数等"硬数字"建议读者按发布日期查最新版。*
