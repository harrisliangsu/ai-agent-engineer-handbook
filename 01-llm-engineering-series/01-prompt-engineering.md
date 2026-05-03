# 提示词工程：含 Transformer 与 LLM 训练 / 推理基础

**位置**：LLM 应用工程四层演进的第一层（最底层基础）
**调研日期**：2026-05-01

---

## Executive Summary

提示词工程（Prompt Engineering）是 2022-2023 年的行业焦点，但要真正讲清它，必须从 **Transformer 架构（2017）→ LLM 训练栈 → 推理与解码 → 提示词技术**这条完整链条说起。原因是：当代提示词技术（CoT 触发推理、ReAct 触发工具调用、模型偏好 XML vs JSON）的"为什么有效"全部根植于底层架构与训练方式。

本章覆盖：

- **§1.1** Transformer 架构 9 年演进——从 Vaswani 2017 的 dense attention 到 2026 年 Flash Attention 4（NVIDIA B200 上 1605 TFLOPs/s、71% 利用率）
- **§1.2** LLM 训练栈三段式 = **预训练 + 后训练 + 微调**。后训练已从单一 RLHF 演进到模块化的 SFT → 偏好对齐（DPO/SimPO/KTO）→ 推理强化（GRPO/DAPO/RLVR）。微调按显存预算分四档：Full FT（$50K）→ LoRA（$10K）→ QLoRA（$1.5K，4090 卡可跑 7B）→ 蒸馏
- **§1.3** 推理栈与解码策略——温度、top-p、grammar-guided 等
- **§1.4** 提示词工程经典技术——Few-shot / CoT（GSM8K 17.9%→56.9%；叠加 self-consistency 后 ~74%）/ Self-Consistency / ReAct（ALFWorld +34%）/ ToT / Self-Refine（NeurIPS 2023, +20%）/ Reflexion / 角色 prompt
- **§1.5** 自动化提示词——APE / OPRO / Promptbreeder / **DSPy**（2025-2026 进入主流，让 prompt 成为编译产物）
- **§1.6** 评估（Promptfoo）、失败模式（OWASP LLM Top 10）、以及为何"提示词工程的黄昏"在 2024 下半年开始被讨论

关键转折：**2024-2025 年起，最值钱的能力从"写好这一句"迁移到"管好整个 context"**——这是后续 02 上下文工程的开端。

---

## 1.1 起点：Transformer 与注意力机制

2017 年 Vaswani 等人在 *Attention Is All You Need* [1] 中提出 Transformer 架构，把循环（RNN）和卷积（CNN）从序列建模里拿掉，只留下"注意力"。其核心思想极为朴素：序列里每个 token 与所有其他 token 通过 Query / Key / Value 三个矩阵做加权聚合，权重由 softmax(QK^T / √d) 决定。这种设计的两个直接结果是——**全局可见性**（任何位置都能直接看到任何位置，无需逐步传递）和**完全并行**（不再像 RNN 那样必须按时间步前向）。

### 1.1.1 Attention 完整公式与 Q / K / V 的物理含义

标准的 **Scaled Dot-Product Attention** 数学定义：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 Q、K、V 是同一个输入序列 X（形状 `[seq_len, d_model]`）经过三个独立的线性变换得到：

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

`W^Q`、`W^K`、`W^V` 都是可学习参数（形状 `[d_model, d_k]`，d_k 通常 = d_model / num_heads）。

理解 Q/K/V 最常用的类比是**"图书馆检索"**：

- **Query（查询）**：我想找的内容（一本"关于 Transformer 的书"）
- **Key（键）**：每本书的"标签 / 主题"（书脊上的关键词）
- **Value（值）**：书的实际内容（你真正读到的）

整个过程：用 Q 去"叩问"所有 K，根据匹配度（点积）算出每本书的"相关度分数"，softmax 归一化成权重，再用权重去加权所有 V，得到"我应该融合哪些内容"。

公式中的几个关键设计：

- **`QK^T` 矩阵乘**：算所有 query-key 对的点积，结果是 `[seq_len, seq_len]` 的"相关度矩阵"，每行 i 是 token i 对所有其他 token 的相关度
- **`/√d_k` 缩放**：防止 d_k 大时点积值过大、softmax 进入饱和区导致梯度消失。Vaswani 论文里专门论证过这个细节
- **`softmax`**：把相关度分数归一化成"注意力权重"（每行加和 = 1）
- **乘 V**：用这些权重加权聚合 V 矩阵

```python
# 简化版 PyTorch 实现
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # decoder 的 causal mask
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### 1.1.2 Multi-Head Attention：为什么是"多头"？

单头 attention 只能学到一种"关联模式"。Multi-Head Attention 把 Q/K/V 各自切成 h 份（h 通常 = 8、12、16），每份独立做 attention，最后把 h 个结果拼接 + 线性投影：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

不同 head 学到的注意力模式各不相同——可视化研究（如 Clark 2019 *What Does BERT Look At?*）发现：

- 有些 head 专门关注"前一个 token"（学短距离依赖）
- 有些 head 关注"句法宾语"（连接动词与其宾语）
- 有些 head 关注"特殊 token"（[CLS] / [SEP]）
- 有些 head 在底层做表层模式，深层 head 做抽象语义

**关键参数对照**（典型 7B 模型）：

| 参数 | 典型值 |
|---|---|
| d_model（隐藏维度） | 4096 |
| num_heads | 32 |
| d_k = d_v（每个 head 维度） | 128 |
| num_layers | 32 |
| FFN 中间维度 | 11008（约 2.7×d_model） |
| 总参数量 | ≈7 B |

### 1.1.3 完整 Transformer Block 的结构

```
        Input X
           │
           ▼
    ┌──────────────┐
    │  RMSNorm     │   ← Pre-Norm（现代主流，原论文是 Post-Norm）
    └──────┬───────┘
           ▼
    ┌─────────────────────────┐
    │  Multi-Head Self-Attn   │   ← 加 RoPE 位置编码、KV cache
    └──────┬──────────────────┘
           ▼
        + X (Residual)
           │
           ▼
    ┌──────────────┐
    │  RMSNorm     │
    └──────┬───────┘
           ▼
    ┌──────────────────────────┐
    │  FFN（SwiGLU 激活）      │   ← 现代模型常用 SwiGLU 替代原 ReLU
    │  W1 → SiLU → ⊙W2 → W3   │
    └──────┬───────────────────┘
           ▼
        + (Residual)
           │
           ▼
        Output
```

2017→2026 期间核心架构改进（按重要性）：
1. **Pre-Norm 替代 Post-Norm**：RMSNorm/LayerNorm 放在 sublayer 前面，训练更稳定
2. **RMSNorm 替代 LayerNorm**：去掉 mean centering，只做 variance normalization，更快更稳
3. **RoPE（Rotary Position Embedding）替代 sinusoidal**：相对位置编码，外推性更好
4. **SwiGLU 替代 ReLU**：FFN 用 GLU 变体，效果更好
5. **GQA（Grouped Query Attention）**：多个 Q head 共享 K/V，KV cache 缩小、推理更快
6. **MoE（Mixture of Experts）**：FFN 拆成多个专家，路由稀疏激活——Mixtral / DeepSeek-V3 / GPT-4 等

### 1.1.4 Tokenization：模型看到的不是文字，是 token

模型不直接处理字符或单词，而是处理 **token**——介于字和词之间的子词单元。Tokenization 是把人类文本转 token id 序列的过程，看似工程细节，**但直接决定 cost、context 长度、跨语言效果、以及一类著名 bug "glitch tokens"**。

#### 三大主流算法

| 算法 | 思路 | 代表使用 |
|---|---|---|
| **BPE（Byte-Pair Encoding）** | 反复合并出现频率最高的字符对，直到达到目标词表大小 | GPT-2/3/4、Llama、Mistral、Qwen、DeepSeek |
| **BBPE（Byte-level BPE）** | BPE 在字节级别上做（256 bytes 的基础词表），保证对任意 Unicode 都不爆词表 | GPT-2 起 OpenAI 全系、Llama 3 起、Mistral Tekken |
| **SentencePiece（Unigram or BPE）** | 不依赖空格预切分，直接从 raw text 训练；是 Google 出品 | T5 / mT5 / Gemma / PaLM / 中日韩 / 多语言模型主流 |
| **WordPiece** | BPE 变种，按互信息合并而非频率 | BERT 系（已不主流） |

#### 为什么中日韩需要特别注意

**BPE 假设空格是词边界**——这对英语没问题，但中日韩泰这些**不用空格分词的语言会被切得很碎**。例如 GPT-3 早期对中文的切分：

```
"今天天气真好" → ['今', '天天', '气', '真', '好']    # 5 个 token，每字约 1 token
等价英文 "Today's weather is great" → 4-5 token
```

中文 token 化效率比英文低 30-50% 是常态，意味着：

- **同样意思的中文 prompt 比英文贵 30-50%**
- **同样的 context window 装中文比英文少 30-50%**

2024-2026 改进：DeepSeek、Qwen、智谱 等中文优化模型用更大词表（150k vs OpenAI 100k）+ 中文优化 SentencePiece，把中文压缩比拉到接近 1:1。

#### tiktoken vs SentencePiece：选哪个

| 工具 | 出品 | 用途 | 训练？ | 速度 |
|---|---|---|---|---|
| **tiktoken**（Rust） | OpenAI | 编码 / 解码 + 算 token 数 | ✗ 仅推理 | 最快（3-6× 其他） |
| **SentencePiece** | Google | 训练 + 编码 + 解码 | ✓ | 中 |
| **HuggingFace Tokenizers**（Rust） | HF | 通用 | ✓ | 快 |
| **karpathy/minbpe** | Karpathy | 教学纯 Python | ✓ | 慢 |

**实战**：

- 算 OpenAI / Anthropic / 任何用 tiktoken 兼容词表模型的 token 数 → `tiktoken`
- 训练自家专用 tokenizer（如垂直领域、中文优化）→ `SentencePiece` 或 `HuggingFace Tokenizers`
- 学习算法本质 → karpathy 的 `minbpe`（300 行 Python）

```python
# tiktoken 算 prompt 的 token 数
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
n_tokens = len(enc.encode("你的 prompt 文本"))
print(f"输入 token 数: {n_tokens}, 估算 cost: ${n_tokens * 5e-6:.4f}")
```

#### Glitch Tokens：tokenization 的暗坑

2023 年发现的 *SolidGoldMagikarp* 现象：让 ChatGPT 重复 "SolidGoldMagikarp" 它会输出 "distribute"。原因是：tokenizer 训练数据里这个用户名出现频繁，得到了一个独立 token id；但模型训练数据里几乎没有，导致这个 token 的 embedding **近乎随机噪声**——模型看到这个 token 就胡言乱语。

GlitchMiner 框架（AAAI 2026）系统性扫描 GPT-4 / Llama 2 / Mistral / DeepSeek-V3，发现**约 4.3% 词表是 glitch tokens**。生产建议：

- 用户输入做白名单过滤极端字符 / 罕见 unicode
- 关键 prompt 用 tiktoken 解码后人工 review 是否有异常 token
- 安全敏感场景禁用 raw text 输入，改 structured input

### 1.1.5 位置编码：从 sinusoidal 到 RoPE 与 YaRN

Transformer 是 **permutation-invariant**——不加位置编码的话，"猫吃鱼"和"鱼吃猫"长得一样。位置编码 9 年的演进：

| 方法 | 原理 | 优势 | 限制 |
|---|---|---|---|
| **Sinusoidal**（原论文 2017） | 给每个位置加一个 sin/cos 向量 | 不需要参数 | 长度外推差 |
| **Learned Absolute** | 每个位置一个可学习向量 | 简单 | 训练长度即上限 |
| **ALiBi**（2022） | 不在 input 加，在 attention score 上加线性偏置 | 长度外推好 | 表达力受限 |
| **RoPE（Rotary Position Embedding）**（Su 2021） | 用旋转矩阵把位置编进 Q/K | **2024-2026 事实标准**，相对位置 + 外推 | 直接外推到训练长度 N× 性能掉 |

**RoPE 数学**（简化）：把位置 m 处的 query/key 向量旋转 mθ 角度，使得 `q_m · k_n` 自然依赖于相对位置 `m-n`：

```
q_m → R(mθ) · q_m
k_n → R(nθ) · k_n
attention score = q_m · k_n = q · R((m-n)θ) · k   ← 仅依赖 m-n
```

不同维度用不同 θ_i = 10000^(-2i/d)，相当于多频率正余弦。

#### 长上下文外推：RoPE 不够要扩

直接拿 4K 训练的 RoPE 模型跑 32K context 会崩——位置在训练时没见过。**长上下文扩展的几种主流方法**（按出现顺序）：

| 方法 | 思路 | 何时用 |
|---|---|---|
| **Position Interpolation (PI)**（Meta 2023） | 把新长度的位置线性压回训练范围 | 简单但损失信息 |
| **NTK-Aware Scaling**（社区 2023） | 高频维度少压、低频维度多压 | 不需要重训，hot patch |
| **YaRN**（Peng 2023） | NTK 分块 + temperature 调整 attention | **128K+ context 主流方法** |
| **LongRoPE**（Microsoft 2024） | 进化算法搜不同 dim 的最佳缩放 | 极致 2M+ context |
| **DCA（Dual Chunk Attention）** | 把长 context 分块再融合 | Qwen 长版用 |

**实战**：基础模型用 4-8K 训练，扩到 128K 用 YaRN，扩到 1M+ 用 LongRoPE 或 DCA。Llama 3.1 / Qwen 2.5 / DeepSeek-V3 都是这条路径。Grok-4-fast 等 2026 模型上 2M context，本质都是 YaRN/LongRoPE 系的演进。

但 Transformer 自诞生起就背着一个数学包袱：**标准自注意力对序列长度 N 是 O(N²) 的时间和显存复杂度**。把上下文窗口从 32K 扩到 128K，计算量并不是 4 倍，而是 16 倍。这一条数学事实驱动了 2022-2026 年间数十亿美元的工程投入。

围绕"如何把 O(N²) 这条墙凿穿"诞生了一系列变体：

- **Linear Attention**（Katharopoulos 2020）通过核函数特征映射 + 矩阵乘法结合律避开显式 attention 矩阵，把复杂度降到 O(N)
- **Sparse Attention**（如 Longformer / BigBird）让每个 token 只看局部窗口 + 少数全局 token
- **Flash Attention**（Dao 2022 起）不是改算法而是改工程——把 attention 计算用 tiling 拆成能塞进 GPU 片上 SRAM 的小块，避免把 N×N 矩阵写回 HBM。**Flash Attention 4 在 2026 年 3 月发布，在 NVIDIA B200 GPU 上达到 1605 TFLOPs/s、71% 硬件利用率**
- **Paged Attention**（vLLM 团队 2023）借用操作系统虚拟内存思想，把 KV cache 切成 16 token 一块的页，使显存碎片低于 4%
- **Local Attention** / **Sliding Window Attention** 用于 Mistral / Qwen 等长序列优化
- **MoE（Mixture of Experts）注意力变体**与稀疏激活路由结合（Mixtral / DeepSeek-V3 等）

到 2026 年，这些技术已经不是"二选一"，而是按层 / 按块混合堆叠：模型底层用 Sliding Window 控成本，关键层用 Full Attention 保表达力，推理时全部走 Flash + Paged 优化。

**Scaling Laws** 是与 Transformer 并行演进的另一条主线。Kaplan 等人 2020 年提出参数量 / 数据量 / 计算量的幂律关系，Hoffmann 等人 2022 年（Chinchilla）修正了"应该用多少 token 训练多少参数的模型"——结论是过去主流（如 GPT-3）严重 under-trained。而 2025-2026 年的关键转折是：**重心从"更多数据"转向"更高质量的数据"**。当 Common Crawl 已经被吃完、合成数据兴起，定律关注的不再是 token 数本身，而是 token 的信息密度与多样性。

---

## 1.2 LLM 训练栈：预训练 → 后训练 → 微调

2024-2026 一种常见拆法是把训练栈分为"三段式"——预训练（Pretraining） → 后训练（Post-training） → 任务级微调（Task-specific Fine-tuning）。Sebastian Raschka 在 *State of LLMs 2025* [22] 中的点评精准：「12 个月前的标准配方是预训练几万亿 token 然后跑 RLHF；那条配方已经死了。从 DeepSeek-R1 到 Nemotron 3 Super 到 GPT-5.3 Codex，过去一年发布的每个主要模型都用了不同的后训练栈。」

### 1.2.1 预训练（Pretraining）

任务非常简单：在巨量语料上做下一个 token 预测（next-token prediction），用 cross-entropy loss 反向传播。但工程难度极高：

- **数据**：从公开 Web（Common Crawl、C4、RefinedWeb）+ 代码（GitHub、StackOverflow）+ 学术（arXiv、Wikipedia、书籍）+ 多语言 + 合成数据（蒸馏自更强模型）配比
- **配比策略**：高质量数据多过几轮（curriculum learning），数学 / 代码 / 推理类数据比例上升
- **基础设施**：千卡到万卡 GPU 集群，3D 并行（数据 / 张量 / 流水线 / Sequence parallel）、checkpointing、容错恢复
- **超参**：BF16 / FP8 混合精度、Lion / AdamW 优化器、cosine learning rate、warmup
- **数据去重 / 去毒 / PII 过滤**

预训练的产物是 **基础模型（base model）**——它能续写但不能"对话"，给它"今天天气如何？"它会接着写一段类似贴吧的杂谈，而不是回答你。要让它变成 ChatGPT 那样能交互的助手，需要后训练。

### 1.2.2 后训练（Post-training）

后训练在 2024-2026 年被拆成三个相对正交的阶段 [23][24]：**SFT（让模型会说话）→ 偏好对齐（让模型说人喜欢的话）→ 推理强化（让模型在可验证任务上更准）**。顺序很重要。

**SFT（Supervised Fine-Tuning，监督微调）**

用人工编写的"指令-回答"对（如 Alpaca / Dolly / OpenAssistant 风格）对基础模型继续训练，让它学会"指令跟随"和对话格式。SFT 教模型**怎么说话**——产生结构化输出、遵循 system prompt、用合适的礼貌度等。技术上和预训练区别很小，差异主要在数据上。

**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**

OpenAI InstructGPT / ChatGPT 的看家本领。完整流程分三步：

```
Step 1: SFT      Base Model + 指令对话数据 → SFT Model（可对话）
   │
   ▼
Step 2: RM 训练   收集 (prompt, chosen, rejected) 偏好对 → 训练 Reward Model
   │              偏好对来自人类标注："回答 A 比 B 更好"
   ▼
Step 3: PPO       SFT Model 作为 actor，RM 给奖励，用 PPO 优化
                  目标：max E[r(x,y)] - β·KL(π‖π_ref)
```

**RLHF 的核心目标函数**：

$$
\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(y|x)} \left[ r(x,y) \right] - \beta \, D_{KL}\left[ \pi(y|x) \,\|\, \pi_{ref}(y|x) \right]
$$

读这个公式：

- **`max_π E[r(x,y)]`**：最大化"奖励模型给当前策略生成的回答打的分"
- **`-β·KL[π || π_ref]`**：减去当前策略与参考策略（通常是 SFT 模型）的 KL 散度，乘以系数 β

KL 项是关键——如果只追求 reward 最大化，模型会"作弊"（reward hacking），输出 RM 喜欢但人类不喜欢的乱码。KL 约束保证模型不要偏离 SFT 太远。β 通常取 0.01-0.1。

PPO 实现极复杂，需要同时维护 **4 个模型**：

| 模型 | 作用 | 是否训练 |
|---|---|---|
| **Actor**（policy，π） | 当前策略，生成 y | ✓ 训练 |
| **Critic**（value model） | 估计 V(state)，算优势 A | ✓ 训练 |
| **Reference**（π_ref） | KL 约束的参考策略 | ✗ frozen |
| **Reward Model**（RM） | 给 (x,y) 打奖励分 | ✗ frozen |

7B 模型 RLHF 训练显存爆炸（4 × 14 GB = 56 GB+），训练不稳定（reward hacking、KL 失控、value loss 发散），是公认最难调的训练阶段。

**DPO（Direct Preference Optimization，直接偏好优化）**

斯坦福 Rafailov 等人 2023 年提出的"无 RL 的 RLHF"。核心洞察：把 RLHF 的目标函数做闭式重写后，发现**最优策略与 reward 之间有一个解析关系**——可以直接用偏好对训练原模型，无需奖励模型、无需 PPO。

**DPO 损失函数**：

$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_c, y_r) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)} \right) \right]
$$

读这个公式（其实非常直观）：

- 给定一个偏好对：prompt `x`、chosen 回答 `y_c`、rejected 回答 `y_r`
- 计算两个**对数概率比**：当前模型对 chosen 的偏好程度 vs 参考模型，以及对 rejected 的偏好程度 vs 参考模型
- 让两者的差异最大化（chosen 比 rejected 概率高得越多越好）
- 包在 sigmoid + log 里，等价于二分类的 logistic loss

DPO 的几个工程优势：

| 维度 | RLHF (PPO) | DPO |
|---|---|---|
| 需训练的模型 | actor + critic + RM | 仅 actor |
| 显存占用 | 4 × 模型 | 2 × 模型（actor + ref） |
| 训练稳定性 | 不稳定（reward hacking、KL 失控） | 稳定（普通分类 loss） |
| 实现复杂度 | 上千行 RL 代码 | 几十行 |
| 效果 | 强 | 与 PPO 相当或更好（Llama 3 等验证） |

**2024 年起绝大多数开源模型（Llama 3 / Qwen 2/3 / Mistral / Phi）都改用 DPO 或其变体（IPO / KTO / SimPO）替代 PPO**。

DPO 的变体（关键差异在 loss 形式）：

- **IPO**（Identity-PO）：解决 DPO 在偏好数据极端时过拟合
- **KTO**（Kahneman-Tversky）：不需要成对偏好，仅需"好/坏"二元标注
- **SimPO**：去掉参考模型 π_ref（等价于 β=∞），更省显存
- **CPO**（Contrastive Preference Optimization）：偏好 + SFT loss 联合优化

**GRPO（Group Relative Policy Optimization）**

DeepSeek 团队 2024 年提出，是 R1 系列爆火背后的核心算法 [22][23]。**GRPO 砍掉了 PPO 的 critic 网络**（PPO 训练不稳定的主要来源就是 critic 难训），改用"组内相对优势"。

**GRPO 流程**：

```
对每个 prompt x：
  1. 用当前 policy π_θ 采样 K 个回答 {y_1, y_2, ..., y_K}（K 通常 8-64）
  2. 用奖励函数 r(·) 给每个回答打分 → {r_1, r_2, ..., r_K}
  3. 计算组内归一化优势：
        A_i = (r_i - mean(r)) / std(r)
  4. 用 PPO 风格的 clipped surrogate loss 更新策略
```

**GRPO 损失**（与 PPO 几乎一致，但优势 A_i 不同）：

$$
\mathcal{L}_{GRPO}(\theta) = \mathbb{E} \left[ \min\left( \rho_i A_i, \, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) \right] - \beta \, D_{KL}[\pi_\theta \| \pi_{ref}]
$$

其中 `ρ_i = π_θ(y_i|x) / π_θ_old(y_i|x)` 是 importance ratio。

**GRPO 的关键创新**：

1. **去 critic**：不需要训 value network，省显存、省调参
2. **组内归一化**：让"什么是好回答"完全由 group 内相对比较决定，避免 reward 绝对值飘移
3. **天然适配 RLVR**：奖励可以是 0/1（程序验证），无需训 reward model

**RLVR（Reinforcement Learning with Verifiable Rewards）**

GRPO 通常和 RLVR 配套——奖励信号不是来自人类偏好或 reward model，而是来自 **可程序化验证的信号**：

- 数学题：跑算式验证答案
- 代码：跑单元测试看通过率
- 形式化推理：用定理证明器检查
- 工具调用：看函数调用是否合法

```python
# RLVR + GRPO 伪代码
for prompt in math_problems:
    responses = [policy.sample(prompt) for _ in range(K)]  # 采 K 个
    rewards = [verify_math(prompt, r) for r in responses]  # 程序验证 → 0 or 1
    advantages = (rewards - mean(rewards)) / std(rewards)  # 组内归一化
    loss = grpo_loss(responses, advantages, policy, ref_policy)
    loss.backward()
```

**为什么 GRPO + RLVR 把推理能力推到新高度**：传统 RLHF 的奖励来自 RM（学到的偏好近似），RM 容易被 hack；RLVR 奖励来自硬验证器（不可 hack）。这让 DeepSeek-R1、OpenAI o1、Qwen-QwQ 等推理模型在数学 / 代码 / 形式化任务上突飞猛进。

**三者横向对比**（实战选型）：

| 算法 | 数据需求 | 训练稳定性 | 显存 | 适用场景 |
|---|---|---|---|---|
| **PPO (RLHF)** | 偏好对 + RM | 难 | 高（4 模型） | 历史路径，大厂仍用 |
| **DPO** | 偏好对 | 稳 | 中（2 模型） | 通用对齐，开源主流 |
| **GRPO + RLVR** | 可验证任务 | 较稳 | 中 | 数学 / 代码 / 推理强化 |

一种常见路径：**SFT → DPO 做通用对齐 → GRPO/RLVR 做推理强化**（参考 Raschka [22]、HuggingFace TRL 文档）。注意：各家具体栈差异很大——Raschka 自己也强调"配方已死，每个主要模型都用了不同的后训练栈"，本节给的是"一种"配方而非业界标准。

**RLAIF（Reinforcement Learning from AI Feedback，基于 AI 反馈的强化学习）**

Anthropic 2022 年 *Constitutional AI* 论文里首次系统化：用更强的 LLM 代替人类标注偏好对，大幅压低数据成本。**2025-2026 年 RLAIF 已成为通用做法**——基础模型对齐用 RLAIF，关键安全场景用人类反馈做最后一道把关。

**RLVR（Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习）**

2025 下半年起在 OpenAI o1 / DeepSeek-R1 上爆红。核心思想：在数学题、代码、单元测试等"答案可被程序自动验证"的任务上，奖励信号不来自人类偏好或奖励模型，而来自硬编码的验证器（运行单元测试、对比标准答案）。这避免了奖励模型的"作弊"（reward hacking）问题，把推理能力推到新高度。**GRPO + RLVR 是 R1 的黄金组合**。

**DAPO**（Decoupled Clip and Dynamic Sampling Policy Optimization）等 2026 年新算法继续在 GRPO 基础上做工程优化（动态采样、分层 clip）。

后训练栈的完整图景：

```
Base Model (next-token predict only)
        │
        ▼ SFT  (instruction-following, format)
        │
        ▼ Preference Optimization  (DPO / SimPO / KTO / RLAIF)
        │
        ▼ Reasoning RL  (GRPO / DAPO / RLVR)
        │
        ▼
    Production-Ready Model
```

### 1.2.3 任务级微调（Fine-Tuning）

如果你拿到的不是"训练新基础模型"的资源，而是"在已有开源模型上做领域适配"，那么 fine-tuning 才是你的工具。这里有个关键的成本分水岭 [25]：

| 方法 | 7B 模型显存 | 硬件成本 | 质量保留 | 适用 |
|---|---|---|---|---|
| **Full Fine-Tune** | 100-120 GB | $50K（H100） | 100%（基线） | 延迟敏感 + 最高精度 |
| **LoRA** | ~25 GB | $10K（A100） | 95-98% | 快速实验 + 多 adapter |
| **QLoRA** | ~6 GB | $1.5K（RTX 4090） | 80-90% | VRAM 极限 + 多客户隔离 |
| **Distillation** | 视 student | 中 | 视任务 | 推理时延 / 成本极致 |

- **LoRA**（Low-Rank Adaptation，Hu 等 2021）只在权重矩阵旁边加一对低秩矩阵 A·B，原权重冻结，仅训练 0.2-0.3% 的参数。适合需要给不同客户 / 任务 / 场景维护多个轻量 adapter 的场景，切换 adapter 只需要换一个几百 MB 的小文件
- **QLoRA**（Dettmers 2023）= 4-bit 量化原模型 + LoRA。把基础模型的常驻显存从 ~26 GB（fp16，7B）压到 ~4 GB，让单张消费级显卡（4090、3090）就能跑 7B 模型的微调。质量损失约 5-10%
- **PEFT**（Parameter-Efficient Fine-Tuning）是个伞型术语，涵盖 LoRA、QLoRA、Prefix Tuning、Prompt Tuning、Adapter Tuning、IA³ 等。HuggingFace 的 `peft` 库是事实标准
- **蒸馏（Knowledge Distillation）** 走另一条路：用大模型（teacher）生成大量数据，训练小模型（student）模仿大模型。代价是 student 通常达不到 teacher 的 ceiling，但推理成本可能降一个数量级。Qwen 2.5 系列、Phi 系列大量使用蒸馏

实践指引 [25]：「如果产品需要极低延迟和最高准确率，做 full fine-tune；如果需要快速实验、多变体、按客户单独 adapter，用 LoRA；如果模型很大且 VRAM 受限，用 QLoRA。」

### 1.2.4 数据飞轮：合成数据 + 自我提升（2026 训练数据的核心来源）

公开互联网数据基本被吃完（Common Crawl + Wikipedia + arXiv + GitHub 都已 ingest），**2025-2026 年新模型几乎都靠合成数据 (synthetic data) + 数据飞轮 (data flywheel)**。

#### 三种合成数据来源

| 类型 | 方法 | 代表 |
|---|---|---|
| **强模型蒸馏** | 用 GPT-4 / Claude Opus 生成大量 instruction-response 对训小模型 | Alpaca / WizardLM / Phi 系列 / DeepSeek-V3 |
| **Self-Instruct** | 让模型自己生成 instruction + response，迭代扩充数据集 | Self-Instruct / Evol-Instruct |
| **Verifier-filtered** | 生成 N 个候选，用程序验证器（数学 solver / 单元测试 / 代码 linter）筛通过的 | DeepSeek-R1 训练核心；STaR / ReST |

**典型蒸馏代码**（用 Opus 给小模型造数据）：

```python
def generate_training_data(seed_prompts, target_size=10_000):
    """用 Claude Opus 生成 instruction-response 对"""
    data = []
    for seed in seed_prompts:
        # 生成多样化变体
        variations = opus.invoke(f"""
            为下面任务生成 5 个不同表述的指令：
            原指令：{seed}
        """).split("\n")
        for v in variations:
            response = opus.invoke(v)  # 用 Opus 生成"标准答案"
            # 质量过滤
            if quality_score(v, response) > 0.8:
                data.append({"instruction": v, "response": response})
        if len(data) >= target_size: break
    return data
```

#### 数据飞轮：生产 → 数据 → 模型 → 生产

2026 年最强模型（DeepSeek-R1 / Claude / GPT-5 系）都跑这个闭环：

```
   ┌──────────────────┐
   │ 部署模型到生产   │
   │（覆盖真实用户）  │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 收集真实交互数据 │
   │ + 用户反馈       │ ← 来自 thumbs up/down / latency / cost
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 自动筛选高价值   │
   │ trace（成功 +    │ ← LLM-as-judge / verifier
   │ 多样 + 高难度）  │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 合成数据增强     │ ← Self-instruct 扩 10×
   │ + LLM 改写       │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 后训练（DPO/GRPO）│
   │ 出新模型版本     │
   └────────┬─────────┘
            │
            └─────────────► 回到部署
```

**飞轮越转越快的原理**：

1. 部署 → 用户产生真实需求分布
2. 真实分布 → 训练数据贴近实际使用
3. 训练后模型对真实分布表现更好
4. 表现更好 → 更多用户 / 更多数据
5. ...

**对比一次性训练 + 部署**：

| 模式 | 数据来源 | 演化速度 |
|---|---|---|
| 一次性训练 | 公开 web | 训完即冻 |
| 数据飞轮 | 真实用户 + 合成增强 | 每周 / 每月 fine-tune 一次 |

#### 工程实践要点

- **隐私先行**：用户数据进训练集前必须 PII 脱敏 + 同意条款（GDPR / CCPA）
- **质量过滤 > 数据量**：1 万高质量样本 > 100 万低质量
- **多样性约束**：避免模型坍塌到狭窄分布（用 embedding 聚类 + 跨 cluster 采样）
- **Active Learning**：优先标注 / 训练模型当前最不擅长的样本（用 entropy 或 LLM-as-judge 找 hard cases）
- **Fine-tune 周期**：典型 4-8 周一次 base model 升级，1-2 周一次 LoRA adapter 升级

#### 注意：模型坍塌 (Model Collapse)

**仅用合成数据训练的模型会逐代退化**——多样性塌缩 + 罕见事件遗忘 + 错误放大。Shumailov 等人 *Nature* 2024 论文系统证明：纯合成数据训练 3-5 代后，模型质量崩溃。

对策：

- 真实数据 + 合成数据**混合**（典型比例 30% 真实 + 70% 合成）
- 每代保留"种子"真实数据（不淘汰）
- 用强 verifier / 人类标注过滤合成数据

---

## 1.3 推理栈与解码策略

模型训完上线，"用什么策略从 logits 里采样下一个 token"决定了输出质感：

- **Greedy / Argmax**：永远选概率最高的 token。结果稳定但容易陷入重复（"the the the..."）
- **Beam Search**：维护 top-K 个候选序列。机器翻译里好用，对话里偏机械
- **Top-K Sampling**：从概率最高的 K 个里随机采。K=40 是常见值
- **Top-P / Nucleus Sampling**（Holtzman 2019）：从累积概率到 P 的最小集合里采。P=0.9 是常见值，比 top-K 更自适应
- **Temperature**：对 logits 除以温度 T 后再 softmax。T<1 偏保守（接近 greedy），T>1 偏发散，T=0 等价 greedy
- **Min-P Sampling**（2024）：相对阈值采样，比 top-p 更稳健地处理高 confidence 分布
- **Repetition Penalty / Frequency Penalty / Presence Penalty**：抑制重复 token

工程上现在的事实标准是 `temperature=0.7 + top_p=0.95`（OpenAI / Anthropic 的默认）。**要做创意生成调高 T，要做严格 JSON 提取调到 T=0 + 配合 grammar-guided generation**（如 Outlines、Guidance、Llama.cpp 的 GBNF）。

**Speculative Decoding**（详见 [04_harness_engineering.md](./04_harness_engineering.md) §3.9.2）和 **Prefix Caching** 等推理服务器层优化也属于推理栈但偏服务部署，本章只述及解码策略层面。

---

## 1.4 提示词工程的经典技术

提示词工程的"经典时代"基本可以用一张技术图谱概括 [28][29][30]：

### 1.4.1 零样本 / 少样本（Zero-shot / Few-shot）

直接问 vs 给几个例子。Brown 等人 2020 年的 GPT-3 论文发现 few-shot 能让小模型逼近大模型 fine-tuned 后的水平——这是"上下文学习（in-context learning）"概念的诞生。**2026 年 few-shot 仍是 80% 任务的最强 baseline**。

### 1.4.2 Chain-of-Thought（CoT，思维链）

Wei 等人 2022 年提出。让模型先写"思考过程"再写答案，大幅提升数学 / 推理任务准确率。简单触发词 "Let's think step by step." **在 GSM8K 上 PaLM-540B 从 17.9% 提升到 56.9%（CoT 单独）；叠加 self-consistency 后到 74%+** [Wei et al. 2022]。CoT 的核心机制：把原本要在前向传播里一次完成的复杂推理，分摊到多个 token 的生成上，等价于动态扩展计算量。

### 1.4.3 Self-Consistency

Wang 等人 2022 年。同一个 CoT prompt 采样 N 次（高 temperature），最后投票选最常出现的答案。在数学 / 编程上又把 CoT 提升 5-10 个百分点，代价是 N 倍 token 成本。

### 1.4.4 ReAct（Reasoning + Acting）

Yao 等人 2022 年 [4]。把思考（Thought）和行动（Action）交错：模型先想"我现在该查什么"，然后调用工具（搜索 / 计算器 / API），观察结果（Observation），再继续想下一步。**在 ALFWorld 交互式任务上比 imitation learning + RL 高 34% 绝对成功率**，在 WebShop 上高 10%。**ReAct 是后来所有 Agent 架构的祖父**——LangGraph 的 react agent template、AutoGen 的 conversable agent、CrewAI 的 task agent，本质都是 ReAct 的工程化变体。详见 [03_agent_engineering.md](./03_agent_engineering.md) §2。

### 1.4.5 Tree of Thoughts（ToT）/ Graph of Thoughts（GoT）

Yao 2023 / Besta 2023。CoT 是单条思考链，ToT 把每一步展开成多个分支，用 BFS/DFS + 启发式打分搜索整棵推理树。GoT 进一步把树推广为 DAG，允许节点合并（"我这两条思路其实指向同一个答案"）。代价是 token 成本数倍到十倍，仅在数学推理 / 规划等"答案稀疏但路径多"的任务有性价比。

### 1.4.6 Self-Refine

Madaan 等人 NeurIPS 2023 [3]。同一个模型扮演三个角色：generator（生成初稿）→ critic（自我评判）→ reviser（重写）。**在 7 个任务上平均提升 20%**（GPT-3.5 / GPT-4 都验证过）。这是后来 Reflexion、CRITIC、Constitutional AI 等"自我改进"路径的原型。

### 1.4.7 Reflexion

Shinn 等人 2023。Self-Refine 是单轮自我改进，Reflexion 把自我反思写入"长期记忆"——如果某次任务失败，把失败原因总结成一段反思，附在下次同类任务的 prompt 里。等价于让 agent 从经验里学习。

### 1.4.8 Prompt Chaining / Decomposition

复杂任务拆成多步骤的 prompt，每步输出喂给下一步（即"workflow"模式）。LangChain、LlamaIndex、Semantic Kernel 大量这套。优点是可调试、可观测；缺点是步骤间信息可能丢失。

### 1.4.9 Role Prompting / System Prompt

"You are an expert Python developer. ..." 给模型注入身份与场景。Claude 系列对 system prompt 的遵循度普遍高于 GPT 系列，所以社区有"Claude 提示工程偏 XML 标签 + 强 system prompt"的共识 [29]。

### 1.4.10 模型差异化

2026 年的提示词工程已经从"通用最佳实践"分化为"模型特异性最佳实践" [29][30]：

- **Claude（Anthropic）**：偏好 XML 标签（`<context>...</context>`、`<instructions>...</instructions>`），长 system prompt，多步推理任务直接说 "think step by step"
- **GPT 系列（OpenAI）**：偏好简洁的 JSON schema，function calling 结构化，markdown 友好
- **Gemini**：多模态原生，对图像 + 文本混合 prompt 支持最好
- **DeepSeek-R1 / o1 / 思考模型**：不要再加 CoT 触发词（模型自带），直接给问题；temperature 偏低

### 1.4.11 可复制的 Prompt 模板速查

下面是上面 10 种技术的 production-ready 模板，复制即可用。

#### 模板 A：零样本 + 角色 + 输出格式（最通用基础模板）

```
你是一名{角色描述}。

## 任务
{清晰的任务描述}

## 上下文
{相关背景，按重要性排列}

## 约束
- {约束 1，例如：仅用提供的资料}
- {约束 2，例如：不超过 200 字}
- {约束 3，例如：用中文回答}

## 输出格式
{Markdown / JSON schema / 段落}

## 输入
{user_input}
```

#### 模板 B：Few-shot（带 N 个示例）

```
你的任务是把英文产品名翻译成中文，保持品牌名不译。

## 示例
英文：iPhone 16 Pro Max
中文：iPhone 16 Pro Max

英文：MacBook Air with M4 chip
中文：搭载 M4 芯片的 MacBook Air

英文：Apple Vision Pro Headset
中文：Apple Vision Pro 头显

## 输入
英文：{input}
中文：
```

**关键点**：示例顺序无关紧要，但**最相似于待解题的示例放最近**召回最高（few-shot 也有 lost-in-the-middle）。

#### 模板 C：Chain-of-Thought（CoT）

```
请逐步推理后给出答案。

问题：{question}

请按以下格式：
1. 理解问题：{重述问题}
2. 拆分子问题：{列出需要解决的子问题}
3. 逐个推理：{每个子问题的思考}
4. 综合：{把子结论组合成最终答案}
5. 答案：{最终结论}
```

简化版（不需要结构化思考）：

```
{question}

Let's think step by step.
```

#### 模板 D：Self-Consistency（单 prompt + N 次采样投票）

```python
# 采样代码层
def self_consistency_answer(question, n=5, temperature=0.7):
    answers = []
    for _ in range(n):
        response = llm.invoke(
            prompt=f"{question}\n\nLet's think step by step.",
            temperature=temperature,
        )
        # 用 regex 或 LLM 抽取最终答案
        final = extract_final_answer(response)
        answers.append(final)
    return Counter(answers).most_common(1)[0][0]  # 投票
```

#### 模板 E：ReAct（Thought-Action-Observation 循环）

```
你可以使用以下工具：
{tool_descriptions, e.g.}
- search(query: str) -> str: 网页搜索
- calculator(expr: str) -> float: 计算
- read_file(path: str) -> str: 读文件

每一步必须按以下格式：
Thought: {你的推理}
Action: {tool_name}
Action Input: {JSON 形式的参数}
Observation: {工具返回的结果}
（重复 Thought/Action/Observation 直到能回答）
Thought: 我现在能回答了
Final Answer: {最终回答}

问题：{question}

Thought:
```

#### 模板 F：Self-Refine（生成-评判-修订）

```
# Step 1: Generator
请回答以下问题：{question}

# Step 2: Critic（用上一步输出做输入）
评估以下回答的质量。指出 3 个具体问题：
- 是否准确？
- 是否完整？
- 是否清晰？

回答：
"""
{step1_output}
"""

请按 JSON 输出：{"issues": ["...", "...", "..."]}

# Step 3: Reviser
根据下列反馈，重写上一步回答：

原回答：{step1_output}
反馈：{step2_output}

重写后的回答：
```

#### 模板 G：结构化输出（JSON Schema 约束）

```python
# OpenAI Structured Output
from pydantic import BaseModel
from typing import Literal

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": f"分析情感：{text}"}],
    response_format=Sentiment,
)
result: Sentiment = response.choices[0].message.parsed

# Anthropic 用 XML
prompt = f"""分析下面文本的情感，按 XML 输出：
<text>{text}</text>

<output>
  <label>positive | negative | neutral</label>
  <confidence>0.0 到 1.0</confidence>
  <reasoning>简要理由</reasoning>
</output>"""
```

#### 模板 H：Tree-of-Thoughts（多路径搜索）

```python
def tot_solve(problem, depth=3, breadth=3):
    """简化版 ToT：每步 generate B 个候选，evaluate 选 top-1 继续。"""
    state = problem
    for d in range(depth):
        # Generate B 个候选下一步
        candidates = []
        for _ in range(breadth):
            cand = llm.invoke(
                f"问题：{problem}\n当前状态：{state}\n生成下一步推理：",
                temperature=0.8,
            )
            candidates.append(cand)
        # Evaluate
        best = llm.invoke(
            f"问题：{problem}\n以下 {breadth} 个候选下一步推理，"
            f"哪一个最有希望解出问题？只回答数字 1-{breadth}：\n" +
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        )
        state = candidates[int(best) - 1]
    return state  # 最终推理路径
```

#### 模板 I：Reflexion（跨任务长期反思）

```python
class ReflexionAgent:
    def __init__(self):
        self.lessons_learned = []   # 跨任务的反思笔记

    def attempt(self, task):
        # 把过去的反思注入新 prompt
        prompt = f"""
任务：{task}

# 我从过去任务总结的经验
{chr(10).join(f'- {l}' for l in self.lessons_learned[-10:])}

请完成任务，注意避开过去的错误。
"""
        result = llm.invoke(prompt)
        success = evaluator.check(task, result)
        if not success:
            # 反思失败原因
            reflection = llm.invoke(f"""
任务：{task}
我的回答：{result}
失败原因分析（写一句话给未来的我看）：
""")
            self.lessons_learned.append(reflection)
        return result
```

#### 模板 J：模型差异化（同一任务三种写法）

| 任务 | Claude（XML） | GPT-4（JSON） | Gemini（多模态友好） |
|---|---|---|---|
| **简单分类** | `<input>...</input>\n<output>label</output>` | `Output JSON: {"label": "..."}` | 同 GPT |
| **多步推理** | `<thinking>步骤</thinking><answer>X</answer>` | `Step 1: ... Step 2: ... Final: X` | 同 |
| **代码生成** | `<requirements>...</requirements>\n<code>` | ` ```python\n...` | 同 GPT |
| **图像 + 文本** | 图像 + `<task>分析</task>` | 同 | 直接 multimodal API |

#### 实战：写好 Prompt 的 7 条戒律

1. **明确角色 / 视角**：「你是一名 X 专家」开场，比模糊指令好
2. **示范优于解释**：给 1-3 个 few-shot 比写 5 段规则有效
3. **拆分长任务**：复杂任务用 prompt chaining 而非塞给单个 prompt
4. **限制输出**：明确长度、格式、风格、禁止项
5. **拒绝回答的兜底**：「如果不确定，回答"我不知道"而不是猜测」
6. **A/B 测试**：用 Promptfoo 跑 20+ 测试用例，不要凭直觉
7. **版本化**：prompt 入 git，每次改动都记 changelog

### 1.4.12 Reasoning Models 的 prompt 哲学完全不同

2024 年 9 月 OpenAI 发布 **o1** 开启 "reasoning model" 范式。2025-2026 年 DeepSeek-R1 / Claude 3.7+ extended thinking / Gemini 2.5 thinking / GPT-5 reasoning 全面跟进。这类模型在生成可见 token 之前会先生成大量**内部推理 token**（2K-30K hidden tokens），相当于把 CoT 内化为模型自身能力。

#### 关键差异

| 维度 | 传统 LLM（GPT-4o / Claude Sonnet 4.5） | Reasoning Model（o1 / R1 / Claude 4.5 thinking / R2） |
|---|---|---|
| 内部推理 | 没有，依赖你的 prompt 触发 CoT | **自动生成 2K-30K thinking tokens** |
| `temperature` | 0.7 常见 | **0** 或忽略（很多模型不支持） |
| CoT 触发词 | "think step by step" 必加 | **不要加**——模型已自带，加了反而干扰 |
| Few-shot | 几乎必备 | **少用甚至不用**——few-shot 反而分散推理 |
| 系统 prompt | 长 / 详细 | **短 / 直接**——告诉它"做什么"即可 |
| 延迟 | 秒级 | 30 秒 - 数分钟 |
| Token 成本 | 实付 = 看到的 | **实付 = 看到的 + 隐藏的 thinking**（10-100×） |
| 适用 | 大多数任务 | 数学 / 代码 / 复杂推理 / 规划 |

#### 反范式：reasoning model 的"少即是多"

OpenAI o1 的官方 prompt 指南直接说：

> 不要用 chain-of-thought prompting。让模型自己思考。
> 不要用 few-shot examples 除非任务非常 niche。
> 写 prompt 时直接给目标和约束，越简单越好。

DeepSeek-R1 的 `<think>` 标签是显式的（其他模型隐藏），可以让你**看到**它在想什么，但**不要**自己拼 `<think>` 进 prompt（会干扰）。

#### 何时用 reasoning model

- ✅ 用：数学题、代码 debug、复杂规划、多步推理、需要"想清楚"的任务
- ❌ 不用：简单 chat、内容生成、需要快速响应（>5s 用户跑掉）、预算敏感
- ⚠️ Hybrid：Anthropic 的 *extended thinking* 可以**按 budget 控制**——给 1024 tokens thinking budget 适合中等推理任务，给 16384+ 适合复杂数学

#### Hybrid Reasoning：2026 主流

Claude Sonnet 4.5+ / Gemini 2.5+ / GPT-5 都做成 **hybrid**：默认快速回答，需要深度时启用 thinking。代码：

```python
# Anthropic extended thinking
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    thinking={"type": "enabled", "budget_tokens": 4096},  # ★ 按需启用
    messages=[{"role": "user", "content": "证明费马小定理"}],
)

# 同一模型不开 thinking 跑普通 chat
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    # 不设 thinking → 走快速路径
    messages=[{"role": "user", "content": "今天的天气怎么样？"}],
)
```

**实战经济学**：reasoning model 在 2025-2026 已经"商品化"——开源 reasoning 模型（Kimi K2 / Qwen3 / GLM-5 等）在每百万 token 的价格区间（具体 input/output 见各家官网定价页）已大幅低于第一代闭源 reasoning 模型（如 o1-pro / o3-pro）。"为质量付十倍价钱"的论点在很多场景已不成立，但具体倍率因模型档次和任务难度差异很大。

### 1.4.13 多模态 prompting：图 / 音 / 视

2024-2026 年主流模型几乎全多模态原生（vision 必备，audio/video 渐次跟进）。多模态 prompting 不是文本 prompt 加张图就完事，有自己的最佳实践。

#### 各家原生支持矩阵（2026-04）

| 模型 | 图像 | PDF | 音频 | 视频 | 强项 |
|---|---|---|---|---|---|
| **Claude Opus 4.7 / Sonnet 4.6** | ✓ | **✓ 业界最强** | ✗ | ✗ | 长 PDF / 多页文档推理 |
| **GPT-5 / GPT-4o** | ✓ | ✓ | **✓ 原生** | 帧级 | 截图 / chart / 音频 |
| **Gemini 3.1 Pro** | ✓ | ✓ | ✓ | **✓ 1 小时视频** | 视频是一等公民 |
| **DeepSeek-V4 Vision** | ✓ | 中 | ✗ | ✗ | 中文场景 |
| **Qwen 3.5 Vision** | ✓ | ✓ | ✓ | 中 | 中文 + 性价比 |

#### 多模态 prompt 的 4 条要点

**1. 媒体放在 prompt 前面**

```python
# ✅ 推荐：媒体 → 文本指令
messages = [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                  "data": image_b64}},
    {"type": "text", "text": "这张架构图里 PostgreSQL 实例数是多少？请数清楚。"},
]}]

# ❌ 避免：文本在前会让模型先建立期望，影响视觉解读
```

**2. 多图时给每张图编号**

```python
content = [
    {"type": "text", "text": "下面给你 3 张图，分别标号 A / B / C："},
    {"type": "text", "text": "Image A:"},
    {"type": "image", ...},
    {"type": "text", "text": "Image B:"},
    {"type": "image", ...},
    {"type": "text", "text": "Image C:"},
    {"type": "image", ...},
    {"type": "text", "text": "请回答：A 和 C 哪个亮度更高？"},
]
```

**3. 任务模型选择**

> 不要一个模型干所有多模态。**异构是对的**：合同走 Claude、截图 / 音频走 GPT、视频走 Gemini。

实战分流：

```python
def route_multimodal(task):
    if task.has_video: return gemini
    if task.has_audio: return gpt5
    if task.is_pdf and task.pages > 50: return claude
    if task.is_screenshot: return gpt5
    return claude  # default
```

**4. 视频处理的 GPT-5 模式：抽帧 + 文本描述**

GPT-5 不接收 video 直接输入，要**自己抽帧 + 用 vision 描述每帧 + 拼起来**：

```python
import cv2
def process_video_with_gpt5(video_path, fps_sample=1):
    cap = cv2.VideoCapture(video_path)
    frame_descriptions = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % fps_sample == 0:
            desc = gpt5_vision(frame, prompt="一句话描述这一帧")
            frame_descriptions.append(f"[{frame_idx/fps:.1f}s] {desc}")
    return llm.invoke(f"视频帧描述：\n{chr(10).join(frame_descriptions)}\n\n请综合说明视频内容")
```

Gemini 直接收 video URL / bytes，省掉抽帧，长视频内置 token efficient 编码。

#### 多模态的 PII 与隐私

图像 / 视频里可能含人脸 / 车牌 / 屏幕截图里的密码——上传给 third-party API 前必须处理：

- 人脸打码：MediaPipe / dlib
- 屏幕截图自动 redact：自己 mask 已知敏感区域 / 用 OCR 检测后 mask
- 视频会议录像必须脱敏后再喂 LLM

---

## 1.5 自动化与高级提示词

### 1.5.1 为什么需要自动化

人工写 prompt 终究有上限：

- **空间太大**：一个任务可能有数百种 prompt 写法，人不可能手工 A/B 全部
- **模型升级即失效**：换 GPT-4o → GPT-5 后，最优 prompt 经常不一样
- **Few-shot 选样难**：从 1000 个候选 example 里挑哪 5 个 demo，组合数爆炸
- **指令措辞敏感**："think step by step" vs "reason carefully" 准确率可能差 5-10%

2023 年起涌现"**让 LLM 自己写 prompt**"的方向，统称 **automated prompt optimization**。

### 1.5.2 四大流派

#### APE（Automatic Prompt Engineer）— Zhou 2023

最朴素：

```
1. 用 LLM 生成 N 个候选 prompt
2. 在 dev set 上跑每个候选 → 打分
3. 选 top-K → 让 LLM 看了它们后再生成 N 个新候选
4. 迭代直到收敛
```

效果：在多个 NLP 任务上比人工 prompt 高 10-20%。开创了"prompt 优化"作为问题。

#### OPRO（Optimization by PROmpting）— Yang 2023, Google

**让 LLM 直接担任优化器**。每轮告诉 LLM：

```
过去这些 prompt 拿到这些分数：
  prompt 1: "Let's think step by step." → 78%
  prompt 2: "Take a deep breath and work step by step." → 80.2%   ← OPRO 发现的真实 prompt
  prompt 3: ...
请提出一个能拿更高分的新 prompt。
```

OPRO 在 GSM8K 上发现了"Take a deep breath..."这条比"think step by step"更高的指令——一条经典 LLM 自我发现案例。

#### Promptbreeder — Fernando 2023, DeepMind

**APE + 进化算法**：维护一个 prompt 种群，每轮做：突变（让 LLM 改写）+ 选择（按 dev 分数） + 交叉（合并两个高分 prompt 的部分）。在数学推理任务上超过人工最优。

#### DSPy — Stanford 2023（事实主流）

把 prompt 工程**当作编译问题**——你写"程序"（Module + Signature），DSPy 编译器自动选最优 demo 和指令。**核心抽象 3 个**：

1. **Signature**：声明输入输出（"question -> answer"）
2. **Module**：可组合的程序单元（`dspy.Predict`、`dspy.ChainOfThought`、`dspy.ReAct`）
3. **Optimizer**（teleprompter）：用训练集 + metric 自动优化 module 内的 prompt

### 1.5.3 DSPy 完整工作流（产线级示例）

#### 基础概念示例

```python
import dspy

# 配置 LM（任何 LiteLLM 支持的 provider）
lm = dspy.LM("anthropic/claude-sonnet-4-6")
dspy.configure(lm=lm)

# 1. 定义 Signature（声明 input → output 类型）
class GenerateAnswer(dspy.Signature):
    """根据上下文回答问题，引用 source。"""
    context: str = dspy.InputField(desc="可能包含答案的相关段落")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="简洁、引用 source 的回答")

# 2. 定义 Module（你的"程序"）
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# 3. 直接跑（用 default prompt，不优化）
rag = RAG()
result = rag(question="2026 年 SWE-bench 头部模型？")
print(result.answer)
```

#### 用 BootstrapFewShot 优化

```python
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

# 准备训练集（30-100 条 (question, answer) 即可）
trainset = [
    dspy.Example(question="...", answer="...").with_inputs("question")
    for q, a in your_qa_pairs
]

# Optimizer：bootstrap few-shot demos
optimizer = BootstrapFewShot(
    metric=answer_exact_match,    # 你的评估函数
    max_bootstrapped_demos=4,     # 自动生成的 demo 数
    max_labeled_demos=4,          # 从 trainset 直接采的 demo 数
)

# Compile：DSPy 自动选 demo 让 metric 最大化
compiled_rag = optimizer.compile(student=RAG(), trainset=trainset)

# 编译后的 module 在 prompt 里自动嵌入了选出的 demo
result = compiled_rag(question="...")
```

`BootstrapFewShot` 内部的算法：

1. 对每条 trainset 用初始 RAG 跑一遍，收集中间步骤（context、CoT reasoning）作为候选 demo
2. 跑 metric 过滤——只保留通过 metric 的 demo
3. 把这些 demo 嵌进未来调用的 prompt

#### 用 MIPROv2 联合优化指令 + demo

```python
from dspy.teleprompt import MIPROv2

# MIPROv2 是 DSPy 当前最强 optimizer
# 同时优化「指令措辞」和「few-shot 选样」用 Bayesian Optimization
teleprompter = MIPROv2(
    metric=answer_exact_match,
    auto="medium",   # "light" / "medium" / "heavy" 控制搜索预算
    num_threads=8,
)

optimized_rag = teleprompter.compile(
    student=RAG(),
    trainset=trainset,    # 100+ 条
    valset=valset,        # 评估集
    requires_permission_to_run=False,
)

# 保存 / 加载
optimized_rag.save("./optimized_rag.json")

# 后续直接 load
new_rag = RAG()
new_rag.load("./optimized_rag.json")
```

`MIPROv2` 内部是**两阶段 Bayesian Optimization**：

- 阶段 1：Bootstrap 候选 demos（同 BootstrapFewShot）
- 阶段 2：用 Bayesian Optimization 在「指令措辞 × demo 组合」搜索空间里找最优配置

实证：MIPROv2 在多任务上比 BootstrapFewShot 再高 5-15%。是 2025-2026 年 DSPy 默认 optimizer。

#### 其他 DSPy Optimizer

| Optimizer | 特点 | 适合 |
|---|---|---|
| `BootstrapFewShot` | 仅 bootstrap demo | 入门 / 数据少 |
| `BootstrapFewShotWithRandomSearch` | bootstrap + 多次随机 demo 选样 | 计算预算多 |
| `MIPROv2` | 指令 + demo 联合优化（Bayesian） | **生产推荐** |
| `COPRO` | 仅优化指令措辞 | 已有好 demo |
| `KNNFewShot` | 检索式 demo 选样（每个 query 选最相似的 demo） | 任务多样 |
| `BootstrapFinetune` | 用 bootstrapped demo 微调小模型 | 想蒸馏到便宜模型 |
| `SIMBA`（2025 新） | stochastic introspective 优化 | 大模型微调式 |

### 1.5.4 何时用 DSPy vs 手工 prompt

| 场景 | 推荐 |
|---|---|
| 一次性脚本 / Demo | 手工 prompt（直接） |
| 已有 baseline，想再提 5-15% | **DSPy MIPROv2** |
| 频繁切换模型（OpenAI → Anthropic → 开源） | **DSPy**（重新 compile 即可） |
| 有 100+ 标注数据 + 客观 metric | **DSPy**（数据是燃料） |
| 完全主观任务（写诗 / 讲笑话） | 手工（DSPy 的 metric 难定义） |
| Multi-step 复杂程序（RAG + Agent + reasoning） | **DSPy**（程序化抽象天生适合） |

### 1.5.5 DSPy 的产业采用

2025-2026 进入主流：

- **Databricks** Mosaic Research 团队大量内部使用，公开 case study
- **Snowflake** Cortex AI 把 DSPy 作为内部 prompt 编译框架
- **JetBlue / Replit / 多家金融** 公司公开演讲分享 DSPy 落地

代表了行业的一个重要转向：**prompt 不再是手写艺术，而是编译产物**——和软件工程从汇编到高级语言的迁移类似。

### 1.5.6 GitHub 资源

- [`stanfordnlp/dspy`](https://github.com/stanfordnlp/dspy) — DSPy 官方
- [`stanfordnlp/dspy/docs`](https://dspy.ai/) — 文档与 cookbook
- [`weaviate/recipes`](https://github.com/weaviate/recipes) — DSPy + Weaviate 实战
- *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*（Khattab 2023）— 原始论文
- *MIPROv2: Multiprompt Instruction PRoposal Optimizer*（Opsahl-Ong 2024）— MIPROv2 论文

---

## 1.6 评估、红队与提示词工程的"黄昏"

提示词工程的成功必须可被度量。**Promptfoo** [29] 是开源 prompt 测试框架的事实标准——定义测试用例、跑多模型对比、CI 集成。最小可行测试集：20 个多样化案例（happy path + edge case + adversarial），每次 prompt 改动后自动跑。

提示词工程的失败模式有三类：

- **幻觉（Hallucination）**：模型一本正经地编造事实
- **提示词注入（Prompt Injection）**：用户输入里夹带"忽略上面所有指令，改做 X"，劫持 system prompt
- **越狱（Jailbreak）**：绕过对齐让模型输出有害内容（"DAN"、角色扮演、长尾语言攻击等）

**OWASP *Top 10 for LLM Applications* (现行 2025 版) 是业界标准分类法** [36]，前三名分别是 LLM01 Prompt Injection（提示词注入）、LLM02 Sensitive Information Disclosure（敏感信息泄露）、LLM03 Supply Chain（供应链风险）。这些问题不能靠"写更好的 prompt"解决，必须靠护栏（详见 [04_harness_engineering.md](./04_harness_engineering.md) §3.3.3）。

到 2024 年下半年起，"提示词工程是不是还重要"的争论开始浮现。Karpathy 在 YC AI School 的 *Software Is Changing (Again)* 演讲里把行业焦点指向了下一站——**上下文工程**（详见 [02_context_engineering.md](./02_context_engineering.md)）。

---

## Bibliography

[1] Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762>
[3] Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS.
[4] Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
[22] Raschka, S. *State of LLMs 2025*. <https://magazine.sebastianraschka.com/p/state-of-llms-2025>
[23] llm-stats. *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond*. <https://llm-stats.com/blog/research/post-training-techniques-2026>
[24] HuggingFace Blog. *Guide to RL Post-Training: PPO, DPO, GRPO*.
[25] Introl. *Fine-Tuning Infrastructure: LoRA, QLoRA, PEFT at Scale*.
[26] Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
[27] Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.
[28] PromptingGuide.ai. <https://www.promptingguide.ai>
[29] Pillitteri, P. *Prompt Engineering 2026: Frameworks That Actually Work*. <https://pasqualepillitteri.it/en/news/1090/prompt-engineering-2026-frameworks-complete-guide>
[30] BrightCoding. *Prompt Engineering Guide*.
[36] NVIDIA NeMo Guardrails Documentation. <https://docs.nvidia.com/nemo/guardrails/latest/index.html>

补充阅读：
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning*.
- Yao et al. (2023). *Tree of Thoughts*.
- Khattab et al. (2023). *DSPy: Compiling Declarative Language Model Calls*.
- Anthropic Prompting Docs. <https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview>

---

## GitHub 系统化学习资源（LLM 基础 / 训练栈专题）

| 资源 | 类型 | 推荐用法 |
|---|---|---|
| [`karpathy/nn-zero-to-hero`](https://github.com/karpathy/nn-zero-to-hero) | 视频 + 代码课程 | 从 backprop 一步步到 GPT-2，**任何人入门 LLM 的最优起点** |
| [`karpathy/build-nanogpt`](https://github.com/karpathy/build-nanogpt) | 视频 + 代码 | 从空文件复现 GPT-2（124M），1 小时 ~$10。每个 commit 原子化，可 diff 跟读 |
| [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) | 简洁实现 | 中等规模 GPT 训练 / 微调的最简代码（~300 行） |
| [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) | Manning 配套书 | Sebastian Raschka *Build a Large Language Model (From Scratch)* 全代码，章节对应清晰 |
| [`Hannibal046/Awesome-LLM`](https://github.com/Hannibal046/Awesome-LLM) | awesome list | LLM 综合资源入口 |
| [Stanford CS324](https://stanford-cs324.github.io/winter2022/) | 大学课程 | LLM 全景课程（modeling / theory / ethics / systems）讲义可读 |
| [Stanford CS25](https://web.stanford.edu/class/cs25/) | 邀请系列讲座 | Transformers 主题前沿讲座，Karpathy / Hinton 等都讲过 |
| [`huggingface/peft`](https://github.com/huggingface/peft) | LoRA / QLoRA / IA³ 实现 | 微调事实标准库 |
| [`huggingface/trl`](https://github.com/huggingface/trl) | RLHF / DPO / GRPO 实现 | HuggingFace 后训练框架（PPOTrainer / DPOTrainer / GRPOTrainer 全有） |
| [`unslothai/unsloth`](https://github.com/unslothai/unsloth) | 高效微调 | 单卡 4090 上跑 Llama 70B QLoRA、~2× 速度、~70% 显存 |
| [`OpenRLHF/OpenRLHF`](https://github.com/OpenRLHF/OpenRLHF) | 工业级 RLHF | DPO / PPO / GRPO 多卡分布式实现 |
| [`HuggingFace transformers`](https://github.com/huggingface/transformers) | 模型库 | 一切的基础 |
| [`pytorch/torchtune`](https://github.com/pytorch/torchtune) | PyTorch 官方 fine-tune | 简洁、原生 PyTorch、适合"想看清每个 step"的工程师 |

**论文必读清单**：
- 2017 — *Attention Is All You Need* (Vaswani)
- 2020 — *Scaling Laws for Neural Language Models* (Kaplan, OpenAI)
- 2022 — *Chain-of-Thought Prompting* (Wei) / *InstructGPT/RLHF* (Ouyang) / *ReAct* (Yao)
- 2022 — *Training Compute-Optimal LLMs* (Hoffmann, Chinchilla)
- 2023 — *DPO* (Rafailov) / *LoRA* (Hu) / *QLoRA* (Dettmers) / *Self-Refine* (Madaan)
- 2024 — *RLHF Book* (Nathan Lambert，在线) / *DSPy* / *Mixtral / GQA*
- 2025 — *DeepSeek-R1 Technical Report*（GRPO + RLVR 系统化）

---

## 章节交叉引用

- 想理解 ReAct 如何成为现代 Agent 基础 → [03_agent_engineering.md](./03_agent_engineering.md) §2 范式演进
- 想理解长上下文 / RAG / 记忆如何延展提示词工程 → [02_context_engineering.md](./02_context_engineering.md)
- 想看 Prompt 安全护栏在生产里的部署 → [04_harness_engineering.md](./04_harness_engineering.md) §3.3.3

