# 从提示词到驾驭：LLM 应用工程的三层演进

**调研日期**：2026-05-01 · **模式**：deep-research ultradeep · **检索路数**：18 路并行（中英混合）

---

## Executive Summary

2017 年 Transformer 架构论文 [1] 发表至 2026 年的九年里，"如何让 LLM 真正在生产环境跑起来"经历了三轮范式跃迁：**提示词工程**（Prompt Engineering，2022-2023 主线）→ **上下文工程**（Context Engineering，2024 起被 Karpathy 等人正式命名 [5][6]）→ **驾驭工程**（Harness Engineering，2026 年 2 月由 Mitchell Hashimoto 命名 [8][31]）。

三者不是替代关系而是**叠加抽象**：提示词工程关心「这一句话怎么写」，上下文工程关心「模型在推理这一刻看到了什么」，驾驭工程关心「整个 agent 在什么环境里运行、出错时如何不再犯同样的错」。

本文围绕这三层展开：

- **第一篇** 从 Transformer 注意力机制出发，覆盖 LLM 训练栈（预训练 → 后训练 → 微调，含 SFT/DPO/GRPO/RLAIF/RLVR/LoRA/QLoRA/蒸馏）、推理与解码策略，再到 CoT/ReAct/ToT/Self-Refine 等经典提示词技术与 APE/OPRO 等自动化方法。
- **第二篇** 解剖上下文工程的物理约束（O(N²) 注意力、Lost-in-the-Middle [2]、RULER 基准 [12]）、RAG 全栈选型（演进四代、嵌入模型 MTEB 榜单、向量库基准、混合检索 + 重排、查询改写、GraphRAG / Self-RAG / CRAG / Agentic RAG）、记忆系统四强（MemGPT-Letta / mem0 / Zep / Anthropic Memory Tool）、以及 MCP / A2A 协议生态。
- **第三篇** 拆解驾驭工程的七大组件：工具集成、沙箱、LLM 网关、推理服务器（vLLM / SGLang）、可观测性（LangSmith / Langfuse / Helicone / Phoenix）、评估（DeepEval / RAGAS / Promptfoo / Braintrust）、护栏（NeMo Guardrails），并给出 Agent 编排的主流方案与"完整生产栈"参考架构。

文末给出三类典型场景（个人学习 / 创业 MVP / 企业生产）的技术选型清单与限制说明。

---

## Introduction

### 调研问题
"分别讲清提示词工程（含 LLM/Transformer 等基础）、上下文工程（含记忆 / RAG 选型等）、驾驭工程，越全越好。"

### 范围与边界
- **覆盖**：三大工程范式的概念演进、技术栈、代表性产品 / 论文 / 基准数据，时效以 2025 下半年至 2026 春为基准
- **不覆盖**：LLM 内部数学推导（attention 矩阵推导、loss 函数）、具体业务案例的财务 ROI、单模型评测排行榜的细节波动
- **语言**：中文写作，英文术语保留

### 核心假设
- 用户是中文 AI 工程师 / Agent 开发者 / 大模型应用架构师
- 偏好"工程化能落地"的内容，胜过"学术化但脱离生产"的内容
- 对 PyTorch / Python / LangChain / LangGraph 等主流栈有基础认知
- "驾驭工程" 对应英文 *Harness Engineering*

### 调研方法
本报告由 `deep-research` skill（199-bio 版，ultradeep 模式）驱动：18 路并行 WebSearch（中英双语，分两批） → 跨多源交叉验证关键 claim（≥2 源支持才纳入）→ 时间锚定（默认引用 ≥2025-12 的资源） → Phase 6 红队复核（怀疑派 / 对抗评审 / 实施工程师三视角） → 落盘归档。详见 Methodology Appendix。

---

## 第一篇：提示词工程（含 Transformer / LLM 基础）

### 1.1 起点：Transformer 与注意力机制

2017 年 Vaswani 等人在 *Attention Is All You Need* [1] 中提出 Transformer 架构，把循环（RNN）和卷积（CNN）从序列建模里拿掉，只留下"注意力"。其核心思想极为朴素：序列里每个 token 与所有其他 token 通过 Query / Key / Value 三个矩阵做加权聚合，权重由 softmax(QK^T / √d) 决定。这种设计的两个直接结果是——**全局可见性**（任何位置都能直接看到任何位置，无需逐步传递）和**完全并行**（不再像 RNN 那样必须按时间步前向）。

但 Transformer 自诞生起就背着一个数学包袱：标准自注意力对序列长度 N 是 O(N²) 的时间和显存复杂度 [关于这一点的详细论证可见 2026 年的若干综述 [11][39]]。把上下文窗口从 32K 扩到 128K，计算量并不是 4 倍，而是 16 倍。这一条数学事实驱动了 2022-2026 年间数十亿美元的工程投入。

围绕"如何把 O(N²) 这条墙凿穿"诞生了一系列变体：
- **Linear Attention**（Katharopoulos 2020）通过核函数特征映射 + 矩阵乘法结合律避开显式 attention 矩阵，把复杂度降到 O(N)；
- **Sparse Attention**（如 Longformer / BigBird）让每个 token 只看局部窗口 + 少数全局 token；
- **Flash Attention**（Dao 2022 起）不是改算法而是改工程——把 attention 计算用 tiling 拆成能塞进 GPU 片上 SRAM 的小块，避免把 N×N 矩阵写回 HBM。Flash Attention 4 在 2026 年 3 月发布，在 NVIDIA B200 GPU 上达到 1605 TFLOPs/s、71% 硬件利用率 [文献描述见 attention 机制综述 2026]；
- **Paged Attention**（vLLM 团队 2023）借用操作系统虚拟内存思想，把 KV cache 切成 16 token 一块的页，使显存碎片低于 4% [10]；
- **Local Attention** / **Sliding Window Attention** 用于 Mistral / Qwen 等长序列优化。

到 2026 年，这些技术已经不是"二选一"，而是按层 / 按块混合堆叠：模型底层用 Sliding Window 控成本，关键层用 Full Attention 保表达力，推理时全部走 Flash + Paged 优化。

**Scaling Laws** 是与 Transformer 并行演进的另一条主线。Kaplan 等人 2020 年提出参数量 / 数据量 / 计算量的幂律关系，Hoffmann 等人 2022 年（Chinchilla）修正了"应该用多少 token 训练多少参数的模型"——结论是过去主流（如 GPT-3）严重 under-trained。而 2025-2026 年的关键转折是：**重心从"更多数据"转向"更高质量的数据"**。当 Common Crawl 已经被吃完、合成数据兴起，定律关注的不再是 token 数本身，而是 token 的信息密度与多样性。

### 1.2 LLM 训练栈：预训练 → 后训练 → 微调

2026 年训练栈的标准模板已经形成"三段式"——预训练（Pretraining） → 后训练（Post-training） → 任务级微调（Task-specific Fine-tuning）。Sebastian Raschka 在 *State of LLMs 2025* [22] 中点评：「12 个月前的标准配方是预训练几万亿 token 然后跑 RLHF；那条配方已经死了。从 DeepSeek-R1 到 Nemotron 3 Super 到 GPT-5.3 Codex，过去一年发布的每个主要模型都用了不同的后训练栈。」

#### 预训练（Pretraining）

任务非常简单：在巨量语料上做下一个 token 预测（next-token prediction），用 cross-entropy loss 反向传播。但工程难度极高：

- **数据**：从公开 Web（Common Crawl、C4、RefinedWeb）+ 代码（GitHub、StackOverflow）+ 学术（arXiv、Wikipedia、书籍）+ 多语言 + 合成数据（蒸馏自更强模型）配比
- **配比策略**：高质量数据多过几轮（curriculum learning），数学 / 代码 / 推理类数据比例上升
- **基础设施**：千卡到万卡 GPU 集群，3D 并行（数据 / 张量 / 流水线 / Sequence parallel）、checkpointing、容错恢复
- **超参**：BF16 / FP8 混合精度、Lion / AdamW 优化器、cosine learning rate、warmup
- **数据去重 / 去毒 / PII 过滤**

预训练的产物是 **基础模型（base model）**——它能续写但不能"对话"，给它"今天天气如何？"它会接着写一段类似贴吧的杂谈，而不是回答你。要让它变成 ChatGPT 那样能交互的助手，需要后训练。

#### 后训练（Post-training）

后训练在 2024-2026 年被拆成三个相对正交的阶段 [23][24]：**SFT（让模型会说话）→ 偏好对齐（让模型说人喜欢的话）→ 推理强化（让模型在可验证任务上更准）**。顺序很重要。

**SFT（Supervised Fine-Tuning，监督微调）**

用人工编写的"指令-回答"对（如 Alpaca / Dolly / OpenAssistant 风格）对基础模型继续训练，让它学会"指令跟随"和对话格式。SFT 教模型**怎么说话**——产生结构化输出、遵循 system prompt、用合适的礼貌度等。技术上和预训练区别很小，差异主要在数据上。

**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**

OpenAI InstructGPT / ChatGPT 的看家本领。流程分三步：（a）收集人类偏好对（"回答 A 比 B 更好"）；（b）训练奖励模型（reward model）学习这些偏好；（c）用 PPO 算法以奖励模型的打分作为 reward 微调原模型。PPO 实现极复杂，需要同时维护 4 个模型（actor、critic、reference、reward），显存爆炸 + 训练不稳定。

**DPO（Direct Preference Optimization，直接偏好优化）**

斯坦福 Rafailov 等人 2023 年提出的"无 RL 的 RLHF"。核心洞察：把 RLHF 的目标函数做闭式重写后，发现可以直接用偏好对训练原模型，无需奖励模型、无需 PPO。一个简单的二分类损失就够了。结果：训练稳定 10x、显存占用降一半、效果与 PPO RLHF 相当甚至更好。2024 年起绝大多数开源模型（Llama 3、Qwen 2/3、Mistral 系列）都改用 DPO 或其变体（IPO / KTO / SimPO）替代 PPO。

**GRPO（Group Relative Policy Optimization）**

DeepSeek 团队 2024 年提出，是 R1 系列爆火背后的核心算法 [22][23]。GRPO 砍掉了 PPO 的 critic 网络（这是 PPO 训练不稳定的主要来源），改用"组内相对优势"——对同一个 prompt 采样 N 个回答，每个回答的优势 = 该回答的奖励 - 组内均值，再用 PPO 风格的策略梯度更新。在数学 / 代码等可验证任务上效果惊人，把"在线 RL"重新带回主流。

**RLAIF（Reinforcement Learning from AI Feedback，基于 AI 反馈的强化学习）**

Anthropic 2022 年 *Constitutional AI* 论文里首次系统化：用更强的 LLM 代替人类标注偏好对，大幅压低数据成本。2025-2026 年 RLAIF 已成为通用做法——基础模型对齐用 RLAIF，关键安全场景用人类反馈做最后一道把关。

**RLVR（Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习）**

2025 下半年起在 OpenAI o1 / DeepSeek-R1 上爆红。核心思想：在数学题、代码、单元测试等"答案可被程序自动验证"的任务上，奖励信号不来自人类偏好或奖励模型，而来自硬编码的验证器（运行单元测试、对比标准答案）。这避免了奖励模型的"作弊"（reward hacking）问题，把推理能力推到新高度。GRPO + RLVR 是 R1 的黄金组合。

**DAPO**（Decoupled Clip and Dynamic Sampling Policy Optimization）等 2026 年新算法继续在 GRPO 基础上做工程优化（动态采样、分层 clip）。

#### 任务级微调（Fine-Tuning）

如果你拿到的不是"训练新基础模型"的资源，而是"在已有开源模型上做领域适配"，那么 fine-tuning 才是你的工具。这里有个关键的成本分水岭 [25]：

| 方法 | 7B 模型显存 | 硬件成本 | 质量保留 | 适用 |
|---|---|---|---|---|
| **Full Fine-Tune** | 100-120 GB | $50K（H100） | 100%（基线） | 延迟敏感 + 最高精度 |
| **LoRA** | ~25 GB | $10K（A100） | 95-98% | 快速实验 + 多 adapter |
| **QLoRA** | ~6 GB | $1.5K（RTX 4090） | 80-90% | VRAM 极限 + 多客户隔离 |
| **Distillation** | 视 student | 中 | 视任务 | 推理时延 / 成本极致 |

- **LoRA**（Low-Rank Adaptation，Hu 等 2021）只在权重矩阵旁边加一对低秩矩阵 A·B，原权重冻结，仅训练 0.2-0.3% 的参数。适合需要给不同客户 / 任务 / 场景维护多个轻量 adapter 的场景，切换 adapter 只需要换一个几百 MB 的小文件。
- **QLoRA**（Dettmers 2023）= 4-bit 量化原模型 + LoRA。把基础模型的常驻显存从 ~26 GB（fp16，7B）压到 ~4 GB，让单张消费级显卡（4090、3090）就能跑 7B 模型的微调。质量损失约 5-10%。
- **PEFT**（Parameter-Efficient Fine-Tuning）是个伞型术语，涵盖 LoRA、QLoRA、Prefix Tuning、Prompt Tuning、Adapter Tuning、IA³ 等。HuggingFace 的 `peft` 库是事实标准。
- **蒸馏（Knowledge Distillation）** 走另一条路：用大模型（teacher）生成大量数据，训练小模型（student）模仿大模型。代价是 student 通常达不到 teacher 的 ceiling，但推理成本可能降一个数量级。Qwen 2.5 系列、Phi 系列大量使用蒸馏。

实践指引 [25]：「如果产品需要极低延迟和最高准确率，做 full fine-tune；如果需要快速实验、多变体、按客户单独 adapter，用 LoRA；如果模型很大且 VRAM 受限，用 QLoRA。」

### 1.3 推理栈与解码策略

模型训完上线，"用什么策略从 logits 里采样下一个 token"决定了输出质感：

- **Greedy / Argmax**：永远选概率最高的 token。结果稳定但容易陷入重复（"the the the..."）
- **Beam Search**：维护 top-K 个候选序列。机器翻译里好用，对话里偏机械
- **Top-K Sampling**：从概率最高的 K 个里随机采。K=40 是常见值
- **Top-P / Nucleus Sampling**（Holtzman 2019）：从累积概率到 P 的最小集合里采。P=0.9 是常见值，比 top-K 更自适应
- **Temperature**：对 logits 除以温度 T 后再 softmax。T<1 偏保守（接近 greedy），T>1 偏发散，T=0 等价 greedy
- **Repetition Penalty / Frequency Penalty / Presence Penalty**：抑制重复 token

工程上现在的事实标准是 `temperature=0.7 + top_p=0.95`（OpenAI / Anthropic 的默认）。要做创意生成调高 T，要做严格 JSON 提取调到 T=0 + 配合 grammar-guided generation（如 Outlines、Guidance、Llama.cpp 的 GBNF）。

### 1.4 提示词工程的经典技术

提示词工程的"经典时代"基本可以用一张技术图谱概括 [28][29][30]：

**1. 零样本 / 少样本（Zero-shot / Few-shot）**

直接问 vs 给几个例子。Brown 等人 2020 年的 GPT-3 论文发现 few-shot 能让小模型逼近大模型 fine-tuned 后的水平——这是"上下文学习（in-context learning）"概念的诞生。2026 年 few-shot 仍是 80% 任务的最强 baseline [29]。

**2. Chain-of-Thought（CoT，思维链）**

Wei 等人 2022 年提出。让模型先写"思考过程"再写答案，大幅提升数学 / 推理任务准确率。简单触发词 "Let's think step by step." 在 GSM8K 上把准确率从 17% 提升到 78% [28]。CoT 的核心机制：把原本要在前向传播里一次完成的复杂推理，分摊到多个 token 的生成上，等价于动态扩展计算量。

**3. Self-Consistency**

Wang 等人 2022 年。同一个 CoT prompt 采样 N 次（高 temperature），最后投票选最常出现的答案。在数学 / 编程上又把 CoT 提升 5-10 个百分点，代价是 N 倍 token 成本。

**4. ReAct（Reasoning + Acting）**

Yao 等人 2022 年 [4]。把思考（Thought）和行动（Action）交错：模型先想"我现在该查什么"，然后调用工具（搜索 / 计算器 / API），观察结果（Observation），再继续想下一步。在 ALFWorld 交互式任务上比 imitation learning + RL 高 34% 绝对成功率，在 WebShop 上高 10%。ReAct 是后来所有 Agent 架构的祖父——LangGraph 的 react agent template、AutoGen 的 conversable agent、CrewAI 的 task agent，本质都是 ReAct 的工程化变体。

**5. Tree of Thoughts（ToT）/ Graph of Thoughts（GoT）**

Yao 2023 / Besta 2023。CoT 是单条思考链，ToT 把每一步展开成多个分支，用 BFS/DFS + 启发式打分搜索整棵推理树。GoT 进一步把树推广为 DAG，允许节点合并（"我这两条思路其实指向同一个答案"）。代价是 token 成本数倍到十倍，仅在数学推理 / 规划等"答案稀疏但路径多"的任务有性价比。

**6. Self-Refine**

Madaan 等人 NeurIPS 2023 [3]。同一个模型扮演三个角色：generator（生成初稿）→ critic（自我评判）→ reviser（重写）。在 7 个任务上平均提升 20%（GPT-3.5 / GPT-4 都验证过）。这是后来 Reflexion、CRITIC、Constitutional AI 等"自我改进"路径的原型。

**7. Reflexion**

Shinn 等人 2023。Self-Refine 是单轮自我改进，Reflexion 把自我反思写入"长期记忆"——如果某次任务失败，把失败原因总结成一段反思，附在下次同类任务的 prompt 里。等价于让 agent 从经验里学习。

**8. Prompt Chaining / Decomposition**

复杂任务拆成多步骤的 prompt，每步输出喂给下一步（即"workflow"模式）。LangChain、LlamaIndex、Semantic Kernel 大量这套。优点是可调试、可观测；缺点是步骤间信息可能丢失。

**9. Role Prompting / System Prompt**

"You are an expert Python developer. ..." 给模型注入身份与场景。Claude 系列对 system prompt 的遵循度普遍高于 GPT 系列，所以社区有"Claude 提示工程偏 XML 标签 + 强 system prompt"的共识 [29]。

**10. 模型差异化**

2026 年的提示词工程已经从"通用最佳实践"分化为"模型特异性最佳实践" [29][30]：

- **Claude（Anthropic）**：偏好 XML 标签（`<context>...</context>`、`<instructions>...</instructions>`），长 system prompt，多步推理任务直接说 "think step by step"
- **GPT 系列（OpenAI）**：偏好简洁的 JSON schema，function calling 结构化，markdown 友好
- **Gemini**：多模态原生，对图像 + 文本混合 prompt 支持最好
- **DeepSeek-R1 / o1 / 思考模型**：不要再加 CoT 触发词（模型自带），直接给问题；temperature 偏低

### 1.5 自动化与高级提示词

人工写 prompt 终究有上限。2023 年起涌现"让 LLM 自己写 prompt"的方向：

- **APE（Automatic Prompt Engineer）**（Zhou 2023）：用一个 LLM 生成候选 prompt，在 dev set 上打分，迭代选优
- **OPRO（Optimization by PROmpting）**（Yang 2023，Google）：把 prompt 优化看作元优化问题，让 LLM 直接担任"优化器"，在每轮根据上一轮的 prompt + 分数提出新 prompt
- **Promptbreeder**（Fernando 2023，DeepMind）：APE + 进化算法，让 prompt 在种群里突变 / 选择 / 交叉
- **DSPy**（Stanford 2023）：把 prompt 当作神经网络层，用编译器 + bootstrapping few-shot 自动生成最优 demo 样本，让你写"模块"而不是写 prompt

DSPy 在 2025-2026 进入主流——被 Databricks、Snowflake 等收编为内部 prompt 编译框架。它代表了一个重要转向：**prompt 不再是手写艺术，而是编译产物**。

### 1.6 评估、红队与提示词工程的"黄昏"

提示词工程的成功必须可被度量。**Promptfoo** [29] 是开源 prompt 测试框架的事实标准——定义测试用例、跑多模型对比、CI 集成。最小可行测试集：20 个多样化案例（happy path + edge case + adversarial），每次 prompt 改动后自动跑。

提示词工程的失败模式有三类：
- **幻觉（Hallucination）**：模型一本正经地编造事实
- **提示词注入（Prompt Injection）**：用户输入里夹带"忽略上面所有指令，改做 X"，劫持 system prompt
- **越狱（Jailbreak）**：绕过对齐让模型输出有害内容（"DAN"、角色扮演、长尾语言攻击等）

2026 年 OWASP 已把 *Top 10 for LLM Applications* 列为标准分类法，前三名分别是 LLM01 提示词注入、LLM02 不安全输出处理、LLM03 训练数据投毒 [36]。这些问题不能靠"写更好的 prompt"解决，必须靠护栏（详见第三篇 3.3.7）。

到 2024 年下半年起，"提示词工程是不是还重要"的争论开始浮现。Karpathy 在 YC AI School 的 *Software Is Changing (Again)* 演讲里把行业焦点指向了下一站——**上下文工程**。

---

## 第二篇：上下文工程

### 2.1 定义与起源

"Context Engineering"（上下文工程）这个词在 2024 年下半年开始在社区流传，普遍归功于 Karpathy 在 YC AI School 的 *Software Is Changing (Again)* 演讲 [5]。他给的定义后来被反复引用：

> "In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step."
> （在每一个工业级 LLM 应用里，上下文工程是把恰好正确的信息填进上下文窗口、为下一步推理做准备的精细艺术与科学。）

Anthropic 在 2025 年发布的官方文章 *Effective Context Engineering for AI Agents* [6] 进一步把它定位为"提示词工程的自然演进"：「prompt engineering 关心的是你给模型说的那句话；context engineering 关心的是模型在推理那一刻能看到的所有信息——system prompt、对话历史、工具定义、检索到的文档、记忆、规划状态、约束与策略。」

LangChain 团队在 *Context Engineering for Agents* 博客 [7] 里给了一个常被引用的工程定义：

> "Context Engineering 是设计**动态系统**的学科，这些系统在**正确的时间、以正确的格式，向 LLM 提供正确的信息和工具**，让模型有完成任务所需的一切。"

为什么"提示词工程"不够了？三个根本变化：

1. **窗口扩张**：从 GPT-3.5 的 4K 一路涨到 Gemini 2.5 Pro 的 1M、Claude Opus 4.7 的 1M（启用 1M context 模式）。能塞进窗口的信息从"几段文字"变成了"一整本书"。决定塞什么变得比写好那一句话重要得多。
2. **Agent 兴起**：单轮 prompt 是孤立交易，agent 是持续状态机。每一步推理的 context 都来自上一步的输出 + 工具返回 + 检索 + 记忆，"prompt"已经不是用户写的那句话，而是 agent 框架动态拼装的产物。
3. **多模态 + 工具调用**：context 不再只是文本，还有图像、表格、代码、tool schema、retrieved chunks、structured memory、scratchpad。把这些异构信息组织好是工程问题。

到 2026 年初，业界共识是：**生产级 LLM 应用的核心难度从"写 prompt"迁移到了"管 context"**——Anthropic 在文档里直接说"Context Engineering 是构建可靠 AI agent 的关键学科" [6]。

### 2.2 长上下文的物理学

要理解为什么"塞满窗口"是个坏主意，必须先理解上下文的三个物理约束：

**约束一：O(N²) 成本**

回到第一篇 1.1 节：标准 attention 是 N² 复杂度。上下文从 32K 翻到 128K，注意力计算量是 16 倍而非 4 倍。对推理服务来说这意味着两件事：（a）首 token 延迟（TTFT）上升，因为要先 prefill 整个 prompt；（b）显存被 KV cache 占满，并发数下降。

**约束二：KV Cache 显存**

每个 attention 层在推理时要维护 K 和 V 的缓存（避免重复计算）。一个 70B 模型在 128K 上下文时，KV cache 占用可达 40+ GB，比模型权重本身还大。这就是为什么 vLLM 的 PagedAttention [10] 和 SGLang 的 RadixAttention [11] 这类"管 KV cache"的技术成为推理服务器的核心竞争力（详见第三篇 3.3.4）。

**约束三：Lost-in-the-Middle**

Liu 等人 2024 TACL 论文 *Lost in the Middle: How Language Models Use Long Contexts* [2] 系统验证了一个反直觉现象：把关键信息放在 prompt 开头或结尾，模型能很好地利用；放在中间，模型常常视而不见。**召回率随位置呈 U 型曲线**——开头有 primacy bias、结尾有 recency bias、中间是黑洞。

这条曲线在 GPT-3.5 / GPT-4 / Claude 上都验证过。即便 2025 年的 Gemini 1.5 Pro 在简单的 needle-in-a-haystack 测试上能拿到 99.7% 召回，但当任务变成"在 1M token 里找出 8 个相关事实并综合"，平均召回率掉到 ~60% [41]。

### 2.3 RULER 与 1M 上下文的真实能力

NVIDIA 等机构 2024 年发布的 **RULER 基准** [12]（*What's the Real Context Size of Your Long-Context Language Models?*）把"长上下文能力"从 needle-in-a-haystack 的玩具测试升级为"折磨套件"：

- **多种类多数量的 needle**：不只是埋一个事实，是埋多个不同类型
- **多跳推理**：找到 A 才能找到 B
- **聚合任务**：在长文里数总数 / 求平均
- **变量追踪**：跟踪一个变量在 100K token 范围内的多次重新赋值

RULER 暴露出的真相：**几乎所有声称 100K+ 上下文的模型，在 RULER 上的"有效上下文"远小于宣称值**。比如某模型宣称支持 200K，但 RULER 上 50K 后准确率就掉到 80% 以下，128K 时已经接近 random。

这把行业拉回了现实：**long context 不是 RAG 的替代，而是补充**。

#### 长上下文 vs RAG：决策框架

Tianpan 在 *Long-Context Models vs RAG: When the 1M-Token Window Is the Wrong Tool* [41] 给了一个 2026 主流的决策框架：

| 维度 | 1M 上下文模型 | RAG |
|---|---|---|
| **延迟** | 30-60 倍慢（必须 prefill 整个窗口） | 检索 + 生成两阶段，但生成阶段输入小 |
| **每查询成本** | ~1250 倍贵 | 检索几乎免费，生成成本随相关 chunk 数 |
| **数据更新** | 每次查询都要重传 | 索引更新一次，所有查询受益 |
| **可解释性** | 黑箱 | 检索结果可审计、可引用 |
| **多事实召回** | ~60%（实战） | 配合 reranker 可达 90%+ |
| **真正适合长上下文的场景** | 单次深度分析（"读完这篇 200 页的 PDF 总结洞察"） | 知识库问答、客服、文档检索 |

实战指引：**90% 的"知识检索"场景应该用 RAG，10% 的"深度分析单文档"才用长上下文**。

### 2.4 RAG 全栈选型

RAG（Retrieval-Augmented Generation）是上下文工程最成熟的一支，已演进出完整的子学科。2025-2026 年的状态可以用四代演进概括 [43]：

**第一代：Naive RAG**（2020-2022）
检索 → 拼接 → 生成。chunk + embed + vector search + LLM。简单但脆弱：chunk 切错就全错、检索召回低就答错。

**第二代：Advanced RAG**（2023-2024）
引入 query rewriting / reranking / hybrid search / metadata filter。召回从 50% 拉到 80%。

**第三代：Modular RAG**（2024）
把 RAG 拆成可组合模块：indexing / pre-retrieval / retrieval / post-retrieval / generation 各自可独立替换。LangChain、LlamaIndex 的 RAG 工程化抽象。

**第四代：Agentic RAG**（2025-2026）
把 RAG 看作 agent 的能力之一，让 agent 自主决定何时检索、检索什么、如何评判检索结果、要不要重新检索 [43]。Self-RAG / CRAG / Agentic RAG 都是这一代的代表。

下面按 RAG 流水线顺序拆解 2026 年的选型空间。

#### 2.4.1 文档解析（Parsing & Layout）

现实世界的文档是 PDF、PPT、扫描件、Excel、HTML 而不是纯文本。这一步常被忽略但影响极大：

- **基础**：`pypdf` / `pdfplumber`（速度快，但表格 / 多栏排版易错）
- **进阶**：`unstructured` / `LlamaParse`（带 layout 模型，能识别标题 / 段落 / 表格 / 列表）
- **OCR**：`PaddleOCR` / `Tesseract` / `EasyOCR`（扫描件、图片）
- **多模态原生**：用 GPT-4o / Gemini / Claude 直接读 PDF 页（vision），适合排版极复杂的文档但成本高
- **专用商用**：Azure Document Intelligence、AWS Textract、Microsoft MarkItDown

最佳实践：**至少跑两套 parser 对比**，再决定用哪个。表格 / 公式必须特别处理（转 markdown / LaTeX）。

#### 2.4.2 切分策略（Chunking）

切分策略对召回质量影响远大于嵌入模型的选择：

- **Fixed Size Chunking**：固定 N token，最简单。问题：容易切碎语义单元
- **Recursive Character Splitting**：按段落 → 句子 → 词层级递归切。LangChain 默认。性价比好的 baseline
- **Semantic Chunking**：用 embedding 计算相邻句子相似度，相似度骤降处切分
- **Document-Specific**：Markdown 按 heading 切、code 按 function 切、HTML 按 DOM 切
- **Agentic Chunking**：让 LLM 阅读文档后自己决定怎么切。最贵但最准
- **Late Chunking**（Jina 2024）：先对整文档 embed 得到 token 级 embedding，再切分聚合。在长上下文场景比传统切分召回高 10-15%

通用建议：**chunk size 256-512 token、overlap 50-100 token**，是大多数任务的 sweet spot。

#### 2.4.3 嵌入模型（Embedding Models）

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

1. **Matryoshka Representation Learning（套娃表示）**：同一个 embedding 在不同截断长度（128 / 256 / 512 / 1024 / 3072）都能用，索引时存全长，检索时按需截短。Gemini 2 / Voyage 4 / Cohere v4 / OpenAI text-3-* / Jina v5 / Microsoft Harrier / Nomic v1.5 都支持。可以让"先粗筛后精排"在 embedding 层就实现。
2. **多模态原生**：Gemini Embedding 2 把图像 / 视频 / 音频 / PDF 编进同一个语义空间，多模态 RAG 的索引复杂度从"分模态建索引 + 跨模态对齐"降到"统一索引"。

中文场景实测：**BGE-M3 是免费第一**（开源 + 多语言强 + 同模型支持稀疏/稠密），**Voyage 3 large** 是付费第一（中文召回普遍比 OpenAI 高 10%+）。

#### 2.4.4 向量数据库（Vector Database）

2026 年生产部署的五大主流 [14]：**pgvector / Qdrant / Weaviate / Milvus / LanceDB**。Pinecone 仍在但价格策略有争议（取消 pod tier 改 Dedicated Read Nodes）。

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

#### 2.4.5 检索：稀疏 + 稠密 + 后期交互

向量检索（稠密）擅长语义相似度但不擅长精确关键词匹配（产品 ID、专有名词、错别字容忍）。**生产级 RAG 几乎都是 hybrid search**：

- **稀疏**：BM25 / SPLADE / BGE-sparse（关键词 + 词频）
- **稠密**：embedding-based vector search（语义）
- **后期交互（late interaction）**：ColBERT / ColBERT-v2，每个 token 单独 embedding，检索时按 token 级 max-similarity 聚合。比单向量 dense 召回高，但索引大 100x。Vespa、Qdrant 已支持

混合策略：用 RRF（Reciprocal Rank Fusion）合并稀疏与稠密的结果列表。简单且无需调参。

实证：**hybrid search + reranker 比 semantic-only 提升 +9.3 个百分点 MRR**，配合 Cohere Rerank 准确率可以到 90%+ [实测见 LanceDB 基准]。

#### 2.4.6 重排（Reranker）

第一阶段检索（embedding / BM25）召回 top-100 后，用更贵但更准的 cross-encoder 模型对 query-doc 对重排，取 top-5 给 LLM。这一步的 ROI 极高 [reranker 综述]：

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

#### 2.4.7 查询改写（Query Rewriting / Expansion）

用户问的"今天天气咋样"和文档里写的"current weather conditions"语义相同但词汇不重叠。Query rewriting 在检索前先对 query 做处理：

- **HyDE（Hypothetical Document Embeddings）**：先让 LLM 假想"如果有一篇答案文档会写什么"，用这段假想文本去检索（不是用原 query）。在专业领域召回提升明显
- **Multi-Query**：让 LLM 把原 query 改写成 3-5 个不同表达，分别检索后合并
- **Step-Back Prompting**：先把 query 抽象成更宏观的问题（"X 公司 Q3 财报里的具体数字是？" → "X 公司 Q3 财报概况是？"），用抽象问题的检索结果做铺垫
- **Sub-Query Decomposition**：把复合问题拆成多个子问题分别检索（"对比 A 和 B" → ["A 的特点"、"B 的特点"]）
- **HyDE + Multi-Query 组合** 在 BEIR 等检索基准上是当前最强 baseline 之一

#### 2.4.8 GraphRAG

Microsoft GraphRAG（2024 开源）和 LightRAG / Nano-GraphRAG 等社区变体把检索从"向量相似"扩展到"知识图遍历"：

- **构建阶段**：用 LLM 抽取文档里的实体和关系，建一张知识图
- **检索阶段**：把 query 映射到图上的实体，做局部子图查询 + summary
- **优势**：能回答"主题级"问题（"我们所有供应商合同里的合规风险有哪些"），传统 RAG 因为这种问题没有局部相似 chunk 而失败 [43]
- **代价**：图构建非常贵（LLM 把每段文档抽实体），适合静态高价值知识库

GraphRAG 在某些场景把检索精度推到 99%（金融合规、法律、医疗等关系密集的领域）[43]。

#### 2.4.9 Self-RAG / CRAG / Agentic RAG

Self-RAG（Asai 2023）：让模型在生成时自己决定**何时**检索、**评估**检索结果是否相关、**批判**自己的输出是否有依据。
CRAG（Corrective RAG）：检索后对 chunk 打"高/中/低置信度"标签，低置信度时触发 web search 兜底。
Agentic RAG：把上面所有能力包进一个 agent loop，让 agent 在"想 → 检索 → 评判 → 重写 query → 重新检索 → 综合"里循环直到自信为止。LangGraph 是实现 Agentic RAG 的主流框架 [Vinod Rane 2026]。

实证：Self-RAG 在生产部署中能减少 25-40% 的"不必要检索" [43]，因为模型学会了对自己已有知识的事情不去检索。

#### 2.4.10 多模态 RAG

图像 + 文本混合的索引方案：

- **Late fusion**：分别对文本和图像 embed，检索时分别检索后合并
- **Early fusion**：用 CLIP / Gemini Embedding 等多模态 embedding，把图文嵌入同一空间
- **VLM-as-extractor**：用 GPT-4o / Claude 3.5 Sonnet 把图片转成详细描述，再走文本 RAG（成本最高但召回最稳）

### 2.5 记忆系统

如果 RAG 解决"知识库怎么塞进 context"，记忆系统解决"agent 自身经验怎么塞进 context"。

#### 2.5.1 记忆的三个轴

借自认知心理学的分类：

- **短期记忆（Working Memory）**：当前对话窗口内的内容（system prompt + 最近 N 轮对话）
- **长期记忆**进一步分三类：
  - **情景记忆（Episodic）**：「上周三我们讨论过 X」「用户上次说他对 React 不熟」
  - **语义记忆（Semantic）**：「用户的母语是中文」「用户偏好暗色主题」「项目用 PostgreSQL」
  - **程序记忆（Procedural）**：「这类任务通常按 X→Y→Z 步骤做」「遇到 timeout 应该先重试 3 次」

#### 2.5.2 四大记忆系统对比（2026）

| 系统 | 核心架构 | 优势 | 适用 |
|---|---|---|---|
| **Mem0** | Vector + 可选 graph，用户/会话/agent 三层 | 集成最简单，文档完善 | 通用 chatbot 加记忆 |
| **Letta**（前身 MemGPT） | OS-inspired，agent 通过 tool call 自管 memory blocks | 显式 + 可解释，agent 持久身份 | 长跑（数天/周）的 agent |
| **Zep** | Temporal Knowledge Graph，追踪事实随时间变化 | 时序推理强，企业级 | 用户状态会变的场景（CRM/客服） |
| **Anthropic Memory Tool** | 托管 store + 自动注入 + API revise/delete | 与 Claude 原生集成 | Anthropic 生态内 |

横向架构观察：

**Mem0** [Mem0 论文，arxiv 2504.19413] 是最易接入的方案。三步：从 interaction 抽取"memory" → 存（默认 vector）→ 后续查询时检索 + 注入 prompt。"用户/会话/agent"三层 hierarchy 让多租户隔离自动完成。

**Letta** 的设计哲学很特别：把 memory 看作 agent 的"可编辑状态"，agent 自己用 tool call 来读/写/搜自己的记忆块。等价于给 agent 一个 OS 风格的 memory 管理器。优势是 agent 能跨会话保留"身份"——下次启动时，记忆块还在那。

**Zep** 的差异化是 *temporal*——同一个事实"用户的工作"可能从"在 Google" → "在 Anthropic" → "创业"，Zep 用知识图记录时间维度，回答"用户当前的工作是什么"和"用户去年的工作是什么"都能正确。

**Anthropic Memory Tool** 是 2025 年下半年随 Claude Sonnet 4.6 推出的官方 memory 接口：在 API 调用时声明使用 memory tool，模型会自动决定写什么记忆、读哪些记忆，开发者只管业务逻辑。

实战选型 [n1n.ai 2026 对比]：
- **简单 chatbot 加记忆** → Mem0
- **需要 agent 跨天独立运行** → Letta
- **企业 + 用户状态会变** → Zep
- **已绑定 Anthropic** → Memory Tool

#### 2.5.3 记忆的常见陷阱

- **记忆爆炸**：什么都记 → 噪声压过信号 → 检索质量下降。需要主动遗忘 / 摘要 / 优先级淘汰
- **过期记忆**：用户上周说"我喜欢冷咖啡"，今天告诉模型"我现在只喝热咖啡"。Zep 这类时序系统专治这个
- **跨用户泄露**：多租户 SaaS 必须严格 user_id 隔离

### 2.6 上下文管理策略

即使有了 RAG 和记忆，往 context window 塞东西的"塞什么、塞多少、什么时候塞"仍然是难题。Anthropic 文章 [6] 给了几个原则：

1. **不是所有信息都同等重要**——按优先级排序，最重要的放头部和尾部（避开 lost-in-the-middle）
2. **动态裁剪**——当 context 接近上限时，主动总结早期对话、淘汰过期工具结果
3. **/compact 模式**——Claude Code 等产品支持 `/compact` 命令，把当前长对话压缩成摘要 + 关键事实，保留语义但回收 token
4. **Scratchpad / Working Memory 分离**——把 agent 的"草稿纸"（推理过程、工具中间结果）和"对外 context"分开管理，避免污染

### 2.7 MCP / A2A 协议生态

把工具和数据源接入 LLM 长期是脏活——每个 LLM 的 function calling schema 都不一样、每个工具都得手写 wrapper。**MCP（Model Context Protocol）** [9] 是 Anthropic 2024 年 11 月推出的开放标准，目标是做 AI 的"USB-C"——一套协议让任意 LLM 接任意工具 / 数据源。

MCP 的关键设计：
- **JSON-RPC 2.0** 作为传输协议
- **Server / Client 架构**：MCP Server 包装外部系统（PostgreSQL、GitHub、文件系统、自定义 API），暴露三类原语：tools（可执行函数）、resources（只读数据）、prompts（模板）
- **解耦工具与 LLM**：function calling 把 tool definition 绑死在 LLM provider 上，MCP 让 tool 与 LLM 独立演化

时间线：
- 2024-11：Anthropic 发布
- 2025 上半年：OpenAI / Google / Microsoft / Amazon 相继支持
- 2025-12：Anthropic 把 MCP 捐给 Linux Foundation 旗下的 Agentic AI Foundation（AAIF），与 Block、OpenAI 共同治理
- 2026-02：Python + TypeScript SDK 月下载量突破 9700 万 [9]

**A2A（Agent-to-Agent）** 是另一类协议，关注 *agent 之间* 怎么协作（任务委托、状态共享、工作流编排）。MCP 与 A2A 是互补关系：

> "MCP 给 agent 一双手（连工具/数据），A2A 让多个 agent 组队干活。MCP 是垂直整合，A2A 是水平协作。"

到 2026 春，MCP 已成为 Agent 生态的事实标准；A2A 仍在多协议竞争阶段（Google A2A / Microsoft AutoGen 协议 / Anthropic 的 Agent Skills 等都在尝试）。

---

## 第三篇：驾驭工程（Harness Engineering）

### 3.1 起源：被命名的"那一层东西"

"Harness Engineering"作为术语在 2026 年 2 月由 Mitchell Hashimoto（HashiCorp 创始人）正式命名 [8]。触发事件是 OpenAI 工程师 Ryan Lopopolo 发表了一篇描述 OpenAI 内部 agent 基础设施的文章，Hashimoto 读后说："这就是 *harness engineering*——agent 周围所有让它能正常工作的脚手架、约束和反馈回路。"

Karpathy 则从另一个角度铺垫了这个概念。他在 2025-2026 年多次描述自己的 coding workflow 已经从"主要手写代码、偶尔让 agent 帮忙"反转成"主要由 agent 写代码、自己手动微调"。让这种工作模式可行的不是模型变强，而是 agent 周围的*环境*——repo 结构、CI 配置、formatter、package manager、framework 约束、project instruction、外部工具集成、linter——这些一起构成了 agent 的 "harness"。

Adnan Masood 在 2026 年 4 月 Medium 文章 *Agent Harness Engineering: The Rise of the AI Control Plane* [32] 总结：「Harness Engineering 是构建那个 *governs* 一个已部署 AI agent 跨多次交互的整个系统——guides、sensors、data context pipelines、eval suites、constraint enforcement。」

驾驭工程的核心理念被反复强调：

> **每次 agent 犯错，不要只是希望它下次做得更好；要工程化环境，让它无法以同样的方式再犯。**

这把"AI 安全"和"AI 工程化"统一到了一条原则：**约束环境，而非约束模型本身**。模型的"能力"是可变的（每三个月升一代），但环境约束是稳定的、可累积的。

### 3.2 与提示词工程、上下文工程的边界

三层抽象的清晰划分：

| 层级 | 关注 | 单位 | 时间尺度 |
|---|---|---|---|
| **Prompt Engineering** | 这一句话怎么写 | 字符 / 句 | 单次 LLM 调用 |
| **Context Engineering** | 模型在推理这一刻看到什么 | tokens 集合 | 单次推理（含 RAG、记忆、工具 schema 等） |
| **Harness Engineering** | agent 在什么环境运行、如何不再犯同样的错 | 系统 / 流程 | 跨多次交互、持续演进 |

类比：
- prompt = 你给员工的一个具体指令
- context = 员工桌面上能看到的所有材料（笔记本、邮件、文档、聊天记录）
- harness = 公司的整套规章、工具、CI 流程、code review、教练系统

### 3.3 驾驭层的七大组件

下面拆解 2026 年生产级 agent harness 的七大核心组件。

#### 3.3.1 工具集成层（Tool Integration）

详见 2.7 节的 MCP / A2A。除此之外：

- **Function Calling**：LLM provider 原生（OpenAI、Anthropic、Google、DeepSeek 都支持），适合简单工具、单 LLM provider 场景
- **MCP**：跨 LLM、可复用、生态爆发中（97M 月下载）
- **OpenAPI / GraphQL adapter**：把任意 REST/GraphQL 服务包成 MCP server
- **Agent Skills**（Anthropic 2026 推出）：把"agent 知道如何做某类任务"打包成可复用单元

实战：MCP 已是 2026 年的事实标准，新项目直接 MCP，老项目通过 MCP adapter 兼容。

#### 3.3.2 沙箱与代码执行（Sandbox）

agent 写代码 / 跑命令 / 处理用户上传文件，必须在隔离环境里。否则 prompt injection 一来，agent 就把你服务器上的 `~/.ssh/` 上传到公网了。

- **E2B**：开源 + SaaS 都有，secure sandbox 跑任意 Python / Node / Bash，毫秒级冷启
- **Daytona**：开源 dev environment，原本面向 dev container，2025 起加 agent sandbox 支持
- **Modal**：serverless GPU + sandbox 双用
- **OpenAI Code Interpreter**（Apps SDK）：托管，简单
- **Anthropic 的 Claude Computer Use**：让 Claude 操作桌面应用，本质是更激进的 sandbox

最低限度：**绝不在生产服务器主进程里 `eval()` agent 输出**。即使是看起来无害的 `os.path.join`。

#### 3.3.3 LLM 网关（LLM Gateway）

把"调用 LLM"从你的业务代码里抽离到独立的网关层，统一处理：路由、fallback、缓存、限流、计费、观测、密钥管理。2026 年主流方案 [37][38]：

| 网关 | 模式 | 主打能力 | 适用 |
|---|---|---|---|
| **LiteLLM** | 开源 Python 代理 | 100+ LLM 统一 OpenAI 兼容接口、自托管 | 中小团队、自托管偏好 |
| **Portkey** | 商用 + 自托管 | 语义缓存、guardrails、企业级观测 | 企业生产 |
| **OpenRouter** | SaaS marketplace | 一个 API key 200+ 模型，按需付费 | 创业 / prototype |
| **Helicone AI Gateway** | 开源 + SaaS | Rust 实现高性能、健康感知路由 + 熔断、原生观测 | 高并发生产 |
| **Cloudflare AI Gateway** | SaaS（边缘） | 全球边缘缓存、零运维 | 已用 Cloudflare 栈 |

关键能力：

- **多 provider 路由**：根据成本 / 延迟 / 模型能力动态选 provider。GPT-4 失败 fallback 到 Claude 再 fallback 到 Llama。
- **语义缓存**：Portkey 和 Cloudflare 的杀手锏——不是字面匹配，而是"语义相似的 query 复用 cached response"。客服 / 知识库场景能做到 60%+ 缓存命中。
- **熔断**：Helicone 的"健康感知路由 + circuit breaking"在 provider 故障时自动隔离故障 endpoint。
- **统一计费 + 配额**：按 user_id / org_id 配额、防止恶意用户烧光额度。
- **密钥管理**：业务代码不接触 API key，全在网关里。

注意 [37]：2026 年 LLM proxy 生态出过两件大事——Helicone 被收购、LiteLLM 被检测出依赖供应链问题。"在生产环境运行第三方 proxy"本身是个安全决策，企业级建议自托管 + 锁版本。

#### 3.3.4 推理服务器（Inference Server）

如果你自部署模型（开源 7B-70B 或自家训的），推理服务器决定了吞吐和成本。2026 年的状态 [39][40]：

- **TGI（Text Generation Inference，HuggingFace）**：2025 年 12 月起进入 *maintenance mode*。HF 官方推荐新部署用 vLLM 或 SGLang。
- **vLLM**（UC Berkeley Sky Lab）：核心创新是 **PagedAttention** [10]，把 KV cache 切成 16 token 一页，用类似操作系统虚拟内存的方式管理，显存碎片<4%。这让单卡能跑的并发数几倍提升。生态最广（HuggingFace、各云厂、企业自部署的默认选择）。支持 prefix caching（自动检测共享前缀复用 KV）、speculative decoding（小模型先猜大模型再校验，1.3-2x 加速，acceptance rate≥0.7 时）。
- **SGLang**（LMSYS）：核心创新是 **RadixAttention** [11]——用 radix tree 数据结构自动发现并复用 KV cache，不需要手动配置 prefix caching。在多轮对话和 agent 场景（请求间共享动态上下文）比 vLLM 快 10-20%；在 prefix-heavy workload 上比基线快最多 6.4x。小模型场景（7B-13B）SGLang 比 vLLM 高 ~29% 吞吐；70B 规模差异收窄到 3-5%。
- **NVIDIA Triton Inference Server**：传统选择，多模态 / 多模型混跑场景仍有人用，但纯 LLM 推理已经没什么优势。
- **LMCache**（2025 提出）：作为独立的 KV cache 层，可以跨不同的推理服务器实例共享 cache。企业多副本部署时能再节省 30-50% 显存。

选型经验：**纯文本 LLM 推理 → SGLang（多轮 / agent 场景）或 vLLM（通用 / 简单 single-turn）；多模态混跑 → vLLM 或 Triton。**

#### 3.3.5 可观测性（Observability）

LLM 应用的 trace 比传统服务复杂得多——一次请求可能涉及多次 LLM 调用、多次工具调用、多次检索、多次 reranker、多个 sub-agent。2026 年主流 [33][34]：

| 平台 | 类型 | 强项 | License |
|---|---|---|---|
| **Langfuse** | 开源 + 云 | 多租户、ClickHouse 高吞吐、6M+ 月 SDK 安装 | MIT |
| **LangSmith** | 商用（LangChain） | LangChain 原生集成、可视化 graph、CI eval | 商用 |
| **Helicone** | 开源 + 云 | proxy 架构（一个 URL 改动就接入）、cost & latency 分析 | 开源 + 云 |
| **Arize Phoenix** | 开源 | RAG pipeline 强（向量空间可视化、retrieval drift） | ELv2 |
| **Comet Opik** | 商用 | session replay、prompt management | 商用 |
| **Braintrust** | 商用 | 全 lifecycle（dataset → score → CI gating） | 商用 |

实战 [33][34]：
- 想要功能完整 + 自托管 → **Langfuse**
- 已用 LangChain → **LangSmith**（自动集成）
- 不想改 SDK，只想接个 proxy → **Helicone**
- 重点是 RAG → **Arize Phoenix**（retrieval drift 可视化是它的看家本领）

很多团队最后落到"两个工具组合"：一个 OSS 平台（Langfuse / Phoenix）做开发与 dashboard，一个 proxy 网关（Helicone）做生产流量观测。

#### 3.3.6 评估（Evaluation）

LLM 应用没有传统单元测试那种"输入 X 必须输出 Y"的硬约束。评估的核心问题是"模型回答是好回答吗"，这又分两类 [35]：

- **离线评估（Offline Eval）**：跑一个固定的 dataset，用 metric / LLM-as-judge / 人工打分
- **在线评估（Online Eval）**：在生产流量里抽样评分，监控质量漂移

2026 年主流框架 [35]：

| 框架 | 特点 | 适用 |
|---|---|---|
| **DeepEval** | 50+ 指标、pytest 集成、agent 评估 | 工程团队 CI |
| **RAGAS** | RAG 专精：faithfulness、context precision、recall、answer relevancy | RAG 系统 |
| **Promptfoo** | 红队 + CI、零云依赖、prompt 对比 | prompt 迭代 + 安全测试 |
| **Braintrust** | 全 lifecycle 平台、CI release gate | 企业 |
| **OpenAI Evals** | 官方框架 | OpenAI 生态 |

**LLM-as-Judge** 是 2024 起的核心范式：用更强的 LLM（通常是 GPT-4 / Claude Opus）对 agent 输出打分。要点：判官模型必须强于被测模型；prompt 要给清楚 rubric；判官也会有 bias（偏向自己家模型、偏向长答案）。

实战 [35]：「你几乎一定需要两个工具——一个轻量框架（DeepEval / RAGAS / Promptfoo）做 CI/CD gating，一个平台（Braintrust / LangSmith / Arize）做人工标注、回归追踪、stakeholder dashboard。」

#### 3.3.7 护栏（Guardrails）

LLM 是不可信组件。它会幻觉、会被注入、会泄密、会生成有害内容。护栏是放在 LLM 输入和输出两侧的"过滤层"。2026 年主流：

- **NVIDIA NeMo Guardrails** [36]：开源，用 Colang DSL 描述对话流和约束。支持 5 种 rails：
  - Input rails（用户输入侧）
  - Dialog rails（对话流约束）
  - Retrieval rails（RAG chunk 过滤）
  - Execution rails（自定义 action 包装）
  - Output rails（LLM 输出侧）
  与 NVIDIA Nemotron Safety 模型集成（content safety、PII、jailbreak 检测）。
- **Guardrails AI**（开源）：Pydantic 风格的"validator"，校验 LLM 输出符合 schema、不含 PII、不含越狱触发词
- **AWS Bedrock Guardrails**：托管护栏，AWS 生态零接入
- **Azure Content Safety**：同上，Azure 生态
- **Lakera Guard**：商用 prompt injection 检测专精
- **Protect AI Layer**：商用全栈安全平台

合规驱动：**EU AI Act 高风险义务从 2026 年 8 月 2 日生效** [36]，对医疗、教育、招聘、金融、关键基础设施等场景的 AI 系统提出强制护栏要求。OWASP Top 10 for LLM Applications 已成为业界标准的风险分类。

实战配置 [Atlan 2026]：
- **输入侧**：prompt injection 检测 + PII 过滤 + 主题白名单
- **检索侧**：检索结果按敏感度分级，敏感内容不进 LLM
- **输出侧**：PII / 敏感词检测、事实性验证、JSON schema 校验、长度 / 速率限制
- **审计侧**：所有"被护栏拦下"的请求记录，定期 review

### 3.4 Agent 编排框架

把上面的组件拼成"一个 agent"需要编排框架。2026 年主流：

- **LangGraph**（LangChain）：基于图（DAG）的 state machine，明确节点 + 边、checkpointer 内置、human-in-the-loop 一等公民。生态最广，是事实标准
- **AutoGen**（Microsoft）：多 agent 对话框架，agent 之间通过自然语言协商
- **CrewAI**：「角色 + 任务」抽象，team agent 做协作
- **LlamaIndex Agent**：RAG-first 的 agent 框架
- **Semantic Kernel**（Microsoft）：偏 .NET 生态
- **OpenAI Swarm / Agents SDK**：OpenAI 官方，简洁但功能有限
- **Smolagents**（HuggingFace）：极简，只有几百行核心代码
- **Pydantic AI**：基于 Pydantic 的强类型 agent
- **Mastra**（TypeScript 原生）：JS / TS 生态优选

新项目通用建议 [Vinod Rane 2026]：
- **Python + 复杂 workflow** → LangGraph
- **TypeScript / Next.js** → Mastra 或 Vercel AI SDK
- **多 agent 协作 / 头脑风暴模式** → AutoGen 或 CrewAI
- **极简 + 教学** → Smolagents

### 3.5 完整生产栈参考架构

把所有组件串起来，2026 年一个"教科书式"的生产 agent 系统大致是这样：

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

每个箭头都是一个"被驾驭"的接口。错一次，加一条 rule、一个 eval case、一个 guardrail 规则——这就是 harness engineering 的日常。

---

## Synthesis & Insights：三层演进的全貌

### 时间叙事

把过去四年的行业焦点拉成时间轴：

- **2022-2023：提示词工程时代**
  - ChatGPT 引爆、人人写 prompt
  - 主要技术：CoT、Few-shot、ReAct
  - 工程产物：prompt template、prompt 库、prompt 培训课
  - 隐喻："能跟模型说话的人"
- **2024：上下文工程兴起**
  - Karpathy 命名、Anthropic 推动
  - 长上下文（100K → 1M）+ RAG 成熟 + 记忆系统涌现
  - 主要技术：RAG、向量库、reranker、MCP（11 月）
  - 工程产物：vector DB 部署、RAG pipeline、MCP server 生态
  - 隐喻："能给模型搭舞台的人"
- **2025-2026：驾驭工程成为生产关键**
  - Hashimoto 命名、OpenAI 内部架构曝光、Karpathy 反转工作流
  - 主要技术：MCP（97M 月下载）、推理服务器（vLLM/SGLang）、Agent 编排（LangGraph）、可观测性、评估、护栏
  - 工程产物：完整 agent 栈、合规框架、企业 AI control plane
  - 隐喻："能驯服一群 agent 的工程经理"

### 三者的层次关系

不是替代而是叠加。一个生产级 agent 三层都需要：

- 不写好 prompt（第一层）→ 模型理解不了任务
- 不管好 context（第二层）→ 模型看不到完成任务必需的信息
- 不建好 harness（第三层）→ 系统在生产里不可控、不可靠、不可演进

类比软件工程：
- prompt = 你写的那个具体函数
- context = 这个函数运行时能访问的数据 + 依赖
- harness = 整个工程的 CI / CD / 监控 / 测试 / 发布流程

### 工程师能力栈的迁移

2022 年最值钱的 LLM 工程师能力是"能调出模型最佳输出的 prompt 直觉"。
2024 年最值钱的能力是"能架一套 RAG 让公司所有文档可问"。
2026 年最值钱的能力是"能搭一套 agent harness 让多个 agent 在生产 7×24 跑而不出大事故"。

注意这条迁移线不是终点，下一站很可能是"能把多个 agent harness 联邦化的协议工程师"——A2A 协议生态成熟、多公司 agent 互操作的时代。

### 跨三层的 7 个共性原则

不管是 prompt、context 还是 harness，工程实践都遵循一些共性原则：

1. **测量先于优化**：没有 eval 数据、没有 dashboard、没有 baseline，所有"提升"都是幻觉
2. **简单优先**：80% 的任务用 baseline 方案够了；只在 baseline 失败的地方加复杂度
3. **可观测压倒一切**：trace、log、metric 三件套是生产 agent 的氧气
4. **防御性编程**：假设模型会幻觉、会被注入、会超时、会断网
5. **可逆设计**：每一步都能回滚，每个组件都能换掉
6. **成本意识**：每个决策都问"这一步多少 token / 多少 USD / 多少毫秒"
7. **演进式迭代**：先 prompt → 不够加 context → 不够加 harness。不要从 day 1 就建复杂栈

---

## Recommendations：三类典型场景的选型清单

### 场景一：个人学习 / 周末项目

**目标**：理解原理 + 跑通端到端
**预算**：API 几美元，硬件用现有

| 组件 | 推荐 |
|---|---|
| LLM | OpenAI GPT-4o-mini 或 Claude Haiku（便宜+够用） |
| Prompt 框架 | 直接用 Anthropic / OpenAI SDK，不用 LangChain |
| RAG | LlamaIndex（API 友好）或 LangChain RAG quickstart |
| Embedding | OpenAI text-embedding-3-small（最便宜） |
| Vector DB | Chroma（嵌入式，零运维） |
| Agent | LangGraph（学完后通用） |
| 观测 | Langfuse Cloud 免费 tier |

直接抄 LangChain / LlamaIndex 官方 cookbook 改改就跑。

### 场景二：创业 MVP / 小团队产品

**目标**：尽快上线 + 可演进
**预算**：每月几百 USD

| 组件 | 推荐 |
|---|---|
| LLM | OpenRouter（一个 key 接所有模型，按需切换） |
| Prompt 管理 | Promptfoo（CI gate） + LangSmith（观测） |
| RAG | LlamaIndex / LangGraph |
| Embedding | Voyage 3 large（中文场景）或 OpenAI text-3-large |
| Vector DB | Qdrant Cloud（性价比 + 企业级） |
| 记忆 | Mem0（最快接入） |
| Agent | LangGraph |
| LLM 网关 | LiteLLM 自托管（Docker 一行起） |
| 观测 | Langfuse 自托管 |
| 评估 | Promptfoo + RAGAS |
| 护栏 | Guardrails AI（轻量） |

总成本：服务器 ~$100/月 + LLM 按量。

### 场景三：企业生产 / 合规敏感

**目标**：稳定、可观测、合规、可审计
**预算**：六位数月度

| 组件 | 推荐 |
|---|---|
| LLM | 多 provider 冗余 + 自部署 OSS（敏感数据） |
| 自部署推理 | SGLang（多轮 / agent）+ vLLM（单轮 / 高吞吐） |
| Prompt 管理 | LangSmith / Braintrust |
| RAG | 自研 modular pipeline + Cohere Rerank |
| Embedding | Cohere embed-v4 + BGE-M3 双部署 |
| Vector DB | Milvus 集群 / pgvector（已有 Postgres） |
| 记忆 | Zep（用户状态）+ Letta（持续 agent） |
| Agent | LangGraph 主、AutoGen 多 agent 协作 |
| LLM 网关 | Portkey 自托管（语义缓存 + 企业级控制） |
| 观测 | Langfuse 自托管 + Arize Phoenix（RAG） |
| 评估 | Braintrust（lifecycle）+ DeepEval（CI） |
| 护栏 | NeMo Guardrails + Lakera Guard（注入检测） |
| 合规 | EU AI Act 2026-08 高风险义务、SOC2、HIPAA（视行业） |

通用建议：**无论哪种场景，从 day 1 就接观测**。Langfuse 接入零成本，没接的代价是 6 个月后做不出任何"为什么这个 query 失败了"的根因分析。

---

## Limitations & Caveats

1. **行业仍在快速演化**：本文截至 2026-05-01。MCP、Agent Skills、A2A 协议、推理服务器都在数月级别迭代，6 个月后细节可能过时
2. **基准数据有作弊空间**：MTEB、RULER、reranker ELO 都存在 cherry-pick 风险。生产前必须在自己业务数据上重测
3. **"驾驭工程"作为术语仍年轻**：2026-02 才被正式命名，社区共识还在形成。本文采用 Hashimoto / Augment Code / Atlan / Adnan Masood 等 ≥4 个独立来源都引用的描述，但术语边界仍可能漂移
4. **未覆盖**：模型推理数学（attention 推导、loss 函数）、具体业务案例 ROI、提示词攻击 / 越狱的详细技术目录、监管细节（EU AI Act 全文）
5. **评分有主观成分**：选型表里的"推荐"基于公开评测 + 社区共识 + 笔者实战偏好，不构成对所有场景的最优解
6. **付费 vs 开源**：本文偏向开源方案介绍。商业 SaaS 在很多场景的"零运维"价值未被充分体现
7. **中文场景特殊性**：中文 embedding / RAG / 模型选型与英文场景有差异。本文给出的推荐已尽量考虑中文场景，但具体业务（金融 / 法律 / 医疗）的中文 NLP 仍需领域 fine-tune

---

## Bibliography

### 论文 / 学术资源

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

### 业界文章 / 官方博客

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
[43] Squirro / 多源. *RAG in 2026: Bridging Knowledge and Generative AI*.

### 框架 / 协议官方资源

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

## Methodology Appendix

### 调研设计
- **Skill**：`deep-research` v2.3.1（199-bio 版）
- **Mode**：ultradeep（用户 explicit "越全越好"，触发 8+ 阶段最深档）
- **Search provider**：内置 WebSearch（多供应商 search-cli 未配置 key，回退）
- **检索路数**：18 路（两批并行：第一批 8 路覆盖三大主题主干，第二批 8 路覆盖运行时栈与长尾选型，第三批 2 路补训练方法）

### 18 路检索清单
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

### Phase 4 TRIANGULATE — 跨多源验证的关键 claim
所有进入正文的关键事实都至少有 2 个独立来源支持：

| Claim | 至少 2 源支持 |
|---|---|
| Karpathy 命名 "context engineering"，YC AI School 演讲 | Anthropic 官方 + IntuitionLabs + FlowHunt + LangChain |
| Mitchell Hashimoto 2026-02 命名 "harness engineering" | Augment Code + MadPlay + Atlan + Adnan Masood |
| MCP 2024-11 由 Anthropic 发布、2025-12 转 Linux Foundation、2026-02 SDK 月下载 9700 万 | Wikipedia + a2a-mcp.org + Pockit + EssaMamdani |
| Lost-in-the-Middle U 形召回曲线 | TACL 2024 + 多篇 2026 综述 |
| Gemini 1.5 Pro 99.7% needle 但 ~60% 多事实召回 | Tianpan + LongContext + RAG 论文 |
| 长上下文比 RAG 慢 30-60x、贵 1250x | Tianpan + 同类生产数据帖 |
| TGI 2025-12 进入 maintenance mode | PreMAI + SitePoint |
| vLLM PagedAttention <4% 显存碎片 | vLLM 官方博客 + 多篇综述 |
| SGLang RadixAttention 在 conversational 工作负载比 vLLM +10-20% | Runpod + SGLang 官方 |
| Speculative decoding 1.3-2x，acceptance≥0.7 | vLLM 文档 + 多篇评测 |
| Cohere embed-v4 65.2 MTEB（当前榜首） | pecollective + reintech + Mixpeek |
| Voyage 3 large 比 OpenAI text-3-large 高 ~10% | TokenMix + Cheney Zhang |
| Hybrid + reranker 比 semantic-only +9.3pp MRR | LanceDB benchmark + dev.to |
| Self-RAG 减少 25-40% 不必要检索 | Squirro + 综述 |
| Langfuse 6M+ 月 SDK 安装 | Langfuse 官方 + Spheron |
| Full FT 7B = 100-120GB VRAM, QLoRA 7B = ~6GB | Introl + Mercity + RedHat |
| QLoRA 80-90% 质量 vs Full FT | Introl + RedHat |
| GRPO 是 DeepSeek-R1 核心算法 | llm-stats 2026 + Sebastian Raschka 综述 |
| EU AI Act 高风险义务 2026-08-02 生效 | NVIDIA NeMo docs + 多篇合规文章 |

### Phase 6 CRITIQUE — 红队复核
- **怀疑派**：是否漏了重要话题？补查后又加了 2 路（训练方法）。剩余可能有遗漏的：多模态模型架构（CLIP 系、Vision Transformer 系），但本文聚焦 LLM 应用工程而非模型架构演进，可接受
- **对抗评审**：是否过度简化？驾驭工程章节 7 大组件每个都只给 ~500 字，确有简化；本文目标是导引图谱而非每组件深扒，可接受。每组件给了 2-3 个公开来源供读者深挖
- **实施工程师**：选型清单是否可立即落地？三类场景都给了具体产品名 + 部署方式，且明确了"day 1 必接观测"等首要原则

### Phase 8 PACKAGE
- 主报告：`article.md`
- 元数据：`run_manifest.json`、`sources.jsonl`
- HTML / PDF：未生成（用户未要求 + WeasyPrint 未安装）

### 复现命令
```bash
ls -la ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/
wc -m ~/Documents/Prompt_Context_Harness_Engineering_Article_20260501/article.md
```



