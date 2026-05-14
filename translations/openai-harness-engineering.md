# Harness 工程：在 agent-first 的世界里驾驭 Codex

> **原标题**：Harness engineering: leveraging Codex in an agent-first world
> **作者**：Ryan Lopopolo（OpenAI Member of the Technical Staff）；致谢 Victor Zhu、Zach Brock
> **原文**：[Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
> **发布**：2026-02-11 · **本中文译稿**：2026-05-14
> 本文为社区学习用途的非官方中文翻译。技术名词以括注英文原文为准；最终解释以原文为准。

过去五个月里，我们团队做了一个实验：构建并发布了一款内部 beta 阶段的软件产品——**手工写代码的行数为 0**。

这款产品已有内部日活用户和外部 alpha 测试者。它会发布、会上线、会出 bug、会被修复。**不同的是：每一行代码——业务逻辑、测试、CI 配置、文档、可观测性、内部工具——都是由 Codex 写的**。我们估算，整体投入大约是手工写代码所需时间的 **1/10**。

**人类把方向，agent 做执行。（Humans steer. Agents execute.）**

我们刻意设下这道约束，倒逼自己去搭建**让工程速度提升一个数量级**所必需的那些东西。我们当时只有几周时间发布最终大约一百万行代码的产品。要做到这一点，**必须搞清楚一件事**：当一个软件工程团队的主业不再是写代码，而是**设计环境、明确意图、构建反馈闭环**让 Codex agent 能可靠地完成工作时，到底有哪些东西需要重新思考？

这篇文章讲的是：用一支由 agent 组成的"团队"从零做一款新产品，我们学到了什么——**什么会失灵、什么会复合、以及如何最大化我们唯一真正稀缺的资源：人的时间和注意力**。

## 我们从一个空 git 仓库开始

第一次向空仓库提交 commit，是 2025 年 8 月底的事。

最初的脚手架——仓库结构、CI 配置、格式化规则、包管理器设置、应用框架——都是用 Codex CLI 在 GPT-5 上生成的，参考了一小撮已有模板。**就连最初指导 agent 如何在仓库里工作的 `AGENTS.md` 文件本身，也是由 Codex 写的**。

**没有任何预先存在的、由人类写的代码作为锚点**。从一开始，整个仓库的形态就是被 agent 塑造的。

五个月之后，这个仓库大约包含**百万行代码**——覆盖业务逻辑、基础设施、工具、文档、内部开发者工具。这段时间内，**3 名工程师驱动 Codex 开出并合并了约 1500 个 PR**。换算成人均产出大约是 **3.5 个 PR / 人 / 天**——而且令人惊讶的是，团队扩到现在的 7 人后，**人均吞吐量反而增加了**。重要的是：这不是为了产出而产出——产品已被数百名内部用户使用，包括重度日活用户。

整个开发过程中，**人类从未直接贡献过一行代码**。这成了团队的核心信条：**no manually-written code（不手写代码）**。

## 重新定义"工程师"这个角色

不亲自动手写代码，**引入了一种不同形态的工程工作——聚焦在系统、脚手架和杠杆上**。

早期进展比我们预期的要慢，但不是因为 Codex 能力不足，而是因为**环境不够清晰**。Agent 缺少为高阶目标推进所需的工具、抽象、内部结构。我们工程团队的主要工作，**变成了帮 agent 把活干漂亮**。

实践中，这意味着**深度优先**：把更大的目标拆成更小的构件（设计、代码、评审、测试等等），prompt agent 去构造这些构件，再用它们去打开更复杂的任务。**出问题时，修复方案几乎从来不是"再多试一次"**。因为唯一能推进的方式就是让 Codex 来做这件事，**人类工程师永远要走进任务、问自己**：**"缺了什么能力？怎么才能让这个能力对 agent 既可读、又可强制？"**

人类几乎完全通过 prompt 与系统交互：工程师描述任务、运行 agent，让它去开 PR。要把一个 PR 推到完成，我们让 Codex **本地评审自己的改动、显式请求其他 agent 评审（本地和云端）、回应人类或 agent 的反馈，并在循环中迭代直到所有 agent 评审通过**——这本质上是一个 [Ralph Wiggum 循环](https://ghuntley.com/loop/)（Ralph Wiggum Loop，一种"傻乎乎不停转直到通过"的循环范式）。Codex 直接使用我们标准的开发工具（`gh`、本地脚本、仓库内置的 skills）去拿上下文，**不需要人类把内容粘贴到 CLI 里**。

人类**可以**评审 PR，但不是必须。随时间推移，**我们已经把几乎所有评审工作都推到了 agent-to-agent**。

## 提升应用的"可读性"

随着代码吞吐量上升，**瓶颈变成了人类的 QA 容量**。由于固定的约束是"人的时间和注意力"，我们一直在给 agent 加能力——让**应用 UI、日志、应用指标**这些东西**对 Codex 直接可读**。

举个例子，我们让应用可以**按 git worktree 启动**，这样 Codex 可以为每个改动启动一个独立实例。我们还把 Chrome DevTools Protocol 接到了 agent runtime 里，并为 DOM 快照、截图、导航这些操作建立了 skills。这让 Codex 能**直接复现 bug、验证修复、对 UI 行为做推理**。

![Codex 通过 Chrome DevTools MCP 驱动应用以验证自己工作的示意图：Codex 选定目标、在触发 UI 路径前后做状态快照、通过 Chrome DevTools 观察运行时事件、施加修复、重启，并循环地重新运行验证直到应用干净](https://images.ctfassets.net/kftzwdyauwt9/1Gu58eNlqDEuITmbqJDmq9/1e2e62f7e15fb16d2da0da5407240564/fig_1__codex_drives_the_app_.png?w=3840&q=90&fm=webp)

我们对可观测性工具也做了同样的事。**日志、metrics、traces 通过一个本地可观测性栈暴露给 Codex**——这套栈针对任意 worktree 都是临时的（ephemeral）：Codex 工作在一个完全隔离的应用版本上，包括它的日志和指标，**任务完成后整套栈被销毁**。Agent 可以用 **LogQL** 查询日志、用 **PromQL** 查询指标。有了这层上下文，像"确保服务启动在 800ms 内完成"或"这 4 条关键用户路径里没有任何一个 span 超过 2 秒"这样的 prompt，**就变成可执行的目标**。

![把完整可观测性栈交给本地开发中的 Codex：应用把日志、指标、traces 发到 Vector，Vector 分发给包含 Victoria Logs / Metrics / Traces 的可观测性栈，分别通过 LogQL / PromQL / TraceQL API 查询；Codex 用这些信号做查询、关联、推理，然后在代码库里实施修复、重启应用、重跑 workload、测试 UI 路径，在反馈闭环中循环](https://images.ctfassets.net/kftzwdyauwt9/4Xr18TZ5G4Bh8zIgsTFIVK/f7ae689ddd8c31664e39d809b0973425/OAI_Harness_engineering_Giving_Codex_a_full_observability_stack_desktop-light__1_.svg)

我们经常看到**单次 Codex run 在一个任务上跑超过 6 小时**（往往是趁人睡觉的时候）。

## 我们把"仓库知识"做成了系统记录源

**上下文管理**是把 agent 用在大型复杂任务上的最大挑战之一。我们最早学到的一课很简单：**给 Codex 一张地图，而不是一本 1000 页的说明书**。

我们试过"一份大而全的 [`AGENTS.md`](https://agents.md/)"方案。它会按以下可预测的方式失败：

- **上下文是稀缺资源。** 一个巨型指令文件会把任务、代码、相关文档都挤掉——结果 agent 要么漏掉关键约束、要么为错误的目标做优化。
- **指引太多就变成了"没指引"（non-guidance）。** 当一切都"重要"，那就什么都不重要了。Agent 最终只会做局部 pattern matching，而不是有意图地导航。
- **会瞬间腐烂。** 一个庞然大物式的说明书会变成"陈旧规则的坟场"。Agent 分不清哪些还有效，人也懒得维护，文件就悄悄地变成"看起来有用但实际危险的诱惑物（attractive nuisance）"。
- **难以校验。** 一坨整体的文件没法做机械检查（覆盖度、新鲜度、所有权、交叉引用），漂移（drift）是必然。

所以**我们不再把 `AGENTS.md` 当百科全书，而是把它当目录页（table of contents）**。

仓库的知识库存放在一个结构化的 `docs/` 目录里，**它才是系统记录源（system of record）**。一份大约 100 行的简短 `AGENTS.md` 被注入到上下文中，主要作为**一张地图**，指向其他地方更深的真理源。

```text
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   ├── new-user-onboarding.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   ├── uv-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

*仓库内知识库的版面布局。*

**设计文档**被编目和索引，包含其验证状态以及定义 agent-first 运行原则的核心信条（core beliefs）。[架构文档](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html)给出领域和包分层的顶层地图。一份质量文档（quality document）给每个产品领域和架构层级打分，跟踪 gap 随时间的变化。

**计划被当作一等产物（first-class artifacts）**。小改动用临时轻量化的 plan；复杂工作用 [execution plans](https://cookbook.openai.com/articles/codex_exec_plans) 来承载，其中包括进展与决策日志，都签入仓库。Active plans、completed plans、known technical debt 全部版本化、共址（co-located），让 agent 不需要依赖外部上下文就能工作。

这就实现了 **渐进式展开（progressive disclosure）**：agent 从一个**小而稳定的入口点**开始，被教会"接下来去看哪里"，而不是一上来就被海量信息淹没。

我们用机械方式**强制执行**这一切。专门的 linter 和 CI job 校验知识库是否最新、交叉链接是否完整、结构是否正确。**一个定期跑的 "doc-gardening" agent（文档园丁）**扫描那些与真实代码行为不符的过时或废弃文档，并开 PR 修复。

## "Agent 可读性"才是终极目标

随着代码库演化，**Codex 做设计决策的框架也必须跟着演化**。

因为这个仓库完全是 agent 生成的，它**首先优化的是 Codex 的可读性（legibility）**——和团队设法提升代码对新工程师的可导航性是一样的道理，我们人类工程师的目标是：**让 agent 能直接从仓库本身推理出完整的业务领域**。

**从 agent 的视角看，运行时上下文里访问不到的东西，等同于不存在**。存在 Google Docs、聊天串、或者别人脑子里的知识，系统就看不到。**只有仓库内的、版本化的产物**（代码、markdown、schema、可执行 plan 等）它才能看到。

![Agent 知识的边界："Codex 看不到的东西就不存在"。Codex 的知识被画成一个有界气泡。下方是它看不到的知识例子——Google Docs、Slack 消息、人脑里的隐性知识。箭头表示要让这些信息对 Codex 可见，必须**把它们编码进代码库里的 markdown**](https://images.ctfassets.net/kftzwdyauwt9/7uWHsJIC6o3uQPsnQ2Avz9/8be3e321892054bd215afb2b250a176a/OAI_Harness_engineering_The_limits_of_agent_knowledge_desktop-light.png?w=3840&q=90&fm=webp)

我们逐渐学到：**得把越来越多的上下文搬进仓库**。那场让团队在某个架构模式上达成一致的 Slack 讨论？**如果对 agent 不可发现，它就和"三个月后加入的新员工没听说过一样不可见"**。

给 Codex 更多上下文，意味着**组织和暴露正确的信息让 agent 能在上面推理**，而不是用一堆零散指令把它淹没。就像你给新队友讲产品原则、工程规范、团队文化（包括偏好哪种 emoji）一样——给 agent 这些信息，**它的输出会与你更对齐**。

这种视角厘清了不少 tradeoff。**我们偏好那些可以被 agent 完整内化、并能在仓库内推理的依赖与抽象**。那些被称为"无聊（boring）"的技术，往往因为**组合性、API 稳定性、以及在训练集里的覆盖度**，更容易被 agent 建模。某些情况下，**让 agent 重新实现某段功能，比绕开公共库不透明的上游行为更便宜**。例如，与其引入一个通用的 `p-limit` 风格的包，**我们让它自己实现了一个 map-with-concurrency helper**——它和我们的 OpenTelemetry 桩深度集成、100% 测试覆盖、行为完全符合 runtime 的预期。

**把更多系统拉进 agent 能直接检视、校验、修改的形式里，就是在放大杠杆**——不仅对 Codex 如此，对其他在这个代码库里干活的 agent（例如 [Aardvark](https://openai.com/index/introducing-aardvark/)）也是如此。

## 强制执行架构与品味

光有文档**留不住**一个完全由 agent 生成的代码库的一致性。**通过强制不变量（invariants），而非微观管理实现，我们让 agent 既能快速发布、又不会从根基上崩坏**。例如，我们要求 Codex [在边界处解析数据（parse, don't validate）](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)，但**不规定具体怎么做**（模型似乎偏爱 Zod，但我们并没指定要用这个库）。

**Agent 在边界严格、结构可预测的环境里最有效**——参见 ["AI 在逼我们写好代码"](https://bits.logic.inc/p/ai-is-forcing-us-to-write-good-code) 这一类观点——所以**我们围绕一套刚性架构模型来组织应用**。每个业务领域被划分成一组**固定的层级**，依赖方向被严格校验，允许的连接边（permissible edges）也很有限。这些约束通过自定义 linter（当然也是 Codex 生成的！）和结构性测试**用机械方式强制执行**。

下图展示了这条规则：在每个业务领域内（如 App Settings），**代码只能"向前"穿过一组固定的层级**——`Types → Config → Repo → Service → Runtime → UI`。横切关注点（auth、connectors、telemetry、feature flags）**通过一个显式接口 Providers 进入**。除此之外的依赖一律禁止，并由机制强制。

![分层领域架构与显式横切边界：在业务逻辑域内部，模块按 Types → Config → Repo，以及 Providers → Service → Runtime → UI 的方向单向排列，底部是 App Wiring + UI；Utils 模块位于边界之外，喂给 Providers](https://images.ctfassets.net/kftzwdyauwt9/4Rlip1H3T9apPlSmWs7Wr8/7708c176bfbe11951e06ad8e2b83bf01/OAI_Harness_engineering_Layered_domain_architecture_with_explicit_cross-cutting_boundries_desktop-light.png?w=3840&q=90&fm=webp)

这是那种**通常要等到团队有几百号工程师才会上**的架构。**有了 coding agent，它反而是个早期前提**——这些约束**正是让你"既能快、又不腐烂、不漂移"的东西**。

实践中，我们用自定义 linter 和结构性测试，再加上少量"品味不变量（taste invariants）"来强制规则。例如，我们静态强制结构化日志、schema 与 type 的命名约定、文件大小上限、以及平台层的可靠性要求。**因为 linter 是自定义的，我们把错误信息写成"会把修复指令注入 agent 上下文"的格式**。

在以人为本的工作流里，这些规则可能让人觉得啰嗦、束手束脚。**但配上 agent，它们就成了乘数**：一旦编码下来，**会同时作用到所有地方**。

与此同时，我们对"约束在哪里重要、在哪里不重要"也很显式。这种方式很像**带一个大型工程平台组织**：**边界集中强制、局部允许自治**。你对边界、正确性、可复现性极度上心；在这些边界之内，你给团队——或 agent——**很大的方案表达自由**。

最终产出的代码**不一定符合人类的风格偏好**，没关系。**只要它正确、可维护、对未来的 agent run 可读**，就达到了门槛。

**人类的品味会被持续反馈进系统**。评审意见、重构 PR、用户反馈的 bug——要么被记入文档更新，要么被直接编码进工具链。**当文档不够用时，我们就把规则升级成代码（promote the rule into code）**。

## 吞吐量改变了合并哲学

随着 Codex 的吞吐量上升，**许多传统工程规范开始适得其反**。

我们的仓库以**最少的阻塞式合并门控（minimal blocking merge gates）**运行。**PR 寿命很短**。测试 flake 通常用"重跑一次"解决，而不是无限地阻塞进度。**在一个 agent 吞吐量远超人类注意力的系统里，纠错很便宜、等待很贵**。

在低吞吐量环境里这样干很不负责任。**但在这里，这往往是正确的 tradeoff**。

## "Agent 生成"到底意味着什么

当我们说"这个代码库是 Codex agent 生成的"时，我们指的是**代码库里的一切**。

Agent 产出：

- 产品代码与测试
- CI 配置与发布工具
- 内部开发者工具
- 文档与设计历史
- 评估 harness
- 评审意见与回复
- 管理仓库本身的脚本
- 生产 dashboard 的定义文件

**人类始终在闭环中（in the loop），但工作在一个不同的抽象层级**。我们排优先级、把用户反馈翻译成验收标准、并验证结果。**当 agent 卡住时，我们把这视为一个信号**：找出缺什么——工具、护栏、文档——再把它喂回仓库，**修复永远由 Codex 自己来写**。

Agent 直接使用我们标准的开发工具：拉评审反馈、内联回复、推更新、并常常自己 squash & merge 自己的 PR。

## 自主性的层级在不断提升

当开发循环越来越多地被直接编码进系统里——测试、校验、评审、反馈处理、恢复——**仓库最近跨过了一道有意义的门槛：Codex 可以端到端驱动一个新功能**。

只给一句 prompt，agent 现在能做到：

- 校验代码库当前状态
- 复现一个被报告的 bug
- 录一段视频演示这个失败
- 实施修复
- 通过驱动应用来验证修复
- 录第二段视频演示修复结果
- 开 PR
- 回应 agent 与人类的反馈
- 检测并修复构建失败
- 仅在需要判断时才升级（escalate）给人
- 合并改动

**这种行为高度依赖这个仓库的特定结构和工具**，不能假定它**在没有同等投入的前提下**就能泛化——至少现在还不行。

## 熵与垃圾回收

**完整 agent 自主性也带来了新的问题**。**Codex 会复现仓库里已经存在的模式——哪怕那些模式参差不齐或是次优的**。随时间推移，这必然导致漂移（drift）。

最开始，人类靠手动来管。我们团队曾经**每周五（每周的 20%）都用来打扫"AI 垃圾代码（AI slop）"**。不出所料，**这一套没法 scale**。

后来我们改成把所谓的**"黄金原则（golden principles）"**直接编码进仓库，并构建了一个**周期性的清理流程**。这些原则是**有立场、机械化**的规则，目的是为未来的 agent run 保持代码库**可读、一致**。举两个例子：(1) 我们偏好**共享 utility 包**而不是手写 helper，把不变量集中管理；(2) 我们不"YOLO 式地探测数据形状"——**要么在边界处校验、要么依赖类型化 SDK**，这样 agent 就不会基于"猜测出来的结构"误打误撞。**按固定节奏跑一组后台 Codex 任务**，扫描偏差、更新质量评分、并开**有针对性的重构 PR**。这些 PR 大多**一分钟内能审完并自动合入**。

这种机制的功能就像**垃圾回收（garbage collection）**。**技术债像一笔高利贷：与其让它复利累积、攒到痛苦的批量整理日，不如**连续小额偿还**。**人类的品味只表达一次，然后被持续地强制执行到每一行代码上**。这也让我们能**每天**捕获并解决坏模式，而不是任由它们在代码库里散播数天甚至数周。

## 我们还在学的东西

至今为止，这套打法在 OpenAI 内部上线和采用阶段表现良好。**给真实用户做一款真实产品**——这件事帮我们把投入锚定在现实里，把方向往长期可维护性上拉。

我们仍不知道的是：**在一套完全由 agent 生成的系统里，架构一致性会如何随年份演化**。我们仍在学**人类判断在哪里增益最大**、以及**如何把这种判断编码下来让它复利**。我们也不知道随着模型继续变强，这套系统会如何演变。

已经清楚的是：**做软件依然需要纪律，但纪律更多体现在脚手架，而不是代码本身**。**让代码库保持一致的工具链、抽象和反馈闭环**变得越来越重要。

**我们当前最难的挑战，已经转向"设计环境、反馈闭环和控制系统"**——这些东西帮 agent 完成我们真正的目标：**大规模地构建并维护复杂、可靠的软件**。

随着 Codex 这样的 agent 接手软件生命周期中越来越大的部分，这些问题只会变得更重要。希望分享这些早期经验，对你思考"该把精力投到哪里"有所帮助，让你**[可以直接造东西](https://openai.com/codex/)**。
