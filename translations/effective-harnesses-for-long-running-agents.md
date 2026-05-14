# 为长时程 Agent 打造高效的 Harness

> **原标题**：Effective harnesses for long-running agents
> **作者**：Justin Young；David Hershey、Prithvi Rajasakeran、Jeremy Hadfield、Naia Bouscal、Michael Tingley、Jesse Mu、Jake Eaton、Marius Buleandara、Maggie Vo、Pedram Navid、Nadine Yasser、Alex Notov 等共同贡献（Anthropic）
> **原文**：[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
> **发布**：2025-11-26 · **本中文译稿**：2026-05-14
> 本文为社区学习用途的非官方中文翻译，技术名词以括注英文原文为准，最终解释以原文为准。

随着 AI agent 能力越来越强，开发者越来越倾向于把那些**需要数小时甚至数天才能完成的复杂任务**交给它们。然而，**让 agent 在跨多个上下文窗口的工作中保持稳定推进**，至今仍是一个开放问题。

长时程 agent 的核心难题在于：**它们必须在一段段离散的会话里工作**，而每一段新会话开始时，对此前发生的一切都没有记忆。想象一个软件项目由轮班工程师负责，每位新班次工程师上岗时对上一班发生了什么一无所知。由于上下文窗口有限、而大多数复杂项目又装不进单次窗口，**agent 需要某种机制在编码会话之间架起桥梁**。

我们开发了一套两段式方案，让 [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) 能跨多个上下文窗口高效工作：一个**初始化 agent（initializer agent）**负责在首次运行时搭建环境，另一个**编码 agent（coding agent）**则负责在每次会话中做出**增量推进**，并为下一次会话留下清晰的工作产物。完整代码示例见配套的 [quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding)。

## 长时程 Agent 的核心难题

Claude Agent SDK 是一个通用、强大的 agent harness——擅长编码，也胜任其他需要模型调用工具收集上下文、规划、执行的任务。它具备**压缩（compaction）**等上下文管理能力，能让 agent 在不耗尽上下文窗口的情况下继续工作。理论上，凭这套设置，agent 应该能任意长时间地持续做有用功。

但**仅有压缩是不够的**。即便是 Opus 4.5 这种前沿编码模型，开箱即用地在 Claude Agent SDK 上跨多个上下文窗口循环运行，仅靠一句高层 prompt（如"做一个 [claude.ai](http://claude.ai/redirect/website.v1.88c2cc68-ad7a-4c57-b8ba-abde3d58c0d1) 的克隆"），也无法构建出生产级质量的 web 应用。

Claude 的失败主要表现为两种模式：

第一，**agent 倾向于一次做太多事**——本质上是想"一把梭"把整个 app 一次性做出来。结果常常是模型在实现到一半时把上下文耗尽，留给下一次会话的是一个**未完成、未文档化、状态半残**的功能。下一个 agent 不得不去猜上一轮发生了什么，花大量时间把基本的 app 重新跑起来。即便开启了压缩，这种情况也会发生——压缩并不总能把指令完美清晰地传递给下一个 agent。

第二种失败模式经常发生在项目较晚期：当一部分功能已经做好后，后来的 agent 实例环顾四周，看到"已经有进展了"，就**宣布任务完成**。

这把问题分解成两部分。**第一**，我们需要搭建一个能承载某个 prompt 所需**全部**功能的初始环境，让 agent 能逐步、按功能推进。**第二**，我们应当 prompt 每个 agent 在向目标推进的同时，**会话结束时把环境留在一个干净状态**。所谓"干净状态"是指**适合合入主分支的代码**：没有重大 bug、代码整齐且有文档，开发者可以直接开始新功能开发，而不必先收拾上一个 agent 留下的烂摊子。

在内部实验中，我们用一套两段式方案解决了这些问题：

1. **初始化 agent（Initializer agent）**：首次 agent 会话使用一份**专门的 prompt**，要求模型搭建初始环境：一个 `init.sh` 脚本、一个记录 agent 已做过哪些事的 `claude-progress.txt` 文件，以及一次显示新添加文件的初始 git commit。
2. **编码 agent（Coding agent）**：之后的每次会话都要求模型**做出增量推进，然后留下结构化的更新**。<sup>1</sup>

这里的关键洞见是：**找到一种方式让 agent 在拿到新的、干净的上下文窗口时，能迅速理解当前工作状态**——靠的就是 `claude-progress.txt` 文件加上 git 历史。这些做法的灵感来自**高效软件工程师每天的工作习惯**。

## 环境管理

在更新版的 [Claude 4 提示词指南](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices#multi-context-window-workflows) 中，我们分享过多上下文窗口工作流的一些最佳实践，包括一种 harness 结构——"**给最开始的那个上下文窗口一份不同的 prompt**"。这份"不同的 prompt"要求初始化 agent **搭建好后续编码 agent 所需的全部环境与上下文**。下面对这种环境的几个关键组件做深入讨论。

### 功能清单（Feature list）

为了解决"一把梭做完整个 app"和"过早宣布项目完成"两个问题，我们让初始化 agent **基于用户最初的 prompt，写一份详尽的功能需求文件**。在 [claude.ai](http://claude.ai/redirect/website.v1.88c2cc68-ad7a-4c57-b8ba-abde3d58c0d1) 克隆这个例子里，这意味着 **200+ 条功能**，比如"用户可以打开新聊天、输入 query、按下回车、看到 AI 回复"。所有这些功能一开始都被标记为"failing"，这样后续编码 agent 就有了一份**完整功能长什么样的清晰提纲**。

```json
{
    "category": "functional",
    "description": "New chat button creates a fresh conversation",
    "steps": [
      "Navigate to main interface",
      "Click the 'New Chat' button",
      "Verify a new conversation is created",
      "Check that chat area shows welcome state",
      "Verify conversation appears in sidebar"
    ],
    "passes": false
  }
```

我们让编码 agent **只能通过修改 `passes` 字段的状态**来编辑这个文件，并用措辞强硬的指令——比如"**删除或编辑测试是不可接受的**，因为这可能导致功能缺失或带 bug"。经过若干实验，我们最终选用 JSON 格式：相比 Markdown，**模型更不容易不当地修改或覆盖 JSON 文件**。

### 增量推进（Incremental progress）

有了上述初始环境脚手架，下一阶段的编码 agent 被要求**一次只做一个功能**。这种增量做法对**抑制 agent 一次做太多事的倾向**至关重要。

仅仅做到增量推进还不够：每次代码改动后，模型必须**把环境留在干净状态**。我们在实验中发现，最能稳定激发这种行为的方式是：**要求模型把进展用清晰的 commit message 提交到 git，并在 progress 文件里写下进展摘要**。这样模型就可以**用 git 回滚有问题的代码改动，恢复代码库的可用状态**。

这些做法同时提升了效率，因为 agent 不再需要花时间去猜上轮发生了什么、不再需要重复把基本的 app 重新跑通。

### 测试

我们观察到的另一个重要失败模式是：**Claude 倾向于在没有充分测试的情况下就把功能标记为完成**。如果没有显式 prompt 引导，Claude 通常会改代码、甚至用单元测试或 `curl` 打开发服务器做测试，但仍然**意识不到这个功能从端到端来看其实没跑通**。

在构建 web app 这个场景里，**一旦显式 prompt 要求 Claude 使用浏览器自动化工具、以人类用户的方式做全部测试**，它在端到端验证功能上的表现就大幅改善。

![Claude 通过 Puppeteer MCP server 测试 claude.ai 克隆时截下的屏幕画面](https://www-cdn.anthropic.com/images/4zrzovbb/website/f94c2257964fb2d623f1e81f874977ebfc0986bc-1920x1080.gif)

把这类测试工具交给 Claude 大幅提升了表现——agent 能识别并修复那些**仅看代码不容易发现**的 bug。

不过仍有一些问题：Claude 的视觉能力和浏览器自动化工具本身的局限，让它难以识别所有种类的 bug。比如 Claude 通过 Puppeteer MCP **看不到浏览器原生的 alert modal**，依赖这类 modal 的功能因此更容易带 bug。

## 让 Agent 快速进入状态

上述基础设施就位后，每个编码 agent 都被 prompt 去走一套"找方向"的步骤——有些非常基本，但确实有用：

1. *执行 `pwd` 看看你当前在哪个目录。你只能编辑这个目录里的文件。*
2. *阅读 git 日志和 progress 文件，了解最近做了哪些工作。*
3. *阅读 features 列表文件，挑出尚未完成、优先级最高的功能去做。*

这种做法**每次会话都能为 Claude 省下一些 token**——它不再需要从头琢磨"该怎么测试这段代码"。同时也很有用的是：让初始化 agent 顺手写一个 `init.sh` 脚本能起动开发服务器，并在开始新功能之前**先跑一遍基本的端到端测试**。

在 claude.ai 克隆这个例子里，这意味着 agent 总是会先启动本地开发服务器，用 Puppeteer MCP 开一个新聊天、发一条消息、收到回复。这样 Claude 能**第一时间发现 app 是否处在损坏状态**、并立即修复已存在的 bug。如果不这么做，直接动手做新功能，反而会**让问题变得更糟**。

基于以上设置，一次典型会话会以这样的 assistant 消息开始：

```text
[Assistant] I'll start by getting my bearings and understanding the current state of the project.
[Tool Use] <bash - pwd>
[Tool Use] <read - claude-progress.txt>
[Tool Use] <read - feature_list.json>
[Assistant] Let me check the git log to see recent work.
[Tool Use] <bash - git log --oneline -20>
[Assistant] Now let me check if there's an init.sh script to restart the servers.
<Starts the development server>
[Assistant] Excellent! Now let me navigate to the application and verify that some fundamental features are still working.
<Tests basic functionality>
[Assistant] Based on my verification testing, I can see that the fundamental functionality is working well. The core chat features, theme switching, conversation loading, and error handling are all functioning correctly. Now let me review the tests.json file more comprehensively to understand what needs to be implemented next.
<Starts work on a new feature>
```

**Agent 失败模式与对策汇总**

| 问题 | 初始化 Agent 的动作 | 编码 Agent 的动作 |
|---|---|---|
| Claude 过早宣布整个项目已完成 | 搭建一份功能清单文件：基于输入规格，生成一份结构化 JSON，列出端到端的功能描述。 | 会话开始时读取功能清单文件，**只选一个功能**开始工作。 |
| Claude 把环境留在带 bug 或未文档化的进展状态 | 写入一个初始 git 仓库和 progress 笔记文件。 | 会话开始时读 progress 笔记和 git commit 日志，并在开发服务器上跑基本测试以发现未文档化的 bug；会话结束时写入一次 git commit 和 progress 更新。 |
| Claude 过早把功能标记为已完成 | 搭建功能清单文件。 | 自验证所有功能；**只有经过细致测试**之后才把功能标记为 "passing"。 |
| Claude 要花时间琢磨怎么把 app 跑起来 | 写一个能启动开发服务器的 `init.sh` 脚本。 | 会话开始时先读 `init.sh`。 |

## 未来工作

本研究展示了一种可能的"长时程 agent harness"方案，让模型能跨多个上下文窗口做出增量推进。但仍有不少开放问题。

最值得探讨的是：**单一通用编码 agent 跨上下文表现最好，还是采用多 agent 架构能取得更优表现**，目前还不清楚。可以合理推测：**像测试 agent、QA agent、代码清理 agent 这样的专门 agent，可能在软件开发生命周期的各类子任务上做得更好**。

此外，本 demo 是为全栈 web 应用开发而优化的。**一个未来方向是把这些发现推广到其他领域**——本文的某些或全部经验，很可能能应用到比如**科学研究、金融建模**等其他需要长时程 agentic 工作的场景中。

### 致谢

由 Justin Young 撰写。特别感谢 David Hershey、Prithvi Rajasakeran、Jeremy Hadfield、Naia Bouscal、Michael Tingley、Jesse Mu、Jake Eaton、Marius Buleandara、Maggie Vo、Pedram Navid、Nadine Yasser、Alex Notov 的贡献。

本工作凝结了 Anthropic 多个团队的集体努力，让 Claude 得以**安全地执行长时程的自主软件工程任务**，特别感谢 code RL 团队与 Claude Code 团队。有意贡献的候选人欢迎在 [anthropic.com/careers](http://anthropic.com/careers) 申请。

### 脚注

1. 我们在此把它们称作"两个不同的 agent"，仅仅是因为它们的**初始用户 prompt** 不同。它们的 system prompt、工具集、整体 agent harness 在其他方面完全相同。
