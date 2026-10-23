# 贡献指南

欢迎校正事实、补来源、加面试题和 JD。正文是给读者用的手册，改动以小、可核对为准。

英文短导览见 [README.en.md](./README.en.md)。行为上的完整说明以本文件为准。

## 用哪种方式开口

| 情况 | 用这个 |
|---|---|
| 某一段写错了、过时了、链接断了，你能指出文件和位置 | [Issue](https://github.com/harrisliangsu/ai-agent-engineer-handbook/issues/new/choose) 或直接 PR |
| 你要加一道题、一条 JD，格式已经按下面准备好 | PR（也可以先开 Issue 对一下范围） |
| 还没落到具体段落：怎么读、求职进度、想讨论一个概念 | [Discussions](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions) |
| 提议新的栏目或题型，还没有成稿 | Discussions 的 **Ideas** |
| 阅读和用法上的提问 | Discussions 的 **Q&A** |
| 自我介绍、进度同步、闲聊 | Discussions 的 **General**。欢迎帖在 [discussions/4](https://github.com/harrisliangsu/ai-agent-engineer-handbook/discussions/4) |

Discussions 适合还没形成 diff 的对话。需要改某一段文字时，请改开 Issue 或 PR，这样能对到文件和行。**Announcements** 留给维护者发更新；投票用 **Polls**。

## 欢迎的改动

- 校正或补充事实，并附一手来源 URL（论文、厂商文档、标准、或带抓取日期的 JD）
- 新增面试题：难度（⭐ / ⭐⭐ / ⭐⭐⭐）+ 约 200–500 字答案 + 来源 URL
- 补充 JD 摘录：公司名 + 岗位名 + 来源 URL + 抓取日期。只拿到摘要时，请在文中写明证据限度
- 修断链、错别字、目录锚点
- 网页阅读体验的缺陷（TOC、窄屏、断掉的章节切换）

工程四层里的新事实，请在 [`engineering-foundations/sources.jsonl`](./engineering-foundations/sources.jsonl) 追加一行，使用下一个 `S##` 编号，并在正文里引用。

## 请不要做的事

- 不要整章重写或删短 `engineering-foundations/` 与 `interview-prep/` 里的长文。局部校正即可
- [`engineering-foundations/overview.md`](./engineering-foundations/overview.md) 是冻结的 v1 合并稿，用来留历史，不要把它改成和现行四章一致
- 驾驭工程一章用的是「两条法则（强约束 + 自愈循环）+ 四大支柱」。不要把「七大组件」写回正文框架
- [`engineering-foundations/run_manifest.json`](./engineering-foundations/run_manifest.json) 是当时调研的历史快照，日常改内容时不要改它
- 不要提交没有来源的数字、薪资或模型排名

## Markdown 和 HTML 要一起改

`*/html/` 是手写维护的网页镜像，不是每次用 pandoc 重新生成的。改了 `.md` 的实质内容，请把同一处改动移植到对应的 `.html`，沿用现有标记（`<h2 id="...">`、`<table>`、`<pre><code>`）。只改 Markdown 会让 [在线阅读](https://harrisliangsu.github.io/ai-agent-engineer-handbook/) 和本地网页落后于源稿。

## 本地预览

```bash
python3 -m http.server 8000
# http://localhost:8000                         门户
# http://localhost:8000/knowledge-map.html      知识图谱
```

只看一组章节时，也可以在 `engineering-foundations/html` 或 `interview-prep/html` 里启动同一个命令。页面使用相对路径，根目录发布到 GitHub Pages 后地址仍然有效。

## Pull request

- 一个 PR 做一件事，标题写清改了哪一节
- 事实类改动在描述里贴来源 URL
- 勾选模板里的核对项，尤其是「HTML 是否已同步」
- 不要顺手重排无关段落

维护者可能直接改字句或请你补来源。没有来源的事实性陈述可能不会合并。
