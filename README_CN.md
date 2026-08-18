<div align="center">

# CoreCoder

**一个从极简 Python 核心演进而来的、面向工程实践的 coding-agent runtime。**

*工具安全 · 代码检索 · Agent 服务 · 多 Agent 编排 · Token/成本预算 · 模型路由 · 可复现实验评估*

中文 | [English](README.md) | [上游源码导读 · 八篇双语](article/)

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/liuzhixin352-gif/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/liuzhixin352-gif/CoreCoder/actions)

</div>

- **端到端 coding-agent 工作流。** 从 GitHub Issue 获取任务，检查仓库、规划与执行修复、本地验证、提交、推送分支、创建 Pull Request，并对 CI 失败执行一次有界的自动修复。
- **生产式 runtime 控制。** 工具权限、人工审批边界、checkpoint/resume、Tracing、上下文管理、Token/成本预算以及能力感知的模型 fallback 都是显式的运行时策略。
- **适合面试讲解的架构。** Orchestration、检索、多 Agent 角色、预算控制、模型路由、评估和失败边界都直接用 Python 实现，可以从源码解释每一个工程取舍。

## 这个 fork 增加了什么

CoreCoder 起点是一个刻意保持精简的 coding-agent 核心。这个 fork 在保留核心执行路径可读性的同时，把它扩展成了一套更接近生产工程的 Agent runtime。

| 领域 | 这个 fork 新增的能力 |
|---|---|
| GitHub 修复工作流 | Issue 获取、仓库验证、专用修复分支、本地验证、提交、推送、Pull Request、CI 监控，以及一次 CI 驱动的修复重试 |
| Runtime 安全 | 工具权限策略、read/write/execute 分类、人工审批边界、仓库与工作区安全检查 |
| Orchestration | 独立的工作流编排、显式状态机、checkpoint/resume、human-in-the-loop 边界 |
| 工具体系 | 仓库、代码搜索、测试和 Issue 工具，以及独立的 MCP server adapter |
| Runtime | 有并发上限的异步工具执行、结构化 tracing、上下文工程 |
| 检索 | Repository Code RAG 与可复现的检索评估 |
| 服务层 | FastAPI Agent Service |
| 多 Agent | Planner → Coder → Reviewer 工作流与角色隔离的工具权限 |
| 资源控制 | 单次运行及共享工作流级别的 Token/美元成本预算 |
| 模型 Runtime | 能力感知路由、瞬态错误 fallback、流式输出安全、预算感知重路由、角色特定模型策略 |
| 评估 | 可复现评估与确定性 benchmark 基础设施，记录成功率、延迟、轮次、工具调用、Token 与美元成本 |

## 这是什么

CoreCoder 是一套 coding-agent runtime。它关注的不只是“模型调用工具”这个核心循环，还把安全、状态管理、检索、可观测性、成本、模型路由和失败恢复这些真正进入工程环境后必须面对的问题显式实现出来。

中心循环仍然刻意保持简单：把上下文发送给模型，执行模型请求的工具，把工具结果追加回上下文，然后继续，直到模型返回最终答案。项目在这个循环周围逐层增加生产环境需要的控制，而不是把这些行为隐藏在大型 Agent framework 里面。

这个项目有意保持 framework-light。CoreCoder 不使用 LangChain、LangGraph、AutoGen 或 CrewAI 作为 orchestration runtime。Agent loop、状态转换、权限策略、多 Agent 工作流、预算跟踪、模型路由、Tracing 和评估协议都直接在 Python 中实现。模型层则可以通过 OpenAI-compatible client 或 LiteLLM backend 替换。

因此它既是一套可以工作的工程系统，也是一份适合面试展示的项目：每个关键子系统都有围绕失败边界设计的测试，同时整个架构仍然能够从请求入口一路解释到工具执行、模型 fallback、预算控制和最终评估。

<p align="center">
  <img src="assets/demo.png" width="820"
       alt="CoreCoder 一次真实运行：corecoder -p 让它修 buggy.py，agent 自己读取代码、修改文件、运行验证并返回结果">
</p>

<p align="center"><sub><i>一个最小 CLI 修复回路：Agent 读取代码、修改文件、验证结果，然后返回最终答案。</i></sub></p>

## 架构

```text
                 CLI / FastAPI / GitHub Issue
                            │
                            ▼
                    工作流 / 服务层
                            │
                            ▼
                          Agent
          ┌─────────────────┼──────────────────┐
          │                 │                  │
        Context           Tools             Tracing
          │                 │
          │         ┌───────┼────────┐
          │         │       │        │
          │       文件    测试     Code Search
          │                          │
          │                          ▼
          │                       Code RAG
          │
          ├──────────────┐
          │              │
          ▼              ▼
 Permission Policy   BudgetTracker
      + HITL              │
                          ▼
                       RoutedLLM
                  ┌───────┴────────┐
                  │                │
             Model Router     Model Catalog
                  │
               能力 / 成本 / 角色
                  │
                  ▼
             瞬态错误 fallback
                  │
                  ▼
             Model backend(s)


Planner ─┐
Coder   ─┼── 角色特定工具集与模型路由
Reviewer─┘
         │
         └── 共享工作流预算


MCP server adapter
      │
      └── 独立的外部工具接口


                    Eval / Benchmark
             成功率 · 延迟 · 轮次 · 工具调用
                    · Token · 美元成本
```

### 关键边界

- **工具执行前必须经过权限判断。** 只读操作可以由策略自动允许；写入和执行操作必须经过显式策略判断，并可以要求人工批准。
- **预算控制独立于模型路由。** 路由可以根据预估剩余成本选择更便宜的合格模型，但 `BudgetTracker` 仍然是响应返回后的硬性预算执行边界。
- **Fallback 只处理合适的失败类型。** 瞬态 provider 故障可以切换到下一个合格模型；错误请求、权限失败、预算失败以及其他不可重试错误不会触发 fallback。
- **流式 fallback 保守处理。** 一旦已经向用户输出文本，CoreCoder 不会静默切换模型重新生成，以避免重复或损坏已经可见的响应。
- **角色路由是 request-local 的。** Planner、Coder 和 Reviewer 可以共享同一个 routed runtime，而不需要修改共享的全局路由状态。
- **实际成本跟随实际成功的模型。** 如果 fallback 改变了最终返回结果的模型，成本统计使用响应真正对应的模型，而不是最初选择的 wrapper。

## 工程取舍

这个项目有几项刻意保留为显式设计的选择，也很适合在 system-design 面试中讨论：

- **自定义 runtime，而不是依赖大型 Agent framework。** 项目自己维护更多代码，但执行语义和失败边界保持可见、可测试。
- **存在美元预算时，对未知价格 fail closed。** 没有已知价格的模型不能静默绕过成本约束。
- **路由使用预估成本，预算执行使用实际成本。** 预估值帮助选择模型，provider 响应中的实际 usage 决定最终预算更新。
- **多 Agent 共享工作流预算。** Planner、Coder、Reviewer 和 revision round 竞争同一个资源额度，而不是每个角色拥有彼此独立的隐藏预算。
- **只在尚未输出流式文本时 fallback。** 提高可靠性的同时，不以破坏用户已经看到的响应为代价。
- **先做确定性 runtime 评估，再做真实模型 benchmark。** Runtime 行为可以廉价、可复现地测试；真实模型 coding quality 可以作为独立评估层加入。

## 确定性 Runtime Benchmark

CoreCoder 包含一套小型确定性 benchmark，用来在不调用真实 API、也不受模型
随机性影响的情况下验证 Runtime contract。

两个 target 都经过真实的 `corecoder.Agent` runtime。LLM 行为由确定性的
script 驱动；区别在于 Baseline 从不使用工具，而 Advanced 会真正经过权限
策略、工具执行、Tracing 和预算统计路径。

| Target | 成功率 | 直接回答 | 单工具 | 两工具 | Token | 成本 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 33.3% | PASS | FAIL | FAIL | 75 | $0.0003 |
| Advanced | 100.0% | PASS | PASS | PASS | 125 | $0.0005 |

当前版本包含三个确定性 case：

```text
direct-response
  1 个 LLM round
  0 次工具调用

single-tool
  2 个 LLM round
  恰好 1 次工具调用

tool-chain
  恰好 2 次工具调用
```

Advanced target 的工具调用由真实 CoreCoder Agent 执行，并真正经过权限策略
和异步工具 Runtime。`tool.permission`、`tool.started` 和
`tool.completed` 等事件来自正式 Tracing 路径，而不是由 benchmark fixture
人工伪造。

Token 用量来自 scripted LLM response 中固定的测试 usage；美元成本随后通过
CoreCoder 正常的预算统计逻辑，并使用项目自身的 pricing table 计算。

这套 benchmark 衡量的是 **Runtime 行为，而不是真实模型的 coding quality**。
它不表示 CoreCoder 能解决 100% 的真实编程任务，也不表示它优于其他 coding
agent。真实模型和真实代码仓库上的 coding quality 应作为独立评估层进行。


## 快速开始


```markdown
Clone 仓库并以 editable mode 安装 CoreCoder：

```bash
git clone https://github.com/liuzhixin352-gif/CoreCoder
cd CoreCoder
pip install -e .
```

CoreCoder 默认使用 OpenAI-compatible client。通过环境变量配置模型和 API key：

| Provider | 环境变量示例 |
|---|---|
| OpenAI（默认 `gpt-5.5`） | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| 本地 Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

其他 OpenAI-compatible provider 可以使用相同配置方式。如需更广泛的 provider 支持，可以安装可选的 LiteLLM backend：

```bash
pip install -e ".[litellm]"
```

然后在环境变量或 `.env` 文件中设置 `CORECODER_PROVIDER=litellm`。

API key 可以直接设置为环境变量，也可以放在项目目录的 `.env` 文件中。

运行交互式 CLI 或一次性模式：

```bash
corecoder
corecoder -p "给 parse_config() 加错误处理"
```

### HTTP 服务

安装 service 依赖：

```bash
pip install -e ".[service]"
```

启动 FastAPI service：

```bash
corecoder-service
```

服务暴露以下接口：

```text
GET  /health
POST /chat
```

`/chat` 请求包含消息以及可选的 session ID；响应包含 Agent 返回内容、session ID 和 run ID。


### DevPilot GitHub Issue 工作流

DevPilot 会把一个 GitHub Issue 转换成受安全边界约束的仓库修复工作流：

```text
GitHub Issue
    ↓
仓库验证
    ↓
Git 工作区安全检查
    ↓
专用修复分支
    ↓
Agent 修复
    ↓
本地验证
    ↓
人工提交审批
    ↓
commit → push → Pull Request
    ↓
GitHub Actions CI
    ├── 成功 → 完成
    └── 失败 → 一次有界的 CI 驱动修复
```

#### 先进行只读分析

```bash
corecoder --issue https://github.com/owner/repository/issues/12 --dry-run
```

`--dry-run` 只分析问题，不修改仓库。可用工具被限制为：

```text
read_file
glob
grep
repo_map
```

只读模式不会创建修复分支，也不会执行修复后验证、创建 commit、推送代码
或创建 Pull Request。

#### 仓库与工作区安全

执行真实修复：

```bash
corecoder --issue https://github.com/owner/repository/issues/12
```

仓库验证采用 fail-closed 策略：

| 仓库状态 | 只读分析 | 新建真实修复 |
|---|---|---|
| `exact` | 允许 | 允许 |
| `fork` | 允许 | 允许 |
| `mismatch` | 拒绝 | 拒绝 |
| `unknown` | 允许只读分析 | 默认拒绝 |

只有在用户已经独立确认当前工作目录正确后，才能显式覆盖 `unknown` 状态：

```bash
corecoder --issue https://github.com/owner/repository/issues/12 \
  --allow-unverified-repository
```

该参数不能绕过已经确认的 repository mismatch。

一次新的真实修复还必须从干净的 Git worktree 开始。存在未提交修改时会直接
拒绝启动，避免把 Issue 修复内容与用户原有修改混在一起。

#### 专用分支与 workflow checkpoint

新的真实修复会在 Agent 启动前创建专用分支：

```text
devpilot/issue-21-fix-repository-scan-limit
```

工作流会在验证、审批、commit、push、Pull Request 和 CI 等状态之间推进时
持续保存 checkpoint。

恢复之前已经保存的工作流：

```bash
corecoder --issue https://github.com/owner/repository/issues/12 \
  --resume-workflow
```

恢复模式会检查 checkpoint 是否属于当前 Issue，同时要求当前 Git 分支与
checkpoint 中保存的 repair branch 一致。

恢复模式不执行“新建修复必须 clean worktree”的启动检查，从而允许继续处理
checkpoint 阶段保留下来的未提交修复状态；checkpoint 与 repair branch 的
身份检查仍然必须通过。

#### 本地验证与人工审批

Agent 产生仓库修改后，DevPilot 会在创建任何 commit 之前先运行本地验证：

```text
Post-repair validation
Command: python -m pytest tests -q
Status: passed
Passed: 275
```

测试失败、中断、配置无效、没有发现测试、启动失败或超过超时时间，都会使
工作流以非零状态码停止。修复分支和当前修改会被保留，供人工检查。

验证成功后，CLI 会展示修改文件与验证结果，并在创建**第一次修复提交**
之前要求显式人工批准：

```text
Commit approval required
Approve commit? [approve/reject] >
```

如果拒绝批准，工作流会在 commit、push 和 Pull Request 创建之前停止。

#### Commit、push 与 Pull Request

批准后，DevPilot 会依次执行：

```text
使用确定性 message 创建 repair commit
    ↓
验证并将修复分支推送到 origin
    ↓
创建 Pull Request，目标分支为
修复工作流启动前检出的原始分支
```

提交信息格式为：

```text
Fix #<Issue 编号>: <Issue 标题>
```

因此 Pull Request 的 base branch **不是硬编码的 `main` 或
`devpilot-v1`**，而是修复任务开始时所在的分支。

#### CI 监控与有界修复

创建 Pull Request 后，DevPilot 会持续轮询该修复 commit 对应的 GitHub
check runs。默认每 5 秒检查一次，最长等待 300 秒。

最终 CI 状态为：

```text
success
failure
```

如果第一次 CI 结果为 `failure`，DevPilot 会利用失败 check-run 的上下文，
自动执行**一次** CI 驱动修复：

```text
CI 失败
   ↓
构造有界失败上下文
   ↓
Agent 修复
   ↓
再次本地验证
   ↓
创建新的 repair commit
   ↓
推送同一个 repair branch
   ↓
再次等待 CI
```

这个 retry 被刻意限制为一次。第二次 CI 仍失败、二次修复没有产生仓库修改，
或者 CI 等待超时，都会让工作流以非零状态码结束。

## 代码地图

当前 runtime 按职责拆分，而不是把执行逻辑隐藏在大型 Agent framework
里面。下面这些是进行架构或 system-design walkthrough 时最值得看的入口：

| 模块 | 职责 |
|---|---|
| `agent.py` | 核心模型/工具循环，以及权限、Context、Tracing 和预算集成 |
| `issue_orchestration.py` | 显式 GitHub Issue 状态机、checkpoint、resume 与 CI 修复生命周期 |
| `repository_guard.py` | 仓库身份、Git worktree 与 clean-start 安全检查 |
| `repair_branch.py` | 创建专用修复分支 |
| `post_repair.py` / `post_repair_validation.py` | 修改收集与强制本地验证 |
| `repair_commit.py` / `repair_push.py` / `repair_pr.py` | Commit、push 与 Pull Request 生命周期 |
| `repair_ci.py` | CI 轮询、有界失败上下文以及一次 CI 驱动修复 |
| `context.py` | Context 估算与压缩 |
| `permissions.py` | read/write/execute 权限策略与审批决策 |
| `tool_runtime.py` | 有并发上限的异步工具执行 |
| `code_rag.py` | Repository indexing 与代码检索 |
| `tools/` | 内置文件、shell、搜索、测试、仓库、代码检索、Issue 与子 Agent 工具 |
| `mcp_tools.py` | 独立 MCP server adapter |
| `agent_service.py` / `service.py` | Session-aware Agent service 与 FastAPI 入口 |
| `multi_agent.py` | Planner → Coder → Reviewer 编排与角色特定工具策略 |
| `budget.py` | Token/美元预算限制、使用量与强制执行 |
| `model_catalog.py` | 模型能力与价格元数据 |
| `model_router.py` | 能力、角色和预算感知的模型选择 |
| `routed_llm.py` | 路由执行、瞬态错误 fallback、响应模型归属与流式安全 |
| `tracing.py` | 结构化 runtime event |
| `eval.py` | 评估指标与报告基础设施 |
| `benchmark.py` | 版本化确定性 benchmark 与多 target 比较基础设施 |

当前 fork 推荐的阅读路径是：

```text
请求
  ↓
CLI / service / GitHub Issue workflow
  ↓
issue_orchestration.py
  ↓
agent.py
  ├── context.py
  ├── permissions.py
  ├── tools/
  │      └── code_search.py
  ├── code_rag.py
  ├── budget.py
  └── routed_llm.py
         ├── model_router.py
         └── model_catalog.py
  ↓
tracing.py
  ↓
eval.py / benchmark.py
```

## Runtime 流程

模型/工具循环仍然是 CoreCoder 的中心，但大多数生产式行为来自这个循环
周围的显式策略。

一次普通 Agent round：

```text
用户输入
   ↓
Context 准备
   ↓
预算预检查
   ↓
模型路由
   ↓
LLM response
   ├── 最终文本 ─────────────────→ 返回
   │
   └── tool calls
          ↓
        权限策略
          ↓
      allow / ask / deny
          ↓
      有界工具执行
          ↓
       追加结果
          ↓
      下一轮模型调用
```

GitHub Issue 修复时，这个内部 Agent loop 会运行在前面介绍的、更大的
有状态仓库工作流中。

各层职责被刻意拆开：

```text
Agent
  负责模型与工具交互

Issue workflow
  负责仓库生命周期、checkpoint、
  验证、审批、Git 操作和 CI 恢复

BudgetTracker
  负责硬性资源限制

Model router
  负责模型选择策略

Tracing / Eval / Benchmark
  负责观察和衡量 runtime 行为
```

这样可以保持失败边界清楚，而不是让一个对象承担整个系统的所有职责。

## 上游源码导读 · 八篇双语

原始 CoreCoder 项目包含一套双语源码导读，介绍这个 fork 起点所使用的
极简 Agent 核心。

这些文章仍然适合理解基础模型循环、工具系统、provider wrapper、
Context 管理、Session 和子 Agent 等概念。不过文章描述的是上游教学型核心，
并不覆盖这个 fork 后来增加的所有子系统，因此当前项目应以本 README
前面的架构和 Runtime 章节为准。

- **[导言 · 用 CoreCoder 读懂 Claude Code，再造一个你自己的](article/00-index.md)**
- **[01 一个 Agent 的本体，是一个 `while` 循环](article/01-the-loop.md)** — 上游主循环、打断与轮次限制
- **[02 工具系统：让模型安全地动手](article/02-tools.md)** — 上游原始工具系统与 bash 安全闸
- **[03 接入任意大模型，顺便把账算清楚](article/03-llm-and-cost.md)** — 上游 provider wrapper、重试与成本统计设计
- **[04 用有限的窗口扛住一个长任务](article/04-context.md)** — Context 压缩与孤儿 tool message
- **[05 并行执行与子 Agent](article/05-parallel-and-subagents.md)** — 上游并发与子 Agent 设计
- **[06 把它跑成一个真正的命令行工具](article/06-session-and-cli.md)** — Session 与路径穿越防护
- **[07 Fork CoreCoder，搭一个你自己的 coding agent](article/07-build-your-own.md)** — 从上游极简核心继续扩展

## 后续扩展方向

CoreCoder 已经实现了很多上游极简核心刻意没有包含的生产式能力。剩余边界
也正好可以继续作为 system-design 方向：

- **真正的 sandbox isolation。** Permission policy 能控制工具是否允许执行，但面对不可信代码时，真正的隔离最终属于进程、容器或操作系统层。
- **生产级持久化状态。** Workflow checkpoint 已经可以持久保存，但真正部署时可以进一步把工作流状态迁移到数据库或专用 workflow engine。
- **分布式执行。** 当前工具并发在本地显式且有上限；更大规模部署可以把工具执行与不同 Agent role 拆到独立 worker。
- **真实模型 coding benchmark。** 确定性 benchmark 用来隔离 Runtime contract；可以再加入跨仓库、跨模型，同时衡量 coding quality、延迟和成本的真实模型评估。
- **自适应路由。** 当前路由是确定性的 policy-driven 选择；未来可以利用历史 trace 做经验式甚至学习式模型选择。
- **生产 observability backend。** 当前已经存在结构化 trace event，可以进一步接入完整 telemetry stack。

## CLI 命令

交互式 REPL 中可以使用 `/help` 查看命令。常用命令包括：

```text
/help            查看命令帮助
/reset           重置当前对话
/model <name>    切换模型
/compact         手动压缩 Context
/tokens          查看 Token 用量和成本估算
/diff            查看当前 Session 修改过的文件
/save            保存当前 Session
/sessions        列出已保存 Session
quit / exit      退出 REPL
```

GitHub Issue workflow 使用前面介绍的 `--issue`、`--dry-run`、
`--allow-unverified-repository` 和 `--resume-workflow` 参数。

## 开发检查

提交修改前运行：

```bash
python -m pytest -q
python -m ruff check corecoder tests
python -m compileall -q corecoder tests
```

项目支持 Python 3.10+。

## 上游归属

CoreCoder 起源于
[Yufeng He 的 CoreCoder 项目](https://github.com/he-yufeng/CoreCoder)，
原项目是一套用于理解 coding-agent 内部机制的紧凑教学实现。

这个 fork 在该基础上加入了 DevPilot GitHub 修复工作流、仓库与工具安全策略、
显式 workflow state 与 checkpoint、人工审批边界、MCP、带并发上限的异步
工具执行、结构化 Tracing、Code RAG、FastAPI service、
Planner/Coder/Reviewer 编排、Token/成本预算、能力感知模型路由与 fallback，
以及可复现的评估和 benchmark 基础设施。

上游源码导读继续作为学习材料保留。复用条款请参见仓库 License。

> CoreCoder 在上游项目中原名 NanoCoder，后来为了避免与
> [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder)
> 混淆而更名。
