<div align="center">

# CoreCoder

**A production-oriented coding-agent runtime built from a minimal Python core.**

*tool safety · code retrieval · agent service · multi-agent orchestration · budgets · model routing · reproducible evaluation*

[中文](README_CN.md) | English | [Upstream source-reading series · 8 bilingual essays](article/00-index_EN.md)


[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/liuzhixin352-gif/CoreCoder/actions/workflows/ci.yml/badge.svg?branch=devpilot-v1)](https://github.com/liuzhixin352-gif/CoreCoder/actions)

</div>

- **End-to-end coding-agent workflow.** Fetch a GitHub Issue, inspect the repository, plan and execute a repair, validate it, commit it, push a branch, open a pull request, and react to CI failures.
- **Production-style runtime controls.** Tool permissions, human approval boundaries, checkpoint/resume, tracing, context management, token/cost budgets, and capability-aware model fallback are explicit runtime policies rather than hidden framework behavior.
- **Interviewable architecture.** The orchestration, retrieval, multi-agent roles, budget enforcement, routing, evaluation, and failure boundaries are implemented directly in Python so each engineering trade-off can be explained from code.

## What this fork adds

CoreCoder started from a deliberately small coding-agent core. This fork extends that core into a production-oriented agent runtime while keeping the major control paths explicit.

| Area | Added in this fork |
|---|---|
| GitHub repair workflow | Issue ingestion, repository verification, dedicated repair branches, validation, commits, pushes, pull requests, CI monitoring, and one CI-driven repair retry |
| Runtime safety | Tool permission policy, read/write/execute classification, approval boundaries, repository/worktree safety |
| Orchestration | Extracted workflow orchestration, explicit state machine, checkpoint/resume, human-in-the-loop boundaries |
| Tool ecosystem | Structured repository, code-search, test, and issue tools plus a separate MCP server adapter |
| Runtime | Bounded asynchronous tool execution, structured tracing, and context engineering |
| Retrieval | Repository Code RAG with reproducible retrieval evaluation |
| Service layer | FastAPI agent service |
| Multi-agent | Planner → Coder → Reviewer workflow with isolated role permissions |
| Resource control | Per-run and shared token/cost budgets |
| Model runtime | Capability-aware routing, transient fallback, streaming safety, budget-aware rerouting, role-specific model policies |
| Evaluation | Reproducible evaluation and deterministic benchmark infrastructure covering success, latency, rounds, tool calls, tokens, and dollar cost |

## What this is

CoreCoder is a coding-agent runtime built to make the difficult parts of agent engineering visible: not just the model/tool loop, but the safety, state, retrieval, observability, cost, routing, and recovery policies around it.

The central loop is still intentionally understandable: send context to a model, execute requested tools, append the results, and continue until the model returns a final answer. The project then layers explicit production concerns around that loop instead of delegating them to a large agent framework.

The result is deliberately framework-light. CoreCoder does not use LangChain, LangGraph, AutoGen, or CrewAI as its orchestration runtime. The agent loop, state transitions, permission policy, multi-agent workflow, budget tracking, routing, tracing, and evaluation contracts are implemented directly in Python. Model access remains replaceable through OpenAI-compatible and LiteLLM-backed clients.

The project is intended as both a working engineering system and an interview artifact: each subsystem has tests around its failure boundaries, and the architecture is small enough to explain from request entry to tool execution, fallback, budget enforcement, and final evaluation.

<p align="center">
  <img src="assets/demo_en.png" width="760"
       alt="A real CoreCoder run: corecoder -p asks it to fix buggy.py; the agent reads the file, edits the code, runs it to confirm, and reports what it changed.">
</p>

<p align="center"><sub><i>A minimal CLI repair loop: the agent reads code, edits it, validates the result, and returns a final answer.</i></sub></p>


## Architecture

```text
                  CLI / FastAPI / GitHub Issue
                            │
                            ▼
                 Workflow / Service Layer
                            │
                            ▼
                          Agent
          ┌─────────────────┼──────────────────┐
          │                 │                  │
       Context            Tools             Tracing
          │                 │
          │         ┌───────┼────────┐
          │         │       │        │
          │       Files   Tests   Code Search
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
         capability / cost / role
                  │
                  ▼
          transient fallback
                  │
                  ▼
            Model backend(s)


Planner ─┐
Coder   ─┼── role-specific tools and routing
Reviewer─┘
         │
         └── shared workflow budget


MCP server adapter
      │
      └── separate external tool surface


                    Eval / Benchmark
          success · latency · rounds · tools
                 · tokens · dollar cost
```


### Key boundaries

- **Permissions are checked before tool execution.** Read-only operations can be allowed automatically, while write and execute operations pass through explicit policy decisions and can require human approval.
- **Budgets are enforced independently of model routing.** Routing may choose a cheaper eligible model from estimated remaining cost, but the `BudgetTracker` remains the hard post-response enforcement boundary.
- **Fallback is selective.** Transient provider failures may move to the next eligible model; bad requests, permission failures, budget failures, and other non-retryable errors do not.
- **Streaming fallback is conservative.** Once text has already been emitted to the user, CoreCoder does not silently retry another model and risk duplicated output.
- **Role routing is request-local.** Planner, Coder, and Reviewer can share one routed runtime without mutating shared routing state.
- **Actual cost follows the actual model.** When fallback changes the winning backend, accounting uses the model attached to that response rather than the originally selected wrapper.


## Engineering trade-offs

The project intentionally makes several choices that are useful to discuss in a system-design interview:

- **Custom runtime over a large agent framework.** More code is owned locally, but execution semantics and failure boundaries remain visible and testable.
- **Fail closed for unknown pricing under a cost budget.** A model without known pricing cannot silently bypass cost enforcement.
- **Estimated cost for routing, actual cost for enforcement.** Estimates guide model selection; provider response usage determines the final budget update.
- **Shared budget across multi-agent roles.** Planner, Coder, Reviewer, and revision rounds compete for one workflow-level resource envelope instead of receiving independent hidden budgets.
- **Fallback only before streamed output.** Reliability does not come at the cost of corrupting already-visible responses.
- **Deterministic evaluation before live-model benchmarking.** Runtime correctness can be tested cheaply and reproducibly; model-quality evaluation can be layered on separately.

## Deterministic runtime benchmark

CoreCoder includes a small deterministic benchmark for exercising runtime
contracts without API calls or model variance.

Both targets run through the real `corecoder.Agent` runtime. The LLM behavior
is scripted and deterministic; the difference is that the baseline never
uses tools, while the advanced target exercises the actual permission,
tool-execution, tracing, and budget-accounting paths.

| Target | Success | Direct response | Single tool | Two tools | Tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 33.3% | PASS | FAIL | FAIL | 75 | $0.0003 |
| Advanced | 100.0% | PASS | PASS | PASS | 125 | $0.0005 |

The three versioned cases are:

```text
direct-response
  1 LLM round
  0 tool calls

single-tool
  2 LLM rounds
  exactly 1 tool call

tool-chain
  exactly 2 tool calls
```

For the advanced target, tool calls are executed by the real CoreCoder Agent
through its permission policy and asynchronous tool runtime. The resulting
`tool.permission`, `tool.started`, and `tool.completed` events come from the
production tracing path rather than being synthesized by the benchmark
fixture.

Token usage is deterministic test usage attached to the scripted LLM
responses. Dollar cost is then calculated through CoreCoder's normal budget
accounting using the project's pricing table.

This benchmark measures **runtime behavior, not live-model coding quality**.
It does not claim that CoreCoder solves 100% of real coding tasks or
outperforms external coding agents. Live-model and repository-level coding
quality remain a separate evaluation layer.


## Quick start

The current DevPilot runtime is developed on the `devpilot-v1` branch. Until the final release is promoted to `main`, clone that branch explicitly:

```bash
git clone --branch devpilot-v1 --single-branch https://github.com/liuzhixin352-gif/CoreCoder
cd CoreCoder
pip install -e .
```

CoreCoder uses an OpenAI-compatible client by default. Configure a model and API key with environment variables:

| Provider | Example env vars |
|---|---|
| OpenAI (default `gpt-5.5`) | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| Local Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Other OpenAI-compatible providers can use the same configuration pattern. For broader provider support, install the optional LiteLLM backend:

```bash
pip install -e ".[litellm]"
```

Then set `CORECODER_PROVIDER=litellm` in the environment or `.env` file.

API keys can be exported directly or placed in a `.env` file. Then run either the interactive CLI or one-shot mode:

```bash
corecoder                                             # interactive REPL
corecoder -p "add error handling to parse_config()"   # one-shot mode, exits when done
```

### HTTP service

Install the service dependencies:

```bash
pip install -e ".[service]"
```

Start the FastAPI service:

```bash
corecoder-service
```

The service exposes:

```text
GET  /health
POST /chat
```

A chat request contains a message and an optional session ID; the response includes the agent response, session ID, and run ID.

### DevPilot GitHub Issue workflow

DevPilot turns a GitHub Issue into a guarded repository repair workflow:

```text
GitHub Issue
    ↓
repository verification
    ↓
worktree safety check
    ↓
dedicated repair branch
    ↓
Agent repair
    ↓
local validation
    ↓
human commit approval
    ↓
commit → push → Pull Request
    ↓
GitHub Actions CI
    ├── success → completed
    └── failure → one bounded CI-driven repair retry
```

#### Start with a read-only dry run

```bash
corecoder --issue https://github.com/owner/repository/issues/12 --dry-run
```

Dry-run mode performs analysis without modifying the repository. Its tool
profile is restricted to:

```text
read_file
glob
grep
repo_map
```

It does not create a repair branch, run post-repair validation, create a
commit, push code, or open a Pull Request.

#### Repository and worktree safety

For a real repair:

```bash
corecoder --issue https://github.com/owner/repository/issues/12
```

Repository verification is fail-closed:

| Repository status | Dry run | New real repair |
|---|---|---|
| `exact` | allowed | allowed |
| `fork` | allowed | allowed |
| `mismatch` | blocked | blocked |
| `unknown` | allowed read-only | blocked by default |

An `unknown` repository can be overridden only after independent
verification:

```bash
corecoder --issue https://github.com/owner/repository/issues/12 \
  --allow-unverified-repository
```

The override does not bypass a confirmed mismatch.

A new real repair must also start inside a clean Git worktree. Existing
uncommitted work is rejected so Issue changes cannot be mixed with unrelated
local changes.

#### Dedicated branch and workflow checkpoints

A new real repair creates a dedicated branch before Agent execution:

```text
devpilot/issue-21-fix-repository-scan-limit
```

The workflow persists checkpoints as it moves through validation, approval,
commit, push, Pull Request, and CI states.

To continue a previously checkpointed workflow:

```bash
corecoder --issue https://github.com/owner/repository/issues/12 \
  --resume-workflow
```

Resume mode requires the saved workflow to belong to the requested Issue and
the current Git branch to match the saved repair branch. Resume mode is exempt
from the clean-worktree start check so checkpointed uncommitted repair state
can be continued; the checkpoint and branch identity checks still apply.

#### Validation and human approval

When the Agent produces repository changes, DevPilot runs local validation
before any commit is created:

```text
Post-repair validation
Command: python -m pytest tests -q
Status: passed
Passed: 275
```

Failed, interrupted, invalid, empty, timed-out, or unstartable test runs stop
the workflow with a non-zero exit status. The repair branch and current
changes are preserved for inspection.

After validation succeeds, the CLI presents the changed files and validation
result and asks for explicit approval before creating the initial repair
commit:

```text
Commit approval required
Approve commit? [approve/reject] >
```

Rejecting approval stops the workflow before commit, push, or Pull Request
creation.

#### Commit, push, and Pull Request

After approval, DevPilot:

```text
creates a repair commit with a deterministic message
    ↓
verifies and pushes the repair branch to origin
    ↓
opens a Pull Request against the branch
that was checked out before the repair started
```

The commit message uses:

```text
Fix #<Issue number>: <Issue title>
```

The Pull Request base is therefore not hard-coded to `main` or
`devpilot-v1`; it follows the branch from which the repair workflow began.

#### CI monitoring and bounded repair

After the Pull Request is created, DevPilot polls GitHub check runs for the
repair commit. The default polling interval is 5 seconds with a 300-second
timeout.

Final CI states are:

```text
success
failure
```

If the first CI result is `failure`, DevPilot performs exactly one
CI-driven repair attempt using failed check-run context:

```text
failed CI
   ↓
prepare bounded failure context
   ↓
Agent repair
   ↓
local validation
   ↓
new repair commit
   ↓
push same repair branch
   ↓
wait for CI again
```

This retry is intentionally bounded to one attempt. A second CI failure, a
retry that produces no repository changes, or a CI timeout terminates the
workflow with a non-zero exit status.

## Code map

The runtime is split by responsibility rather than hidden behind a large
agent framework. These are the main entry points for an architecture or
system-design walkthrough:

| Module | Responsibility |
|---|---|
| `agent.py` | Core model/tool loop plus permission, context, tracing, and budget integration |
| `issue_orchestration.py` | Explicit GitHub Issue workflow state machine, checkpoints, resume, and CI-repair lifecycle |
| `repository_guard.py` | Repository identity, Git worktree, and clean-start safety checks |
| `repair_branch.py` | Dedicated repair branch creation |
| `post_repair.py` / `post_repair_validation.py` | Change collection and mandatory local validation |
| `repair_commit.py` / `repair_push.py` / `repair_pr.py` | Commit, push, and Pull Request lifecycle |
| `repair_ci.py` | CI polling, bounded failure context, and one CI-driven repair retry |
| `context.py` | Context estimation and compression |
| `permissions.py` | Read/write/execute permission policy and approval decisions |
| `tool_runtime.py` | Bounded asynchronous tool execution |
| `code_rag.py` | Repository indexing and code retrieval |
| `tools/` | Built-in file, shell, search, test, repository, code-search, issue, and sub-agent tools |
| `mcp_tools.py` | Separate MCP server adapter |
| `agent_service.py` / `service.py` | Session-aware Agent service and FastAPI entry point |
| `multi_agent.py` | Planner → Coder → Reviewer orchestration and role-specific tool policies |
| `budget.py` | Token and dollar-budget limits, usage, and enforcement |
| `model_catalog.py` | Model capabilities and pricing metadata |
| `model_router.py` | Capability-, role-, and budget-aware model selection |
| `routed_llm.py` | Routed execution, transient fallback, response attribution, and streaming safety |
| `tracing.py` | Structured runtime events |
| `eval.py` | Evaluation metrics and report primitives |
| `benchmark.py` | Versioned deterministic benchmark and multi-target comparison infrastructure |

A useful reading path for the current fork is:

```text
request
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

## Runtime flow

The model/tool loop is still the center of CoreCoder, but most production
behavior comes from the policies around that loop.

For a normal Agent round:

```text
user input
   ↓
context preparation
   ↓
budget pre-check
   ↓
model routing
   ↓
LLM response
   ├── final text ─────────────────→ return
   │
   └── tool calls
          ↓
      permission policy
          ↓
      allow / ask / deny
          ↓
      bounded tool execution
          ↓
      append results
          ↓
      next model round
```

For a GitHub Issue repair, that inner Agent loop runs inside the larger
stateful repository workflow described above.

The separation is intentional:

```text
Agent
  owns model/tool interaction

Issue workflow
  owns repository lifecycle, checkpoints,
  validation, approval, Git operations, and CI recovery

BudgetTracker
  owns hard resource enforcement

Model router
  owns model-selection policy

Tracing / Eval / Benchmark
  observe and measure runtime behavior
```

This keeps failure boundaries explicit instead of making one object
responsible for the entire system.

## Upstream source-reading series · 8 bilingual essays

The original CoreCoder project includes a bilingual source-reading series
covering the minimal Agent core from which this fork started.

The essays are still useful for understanding the foundational loop, tools,
provider wrapper, context management, sessions, and sub-agent ideas. They
describe the upstream teaching-oriented core rather than every subsystem in
this fork, so the architecture and runtime sections above are the
authoritative overview of the current project.

- **[Intro · Read Claude Code through CoreCoder, then build your own](article/00-index_EN.md)**
- **[01 · An agent, at its core, is a `while` loop](article/01-the-loop_EN.md)** — the original main loop, interrupts, and round limit
- **[02 · The tool system: letting the model act, safely](article/02-tools_EN.md)** — the original tool system and bash safety gate
- **[03 · Plug in any LLM, and keep the bill honest](article/03-llm-and-cost_EN.md)** — the original provider wrapper, retry, and cost-accounting design
- **[04 · Surviving a long task on a finite window](article/04-context_EN.md)** — context compression and orphaned tool messages
- **[05 · Parallel execution and sub-agents](article/05-parallel-and-subagents_EN.md)** — the original concurrency and sub-agent design
- **[06 · Turning it into a real command-line tool](article/06-session-and-cli_EN.md)** — sessions and path-traversal defense
- **[07 · Fork CoreCoder into your own coding agent](article/07-build-your-own_EN.md)** — extending the original minimal core

## Extension points

CoreCoder now implements many of the production concerns that were
intentionally absent from the original minimal core. The remaining gaps are
also useful system-design directions:

- **Real sandbox isolation.** Permission policy controls whether a tool may execute, but hostile-code isolation ultimately belongs at the process, container, or OS boundary.
- **Persistent production state.** Workflow checkpoints are stored durably, but a deployed system could move workflow state into a database or dedicated workflow engine.
- **Distributed execution.** Tool concurrency is explicit and bounded locally; larger deployments could move tools and Agent roles onto independent workers.
- **Live-model coding benchmarks.** The deterministic benchmark isolates runtime contracts; a separate benchmark can measure real coding quality across repositories, models, latency, and cost.
- **Adaptive routing.** Routing is currently deterministic and policy-driven. Historical traces could later support empirical or learned model selection.
- **Production telemetry backends.** Structured trace events already exist and can be exported to a larger observability stack.

## CLI commands

Inside the interactive REPL, `/help` shows the available commands. Common
ones include:

```text
/help            show command help
/reset           reset the conversation
/model <name>    switch model
/compact         compact context manually
/tokens          show token usage and estimated cost
/diff            show files modified in this session
/save            save the current session
/sessions        list saved sessions
quit / exit      exit the REPL
```

GitHub Issue workflows use the separate `--issue`, `--dry-run`,
`--allow-unverified-repository`, and `--resume-workflow` CLI flags described
earlier.

## Development checks

Before submitting a change, run:

```bash
python -m pytest -q
python -m ruff check corecoder tests
python -m compileall -q corecoder tests
```

The project targets Python 3.10+.

## Upstream attribution

CoreCoder originated from
[Yufeng He's CoreCoder project](https://github.com/he-yufeng/CoreCoder), a
compact educational implementation for understanding coding-agent internals.

This fork extends that foundation with the DevPilot GitHub repair workflow,
repository and tool safety policies, explicit workflow state and checkpoints,
human approval boundaries, MCP support, bounded asynchronous tool execution,
structured tracing, Code RAG, a FastAPI service, Planner/Coder/Reviewer
orchestration, Token/cost budgets, capability-aware model routing and
fallback, and reproducible evaluation and benchmark infrastructure.

The upstream source-reading articles are retained as learning material. See
the repository license for reuse terms.

> CoreCoder was formerly named NanoCoder in the upstream project. It was
> renamed to avoid confusion with
> [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder).