"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .tools import ALL_TOOLS
from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .github_issue import (
    GitHubIssueFetchError,
    GitHubIssueReferenceError,
    fetch_github_issue,
)
from .issue_workflow import (
    IssueWorkflowError,
    build_issue_repair_prompt,
)
from .repair_branch import (
    RepairBranchError,
    create_repair_branch,
)
from .repository_guard import (
    RepositoryGuardError,
    check_issue_repository,
    check_worktree,
)
from .session import save_session, load_session, list_sessions
from . import __version__

console = Console()
_DRY_RUN_TOOL_NAMES = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "repo_map",
    }
)


def _tools_for_issue_workflow(*, dry_run: bool):
    """Return the Agent tool profile for an Issue workflow."""
    if not dry_run:
        return None

    return [
        tool
        for tool in ALL_TOOLS
        if tool.name in _DRY_RUN_TOOL_NAMES
    ]

def _format_local_repositories(preflight) -> str:
    """Return local GitHub repositories for CLI messages."""
    if not preflight.local_repositories:
        return "no GitHub remotes found"

    return ", ".join(
        repository.full_name
        for repository in preflight.local_repositories
    )

def _parse_args():
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-5.5)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    input_group = p.add_mutually_exclusive_group()

    input_group.add_argument(
        "-p",
        "--prompt",
        help="One-shot prompt (non-interactive mode)",
    )
    input_group.add_argument(
        "--issue",
        metavar="URL",
        help=(
            "Fetch a GitHub Issue URL and run a structured "
            "DevPilot repair workflow"
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze the GitHub Issue and propose a repair "
            "without modifying files"
        ),
    )
    p.add_argument(
        "--allow-unverified-repository",
        action="store_true",
        help=(
            "Allow a real Issue repair when the current "
            "GitHub repository cannot be verified"
        ),
    )
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args()

    if args.dry_run and not args.issue:
        p.error("--dry-run requires --issue")

    if args.allow_unverified_repository and not args.issue:
        p.error(
        "--allow-unverified-repository requires --issue"
        )

    return args


def main():
    args = _parse_args()

    issue_task = None
    issue_prompt = None
    repository_preflight = None


    if args.issue:
        try:
            issue_task = fetch_github_issue(
                issue_url=args.issue,
            )

            repository_preflight = check_issue_repository(
                issue_task
            )

            if repository_preflight.status == "mismatch":
                local_repositories = _format_local_repositories(
                    repository_preflight
                )

                console.print(
                    "[red bold]Repository mismatch:[/] "
                    f"Issue belongs to "
                    f"[cyan]{repository_preflight.target.full_name}[/cyan], "
                    f"but the current repository remotes are "
                    f"[yellow]{local_repositories}[/yellow]."
                )
                console.print(
                    "Run DevPilot from the matching repository "
                    "before attempting this Issue."
                )
                sys.exit(1)

            if repository_preflight.status == "unknown":
                if args.dry_run:
                    console.print(
                        "[yellow bold]Repository verification warning:[/] "
                        "No supported GitHub remote was found. "
                        "Continuing in read-only dry-run mode."
                    )
                elif not args.allow_unverified_repository:
                    console.print(
                        "[red bold]Repository verification required:[/] "
                        "No supported GitHub remote was found. "
                        "Real Issue repair is blocked by default."
                    )
                    console.print(
                        "Use --dry-run for read-only analysis, or pass "
                        "--allow-unverified-repository only after "
                        "independently verifying the current repository."
                    )
                    sys.exit(1)
                else:
                    console.print(
                        "[yellow bold]Repository verification override:[/] "
                        "No supported GitHub remote was found. "
                        "Continuing because "
                        "--allow-unverified-repository was provided."
                    )


            if not args.dry_run:
                worktree_preflight = check_worktree()

                if worktree_preflight.status == "not_repository":
                    console.print(
                        "[red bold]Git worktree required:[/] "
                        "Real Issue repair must run inside "
                        "a Git worktree."
                    )
                    console.print(
                        "Run DevPilot from the target Git repository, "
                        "or use --dry-run for read-only analysis."
                    )
                    sys.exit(1)

                if worktree_preflight.status == "dirty":
                    console.print(
                        "[red bold]Clean worktree required:[/] "
                        "Real Issue repair cannot start while "
                        "the current Git worktree has "
                        "uncommitted changes."
                    )

                    for change in worktree_preflight.changes:
                        console.print(
                            f"  [yellow]{change}[/yellow]"
                        )

                    console.print(
                        "Commit, stash, or discard these changes "
                        "before starting the repair."
                    )
                    sys.exit(1)
                repair_branch = create_repair_branch(
                    issue_task
                )
                console.print(
                    "[green bold]Repair branch created:[/] "
                    f"[cyan]{repair_branch}[/cyan]"
                )
            issue_prompt = build_issue_repair_prompt(
                issue_task,
                dry_run=args.dry_run,
            )

        except (
            GitHubIssueReferenceError,
            GitHubIssueFetchError,
            IssueWorkflowError,
            RepositoryGuardError,
            RepairBranchError,
        ) as error:
            console.print(
                f"[red bold]Issue workflow error:[/] {error}"
            )
            sys.exit(1)
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent_tools = None

    if args.issue:
        agent_tools = _tools_for_issue_workflow(
            dry_run=args.dry_run,
        )

    agent = Agent(
        llm=llm,
        tools=agent_tools,
        max_context_tokens=config.max_context_tokens,
    )

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # GitHub Issue workflow mode
    if args.issue:
        assert issue_task is not None
        assert issue_prompt is not None
        assert repository_preflight is not None
        mode = "dry run" if args.dry_run else "repair"

        issue_number = (
            f"#{issue_task.issue_number}"
            if issue_task.issue_number is not None
            else "unknown number"
        )

        repository_status = repository_preflight.status

        console.print(
            Panel(
                (
                    f"[bold]{issue_task.title}[/bold]\n"
                    f"Issue: [cyan]{issue_number}[/cyan]\n"
                    f"Repository: "
                    f"[cyan]{repository_preflight.target.full_name}[/cyan]\n"
                    f"Preflight: [cyan]{repository_status}[/cyan]\n"
                    f"Mode: [cyan]{mode}[/cyan]"
                ),
                title="DevPilot Issue Workflow",
                border_style="blue",
            )
        )

        _run_once(agent, issue_prompt)
        return

    # one-shot prompt mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config)


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    try:
        agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    print()


def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        # an unknown /command shouldn't be sent to the model as a prompt
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    """Format tool keyword arguments for a compact terminal preview.

    Each value's repr is capped at ~40 characters.  Long string reprs are
    shortened with an ellipsis while preserving both opening and closing
    quotes so the preview never shows an orphaned quote character.
    The final result is further trimmed to *maxlen*.
    """
    parts = []
    for k, v in kwargs.items():
        r = repr(v)
        # Safely truncate long string reprs so the closing quote is kept
        if len(r) > 40:
            q = r[0]
            if q in ("'", '"') and r[-1] == q:
                inner = r[1:-1]
                target = 40 - len(q) * 2 - 3  # room for q + inner + ... + q
                if target > 0 and len(inner) > target:
                    r = q + inner[:target] + "..." + q
        parts.append(f"{k}={r}")

    s = ", ".join(parts)

    if len(s) <= maxlen:
        return s

    if maxlen <= 0:
        return ""

    if maxlen <= 3:
        return "..."[:maxlen]

    kept_parts = []

    for part in parts:
        full_preview = ", ".join([*kept_parts, part])

        if len(full_preview) <= maxlen:
            kept_parts.append(part)
            continue

        # The complete value does not fit. Preserve the argument name
        # without cutting through its repr or leaving an open quote.
        key, _, _ = part.partition("=")
        shortened_part = f"{key}=..."
        shortened_preview = ", ".join([*kept_parts, shortened_part])

        if len(shortened_preview) <= maxlen:
            return shortened_preview

        # If even key=... does not fit, preserve as many earlier complete
        # arguments as possible and mark the omitted remainder.
        while kept_parts:
            omitted_preview = ", ".join([*kept_parts, "..."])

            if len(omitted_preview) <= maxlen:
                return omitted_preview

            kept_parts.pop()

        return "..."[:maxlen]

    return s
