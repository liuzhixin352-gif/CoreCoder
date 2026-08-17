"""Tests for repository code retrieval."""

from corecoder.code_rag import (
    CodeChunk,
    CodeSearchResult,
    LexicalCodeIndex,
    discover_python_files,
    extract_python_chunks,
    index_repository,
    search_repository,
)
import json
import pytest
from corecoder.permissions import ToolPermission
from corecoder.tools.code_search import CodeSearchTool
from corecoder.tools import get_tool
from types import SimpleNamespace

from corecoder.agent import Agent
from corecoder.eval import (
    CodeRetrievalEvalCase,
    CodeRetrievalEvalResult,
    CodeRetrievalEvalRunner,
    CodeRetrievalEvalReport,
)

def test_code_chunk_preserves_source_metadata():
    chunk = CodeChunk(
        path="corecoder/agent.py",
        start_line=10,
        end_line=20,
        content="class Agent:\n    pass",
        symbol="Agent",
    )

    assert chunk.path == "corecoder/agent.py"
    assert chunk.start_line == 10
    assert chunk.end_line == 20
    assert chunk.content == "class Agent:\n    pass"
    assert chunk.symbol == "Agent"


def test_extract_python_chunks_creates_function_chunk():
    source = (
        "x = 1\n"
        "\n"
        "def greet(name):\n"
        '    return f"hello {name}"\n'
    )

    chunks = extract_python_chunks(source, path="example.py")

    assert chunks == [
        CodeChunk(
            path="example.py",
            start_line=3,
            end_line=4,
            content='def greet(name):\n    return f"hello {name}"',
            symbol="greet",
        )
    ]


def test_extract_python_chunks_creates_qualified_method_chunk():
    source = (
        "class Greeter:\n"
        "    def greet(self, name):\n"
        '        return f"hello {name}"\n'
    )

    chunks = extract_python_chunks(source, path="example.py")

    assert chunks == [
        CodeChunk(
            path="example.py",
            start_line=1,
            end_line=3,
            content=(
                "class Greeter:\n"
                "    def greet(self, name):\n"
                '        return f"hello {name}"'
            ),
            symbol="Greeter",
        ),
        CodeChunk(
            path="example.py",
            start_line=2,
            end_line=3,
            content=(
                "    def greet(self, name):\n"
                '        return f"hello {name}"'
            ),
            symbol="Greeter.greet",
        ),
    ]


def test_extract_python_chunks_preserves_function_decorators():
    source = (
        "@trace\n"
        "def greet(name):\n"
        '    return f"hello {name}"\n'
    )

    chunks = extract_python_chunks(source, path="example.py")

    assert chunks == [
        CodeChunk(
            path="example.py",
            start_line=1,
            end_line=3,
            content=(
                "@trace\n"
                "def greet(name):\n"
                '    return f"hello {name}"'
            ),
            symbol="greet",
        )
    ]


def test_extract_python_chunks_supports_async_functions():
    source = (
        "async def fetch_data():\n"
        "    return 42\n"
    )

    chunks = extract_python_chunks(source, path="example.py")

    assert chunks == [
        CodeChunk(
            path="example.py",
            start_line=1,
            end_line=2,
            content=(
                "async def fetch_data():\n"
                "    return 42"
            ),
            symbol="fetch_data",
        )
    ]


def test_extract_python_chunks_preserves_decorated_class_and_method():
    source = (
        "@dataclass\n"
        "class Greeter:\n"
        "    @staticmethod\n"
        "    def greet(name):\n"
        "        return name\n"
    )

    chunks = extract_python_chunks(source, path="example.py")

    assert chunks == [
        CodeChunk(
            path="example.py",
            start_line=1,
            end_line=5,
            content=(
                "@dataclass\n"
                "class Greeter:\n"
                "    @staticmethod\n"
                "    def greet(name):\n"
                "        return name"
            ),
            symbol="Greeter",
        ),
        CodeChunk(
            path="example.py",
            start_line=3,
            end_line=5,
            content=(
                "    @staticmethod\n"
                "    def greet(name):\n"
                "        return name"
            ),
            symbol="Greeter.greet",
        ),
    ]


def test_discover_python_files_returns_indexable_repository_files(
    tmp_path,
):
    repo = tmp_path / "repo"

    (repo / "corecoder").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / ".venv").mkdir()
    (repo / "__pycache__").mkdir()

    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "corecoder" / "agent.py").write_text(
        "class Agent:\n    pass\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Repo\n", encoding="utf-8")

    (repo / ".git" / "hook.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )
    (repo / ".venv" / "package.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )
    (repo / "__pycache__" / "cached.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )

    paths = discover_python_files(repo)

    assert paths == [
        "app.py",
        "corecoder/agent.py",
    ]

def test_discover_python_files_ignores_generated_and_dependency_dirs(
    tmp_path,
):
    repo = tmp_path / "repo"

    (repo / "src").mkdir(parents=True)
    (repo / ".tox").mkdir()
    (repo / "node_modules").mkdir()
    (repo / "build").mkdir()
    (repo / "dist").mkdir()

    (repo / "src" / "main.py").write_text(
        "def main():\n    pass\n",
        encoding="utf-8",
    )

    (repo / ".tox" / "generated.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )
    (repo / "node_modules" / "dependency.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )
    (repo / "build" / "built.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )
    (repo / "dist" / "packaged.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )

    paths = discover_python_files(repo)

    assert paths == [
        "src/main.py",
    ]

def test_index_repository_builds_chunks_from_discovered_files(
    tmp_path,
):
    repo = tmp_path / "repo"

    (repo / "corecoder").mkdir(parents=True)
    (repo / ".venv").mkdir()

    (repo / "corecoder" / "alpha.py").write_text(
        "def alpha():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (repo / "corecoder" / "worker.py").write_text(
        "class Worker:\n"
        "    def run(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    (repo / ".venv" / "ignored.py").write_text(
        "def ignored():\n"
        "    return 3\n",
        encoding="utf-8",
    )

    chunks = index_repository(repo)

    assert chunks == [
        CodeChunk(
            path="corecoder/alpha.py",
            start_line=1,
            end_line=2,
            content=(
                "def alpha():\n"
                "    return 1"
            ),
            symbol="alpha",
        ),
        CodeChunk(
            path="corecoder/worker.py",
            start_line=1,
            end_line=3,
            content=(
                "class Worker:\n"
                "    def run(self):\n"
                "        return 2"
            ),
            symbol="Worker",
        ),
        CodeChunk(
            path="corecoder/worker.py",
            start_line=2,
            end_line=3,
            content=(
                "    def run(self):\n"
                "        return 2"
            ),
            symbol="Worker.run",
        ),
    ]

def test_lexical_code_index_finds_chunks_by_normalized_token():
    permission_chunk = CodeChunk(
        path="corecoder/agent.py",
        start_line=10,
        end_line=12,
        content=(
            "def _permission_error(self):\n"
            "    permission = self.permission_policy\n"
            "    return permission"
        ),
        symbol="Agent._permission_error",
    )
    context_chunk = CodeChunk(
        path="corecoder/context.py",
        start_line=20,
        end_line=22,
        content=(
            "def maybe_compress(self):\n"
            "    return self.messages"
        ),
        symbol="ContextManager.maybe_compress",
    )

    index = LexicalCodeIndex(
        [
            permission_chunk,
            context_chunk,
        ]
    )

    assert index.lookup("permission") == [
        permission_chunk,
    ]
    assert index.lookup("PERMISSION") == [
        permission_chunk,
    ]

def test_lexical_code_index_splits_code_identifiers():
    permission_chunk = CodeChunk(
        path="corecoder/agent.py",
        start_line=1,
        end_line=3,
        content=(
            "def _permission_error(self):\n"
            "    return None"
        ),
        symbol="Agent._permission_error",
    )
    context_chunk = CodeChunk(
        path="corecoder/context.py",
        start_line=1,
        end_line=3,
        content=(
            "class ContextManager:\n"
            "    pass"
        ),
        symbol="ContextManager",
    )

    index = LexicalCodeIndex(
        [
            permission_chunk,
            context_chunk,
        ]
    )

    assert index.lookup("error") == [
        permission_chunk,
    ]
    assert index.lookup("manager") == [
        context_chunk,
    ]

def test_lexical_code_index_ranks_chunks_by_query_token_overlap():
    approval_chunk = CodeChunk(
        path="corecoder/permissions.py",
        start_line=1,
        end_line=3,
        content=(
            "def request_permission_approval():\n"
            "    return True"
        ),
        symbol="request_permission_approval",
    )
    permission_chunk = CodeChunk(
        path="corecoder/agent.py",
        start_line=1,
        end_line=3,
        content=(
            "def check_permission():\n"
            "    return True"
        ),
        symbol="check_permission",
    )
    context_chunk = CodeChunk(
        path="corecoder/context.py",
        start_line=1,
        end_line=3,
        content=(
            "def compress_context():\n"
            "    return True"
        ),
        symbol="compress_context",
    )

    index = LexicalCodeIndex(
        [
            permission_chunk,
            context_chunk,
            approval_chunk,
        ]
    )

    results = index.search("permission approval")

    assert results == [
        CodeSearchResult(
            chunk=approval_chunk,
            score=6.0,
            matched_tokens=("approval", "permission"),
        ),
        CodeSearchResult(
            chunk=permission_chunk,
            score=3.0,
            matched_tokens=("permission",),
        ),
    ]

def test_lexical_code_index_limits_ranked_results_with_top_k():
    first_chunk = CodeChunk(
        path="first.py",
        start_line=1,
        end_line=2,
        content="permission approval policy",
        symbol="first",
    )
    second_chunk = CodeChunk(
        path="second.py",
        start_line=1,
        end_line=2,
        content="permission approval",
        symbol="second",
    )
    third_chunk = CodeChunk(
        path="third.py",
        start_line=1,
        end_line=2,
        content="permission",
        symbol="third",
    )

    index = LexicalCodeIndex(
        [
            third_chunk,
            second_chunk,
            first_chunk,
        ]
    )

    results = index.search(
        "permission approval policy",
        top_k=2,
    )

    assert [result.chunk for result in results] == [
        first_chunk,
        second_chunk,
    ]

@pytest.mark.parametrize("top_k", [0, -1])
def test_lexical_code_index_rejects_non_positive_top_k(top_k):
    chunk = CodeChunk(
        path="example.py",
        start_line=1,
        end_line=1,
        content="permission",
        symbol="permission",
    )
    index = LexicalCodeIndex([chunk])

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        index.search("permission", top_k=top_k)

def test_lexical_code_index_weights_symbol_and_path_matches():
    symbol_chunk = CodeChunk(
        path="z_symbol.py",
        start_line=1,
        end_line=2,
        content="def check():\n    return True",
        symbol="check_permission",
    )
    path_chunk = CodeChunk(
        path="corecoder/permission_helpers.py",
        start_line=1,
        end_line=2,
        content="def check():\n    return True",
        symbol="check",
    )
    content_chunk = CodeChunk(
        path="a_content.py",
        start_line=1,
        end_line=2,
        content="def check():\n    permission = True",
        symbol="check",
    )

    index = LexicalCodeIndex(
        [
            content_chunk,
            path_chunk,
            symbol_chunk,
        ]
    )

    results = index.search("permission")

    assert results == [
        CodeSearchResult(
            chunk=symbol_chunk,
            score=3.0,
            matched_tokens=("permission",),
        ),
        CodeSearchResult(
            chunk=path_chunk,
            score=2.0,
            matched_tokens=("permission",),
        ),
        CodeSearchResult(
            chunk=content_chunk,
            score=1.0,
            matched_tokens=("permission",),
        ),
    ]

def test_lexical_code_index_deduplicates_repeated_query_tokens():
    chunk = CodeChunk(
        path="corecoder/agent.py",
        start_line=1,
        end_line=2,
        content=(
            "def check_permission():\n"
            "    return True"
        ),
        symbol="check_permission",
    )

    index = LexicalCodeIndex([chunk])

    results = index.search(
        "permission permission permission",
    )

    assert results == [
        CodeSearchResult(
            chunk=chunk,
            score=3.0,
            matched_tokens=("permission",),
        )
    ]

def test_search_repository_returns_ranked_repository_results(
    tmp_path,
):
    repo = tmp_path / "repo"
    (repo / "corecoder").mkdir(parents=True)

    (repo / "corecoder" / "permissions.py").write_text(
        "def request_permission_approval():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "corecoder" / "context.py").write_text(
        "def compress_context():\n"
        "    return True\n",
        encoding="utf-8",
    )

    results = search_repository(
        repo,
        "permission approval",
        top_k=1,
    )

    assert results == [
        CodeSearchResult(
            chunk=CodeChunk(
                path="corecoder/permissions.py",
                start_line=1,
                end_line=2,
                content=(
                    "def request_permission_approval():\n"
                    "    return True"
                ),
                symbol="request_permission_approval",
            ),
            score=6.0,
            matched_tokens=("approval", "permission"),
        )
    ]

def test_code_search_tool_declares_read_permission():
    tool = CodeSearchTool()

    assert tool.name == "code_search"
    assert tool.permission is ToolPermission.READ

def test_code_search_tool_returns_ranked_results_as_json(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "permissions.py").write_text(
        "def request_permission_approval():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "context.py").write_text(
        "def compress_context():\n"
        "    return True\n",
        encoding="utf-8",
    )

    tool = CodeSearchTool()

    result = json.loads(
        tool.execute(
            query="permission approval",
            path=str(repo),
            top_k=1,
        )
    )

    assert result == {
        "status": "ok",
        "query": "permission approval",
        "result_count": 1,
        "results": [
            {
                "path": "permissions.py",
                "symbol": "request_permission_approval",
                "start_line": 1,
                "end_line": 2,
                "score": 6.0,
                "matched_tokens": [
                    "approval",
                    "permission",
                ],
                "content": (
                    "def request_permission_approval():\n"
                    "    return True"
                ),
            }
        ],
    }

def test_code_search_tool_returns_error_for_invalid_top_k(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    tool = CodeSearchTool()

    result = json.loads(
        tool.execute(
            query="permission",
            path=str(repo),
            top_k=0,
        )
    )

    assert result == {
        "status": "error",
        "error": "top_k must be greater than 0",
    }

def test_search_repository_rejects_missing_repository(tmp_path):
    missing_repo = tmp_path / "missing"

    with pytest.raises(
        ValueError,
        match="Repository path not found",
    ):
        search_repository(
            missing_repo,
            "permission",
            top_k=5,
        )

def test_code_search_tool_returns_error_for_missing_repository(
    tmp_path,
):
    missing_repo = tmp_path / "missing"
    tool = CodeSearchTool()

    result = json.loads(
        tool.execute(
            query="permission",
            path=str(missing_repo),
            top_k=5,
        )
    )

    assert result == {
        "status": "error",
        "error": f"Repository path not found: {missing_repo}",
    }

def test_code_search_tool_is_registered():
    tool = get_tool("code_search")

    assert isinstance(tool, CodeSearchTool)
    assert tool.permission is ToolPermission.READ

def test_code_search_tool_declares_high_context_priority():
    tool = CodeSearchTool()

    assert tool.context_priority == "high"


def test_agent_marks_code_search_results_as_high_priority(
    monkeypatch,
):
    tool = CodeSearchTool()

    tool_call = SimpleNamespace(
        id="call-1",
        name="code_search",
        arguments={
            "query": "permission",
            "path": ".",
            "top_k": 1,
        },
    )

    responses = iter(
        [
            SimpleNamespace(
                tool_calls=[tool_call],
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "code_search",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                content="",
            ),
            SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            ),
        ]
    )

    class StubLLM:
        def chat(self, **kwargs):
            return next(responses)

    agent = Agent(
        llm=StubLLM(),
        tools=[tool],
        max_rounds=2,
    )

    monkeypatch.setattr(
        agent,
        "_exec_tools_parallel",
        lambda tool_calls, on_tool: ["retrieved code"],
    )

    assert agent.chat("find permission code") == "done"

    tool_messages = [
        message
        for message in agent.messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["context_priority"] == "high"

def test_agent_preserves_high_priority_code_search_result_during_compression(
    monkeypatch,
):
    tool = CodeSearchTool()

    tool_call = SimpleNamespace(
        id="call-1",
        name="code_search",
        arguments={
            "query": "permission",
            "path": ".",
            "top_k": 1,
        },
    )

    responses = iter(
        [
            SimpleNamespace(
                tool_calls=[tool_call],
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "code_search",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                content="",
            ),
            SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            ),
        ]
    )

    class StubLLM:
        def chat(self, **kwargs):
            return next(responses)

    retrieved_code = (
        "def important_permission_logic():\n"
        "    return True\n"
    ) * 100

    agent = Agent(
        llm=StubLLM(),
        tools=[tool],
        max_context_tokens=200,
        max_rounds=2,
    )

    monkeypatch.setattr(
        agent,
        "_exec_tools_parallel",
        lambda tool_calls, on_tool: [retrieved_code],
    )

    assert agent.chat("find permission code") == "done"

    tool_messages = [
        message
        for message in agent.messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["context_priority"] == "high"
    assert tool_messages[0]["content"] == retrieved_code

    metrics = agent.context.last_metrics

    assert metrics is not None
    assert metrics.high_priority_messages == 1
    assert metrics.priority_preserved_messages == 2

def test_default_agent_exposes_code_search_tool_to_llm():
    captured = {}

    class StubLLM:
        def chat(self, **kwargs):
            captured["tools"] = kwargs["tools"]

            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=StubLLM(),
        max_rounds=1,
    )

    assert agent.chat("find permission code") == "done"

    tool_names = [
        schema["function"]["name"]
        for schema in captured["tools"]
    ]

    assert "code_search" in tool_names

def test_code_retrieval_eval_case_preserves_expected_symbol():
    case = CodeRetrievalEvalCase(
        name="permission-search",
        query="permission approval",
        expected_symbol="request_permission_approval",
        top_k=3,
    )

    assert case.name == "permission-search"
    assert case.query == "permission approval"
    assert case.expected_symbol == "request_permission_approval"
    assert case.top_k == 3

def test_code_retrieval_eval_result_preserves_rank_and_symbols():
    result = CodeRetrievalEvalResult(
        case_name="permission-search",
        success=True,
        rank=2,
        retrieved_symbols=(
            "ToolPermissionPolicy.evaluate",
            "request_permission_approval",
            "ContextManager.maybe_compress",
        ),
    )

    assert result.case_name == "permission-search"
    assert result.success is True
    assert result.rank == 2
    assert result.retrieved_symbols == (
        "ToolPermissionPolicy.evaluate",
        "request_permission_approval",
        "ContextManager.maybe_compress",
    )

def test_code_retrieval_eval_runner_records_expected_symbol_rank(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "permissions.py").write_text(
        "def request_permission_approval():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "agent.py").write_text(
        "def check_permission():\n"
        "    return True\n",
        encoding="utf-8",
    )

    runner = CodeRetrievalEvalRunner(repo)

    result = runner.run_case(
        CodeRetrievalEvalCase(
            name="permission-search",
            query="permission approval",
            expected_symbol="request_permission_approval",
            top_k=2,
        )
    )

    assert result == CodeRetrievalEvalResult(
        case_name="permission-search",
        success=True,
        rank=1,
        retrieved_symbols=(
            "request_permission_approval",
            "check_permission",
        ),
    )

def test_code_retrieval_eval_runner_records_miss_outside_top_k(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "permissions.py").write_text(
        "def request_permission_approval():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "agent.py").write_text(
        "def check_permission():\n"
        "    return True\n",
        encoding="utf-8",
    )

    runner = CodeRetrievalEvalRunner(repo)

    result = runner.run_case(
        CodeRetrievalEvalCase(
            name="permission-miss",
            query="permission approval",
            expected_symbol="check_permission",
            top_k=1,
        )
    )

    assert result == CodeRetrievalEvalResult(
        case_name="permission-miss",
        success=False,
        rank=None,
        retrieved_symbols=(
            "request_permission_approval",
        ),
    )

def test_code_retrieval_eval_report_computes_hit_rate_and_mrr():
    report = CodeRetrievalEvalReport(
        results=(
            CodeRetrievalEvalResult(
                case_name="rank-one",
                success=True,
                rank=1,
                retrieved_symbols=("target",),
            ),
            CodeRetrievalEvalResult(
                case_name="rank-two",
                success=True,
                rank=2,
                retrieved_symbols=("other", "target"),
            ),
            CodeRetrievalEvalResult(
                case_name="miss",
                success=False,
                rank=None,
                retrieved_symbols=("other",),
            ),
        )
    )

    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx(0.5)

def test_code_retrieval_eval_runner_runs_multiple_cases(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "permissions.py").write_text(
        "def request_permission_approval():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "agent.py").write_text(
        "def check_permission():\n"
        "    return True\n",
        encoding="utf-8",
    )

    runner = CodeRetrievalEvalRunner(repo)

    report = runner.run(
        [
            CodeRetrievalEvalCase(
                name="permission-hit",
                query="permission approval",
                expected_symbol="request_permission_approval",
                top_k=2,
            ),
            CodeRetrievalEvalCase(
                name="permission-miss",
                query="permission approval",
                expected_symbol="check_permission",
                top_k=1,
            ),
        ]
    )

    assert report == CodeRetrievalEvalReport(
        results=(
            CodeRetrievalEvalResult(
                case_name="permission-hit",
                success=True,
                rank=1,
                retrieved_symbols=(
                    "request_permission_approval",
                    "check_permission",
                ),
            ),
            CodeRetrievalEvalResult(
                case_name="permission-miss",
                success=False,
                rank=None,
                retrieved_symbols=(
                    "request_permission_approval",
                ),
            ),
        )
    )

    assert report.hit_rate == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)

def test_code_retrieval_eval_report_serializes_metrics_and_results():
    report = CodeRetrievalEvalReport(
        results=(
            CodeRetrievalEvalResult(
                case_name="hit",
                success=True,
                rank=1,
                retrieved_symbols=(
                    "request_permission_approval",
                    "check_permission",
                ),
            ),
            CodeRetrievalEvalResult(
                case_name="miss",
                success=False,
                rank=None,
                retrieved_symbols=(
                    "compress_context",
                ),
            ),
        )
    )

    assert report.to_dict() == {
        "hit_rate": 0.5,
        "mrr": 0.5,
        "results": [
            {
                "case_name": "hit",
                "success": True,
                "rank": 1,
                "retrieved_symbols": [
                    "request_permission_approval",
                    "check_permission",
                ],
            },
            {
                "case_name": "miss",
                "success": False,
                "rank": None,
                "retrieved_symbols": [
                    "compress_context",
                ],
            },
        ],
    }

def test_code_retrieval_eval_report_serializes_to_json():
    report = CodeRetrievalEvalReport(
        results=(
            CodeRetrievalEvalResult(
                case_name="hit",
                success=True,
                rank=1,
                retrieved_symbols=("target",),
            ),
        )
    )

    assert json.loads(report.to_json()) == report.to_dict()

def test_search_repository_skips_python_file_with_invalid_syntax(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "valid.py").write_text(
        "def target_symbol():\n"
        "    return True\n",
        encoding="utf-8",
    )

    (repo / "broken.py").write_text(
        "def broken(:\n"
        "    return False\n",
        encoding="utf-8",
    )

    results = search_repository(
        repo,
        "target symbol",
        top_k=5,
    )

    assert [
        result.chunk.symbol
        for result in results
    ] == ["target_symbol"]