import asyncio
import pytest
from mcp.client import Client

from corecoder.mcp_tools import create_mcp_server

from corecoder.tools.mcp import _result_text, load_mcp_tools
from mcp.types import CallToolResult, TextContent

def test_mcp_server_exposes_read_repository_file():
    async def scenario():
        server = create_mcp_server()

        async with Client(server) as client:
            result = await client.list_tools()

        assert [tool.name for tool in result.tools] == [
            "read_repository_file"
        ]

    asyncio.run(scenario())

def test_mcp_read_repository_file_returns_file_contents(tmp_path):
    repository_root = tmp_path
    target = repository_root / "example.txt"
    target.write_text("hello from CoreCoder", encoding="utf-8")

    async def scenario():
        server = create_mcp_server(
            repository_root=repository_root
        )

        async with Client(server) as client:
            result = await client.call_tool(
                "read_repository_file",
                {"path": "example.txt"},
            )

        assert result.is_error is False
        assert result.content[0].text == "hello from CoreCoder"

    asyncio.run(scenario())

def test_mcp_read_repository_file_rejects_paths_outside_repository(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text(
        "secret outside repository",
        encoding="utf-8",
    )

    async def scenario():
        server = create_mcp_server(
            repository_root=repository_root
        )

        async with Client(server) as client:
            result = await client.call_tool(
                "read_repository_file",
                {"path": "../outside.txt"},
            )

        assert result.is_error is True

    asyncio.run(scenario())

def test_load_mcp_tools_adapts_discovered_tools_to_corecoder_schema():
    server = create_mcp_server()

    tools = load_mcp_tools(server)

    assert [tool.name for tool in tools] == [
        "read_repository_file"
    ]

    schema = tools[0].schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_repository_file"
    assert schema["function"]["parameters"]["type"] == "object"
    assert "path" in schema["function"]["parameters"]["properties"]

def test_mcp_tool_adapter_executes_tool_through_mcp(tmp_path):
    target = tmp_path / "example.txt"
    target.write_text(
        "hello through MCP adapter",
        encoding="utf-8",
    )

    server = create_mcp_server(
        repository_root=tmp_path
    )

    tools = load_mcp_tools(server)

    result = tools[0].execute(
        path="example.txt"
    )

    assert result == "hello through MCP adapter"

def test_mcp_tool_adapter_aexecute_calls_mcp_directly(tmp_path):
    target = tmp_path / "example.txt"
    target.write_text(
        "hello through async MCP adapter",
        encoding="utf-8",
    )

    server = create_mcp_server(
        repository_root=tmp_path
    )

    tool = load_mcp_tools(server)[0]

    def fail_execute(**kwargs):
        raise AssertionError(
            "sync execute should not be used"
        )

    tool.execute = fail_execute

    async def scenario():
        return await tool.aexecute(
            path="example.txt"
        )

    assert asyncio.run(scenario()) == (
        "hello through async MCP adapter"
    )

def test_mcp_tool_adapter_raises_when_mcp_tool_fails(tmp_path):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text(
        "secret outside repository",
        encoding="utf-8",
    )

    server = create_mcp_server(
        repository_root=repository_root
    )

    tools = load_mcp_tools(server)

    with pytest.raises(RuntimeError):
        tools[0].execute(
            path="../outside.txt"
        )

def test_mcp_result_text_preserves_multiple_text_blocks():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="first"),
            TextContent(type="text", text="second"),
        ]
    )

    assert _result_text(result) == "first\nsecond"
