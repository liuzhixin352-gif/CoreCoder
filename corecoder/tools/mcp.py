import asyncio

from mcp.client import Client
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from .base import Tool

def _result_text(result: CallToolResult) -> str:
    return "\n".join(
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    )

class MCPToolAdapter(Tool):
    def __init__(
        self,
        *,
        server: MCPServer,
        name: str,
        description: str | None,
        parameters: dict,
    ):
        self._server = server
        self.name = name
        self.description = description or ""
        self.parameters = parameters

    async def aexecute(self, **kwargs) -> str:
        async with Client(self._server) as client:
            result = await client.call_tool(
                self.name,
                kwargs,
            )

        text = _result_text(result)

        if result.is_error:
            raise RuntimeError(text)

        return text

    def execute(self, **kwargs) -> str:
        return asyncio.run(
            self.aexecute(**kwargs)
        )


async def _load_mcp_tools(
    server: MCPServer,
) -> list[Tool]:
    async with Client(server) as client:
        result = await client.list_tools()

    return [
        MCPToolAdapter(
            server=server,
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in result.tools
    ]


def load_mcp_tools(
    server: MCPServer,
) -> list[Tool]:
    return asyncio.run(_load_mcp_tools(server))