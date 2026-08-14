from pathlib import Path

from mcp.server import MCPServer


def create_mcp_server(
    repository_root: Path | None = None,
) -> MCPServer:
    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )

    server = MCPServer("CoreCoder")

    @server.tool()
    def read_repository_file(path: str) -> str:
        """Read a file from the repository."""
        target = (root / path).resolve()

        if not target.is_relative_to(root):
            raise ValueError(
                "path must stay within repository root"
            )

        return target.read_text(encoding="utf-8")

    return server