"""Repository code search tool."""

import json

from corecoder.code_rag import search_repository
from corecoder.permissions import ToolPermission

from .base import Tool


class CodeSearchTool(Tool):
    """Search repository source code using the Code RAG index."""

    name = "code_search"
    permission = ToolPermission.READ
    context_priority = "high"

    description = (
        "Search Python source code in a repository and return "
        "the most relevant symbols and source chunks."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Code or concept to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Repository directory to search. "
                    "Defaults to the current directory."
                ),
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Maximum number of ranked results to return."
                ),
            },
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str,
        path: str = ".",
        top_k: int = 5,
    ) -> str:
        """Execute a repository code search."""
        try:
            results = search_repository(
                path,
                query,
                top_k=top_k,
            )
        except ValueError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )

        payload = {
            "status": "ok",
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "path": result.chunk.path,
                    "symbol": result.chunk.symbol,
                    "start_line": result.chunk.start_line,
                    "end_line": result.chunk.end_line,
                    "score": result.score,
                    "matched_tokens": list(result.matched_tokens),
                    "content": result.chunk.content,
                }
                for result in results
            ],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
