"""Base class for all tools."""
import asyncio
from abc import ABC, abstractmethod
from ..permissions import ToolPermission

class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args
    permission: ToolPermission = ToolPermission.UNKNOWN

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    async def aexecute(self, **kwargs) -> str:
        """Run the tool asynchronously."""
        return await asyncio.to_thread(
            self.execute,
            **kwargs,
        )

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
