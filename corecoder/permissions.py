"""Tool permission policy primitives."""

from dataclasses import dataclass
from enum import Enum


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    UNKNOWN = "unknown"


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

@dataclass(frozen=True)
class ToolApprovalRequest:
    """Describe one tool call awaiting human approval."""

    tool_name: str
    permission: ToolPermission
    arguments: dict

class ToolPermissionPolicy:
    """Default permission policy for tool execution."""

    def evaluate(
        self,
        permission: ToolPermission,
    ) -> PermissionDecision:
        if permission is ToolPermission.READ:
            return PermissionDecision.ALLOW

        return PermissionDecision.ASK