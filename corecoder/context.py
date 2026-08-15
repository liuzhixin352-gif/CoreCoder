"""Multi-layer context compression.

Claude Code uses a 4-layer strategy:
  1. HISTORY_SNIP   - trim old tool outputs to a one-line summary
  2. Microcompact   - LLM-powered summary of old turns (cached)
  3. CONTEXT_COLLAPSE - aggressive compression when nearing hard limit
  4. Autocompact    - periodic background compaction

CoreCoder implements the same idea in 3 layers:
  Layer 1 (tool_snip)   - replace verbose tool results with truncated versions
  Layer 2 (summarize)   - LLM-powered summary of old conversation
  Layer 3 (hard_collapse) - last resort: drop everything except summary + recent
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


def _approx_tokens(text: str) -> int:
    """Rough token count, roughly 3 chars per token for mixed en/zh content."""
    return len(text) // 3


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        if m.get("content"):
            total += _approx_tokens(m["content"])
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total

@dataclass(frozen=True)
class ContextMetrics:
    """Token usage metrics for one context-management pass."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    applied_layers: tuple[str, ...]
    high_priority_messages: int
    priority_preserved_messages: int


class ContextManager:
    def __init__(
        self,
        max_tokens: int = 128_000,
        reserved_output_tokens: int = 0,
        keep_recent_messages: int = 8,
    ):
        self.max_tokens = max_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.keep_recent_messages = keep_recent_messages
        self.last_metrics: ContextMetrics | None = None

        if reserved_output_tokens < 0:
            raise ValueError(
                "reserved_output_tokens must be non-negative"
            )

        if reserved_output_tokens >= max_tokens:
            raise ValueError(
                "reserved_output_tokens must be less than max_tokens"
            )

        if keep_recent_messages <= 0:
            raise ValueError(
                "keep_recent_messages must be greater than 0"
            )

        self.input_budget = max_tokens - reserved_output_tokens

        # layer thresholds (fraction of usable input budget)
        self._snip_at = int(self.input_budget * 0.50)
        self._summarize_at = int(self.input_budget * 0.70)
        self._collapse_at = int(self.input_budget * 0.90)

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Apply compression layers as needed. Returns True if any compression happened."""
        tokens_before = estimate_tokens(messages)
        current = tokens_before
        compressed = False
        applied_layers = []

        # Layer 1: snip verbose tool outputs
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                applied_layers.append("tool_snip")
                current = estimate_tokens(messages)

        # Layer 2: LLM-powered summarization of old turns
        if (
            current > self._summarize_at
            and len(messages) > self.keep_recent_messages + 2
        ):
            if self._summarize_old(messages, llm, keep_recent=self.keep_recent_messages):
                compressed = True
                applied_layers.append("summarize")
                current = estimate_tokens(messages)

        # Layer 3: hard collapse - last resort
        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True
            applied_layers.append("hard_collapse")

        tokens_after = estimate_tokens(messages)

        self.last_metrics = ContextMetrics(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            applied_layers=tuple(applied_layers),
            high_priority_messages=sum(
                1
                for message in messages
                if message.get("context_priority") == "high"
            ),
            priority_preserved_messages=len(
                self._priority_preserved_indexes(messages)
            ),
        )
        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """Layer 1: Truncate tool results over 1500 chars to their first/last lines.

        This mirrors Claude Code's HISTORY_SNIP which replaces old tool outputs
        with a one-line summary to reclaim context space.
        """
        changed = False
        preserved_indexes = ContextManager._priority_preserved_indexes(
            messages
        )
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            if i in preserved_indexes:
                continue

            content = m.get("content", "")

            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            # keep first 3 + last 3 lines
            snipped = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            m["content"] = snipped
            changed = True
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Index where the kept tail should start.

        Walk the boundary back so a 'tool' result is never separated from the
        assistant message whose tool_calls produced it - an orphaned tool
        message has no preceding tool_calls and OpenAI-compatible APIs reject it.
        """
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    @staticmethod
    def _priority_preserved_indexes(messages: list[dict]) -> set[int]:
        preserved_indexes = {
            i
            for i, message in enumerate(messages)
            if message.get("context_priority") == "high"
            and message.get("role") != "tool"
        }

        for assistant_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls") or []
            tool_call_ids = {
                tool_call.get("id")
                for tool_call in tool_calls
                if tool_call.get("id")
            }

            if not tool_call_ids:
                continue

            tool_result_indexes = {
                i
                for i, candidate in enumerate(messages)
                if candidate.get("role") == "tool"
                and candidate.get("tool_call_id") in tool_call_ids
            }

            group_is_high_priority = (
                message.get("context_priority") == "high"
                or any(
                    messages[i].get("context_priority") == "high"
                    for i in tool_result_indexes
                )
            )

            if group_is_high_priority:
                preserved_indexes.add(assistant_index)
                preserved_indexes.update(tool_result_indexes)

        return preserved_indexes

    def _summarize_old(self, messages: list[dict], llm: LLM | None,
                       keep_recent: int = 8) -> bool:
        """Layer 2: Summarize old conversation, keep recent messages intact."""
        if len(messages) <= keep_recent:
            return False

        split = self._safe_split(messages, keep_recent)
        old = messages[:split]
        tail = messages[split:]

        preserved_indexes = self._priority_preserved_indexes(old)

        high_priority = [
            message
            for i, message in enumerate(old)
            if i in preserved_indexes
        ]

        summarizable = [
            message
            for i, message in enumerate(old)
            if i not in preserved_indexes
        ]

        summary = self._get_summary(summarizable, llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(high_priority)
        messages.extend(tail)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """Layer 3: Emergency compression. Keep only last 4 messages + summary."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        old = messages[:split]
        tail = messages[split:]

        preserved_indexes = self._priority_preserved_indexes(old)

        high_priority = [
            message
            for i, message in enumerate(old)
            if i in preserved_indexes
        ]

        summarizable = [
            message
            for i, message in enumerate(old)
            if i not in preserved_indexes
        ]

        summary = self._get_summary(summarizable, llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(high_priority)
        messages.extend(tail)

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fallback to extraction."""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                pass

        # fallback: extract key lines
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        import re
        files_seen = set()
        errors = []

        for m in messages:
            text = m.get("content", "") or ""
            # extract file paths
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # extract error lines
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
