"""Structured representation and parsing of software issue tasks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable


_CHECKBOX_PATTERN = re.compile(
    r"^\s*[-*]\s*\[(?: |x|X)\]\s*(?P<text>.+?)\s*$"
)

_FILE_PATH_PATTERN = re.compile(
    r"(?<![\w.-])"
    r"(?:[\w.-]+/)+"
    r"[\w.-]+\."
    r"(?:py|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|md|toml|yaml|yml|json)"
)


@dataclass
class IssueTask:
    """Structured repair task derived from an issue."""

    title: str
    body: str = ""
    labels: list[str] = field(default_factory=list)
    issue_number: int | None = None
    issue_url: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)
    referenced_files: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the task as a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the task as formatted JSON text."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
        )


def _normalize_labels(labels: Iterable[str] | None) -> list[str]:
    """Normalize labels while preserving their original order."""
    normalized: list[str] = []

    for label in labels or []:
        cleaned = str(label).strip()

        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)

    return normalized


def _extract_acceptance_criteria(body: str) -> list[str]:
    """Extract Markdown checkbox items as acceptance criteria."""
    criteria: list[str] = []

    for line in body.splitlines():
        match = _CHECKBOX_PATTERN.match(line)

        if not match:
            continue

        text = match.group("text").strip()

        if text and text not in criteria:
            criteria.append(text)

    return criteria


def _extract_referenced_files(text: str) -> list[str]:
    """Extract likely repository file paths from issue text."""
    referenced_files: list[str] = []

    for match in _FILE_PATH_PATTERN.finditer(text):
        file_path = match.group(0)

        if file_path not in referenced_files:
            referenced_files.append(file_path)

    return referenced_files


def parse_issue_task(
    title: str,
    body: str = "",
    labels: Iterable[str] | None = None,
    issue_number: int | None = None,
    issue_url: str | None = None,
) -> IssueTask:
    """Convert raw issue fields into a structured repair task."""
    clean_title = title.strip()
    clean_body = body.strip()

    acceptance_criteria = _extract_acceptance_criteria(clean_body)
    referenced_files = _extract_referenced_files(
        f"{clean_title}\n{clean_body}"
    )

    ambiguities: list[str] = []

    if not clean_title:
        ambiguities.append("Issue title is empty")

    if not clean_body:
        ambiguities.append("Issue body is empty")

    if not acceptance_criteria:
        ambiguities.append(
            "No explicit Markdown acceptance criteria found"
        )

    return IssueTask(
        title=clean_title,
        body=clean_body,
        labels=_normalize_labels(labels),
        issue_number=issue_number,
        issue_url=issue_url,
        acceptance_criteria=acceptance_criteria,
        referenced_files=referenced_files,
        ambiguities=ambiguities,
    )