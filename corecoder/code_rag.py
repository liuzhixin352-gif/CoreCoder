"""Repository code retrieval primitives."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path


_IGNORED_REPOSITORY_DIRS = {
    ".git",
    ".venv",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

_IDENTIFIER_PART_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\Z)"
    r"|[A-Z]?[a-z]+"
    r"|[A-Z]+"
    r"|\d+"
)

_COARSE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_AUXILIARY_CODE_DIRS = {
    "test",
    "tests",
    "testing",
    "benchmark",
    "benchmarks",
    "spec",
    "specs",
}

_AUXILIARY_CODE_WEIGHT = 0.5


@dataclass(frozen=True)
class CodeChunk:
    """A source-code chunk with repository location metadata."""

    path: str
    start_line: int
    end_line: int
    content: str
    symbol: str


@dataclass(frozen=True)
class CodeSearchResult:
    """One ranked result for a code search query."""

    chunk: CodeChunk
    score: float
    matched_tokens: tuple[str, ...]


def _node_start_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    """Return the first source line belonging to a symbol."""
    if not node.decorator_list:
        return node.lineno

    return min(
        node.lineno,
        *(decorator.lineno for decorator in node.decorator_list),
    )


def _chunk_from_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    *,
    path: str,
    lines: list[str],
    symbol: str,
) -> CodeChunk:
    """Create a CodeChunk from one AST node."""
    start_line = _node_start_line(node)
    end_line = node.end_lineno or node.lineno

    return CodeChunk(
        path=path,
        start_line=start_line,
        end_line=end_line,
        content="\n".join(lines[start_line - 1:end_line]),
        symbol=symbol,
    )


def extract_python_chunks(
    source: str,
    *,
    path: str,
) -> list[CodeChunk]:
    """Extract top-level symbols and class methods from Python source."""
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _chunk_from_node(
                    node,
                    path=path,
                    lines=lines,
                    symbol=node.name,
                )
            )
            continue

        if not isinstance(node, ast.ClassDef):
            continue

        chunks.append(
            _chunk_from_node(
                node,
                path=path,
                lines=lines,
                symbol=node.name,
            )
        )

        for child in node.body:
            if not isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            chunks.append(
                _chunk_from_node(
                    child,
                    path=path,
                    lines=lines,
                    symbol=f"{node.name}.{child.name}",
                )
            )

    return chunks


def discover_python_files(repo_root: str | Path) -> list[str]:
    """Discover indexable Python files under a repository root."""
    root = Path(repo_root)
    paths: list[str] = []

    for path in root.rglob("*.py"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)

        if any(
            part in _IGNORED_REPOSITORY_DIRS
            for part in relative.parts[:-1]
        ):
            continue

        paths.append(relative.as_posix())

    return sorted(paths)


def index_repository(repo_root: str | Path) -> list[CodeChunk]:
    """Build source-code chunks for indexable repository files."""
    root = Path(repo_root)
    chunks: list[CodeChunk] = []

    for relative_path in discover_python_files(root):
        source_path = root / Path(relative_path)
        source = source_path.read_text(encoding="utf-8")

        try:
            file_chunks = extract_python_chunks(
                source,
                path=relative_path,
            )
        except SyntaxError:
            continue

        chunks.extend(file_chunks)

    return chunks

def _is_auxiliary_code_path(path: str) -> bool:
    """Return whether a path primarily contains tests or benchmarks."""
    normalized = Path(path)
    parts = [
        part.lower()
        for part in normalized.parts
    ]
    filename = normalized.name.lower()

    if any(
        part in _AUXILIARY_CODE_DIRS
        for part in parts[:-1]
    ):
        return True

    return (
        filename == "conftest.py"
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.startswith("benchmark_")
        or filename.startswith("bench_")
        or filename.endswith("_spec.py")
    )

def _tokenize_code(text: str) -> set[str]:
    """Normalize source text into searchable code-aware tokens."""
    tokens: set[str] = set()

    for coarse_token in _COARSE_TOKEN_RE.findall(text):
        normalized = coarse_token.lower()
        tokens.add(normalized)

        for part in _IDENTIFIER_PART_RE.findall(coarse_token):
            tokens.add(part.lower())

    return tokens


class LexicalCodeIndex:
    """In-memory inverted index over repository code chunks."""

    def __init__(self, chunks: list[CodeChunk]):
        self.chunks = list(chunks)
        self._postings: dict[str, list[CodeChunk]] = {}
        self._token_weights: dict[tuple[CodeChunk, str], float] = {}

        for chunk in self.chunks:
            token_weights: dict[str, float] = {}

            searchable_fields = (
                (chunk.content, 1.0),
                (chunk.path, 2.0),
                (chunk.symbol, 3.0),
            )

            for text, weight in searchable_fields:
                for token in _tokenize_code(text):
                    token_weights[token] = max(
                        token_weights.get(token, 0.0),
                        weight,
                    )

            for token, weight in token_weights.items():
                self._postings.setdefault(token, []).append(chunk)
                self._token_weights[(chunk, token)] = weight

    def lookup(self, token: str) -> list[CodeChunk]:
        """Return chunks containing a normalized lexical token."""
        normalized_tokens = _tokenize_code(token)

        if not normalized_tokens:
            return []

        matches: list[CodeChunk] = []
        seen: set[CodeChunk] = set()

        for normalized_token in sorted(normalized_tokens):
            for chunk in self._postings.get(normalized_token, []):
                if chunk in seen:
                    continue

                seen.add(chunk)
                matches.append(chunk)

        return matches

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[CodeSearchResult]:
        """Rank chunks by distinct query-token overlap."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        query_tokens = _tokenize_code(query)

        if not query_tokens:
            return []

        matches: dict[CodeChunk, dict[str, float]] = {}

        for token in query_tokens:
            for chunk in self._postings.get(token, []):
                matches.setdefault(chunk, {})[token] = (
                    self._token_weights[(chunk, token)]
                )

        results = [
            CodeSearchResult(
                chunk=chunk,
                score=sum(token_weights.values()),
                matched_tokens=tuple(sorted(token_weights)),
            )
            for chunk, token_weights in matches.items()
        ]

        def ranking_score(
            result: CodeSearchResult,
        ) -> float:
            score = result.score

            if _is_auxiliary_code_path(
                result.chunk.path
            ):
                score *= _AUXILIARY_CODE_WEIGHT

            return score


        ranked_results = sorted(
            results,
            key=lambda result: (
                -ranking_score(result),
                _is_auxiliary_code_path(
                    result.chunk.path
                ),
                -result.score,
                result.chunk.path,
                result.chunk.start_line,
                result.chunk.end_line,
                result.chunk.symbol,
            ),
        )

        if top_k is None:
            return ranked_results

        return ranked_results[:top_k]

def search_repository(
    repo_root: str | Path,
    query: str,
    *,
    top_k: int | None = None,
) -> list[CodeSearchResult]:
    """Search a repository for code relevant to a lexical query."""
    root = Path(repo_root)

    if not root.exists():
        raise ValueError(f"Repository path not found: {repo_root}")

    chunks = index_repository(root)
    index = LexicalCodeIndex(chunks)

    return index.search(
        query,
        top_k=top_k,
    )
