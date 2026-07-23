"""Prepare tokenized corpora and assemble bounded long-context prompts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


CONTEXTS_ROOT = Path(__file__).resolve().parents[2] / "data" / "contexts"
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"


class ContextError(ValueError):
    """Raised when a named context cannot be prepared or loaded safely."""


@dataclass(frozen=True)
class ContextChunk:
    """One token-counted corpus fragment stored by a context provider."""
    chunk_id: str
    source: str
    text: str
    token_count: int


@dataclass(frozen=True)
class PreparedContext:
    """Summary returned after writing a prepared context and manifest."""
    name: str
    provider: str
    chunks_path: str
    chunks: int
    corpus_tokens: int
    chunk_tokens: int
    chunk_overlap: int


@dataclass(frozen=True)
class ContextPrompt:
    """Completed prompt text paired with compact context provenance metadata."""
    prompt: str
    metadata: dict[str, Any]


class ContextProvider(Protocol):
    """Strategy interface for retrieving prepared chunks by context name."""
    name: str

    def load(self, context_name: str) -> list[ContextChunk]:
        """Load prepared chunks in deterministic order."""
        ...


def validate_context_name(name: str) -> str:
    """Validate a context name for safe use below the registry root.

    Raises:
        ContextError: If path traversal or separators are present.
    """
    if not name or name in {".", ".."} or any(part in name for part in ("/", "\\")):
        raise ContextError("Context names may contain no path separators")
    return name


class FolderContextProvider:
    """Read prepared context chunks from the filesystem registry."""
    name = "folder"

    def __init__(self, root: Path = CONTEXTS_ROOT) -> None:
        self.root = root

    def directory(self, context_name: str) -> Path:
        """Return the validated directory for ``context_name``."""
        return self.root / validate_context_name(context_name)

    def chunks_path(self, context_name: str) -> Path:
        """Return the JSONL chunk path for ``context_name``."""
        return self.directory(context_name) / "chunks.jsonl"

    def load(self, context_name: str) -> list[ContextChunk]:
        """Read context chunks, raising ``ContextError`` for invalid data."""
        path = self.chunks_path(context_name)
        if not path.is_file():
            raise ContextError(
                f"Prepared context not found: {path}. Run --prepare-context {context_name}."
            )
        chunks: list[ContextChunk] = []
        with path.open(encoding="utf-8") as context_file:
            for line_number, line in enumerate(context_file, 1):
                try:
                    record = json.loads(line)
                    chunks.append(ContextChunk(**record))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    raise ContextError(f"Invalid context chunk at {path}:{line_number}") from exc
        return chunks


CONTEXT_PROVIDERS: dict[str, Callable[[], ContextProvider]] = {
    "folder": FolderContextProvider,
}


def get_context_provider(name: str = "folder") -> ContextProvider:
    """Construct a registered provider or raise ``ContextError``."""
    try:
        return CONTEXT_PROVIDERS[name]()
    except KeyError as exc:
        raise ContextError(f"Unknown context provider: {name}") from exc


def _file_documents(path: Path) -> Iterator[tuple[str, str]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for source in paths:
        if source.is_file():
            yield str(source), source.read_text(encoding="utf-8")


def corpus_documents(corpus: str) -> Iterable[tuple[str, str]]:
    """Return source/text pairs from a path or WikiText-103 Raw.

    Side Effects:
        Reads files or downloads/reads the dataset through ``datasets``.
    """
    if corpus == "wikitext-103-raw":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ContextError("WikiText preparation requires the installed datasets package") from exc
        dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
        return ((f"wikitext-103-raw:train:{index}", row["text"]) for index, row in enumerate(dataset))
    path = Path(corpus).expanduser()
    if not path.exists():
        raise ContextError(f"Corpus does not exist: {path}")
    return _file_documents(path)


def chunk_documents(
    documents: Iterable[tuple[str, str]],
    tokenizer: Any,
    *,
    chunk_tokens: int,
    overlap: int,
) -> Iterator[ContextChunk]:
    """Yield tokenizer-bounded chunks with configurable token overlap.

    Raises:
        ContextError: If chunk size or overlap is invalid.
    """
    if chunk_tokens <= 0:
        raise ContextError("chunk_tokens must be greater than zero")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ContextError("chunk_overlap must be at least zero and smaller than chunk_tokens")
    step = chunk_tokens - overlap
    chunk_number = 0
    for source, text in documents:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(token_ids), step):
            selected = token_ids[start : start + chunk_tokens]
            if not selected:
                break
            chunk_text = tokenizer.decode(selected, skip_special_tokens=True)
            stored_token_count = len(
                tokenizer.encode(chunk_text, add_special_tokens=False)
            )
            yield ContextChunk(
                chunk_id=f"chunk-{chunk_number:08d}",
                source=source,
                text=chunk_text,
                token_count=stored_token_count,
            )
            chunk_number += 1
            if start + chunk_tokens >= len(token_ids):
                break


def prepare_folder_context(
    name: str,
    corpus: str,
    tokenizer: Any,
    *,
    chunk_tokens: int = 1024,
    overlap: int = 128,
    root: Path = CONTEXTS_ROOT,
) -> PreparedContext:
    """Tokenize a corpus and write folder-provider JSONL plus a manifest.

    Side Effects:
        Creates or replaces files under the named context directory.
    """
    provider = FolderContextProvider(root)
    directory = provider.directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = provider.chunks_path(name)
    count = 0
    corpus_tokens = 0
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunk_documents(
            corpus_documents(corpus), tokenizer,
            chunk_tokens=chunk_tokens, overlap=overlap,
        ):
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
            corpus_tokens += chunk.token_count
    manifest = PreparedContext(
        name=name, provider="folder", chunks_path=str(path), chunks=count,
        corpus_tokens=corpus_tokens, chunk_tokens=chunk_tokens,
        chunk_overlap=overlap,
    )
    (directory / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def _render_prompt(question: str, chunks: list[ContextChunk]) -> str:
    corpus = "\n\n".join(
        f"[Context chunk {chunk.chunk_id}; source: {chunk.source}]\n{chunk.text}"
        for chunk in chunks
    )
    return (
        "Use the supplied context when it is relevant. Answer the question after the context.\n\n"
        f"Context:\n{corpus}\n\nQuestion:\n{question}"
    )


def build_context_prompt(
    *,
    name: str,
    requested_tokens: int,
    question: str,
    tokenizer: Any,
    count_prompt_tokens: Callable[[str], int],
    model_context_limit: int,
    max_new_tokens: int,
    provider_name: str = "folder",
) -> ContextPrompt:
    """Build a context prompt while reserving output tokens exactly.

    Returns:
        Bounded prompt and compact context metadata.

    Raises:
        ContextError: If no usable context or valid prompt can fit.
    """
    if requested_tokens <= 0:
        raise ContextError("context_tokens must be greater than zero")
    available_input = model_context_limit - max_new_tokens
    if available_input <= 0:
        raise ContextError("max_new_tokens leaves no room for an input prompt")
    provider = get_context_provider(provider_name)
    source_chunks = provider.load(name)
    selected: list[ContextChunk] = []
    corpus_tokens = 0
    final_chunk_trimmed = False
    for chunk in source_chunks:
        remaining = requested_tokens - corpus_tokens
        if remaining <= 0:
            break
        if chunk.token_count <= remaining:
            selected.append(chunk)
            corpus_tokens += chunk.token_count
        else:
            ids = tokenizer.encode(chunk.text, add_special_tokens=False)[:remaining]
            selected.append(ContextChunk(chunk.chunk_id, chunk.source, tokenizer.decode(ids, skip_special_tokens=True), len(ids)))
            corpus_tokens += len(ids)
            final_chunk_trimmed = True
            break

    if not selected:
        raise ContextError(f"Context {name!r} contains no usable chunks")

    prompt = _render_prompt(question, selected)
    prompt_tokens = count_prompt_tokens(prompt)
    while prompt_tokens > available_input and selected:
        final = selected[-1]
        overflow = prompt_tokens - available_input
        ids = tokenizer.encode(final.text, add_special_tokens=False)
        keep = len(ids) - max(1, overflow)
        if keep <= 0:
            corpus_tokens -= final.token_count
            selected.pop()
        else:
            trimmed_ids = ids[:keep]
            corpus_tokens -= final.token_count - len(trimmed_ids)
            selected[-1] = ContextChunk(final.chunk_id, final.source, tokenizer.decode(trimmed_ids, skip_special_tokens=True), len(trimmed_ids))
            final_chunk_trimmed = True
        prompt = _render_prompt(question, selected)
        prompt_tokens = count_prompt_tokens(prompt)
    if not selected or prompt_tokens > available_input:
        raise ContextError("The question and chat template do not fit within the model context limit")

    return ContextPrompt(prompt, {
        "context_name": name,
        "context_provider": provider_name,
        "benchmark_question": question,
        "requested_context_tokens": requested_tokens,
        "actual_corpus_tokens": corpus_tokens,
        "final_prompt_tokens": prompt_tokens,
        "chunks_used_count": len(selected),
        "maximum_output_tokens": max_new_tokens,
        "model_context_limit": model_context_limit,
        "input_truncation_status": "final_chunk_trimmed" if final_chunk_trimmed else "not_truncated",
    })
