from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_benchmark.app import parse_args, parse_context_command, prompt_preview
from llm_benchmark.contexts import (
    ContextChunk,
    ContextError,
    FolderContextProvider,
    build_context_prompt,
    chunk_documents,
    prepare_folder_context,
)


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(tokens)


class ContextPreparationTests(unittest.TestCase):
    def test_token_chunks_and_overlap(self) -> None:
        chunks = list(chunk_documents(
            [("source.txt", "zero one two three four five")],
            WordTokenizer(), chunk_tokens=4, overlap=2,
        ))
        self.assertEqual([chunk.token_count for chunk in chunks], [4, 4])
        self.assertEqual(chunks[1].text, "two three four five")

    def test_prepare_and_load_folder_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "contexts"
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text("one two three four five", encoding="utf-8")
            manifest = prepare_folder_context(
                "sample", str(corpus), WordTokenizer(),
                chunk_tokens=3, overlap=1, root=root,
            )
            loaded = FolderContextProvider(root).load("sample")
            self.assertEqual(manifest.chunks, 2)
            self.assertEqual(loaded[0].chunk_id, "chunk-00000000")
            record = json.loads((root / "sample" / "chunks.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(record), {"chunk_id", "source", "text", "token_count"})

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContextError, "smaller"):
            list(chunk_documents([], WordTokenizer(), chunk_tokens=10, overlap=10))


class ContextPromptTests(unittest.TestCase):
    def test_complete_chunks_are_selected_to_requested_budget(self) -> None:
        provider = type("Provider", (), {
            "name": "mock",
            "load": lambda self, name: [
                ContextChunk("a", "one", "1 2 3", 3),
                ContextChunk("b", "two", "4 5 6", 3),
            ],
        })()
        from llm_benchmark import contexts
        original = contexts.CONTEXT_PROVIDERS
        contexts.CONTEXT_PROVIDERS = {"mock": lambda: provider}
        try:
            result = build_context_prompt(
                name="test", requested_tokens=5, question="question",
                tokenizer=WordTokenizer(), count_prompt_tokens=lambda text: len(text.split()),
                model_context_limit=100, max_new_tokens=10, provider_name="mock",
            )
        finally:
            contexts.CONTEXT_PROVIDERS = original
        self.assertEqual(result.metadata["actual_corpus_tokens"], 5)
        self.assertEqual(result.metadata["input_truncation_status"], "final_chunk_trimmed")
        self.assertEqual(result.metadata["chunks_used_count"], 2)
        self.assertNotIn("chunks_used", result.metadata)

    def test_prompt_budget_trims_only_last_chunk(self) -> None:
        provider = type("Provider", (), {
            "name": "mock",
            "load": lambda self, name: [
                ContextChunk("a", "one", "1 2 3", 3),
                ContextChunk("b", "two", "4 5 6 7 8", 5),
            ],
        })()
        from llm_benchmark import contexts
        original = contexts.CONTEXT_PROVIDERS
        contexts.CONTEXT_PROVIDERS = {"mock": lambda: provider}
        try:
            result = build_context_prompt(
                name="test", requested_tokens=8, question="q",
                tokenizer=WordTokenizer(), count_prompt_tokens=lambda text: len(text.split()),
                model_context_limit=42, max_new_tokens=10, provider_name="mock",
            )
        finally:
            contexts.CONTEXT_PROVIDERS = original
        self.assertEqual(result.metadata["chunks_used_count"], 2)
        self.assertEqual(result.metadata["actual_corpus_tokens"], 5)
        self.assertLessEqual(result.metadata["final_prompt_tokens"], 32)


class ContextCliTests(unittest.TestCase):
    def test_context_prompt_preview_is_word_limited(self) -> None:
        preview = prompt_preview("one two three four five", word_limit=3)
        self.assertEqual(preview, "one two three … [truncated]")

    def test_benchmark_context_arguments(self) -> None:
        args = parse_args(["--context", "wikitext", "--context-tokens", "32000"])
        self.assertEqual((args.context, args.context_tokens), ("wikitext", 32000))

    def test_prepare_context_arguments(self) -> None:
        args = parse_args(["--prepare-context", "wikitext", "--corpus", "wikitext-103-raw", "--chunk-tokens", "2048", "--chunk-overlap", "256"])
        self.assertEqual((args.prepare_context, args.chunk_tokens, args.chunk_overlap), ("wikitext", 2048, 256))

    def test_runtime_context_command(self) -> None:
        self.assertEqual(
            parse_context_command("/context wikitext 32000"),
            ("wikitext", 32000),
        )

    def test_runtime_context_can_be_disabled(self) -> None:
        self.assertEqual(parse_context_command("/context off"), (None, None))

    def test_runtime_context_rejects_invalid_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            parse_context_command("/context wikitext 0")
