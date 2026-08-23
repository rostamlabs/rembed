#!/usr/bin/env python3
"""Dev-only: regenerate the Gemma byte-fallback BPE tokenizer fixture.

Writes tokenizer/gemma/testdata/gemma_fixture.json: HF-produced input_ids
(the <bos>...<eos> framing included) for a battery hitting the ways Gemma's
SentencePiece-style BPE differs from byte-level BPE — metaspace (space ->
U+2581) normalization, byte_fallback for out-of-vocab characters (emoji,
rare scripts), fuse_unk, digit and whitespace handling, and the task-prompt
prefixes EmbeddingGemma ships. The Go test requires token-for-token
equality.

Usage: python gemma_fixture.py [model_id]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    "hello world", "Hello World",
    "The quick brown fox jumps over the lazy dog.",
    "it's don't we're I'm they'll",
    "café résumé naïve déjà vu", "€100 costs $5, or ±43 at 20°",
    "你好世界 mixed english", "emoji 🚀 🧪 �e",  # byte_fallback territory
    "a", "", " ", "  ", "   ",
    "2024 and 1,234,567", "123 456 7890",
    "foo.bar_baz a1b2c3",
    "punctuation, splitting: does it (really) work?!",
    "line one\nline two\n\nline three", "tab\there\ttab",
    "  leading", "trailing  ", "word\n  indented",
    "سلام دنیا! می‌کند", "①②③ ½ ¾",
    "https://example.com/path?q=1", "SELECT * FROM t WHERE a='b';",
    # EmbeddingGemma task prompts (query/document framing) + content
    "task: search result | query: what is a vector database",
    "title: none | text: Rostam is a vector database written in Go.",
    "▁literal metaspace char", "null\x00byte",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-300m"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    cases = [{"text": t, "input_ids": tok(t)["input_ids"]} for t in TEXTS]
    out = REPO_ROOT / "tokenizer" / "gemma" / "testdata" / "gemma_fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
