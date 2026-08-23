#!/usr/bin/env python3
"""Dev-only: regenerate the Qwen3 byte-level BPE tokenizer fixture.

Writes tokenizer/bpe/testdata/qwen_fixture.json: HF-produced input_ids
(the appended <|endoftext|> included) for a battery hitting the ways the
Qwen/GPT-NeoX pre-tokenizer differs from GPT-2 — case-insensitive
contractions, single-digit splitting, the broad letter prefix, newline
handling — plus NFC and the suffix-only framing. The Go test requires
token-for-token equality.

Usage: python qwen_fixture.py [model_id]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    "hello world", "Hello World",
    "The quick brown fox jumps over the lazy dog.",
    "it's don't we're WE'RE Can't I'M they'll",  # case-insensitive contractions
    "don'T CAN'T won'T",
    "café résumé naïve déjà vu", "€100 costs $5, or ±43 at 20°",
    "你好世界 mixed english", "emoji 🚀 🧪",
    "a", "", "  ", "   ",
    "2024 and 1,234,567", "123 456 7890",  # single-digit splitting
    "foo.bar_baz a1b2c3",
    "punctuation, splitting: does it (really) work?!",
    "line one\nline two\n\nline three", "tab\there\ttab",
    "  leading", "trailing  ", "word\n  indented",
    "<|endoftext|> <think> mid",  # specials mid-text
    "سلام دنیا! می‌کند", "①②③ ½ ¾",
    "https://example.com/path?q=1", "SELECT * FROM t WHERE a='b';",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    cases = [{"text": t, "input_ids": tok(t)["input_ids"]} for t in TEXTS]
    out = REPO_ROOT / "tokenizer" / "bpe" / "testdata" / "qwen_fixture.json"
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
