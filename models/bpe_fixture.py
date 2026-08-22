#!/usr/bin/env python3
"""Dev-only: regenerate the byte-level BPE tokenizer fixture.

Writes tokenizer/bpe/testdata/fixture.json: HF-produced input_ids
(specials included) for a battery of texts chosen to hit every
pre-tokenizer branch — contractions, space-joining, whitespace-run
backtracking, non-space whitespace, unicode classes, and raw bytes.
The Go test requires token-for-token equality.

Usage: python bpe_fixture.py [model_id]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    "hello world",
    "Hello World",  # case preserved — byte-level BPE never lowercases
    "The quick brown fox jumps over the lazy dog.",
    "it's don't we're you've I'm they'll she'd",  # every contraction branch
    "it'S DON'T",  # contractions are case-sensitive in the pattern
    " leading space",
    "trailing space ",
    "double  space and   triple",  # \s+(?!\S) backtracking
    "tab\tand\nnewline\r\nmix",  # non-space whitespace cannot join as prefix
    "line one\n line two",  # newline then space then word
    "a",
    "",
    " ",
    "   ",
    "punctuation, splitting: does it (really) work?! yes -- it does...",
    "!!!'s apostrophe inside a punctuation run",
    " 're a space then an apostrophe",
    "café résumé naïve déjà vu at the sørensen-müller café",
    "€100 costs $5, or ±43 at 20° ± 3°",
    "你好世界 mixed with english text",
    "emoji 🚀 and 🧪 in text",
    "numbers 123 4567 89, and ½ ¾ Ⅻ",  # No and Nl number categories are \p{N}
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "snake_case and kebab-case and CamelCase and dot.case",
    "a1b2c3 mixes letters and digits per branch",
    "quotes \"double\" and 'single' and `backtick`",
    "In 2024, the model processed 1,234,567 queries at 99.9% availability, "
    "averaging 3.14 ms per request across 42 regions.",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-distilroberta-v1"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    cases = []
    for text in TEXTS:
        enc = tok(text, truncation=True, max_length=512)
        cases.append({"text": text, "input_ids": enc["input_ids"]})
    out = REPO_ROOT / "tokenizer" / "bpe" / "testdata" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
