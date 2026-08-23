#!/usr/bin/env python3
"""Dev-only: regenerate the ModernBERT byte-level BPE tokenizer fixture.

Writes tokenizer/bpe/testdata/mb_fixture.json: HF-produced input_ids
(specials included) for a battery of texts chosen to hit the parts that
differ from the RoBERTa byte-level BPE — NFC normalization, and the
OLMo-inherited added tokens (whitespace runs of 2..24 spaces, the
|||MARKER||| tokens, and [unusedN]) which HF matches leftmost-longest
BEFORE the BPE merges. The Go test requires token-for-token equality.

Usage: python mb_fixture.py [model_id]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    # Ordinary text (shared with the byte-BPE battery — must still agree).
    "hello world",
    "Hello World",
    "The quick brown fox jumps over the lazy dog.",
    "it's don't we're you've I'm they'll she'd",
    "punctuation, splitting: does it (really) work?! yes -- it does...",
    "café résumé naïve déjà vu at the sørensen-müller café",
    "€100 costs $5, or ±43 at 20° ± 3°",
    "你好世界 mixed with english text",
    "emoji 🚀 and 🧪 in text",
    "a", "", " ",
    # Whitespace-run added tokens: 1..30 spaces (1 is BPE, 2..24 are added
    # tokens, 25+ is longest-match 24 then remainder).
    *[" " * n for n in range(1, 31)],
    "code:\n    indented four\n        indented eight\n",
    "def f():\n        return 1  # eight-space body",
    "a" + " " * 24 + "b",
    "x" + " " * 25 + "y",
    "trailing spaces then word" + " " * 12 + "END",
    # Marker added tokens.
    "contact |||EMAIL_ADDRESS||| or |||PHONE_NUMBER|||",
    "ip |||IP_ADDRESS||| here",
    "[unused0] and [unused42] and [unused82]",
    "literal [CLS] and [SEP] and [MASK] in text",
    "<|endoftext|> mid sentence <|padding|>",
    # NFC normalization: decomposed sequences must fold to composed.
    "café vs café",          # e+combining acute vs é
    "Å ring vs Å",           # A+ring vs Å
    "ẛ̣",                       # long s with dot (multi-combining)
    "ﬀ ﬁ ﬂ ligatures",                  # NFC leaves these (compat only in NFKC)
    "① ② ③ ½ ¾",
    # Zero-width joiners/non-joiners (Persian, emoji ZWJ).
    "می‌کند",                       # ZWNJ inside Persian
    "test‌test‍test",
    "👨‍👩‍👧",                 # family emoji via ZWJ
    "سلام دنیا! جستجوی معنایی به زبان فارسی هم کار می‌کند.",
    # Mixed everything.
    "  \t\n  mixed  ws\ttab" + " " * 6 + "café‌①",
    "https://example.com/path?q=1&x=2#frag",
    "SELECT * FROM t WHERE a='b';   -- comment",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "nomic-ai/modernbert-embed-base"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    cases = [{"text": t, "input_ids": tok(t)["input_ids"]} for t in TEXTS]
    out = REPO_ROOT / "tokenizer" / "bpe" / "testdata" / "mb_fixture.json"
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
