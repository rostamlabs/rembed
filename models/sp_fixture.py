#!/usr/bin/env python3
"""Dev-only: regenerate the SentencePiece tokenizer fixture.

Writes tokenizer/sentencepiece/testdata/fixture.json with three layers of
reference output per text — sentencepiece's own normalization, its piece
segmentation, and HF XLMRobertaTokenizer input_ids — so a Go mismatch is
attributable to the exact layer (normalizer vs Viterbi vs id mapping).

Usage: python sp_fixture.py [model_id]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    "hello world",
    "Hello World",
    "The quick brown fox jumps over the lazy dog.",
    "a",
    "",
    " ",
    "   ",
    " leading and trailing ",
    "double  space and   triple",
    "tab\tand\nnewline\r\nmix",
    # NFKC-visible normalizations
    "Ｈｅｌｌｏ　ｆｕｌｌｗｉｄｔｈ",     # fullwidth + ideographic space
    "ﬁ ﬂ ligatures and ㎒ ㎡ units",
    "nbsp thin hair spaces",
    "café résumé naïve déjà vu",
    "ＡＢＣ１２３ meets abc123",
    # scripts
    "سلام دنیا! این یک آزمایش فارسی است.",
    "مرحبا بالعالم",
    "你好世界，这是一个测试。",
    "こんにちは世界",
    "Привет мир",
    "Γειά σου Κόσμε",
    "שלום עולם",
    "안녕하세요 세계",
    "नमस्ते दुनिया",
    # mixed + symbols + emoji
    "€100 costs $5, or ±43 at 20° ± 3°",
    "emoji 🚀 and 🧪 in text",
    "punctuation, splitting: does it (really) work?! yes -- it does...",
    "code_like_identifiers and kebab-case and CamelCase",
    "In 2024, the model processed 1,234,567 queries at 99.9% availability.",
    "rare unicode: ᚠᚢᚦ runes and ⠃⠗⠁ braille",
    # long multilingual text (>512 tokens once encoded, exercises truncation)
    ("زبان فارسی یکی از زبان‌های هندواروپایی است که در ایران و افغانستان و "
     "تاجیکستان به آن سخن می‌گویند و ادبیاتی کهن و پربار دارد. " * 30),
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(model_id, "sentencepiece.bpe.model"))
    tok = AutoTokenizer.from_pretrained(model_id)

    cases = []
    for text in TEXTS:
        enc = tok(text, truncation=True, max_length=512)
        cases.append({
            "text": text,
            "normalized": sp.normalize(text),
            "pieces": sp.encode(text, out_type=str),
            "input_ids": enc["input_ids"],
        })
    out = REPO_ROOT / "tokenizer" / "sentencepiece" / "testdata" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
