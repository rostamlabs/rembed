#!/usr/bin/env python3
"""Dev-only: regenerate the NFD-divergence fixture.

These inputs are where HF's FAST tokenizer is a known outlier: it applies
the charsmap per grapheme cluster and skips clusters >= 6 bytes, so NFD
(decomposed) Hangul and kana never compose there. rembed deliberately
matches the sentencepiece C++ reference instead, so this fixture carries
ONLY the sentencepiece layers (normalize + pieces), no HF ids.

Usage: python sp_nfd_fixture.py [model_id]
"""

import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NFC_TEXTS = [
    "안녕하세요 세계, 오늘 날씨가 좋네요.",
    "한국어 문장 임베딩 테스트입니다",
    "ドイツ語とバングラデシュ",
    "がぎぐげご ぱぴぷぺぽ",
    "café résumé naïve",
    "नमस्ते दुनिया",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(model_id, "sentencepiece.bpe.model"))
    cases = []
    for nfc in NFC_TEXTS:
        nfd = unicodedata.normalize("NFD", nfc)
        if nfd == nfc:
            continue  # no precomposed form to decompose (e.g. Devanagari)
        cases.append({
            "text": nfd,
            "normalized": sp.normalize(nfd),
            "pieces": sp.encode(nfd, out_type=str),
            # The composed form must tokenize identically — NFKC inside
            # the charsmap composes NFD back.
            "nfc_pieces": sp.encode(nfc, out_type=str),
        })
    out = REPO_ROOT / "tokenizer" / "sentencepiece" / "testdata" / "fixture-nfd.json"
    out.write_text(json.dumps({"model": model_id, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
