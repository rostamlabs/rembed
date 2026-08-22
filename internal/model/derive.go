// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DeriveConfig builds a Config directly from a Hugging Face
// sentence-transformers model directory (config.json,
// tokenizer_config.json, 1_Pooling/config.json, optional modules.json) —
// the same derivation models/convert.py performs in Python, with the same
// refuse-what-we-cannot-compute guards. This is what lets rembed.Load work
// on a hub download or a plain git-cloned HF repo with no conversion step;
// manifest.json remains supported and wins when present.
func DeriveConfig(dir, name string) (Config, error) {
	var c Config
	readJSON := func(rel string, dst any) error {
		raw, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		if err := json.Unmarshal(raw, dst); err != nil {
			return fmt.Errorf("model dir %s: %s: %w", dir, rel, err)
		}
		return nil
	}

	var hf struct {
		ModelType             string   `json:"model_type"`
		HiddenAct             string   `json:"hidden_act"`
		PositionEmbeddingType string   `json:"position_embedding_type"`
		HiddenSize            int      `json:"hidden_size"`
		NumHiddenLayers       int      `json:"num_hidden_layers"`
		NumAttentionHeads     int      `json:"num_attention_heads"`
		IntermediateSize      int      `json:"intermediate_size"`
		VocabSize             int      `json:"vocab_size"`
		MaxPositionEmbeddings int      `json:"max_position_embeddings"`
		LayerNormEps          *float32 `json:"layer_norm_eps"`
		RelAttnNumBuckets     int      `json:"relative_attention_num_buckets"`
		PadTokenID            *int     `json:"pad_token_id"`
	}
	if err := readJSON("config.json", &hf); err != nil {
		return c, err
	}
	// The engine hardcodes two encoder architectures, exact-erf GELU, and
	// absolute position embeddings; anything else would produce a
	// valid-looking wrong answer, so refuse loudly here (mirrors
	// convert.py's export-time guards).
	switch hf.ModelType {
	case "bert":
	case "roberta":
		if hf.PadTokenID == nil {
			// RoBERTa's position rows are offset by pad_token_id+1;
			// guessing it wrong shifts every position embedding.
			return c, fmt.Errorf("model dir %s: roberta config.json lacks pad_token_id", dir)
		}
	case "mpnet":
		if hf.RelAttnNumBuckets <= 0 {
			return c, fmt.Errorf("model dir %s: mpnet config.json lacks relative_attention_num_buckets", dir)
		}
		if hf.PadTokenID == nil {
			// Every MPNet checkpoint sets it, and validate() then insists
			// on 1 — the value HF's code hardcodes as padding_idx no
			// matter what config says.
			return c, fmt.Errorf("model dir %s: mpnet config.json lacks pad_token_id", dir)
		}
	default:
		return c, fmt.Errorf("model dir %s: model_type=%q — rembed supports bert, roberta, and mpnet encoders", dir, hf.ModelType)
	}
	if hf.HiddenAct != "" && hf.HiddenAct != "gelu" {
		return c, fmt.Errorf("model dir %s: hidden_act=%q — only exact GELU is supported", dir, hf.HiddenAct)
	}
	if hf.PositionEmbeddingType != "" && hf.PositionEmbeddingType != "absolute" {
		return c, fmt.Errorf("model dir %s: position_embedding_type=%q — only absolute is supported", dir, hf.PositionEmbeddingType)
	}

	var tok struct {
		DoLowerCase    *bool  `json:"do_lower_case"`
		ClsToken       string `json:"cls_token"`
		SepToken       string `json:"sep_token"`
		UnkToken       string `json:"unk_token"`
		TokenizerClass string `json:"tokenizer_class"`
	}
	if err := readJSON("tokenizer_config.json", &tok); err != nil {
		return c, err
	}

	var pooling map[string]any
	if err := readJSON("1_Pooling/config.json", &pooling); err != nil {
		return c, fmt.Errorf("model dir %s: no 1_Pooling/config.json — not sentence-transformers format: %w", dir, err)
	}
	mode, err := poolingMode(pooling)
	if err != nil {
		return c, fmt.Errorf("model dir %s: %w", dir, err)
	}

	// XLM-R-family repos ship a SentencePiece model even on a plain BERT
	// architecture (multilingual-e5, paraphrase-multilingual MiniLM). The
	// FILE is the authoritative signal: older exports lack the
	// tokenizer_class field entirely (the multilingual MiniLM does).
	// Only BERT-architecture repos wear the XLM-R tokenizer (the ST
	// multilingual exports are all model_type=bert); keying the file probe
	// on that keeps a stray sentencepiece.bpe.model in e.g. a roberta dir
	// from silently switching tokenizer family.
	sentencePiece := hf.ModelType == "bert" && strings.HasPrefix(tok.TokenizerClass, "XLMRobertaTokenizer")
	if !sentencePiece && hf.ModelType == "bert" {
		if fi, err := os.Stat(filepath.Join(dir, "sentencepiece.bpe.model")); err == nil && !fi.IsDir() {
			sentencePiece = true
		}
	}

	// The WordPiece tokenizer couples lowercasing to accent stripping, so
	// guessing this wrong yields plausible-but-wrong embeddings. Every
	// surveyed BERT/ST repo sets the field explicitly; refuse rather than
	// guess. RoBERTa's byte-level BPE never carries the field. SentencePiece
	// repos sometimes DO carry a stale do_lower_case=true that HF's
	// XLMRobertaTokenizer never reads (the multilingual MiniLM does) — so
	// it is IGNORED there, matching HF, rather than refused.
	doLower := false
	switch {
	case sentencePiece:
	case hf.ModelType == "roberta":
		if tok.DoLowerCase != nil && *tok.DoLowerCase {
			return c, fmt.Errorf("model dir %s: do_lower_case=true on a roberta model — byte-level BPE is case-preserving", dir)
		}
	default:
		if tok.DoLowerCase == nil {
			return c, fmt.Errorf("model dir %s: tokenizer_config.json does not set do_lower_case — set it there or provide a manifest.json", dir)
		}
		doLower = *tok.DoLowerCase
	}

	// modules.json is required: silently assuming "no Normalize module"
	// for a repo that has one would shift every downstream cosine
	// threshold (convert.py enforces the same).
	normalize := false
	var modules []struct {
		Type string `json:"type"`
	}
	if err := readJSON("modules.json", &modules); err != nil {
		return c, fmt.Errorf("model dir %s: modules.json: %w", dir, err)
	}
	for _, m := range modules {
		if strings.HasSuffix(m.Type, "models.Normalize") {
			normalize = true
		} else if !strings.HasSuffix(m.Type, "models.Transformer") && !strings.HasSuffix(m.Type, "models.Pooling") {
			return c, fmt.Errorf("model dir %s: unsupported sentence-transformers module %q", dir, m.Type)
		}
	}

	c = Config{
		Name:                  name,
		ModelType:             hf.ModelType,
		HiddenSize:            hf.HiddenSize,
		NumHiddenLayers:       hf.NumHiddenLayers,
		NumAttentionHeads:     hf.NumAttentionHeads,
		IntermediateSize:      hf.IntermediateSize,
		VocabSize:             hf.VocabSize,
		MaxPositionEmbeddings: hf.MaxPositionEmbeddings,
		DoLowerCase:           doLower,
		ClsToken:              tok.ClsToken,
		SepToken:              tok.SepToken,
		UnkToken:              tok.UnkToken,
		Pooling:               mode,
		Normalize:             normalize,
	}
	if hf.LayerNormEps != nil {
		c.LayerNormEps = *hf.LayerNormEps
	}
	if sentencePiece {
		c.Tokenizer = "sentencepiece"
	}
	switch hf.ModelType {
	case "mpnet":
		c.RelativeAttentionNumBuckets = hf.RelAttnNumBuckets
		c.PadTokenID = *hf.PadTokenID
	case "roberta":
		c.PadTokenID = *hf.PadTokenID
	}
	return c, validate(&c, dir)
}

// poolingMode maps a 1_Pooling/config.json onto rembed's pooling
// strategies. The check is an ALLOWLIST over every pooling_mode_* key —
// truthy in the JSON sense, so a future or nonstandard mode (or "1"
// instead of true) refuses loudly instead of being silently ignored.
func poolingMode(p map[string]any) (string, error) {
	truthy := func(v any) bool {
		switch x := v.(type) {
		case bool:
			return x
		case float64:
			return x != 0
		case string:
			return x != "" && x != "false" && x != "False"
		default:
			return false
		}
	}
	var mean, cls bool
	for k, v := range p {
		if !strings.HasPrefix(k, "pooling_mode_") || !truthy(v) {
			continue
		}
		switch k {
		case "pooling_mode_mean_tokens":
			mean = true
		case "pooling_mode_cls_token":
			cls = true
		default:
			return "", fmt.Errorf("unsupported pooling mode %q (rembed supports mean or cls)", k)
		}
	}
	switch {
	case mean && cls, !mean && !cls:
		return "", fmt.Errorf("unsupported pooling config %v (rembed supports exactly one of mean or cls)", p)
	case mean:
		return "mean", nil
	default:
		return "cls", nil
	}
}

// LoadConfigOrDerive prefers manifest.json (a converted model dir) and
// falls back to deriving from the HF files (a hub download or repo clone).
func LoadConfigOrDerive(dir, name string) (Config, error) {
	manifest := filepath.Join(dir, "manifest.json")
	if _, err := os.Stat(manifest); err == nil {
		return LoadConfig(manifest)
	}
	return DeriveConfig(dir, name)
}
