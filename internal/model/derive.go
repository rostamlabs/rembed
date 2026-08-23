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

		// DistilBERT spells its architecture with different keys.
		Dim                *int   `json:"dim"`
		NLayers            *int   `json:"n_layers"`
		NHeads             *int   `json:"n_heads"`
		HiddenDim          *int   `json:"hidden_dim"`
		Activation         string `json:"activation"`
		SinusoidalPosEmbds bool   `json:"sinusoidal_pos_embds"`

		// ModernBERT spells its architecture and knobs its own way too.
		HiddenActivation       string   `json:"hidden_activation"`
		NormEps                *float32 `json:"norm_eps"`
		NormBias               bool     `json:"norm_bias"`
		AttentionBias          bool     `json:"attention_bias"`
		MLPBias                bool     `json:"mlp_bias"`
		GlobalAttnEveryNLayers int      `json:"global_attn_every_n_layers"`
		LocalAttention         int      `json:"local_attention"`
		GlobalRopeTheta        float64  `json:"global_rope_theta"`
		LocalRopeTheta         float64  `json:"local_rope_theta"`

		// Qwen3 (decoder embedder).
		HeadDim          int             `json:"head_dim"`
		NumKeyValueHeads int             `json:"num_key_value_heads"`
		RopeTheta        float64         `json:"rope_theta"`
		RMSNormEps       *float32        `json:"rms_norm_eps"`
		RopeScaling      json.RawMessage `json:"rope_scaling"`
		SlidingWindow    *int            `json:"sliding_window"`
		UseSlidingWindow bool            `json:"use_sliding_window"`
	}
	if err := readJSON("config.json", &hf); err != nil {
		return c, err
	}
	if hf.ModelType == "distilbert" {
		// Fold DistilBERT's key names into the BERT ones so the rest of
		// the derivation (and validate) sees one shape of config.
		if hf.Dim == nil || hf.NLayers == nil || hf.NHeads == nil || hf.HiddenDim == nil {
			return c, fmt.Errorf("model dir %s: distilbert config.json lacks dim/n_layers/n_heads/hidden_dim", dir)
		}
		hf.HiddenSize, hf.NumHiddenLayers = *hf.Dim, *hf.NLayers
		hf.NumAttentionHeads, hf.IntermediateSize = *hf.NHeads, *hf.HiddenDim
		hf.HiddenAct = hf.Activation
		if hf.SinusoidalPosEmbds {
			// The engine only has learned absolute positions; sinusoidal
			// tables would load as garbage rows.
			return c, fmt.Errorf("model dir %s: sinusoidal_pos_embds=true is not supported", dir)
		}
	}
	// The engine hardcodes two encoder architectures, exact-erf GELU, and
	// absolute position embeddings; anything else would produce a
	// valid-looking wrong answer, so refuse loudly here (mirrors
	// convert.py's export-time guards).
	switch hf.ModelType {
	case "bert", "distilbert":
	case "modernbert":
		// ModernBERT is bias-free by design; the loader never allocates bias
		// tensors, so a checkpoint that enabled any bias would silently drop
		// it. Refuse loudly instead. (The one bias HF's ModernBERT carries —
		// the MLM decoder — is not part of the embedding path.)
		if hf.NormBias || hf.AttentionBias || hf.MLPBias {
			return c, fmt.Errorf("model dir %s: modernbert with norm_bias/attention_bias/mlp_bias is not supported (rembed's modernbert path is bias-free)", dir)
		}
		if hf.HiddenActivation != "" && hf.HiddenActivation != "gelu" {
			return c, fmt.Errorf("model dir %s: hidden_activation=%q — only exact GELU is supported", dir, hf.HiddenActivation)
		}
	case "qwen3":
		// A causal decoder used as an embedder. SwiGLU uses SiLU (not the
		// GELU the encoders use). Refuse anything that would silently change
		// the geometry: a rope scaling (YaRN — this checkpoint ships none),
		// any sliding-window attention (Qwen3-Embedding is full causal), and
		// attention_bias (the qwen3 loader allocates no bias tensors, so a
		// biased checkpoint would drop them silently — QK-norm replaced the
		// Qwen2 QKV bias, so real Qwen3 has none).
		if hf.HiddenAct != "" && hf.HiddenAct != "silu" {
			return c, fmt.Errorf("model dir %s: hidden_act=%q — qwen3 supports only silu (SwiGLU)", dir, hf.HiddenAct)
		}
		if hf.AttentionBias {
			return c, fmt.Errorf("model dir %s: qwen3 with attention_bias is not supported (rembed's qwen3 path is bias-free)", dir)
		}
		if len(hf.RopeScaling) > 0 && string(hf.RopeScaling) != "null" {
			return c, fmt.Errorf("model dir %s: qwen3 rope_scaling=%s is not supported (only default RoPE)", dir, hf.RopeScaling)
		}
		if hf.UseSlidingWindow || (hf.SlidingWindow != nil && *hf.SlidingWindow > 0) {
			return c, fmt.Errorf("model dir %s: qwen3 sliding-window attention is not supported (only full causal)", dir)
		}
	case "roberta", "xlm-roberta":
		// XLM-RoBERTa is architecturally identical to RoBERTa (same encoder,
		// same fairseq position offset); only the tokenizer differs — it wears
		// the SentencePiece Unigram model instead of byte-level BPE. It is
		// normalized to model_type="roberta" below so the whole RoBERTa
		// forward/quantize path applies unchanged.
		if hf.PadTokenID == nil {
			// RoBERTa's position rows are offset by pad_token_id+1;
			// guessing it wrong shifts every position embedding.
			return c, fmt.Errorf("model dir %s: %s config.json lacks pad_token_id", dir, hf.ModelType)
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
		return c, fmt.Errorf("model dir %s: model_type=%q — rembed supports bert, distilbert, modernbert, qwen3, roberta, xlm-roberta, and mpnet encoders", dir, hf.ModelType)
	}
	// qwen3's activation (silu, for SwiGLU) is validated in its case above;
	// the encoders' exact-GELU requirement does not apply to it.
	if hf.ModelType != "qwen3" && hf.HiddenAct != "" && hf.HiddenAct != "gelu" {
		key := "hidden_act"
		if hf.ModelType == "distilbert" {
			key = "activation" // distilbert's spelling of the same knob
		}
		return c, fmt.Errorf("model dir %s: %s=%q — only exact GELU is supported", dir, key, hf.HiddenAct)
	}
	if hf.PositionEmbeddingType != "" && hf.PositionEmbeddingType != "absolute" {
		return c, fmt.Errorf("model dir %s: position_embedding_type=%q — only absolute is supported", dir, hf.PositionEmbeddingType)
	}

	var tok struct {
		DoLowerCase    *bool     `json:"do_lower_case"`
		ClsToken       tokString `json:"cls_token"`
		SepToken       tokString `json:"sep_token"`
		UnkToken       tokString `json:"unk_token"`
		TokenizerClass string    `json:"tokenizer_class"`
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
	// XLM-RoBERTa always wears the SentencePiece Unigram tokenizer; a plain
	// BERT-architecture multilingual export (multilingual-e5, paraphrase
	// MiniLM) wears it too and is detected by class or file probe below.
	sentencePiece := hf.ModelType == "xlm-roberta" ||
		(hf.ModelType == "bert" && strings.HasPrefix(tok.TokenizerClass, "XLMRobertaTokenizer"))
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
	case hf.ModelType == "roberta" || hf.ModelType == "modernbert" || hf.ModelType == "qwen3":
		if tok.DoLowerCase != nil && *tok.DoLowerCase {
			return c, fmt.Errorf("model dir %s: do_lower_case=true on a %s model — byte-level BPE is case-preserving", dir, hf.ModelType)
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
		ClsToken:              string(tok.ClsToken),
		SepToken:              string(tok.SepToken),
		UnkToken:              string(tok.UnkToken),
		Pooling:               mode,
		Normalize:             normalize,
	}
	if hf.LayerNormEps != nil && hf.ModelType != "distilbert" {
		// HF's DistilBERT hardcodes LayerNorm eps 1e-12 and never reads
		// the config key; honoring a stray value would silently diverge.
		c.LayerNormEps = *hf.LayerNormEps
	}
	if sentencePiece {
		c.Tokenizer = "sentencepiece"
	}
	switch hf.ModelType {
	case "mpnet":
		c.RelativeAttentionNumBuckets = hf.RelAttnNumBuckets
		c.PadTokenID = *hf.PadTokenID
	case "roberta", "xlm-roberta":
		c.PadTokenID = *hf.PadTokenID
		// XLM-RoBERTa collapses onto the RoBERTa architecture — the only
		// distinction (its SentencePiece tokenizer) is already carried by
		// c.Tokenizer above, so the rest of the engine sees plain roberta.
		c.ModelType = "roberta"
	case "modernbert":
		// ModernBERT spells its LayerNorm eps norm_eps (not layer_norm_eps),
		// so the block above never sets it; carry it here. The RoPE thetas
		// and the global/local attention schedule are all read from config.
		if hf.NormEps != nil {
			c.LayerNormEps = *hf.NormEps
		}
		c.GlobalAttnEveryNLayers = hf.GlobalAttnEveryNLayers
		c.LocalAttention = hf.LocalAttention
		c.GlobalRopeTheta = hf.GlobalRopeTheta
		c.LocalRopeTheta = hf.LocalRopeTheta
	case "qwen3":
		// Qwen3 spells its norm eps rms_norm_eps; carry it plus the explicit
		// head_dim, the kv-head count (GQA), and the RoPE base.
		if hf.RMSNormEps != nil {
			c.LayerNormEps = *hf.RMSNormEps
		}
		c.HeadDim = hf.HeadDim
		c.NumKeyValueHeads = hf.NumKeyValueHeads
		c.RopeTheta = hf.RopeTheta
		// Qwen3-Embedding frames with a trailing <|endoftext|> (the token
		// config.eos_token_id points at, id 151643) and no prefix; last-token
		// pooling reads that position. SepToken carries the suffix content.
		c.ClsToken = ""
		c.SepToken = "<|endoftext|>"
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
	var mean, cls, last bool
	for k, v := range p {
		if !strings.HasPrefix(k, "pooling_mode_") || !truthy(v) {
			continue
		}
		switch k {
		case "pooling_mode_mean_tokens":
			mean = true
		case "pooling_mode_cls_token":
			cls = true
		case "pooling_mode_lasttoken":
			last = true
		default:
			return "", fmt.Errorf("unsupported pooling mode %q (rembed supports mean, cls, or lasttoken)", k)
		}
	}
	switch {
	case b2i(mean)+b2i(cls)+b2i(last) != 1:
		return "", fmt.Errorf("unsupported pooling config %v (rembed supports exactly one of mean, cls, or lasttoken)", p)
	case mean:
		return "mean", nil
	case cls:
		return "cls", nil
	default:
		return "lasttoken", nil
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
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

// tokString unmarshals a special token that HF writes either as a plain
// string or as an AddedToken object ({"content": "<s>", "lstrip": …}) —
// paraphrase-mpnet-base-v2 ships the object form.
type tokString string

func (t *tokString) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*t = tokString(s)
		return nil
	}
	var obj struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	*t = tokString(obj.Content)
	return nil
}
