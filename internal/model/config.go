// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the model manifest written by models/convert.py. It carries the
// architecture hyperparameters plus the tokenizer/pooling settings that HF
// spreads across config.json, tokenizer_config.json, and the
// sentence-transformers pooling config.
type Config struct {
	Name                  string  `json:"name"`
	ModelType             string  `json:"model_type,omitempty"` // "", bert, distilbert, modernbert, roberta, or mpnet
	HiddenSize            int     `json:"hidden_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	IntermediateSize      int     `json:"intermediate_size"`
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	LayerNormEps          float32 `json:"layer_norm_eps"`
	DoLowerCase           bool    `json:"do_lower_case"`
	ClsToken              string  `json:"cls_token"`
	SepToken              string  `json:"sep_token"`
	UnkToken              string  `json:"unk_token"`
	Pooling               string  `json:"pooling"`   // "mean" or "cls"
	Normalize             bool    `json:"normalize"` // L2-normalize the pooled vector
	// Tokenizer overrides the architecture's default tokenizer family.
	// "" = by model_type (bert/mpnet → WordPiece, roberta → byte-level
	// BPE); "sentencepiece" = SentencePiece Unigram (the XLM-R tokenizer,
	// which BERT-architecture multilingual models like multilingual-e5 and
	// paraphrase-multilingual MiniLM use).
	Tokenizer string `json:"tokenizer,omitempty"`

	// MPNet only. Positions are offset by PadTokenID+1 (fairseq
	// convention: position ids start at padding_idx+1), and every layer's
	// attention scores share one bucketed relative-position bias table.
	RelativeAttentionNumBuckets int `json:"relative_attention_num_buckets,omitempty"`
	PadTokenID                  int `json:"pad_token_id,omitempty"`

	// ModernBERT only. There are no learned position embeddings: positions
	// enter through RoPE, applied per head with a theta that DIFFERS by
	// attention type. Every GlobalAttnEveryNLayers-th layer (0, N, 2N, …)
	// attends globally with GlobalRopeTheta; the rest attend within a
	// sliding window of LocalAttention tokens (±LocalAttention/2) with
	// LocalRopeTheta. All linear layers and LayerNorms are bias-free.
	GlobalAttnEveryNLayers int     `json:"global_attn_every_n_layers,omitempty"`
	LocalAttention         int     `json:"local_attention,omitempty"`
	GlobalRopeTheta        float64 `json:"global_rope_theta,omitempty"`
	LocalRopeTheta         float64 `json:"local_rope_theta,omitempty"`

	// Qwen3 (decoder embedder) only. A causal decoder used for embeddings:
	// RoPE (single RopeTheta), RMSNorm, per-head QK-norm, grouped-query
	// attention (NumKeyValueHeads < NumAttentionHeads), SwiGLU, and
	// last-token pooling. HeadDim is carried explicitly because it is NOT
	// HiddenSize/NumAttentionHeads (Qwen3-0.6B: 128 vs 1024/16=64).
	HeadDim          int     `json:"head_dim,omitempty"`
	NumKeyValueHeads int     `json:"num_key_value_heads,omitempty"`
	RopeTheta        float64 `json:"rope_theta,omitempty"`
}

// PositionOffset is the value added to a token's index to form its
// position-embedding row: 0 for BERT; PadTokenID+1 for MPNet and RoBERTa
// (the fairseq convention both inherit — the first PadTokenID+1 rows of
// their position tables are padding slots and are never used).
func (c *Config) PositionOffset() int {
	if c.ModelType == "mpnet" || c.ModelType == "roberta" {
		return c.PadTokenID + 1
	}
	return 0
}

// MaxSeqLen is the longest token sequence the model can encode — the
// position-table size minus the offset rows MPNet reserves.
func (c *Config) MaxSeqLen() int {
	return c.MaxPositionEmbeddings - c.PositionOffset()
}

// LoadConfig reads and sanity-checks a manifest.json.
func LoadConfig(path string) (Config, error) {
	var c Config
	raw, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, fmt.Errorf("manifest %s: %w", path, err)
	}
	return c, validate(&c, path)
}

// validate sanity-checks a Config from either source (manifest or HF
// derivation) and defaults LayerNormEps, so both paths share the rules.
func validate(c *Config, source string) error {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 ||
		c.IntermediateSize <= 0 || c.VocabSize <= 0 || c.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("%s: missing or non-positive architecture fields: %+v", source, c)
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return fmt.Errorf("%s: hidden_size %d not divisible by num_attention_heads %d", source, c.HiddenSize, c.NumAttentionHeads)
	}
	if c.Pooling != "mean" && c.Pooling != "cls" && c.Pooling != "lasttoken" {
		return fmt.Errorf("%s: pooling %q unsupported (mean, cls, or lasttoken)", source, c.Pooling)
	}
	switch c.Tokenizer {
	case "", "sentencepiece":
	default:
		return fmt.Errorf("%s: tokenizer %q unsupported (\"\" or sentencepiece)", source, c.Tokenizer)
	}
	switch c.ModelType {
	case "", "bert", "distilbert":
	case "modernbert":
		// ModernBERT has no learned position embeddings; positions enter
		// through RoPE, and its attention alternates global/local layers.
		// These knobs are all read from config (unlike MPNet's, which HF
		// hardcodes) — refuse a manifest missing any of them rather than
		// pick a plausible-but-wrong default.
		if c.GlobalAttnEveryNLayers < 1 {
			return fmt.Errorf("%s: modernbert requires global_attn_every_n_layers >= 1 (found %d)", source, c.GlobalAttnEveryNLayers)
		}
		if c.LocalAttention < 2 || c.LocalAttention%2 != 0 {
			return fmt.Errorf("%s: modernbert requires an even local_attention >= 2 (found %d)", source, c.LocalAttention)
		}
		if c.GlobalRopeTheta <= 0 || c.LocalRopeTheta <= 0 {
			return fmt.Errorf("%s: modernbert requires positive global_rope_theta and local_rope_theta (found %g and %g)", source, c.GlobalRopeTheta, c.LocalRopeTheta)
		}
		// RoPE rotates head_dim in pairs, so the per-head dimension must be
		// even. HiddenSize%NumAttentionHeads==0 is checked above.
		if dh := c.HiddenSize / c.NumAttentionHeads; dh%2 != 0 {
			return fmt.Errorf("%s: modernbert head_dim %d must be even for RoPE", source, dh)
		}
	case "qwen3":
		// A causal decoder used as an embedder. head_dim is explicit (not
		// HiddenSize/heads) and even for RoPE; GQA requires the query heads
		// to partition evenly over the kv heads.
		if c.HeadDim <= 0 || c.HeadDim%2 != 0 {
			return fmt.Errorf("%s: qwen3 requires an even head_dim > 0 (found %d)", source, c.HeadDim)
		}
		if c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
			return fmt.Errorf("%s: qwen3 requires num_attention_heads (%d) divisible by num_key_value_heads (%d)", source, c.NumAttentionHeads, c.NumKeyValueHeads)
		}
		if c.RopeTheta <= 0 {
			return fmt.Errorf("%s: qwen3 requires a positive rope_theta (found %g)", source, c.RopeTheta)
		}
	case "roberta":
		// Unlike MPNet, HF's RoBERTa reads padding_idx from config — but
		// zero is refused deliberately: PadTokenID is a plain int, so a
		// manifest MISSING the field decodes as 0 and would silently
		// shift every position embedding by one row (measured: cosine
		// 0.66 against correct output — plausible-looking garbage). No
		// real RoBERTa uses pad 0; id 0 is <s>.
		if c.PadTokenID < 1 {
			return fmt.Errorf("%s: roberta requires pad_token_id >= 1 (found %d — a manifest without the field reads as 0)", source, c.PadTokenID)
		}
		if c.MaxSeqLen() <= 0 {
			return fmt.Errorf("%s: pad_token_id %d leaves no usable positions (max_position_embeddings %d)", source, c.PadTokenID, c.MaxPositionEmbeddings)
		}
	case "mpnet":
		// HF's MPNet implementation HARDCODES both values in code:
		// compute_position_bias buckets with its default num_buckets=32
		// (config only sizes the embedding table), and MPNetEmbeddings
		// sets padding_idx=1 unconditionally. A checkpoint declaring
		// anything else would make rembed and HF silently diverge —
		// refuse loudly instead of computing a valid-looking wrong answer.
		if c.RelativeAttentionNumBuckets != 32 {
			return fmt.Errorf("%s: relative_attention_num_buckets=%d — HF's MPNet hardcodes 32 regardless of config, so rembed only accepts 32", source, c.RelativeAttentionNumBuckets)
		}
		if c.PadTokenID != 1 {
			return fmt.Errorf("%s: pad_token_id=%d — HF's MPNet embeddings hardcode padding_idx=1 regardless of config, so rembed only accepts 1", source, c.PadTokenID)
		}
	default:
		return fmt.Errorf("%s: model_type %q unsupported (bert, distilbert, modernbert, qwen3, roberta, or mpnet)", source, c.ModelType)
	}
	if c.LayerNormEps == 0 {
		c.LayerNormEps = 1e-12
	}
	return nil
}
