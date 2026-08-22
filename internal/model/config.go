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
	ModelType             string  `json:"model_type,omitempty"` // "" or "bert", or "mpnet"
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

	// MPNet only. Positions are offset by PadTokenID+1 (fairseq
	// convention: position ids start at padding_idx+1), and every layer's
	// attention scores share one bucketed relative-position bias table.
	RelativeAttentionNumBuckets int `json:"relative_attention_num_buckets,omitempty"`
	PadTokenID                  int `json:"pad_token_id,omitempty"`
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
	if c.Pooling != "mean" && c.Pooling != "cls" {
		return fmt.Errorf("%s: pooling %q unsupported (mean or cls)", source, c.Pooling)
	}
	switch c.ModelType {
	case "", "bert":
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
		return fmt.Errorf("%s: model_type %q unsupported (bert, roberta, or mpnet)", source, c.ModelType)
	}
	if c.LayerNormEps == 0 {
		c.LayerNormEps = 1e-12
	}
	return nil
}
