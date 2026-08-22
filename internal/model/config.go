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
	if c.LayerNormEps == 0 {
		c.LayerNormEps = 1e-12
	}
	return nil
}
