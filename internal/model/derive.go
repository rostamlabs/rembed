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
	}
	if err := readJSON("config.json", &hf); err != nil {
		return c, err
	}
	// The engine hardcodes the BERT encoder, exact-erf GELU, and absolute
	// positions; anything else would produce a valid-looking wrong answer,
	// so refuse loudly here (mirrors convert.py's export-time guards).
	if hf.ModelType != "bert" {
		return c, fmt.Errorf("model dir %s: model_type=%q — rembed supports BERT-family encoders", dir, hf.ModelType)
	}
	if hf.HiddenAct != "" && hf.HiddenAct != "gelu" {
		return c, fmt.Errorf("model dir %s: hidden_act=%q — only exact GELU is supported", dir, hf.HiddenAct)
	}
	if hf.PositionEmbeddingType != "" && hf.PositionEmbeddingType != "absolute" {
		return c, fmt.Errorf("model dir %s: position_embedding_type=%q — only absolute is supported", dir, hf.PositionEmbeddingType)
	}

	var tok struct {
		DoLowerCase *bool  `json:"do_lower_case"`
		ClsToken    string `json:"cls_token"`
		SepToken    string `json:"sep_token"`
		UnkToken    string `json:"unk_token"`
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

	normalize := false
	var modules []struct {
		Type string `json:"type"`
	}
	if err := readJSON("modules.json", &modules); err == nil {
		for _, m := range modules {
			if strings.HasSuffix(m.Type, "models.Normalize") {
				normalize = true
			} else if !strings.HasSuffix(m.Type, "models.Transformer") && !strings.HasSuffix(m.Type, "models.Pooling") {
				return c, fmt.Errorf("model dir %s: unsupported sentence-transformers module %q", dir, m.Type)
			}
		}
	} else if !os.IsNotExist(err) {
		return c, err
	}

	c = Config{
		Name:                  name,
		HiddenSize:            hf.HiddenSize,
		NumHiddenLayers:       hf.NumHiddenLayers,
		NumAttentionHeads:     hf.NumAttentionHeads,
		IntermediateSize:      hf.IntermediateSize,
		VocabSize:             hf.VocabSize,
		MaxPositionEmbeddings: hf.MaxPositionEmbeddings,
		DoLowerCase:           tok.DoLowerCase == nil || *tok.DoLowerCase,
		ClsToken:              tok.ClsToken,
		SepToken:              tok.SepToken,
		UnkToken:              tok.UnkToken,
		Pooling:               mode,
		Normalize:             normalize,
	}
	if hf.LayerNormEps != nil {
		c.LayerNormEps = *hf.LayerNormEps
	}
	return c, validate(&c, dir)
}

// poolingMode maps a 1_Pooling/config.json onto rembed's pooling
// strategies, refusing combinations the engine does not compute.
func poolingMode(p map[string]any) (string, error) {
	truthy := func(k string) bool {
		v, ok := p[k].(bool)
		return ok && v
	}
	mean := truthy("pooling_mode_mean_tokens")
	cls := truthy("pooling_mode_cls_token")
	other := truthy("pooling_mode_max_tokens") || truthy("pooling_mode_lasttoken") ||
		truthy("pooling_mode_mean_sqrt_len_tokens") || truthy("pooling_mode_weightedmean_tokens")
	switch {
	case other, mean && cls:
		return "", fmt.Errorf("unsupported pooling config %v (rembed supports mean or cls)", p)
	case mean:
		return "mean", nil
	case cls:
		return "cls", nil
	default:
		return "", fmt.Errorf("no supported pooling mode in %v", p)
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
