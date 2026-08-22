// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestRelPosBucket pins the Go port against HF's relative_position_bucket
// (num_buckets=32, max_distance=128). Expected values were computed with
// the reference formula under float32 semantics (torch computes the log in
// float32); the cases bracket every regime boundary — the exact/log split
// at |n|=8, the max-distance clamp at 128, the sign split, and the
// extremes a 512-token sequence can produce.
func TestRelPosBucket(t *testing.T) {
	cases := [][2]int{
		{-513, 15}, {-512, 15}, {-129, 15}, {-128, 15}, {-127, 15},
		{-64, 14}, {-17, 10}, {-16, 10}, {-9, 8}, {-8, 8}, {-7, 7},
		{-1, 1}, {0, 0}, {1, 17}, {7, 23}, {8, 24}, {9, 24},
		{16, 26}, {17, 26}, {64, 30}, {127, 31}, {128, 31}, {129, 31},
		{512, 31}, {513, 31},
	}
	for _, c := range cases {
		if got := relPosBucket(c[0], 32); got != c[1] {
			t.Errorf("relPosBucket(%d) = %d, want %d", c[0], got, c[1])
		}
	}
	// Buckets must stay in range for every delta a max-length sequence
	// can produce, in both directions.
	for d := -513; d <= 513; d++ {
		if b := relPosBucket(d, 32); b < 0 || b >= 32 {
			t.Fatalf("relPosBucket(%d) = %d out of [0,32)", d, b)
		}
	}
}

// TestDeriveConfigMPNet drives the derivation on a synthetic MPNet repo
// layout and checks the mpnet-specific plumbing: model_type, buckets,
// pad offset, and the usable-sequence ceiling.
func TestDeriveConfigMPNet(t *testing.T) {
	dir := t.TempDir()
	write := func(rel, content string) {
		t.Helper()
		path := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("config.json", `{"model_type":"mpnet","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":30527,"max_position_embeddings":514,"layer_norm_eps":1e-5,
		"relative_attention_num_buckets":32,"pad_token_id":1}`)
	write("tokenizer_config.json", `{"do_lower_case":true,"cls_token":"<s>","sep_token":"</s>","unk_token":"[UNK]"}`)
	write("1_Pooling/config.json", `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`)
	write("modules.json", `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"},
		{"type":"sentence_transformers.models.Normalize"}]`)

	cfg, err := DeriveConfig(dir, "test-mpnet")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "mpnet" || cfg.RelativeAttentionNumBuckets != 32 || cfg.PadTokenID != 1 {
		t.Fatalf("mpnet fields not derived: %+v", cfg)
	}
	if cfg.PositionOffset() != 2 || cfg.MaxSeqLen() != 512 {
		t.Fatalf("offset/ceiling wrong: offset=%d maxSeq=%d", cfg.PositionOffset(), cfg.MaxSeqLen())
	}
	if !cfg.Normalize || cfg.Pooling != "mean" || cfg.LayerNormEps != 1e-5 {
		t.Fatalf("shared fields wrong: %+v", cfg)
	}

	// An mpnet config without the bias table's size must refuse: loading
	// would otherwise guess the attention arithmetic.
	write("config.json", `{"model_type":"mpnet","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":30527,"max_position_embeddings":514,"pad_token_id":1}`)
	if _, err := DeriveConfig(dir, "test-mpnet"); err == nil {
		t.Fatal("expected error for missing relative_attention_num_buckets")
	}
}

// TestManifestRoundTripMPNet ensures the mpnet fields survive
// manifest.json (convert.py writes them; LoadConfig must read them).
func TestManifestRoundTripMPNet(t *testing.T) {
	cfg := Config{
		Name: "m", ModelType: "mpnet", HiddenSize: 768, NumHiddenLayers: 12,
		NumAttentionHeads: 12, IntermediateSize: 3072, VocabSize: 30527,
		MaxPositionEmbeddings: 514, LayerNormEps: 1e-5, Pooling: "mean",
		Normalize: true, RelativeAttentionNumBuckets: 32, PadTokenID: 1,
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != cfg {
		t.Fatalf("round trip changed config:\n got %+v\nwant %+v", got, cfg)
	}
}
