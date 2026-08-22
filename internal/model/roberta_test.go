// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDeriveConfigRoberta checks the roberta derivation: pad-offset
// plumbing, the do_lower_case exemption (byte-level BPE is
// case-preserving and roberta configs do not carry the field), and the
// refusal when a roberta config claims lowercasing.
func TestDeriveConfigRoberta(t *testing.T) {
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
	write("config.json", `{"model_type":"roberta","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":6,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":50265,"max_position_embeddings":514,"layer_norm_eps":1e-5,
		"position_embedding_type":"absolute","pad_token_id":1}`)
	write("tokenizer_config.json", `{"cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	write("1_Pooling/config.json", `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`)
	write("modules.json", `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"},
		{"type":"sentence_transformers.models.Normalize"}]`)

	cfg, err := DeriveConfig(dir, "test-roberta")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "roberta" || cfg.PadTokenID != 1 {
		t.Fatalf("roberta fields not derived: %+v", cfg)
	}
	if cfg.PositionOffset() != 2 || cfg.MaxSeqLen() != 512 {
		t.Fatalf("offset/ceiling wrong: offset=%d maxSeq=%d", cfg.PositionOffset(), cfg.MaxSeqLen())
	}
	if cfg.DoLowerCase {
		t.Fatal("roberta derived do_lower_case=true; byte-level BPE never lowercases")
	}
	if !cfg.Normalize || cfg.Pooling != "mean" {
		t.Fatalf("shared fields wrong: %+v", cfg)
	}

	// A roberta repo claiming lowercasing is inconsistent with its own
	// tokenizer class — refuse rather than silently ignore.
	write("tokenizer_config.json", `{"do_lower_case":true,"cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	if _, err := DeriveConfig(dir, "test-roberta"); err == nil {
		t.Fatal("expected refusal for do_lower_case=true on roberta")
	}

	// Missing pad_token_id must refuse: the position offset depends on it.
	write("tokenizer_config.json", `{"cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	write("config.json", `{"model_type":"roberta","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":6,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":50265,"max_position_embeddings":514,"layer_norm_eps":1e-5}`)
	if _, err := DeriveConfig(dir, "test-roberta"); err == nil {
		t.Fatal("expected refusal for missing pad_token_id")
	}
}

// TestManifestMissingPadRefused is the review's HIGH: PadTokenID is a
// plain int, so a roberta manifest WITHOUT the field decodes as 0 — and
// before the guard that loaded silently with every position embedding
// shifted one row (cosine 0.66 vs correct). The manifest path must
// refuse it exactly like the config.json path does.
func TestManifestMissingPadRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	manifest := `{"name":"m","model_type":"roberta","hidden_size":768,
		"num_hidden_layers":6,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":50265,"max_position_embeddings":514,"layer_norm_eps":1e-5,
		"pooling":"mean","normalize":true}`
	if err := os.WriteFile(path, []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Fatal("roberta manifest without pad_token_id was accepted; it must refuse")
	}
}
