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

// TestDeriveConfigSentencePiece: a BERT-architecture repo with the XLM-R
// tokenizer (multilingual MiniLM / multilingual-e5 shape) derives
// tokenizer=sentencepiece, is exempt from do_lower_case, and refuses a
// config claiming lowercasing.
func TestDeriveConfigSentencePiece(t *testing.T) {
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
	write("config.json", `{"model_type":"bert","hidden_act":"gelu","hidden_size":384,
		"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":1536,
		"vocab_size":250037,"max_position_embeddings":512,"layer_norm_eps":1e-12}`)
	write("tokenizer_config.json", `{"tokenizer_class":"XLMRobertaTokenizer","cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	write("1_Pooling/config.json", `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`)
	write("modules.json", `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"}]`)

	cfg, err := DeriveConfig(dir, "test-xlmr")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Tokenizer != "sentencepiece" || cfg.DoLowerCase || cfg.ModelType != "bert" {
		t.Fatalf("sentencepiece derivation wrong: %+v", cfg)
	}
	if cfg.PositionOffset() != 0 || cfg.MaxSeqLen() != 512 {
		t.Fatalf("bert-architecture offsets wrong: %+v", cfg)
	}

	// The real multilingual MiniLM repo carries a stale do_lower_case=true
	// that HF's XLM-R tokenizer never reads — it must be IGNORED, and the
	// derivation must also work with no tokenizer_class field when the
	// sentencepiece.bpe.model file is present (older exports lack the
	// field).
	write("tokenizer_config.json", `{"do_lower_case":true,"cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	write("sentencepiece.bpe.model", "not parsed by derive")
	cfg, err = DeriveConfig(dir, "test-xlmr")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Tokenizer != "sentencepiece" || cfg.DoLowerCase {
		t.Fatalf("file-presence detection or stale do_lower_case handling wrong: %+v", cfg)
	}
}

// TestDeriveConfigXLMRoberta: a genuine model_type=xlm-roberta repo
// (multilingual-e5-base/large, bge-m3 shape) folds onto the RoBERTa
// architecture — SentencePiece tokenizer, fairseq pad offset — so the whole
// RoBERTa forward path applies with no xlm-roberta-specific code downstream.
func TestDeriveConfigXLMRoberta(t *testing.T) {
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
	write("config.json", `{"model_type":"xlm-roberta","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":250002,"max_position_embeddings":514,"layer_norm_eps":1e-5,
		"pad_token_id":1,"position_embedding_type":"absolute"}`)
	// Real XLM-R repos carry a stale do_lower_case that HF's XLM-R tokenizer
	// never reads; it must be ignored, exactly as for the BERT-arch case.
	write("tokenizer_config.json", `{"do_lower_case":true,"tokenizer_class":"XLMRobertaTokenizer","cls_token":"<s>","sep_token":"</s>","unk_token":"<unk>"}`)
	write("1_Pooling/config.json", `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`)
	write("modules.json", `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"}]`)

	cfg, err := DeriveConfig(dir, "test-xlmr-arch")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "roberta" {
		t.Fatalf("xlm-roberta must fold onto roberta, got %q", cfg.ModelType)
	}
	if cfg.Tokenizer != "sentencepiece" || cfg.DoLowerCase {
		t.Fatalf("xlm-roberta tokenizer/lowercasing wrong: %+v", cfg)
	}
	// pad_token_id=1 → positions offset by 2 (pad+1), and two rows are
	// consumed from the 514-row table, matching HF's XLMRobertaModel.
	if cfg.PadTokenID != 1 || cfg.PositionOffset() != 2 || cfg.MaxSeqLen() != 512 {
		t.Fatalf("xlm-roberta fairseq offsets wrong: %+v", cfg)
	}

	// A config missing pad_token_id must refuse rather than shift positions.
	write("config.json", `{"model_type":"xlm-roberta","hidden_act":"gelu","hidden_size":768,
		"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":3072,
		"vocab_size":250002,"max_position_embeddings":514,"layer_norm_eps":1e-5}`)
	if _, err := DeriveConfig(dir, "test-xlmr-arch"); err == nil {
		t.Fatal("xlm-roberta without pad_token_id was accepted; it must refuse")
	}
}

// TestDeriveConfigDistilBERT: DistilBERT spells its architecture with
// different config keys (dim/n_layers/n_heads/hidden_dim/activation) —
// the derivation folds them into the BERT names, and refuses the
// sinusoidal-position variant the engine cannot represent.
func TestDeriveConfigDistilBERT(t *testing.T) {
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
	write("config.json", `{"model_type":"distilbert","activation":"gelu","dim":768,
		"n_layers":6,"n_heads":12,"hidden_dim":3072,"vocab_size":30522,
		"max_position_embeddings":512,"sinusoidal_pos_embds":false}`)
	write("tokenizer_config.json", `{"do_lower_case":true,"cls_token":"[CLS]","sep_token":"[SEP]","unk_token":"[UNK]"}`)
	write("1_Pooling/config.json", `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`)
	write("modules.json", `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"},
		{"type":"sentence_transformers.models.Normalize"}]`)

	cfg, err := DeriveConfig(dir, "test-distil")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "distilbert" || cfg.HiddenSize != 768 || cfg.NumHiddenLayers != 6 ||
		cfg.NumAttentionHeads != 12 || cfg.IntermediateSize != 3072 {
		t.Fatalf("distilbert key folding wrong: %+v", cfg)
	}
	if cfg.PositionOffset() != 0 || cfg.MaxSeqLen() != 512 || cfg.LayerNormEps != 1e-12 {
		t.Fatalf("distilbert defaults wrong: %+v", cfg)
	}

	write("config.json", `{"model_type":"distilbert","activation":"gelu","dim":768,
		"n_layers":6,"n_heads":12,"hidden_dim":3072,"vocab_size":30522,
		"max_position_embeddings":512,"sinusoidal_pos_embds":true}`)
	if _, err := DeriveConfig(dir, "test-distil"); err == nil {
		t.Fatal("expected refusal for sinusoidal_pos_embds=true")
	}
}
