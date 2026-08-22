// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"testing"
)

// writeHFDir lays down a minimal Hugging Face sentence-transformers
// checkout for derivation tests.
func writeHFDir(t *testing.T, pooling, modules string) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"config.json": `{"model_type":"bert","hidden_size":384,"num_hidden_layers":6,
			"num_attention_heads":12,"intermediate_size":1536,"vocab_size":30522,
			"max_position_embeddings":512,"layer_norm_eps":1e-12,"hidden_act":"gelu"}`,
		"tokenizer_config.json": `{"do_lower_case":true,"cls_token":"[CLS]","sep_token":"[SEP]","unk_token":"[UNK]"}`,
		"1_Pooling/config.json": pooling,
	}
	if modules != "" {
		files["modules.json"] = modules
	}
	for rel, content := range files {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestDeriveConfigFromHFFiles(t *testing.T) {
	stModules := `[{"type":"sentence_transformers.models.Transformer"},
		{"type":"sentence_transformers.models.Pooling"},
		{"type":"sentence_transformers.models.Normalize"}]`

	dir := writeHFDir(t, `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`, stModules)
	c, err := DeriveConfig(dir, "test/mean")
	if err != nil {
		t.Fatal(err)
	}
	if c.Pooling != "mean" || !c.Normalize || c.HiddenSize != 384 || !c.DoLowerCase || c.LayerNormEps != 1e-12 {
		t.Fatalf("derived %+v", c)
	}

	// CLS pooling (BGE-style), no Normalize module.
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":false,"pooling_mode_cls_token":true}`,
		`[{"type":"sentence_transformers.models.Transformer"},{"type":"sentence_transformers.models.Pooling"}]`)
	c, err = DeriveConfig(dir, "test/cls")
	if err != nil {
		t.Fatal(err)
	}
	if c.Pooling != "cls" || c.Normalize {
		t.Fatalf("derived %+v", c)
	}

	// Refusals: missing modules.json (silent normalize=false would shift
	// every downstream cosine threshold), unsupported/unknown pooling,
	// alien module, wrong arch, unset do_lower_case.
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":true}`, "")
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for missing modules.json")
	}
	stPlain := `[{"type":"sentence_transformers.models.Transformer"},{"type":"sentence_transformers.models.Pooling"}]`
	dir = writeHFDir(t, `{"pooling_mode_max_tokens":true}`, stPlain)
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for max pooling")
	}
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":true,"pooling_mode_shiny_new_mode":true}`, stPlain)
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for unknown pooling mode (allowlist)")
	}
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":true}`, stPlain)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"cls_token":"[CLS]"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for unset do_lower_case")
	}
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":true}`,
		`[{"type":"sentence_transformers.models.Dense"}]`)
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for Dense module")
	}
	dir = writeHFDir(t, `{"pooling_mode_mean_tokens":true}`, "")
	if err := os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(`{"model_type":"roberta"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = DeriveConfig(dir, "x"); err == nil {
		t.Fatal("expected error for non-BERT model_type")
	}
}
