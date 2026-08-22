// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/rostamlabs/rembed"
)

func serveTestEmbedder(t *testing.T) *rembed.Embedder {
	t.Helper()
	dir := "../../models/all-MiniLM-L6-v2"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present; skipping serve e2e", dir)
	}
	emb, err := rembed.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return emb
}

func post(t *testing.T, h http.HandlerFunc, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("non-JSON response %q: %v", rec.Body.String(), err)
	}
	return rec, out
}

// TestServeEmbeddings pins the OpenAI response contract: object/list
// envelope, per-item index and vector, usage token counts — and that the
// vectors are the engine's own (unit-norm, correct dim).
func TestServeEmbeddings(t *testing.T) {
	emb := serveTestEmbedder(t)
	h := func(w http.ResponseWriter, r *http.Request) { handleEmbeddings(w, r, emb, 8) }

	rec, out := post(t, h, `{"input": ["hello world", "second text"], "model": "whatever"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %v", rec.Code, out)
	}
	if out["object"] != "list" || out["model"] != emb.Model() {
		t.Fatalf("envelope: %v", out)
	}
	data := out["data"].([]any)
	if len(data) != 2 {
		t.Fatalf("want 2 embeddings, got %d", len(data))
	}
	first := data[0].(map[string]any)
	if first["object"] != "embedding" || first["index"] != float64(0) {
		t.Fatalf("item envelope: %v", first)
	}
	vec := first["embedding"].([]any)
	if len(vec) != emb.Dim() {
		t.Fatalf("dim %d want %d", len(vec), emb.Dim())
	}
	var norm float64
	for _, x := range vec {
		norm += x.(float64) * x.(float64)
	}
	if math.Abs(norm-1) > 1e-4 {
		t.Fatalf("vector not unit-norm: %v", norm)
	}
	usage := out["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) < 4 {
		t.Fatalf("usage: %v", usage)
	}

	// Single-string input is one embedding.
	rec, out = post(t, h, `{"input": "hello world"}`)
	if rec.Code != http.StatusOK || len(out["data"].([]any)) != 1 {
		t.Fatalf("single input: %d %v", rec.Code, out)
	}

	// Error paths use the OpenAI error envelope with actionable messages.
	for body, wantSub := range map[string]string{
		`{"input": []}`:        "empty",
		`{"input": [[1,2,3]]}`: "array of strings",
		`{"input": 42}`:        "array of strings",
		`{}`:                   "missing input",
		`not json`:             "invalid JSON",
		`{"input": ["a","b","c","d","e","f","g","h","i"]}`: "max-batch",
	} {
		rec, out := post(t, h, body)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: status %d", body, rec.Code)
		}
		msg := out["error"].(map[string]any)["message"].(string)
		if !strings.Contains(msg, wantSub) {
			t.Fatalf("%s: error %q missing %q", body, msg, wantSub)
		}
	}
}
