// SPDX-License-Identifier: Apache-2.0

package rembed

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"runtime/debug"
	"slices"
	"strings"
	"testing"
)

// modelDir returns the model directory for e2e tests, skipping when the
// weights have not been exported (models are never committed; run
// models/convert.py to produce them).
func modelDir(tb testing.TB) string {
	tb.Helper()
	dir := os.Getenv("REMBED_MODEL_DIR")
	if dir == "" {
		dir = "models/all-MiniLM-L6-v2"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		tb.Skipf("model dir %s not present (run models/convert.py); skipping e2e", dir)
	}
	return dir
}

type goldenFile struct {
	Model string `json:"model"`
	Cases []struct {
		Text      string    `json:"text"`
		InputIDs  []int64   `json:"input_ids"`
		Embedding []float32 `json:"embedding"`
	} `json:"cases"`
}

// TestGoldenAgainstONNXReference is the M0 acceptance test and the permanent
// safety net of the optimization ladder: every kernel or model change must
// keep matching the ONNX Runtime reference within 1e-4.
func TestGoldenAgainstONNXReference(t *testing.T) {
	dir := modelDir(t)
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatalf("golden file missing (run models/convert.py): %v", err)
	}
	var golden goldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("golden file has no cases — an empty reference must not pass")
	}
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if emb.Dim() != len(golden.Cases[0].Embedding) {
		t.Fatalf("Dim()=%d, golden dim=%d", emb.Dim(), len(golden.Cases[0].Embedding))
	}
	const tol = 1e-4
	for _, c := range golden.Cases {
		// Token ids first, so a failure is attributed to the tokenizer, not
		// the numerics.
		if ids := emb.Tokenize(c.Text); !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch\n got  %v\n want %v", c.Text, ids, c.InputIDs)
			continue
		}
		vecs, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		var maxDiff float64
		for i, want := range c.Embedding {
			if d := math.Abs(float64(vecs[0][i] - want)); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > tol {
			t.Errorf("%.40q: maxAbsDiff=%g > %g", c.Text, maxDiff, tol)
		} else {
			t.Logf("%.40q: seq=%d maxAbsDiff=%.2e", c.Text, len(c.InputIDs), maxDiff)
		}
	}
}

func TestEmbedContextCancellation(t *testing.T) {
	dir := modelDir(t)
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := emb.Embed(ctx, []string{"hello"}); err == nil {
		t.Fatal("expected context error")
	}
}

// TestEmbedSteadyStateAllocs pins the M1 alloc-free property. The real
// invariant lives at the Forward level: after warm-up, a forward pass
// allocates ONLY the returned vector, so its alloc count must be identical
// for a short and a near-max-length input (M0, which allocated every scratch
// buffer per call, fails that immediately). GC is pinned off so a
// mid-measurement pool drain can't flake the run.
func TestEmbedSteadyStateAllocs(t *testing.T) {
	dir := modelDir(t)
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	short := emb.Tokenize("The quick brown fox jumps over the lazy dog.")
	long := emb.Tokenize(strings.Repeat("embedding inference engines should be boring and fast ", 60))
	if len(long) < 400 {
		t.Fatalf("long input only tokenized to %d tokens", len(long))
	}
	// 3 runs each: with GC pinned off the count is deterministic, and the
	// seq=512 forward passes are ~2s apiece.
	forwardAllocs := func(ids []int64) float64 {
		if _, err := emb.m.Forward(ids); err != nil { // warm this size class
			t.Fatal(err)
		}
		return testing.AllocsPerRun(3, func() {
			if _, err := emb.m.Forward(ids); err != nil {
				t.Fatal(err)
			}
		})
	}
	// Warm the long size first so the short runs reuse grown buffers too.
	aLong := forwardAllocs(long)
	aShort := forwardAllocs(short)
	if aShort != aLong || aShort > 2 {
		t.Fatalf("Forward allocs must be tiny and independent of seq: short=%v long=%v (want equal, <= 2)", aShort, aLong)
	}

	// End-to-end sanity: tokenizer allocations scale with token count, so
	// only the short text gets an absolute bound (steady state measured 29).
	ctx := context.Background()
	texts := []string{"The quick brown fox jumps over the lazy dog."}
	embedAllocs := testing.AllocsPerRun(20, func() {
		if _, err := emb.Embed(ctx, texts); err != nil {
			t.Fatal(err)
		}
	})
	if embedAllocs > 32 {
		t.Fatalf("steady-state Embed does %v allocs/op, want <= 32", embedAllocs)
	}
}

// BenchmarkEmbed is the end-to-end latency benchmark for the ladder deltas.
func BenchmarkEmbed(b *testing.B) {
	dir := modelDir(b)
	emb, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	texts := []string{"The quick brown fox jumps over the lazy dog."}
	for b.Loop() {
		if _, err := emb.Embed(ctx, texts); err != nil {
			b.Fatal(err)
		}
	}
}
