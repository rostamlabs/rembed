// SPDX-License-Identifier: Apache-2.0

package rembed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
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

// TestGoldenInt8 pins the int8 accuracy contract against the same fp32
// ONNX reference: weight-only per-channel quantization measured worst
// maxAbsDiff 0.0126 and worst cosine 0.99906 across the golden set; the
// bounds below leave ~2× headroom without letting real degradation hide.
func TestGoldenInt8(t *testing.T) {
	dir := modelDir(t)
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden goldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("golden file has no cases")
	}
	emb, err := Load(dir, WithInt8())
	if err != nil {
		t.Fatal(err)
	}
	if !emb.Quantized() {
		t.Skip("int8 path unavailable on this CPU; nothing to assert")
	}
	if emb.Dim() != len(golden.Cases[0].Embedding) {
		t.Fatalf("Dim()=%d, golden dim=%d", emb.Dim(), len(golden.Cases[0].Embedding))
	}
	var worst float64
	for _, c := range golden.Cases {
		if ids := emb.Tokenize(c.Text); !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch", c.Text)
			continue
		}
		v, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		// dot IS cosine here because both vectors are unit-norm (the
		// manifest sets normalize: true, and the golden is normalized).
		var dot, maxd float64
		for i, want := range c.Embedding {
			dot += float64(v[0][i]) * float64(want)
			if d := math.Abs(float64(v[0][i] - want)); d > maxd {
				maxd = d
			}
		}
		if maxd > worst {
			worst = maxd
		}
		if maxd > 0.03 || dot < 0.998 {
			t.Errorf("%.40q: int8 maxAbsDiff=%.4g cosine=%.6f (want <= 0.03, >= 0.998)", c.Text, maxd, dot)
		} else {
			t.Logf("%.40q: int8 maxAbsDiff=%.4g cosine=%.6f", c.Text, maxd, dot)
		}
	}
	// Quantization error must be PRESENT: a run indistinguishable from fp32
	// means the int8 path silently isn't executing, and this test would
	// otherwise pass most easily with the feature under test disabled.
	if worst < 1e-4 {
		t.Fatalf("int8 error (%.3g) indistinguishable from fp32 — int8 path appears inactive", worst)
	}
}

// TestWithWorkersSerialMatchesDefault pins WithWorkers(1): fully serial,
// same embeddings as the default full-machine configuration.
func TestWithWorkersSerialMatchesDefault(t *testing.T) {
	dir := modelDir(t)
	def, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ser, err := Load(dir, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	for _, text := range []string{"hello world", "The quick brown fox jumps over the lazy dog."} {
		a, err := def.Embed(ctx, []string{text})
		if err != nil {
			t.Fatal(err)
		}
		b, err := ser.Embed(ctx, []string{text})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(a[0], b[0]) {
			t.Fatalf("%q: WithWorkers(1) diverged from default", text)
		}
	}
}

// TestEmbedConcurrentMatchesSerial pins two properties of the M2 parallel
// forward pass at once: determinism (parallel execution must be bit-identical
// to a serial reference — workers own disjoint outputs and never change
// accumulation order) and scratch-pool isolation under concurrent Embed
// calls. Run with -race for the full effect.
func TestEmbedConcurrentMatchesSerial(t *testing.T) {
	dir := modelDir(t)
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	texts := []string{
		"hello world",
		"The quick brown fox jumps over the lazy dog.",
		"Rostam is a vector database written in Go, with BM25 and semantic caching built in.",
		strings.Repeat("a longer text to hit a different scratch size class ", 20),
	}
	// The reference is computed with GOMAXPROCS=1, which drives every
	// ParallelFor (matmul tiles, heads, GELU rows) down its inline serial
	// path — so this really is parallel-vs-serial bit-equality, not just
	// reproducibility of the parallel path against itself.
	runtime.GOMAXPROCS(1)
	want := make([][]float32, len(texts))
	for i, text := range texts {
		v, err := emb.Embed(ctx, []string{text})
		if err != nil {
			runtime.GOMAXPROCS(runtime.NumCPU())
			t.Fatal(err)
		}
		want[i] = v[0]
	}
	runtime.GOMAXPROCS(runtime.NumCPU())
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for w := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rep := range 4 {
				i := (w + rep) % len(texts)
				v, err := emb.Embed(ctx, []string{texts[i]})
				if err != nil {
					errs <- err
					return
				}
				if !slices.Equal(v[0], want[i]) {
					errs <- fmt.Errorf("concurrent embed of text %d diverged from serial result", i)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
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

	// Three size classes: a tiny query (seq≈3 — the class the original
	// total-work parallelization gate left serial, which made alloc counts
	// regime-dependent), a typical sentence, and a near-max input.
	tiny := emb.Tokenize("hi")
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
	// Warm the long size first so the smaller runs reuse grown buffers too.
	// The absolute bound covers the M2 fan-out's fixed goroutine/closure
	// allocations; equality across ALL size classes is the invariant that
	// catches both a regression to per-call buffer allocation and a
	// parallelization gate that silently changes regime with seq.
	aLong := forwardAllocs(long)
	aShort := forwardAllocs(short)
	aTiny := forwardAllocs(tiny)
	if aTiny != aShort || aShort != aLong || aShort > 64 {
		t.Fatalf("Forward allocs must be small and independent of seq: tiny=%v short=%v long=%v (want all equal, <= 64)", aTiny, aShort, aLong)
	}

	// End-to-end sanity: tokenizer allocations scale with token count, so
	// only the short text gets an absolute bound (Forward's fan-out plus
	// ~28 tokenizer/output allocations).
	ctx := context.Background()
	texts := []string{"The quick brown fox jumps over the lazy dog."}
	embedAllocs := testing.AllocsPerRun(20, func() {
		if _, err := emb.Embed(ctx, texts); err != nil {
			t.Fatal(err)
		}
	})
	if embedAllocs > 96 {
		t.Fatalf("steady-state Embed does %v allocs/op, want <= 96", embedAllocs)
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
