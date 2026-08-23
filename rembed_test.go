// SPDX-License-Identifier: Apache-2.0

package rembed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
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

// TestGoldenInt8Activations pins the FULL int8 mode (VNNI: u8 activations
// × s8 weights): accuracy against the ONNX golden at the measured bound
// (worst case 0.991652 on this model; the bound leaves margin), an error
// large enough that a silently-disabled feature cannot pass (the R4-era
// inverted-test lesson), and bit-identity between serial and parallel —
// per-row quantization and disjoint tiles make worker count invisible.
func TestGoldenInt8Activations(t *testing.T) {
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
	emb, err := Load(dir, WithInt8Activations())
	if err != nil {
		t.Fatal(err)
	}
	if !emb.QuantizedActivations() {
		t.Skip("no VNNI encoding available on this CPU; nothing to assert")
	}
	t.Logf("full-int8 path ACTIVE (VEX or EVEX per CPU) — accuracy bound enforced")
	serial, err := Load(dir, WithInt8Activations(), WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	var worstCos = 1.0
	var maxErr float64
	for _, c := range golden.Cases {
		v, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		sv, err := serial.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		for i := range v[0] {
			if v[0][i] != sv[0][i] {
				t.Fatalf("%.40q: parallel[%d]=%v differs from serial %v", c.Text, i, v[0][i], sv[0][i])
			}
		}
		var dot float64
		for i, want := range c.Embedding {
			dot += float64(v[0][i]) * float64(want)
			if d := math.Abs(float64(v[0][i] - want)); d > maxErr {
				maxErr = d
			}
		}
		if dot < worstCos {
			worstCos = dot
		}
		if dot < 0.991 {
			t.Errorf("%.40q: cosine %.6f below the full-int8 bound 0.991", c.Text, dot)
		}
	}
	// A full-int8 run that matches fp32 too well means the mode is not
	// actually engaged.
	if maxErr < 1e-4 {
		t.Fatalf("full-int8 output within %.2g of the fp32 golden — is the VNNI path actually active?", maxErr)
	}
	t.Logf("full-int8 worst cosine %.6f, maxAbs %.4g", worstCos, maxErr)
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

// TestGoldenCLSPooling validates the CLS pooling path end to end against
// BGE-small's own ONNX reference — the pooling mode the BGE/arctic model
// families use. Skips when the model dir has not been exported.
func TestGoldenCLSPooling(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_BGE")
	if dir == "" {
		dir = "models/bge-small-en-v1.5"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present (models/convert.py BAAI/bge-small-en-v1.5, or set REMBED_MODEL_BGE)", dir)
	}
	raw, err := os.ReadFile("testdata/golden-bge-small.json")
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
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := emb.Dim(); got != len(golden.Cases[0].Embedding) {
		t.Fatalf("Dim()=%d, golden dim=%d", got, len(golden.Cases[0].Embedding))
	}
	for _, c := range golden.Cases {
		if ids := emb.Tokenize(c.Text); !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch", c.Text)
			continue
		}
		v, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		var maxd float64
		for i, want := range c.Embedding {
			if d := math.Abs(float64(v[0][i] - want)); d > maxd {
				maxd = d
			}
		}
		if maxd > 1e-4 {
			t.Errorf("%.40q: cls maxAbsDiff=%g > 1e-4", c.Text, maxd)
		}
	}
}

// TestGoldenMPNetParallelMatchesSerial pins the CONCURRENT form of the
// MPNet-only code: the per-head workers reading the shared biasDelta table
// from inside Pool.Run goroutines. TestGoldenMatrix runs serially (it is a
// numerics test), so without this the relative-position bias path would
// have no coverage in its parallel form — and the input is long enough
// (≥300 tokens) that the far buckets are exercised concurrently too.
// Parallel must be BIT-identical to serial: partitioning never changes any
// output element's accumulation order.
func TestGoldenMPNetParallelMatchesSerial(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_MPNET")
	if dir == "" {
		dir = "models/all-mpnet-base-v2"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	serial, err := Load(dir, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the antikythera mechanism modeled the motions of the sun and moon with startling precision, ", 25)
	if n := len(serial.Tokenize(text)); n < 300 {
		t.Fatalf("test input tokenizes to %d tokens; need >=300 to reach the far relative-position buckets", n)
	}
	a, err := serial.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	b, err := parallel.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("parallel diverges from serial at [%d]: %v vs %v", i, b[0][i], a[0][i])
		}
	}
}

// TestGoldenModernBERTParallelMatchesSerial pins that ModernBERT's
// parallel head fan-out (RoPE, the sliding-window mask, GeGLU all run
// inside Pool.Run goroutines) is BIT-identical to the serial path, and
// gives the race detector something to chew on for the ModernBERT code —
// TestGoldenMatrix runs at WithWorkers(1). The input is long enough
// (>=300 tokens) that the global/local attention split and the ±64 window
// mask are exercised concurrently.
func TestGoldenModernBERTParallelMatchesSerial(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_MODERNBERT")
	if dir == "" {
		dir = "models/modernbert-embed-base"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	serial, err := Load(dir, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the antikythera mechanism modeled the motions of the sun and moon with startling precision, ", 25)
	if n := len(serial.Tokenize(text)); n < 300 {
		t.Fatalf("test input tokenizes to %d tokens; need >=300 to exercise the sliding-window mask", n)
	}
	a, err := serial.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	b, err := parallel.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("parallel diverges from serial at [%d]: %v vs %v", i, b[0][i], a[0][i])
		}
	}
}

// TestGoldenTokens validates EmbedTokens — per-token hidden states —
// against ONNX Runtime's last_hidden_state for the committed token-level
// golden. This is the raw, unpooled output, so nothing downstream
// (pooling, normalization) can mask a defect here.
func TestGoldenTokens(t *testing.T) {
	dir := modelDir(t)
	raw, err := os.ReadFile("testdata/golden-tokens.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Model string `json:"model"`
		Cases []struct {
			Text     string      `json:"text"`
			InputIDs []int64     `json:"input_ids"`
			Hidden   [][]float32 `json:"hidden"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("empty token golden")
	}
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		res, err := emb.EmbedTokens(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		te := res[0]
		if !slices.Equal(te.IDs, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch", c.Text)
			continue
		}
		if len(te.Vectors) != len(c.Hidden) {
			t.Fatalf("%.40q: %d token vectors, want %d", c.Text, len(te.Vectors), len(c.Hidden))
		}
		var maxd float64
		for i, wantRow := range c.Hidden {
			for j, want := range wantRow {
				if d := math.Abs(float64(te.Vectors[i][j] - want)); d > maxd {
					maxd = d
				}
			}
		}
		// Raw hidden states have magnitudes up to ~10 (unnormalized), so
		// the tolerance is the golden rule's 1e-4 relative to that scale.
		if maxd > 1e-4 {
			t.Errorf("%.40q: token maxAbsDiff=%g > 1e-4 (observed baseline 2.9e-6 — see PR #13 review)", c.Text, maxd)
		} else {
			t.Logf("%.40q: %d tokens, maxAbsDiff=%.2e", c.Text, len(te.Vectors), maxd)
		}
	}
}

// benchText is ~120 tokens — long enough that attention (O(seq²)) and the
// per-element ops (softmax/gelu/layernorm/rope) register against the matmuls.
const benchText = "The history of mechanical computation stretches back further than most people assume, " +
	"beginning with devices like the antikythera mechanism, a bronze assembly of interlocking gears " +
	"recovered from a shipwreck in the aegean sea that modeled the motions of the sun and moon with " +
	"startling precision across many decades of careful astronomical observation and calculation."

// BenchmarkEmbedProfile measures a full forward pass on a ~120-token input,
// SERIAL (WithWorkers(1)) so a CPU profile shows where compute actually goes
// rather than pool-spin. Select the model with REMBED_BENCH_MODEL (default:
// mpnet); this is the profiling counterpart to the parallel BenchmarkEmbed.
//
//	go test -run=^$ -bench=BenchmarkEmbedProfile -benchtime=3s -cpuprofile=/tmp/cpu.prof .
//	go tool pprof -top -nodecount=30 /tmp/cpu.prof
func BenchmarkEmbedProfile(b *testing.B) {
	dir := os.Getenv("REMBED_BENCH_MODEL")
	if dir == "" {
		dir = "models/all-mpnet-base-v2"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		b.Skipf("model dir %s not present", dir)
	}
	emb, err := Load(dir, WithWorkers(1))
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	if _, err := emb.Embed(ctx, []string{benchText}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := emb.Embed(ctx, []string{benchText}); err != nil {
			b.Fatal(err)
		}
	}
}

// TestQwen3_4BDiskSanity exercises the R11 "run larger than RAM" path on a
// real large model: Qwen3-Embedding-4B (sharded, ~15GB in fp32) loaded
// WithDiskWeights. It has no torch golden here (the reference needs ~16GB of
// RAM this class of box lacks), so it asserts the properties a correct
// embedder must have — right dimensionality, unit-normalized output, a
// paraphrase scoring closer than an unrelated sentence, and deterministic
// re-embedding — which together would catch a broken GQA/shard/mmap load.
// Opt-in: set REMBED_MODEL_QWEN3_4B to a 4B model dir (or hub cache dir);
// skips otherwise, so it never runs in CI.
func TestQwen3_4BDiskSanity(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_QWEN3_4B")
	if dir == "" {
		t.Skip("set REMBED_MODEL_QWEN3_4B to a Qwen3-Embedding-4B dir to run")
	}
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	emb, err := Load(dir, WithDiskWeights(), WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	if emb.Dim() != 2560 {
		t.Fatalf("dim %d, want 2560", emb.Dim())
	}
	texts := []string{
		"The cat sat on the warm windowsill in the afternoon sun.",
		"A feline rested by the sunny window during the day.",
		"Quarterly revenue grew twelve percent driven by cloud services.",
	}
	v, err := emb.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, vec := range v {
		var n float64
		for _, x := range vec {
			n += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(n)-1) > 1e-3 {
			t.Errorf("vec %d not unit-normalized: L2=%.6f", i, math.Sqrt(n))
		}
	}
	cosf := func(a, b []float32) float64 {
		var d, na, nb float64
		for i := range a {
			d += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		return d / math.Sqrt(na*nb)
	}
	if para, unrel := cosf(v[0], v[1]), cosf(v[0], v[2]); para <= unrel {
		t.Errorf("paraphrase cos %.4f not > unrelated cos %.4f", para, unrel)
	}
	v2, err := emb.Embed(context.Background(), texts[:1])
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(v[0], v2[0]) {
		t.Error("re-embed not deterministic")
	}
}

// TestGoldenQwen3DiskWeights validates the memory-mapped disk-weights path
// (WithDiskWeights) against the same torch golden: the weights are packed to
// disk on first use, mmapped, and run through the unpacked matmul. This is
// the R11 "run larger than RAM" path — here proven correct on the 0.6B, where
// it also fits in RAM. Tolerance is the golden rule's 1e-4 (the unpacked
// matmul's accumulation order differs slightly from the packed path, but both
// match the reference).
func TestGoldenQwen3DiskWeights(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_QWEN3")
	if dir == "" {
		dir = "models/Qwen3-Embedding-0.6B"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	raw, err := os.ReadFile("testdata/golden-Qwen3-Embedding-0.6B.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden goldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	emb, err := Load(dir, WithDiskWeights(), WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	for _, c := range golden.Cases {
		v, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		var maxd, dot, na, nb float64
		for i, want := range c.Embedding {
			g, w := float64(v[0][i]), float64(want)
			if d := math.Abs(g - w); d > maxd {
				maxd = d
			}
			dot += g * w
			na += g * g
			nb += w * w
		}
		cos := dot / math.Sqrt(na*nb)
		if maxd > 1e-4 || cos < 0.9999 {
			t.Errorf("%.40q: disk maxAbs=%.3g cos=%.6f (want <=1e-4, >=0.9999)", c.Text, maxd, cos)
		}
	}
}

// TestGoldenQwen3ParallelMatchesSerial pins that Qwen3's parallel head
// fan-out (GQA repack, QK-norm, RoPE, causal mask all run inside Pool.Run
// goroutines) is BIT-identical to serial, and gives the race detector the
// Qwen3 code to exercise (the golden matrix runs WithWorkers(1)).
func TestGoldenQwen3ParallelMatchesSerial(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_QWEN3")
	if dir == "" {
		dir = "models/Qwen3-Embedding-0.6B"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	serial, err := Load(dir, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the antikythera mechanism modeled the motions of the sun and moon, ", 20)
	a, err := serial.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	b, err := parallel.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatal(err)
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("parallel diverges from serial at [%d]: %v vs %v", i, b[0][i], a[0][i])
		}
	}
}

// TestGoldenQwen3Tokens validates Qwen3's RAW per-token hidden states
// (EmbedTokens) against the torch reference — the unpooled output, so a
// per-token defect that last-token pooling would only expose at the final
// position is caught at every position. Tolerance is 5e-4 on the ~O(10)
// unnormalized magnitudes (28-layer fp32 accumulation, as for ModernBERT).
func TestGoldenQwen3Tokens(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_QWEN3")
	if dir == "" {
		dir = "models/Qwen3-Embedding-0.6B"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	raw, err := os.ReadFile("testdata/golden-tokens-qwen3.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Text     string      `json:"text"`
			InputIDs []int64     `json:"input_ids"`
			Hidden   [][]float32 `json:"hidden"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("empty token golden")
	}
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		res, err := emb.EmbedTokens(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		te := res[0]
		if !slices.Equal(te.IDs, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch", c.Text)
			continue
		}
		if len(te.Vectors) != len(c.Hidden) {
			t.Fatalf("%.40q: %d token vectors, want %d", c.Text, len(te.Vectors), len(c.Hidden))
		}
		var maxd float64
		for i, wantRow := range c.Hidden {
			for j, want := range wantRow {
				if d := math.Abs(float64(te.Vectors[i][j] - want)); d > maxd {
					maxd = d
				}
			}
		}
		if maxd > 5e-4 {
			t.Errorf("%.40q: token maxAbsDiff=%g > 5e-4", c.Text, maxd)
		} else {
			t.Logf("%.40q: %d tokens, maxAbsDiff=%.2e", c.Text, len(te.Vectors), maxd)
		}
	}
}

// TestGoldenModernBERTTokens validates ModernBERT's RAW per-token hidden
// states (EmbedTokens) against the torch reference — the unpooled output,
// so nothing downstream (mean pooling, normalization) can average out a
// per-token defect that the pooled golden might mask. Reference is the
// canonical PyTorch ModernBertModel (torch, eager); tolerance is the
// golden rule's 1e-4 on the unnormalized ~O(10) hidden magnitudes.
func TestGoldenModernBERTTokens(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_MODERNBERT")
	if dir == "" {
		dir = "models/modernbert-embed-base"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	raw, err := os.ReadFile("testdata/golden-tokens-modernbert.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Text     string      `json:"text"`
			InputIDs []int64     `json:"input_ids"`
			Hidden   [][]float32 `json:"hidden"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("empty token golden")
	}
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		res, err := emb.EmbedTokens(context.Background(), []string{c.Text})
		if err != nil {
			t.Fatal(err)
		}
		te := res[0]
		if !slices.Equal(te.IDs, c.InputIDs) {
			t.Errorf("%.40q: tokenizer mismatch", c.Text)
			continue
		}
		if len(te.Vectors) != len(c.Hidden) {
			t.Fatalf("%.40q: %d token vectors, want %d", c.Text, len(te.Vectors), len(c.Hidden))
		}
		var maxd float64
		for i, wantRow := range c.Hidden {
			for j, want := range wantRow {
				if d := math.Abs(float64(te.Vectors[i][j] - want)); d > maxd {
					maxd = d
				}
			}
		}
		// Tolerance is 5e-4 on the RAW unnormalized hidden states, looser
		// than the pooled golden's 1e-4 for a principled reason: these
		// magnitudes reach ~10, and over 22 layers the approximate GELU/exp
		// kernels (each ~3e-7) accumulate to ~1.3e-4 absolute here — the SAME
		// ~1.3e-5 RELATIVE error the pooled golden shows as 5.4e-7 on its
		// tiny normalized components. A real defect (wrong RoPE, window, or
		// GeGLU order) would be orders of magnitude larger, so this still
		// catches bugs while not flagging honest fp32 accumulation.
		if maxd > 5e-4 {
			t.Errorf("%.40q: token maxAbsDiff=%g > 5e-4", c.Text, maxd)
		} else {
			t.Logf("%.40q: %d tokens, maxAbsDiff=%.2e", c.Text, len(te.Vectors), maxd)
		}
	}
}

// TestGoldenMatrix makes every committed golden reproducible: each model
// in the README matrix validates against its own ONNX-reference golden
// when its model dir is present (env-overridable for CI). gte's bounds
// follow the review's guidance for an F16 checkpoint validated against a
// FP32 ONNX export: the maxAbs gap is dominated by the checkpoint's own
// f16 rounding, so a loose maxAbs alone could hide a systematic defect —
// cosine and mean-abs bounds close that hole.
func TestGoldenMatrix(t *testing.T) {
	cases := []struct {
		golden, dir, env        string
		maxAbs, minCos, maxMean float64
		// int8MinCos > 0 also validates WithInt8 against the same golden
		// (cosine only — int8 trades absolute error for speed by design),
		// so every documented int8 figure is test-enforced. int8ActMinCos
		// does the same for the FULL int8 mode (WithInt8Activations) —
		// bounds are per-model measured worst cases less margin, and they
		// vary widely: RoBERTa-family activation outliers make full int8
		// markedly worse there (see the README table).
		int8MinCos    float64
		int8ActMinCos float64
	}{
		{"testdata/golden-all-MiniLM-L12-v2.json", "models/all-MiniLM-L12-v2", "REMBED_MODEL_L12", 1e-4, 0, 1e-5, 0, 0.992},
		{"testdata/golden-paraphrase-MiniLM-L3-v2.json", "models/paraphrase-MiniLM-L3-v2", "REMBED_MODEL_L3", 1e-4, 0, 1e-5, 0, 0.997},
		{"testdata/golden-gte-small.json", "models/gte-small", "REMBED_MODEL_GTE", 2e-3, 0.9999, 2e-4, 0, 0.9985},
		// MPNet exercises the second architecture end to end: offset
		// positions, no segment table, shared relative-position bias. The
		// 12-case golden's 324-token case reaches |j−i| ≥ 128, pinning the
		// log-spaced far buckets and the max-distance clamp. int8 bound is
		// the measured worst case (0.997874, the punctuation text) less
		// margin — the review caught the docs claiming an optimistic
		// 0.9989 measured on 2 texts.
		{"testdata/golden-all-mpnet-base-v2.json", "models/all-mpnet-base-v2", "REMBED_MODEL_MPNET", 1e-4, 0, 1e-5, 0.9978, 0.9905},
		// RoBERTa exercises the byte-level BPE tokenizer end to end (the
		// tokenizer-mismatch check above bites hardest here) plus the
		// fairseq position offset on the plain BERT encoder path. int8
		// bound is the measured worst case (0.998727, the 326-token text)
		// less margin.
		{"testdata/golden-all-distilroberta-v1.json", "models/all-distilroberta-v1", "REMBED_MODEL_ROBERTA", 1e-4, 0, 1e-5, 0.9985, 0.972},
		// The multilingual model exercises the SentencePiece tokenizer end
		// to end (normalizer + Viterbi + fairseq ids) on a BERT encoder
		// with a padded embedding table (250037 rows for 250002 ids).
		// int8 bound: measured worst 0.999642 (the 402-token text) less margin.
		{"testdata/golden-paraphrase-multilingual-MiniLM-L12-v2.json", "models/paraphrase-multilingual-MiniLM-L12-v2", "REMBED_MODEL_XLMR", 1e-4, 0, 1e-5, 0.9995, 0.9975},
		// The coverage expansion. Both int8 columns are measured worst
		// cases less margin — and the full-int8 spread is wide: bge-base,
		// a PLAIN BERT, measures 0.9593, WORSE than distilroberta's
		// cautionary 0.9747, so activation outliers are a per-checkpoint
		// property, not a RoBERTa-family one.
		{"testdata/golden-multilingual-e5-small.json", "models/multilingual-e5-small", "REMBED_MODEL_E5", 1e-4, 0, 1e-5, 0.9995, 0.998},
		// multilingual-e5-base is a genuine model_type=xlm-roberta checkpoint
		// (the RoBERTa encoder with the SentencePiece tokenizer) — it folds
		// onto the roberta path and exercises the fairseq position offset with
		// SentencePiece ids together for the first time. Reference is torch
		// XLMRobertaModel (fp32, eager); XLM-R ST repos do not reliably ship
		// ONNX. int8 bounds are measured worst cases less margin.
		// int8 worst 0.999162, full int8 worst 0.984945 (measured), less margin.
		{"testdata/golden-multilingual-e5-base.json", "models/multilingual-e5-base", "REMBED_MODEL_E5B", 1e-4, 0, 1e-5, 0.999, 0.98},
		{"testdata/golden-bge-base-en-v1.5.json", "models/bge-base-en-v1.5", "REMBED_MODEL_BGEB", 1e-4, 0, 1e-5, 0.995, 0.958},
		{"testdata/golden-gte-base.json", "models/gte-base", "REMBED_MODEL_GTEB", 1e-4, 0, 1e-5, 0.988, 0.973},
		{"testdata/golden-paraphrase-mpnet-base-v2.json", "models/paraphrase-mpnet-base-v2", "REMBED_MODEL_PMPNET", 1e-4, 0, 1e-5, 0.9945, 0.986},
		{"testdata/golden-multi-qa-MiniLM-L6-cos-v1.json", "models/multi-qa-MiniLM-L6-cos-v1", "REMBED_MODEL_MQAM", 1e-4, 0, 1e-5, 0.998, 0.994},
		{"testdata/golden-snowflake-arctic-embed-s.json", "models/snowflake-arctic-embed-s", "REMBED_MODEL_ARCTIC", 1e-4, 0, 1e-5, 0.995, 0.992},
		// DistilBERT: the fourth architecture, and the golden whose
		// Persian case caught the WordPiece ZWNJ (Cf-dropping) gap.
		{"testdata/golden-multi-qa-distilbert-cos-v1.json", "models/multi-qa-distilbert-cos-v1", "REMBED_MODEL_DISTIL", 1e-4, 0, 1e-5, 0.999, 0.985},
		// ModernBERT: the fifth architecture — RoPE (dual theta), pre-norm,
		// GeGLU, bias-free, sliding-window local attention. The golden's
		// ~400-token case runs seq well past the 128-token local window, so
		// the global/local split and the ±64 mask are exercised end to end.
		// Reference is the canonical torch ModernBertModel (fp32, eager),
		// not ONNX. int8 bounds are measured worst cases less margin:
		// weight-only holds up well (0.9984, the Persian case — note the
		// ~4e-4 slack under the 0.998 bound is deliberately tight but safe:
		// int8 is deterministic integer math, so it cannot flake across CPUs
		// or runs, and a trip means a real quant regression, not noise), but
		// FULL int8 drops to 0.966 — GeGLU's gated activations have outliers
		// the per-row u8 scale can't hold, so like the RoBERTa family, full
		// int8 is not recommended for ModernBERT.
		{"testdata/golden-modernbert-embed-base.json", "models/modernbert-embed-base", "REMBED_MODEL_MODERNBERT", 1e-4, 0, 1e-5, 0.998, 0.96},
		// Qwen3-Embedding: the sixth architecture and the first DECODER
		// embedder — causal attention, RMSNorm, per-head QK-norm, GQA
		// (16 q / 8 kv), SwiGLU, RoPE, and last-token pooling on the appended
		// <|endoftext|>. Reference is the canonical torch Qwen3Model (fp32,
		// eager). int8 bounds are measured worst cases less margin:
		// weight-only 0.9978, full int8 0.9747 (last-token pooling reads a
		// single position, so it has no averaging to hide activation
		// quantization error — full int8 is least suited here; flash-attention's
		// streaming-softmax accumulation shifted the worst full-int8 case to
		// 0.9693, so the bound is 0.96).
		{"testdata/golden-Qwen3-Embedding-0.6B.json", "models/Qwen3-Embedding-0.6B", "REMBED_MODEL_QWEN3", 1e-4, 0, 1e-5, 0.997, 0.96},
		// EmbeddingGemma: the seventh architecture — a bidirectional Gemma 3
		// backbone (unit-offset RMSNorm, per-head QK-norm, GQA, dual-theta
		// RoPE, alternating sliding/global bidirectional attention, tanh-GELU
		// GeGLU, four LayerNorms per layer) plus mean pooling and a two-layer
		// bias-free Dense head. Reference is torch Gemma3TextModel (fp32, eager)
		// with the explicit ST pooling+dense+normalize pipeline.
		// int8 worst 0.998140, full int8 worst 0.993849 (measured), less margin.
		{"testdata/golden-embeddinggemma-300m.json", "models/embeddinggemma-300m", "REMBED_MODEL_GEMMA", 1e-4, 0, 1e-5, 0.998, 0.99},
	}
	for _, tc := range cases {
		t.Run(filepath.Base(tc.dir), func(t *testing.T) {
			dir := os.Getenv(tc.env)
			if dir == "" {
				dir = tc.dir
			}
			if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
				t.Skipf("model dir %s not present", dir)
			}
			raw, err := os.ReadFile(tc.golden)
			if err != nil {
				t.Fatal(err)
			}
			var golden goldenFile
			if err := json.Unmarshal(raw, &golden); err != nil {
				t.Fatal(err)
			}
			if len(golden.Cases) == 0 {
				t.Fatal("empty golden")
			}
			// Serial on purpose: this is a numerics test, and worker count
			// does not change results (TestWithWorkersSerialMatchesDefault
			// pins that). The spinning pool is pathologically slow under
			// -race, and five models × eleven texts of it was most of the
			// race suite's budget.
			emb, err := Load(dir, WithWorkers(1))
			if err != nil {
				t.Fatal(err)
			}
			for _, c := range golden.Cases {
				if ids := emb.Tokenize(c.Text); !slices.Equal(ids, c.InputIDs) {
					t.Errorf("%.40q: tokenizer mismatch", c.Text)
					continue
				}
				v, err := emb.Embed(context.Background(), []string{c.Text})
				if err != nil {
					t.Fatal(err)
				}
				var maxd, sum, dot, na, nb float64
				for i, want := range c.Embedding {
					g, w := float64(v[0][i]), float64(want)
					d := math.Abs(g - w)
					if d > maxd {
						maxd = d
					}
					sum += d
					dot += g * w
					na += g * g
					nb += w * w
				}
				mean := sum / float64(len(c.Embedding))
				cos := dot / math.Sqrt(na*nb)
				if maxd > tc.maxAbs || cos < tc.minCos || mean > tc.maxMean {
					t.Errorf("%.40q: maxAbs=%.3g (<=%.3g) cos=%.6f (>=%.4f) meanAbs=%.3g (<=%.3g)",
						c.Text, maxd, tc.maxAbs, cos, tc.minCos, mean, tc.maxMean)
				}
			}
			checkQuant := func(name string, minCos float64, opts ...Option) {
				q, err := Load(dir, append(opts, WithWorkers(1))...)
				if err != nil {
					t.Fatal(err)
				}
				if !q.Quantized() {
					t.Skipf("%s path not active on this CPU", name)
				}
				for _, c := range golden.Cases {
					v, err := q.Embed(context.Background(), []string{c.Text})
					if err != nil {
						t.Fatal(err)
					}
					var dot, na, nb float64
					for i, want := range c.Embedding {
						g, w := float64(v[0][i]), float64(want)
						dot += g * w
						na += g * g
						nb += w * w
					}
					if cos := dot / math.Sqrt(na*nb); cos < minCos {
						t.Errorf("%s %.40q: cos=%.6f (>=%.4f)", name, c.Text, cos, minCos)
					}
				}
			}
			if tc.int8MinCos > 0 {
				checkQuant("int8", tc.int8MinCos, WithInt8())
			}
			if tc.int8ActMinCos > 0 {
				full, err := Load(dir, WithInt8Activations(), WithWorkers(1))
				if err != nil {
					t.Fatal(err)
				}
				if full.QuantizedActivations() {
					t.Logf("int8-full bound %.4f enforced (VNNI active)", tc.int8ActMinCos)
					checkQuant("int8-full", tc.int8ActMinCos, WithInt8Activations())
				} else {
					t.Logf("int8-full bound NOT enforced: no VNNI on this CPU")
				}
			}
		})
	}
}

// TestGoldenMatryoshka validates WithDim(d) truncation against the
// deterministic slice+renormalize of the full-768 EmbeddingGemma golden
// (the independent torch reference): MRL keeps the first d dimensions and
// re-L2-normalizes. Pins that rembed's truncation matches that operation
// applied to the reference — so both the truncation math and the underlying
// full vector are correct.
func TestGoldenMatryoshka(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_GEMMA")
	if dir == "" {
		dir = "models/embeddinggemma-300m"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	raw, err := os.ReadFile("testdata/golden-embeddinggemma-300m.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden goldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	for _, d := range []int{512, 256, 128} {
		emb, err := Load(dir, WithDim(d), WithWorkers(1))
		if err != nil {
			t.Fatal(err)
		}
		if emb.Dim() != d {
			t.Fatalf("Dim()=%d, want %d", emb.Dim(), d)
		}
		for _, c := range golden.Cases {
			// Reference: renormalize(full[:d]).
			want := make([]float32, d)
			var norm float64
			for i := range d {
				want[i] = c.Embedding[i]
				norm += float64(want[i]) * float64(want[i])
			}
			inv := float32(1 / math.Sqrt(norm))
			for i := range want {
				want[i] *= inv
			}
			v, err := emb.Embed(context.Background(), []string{c.Text})
			if err != nil {
				t.Fatal(err)
			}
			if len(v[0]) != d {
				t.Fatalf("d=%d: got vector dim %d", d, len(v[0]))
			}
			var maxd float64
			for i := range want {
				if dd := math.Abs(float64(v[0][i] - want[i])); dd > maxd {
					maxd = dd
				}
			}
			if maxd > 1e-4 {
				t.Errorf("d=%d %.30q: maxAbs=%g > 1e-4", d, c.Text, maxd)
			}
		}
	}
}

// TestWithDimRange pins the WithDim bounds: 0 keeps the full dim, and an
// out-of-range value is refused at Load rather than truncating silently.
func TestWithDimRange(t *testing.T) {
	dir := os.Getenv("REMBED_MODEL_GEMMA")
	if dir == "" {
		dir = "models/embeddinggemma-300m"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model dir %s not present", dir)
	}
	full, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir, WithDim(full.Dim()+1)); err == nil {
		t.Fatal("WithDim above the full dim was accepted; it must refuse")
	}
	if _, err := Load(dir, WithDim(-1)); err == nil {
		t.Fatal("WithDim(-1) was accepted; it must refuse")
	}
	if e, err := Load(dir, WithDim(0)); err != nil || e.Dim() != full.Dim() {
		t.Fatalf("WithDim(0) should keep the full dim: dim=%d err=%v", e.Dim(), err)
	}
}

// TestLoadPrefersLocalPaths pins the review's HIGH finding: a ref that is
// valid org/name syntax but whose parent IS a local directory must fail as
// a missing local path — never silently reach the network (with HF_TOKEN)
// because of a typo. "hf:" is the explicit override.
func TestLoadPrefersLocalPaths(t *testing.T) {
	parent := t.TempDir()
	ref := filepath.Join(filepath.Base(parent), "no-such-model")
	t.Chdir(filepath.Dir(parent))
	_, err := Load(ref)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "does not exist") || !strings.Contains(err.Error(), "hf:") {
		t.Fatalf("want a missing-local-path error naming the hf: override, got: %v", err)
	}
	// Plain single-segment nonsense is neither.
	if _, err := Load("definitely-not-a-thing"); err == nil ||
		!strings.Contains(err.Error(), "neither") {
		t.Fatalf("got: %v", err)
	}
}

// TestLoadFromHub exercises the pure-Go Hugging Face download path into a
// fresh cache. Network-dependent, so it runs only when opted in.
func TestLoadFromHub(t *testing.T) {
	if os.Getenv("REMBED_NETWORK_TESTS") == "" {
		t.Skip("set REMBED_NETWORK_TESTS=1 to run hub download tests")
	}
	t.Setenv("REMBED_CACHE", t.TempDir())
	emb, err := Load("sentence-transformers/all-MiniLM-L6-v2")
	if err != nil {
		t.Fatal(err)
	}
	if emb.Dim() != 384 {
		t.Fatalf("Dim()=%d", emb.Dim())
	}
	v, err := emb.Embed(context.Background(), []string{"hello world"})
	if err != nil {
		t.Fatal(err)
	}
	// Hub-derived config must produce the same vectors as the converted
	// model dir when both are present.
	if _, err := os.Stat("models/all-MiniLM-L6-v2/model.safetensors"); err == nil {
		ref, err := Load("models/all-MiniLM-L6-v2")
		if err != nil {
			t.Fatal(err)
		}
		w, err := ref.Embed(context.Background(), []string{"hello world"})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(v[0], w[0]) {
			t.Fatal("hub-loaded model diverged from converted model dir")
		}
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
	if raceDetectorEnabled {
		// The detector's shadow memory and the pool's race-mode sleep
		// polling both perturb alloc counts (measured 97 vs the 96 bound).
		// The efficiency contract is enforced by every non-race run.
		t.Skip("alloc counts are not meaningful under the race detector")
	}
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

// TestEmbedBatchMatchesSingle pins the R2 contract: the batched path must
// be BIT-identical to embedding each text alone (worker counts change
// scheduling, never results).
func TestEmbedBatchMatchesSingle(t *testing.T) {
	dir := modelDir(t)
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	texts := []string{
		"hello world",
		"The quick brown fox jumps over the lazy dog.",
		"a third, somewhat longer text to vary the sequence lengths in play",
		"short", "and", "many", "more", "texts",
	}
	// Pin BOTH parallelism regimes regardless of the host's core count
	// (the review caught within>1 never running on small CI runners):
	// GOMAXPROCS=8 with 2 texts forces across=2/within=4; with 16 texts
	// forces across=8/within=1.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	for _, tc := range []struct {
		procs int
		n     int
	}{{8, 2}, {8, 16}, {runtime.NumCPU(), len(texts)}} {
		runtime.GOMAXPROCS(tc.procs)
		sub := texts
		for len(sub) < tc.n {
			sub = append(sub, texts...)
		}
		sub = sub[:tc.n]
		batch, err := emb.Embed(ctx, sub)
		if err != nil {
			t.Fatal(err)
		}
		runtime.GOMAXPROCS(runtime.NumCPU())
		for i, text := range sub {
			one, err := emb.Embed(ctx, []string{text})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(batch[i], one[0]) {
				t.Fatalf("procs=%d n=%d text %d: batch diverged from single embed", tc.procs, tc.n, i)
			}
		}
	}

	// A cancelled batch reports the ctx error, like the single-text path.
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := emb.Embed(cctx, texts[:4]); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch returned %v, want context.Canceled", err)
	}
}

// TestEmbedTokensPoolsToEmbed pins internal consistency: mean-pooling and
// L2-normalizing EmbedTokens' output must reproduce Embed. Catches any
// future divergence between encode()'s output and Forward's pooling
// without needing a golden refresh.
func TestEmbedTokensPoolsToEmbed(t *testing.T) {
	dir := modelDir(t)
	emb, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	texts := []string{"hello world", "The quick brown fox jumps over the lazy dog.", "third text"}
	toks, err := emb.EmbedTokens(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	for ti, te := range toks {
		dim := emb.Dim()
		pooled := make([]float64, dim)
		for _, row := range te.Vectors {
			for j, v := range row {
				pooled[j] += float64(v)
			}
		}
		var norm float64
		for j := range pooled {
			pooled[j] /= float64(len(te.Vectors))
			norm += pooled[j] * pooled[j]
		}
		norm = math.Sqrt(norm)
		for j := range pooled {
			if d := math.Abs(pooled[j]/norm - float64(vecs[ti][j])); d > 1e-6 {
				t.Fatalf("text %d dim %d: pooled tokens %g vs Embed %g", ti, j, pooled[j]/norm, vecs[ti][j])
			}
		}
	}
}

// BenchmarkEmbedBatch32 measures throughput-mode: 32 texts per call.
func BenchmarkEmbedBatch32(b *testing.B) {
	dir := modelDir(b)
	emb, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	// Mixed lengths: uniform batches are the fan-out's best case (no
	// stragglers), so they overstate throughput for real workloads.
	base := []string{
		"short",
		"The quick brown fox jumps over the lazy dog.",
		strings.Repeat("a mid-length passage about nothing in particular ", 4),
		strings.Repeat("a long document body that pushes the sequence length up considerably ", 12),
	}
	texts := make([]string, 32)
	for i := range texts {
		texts[i] = base[i%len(base)]
	}
	for b.Loop() {
		if _, err := emb.Embed(ctx, texts); err != nil {
			b.Fatal(err)
		}
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
