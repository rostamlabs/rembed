// SPDX-License-Identifier: Apache-2.0

// Command rembed embeds text, validates against the golden ONNX reference,
// and benchmarks the engine.
//
//	rembed embed    -model DIR [-full] text...
//	rembed validate -model DIR [-golden FILE] [-tol 1e-4]
//	rembed bench    -model DIR [-runs 30] [-warmup 5] [-text S]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"time"

	"github.com/rostamlabs/rembed"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "embed":
		err = cmdEmbed(os.Args[2:])
	case "validate":
		err = cmdValidate(os.Args[2:])
	case "bench":
		err = cmdBench(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "rembed:", err)
		os.Exit(1)
	}
}

func int8Opts(useInt8 bool) []rembed.Option {
	if useInt8 {
		return []rembed.Option{rembed.WithInt8()}
	}
	return nil
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: rembed {embed|validate|bench} -model DIR [flags] [args]")
	os.Exit(2)
}

func cmdEmbed(args []string) error {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	modelDir := fs.String("model", "models/all-MiniLM-L6-v2", "model directory")
	full := fs.Bool("full", false, "print full vectors as JSON instead of a preview")
	useInt8 := fs.Bool("int8", false, "weight-only int8 inference")
	_ = fs.Parse(args)
	texts := fs.Args()
	if len(texts) == 0 {
		return fmt.Errorf("embed: no texts given")
	}
	emb, err := rembed.Load(*modelDir, int8Opts(*useInt8)...)
	if err != nil {
		return err
	}
	vecs, err := emb.Embed(context.Background(), texts)
	if err != nil {
		return err
	}
	if *full {
		return json.NewEncoder(os.Stdout).Encode(vecs)
	}
	for i, v := range vecs {
		n := min(8, len(v))
		fmt.Printf("%q -> dim=%d %v...\n", texts[i], len(v), v[:n])
	}
	return nil
}

// goldenFile mirrors what models/convert.py writes.
type goldenFile struct {
	Model string `json:"model"`
	Cases []struct {
		Text      string    `json:"text"`
		InputIDs  []int64   `json:"input_ids"`
		Embedding []float32 `json:"embedding"`
	} `json:"cases"`
}

func cmdValidate(args []string) error {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	modelDir := fs.String("model", "models/all-MiniLM-L6-v2", "model directory")
	goldenPath := fs.String("golden", "testdata/golden.json", "golden reference file")
	tol := fs.Float64("tol", 1e-4, "max abs difference tolerance (int8 default: 0.03)")
	useInt8 := fs.Bool("int8", false, "weight-only int8 inference (loosens default -tol)")
	force := fs.Bool("force", false, "proceed even if the golden was generated for a different model")
	_ = fs.Parse(args)
	if *useInt8 && *tol == 1e-4 {
		*tol = 0.03 // quantization bound pinned by TestGoldenInt8
	}

	raw, err := os.ReadFile(*goldenPath)
	if err != nil {
		return err
	}
	var golden goldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		return fmt.Errorf("golden %s: %w", *goldenPath, err)
	}
	if len(golden.Cases) == 0 {
		return fmt.Errorf("golden %s: no cases — refusing to report success on an empty reference", *goldenPath)
	}
	emb, err := rembed.Load(*modelDir, int8Opts(*useInt8)...)
	if err != nil {
		return err
	}
	if emb.Model() != golden.Model {
		if !*force {
			return fmt.Errorf("golden is for %q but model dir is %q (use -force to compare anyway)", golden.Model, emb.Model())
		}
		fmt.Printf("note: golden is for %q, model dir is %q\n", golden.Model, emb.Model())
	}

	failures := 0
	for _, c := range golden.Cases {
		ids := emb.Tokenize(c.Text)
		if !slices.Equal(ids, c.InputIDs) {
			failures++
			fmt.Printf("FAIL %-40.40q tokenizer mismatch:\n  got  %v\n  want %v\n", c.Text, ids, c.InputIDs)
			continue
		}
		vecs, err := emb.Embed(context.Background(), []string{c.Text})
		if err != nil {
			return err
		}
		if len(c.Embedding) != len(vecs[0]) {
			return fmt.Errorf("golden %q has dim %d but model produced dim %d", c.Text, len(c.Embedding), len(vecs[0]))
		}
		maxDiff := 0.0
		for i, want := range c.Embedding {
			if d := math.Abs(float64(vecs[0][i] - want)); d > maxDiff {
				maxDiff = d
			}
		}
		status := "ok  "
		if maxDiff > *tol {
			status = "FAIL"
			failures++
		}
		fmt.Printf("%s %-40.40q seq=%-3d maxAbsDiff=%.2e\n", status, c.Text, len(ids), maxDiff)
	}
	if failures > 0 {
		return fmt.Errorf("validate: %d/%d cases failed (tol %g)", failures, len(golden.Cases), *tol)
	}
	fmt.Printf("validate: all %d cases within %g of the ONNX reference\n", len(golden.Cases), *tol)
	return nil
}

func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	modelDir := fs.String("model", "models/all-MiniLM-L6-v2", "model directory")
	runs := fs.Int("runs", 30, "measured runs")
	warmup := fs.Int("warmup", 5, "discarded warm-up runs")
	useInt8 := fs.Bool("int8", false, "weight-only int8 inference")
	text := fs.String("text", "The quick brown fox jumps over the lazy dog.", "input text")
	asJSON := fs.Bool("json", false, "emit machine-readable per-run latencies (for bench/compare.py)")
	_ = fs.Parse(args)
	if *runs <= 0 || *warmup < 0 {
		return fmt.Errorf("bench: -runs must be > 0 and -warmup >= 0 (got %d, %d)", *runs, *warmup)
	}

	emb, err := rembed.Load(*modelDir, int8Opts(*useInt8)...)
	if err != nil {
		return err
	}
	ids := emb.Tokenize(*text)
	ctx := context.Background()
	for range *warmup {
		if _, err := emb.Embed(ctx, []string{*text}); err != nil {
			return err
		}
	}
	durs := make([]time.Duration, 0, *runs)
	for range *runs {
		start := time.Now()
		if _, err := emb.Embed(ctx, []string{*text}); err != nil {
			return err
		}
		durs = append(durs, time.Since(start))
	}
	if *asJSON {
		runsSec := make([]float64, len(durs))
		for i, d := range durs {
			runsSec[i] = d.Seconds()
		}
		// seq lets compare.py verify both engines tokenize to the same length.
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"runs_sec": runsSec, "seq": len(ids)})
	}
	slices.Sort(durs)
	median := durs[len(durs)/2]
	p10, p90 := durs[len(durs)/10], durs[len(durs)*9/10]
	spreadPct := 100 * float64(p90-p10) / float64(median)
	fmt.Printf("model=%s seq=%d runs=%d (+%d warmup)\n", emb.Model(), len(ids), *runs, *warmup)
	fmt.Printf("median=%v p10=%v p90=%v spread=%.1f%% of median\n", median, p10, p90, spreadPct)
	fmt.Println("note: single-machine numbers are relative-only; see bench/ for the us-vs-ONNX protocol")
	return nil
}
