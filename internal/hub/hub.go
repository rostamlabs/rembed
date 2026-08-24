// SPDX-License-Identifier: Apache-2.0

// Package hub fetches model files straight from the Hugging Face Hub in
// pure Go — no Python, no conversion step. HuggingFace ships
// model.safetensors natively for the supported models, so "conversion" was
// only ever a download plus a manifest, and the manifest is now derived in
// Go (internal/model). Files land in a local cache directory and are
// reused on every later Load.
package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/rostamlabs/rembed/internal/safetensors"
)

// required are the files every sentence-transformers-format repo must
// provide, regardless of architecture — fetched AFTER config.json's
// model_type has been checked, so an unsupported architecture is refused
// having downloaded one small JSON file, not the weights. modules.json is
// required deliberately: silently assuming "no Normalize module" for a
// repo that actually has one would shift every downstream cosine
// threshold, and every genuine ST repo ships the file.
var required = []string{
	"1_Pooling/config.json",
	"modules.json",
}

// tokenizerFiles returns the tokenizer artifacts to fetch and whether
// sentencepiece.bpe.model must be PROBED first. tokenizer_class is only a
// fast path: real repos ship classes like "PreTrainedTokenizerFast" (the
// multilingual MiniLM does — the review caught the probe being gated on
// an empty class, which made the flagship multilingual model unfetchable),
// so any class that is not explicitly XLM-R triggers the probe, and a 404
// falls back to the model_type rules: byte-level BPE (RoBERTa) ships
// vocab.json + merges.txt; WordPiece (BERT, MPNet) ships vocab.txt.
func tokenizerFiles(modelType, tokenizerClass string) (files []string, probe bool) {
	if modelType == "xlm-roberta" || strings.HasPrefix(tokenizerClass, "XLMRobertaTokenizer") {
		return []string{"sentencepiece.bpe.model"}, false
	}
	if modelType == "roberta" {
		return []string{"vocab.json", "merges.txt"}, true
	}
	if modelType == "modernbert" || modelType == "qwen3" {
		// Byte-level BPE with vocab and merges embedded in tokenizer.json;
		// no sentencepiece probe.
		return []string{"tokenizer.json"}, false
	}
	if modelType == "gemma3_text" || modelType == "gemma3" {
		// EmbeddingGemma: SentencePiece-style byte-fallback BPE, all in
		// tokenizer.json; no sentencepiece.bpe.model probe.
		return []string{"tokenizer.json"}, false
	}
	if modelType == "nomic_bert" {
		// nomic-embed uses the BERT WordPiece vocab (vocab.txt); no probe.
		return []string{"vocab.txt"}, false
	}
	return []string{"vocab.txt"}, true
}

// supported mirrors internal/model's architecture allowlist. Duplicated
// here on purpose: hub must refuse BEFORE downloading gigabytes, and
// cannot import internal/model (which would invert the dependency).
func supported(modelType string) bool {
	switch modelType {
	case "bert", "distilbert", "modernbert", "qwen3", "gemma3_text", "gemma3", "nomic_bert", "roberta", "xlm-roberta", "mpnet":
		return true
	}
	return false
}

// errNotFound marks a per-file 404, letting Ensure probe for optional
// files (a repo either ships sentencepiece.bpe.model or it does not).
var errNotFound = errors.New("")

var idRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$`)

// IsModelID reports whether ref looks like a Hugging Face model id
// (org/name) rather than a filesystem path.
func IsModelID(ref string) bool {
	return idRe.MatchString(ref) && !strings.Contains(ref, "..")
}

// CacheDir returns the model cache root: $REMBED_CACHE if set, else
// <user cache dir>/rembed.
func CacheDir() (string, error) {
	if env := os.Getenv("REMBED_CACHE"); env != "" {
		return env, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("hub: no cache dir (set REMBED_CACHE): %w", err)
	}
	return filepath.Join(base, "rembed"), nil
}

// Ensure downloads modelID's files into the cache (skipping files already
// present) and returns the model directory. Set HF_TOKEN for gated or
// rate-limited access.
func Ensure(modelID, cacheDir string) (string, error) {
	if !IsModelID(modelID) {
		return "", fmt.Errorf("hub: %q is not a model id (want org/name)", modelID)
	}
	dir := filepath.Join(cacheDir, strings.ReplaceAll(modelID, "/", "--"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	sweepStaleTemps(dir)
	cleanup := func(files ...string) {
		// A failed fetch should not litter the cache: remove whatever was
		// fetched this far, then the (now empty) directories — os.Remove
		// refuses non-empty ones, so a dir with unrelated content stays.
		for _, f := range files {
			_ = os.Remove(filepath.Join(dir, filepath.FromSlash(f)))
		}
		for _, sub := range []string{"1_Pooling", "2_Dense", "3_Dense"} {
			_ = os.Remove(filepath.Join(dir, sub))
		}
		_ = os.Remove(dir)
	}
	// config.json first and alone: model_type decides whether to continue
	// at all, before anything else is spent.
	if err := fetch(modelID, "config.json", dir); err != nil {
		cleanup()
		return "", err
	}
	var hf struct {
		ModelType string `json:"model_type"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err == nil {
		err = json.Unmarshal(raw, &hf)
	}
	if err != nil {
		cleanup("config.json")
		return "", fmt.Errorf("hub: %s: config.json: %w", modelID, err)
	}
	if !supported(hf.ModelType) {
		cleanup("config.json")
		return "", fmt.Errorf("hub: %s: model_type %q is not supported (rembed runs bert, distilbert, modernbert, qwen3, gemma3, roberta, xlm-roberta, and mpnet encoders)", modelID, hf.ModelType)
	}
	fetched := []string{"config.json"}
	if err := fetch(modelID, "tokenizer_config.json", dir); err != nil {
		cleanup(fetched...)
		return "", err
	}
	fetched = append(fetched, "tokenizer_config.json")
	var tc struct {
		TokenizerClass string `json:"tokenizer_class"`
	}
	raw, err = os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err == nil {
		err = json.Unmarshal(raw, &tc)
	}
	if err != nil {
		cleanup(fetched...)
		return "", fmt.Errorf("hub: %s: tokenizer_config.json: %w", modelID, err)
	}
	// Tokenizer files come BEFORE the weights: they are small, and a
	// probe miss or 404 must not cost a 470 MB download per retry.
	tokFiles, probe := tokenizerFiles(hf.ModelType, tc.TokenizerClass)
	if probe {
		err := fetch(modelID, "sentencepiece.bpe.model", dir)
		switch {
		case err == nil:
			tokFiles = nil
			fetched = append(fetched, "sentencepiece.bpe.model")
		case errors.Is(err, errNotFound):
			// not a SentencePiece repo; keep the model_type files
		default:
			cleanup(fetched...)
			return "", err
		}
	}
	for _, f := range append(tokFiles, required...) {
		if err := fetch(modelID, f, dir); err != nil {
			cleanup(fetched...)
			return "", err
		}
		fetched = append(fetched, f)
	}
	// EmbeddingGemma ships its projection head as two separate
	// sentence-transformers Dense modules (2_Dense/, 3_Dense/), each a small
	// config + safetensors that loadGemma3 reads alongside the backbone.
	if hf.ModelType == "gemma3_text" || hf.ModelType == "gemma3" {
		for _, f := range []string{
			"2_Dense/config.json", "2_Dense/model.safetensors",
			"3_Dense/config.json", "3_Dense/model.safetensors",
		} {
			if err := fetch(modelID, f, dir); err != nil {
				cleanup(fetched...)
				return "", err
			}
			fetched = append(fetched, f)
		}
	}
	// Weights come last (largest). A single model.safetensors is the common
	// case; large checkpoints (Qwen3-4B/8B) ship a sharded set named by
	// model.safetensors.index.json, so a 404 there falls back to fetching the
	// index and every shard it lists.
	weightFiles, err := fetchWeights(modelID, dir)
	if err != nil {
		cleanup(fetched...)
		return "", err
	}
	fetched = append(fetched, weightFiles...)
	return dir, nil
}

// fetchWeights downloads model.safetensors, or — when the repo shards its
// weights — model.safetensors.index.json plus every shard it names. Returns
// the files fetched (for cleanup on a later error).
func fetchWeights(modelID, dir string) ([]string, error) {
	err := fetch(modelID, "model.safetensors", dir)
	if err == nil {
		return []string{"model.safetensors"}, nil
	}
	if !errors.Is(err, errNotFound) {
		return nil, err
	}
	if err := fetch(modelID, "model.safetensors.index.json", dir); err != nil {
		return nil, fmt.Errorf("hub: %s has neither model.safetensors nor a shard index: %w", modelID, err)
	}
	got := []string{"model.safetensors.index.json"}
	raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return got, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return got, fmt.Errorf("hub: %s: bad shard index: %w", modelID, err)
	}
	seen := make(map[string]struct{})
	shards := make([]string, 0, len(idx.WeightMap))
	for _, f := range idx.WeightMap {
		// The index is downloaded and untrusted: a shard name is joined into
		// a filesystem path (and a resolve URL), so reject anything that is
		// not a plain filename before it can escape the cache dir.
		if !safetensors.ValidShardName(f) {
			return got, fmt.Errorf("hub: %s: unsafe shard name %q in index", modelID, f)
		}
		if _, ok := seen[f]; ok {
			continue
		}
		seen[f] = struct{}{}
		shards = append(shards, f)
	}
	sort.Strings(shards)
	for _, shard := range shards {
		if err := fetch(modelID, shard, dir); err != nil {
			return got, err
		}
		got = append(got, shard)
	}
	return got, nil
}

// sweepStaleTemps removes download temp files older than an hour — a
// SIGKILL mid-download orphans them, and nothing else reclaims the space.
func sweepStaleTemps(dir string) {
	matches, _ := filepath.Glob(filepath.Join(dir, ".rembed-dl-*"))
	for _, m := range matches {
		if fi, err := os.Stat(m); err == nil && time.Since(fi.ModTime()) > time.Hour {
			_ = os.Remove(m)
		}
	}
}

// fetch downloads one file if it is not cached, atomically (tmp + rename),
// so a concurrent or interrupted download never leaves a truncated file
// under the final name. Large (LFS) files carry a sha256 in the redirect
// hop's X-Linked-Etag header; when present the body is verified against
// it, so a proxy-served error page or truncated transfer can never poison
// the cache.
// hfTokenOnce caches the resolved Hugging Face token for the process.
var (
	hfTokenOnce  sync.Once
	hfTokenValue string
)

// hfToken resolves a Hugging Face access token for gated/private repos:
// $HF_TOKEN (or $HUGGING_FACE_HUB_TOKEN) first, then the token file the
// `hf`/`huggingface-cli` login writes ($HF_HOME/token, else
// ~/.cache/huggingface/token) — so a user who ran `hf auth login` needs no
// extra env var. Returns "" when none is found (anonymous access).
func hfToken() string {
	hfTokenOnce.Do(func() {
		for _, env := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
			if v := strings.TrimSpace(os.Getenv(env)); v != "" {
				hfTokenValue = v
				return
			}
		}
		paths := []string{}
		if home := os.Getenv("HF_HOME"); home != "" {
			paths = append(paths, filepath.Join(home, "token"))
		}
		if home, err := os.UserHomeDir(); err == nil {
			paths = append(paths, filepath.Join(home, ".cache", "huggingface", "token"))
		}
		for _, p := range paths {
			if b, err := os.ReadFile(p); err == nil {
				if v := strings.TrimSpace(string(b)); v != "" {
					hfTokenValue = v
					return
				}
			}
		}
	})
	return hfTokenValue
}

func fetch(modelID, name, dir string) error {
	dst := filepath.Join(dir, filepath.FromSlash(name))
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	// The integrity headers appear on the FIRST response (the redirect to
	// the CDN), so capture them in CheckRedirect. One client per fetch
	// keeps that capture race-free.
	var wantSHA string
	var wantSize int64 = -1
	client := &http.Client{
		Timeout: 15 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if r := via[len(via)-1].Response; r != nil {
				if etag := strings.Trim(r.Header.Get("X-Linked-Etag"), `W/"`); len(etag) == 64 {
					wantSHA = etag
				}
				if v := r.Header.Get("X-Linked-Size"); v != "" {
					_, _ = fmt.Sscan(v, &wantSize)
				}
			}
			return nil
		},
	}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelID, name)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "rembed/0.3")
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("hub: fetching %s/%s: %w", modelID, name, err)
	}
	defer func() { _ = resp.Body.Close() }()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("hub: %s has no %s — not a sentence-transformers-format encoder repo%w", modelID, name, errNotFound)
	case http.StatusUnauthorized, http.StatusForbidden:
		// HF answers 401 for repos that DO NOT EXIST as well as for gated
		// and private ones — never assume this is an auth problem.
		hint := "no HF_TOKEN is set"
		if hfToken() != "" {
			hint = "an HF token is set but may be invalid or lack access"
		}
		return fmt.Errorf("hub: %s: HTTP %s — the repo does not exist, is gated, or is private (%s)", modelID, resp.Status, hint)
	default:
		return fmt.Errorf("hub: fetching %s/%s: HTTP %s", modelID, name, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".rembed-dl-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	hasher := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, hasher), resp.Body)
	if err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return fmt.Errorf("hub: downloading %s/%s: %w", modelID, name, err)
	}
	if wantSize > 0 && n != wantSize {
		return fmt.Errorf("hub: %s/%s truncated: %d of %d bytes", modelID, name, n, wantSize)
	}
	if resp.ContentLength > 0 && n != resp.ContentLength {
		return fmt.Errorf("hub: %s/%s truncated: %d of %d bytes", modelID, name, n, resp.ContentLength)
	}
	if wantSHA != "" {
		if got := hex.EncodeToString(hasher.Sum(nil)); got != wantSHA {
			return fmt.Errorf("hub: %s/%s content hash mismatch (got %s want %s) — corrupt transfer or interfering proxy", modelID, name, got, wantSHA)
		}
	}
	return os.Rename(tmp.Name(), dst)
}
