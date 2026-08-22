// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rostamlabs/rembed"
)

// cmdServe runs an OpenAI-compatible embeddings endpoint:
//
//	POST /v1/embeddings  {"input": "text" | ["texts", ...]}
//	GET  /healthz
//
// so anything that can call the OpenAI embeddings API — any language, any
// framework — can consume rembed by pointing a base URL at it.
func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelRef := fs.String("model", "sentence-transformers/all-MiniLM-L6-v2", "model directory or Hugging Face id")
	addr := fs.String("addr", ":8080", "listen address")
	useInt8 := fs.Bool("int8", false, "weight-only int8 inference")
	workers := fs.Int("workers", 0, "CPU workers per request (0 = all cores; set low for many concurrent clients)")
	maxBatch := fs.Int("max-batch", 256, "maximum texts per request")
	_ = fs.Parse(args)
	if *maxBatch < 1 {
		return fmt.Errorf("serve: -max-batch must be >= 1")
	}

	opts := int8Opts(*useInt8)
	if *workers > 0 {
		opts = append(opts, rembed.WithWorkers(*workers))
	}
	log.Printf("loading %s (int8=%v workers=%d)...", *modelRef, *useInt8, *workers)
	emb, err := rembed.Load(*modelRef, opts...)
	if err != nil {
		return err
	}
	log.Printf("model %s ready: dim=%d quantized=%v", emb.Model(), emb.Dim(), emb.Quantized())

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model": emb.Model(), "dim": emb.Dim()})
	})
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		handleEmbeddings(w, r, emb, *maxBatch)
	})

	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()
	log.Printf("serving OpenAI-compatible embeddings on %s", *addr)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	select {
	case err := <-errCh:
		return err
	case sig := <-stop:
		log.Printf("received %v, draining...", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return srv.Shutdown(ctx)
	}
}

// embeddingsRequest is the OpenAI request shape. input is a string or an
// array of strings; token-id arrays (which the OpenAI API also accepts)
// are rejected with a clear error rather than mis-embedded.
type embeddingsRequest struct {
	Input json.RawMessage `json:"input"`
	Model string          `json:"model"` // accepted and echoed; rembed serves one model
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request, emb *rembed.Embedder, maxBatch int) {
	var req embeddingsRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	texts, err := parseInput(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(texts) == 0 {
		writeError(w, http.StatusBadRequest, "input is empty")
		return
	}
	if len(texts) > maxBatch {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("batch of %d exceeds max-batch %d", len(texts), maxBatch))
		return
	}

	vecs, err := emb.Embed(r.Context(), texts)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return // client went away
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	tokens := 0
	for _, t := range texts {
		tokens += len(emb.Tokenize(t))
	}
	data := make([]map[string]any, len(vecs))
	for i, v := range vecs {
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": v}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  emb.Model(),
		"usage":  map[string]int{"prompt_tokens": tokens, "total_tokens": tokens},
	})
}

// parseInput accepts the OpenAI input forms rembed can serve: a JSON
// string or an array of strings.
func parseInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("missing input")
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return many, nil
	}
	return nil, fmt.Errorf("input must be a string or an array of strings (token-id arrays are not supported)")
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError uses the OpenAI error envelope so client SDKs surface the
// message instead of a generic failure.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}
