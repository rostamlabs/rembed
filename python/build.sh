#!/bin/sh
# Builds the rembed shared library for the Python bindings.
# Requires a C toolchain (cgo) — the Go library itself stays pure Go;
# this artifact exists only for foreign-function callers.
set -e
cd "$(dirname "$0")/.."
CGO_ENABLED=1 go build -buildmode=c-shared -o python/rembed/librembed.so ./python/capi
echo "built python/rembed/librembed.so"
