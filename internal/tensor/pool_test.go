// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"sync/atomic"
	"testing"
)

// Round counts here are deliberately tiny: the race detector makes the
// spin loops ~10,000× slower per fan-out, and these tests run under -race
// in CI paths (see the note on Pool).

func TestPoolRunsEveryUnitOnce(t *testing.T) {
	p := NewPool(3)
	defer p.Stop()
	for _, units := range []int{0, 1, 2, 7, 64} {
		hits := make([]atomic.Int32, units)
		p.Run(units, func(u int) { hits[u].Add(1) })
		for u := range hits {
			if got := hits[u].Load(); got != 1 {
				t.Fatalf("units=%d: unit %d executed %d times", units, u, got)
			}
		}
	}
}

// TestPoolPropagatesPanicAfterJoin pins the library contract the M4 review
// found broken: a panic in fn must (a) surface on the Run caller where a
// host's recover can handle it — never kill the process — and (b) surface
// only AFTER every unit completed, so workers are never still writing into
// buffers the unwinding caller has released back to a pool.
func TestPoolPropagatesPanicAfterJoin(t *testing.T) {
	p := NewPool(3)
	defer p.Stop()
	var completed atomic.Int32
	const units = 32
	defer func() {
		if r := recover(); r != "boom" {
			t.Fatalf("recovered %v, want \"boom\"", r)
		}
		if got := completed.Load(); got != units {
			t.Fatalf("panic surfaced before the join: %d/%d units completed", got, units)
		}
		// The pool must remain usable after a panicking task. ok is atomic:
		// the two units can run on different worker goroutines, so a plain
		// bool would be a data race even though both writes store true.
		var ok atomic.Bool
		p.Run(2, func(u int) { ok.Store(true) })
		if !ok.Load() {
			t.Fatal("pool dead after panic")
		}
	}()
	p.Run(units, func(u int) {
		defer completed.Add(1)
		if u == 5 {
			panic("boom")
		}
	})
	t.Fatal("Run returned instead of panicking")
}

func TestPoolConcurrentRunPanics(t *testing.T) {
	p := NewPool(2)
	defer p.Stop()
	started := make(chan struct{})
	release := make(chan struct{})
	go func() {
		p.Run(2, func(u int) {
			if u == 0 {
				close(started)
				<-release
			}
		})
	}()
	<-started
	func() {
		defer func() {
			if recover() == nil {
				t.Error("concurrent Run did not panic")
			}
			close(release)
		}()
		p.Run(2, func(int) {})
	}()
}
