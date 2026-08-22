// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"runtime"
	"sync/atomic"
)

// Pool is a spinning fork-join pool for a latency-critical SEQUENCE of
// fan-outs (one forward pass issues ~36 of them within a few ms). Spawning
// goroutines per fan-out costs a wake latency every time — measured, that
// overhead was ~half the short-sequence forward pass. Pool workers are
// spawned once, spin between tasks (they never park — the pool's lifetime
// is one forward pass, and burning a few idle cores for its duration is
// the latency trade), and exit on Stop.
//
// Concurrency contract: one Run at a time (the forward pass is a single
// orchestrator); distinct concurrent forward passes each own their own
// Pool.
type Pool struct {
	cur     atomic.Pointer[poolTask]
	stop    atomic.Bool
	workers int
}

type poolTask struct {
	fn        func(int)
	units     int64
	next      atomic.Int64
	completed atomic.Int64
}

// NewPool starts workers spinning goroutines. workers <= 0 gives a pool
// that runs everything inline on the caller.
func NewPool(workers int) *Pool {
	p := &Pool{workers: max(workers, 0)}
	for range p.workers {
		go p.worker()
	}
	return p
}

func (p *Pool) worker() {
	var last *poolTask
	for i := 0; ; i++ {
		t := p.cur.Load()
		if t == last || t == nil {
			if p.stop.Load() {
				return
			}
			if i%(1<<14) == 0 {
				// Stay preemptible without giving up the spin's latency.
				runtime.Gosched()
			}
			continue
		}
		last = t
		p.runTask(t)
	}
}

// runTask pulls units until t's counter is exhausted. A worker that wakes
// late pulls from ITS task's counter, never a newer task's — task state is
// per-task, so a stale worker simply finds the counter drained.
func (p *Pool) runTask(t *poolTask) {
	for {
		u := t.next.Add(1) - 1
		if u >= t.units {
			return
		}
		t.fn(int(u))
		t.completed.Add(1)
	}
}

// Run executes fn(u) for u in [0, units), the caller working alongside the
// pool's spinning workers, and returns when every unit has COMPLETED (not
// merely been claimed). The atomic completed counter gives the
// happens-before edge that makes workers' writes visible to the caller.
func (p *Pool) Run(units int, fn func(u int)) {
	if p == nil || p.workers == 0 || units <= 1 {
		for u := range units {
			fn(u)
		}
		return
	}
	t := &poolTask{fn: fn, units: int64(units)}
	p.cur.Store(t) // release: workers' acquire-load of cur sees fn/units
	p.runTask(t)
	for i := 0; t.completed.Load() < t.units; i++ {
		if i%(1<<14) == 0 && i > 0 {
			runtime.Gosched()
		}
	}
}

// Stop makes all workers exit at their next poll. The pool must not be
// used again.
func (p *Pool) Stop() { p.stop.Store(true) }
