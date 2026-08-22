// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"runtime"
	"sync/atomic"
)

// Pool is a spinning fork-join pool for a latency-critical SEQUENCE of
// fan-outs (one forward pass issues ~36 of them within a few ms). Spawning
// goroutines per fan-out costs a wake latency every time — measured, that
// overhead was ~half the short-sequence forward pass.
//
// THE COST, measured, not hand-waved: workers never park, so a pool of
// GOMAXPROCS-1 workers pins ~every core for the pool's lifetime. At
// concurrency 1 on a 20-core box a seq=12 embed spends ~68 CPU-ms to buy
// ~4 ms of wall time (~10× the useful work); under saturation the spinners
// still cost ~2.6× the serial CPU. That is the right trade for latency and
// the wrong one for throughput-saturated servers — rembed.WithWorkers
// exists for the latter (workers=1 gives a fully-inline, zero-spin pool).
//
// A panic inside fn is recovered on the worker, the fan-out completes, and
// the panic is re-raised on the Run caller — a library must not let a
// worker panic kill the host process, and Run must not return (or unwind)
// while workers may still touch the caller's buffers.
//
// NOTE: the race detector makes spinning pathologically slow (~10,000×
// per fan-out); keep any pool stress test's round count tiny under -race.
//
// Concurrency contract: one Run at a time (the forward pass is a single
// orchestrator); distinct concurrent forward passes each own their own
// Pool. Violations panic via the inRun guard rather than corrupting.
type Pool struct {
	cur     atomic.Pointer[poolTask]
	stop    atomic.Bool
	inRun   atomic.Bool
	workers int
}

type poolTask struct {
	fn        func(int)
	units     int64
	next      atomic.Int64
	completed atomic.Int64
	panicked  atomic.Pointer[any]
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
		runTask(t)
	}
}

// runTask pulls units until t's counter is exhausted. A worker that wakes
// late pulls from ITS task's counter, never a newer task's — task state is
// per-task, so a stale worker simply finds the counter drained. A panic in
// fn is captured per-unit (first wins) and the unit still counts as
// completed, so Run's join never hangs and never unwinds early.
func runTask(t *poolTask) {
	for {
		u := t.next.Add(1) - 1
		if u >= t.units {
			return
		}
		runUnit(t, int(u))
		t.completed.Add(1)
	}
}

func runUnit(t *poolTask, u int) {
	defer func() {
		if r := recover(); r != nil {
			t.panicked.CompareAndSwap(nil, &r)
		}
	}()
	t.fn(u)
}

// Run executes fn(u) for u in [0, units), the caller working alongside the
// pool's spinning workers, and returns when every unit has COMPLETED (not
// merely been claimed) — including when some fn panicked, in which case the
// first panic is re-raised here AFTER the join, so workers are never left
// writing into buffers the caller has released. The atomic completed
// counter gives the happens-before edge that makes workers' writes visible
// to the caller.
func (p *Pool) Run(units int, fn func(u int)) {
	if p == nil || p.workers == 0 || units <= 1 {
		for u := range units {
			fn(u)
		}
		return
	}
	if !p.inRun.CompareAndSwap(false, true) {
		panic("tensor: concurrent Pool.Run — a Pool serves one orchestrator at a time")
	}
	defer p.inRun.Store(false)
	t := &poolTask{fn: fn, units: int64(units)}
	p.cur.Store(t) // release: workers' acquire-load of cur sees fn/units
	runTask(t)
	for i := 0; t.completed.Load() < t.units; i++ {
		if i%(1<<14) == 0 && i > 0 {
			runtime.Gosched()
		}
	}
	if r := t.panicked.Load(); r != nil {
		panic(*r)
	}
}

// Stop makes all workers exit at their next poll. The pool must not be
// used again.
func (p *Pool) Stop() { p.stop.Store(true) }
