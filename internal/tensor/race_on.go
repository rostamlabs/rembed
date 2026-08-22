// SPDX-License-Identifier: Apache-2.0

//go:build race

package tensor

// raceEnabled marks race-detector builds. The spin pool's idle loops are
// pathological under the detector (every polling load is instrumented, and
// the instrumentation serializes), so they degrade to micro-sleep polling
// — same concurrency, same code paths, ~1000× fewer instrumented ops.
const raceEnabled = true
