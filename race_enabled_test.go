// SPDX-License-Identifier: Apache-2.0

//go:build race

package rembed

// raceDetectorEnabled lets tests whose assertions are perturbed by the
// race detector itself (allocation counts) skip under -race.
const raceDetectorEnabled = true
