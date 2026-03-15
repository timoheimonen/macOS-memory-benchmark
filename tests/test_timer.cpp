// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//
/**
 * @file test_timer.cpp
 * @brief Unit tests for HighResTimer (src/core/timing/timer.cpp)
 *
 * Tests cover factory construction, basic elapsed-time correctness, monotonicity,
 * and nanosecond vs. seconds consistency.  All tests rely only on spin-wait
 * loops (no OS sleep) to keep them fast and deterministic.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "core/timing/timer.h"

// ---------------------------------------------------------------------------
// Factory: create() must succeed on Apple Silicon macOS.
// ---------------------------------------------------------------------------

TEST(HighResTimerTest, CreateReturnsValidTimer) {
  auto t = HighResTimer::create();
  EXPECT_TRUE(t.has_value()) << "HighResTimer::create() returned nullopt";
}

// ---------------------------------------------------------------------------
// Basic elapsed-time correctness
// ---------------------------------------------------------------------------

// After start() and immediate stop(), elapsed seconds must be non-negative.
TEST(HighResTimerTest, ElapsedSecondsNonNegative) {
  auto t = HighResTimer::create();
  ASSERT_TRUE(t.has_value());
  t->start();
  double elapsed = t->stop();
  EXPECT_GE(elapsed, 0.0);
}

// After start() and immediate stop_ns(), elapsed nanoseconds must be
// non-negative.
TEST(HighResTimerTest, ElapsedNanosecondsNonNegative) {
  auto t = HighResTimer::create();
  ASSERT_TRUE(t.has_value());
  t->start();
  double elapsed_ns = t->stop_ns();
  EXPECT_GE(elapsed_ns, 0.0);
}

// ---------------------------------------------------------------------------
// Monotonicity: two consecutive stop() calls must be non-decreasing.
// ---------------------------------------------------------------------------

TEST(HighResTimerTest, ConsecutiveStopsAreMonotonic) {
  auto t = HighResTimer::create();
  ASSERT_TRUE(t.has_value());
  t->start();
  // Burn a few cycles so the two samples are distinguishable.
  volatile int x = 0;
  for (int i = 0; i < 10000; ++i) x += i;
  double first = t->stop();
  for (int i = 0; i < 10000; ++i) x += i;
  double second = t->stop();
  (void)x;
  EXPECT_GE(second, first);
}

// ---------------------------------------------------------------------------
// Seconds / nanoseconds consistency
// ---------------------------------------------------------------------------

// stop_ns() should return a value approximately 1e9× larger than stop()
// for the same interval.  We verify with a very loose ratio bound to avoid
// timer-resolution sensitivity.
TEST(HighResTimerTest, NanosecondsConsistentWithSeconds) {
  auto ts = HighResTimer::create();
  auto tns = HighResTimer::create();
  ASSERT_TRUE(ts.has_value());
  ASSERT_TRUE(tns.has_value());

  volatile int x = 0;
  ts->start();
  for (int i = 0; i < 1000000; ++i) x += i;
  double sec = ts->stop();

  tns->start();
  for (int i = 0; i < 1000000; ++i) x += i;
  double ns = tns->stop_ns();
  (void)x;

  // Both intervals should be in a similar ballpark (within 100×) given the
  // same workload.  Primary check: ns ≈ sec * 1e9 within two orders of
  // magnitude.
  EXPECT_GT(sec, 0.0);
  EXPECT_GT(ns, 0.0);
  if (sec > 0.0) {
    double ratio = ns / (sec * 1e9);
    EXPECT_GT(ratio, 0.01) << "ns/sec*1e9 ratio too small: " << ratio;
    EXPECT_LT(ratio, 100.0) << "ns/sec*1e9 ratio too large: " << ratio;
  }
}

// ---------------------------------------------------------------------------
// Repeated start()/stop() — timer is reusable.
// ---------------------------------------------------------------------------

TEST(HighResTimerTest, TimerIsReusable) {
  auto t = HighResTimer::create();
  ASSERT_TRUE(t.has_value());

  for (int round = 0; round < 5; ++round) {
    t->start();
    volatile double sink = 0.0;
    for (int i = 0; i < 5000; ++i) sink += static_cast<double>(i);
    (void)sink;
    double elapsed = t->stop();
    EXPECT_GE(elapsed, 0.0) << "Non-negative elapsed on round " << round;
  }
}
