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

#include <gtest/gtest.h>
#include <mach/mach_error.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"

namespace {

struct FakeTimerState {
  std::array<uint64_t, 8> ticks{};
  size_t tick_count = 0;
  size_t next_tick = 0;
  kern_return_t timebase_result = KERN_SUCCESS;
  uint32_t numer = 1;
  uint32_t denom = 1;
};

FakeTimerState* active_fake_timer = nullptr;

uint64_t fake_absolute_time() {
  if (active_fake_timer->next_tick >= active_fake_timer->tick_count) {
    return active_fake_timer->ticks[active_fake_timer->tick_count - 1];
  }
  return active_fake_timer->ticks[active_fake_timer->next_tick++];
}

kern_return_t fake_timebase_info(mach_timebase_info_t info) {
  info->numer = active_fake_timer->numer;
  info->denom = active_fake_timer->denom;
  return active_fake_timer->timebase_result;
}

class HighResTimerTest : public testing::Test {
 protected:
  void SetUp() override {
    active_fake_timer = &state;
    set_timer_system_calls_for_testing({fake_absolute_time, fake_timebase_info});
  }

  void TearDown() override {
    reset_timer_system_calls_for_testing();
    active_fake_timer = nullptr;
  }

  void set_ticks(std::initializer_list<uint64_t> ticks) {
    state.tick_count = ticks.size();
    state.next_tick = 0;
    size_t index = 0;
    for (uint64_t tick : ticks) {
      state.ticks[index++] = tick;
    }
  }

  FakeTimerState state;
};

}  // namespace

TEST_F(HighResTimerTest, TickConversionRatioAndCreationErrors) {
  const struct { const char* name; uint64_t ticks; uint32_t numer; uint32_t denom; bool valid; double ns; } ratios[] = {
      {"125/3 ratio", 3, 125, 3, true, 125.0},
      {"2/5 ratio", 250, 2, 5, true, 100.0},
      {"zero denominator", 1, 1, 0, false, 0.0},
  };
  for (const auto& entry : ratios) {
    SCOPED_TRACE(entry.name);
    const std::optional<double> converted = convert_mach_ticks_to_nanoseconds(entry.ticks, entry.numer, entry.denom);
    EXPECT_EQ(converted.has_value(), entry.valid);
    if (entry.valid) EXPECT_DOUBLE_EQ(*converted, entry.ns);
  }

  const struct { const char* name; kern_return_t rc; uint32_t denom; std::string expected; } errors[] = {
      {"mach failure", KERN_FAILURE, 1,
       Messages::error_prefix() + Messages::error_mach_timebase_info_failed(mach_error_string(KERN_FAILURE)) + "\n"},
      {"zero denominator", KERN_SUCCESS, 0,
       Messages::error_prefix() + "timebase denominator is zero (invalid timebase)\n"},
  };
  for (const auto& entry : errors) {
    SCOPED_TRACE(entry.name);
    state = FakeTimerState{};
    state.timebase_result = entry.rc;
    state.denom = entry.denom;
    testing::internal::CaptureStderr();
    const std::optional<HighResTimer> timer = HighResTimer::create();
    const std::string error = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(timer.has_value());
    EXPECT_EQ(error, entry.expected);
  }
}

TEST_F(HighResTimerTest, DeterministicClockAndSnapshotRetainExactBoundaries) {
  {
    SCOPED_TRACE("deterministic nanoseconds and seconds");
    state = FakeTimerState{};
    state.numer = 2;
    state.denom = 5;
    set_ticks({100, 350, 500, 750});
    std::optional<HighResTimer> timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());

    timer->start();
    EXPECT_DOUBLE_EQ(timer->stop_ns(), 100.0);
    timer->start();
    EXPECT_DOUBLE_EQ(timer->stop(), 100.0 / 1e9);
  }

  SCOPED_TRACE("snapshot retains original boundaries without extra clock reads");
  state = FakeTimerState{};
  state.numer = 125;
  state.denom = 3;
  set_ticks({100, 103, 200, 200});
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer);
  EXPECT_FALSE(timer->last_snapshot);
  timer->start();
  EXPECT_FALSE(timer->last_snapshot);
  EXPECT_DOUBLE_EQ(timer->stop(), 125.0 / 1e9);
  ASSERT_TRUE(timer->last_snapshot);
  EXPECT_EQ(timer->last_snapshot->start_ticks, 100u);
  EXPECT_EQ(timer->last_snapshot->stop_ticks, 103u);
  EXPECT_EQ(timer->last_snapshot->delta_ticks, 3u);
  EXPECT_EQ(timer->last_snapshot->numer, 125u);
  EXPECT_EQ(timer->last_snapshot->denom, 3u);
  EXPECT_EQ(state.next_tick, 2u);
  timer->start();
  EXPECT_FALSE(timer->last_snapshot);
  EXPECT_DOUBLE_EQ(timer->stop_ns(), 0.0);
  EXPECT_EQ(timer->last_snapshot->delta_ticks, 0u);
  EXPECT_EQ(state.next_tick, 4u);
}

TEST_F(HighResTimerTest, SnapshotWrapAndLargeConversionAvoidIntegerOverflow) {
  state.numer = std::numeric_limits<uint32_t>::max();
  state.denom = 1;
  set_ticks({10, 9});
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer);
  timer->start();
  const double ns = timer->stop_ns();
  ASSERT_TRUE(timer->last_snapshot);
  EXPECT_EQ(timer->last_snapshot->delta_ticks, std::numeric_limits<uint64_t>::max());
  EXPECT_DOUBLE_EQ(ns, static_cast<double>(std::numeric_limits<uint64_t>::max()) * state.numer);
}
