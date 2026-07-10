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

#include <array>
#include <cstddef>
#include <limits>
#include <mach/mach_error.h>
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
    set_timer_system_calls_for_testing(
        {fake_absolute_time, fake_timebase_info});
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

TEST_F(HighResTimerTest, TickConversionUsesExactTimebaseRatio) {
  const std::optional<double> first =
      convert_mach_ticks_to_nanoseconds(3, 125, 3);
  const std::optional<double> second =
      convert_mach_ticks_to_nanoseconds(250, 2, 5);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_DOUBLE_EQ(*first, 125.0);
  EXPECT_DOUBLE_EQ(*second, 100.0);
  EXPECT_FALSE(convert_mach_ticks_to_nanoseconds(1, 1, 0).has_value());
}

TEST_F(HighResTimerTest, CreateRejectsMachFailureWithCentralizedError) {
  state.timebase_result = KERN_FAILURE;
  testing::internal::CaptureStderr();
  const std::optional<HighResTimer> timer = HighResTimer::create();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(timer.has_value());
  EXPECT_EQ(error, Messages::error_prefix() +
                       Messages::error_mach_timebase_info_failed(
                           mach_error_string(KERN_FAILURE)) +
                       "\n");
}

TEST_F(HighResTimerTest, CreateRejectsZeroDenominator) {
  state.denom = 0;
  testing::internal::CaptureStderr();
  const std::optional<HighResTimer> timer = HighResTimer::create();
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(timer.has_value());
  EXPECT_EQ(error, Messages::error_prefix() +
                       "timebase denominator is zero (invalid timebase)\n");
}

TEST_F(HighResTimerTest, DeterministicClockProducesExactNanosecondsAndSeconds) {
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

TEST_F(HighResTimerTest, UnsignedTickSubtractionPreservesWraparound) {
  state.numer = 1;
  state.denom = 1;
  set_ticks({std::numeric_limits<uint64_t>::max() - 4, 5});
  std::optional<HighResTimer> timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  timer->start();
  EXPECT_DOUBLE_EQ(timer->stop_ns(), 10.0);
}

TEST(HighResTimerIntegrationTest, MonotonicReusableSmokeIntegration) {
  std::optional<HighResTimer> timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  timer->start();
  const double first = timer->stop_ns();
  const double second = timer->stop_ns();
  EXPECT_GE(first, 0.0);
  EXPECT_GE(second, first);

  timer->start();
  EXPECT_GE(timer->stop(), 0.0);
}
