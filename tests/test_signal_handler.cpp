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

#include "core/signal/signal_handler.h"

#include <pthread.h>
#include <signal.h>

namespace {

class OriginalSignalMaskRestorer {
 public:
  OriginalSignalMaskRestorer() noexcept
      : captured_(pthread_sigmask(SIG_SETMASK, nullptr, &original_mask_) == 0) {}

  ~OriginalSignalMaskRestorer() noexcept {
    if (captured_) {
      pthread_sigmask(SIG_SETMASK, &original_mask_, nullptr);
    }
  }

  OriginalSignalMaskRestorer(const OriginalSignalMaskRestorer&) = delete;
  OriginalSignalMaskRestorer& operator=(const OriginalSignalMaskRestorer&) = delete;

  bool captured() const noexcept {
    return captured_;
  }

 private:
  sigset_t original_mask_{};
  bool captured_ = false;
};

sigset_t current_signal_mask() {
  sigset_t mask{};
  EXPECT_EQ(pthread_sigmask(SIG_SETMASK, nullptr, &mask), 0);
  return mask;
}

void expect_same_signal_mask(const sigset_t& actual, const sigset_t& expected) {
  for (int signal_number = 1; signal_number < NSIG; ++signal_number) {
    SCOPED_TRACE(signal_number);
    EXPECT_EQ(sigismember(&actual, signal_number),
              sigismember(&expected, signal_number));
  }
}

void set_signal_blocked(sigset_t* mask, int signal_number, bool blocked) {
  ASSERT_NE(mask, nullptr);
  ASSERT_EQ(blocked ? sigaddset(mask, signal_number)
                    : sigdelset(mask, signal_number),
            0);
}

void install_signal_mask(const sigset_t& mask) {
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, &mask, nullptr), 0);
}

}  // namespace

TEST(BenchmarkSignalMaskGuardTest, RestoresTheExactPreviousThreadMask) {
  OriginalSignalMaskRestorer restore_original;
  ASSERT_TRUE(restore_original.captured());

  sigset_t expected = current_signal_mask();
  set_signal_blocked(&expected, SIGINT, false);
  set_signal_blocked(&expected, SIGTERM, false);
  set_signal_blocked(&expected, SIGUSR1, true);
  set_signal_blocked(&expected, SIGUSR2, false);
  install_signal_mask(expected);

  {
    BenchmarkSignalMaskGuard guard;
    sigset_t changed = current_signal_mask();
    EXPECT_EQ(sigismember(&changed, SIGINT), 1);
    EXPECT_EQ(sigismember(&changed, SIGTERM), 1);

    set_signal_blocked(&changed, SIGUSR2, true);
    install_signal_mask(changed);
  }

  expect_same_signal_mask(current_signal_mask(), expected);
}

TEST(BenchmarkSignalMaskGuardTest, PreservesBenchmarkSignalsAlreadyBlockedByCaller) {
  OriginalSignalMaskRestorer restore_original;
  ASSERT_TRUE(restore_original.captured());

  sigset_t expected = current_signal_mask();
  set_signal_blocked(&expected, SIGINT, true);
  set_signal_blocked(&expected, SIGTERM, false);
  install_signal_mask(expected);

  {
    BenchmarkSignalMaskGuard guard;
    const sigset_t blocked = current_signal_mask();
    EXPECT_EQ(sigismember(&blocked, SIGINT), 1);
    EXPECT_EQ(sigismember(&blocked, SIGTERM), 1);
  }

  const sigset_t restored = current_signal_mask();
  EXPECT_EQ(sigismember(&restored, SIGINT), 1);
  EXPECT_EQ(sigismember(&restored, SIGTERM), 0);
  expect_same_signal_mask(restored, expected);
}

TEST(BenchmarkSignalMaskGuardTest, NestedGuardsRestoreTheirOwnPreviousMasks) {
  OriginalSignalMaskRestorer restore_original;
  ASSERT_TRUE(restore_original.captured());

  sigset_t expected = current_signal_mask();
  set_signal_blocked(&expected, SIGINT, false);
  set_signal_blocked(&expected, SIGTERM, false);
  install_signal_mask(expected);

  {
    BenchmarkSignalMaskGuard outer_guard;
    const sigset_t outer_mask = current_signal_mask();
    EXPECT_EQ(sigismember(&outer_mask, SIGINT), 1);
    EXPECT_EQ(sigismember(&outer_mask, SIGTERM), 1);

    {
      BenchmarkSignalMaskGuard inner_guard;
      const sigset_t inner_mask = current_signal_mask();
      expect_same_signal_mask(inner_mask, outer_mask);
    }

    expect_same_signal_mask(current_signal_mask(), outer_mask);
  }

  expect_same_signal_mask(current_signal_mask(), expected);
}
