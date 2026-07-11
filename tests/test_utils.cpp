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

/**
 * @file test_utils.cpp
 * @brief Unit tests for shared progress and thread utilities
 */

#include <gtest/gtest.h>

#include "utils/utils.h"

#include <atomic>
#include <sstream>
#include <thread>
#include <vector>

TEST(ProgressSpinnerTest, ForcedDisabledModeProducesNoOutput) {
  std::ostringstream output;
  {
    ProgressSpinner spinner(output, false);
    spinner.tick("hidden");
    spinner.clear();
  }

  EXPECT_TRUE(output.str().empty());
}

TEST(ProgressSpinnerTest, ForcedEnabledModeCyclesFramesExactly) {
  std::ostringstream output;
  {
    ProgressSpinner spinner(output, true);
    spinner.tick("work");
    spinner.tick("work");
    spinner.tick("work");
    spinner.tick("work");
    spinner.tick("work");
  }

  EXPECT_EQ(output.str(),
            "\r| work"
            "\r/ work"
            "\r- work"
            "\r\\ work"
            "\r| work"
            "\r      \r");
}

TEST(ProgressSpinnerTest, ShorterMessageIsPaddedAndClearErasesFullWidth) {
  std::ostringstream output;
  ProgressSpinner spinner(output, true);

  spinner.tick("longer");
  spinner.tick("x");
  spinner.clear();

  EXPECT_EQ(output.str(),
            "\r| longer"
            "\r/ x     "
            "\r        \r");
}

TEST(ProgressSpinnerTest, ExplicitClearMakesClearAndDestructorIdempotent) {
  std::ostringstream output;
  {
    ProgressSpinner spinner(output, true);
    spinner.tick("done");
    spinner.clear();
    spinner.clear();
  }

  EXPECT_EQ(output.str(), "\r| done\r      \r");
}

TEST(UtilsTest, JoinThreadsJoinsEveryWorkerAndClearsTheVector) {
  std::atomic<size_t> completed{0};
  std::vector<std::thread> threads;
  threads.emplace_back([&completed]() { completed.fetch_add(1); });
  threads.emplace_back([&completed]() { completed.fetch_add(1); });

  join_threads(threads);

  EXPECT_EQ(completed.load(), 2U);
  EXPECT_TRUE(threads.empty());
}
