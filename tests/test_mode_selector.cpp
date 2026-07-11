// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/config/mode_selector.h"

namespace {

PrimaryModeSelection select(const std::vector<std::string>& arguments) {
  std::vector<std::string> storage = arguments;
  std::vector<char*> argv;
  argv.reserve(storage.size());
  for (std::string& argument : storage) {
    argv.push_back(argument.data());
  }
  return select_primary_benchmark_mode(static_cast<int>(argv.size()),
                                       argv.data());
}

}  // namespace

TEST(ModeSelectorTest, RecognizesEveryShortAndLongPrimaryMode) {
  EXPECT_EQ(select({"program", "-B"}).mode, PrimaryBenchmarkMode::Standard);
  EXPECT_EQ(select({"program", "--patterns"}).mode,
            PrimaryBenchmarkMode::Patterns);
  EXPECT_EQ(select({"program", "-T"}).mode,
            PrimaryBenchmarkMode::AnalyzeTlb);
  EXPECT_EQ(select({"program", "--analyze-core2core"}).mode,
            PrimaryBenchmarkMode::AnalyzeCoreToCore);
  EXPECT_EQ(select({"program", "-G"}).mode,
            PrimaryBenchmarkMode::GpuBandwidth);
}

TEST(ModeSelectorTest, DistinctModesConflictIndependentOfArgvOrder) {
  const PrimaryModeSelection gpu_first =
      select({"program", "--gpu-bandwidth", "--analyze-core2core"});
  const PrimaryModeSelection core_first =
      select({"program", "--analyze-core2core", "--gpu-bandwidth"});

  EXPECT_EQ(gpu_first.mode, PrimaryBenchmarkMode::Conflict);
  EXPECT_EQ(core_first.mode, PrimaryBenchmarkMode::Conflict);
  ASSERT_EQ(gpu_first.selected_options.size(), 2u);
  ASSERT_EQ(core_first.selected_options.size(), 2u);
}

TEST(ModeSelectorTest, RepeatedOneModeRemainsOwnedByItsParser) {
  const PrimaryModeSelection selection =
      select({"program", "-G", "--gpu-bandwidth"});

  EXPECT_EQ(selection.mode, PrimaryBenchmarkMode::GpuBandwidth);
  ASSERT_EQ(selection.selected_options.size(), 1u);
  EXPECT_EQ(selection.selected_options.front(), "--gpu-bandwidth");
}

TEST(ModeSelectorTest, OptionsWithoutPrimaryModeReturnNone) {
  EXPECT_EQ(select({"program", "--buffer-size", "512"}).mode,
            PrimaryBenchmarkMode::None);
}
