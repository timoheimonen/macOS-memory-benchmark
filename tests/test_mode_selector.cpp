// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "core/config/mode_selector.h"

namespace {

struct ModeCase {
  PrimaryBenchmarkMode mode;
  const char* short_option;
  const char* long_option;
};

constexpr std::array<ModeCase, 5> kModeCases{{
    {PrimaryBenchmarkMode::Standard, "-B", "--benchmark"},
    {PrimaryBenchmarkMode::Patterns, "-P", "--patterns"},
    {PrimaryBenchmarkMode::AnalyzeTlb, "-T", "--analyze-tlb"},
    {PrimaryBenchmarkMode::AnalyzeCoreToCore, "-C", "--analyze-core2core"},
    {PrimaryBenchmarkMode::GpuBandwidth, "-G", "--gpu-bandwidth"},
}};

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
  for (const ModeCase& mode_case : kModeCases) {
    for (const char* option : {mode_case.short_option,
                               mode_case.long_option}) {
      SCOPED_TRACE(option);
      EXPECT_EQ(select({"program", option}).mode, mode_case.mode);
    }
  }
}

TEST(ModeSelectorTest,
     EveryShortPrimaryModeSpellingIsOpaqueAfterEitherOutputOption) {
  for (const ModeCase& mode_case : kModeCases) {
    for (const char* output_option : {"-o", "--output"}) {
      SCOPED_TRACE(std::string(output_option) + " " +
                   mode_case.short_option);
      const PrimaryModeSelection selection =
          select({"program", output_option, mode_case.short_option});

      EXPECT_EQ(selection.mode, PrimaryBenchmarkMode::None);
      EXPECT_TRUE(selection.selected_options.empty());
    }
  }
}

TEST(ModeSelectorTest,
     EveryLongPrimaryModeSpellingIsOpaqueAfterLongOutputOption) {
  for (const ModeCase& mode_case : kModeCases) {
    SCOPED_TRACE(mode_case.long_option);
    const PrimaryModeSelection selection =
        select({"program", "--output", mode_case.long_option});

    EXPECT_EQ(selection.mode, PrimaryBenchmarkMode::None);
    EXPECT_TRUE(selection.selected_options.empty());
  }
}

TEST(ModeSelectorTest, OutputMayAppearBeforeOrAfterTheActualSelectedMode) {
  for (size_t mode_index = 0; mode_index < kModeCases.size(); ++mode_index) {
    const ModeCase& actual_mode = kModeCases[mode_index];
    const ModeCase& output_value =
        kModeCases[(mode_index + 1) % kModeCases.size()];
    SCOPED_TRACE(actual_mode.long_option);

    const PrimaryModeSelection output_first =
        select({"program", "--output", output_value.short_option,
                actual_mode.long_option});
    const PrimaryModeSelection mode_first =
        select({"program", actual_mode.long_option, "-o",
                output_value.short_option});

    EXPECT_EQ(output_first.mode, actual_mode.mode);
    EXPECT_EQ(mode_first.mode, actual_mode.mode);
    ASSERT_EQ(output_first.selected_options.size(), 1u);
    ASSERT_EQ(mode_first.selected_options.size(), 1u);
    EXPECT_EQ(output_first.selected_options.front(), actual_mode.long_option);
    EXPECT_EQ(mode_first.selected_options.front(), actual_mode.long_option);
  }
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

TEST(ModeSelectorTest,
     RealModeOutsideConsumedOutputValueStillConflictsInEitherOrder) {
  const PrimaryModeSelection benchmark_first =
      select({"program", "--output", "-P", "--benchmark",
              "--gpu-bandwidth"});
  const PrimaryModeSelection gpu_first =
      select({"program", "--gpu-bandwidth", "--output", "-P",
              "--benchmark"});

  EXPECT_EQ(benchmark_first.mode, PrimaryBenchmarkMode::Conflict);
  EXPECT_EQ(gpu_first.mode, PrimaryBenchmarkMode::Conflict);
  ASSERT_EQ(benchmark_first.selected_options.size(), 2u);
  ASSERT_EQ(gpu_first.selected_options.size(), 2u);
  EXPECT_EQ(benchmark_first.selected_options[0], "--benchmark");
  EXPECT_EQ(benchmark_first.selected_options[1], "--gpu-bandwidth");
  EXPECT_EQ(gpu_first.selected_options[0], "--gpu-bandwidth");
  EXPECT_EQ(gpu_first.selected_options[1], "--benchmark");
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

TEST(ModeSelectorTest, MissingOutputValueRemainsOwnedByTheParser) {
  for (const char* output_option : {"-o", "--output"}) {
    SCOPED_TRACE(output_option);
    const PrimaryModeSelection no_selected_mode =
        select({"program", output_option});
    const PrimaryModeSelection selected_mode =
        select({"program", "--benchmark", output_option});

    EXPECT_EQ(no_selected_mode.mode, PrimaryBenchmarkMode::None);
    EXPECT_TRUE(no_selected_mode.selected_options.empty());
    EXPECT_EQ(selected_mode.mode, PrimaryBenchmarkMode::Standard);
    ASSERT_EQ(selected_mode.selected_options.size(), 1u);
    EXPECT_EQ(selected_mode.selected_options.front(), "--benchmark");
  }
}
