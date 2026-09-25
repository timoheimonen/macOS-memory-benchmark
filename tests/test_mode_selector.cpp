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

constexpr std::array<ModeCase, 6> kModeCases{{
    {PrimaryBenchmarkMode::Standard, "-B", "--benchmark"},
    {PrimaryBenchmarkMode::Patterns, "-P", "--patterns"},
    {PrimaryBenchmarkMode::AnalyzeTlb, "-T", "--analyze-tlb"},
    {PrimaryBenchmarkMode::AnalyzeCoreToCore, "-C", "--analyze-core2core"},
    {PrimaryBenchmarkMode::GpuBandwidth, "-G", "--gpu-bandwidth"},
    {PrimaryBenchmarkMode::LlmMemory, "-M", "--llm-memory"},
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

TEST(ModeSelectorTest, EveryPrimaryModeSpellingIsOpaqueAfterEitherOutputOption) {
  for (const ModeCase& mode_case : kModeCases) {
    for (const char* mode_option : {mode_case.short_option, mode_case.long_option}) {
      for (const char* output_option : {"-o", "--output"}) {
        SCOPED_TRACE(std::string(output_option) + " " + mode_option);
        const PrimaryModeSelection selection = select({"program", output_option, mode_option});
        EXPECT_EQ(selection.mode, PrimaryBenchmarkMode::None);
        EXPECT_TRUE(selection.selected_options.empty());
      }
    }
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
  for (size_t first_index = 0; first_index < kModeCases.size();
       ++first_index) {
    for (size_t second_index = first_index + 1;
         second_index < kModeCases.size(); ++second_index) {
      const ModeCase& first = kModeCases[first_index];
      const ModeCase& second = kModeCases[second_index];
      for (bool reverse : {false, true}) {
        const ModeCase& argv_first = reverse ? second : first;
        const ModeCase& argv_second = reverse ? first : second;
        SCOPED_TRACE(std::string(argv_first.long_option) + " " +
                     argv_second.long_option);
        const PrimaryModeSelection selection =
            select({"program", argv_first.short_option,
                    argv_second.long_option});

        EXPECT_EQ(selection.mode, PrimaryBenchmarkMode::Conflict);
        ASSERT_EQ(selection.selected_options.size(), 2u);
        EXPECT_EQ(selection.selected_options[0], argv_first.long_option);
        EXPECT_EQ(selection.selected_options[1], argv_second.long_option);
      }
    }
  }
}

TEST(ModeSelectorTest, RepeatedOneModeRemainsOwnedByItsParser) {
  for (const ModeCase& mode_case : kModeCases) {
    SCOPED_TRACE(mode_case.long_option);
    const PrimaryModeSelection selection =
        select({"program", mode_case.short_option, mode_case.long_option});

    EXPECT_EQ(selection.mode, mode_case.mode);
    ASSERT_EQ(selection.selected_options.size(), 1u);
    EXPECT_EQ(selection.selected_options.front(), mode_case.long_option);
  }
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
