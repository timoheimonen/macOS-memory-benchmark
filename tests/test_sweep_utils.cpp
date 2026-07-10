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
 * @file test_sweep_utils.cpp
 * @brief Unit tests for shared sweep parsing and Cartesian counting.
 */

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config/sweep_utils.h"

namespace {

struct TestSweepSpec {
  std::vector<int> values;
};

void expect_sweep_parse_failure(const std::string& specification,
                                const std::string& expected_reason) {
  try {
    static_cast<void>(parse_sweep_text(specification));
    FAIL() << "Expected parse failure for: " << specification;
  } catch (const std::invalid_argument& error) {
    EXPECT_EQ(error.what(), expected_reason);
  }
}

}  // namespace

TEST(SweepUtilsTest, ParsesKeyAndCommaSeparatedValues) {
  const ParsedSweepText parsed = parse_sweep_text("threads=1,2,4");

  EXPECT_EQ(parsed.key, "threads");
  EXPECT_EQ(parsed.values, (std::vector<std::string>{"1", "2", "4"}));
}

TEST(SweepUtilsTest, PreservesTokensAndUsesOnlyTheFirstEqualsAsSeparator) {
  const ParsedSweepText parsed = parse_sweep_text(" key = value=with=equals, second ");

  EXPECT_EQ(parsed.key, " key ");
  EXPECT_EQ(parsed.values,
            (std::vector<std::string>{" value=with=equals", " second "}));
}

TEST(SweepUtilsTest, RejectsMissingKeyValueStructureWithExactReason) {
  constexpr const char* kExpectedReason = "sweep must use key=value1,value2 syntax";
  for (const std::string& specification : {"", "threads", "=1", "threads="}) {
    SCOPED_TRACE(specification);
    expect_sweep_parse_failure(specification, kExpectedReason);
  }
}

TEST(SweepUtilsTest, RejectsEmptyValueTokensWithExactReason) {
  constexpr const char* kExpectedReason =
      "sweep value list cannot contain empty values";
  for (const std::string& specification :
       {"threads=,1", "threads=1,", "threads=1,,2"}) {
    SCOPED_TRACE(specification);
    expect_sweep_parse_failure(specification, kExpectedReason);
  }
}

TEST(SweepUtilsTest, CalculatesCartesianProductAndIdentity) {
  EXPECT_EQ(calculate_cartesian_run_count({}), 1u);
  EXPECT_EQ(calculate_cartesian_run_count({2, 3, 4}), 24u);

  const std::vector<TestSweepSpec> specs = {{{1, 2}}, {{3, 4, 5}}};
  EXPECT_EQ(calculate_sweep_run_count_from_specs(specs), 6u);
}

TEST(SweepUtilsTest, EmptyDimensionProducesNoRuns) {
  EXPECT_EQ(calculate_cartesian_run_count({2, 0, 4}), 0u);

  const std::vector<TestSweepSpec> specs = {{{1, 2}}, {{}}, {{3, 4}}};
  EXPECT_EQ(calculate_sweep_run_count_from_specs(specs), 0u);
}

TEST(SweepUtilsTest, CartesianProductSaturatesOnOverflow) {
  const size_t overflowing_factor = std::numeric_limits<size_t>::max() / 2 + 1;
  EXPECT_EQ(calculate_cartesian_run_count({overflowing_factor, 2}),
            std::numeric_limits<size_t>::max());
}
