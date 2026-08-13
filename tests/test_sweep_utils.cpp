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

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config/sweep_utils.h"

namespace {

struct TestSweepValues {
  size_t cardinality = 0;

  size_t size() const noexcept { return cardinality; }
};

struct TestSweepSpec {
  TestSweepValues values;
};

void expect_sweep_parse_failure(const std::string& specification, const std::string& expected_reason) {
  try {
    static_cast<void>(parse_sweep_text(specification));
    FAIL() << "Expected parse failure for: " << specification;
  } catch (const std::invalid_argument& error) {
    EXPECT_EQ(error.what(), expected_reason);
  }
}

}  // namespace

TEST(SweepUtilsTest, ParsesCommaSeparatedValuesAndPreservesTokens) {
  struct ParseCase {
    const char* specification;
    const char* expected_key;
    std::vector<std::string> expected_values;
  };

  const ParseCase cases[] = {
      {"threads=1,2,4", "threads", {"1", "2", "4"}},
      {" key = value=with=equals, second ", " key ", {" value=with=equals", " second "}},
  };

  for (const ParseCase& test_case : cases) {
    SCOPED_TRACE(test_case.specification);
    const ParsedSweepText parsed = parse_sweep_text(test_case.specification);
    EXPECT_EQ(parsed.key, test_case.expected_key);
    EXPECT_EQ(parsed.values, test_case.expected_values);
  }
}

TEST(SweepUtilsTest, RejectsMissingKeyValueStructureWithExactReason) {
  constexpr const char* kExpectedReason = "sweep must use key=value1,value2 syntax";
  for (const std::string& specification : {"", "threads", "=1", "threads="}) {
    SCOPED_TRACE(specification);
    expect_sweep_parse_failure(specification, kExpectedReason);
  }
}

TEST(SweepUtilsTest, RejectsEmptyValueTokensWithExactReason) {
  constexpr const char* kExpectedReason = "sweep value list cannot contain empty values";
  for (const std::string& specification : {"threads=,1", "threads=1,", "threads=1,,2"}) {
    SCOPED_TRACE(specification);
    expect_sweep_parse_failure(specification, kExpectedReason);
  }
}

TEST(SweepUtilsTest, CalculatesCartesianProductIdentityAndEmptyDimensions) {
  const std::vector<TestSweepSpec> no_specs;
  EXPECT_EQ(calculate_sweep_run_count_from_specs(no_specs), 1u);

  const std::vector<TestSweepSpec> specs = {{{2}}, {{3}}, {{4}}};
  EXPECT_EQ(calculate_sweep_run_count_from_specs(specs), 24u);

  const std::vector<TestSweepSpec> specs_with_empty_dimension = {{{2}}, {{0}}, {{4}}};
  EXPECT_EQ(calculate_sweep_run_count_from_specs(specs_with_empty_dimension), 0u);
}

TEST(SweepUtilsTest, CartesianProductSaturatesOnOverflow) {
  const size_t overflowing_factor = std::numeric_limits<size_t>::max() / 2 + 1;
  const std::vector<TestSweepSpec> specs = {{{overflowing_factor}}, {{2}}};
  EXPECT_EQ(calculate_sweep_run_count_from_specs(specs), std::numeric_limits<size_t>::max());
}
