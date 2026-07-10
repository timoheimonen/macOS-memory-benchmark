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
#include <gtest/gtest.h>

#include <limits>

#include "core/config/constants.h"
#include "pattern_benchmark/pattern_work_plan.h"

namespace {

PatternWorkPlan build_plan(size_t buffer_size, size_t stride, int threads, int base_passes = 1,
                           size_t minimum_payload = 0) {
  return build_strided_pattern_work_plan(buffer_size, stride, Constants::PATTERN_ACCESS_SIZE_BYTES, threads,
                                         base_passes, minimum_payload);
}

}  // namespace

TEST(PatternWorkPlanTest, RejectsInvalidParameters) {
  EXPECT_EQ(build_plan(4096, 0, 1).status, PatternMeasurementStatus::Invalid);
  EXPECT_EQ(build_plan(4096, 64, 0).status, PatternMeasurementStatus::Invalid);
  EXPECT_EQ(build_plan(4096, 64, 1, 0).status, PatternMeasurementStatus::Invalid);
}

TEST(PatternWorkPlanTest, SkipsBufferWithoutStrideTransition) {
  const PatternWorkPlan plan =
      build_plan(Constants::PATTERN_STRIDE_CACHE_LINE + Constants::PATTERN_ACCESS_SIZE_BYTES - 1,
                 Constants::PATTERN_STRIDE_CACHE_LINE, 1);

  EXPECT_EQ(plan.status, PatternMeasurementStatus::Skipped);
  EXPECT_EQ(plan.effective_threads, 0);
  EXPECT_TRUE(plan.workers.empty());
}

TEST(PatternWorkPlanTest, IncludesLastExactlyFittingAccess) {
  const size_t buffer_size = Constants::PATTERN_STRIDE_PAGE + Constants::PATTERN_ACCESS_SIZE_BYTES;
  const PatternWorkPlan plan = build_plan(buffer_size, Constants::PATTERN_STRIDE_PAGE, 1);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  ASSERT_EQ(plan.workers.size(), 1u);
  EXPECT_EQ(plan.workers[0].span_bytes, buffer_size);
  EXPECT_EQ(plan.workers[0].accesses_per_pass, 2u);
  EXPECT_EQ(plan.accesses_per_pass, 2u);
  EXPECT_EQ(plan.payload_bytes_per_pass, 2u * Constants::PATTERN_ACCESS_SIZE_BYTES);
}

TEST(PatternWorkPlanTest, ReducesThreadsUntilEveryWorkerHasTransition) {
  const size_t buffer_size = 8 * Constants::BYTES_PER_MB;
  const PatternWorkPlan plan = build_plan(buffer_size, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, 10);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  EXPECT_EQ(plan.requested_threads, 10);
  EXPECT_EQ(plan.effective_threads, 3);
  ASSERT_EQ(plan.workers.size(), 3u);

  size_t next_offset = 0;
  size_t summed_accesses = 0;
  size_t summed_payload = 0;
  for (const PatternWorkerRange& worker : plan.workers) {
    EXPECT_EQ(worker.offset_bytes, next_offset);
    EXPECT_GE(worker.accesses_per_pass, 2u);
    next_offset += worker.span_bytes;
    summed_accesses += worker.accesses_per_pass;
    summed_payload += worker.payload_bytes_per_pass;
  }
  EXPECT_EQ(next_offset, buffer_size);
  EXPECT_EQ(summed_accesses, plan.accesses_per_pass);
  EXPECT_EQ(summed_payload, plan.payload_bytes_per_pass);
  EXPECT_EQ(plan.accesses_per_pass, 6u);
}

TEST(PatternWorkPlanTest, UsesFinalWorkerCountsForPayloadAccounting) {
  const PatternWorkPlan plan = build_plan(512 * Constants::BYTES_PER_MB, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, 10,
                                          Constants::DEFAULT_ITERATIONS, Constants::PATTERN_STRIDED_MIN_TOUCHED_BYTES);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  EXPECT_EQ(plan.effective_threads, 10);
  EXPECT_EQ(plan.accesses_per_pass, 260u);
  EXPECT_EQ(plan.payload_bytes_per_pass, 260u * Constants::PATTERN_ACCESS_SIZE_BYTES);
  EXPECT_EQ(plan.total_accesses, plan.accesses_per_pass * plan.passes);
  EXPECT_EQ(plan.total_payload_bytes, plan.payload_bytes_per_pass * plan.passes);
}

TEST(PatternWorkPlanTest, PreservesExactCoverageForUnevenSplit) {
  const size_t buffer_size = (7 * Constants::BYTES_PER_MB) + 123;
  const PatternWorkPlan plan = build_plan(buffer_size, Constants::PATTERN_STRIDE_PAGE_16K, 4);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  size_t covered_bytes = 0;
  for (const PatternWorkerRange& worker : plan.workers) {
    EXPECT_EQ(worker.offset_bytes, covered_bytes);
    covered_bytes += worker.span_bytes;
  }
  EXPECT_EQ(covered_bytes, buffer_size);
}

TEST(PatternWorkPlanTest, RejectsPassCountBeyondExecutorLimit) {
  const PatternWorkPlan plan =
      build_plan(Constants::PATTERN_STRIDE_CACHE_LINE + Constants::PATTERN_ACCESS_SIZE_BYTES,
                 Constants::PATTERN_STRIDE_CACHE_LINE, 1, 1, std::numeric_limits<size_t>::max());

  EXPECT_EQ(plan.status, PatternMeasurementStatus::Invalid);
  EXPECT_TRUE(plan.workers.empty());
}

TEST(PatternWorkPlanTest, SerializesMeasurementStatus) {
  EXPECT_STREQ(pattern_measurement_status_to_string(PatternMeasurementStatus::Measured), "measured");
  EXPECT_STREQ(pattern_measurement_status_to_string(PatternMeasurementStatus::Skipped), "skipped");
  EXPECT_STREQ(pattern_measurement_status_to_string(PatternMeasurementStatus::Interrupted), "interrupted");
  EXPECT_STREQ(pattern_measurement_status_to_string(PatternMeasurementStatus::Invalid), "invalid");
}
