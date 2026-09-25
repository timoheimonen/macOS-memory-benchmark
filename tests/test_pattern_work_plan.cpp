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

#include <algorithm>
#include <limits>
#include <unordered_set>

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

TEST(PatternWorkPlanTest, IncludesLastExactlyFittingAccess) {
  const size_t buffer_size = Constants::PATTERN_STRIDE_PAGE + Constants::PATTERN_ACCESS_SIZE_BYTES;
  const PatternWorkPlan plan = build_plan(buffer_size, Constants::PATTERN_STRIDE_PAGE, 1);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  ASSERT_EQ(plan.workers.size(), 1u);
  EXPECT_EQ(plan.workers[0].span_bytes, buffer_size);
  EXPECT_EQ(plan.workers[0].accesses_per_pass, 2u);
  EXPECT_EQ(plan.accesses_per_pass, 2u);
  EXPECT_EQ(plan.min_accesses_per_pass, 2u);
  EXPECT_EQ(plan.max_accesses_per_pass, 2u);
  EXPECT_EQ(plan.payload_bytes_per_pass, 2u * Constants::PATTERN_ACCESS_SIZE_BYTES);
}

TEST(PatternWorkPlanTest, PartitionsUnevenBuffersAndReducesThreadsUntilEveryWorkerHasTransition) {
  struct SplitCase {
    size_t bytes;
    size_t stride;
    int requested;
    int effective;
  };
  for (const SplitCase& entry :
       {SplitCase{8 * Constants::BYTES_PER_MB, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, 10, 3},
        SplitCase{7 * Constants::BYTES_PER_MB + 123, Constants::PATTERN_STRIDE_PAGE_16K, 4, 4}}) {
    SCOPED_TRACE(entry.bytes);
    const size_t buffer_size = entry.bytes;
    const PatternWorkPlan plan = build_plan(buffer_size, entry.stride, entry.requested);

    ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
    EXPECT_EQ(plan.requested_threads, entry.requested);
    EXPECT_EQ(plan.effective_threads, entry.effective);
    ASSERT_EQ(plan.workers.size(), static_cast<size_t>(entry.effective));

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
    if (entry.requested == 10) EXPECT_EQ(plan.accesses_per_pass, 6u);
  }
}

TEST(PatternWorkPlanTest, UsesFinalWorkerCountsForPayloadAccounting) {
  const size_t minimum_payload = 256 * Constants::BYTES_PER_MB;
  const PatternWorkPlan plan = build_plan(512 * Constants::BYTES_PER_MB, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, 10,
                                          Constants::DEFAULT_ITERATIONS, minimum_payload);

  ASSERT_EQ(plan.status, PatternMeasurementStatus::Measured);
  EXPECT_EQ(plan.effective_threads, 10);
  EXPECT_EQ(plan.accesses_per_pass, 260u);
  EXPECT_EQ(plan.payload_bytes_per_pass, 260u * Constants::PATTERN_ACCESS_SIZE_BYTES);
  EXPECT_GE(plan.total_payload_bytes, minimum_payload);
  PatternWorkPlan previous = plan;
  ASSERT_GT(plan.passes, 1u);
  ASSERT_TRUE(set_strided_pattern_passes(previous, plan.passes - 1));
  EXPECT_LT(previous.total_payload_bytes, minimum_payload);
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

TEST(PatternWorkPlanTest, PartitionsRandomIndicesBeforeMeasurement) {
  const std::vector<size_t> global_indices = {0, 96, 128, 224, 225, 256};
  const std::vector<PatternRandomWorkerIndices> workers =
      build_random_worker_indices(256, Constants::PATTERN_ACCESS_SIZE_BYTES, 2, global_indices);

  ASSERT_EQ(workers.size(), 2u);
  EXPECT_EQ(workers[0].offset_bytes, 0u);
  EXPECT_EQ(workers[0].span_bytes, 128u);
  EXPECT_EQ(workers[0].indices, (std::vector<size_t>{0, 96}));
  EXPECT_EQ(workers[1].offset_bytes, 128u);
  EXPECT_EQ(workers[1].span_bytes, 128u);
  EXPECT_EQ(workers[1].indices, (std::vector<size_t>{0, 96}));
}

TEST(PatternWorkPlanTest, RejectsInvalidRandomPartitionParameters) {
  const std::vector<size_t> indices = {0, 32};
  EXPECT_TRUE(build_random_worker_indices(64, 0, 1, indices).empty());
  EXPECT_TRUE(build_random_worker_indices(64, 32, 0, indices).empty());
  EXPECT_TRUE(build_random_worker_indices(16, 32, 1, indices).empty());
}

TEST(PatternWorkPlanTest, RotatesPhaseAndCountsEveryAccessExactly) {
  PatternWorkPlan plan = build_plan(Constants::PATTERN_STRIDE_CACHE_LINE + Constants::PATTERN_ACCESS_SIZE_BYTES,
                                    Constants::PATTERN_STRIDE_CACHE_LINE, 1);
  ASSERT_TRUE(set_strided_pattern_passes(plan, 3));

  EXPECT_EQ(plan.phase_period_passes, 2u);
  EXPECT_EQ(plan.total_accesses, 5u);
  EXPECT_EQ(plan.total_payload_bytes, 5u * Constants::PATTERN_ACCESS_SIZE_BYTES);
  EXPECT_EQ(plan.distinct_address_count, 3u);
  EXPECT_EQ(plan.logical_working_set_bytes, 96u);
  EXPECT_EQ(plan.completed_phase_cycles, 1u);
  EXPECT_EQ(plan.min_accesses_per_pass, 1u);
  EXPECT_EQ(plan.max_accesses_per_pass, 2u);
}

TEST(PatternWorkPlanTest, FullSparsePhaseCycleCoversAlignedBufferSlots) {
  const size_t buffer_size = 8 * Constants::BYTES_PER_MB;
  PatternWorkPlan plan = build_plan(buffer_size, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, 10);
  const size_t phase_period = Constants::PATTERN_STRIDE_SUPERPAGE_2MB / Constants::PATTERN_ACCESS_SIZE_BYTES;
  ASSERT_TRUE(set_strided_pattern_passes(plan, phase_period));

  EXPECT_EQ(plan.completed_phase_cycles, 1u);
  EXPECT_EQ(plan.distinct_address_count, buffer_size / Constants::PATTERN_ACCESS_SIZE_BYTES);
  EXPECT_EQ(plan.total_accesses, plan.distinct_address_count);
  EXPECT_EQ(plan.logical_working_set_bytes, buffer_size);
}

TEST(PatternWorkPlanTest, RandomPermutationIsSeededUniqueAlignedAndBounded) {
  for (const size_t buffer_size : {size_t{4096}, size_t{96}}) {
    SCOPED_TRACE(buffer_size);
    const std::vector<size_t> first = generate_random_indices(buffer_size, 100, 42);
    const std::vector<size_t> repeated = generate_random_indices(buffer_size, 100, 42);
    const std::vector<size_t> different = generate_random_indices(buffer_size, 100, 43);

    EXPECT_EQ(first, repeated);
    if (buffer_size == 4096) EXPECT_NE(first, different);
    EXPECT_EQ(first.size(), std::min(size_t{100}, buffer_size / Constants::PATTERN_ACCESS_SIZE_BYTES));
    EXPECT_EQ(std::unordered_set<size_t>(first.begin(), first.end()).size(), first.size());
    for (size_t offset : first) {
      EXPECT_EQ(offset % Constants::PATTERN_ACCESS_SIZE_BYTES, 0u);
      EXPECT_LE(offset + Constants::PATTERN_ACCESS_SIZE_BYTES, buffer_size);
    }
  }
}
