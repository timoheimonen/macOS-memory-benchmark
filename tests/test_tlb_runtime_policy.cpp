// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "benchmark/tlb_runtime_policy.h"
#include "core/config/constants.h"

TEST(TlbRuntimePolicyTest, ProfilesExposeQuickStandardAndExhaustiveBounds) {
  const TlbRuntimeProfile quick =
      tlb_runtime_profile_for_density(TlbSweepDensity::Low);
  const TlbRuntimeProfile standard =
      tlb_runtime_profile_for_density(TlbSweepDensity::Medium);
  const TlbRuntimeProfile exhaustive =
      tlb_runtime_profile_for_density(TlbSweepDensity::High);

  EXPECT_EQ(quick.name, "quick");
  EXPECT_EQ(standard.name, "standard");
  EXPECT_EQ(exhaustive.name, "exhaustive");
  EXPECT_GE(quick.min_rounds, 7u);
  EXPECT_LT(quick.max_rounds, standard.max_rounds);
  EXPECT_LT(standard.max_rounds, exhaustive.max_rounds);
  EXPECT_EQ(exhaustive.max_rounds, 30u);
}

TEST(TlbRuntimePolicyTest, CalibratedAccessesAreWholeCyclesAndRespectMinimum) {
  const TlbRuntimeProfile profile =
      tlb_runtime_profile_for_density(TlbSweepDensity::Medium);
  const size_t node_count = 1000;
  const size_t pilot_accesses = calculate_tlb_pilot_accesses(node_count);
  const size_t accesses = calculate_tlb_calibrated_accesses(
      node_count, pilot_accesses, 5.0 * pilot_accesses, profile);

  EXPECT_EQ(pilot_accesses % node_count, 0u);
  EXPECT_EQ(accesses % node_count, 0u);
  EXPECT_GE(accesses, node_count * profile.minimum_chain_cycles);
  EXPECT_LE(accesses, profile.maximum_accesses);
}

TEST(TlbRuntimePolicyTest, CalibrationRejectsInvalidPilotAndAvoidsOverflow) {
  const TlbRuntimeProfile profile =
      tlb_runtime_profile_for_density(TlbSweepDensity::Medium);
  EXPECT_EQ(calculate_tlb_calibrated_accesses(0, 100, 100.0, profile), 0u);
  EXPECT_EQ(calculate_tlb_calibrated_accesses(10, 0, 100.0, profile), 0u);
  EXPECT_EQ(calculate_tlb_calibrated_accesses(10, 100, 0.0, profile), 0u);
  EXPECT_EQ(calculate_tlb_pilot_accesses(
                std::numeric_limits<size_t>::max()),
            0u);
  EXPECT_EQ(calculate_tlb_calibrated_accesses(
                std::numeric_limits<size_t>::max(),
                100,
                100.0,
                profile),
            0u);
}

TEST(TlbRuntimePolicyTest, QuietSamplesConvergeAndNoisySamplesDoNot) {
  const TlbRuntimeProfile profile =
      tlb_runtime_profile_for_density(TlbSweepDensity::Low);
  std::vector<std::vector<double>> quiet(3);
  std::vector<std::vector<double>> noisy(3);
  for (size_t point = 0; point < 3; ++point) {
    for (size_t round = 0; round < profile.min_rounds; ++round) {
      quiet[point].push_back(1.0 + 0.01 * static_cast<double>(round % 2));
      noisy[point].push_back((round % 2) == 0 ? 0.0 : 4.0);
    }
  }

  const TlbConvergenceSummary quiet_summary =
      evaluate_tlb_convergence(quiet, profile, 123);
  const TlbConvergenceSummary repeated_summary =
      evaluate_tlb_convergence(quiet, profile, 123);
  const TlbConvergenceSummary noisy_summary =
      evaluate_tlb_convergence(noisy, profile, 123);

  EXPECT_TRUE(quiet_summary.converged);
  EXPECT_DOUBLE_EQ(quiet_summary.maximum_ci_width_ns,
                   repeated_summary.maximum_ci_width_ns);
  EXPECT_FALSE(noisy_summary.converged);
}

TEST(TlbRuntimePolicyTest, MemoryBudgetSelectsOnlySafePeakFootprints) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const size_t budget_mb = calculate_tlb_memory_budget_mb(2048);
  size_t peak_bytes = 0;

  EXPECT_EQ(budget_mb, 614u);
  EXPECT_TRUE(tlb_buffer_fits_memory_budget(
      512, page_size, budget_mb, peak_bytes));
  EXPECT_GT(peak_bytes, 512u * Constants::BYTES_PER_MB);
  EXPECT_FALSE(tlb_buffer_fits_memory_budget(
      1024, page_size, budget_mb, peak_bytes));

  EXPECT_EQ(calculate_tlb_memory_budget_mb(0), 384u);
  EXPECT_TRUE(tlb_buffer_fits_memory_budget(
      256, 4 * Constants::BYTES_PER_KB, 384, peak_bytes));
  EXPECT_FALSE(tlb_buffer_fits_memory_budget(
      256, page_size, calculate_tlb_memory_budget_mb(512), peak_bytes));
}

TEST(TlbRuntimePolicyTest, WorkEstimateIncludesPairsRoundsAndPeakMemory) {
  const TlbRuntimeProfile profile =
      tlb_runtime_profile_for_density(TlbSweepDensity::Medium);
  const TlbWorkEstimate estimate = estimate_tlb_work(
      15, 600 * Constants::BYTES_PER_MB, 1000, profile);

  EXPECT_EQ(estimate.point_count, 15u);
  EXPECT_EQ(estimate.min_rounds, profile.min_rounds);
  EXPECT_EQ(estimate.max_rounds, profile.max_rounds);
  EXPECT_EQ(estimate.maximum_accesses_per_measurement,
            profile.maximum_accesses);
  EXPECT_EQ(estimate.maximum_pointer_accesses,
            15u * profile.max_rounds * 2u *
                (estimate.maximum_pilot_accesses_per_measurement +
                 profile.maximum_accesses));
  EXPECT_GT(estimate.estimated_max_duration_sec,
            estimate.estimated_min_duration_sec);
  EXPECT_EQ(estimate.estimated_peak_memory_bytes,
            600u * Constants::BYTES_PER_MB);
}

TEST(TlbRuntimePolicyTest, WorkEstimateHonorsWholeChainCycleFloor) {
  const TlbRuntimeProfile profile =
      tlb_runtime_profile_for_density(TlbSweepDensity::Medium);
  const size_t node_count = profile.maximum_accesses;
  const TlbWorkEstimate estimate =
      estimate_tlb_work(1, 0, node_count, profile);

  EXPECT_EQ(estimate.maximum_accesses_per_measurement,
            node_count * profile.minimum_chain_cycles);
  EXPECT_EQ(estimate.maximum_pointer_accesses,
            profile.max_rounds * 2u *
                (estimate.maximum_pilot_accesses_per_measurement +
                 estimate.maximum_accesses_per_measurement));
}
