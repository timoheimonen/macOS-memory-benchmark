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

#include <vector>

#include "core/audit/deterministic_rng.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"

namespace {

CpuBandwidthSystemInfo make_sys_info(size_t avail_gb = 16) {
  CpuBandwidthSystemInfo info;
  info.l1_cache_bytes = 128ULL * 1024ULL;
  info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  info.page_size_bytes = 16384ULL;
  info.available_memory_bytes = avail_gb * 1024ULL * 1024ULL * 1024ULL;
  return info;
}

TEST(CpuBandwidthWorkPlanTest, CanonicalTargetOrderAndWorkerClamping) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 16;
  cfg.requested_workers = 10;
  cfg.available_workers = 10;
  cfg.user_specified_workers = false;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  cfg.output_file = "out.json";

  const CpuBandwidthWorkPlan plan =
      build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.active_targets.size(), 3u);
  EXPECT_EQ(plan.active_targets[0], CpuBandwidthTarget::MainMemory);
  EXPECT_EQ(plan.active_targets[1], CpuBandwidthTarget::L1);
  EXPECT_EQ(plan.active_targets[2], CpuBandwidthTarget::L2);

  // Auto worker source uses all available for MainMemory, 1 for caches.
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::MainMemory)->effective_workers,
            10u);
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::L1)->effective_workers, 1u);
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::L2)->effective_workers, 1u);

  // Explicit excessive threads (-t 9999) clamps effective_workers while
  // preserving config.requested_workers == 9999.
  cfg.user_specified_workers = true;
  cfg.requested_workers = 9999;
  cfg.custom_cache_size_kb = 16;
  const CpuBandwidthWorkPlan clamped =
      build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(clamped.valid) << clamped.reason_code;
  ASSERT_EQ(clamped.active_targets.size(), 2u);
  EXPECT_EQ(clamped.active_targets[0], CpuBandwidthTarget::MainMemory);
  EXPECT_EQ(clamped.active_targets[1], CpuBandwidthTarget::Custom);
  EXPECT_EQ(clamped.find_target(CpuBandwidthTarget::MainMemory)->effective_workers,
            10u);
  EXPECT_EQ(clamped.find_target(CpuBandwidthTarget::Custom)->effective_workers,
            10u);
}

TEST(CpuBandwidthWorkPlanTest, Partition64bPreservesAlignmentAndContiguity) {
  const std::vector<size_t> sizes = {63, 64, 65, 127, 128, 193, 16384, 524288};
  const std::vector<size_t> workers_list = {1, 2, 3, 4, 10};
  for (size_t sz : sizes) {
    for (size_t w : workers_list) {
      const auto spans = partition_cpu_bandwidth_64b(sz, w);
      ASSERT_EQ(spans.size(), w);
      size_t cursor = 0;
      for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i].span_bytes > 0) {
          EXPECT_EQ(spans[i].offset_bytes, cursor);
          if (i > 0 && sz >= w * 64) {
            EXPECT_EQ(spans[i].offset_bytes % 64u, 0u);
          }
          cursor += spans[i].span_bytes;
        }
      }
      EXPECT_EQ(cursor, sz);
    }
  }
}

TEST(CpuBandwidthWorkPlanTest, StrictMemoryBudgetRejectsOverflowWithoutSilentCapping) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 999999999ULL;
  cfg.custom_cache_size_kb = 0;
  cfg.requested_workers = 4;
  cfg.available_workers = 4;
  cfg.iterations = 1;
  cfg.loop_count = 1;
  cfg.output_file = "out.json";

  const CpuBandwidthWorkPlan plan =
      build_cpu_bandwidth_work_plan(cfg, make_sys_info(16));
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason_code, CpuBandwidthWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_FALSE(plan.memory_budget.valid);
  EXPECT_EQ(plan.memory_budget.runtime_peak_bytes,
            plan.memory_budget.known_owned_peak_bytes);
  // Target bytes were NOT silently truncated!
  ASSERT_NE(plan.find_target(CpuBandwidthTarget::MainMemory), nullptr);
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::MainMemory)->bytes,
            999999999ULL * 1024ULL * 1024ULL);
}

TEST(CpuBandwidthWorkPlanTest, FallbackBudgetAndWorkCaps) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 16;
  cfg.custom_cache_size_kb = 0;
  cfg.requested_workers = 2;
  cfg.available_workers = 2;
  cfg.loop_count = 1;

  CpuBandwidthSystemInfo sys = make_sys_info(0);
  const CpuBandwidthWorkPlan fallback_plan =
      build_cpu_bandwidth_work_plan(cfg, sys);
  ASSERT_TRUE(fallback_plan.valid);
  EXPECT_TRUE(fallback_plan.memory_budget.used_fallback);
  EXPECT_EQ(fallback_plan.memory_budget.allowed_memory_bytes,
            Constants::FALLBACK_TOTAL_LIMIT_MB * Constants::BYTES_PER_MB);

  // 64 GiB payload cap check.
  cfg.buffer_size_mb = 1024;
  cfg.iterations = 65;  // 65 * 1 GiB = 65 GiB > 64 GiB
  const CpuBandwidthWorkPlan capped =
      build_cpu_bandwidth_work_plan(cfg, make_sys_info(16));
  EXPECT_FALSE(capped.valid);
  EXPECT_EQ(capped.reason_code, CpuBandwidthWorkPlanReason::WORK_CAP_EXCEEDED);
}

TEST(CpuBandwidthWorkPlanTest, DomainSeparatedSeedsAndComponentIdentities) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 2;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.iterations = 2;
  cfg.loop_count = 3;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;

  const CpuBandwidthWorkPlan plan =
      build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.component_identities.version, "cpu-bandwidth-components-v1");
  EXPECT_EQ(plan.component_identities.work_plan_version,
            "cpu-bandwidth-work-plan-v1");
  EXPECT_EQ(plan.component_identities.buffer_pattern_version,
            "cpu-bw-buffer-pattern-v1");
  EXPECT_EQ(plan.component_identities.read_checksum_version,
            "cpu-bw-read-checksum-v1");
  EXPECT_EQ(plan.component_identities.run_policy_version,
            "bounded-loop-snapshots-v1");
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::MainMemory)->src_seed,
            splitmix64_mix(424242ULL ^ kCpuBandwidthDomainMainMemory));
  EXPECT_EQ(plan.find_target(CpuBandwidthTarget::Custom)->src_seed,
            splitmix64_mix(424242ULL ^ kCpuBandwidthDomainCustom));
  EXPECT_EQ(plan.pre_copy_sentinel_seed,
            splitmix64_mix(424242ULL ^ kCpuBandwidthDomainPreCopySentinel));
}

}  // namespace
