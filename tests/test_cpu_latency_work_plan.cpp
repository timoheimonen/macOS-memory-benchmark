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

#include <cstdint>
#include <vector>

#include "cpu_latency/cpu_latency_work_plan.h"

namespace {

CpuLatencySystemInfo make_sys_info(size_t avail_bytes = 8ULL * 1024 * 1024 * 1024) {
  CpuLatencySystemInfo sys;
  sys.l1_cache_bytes = 128ULL * 1024ULL;
  sys.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  sys.page_size_bytes = 16384ULL;
  sys.available_memory_bytes = avail_bytes;
  return sys;
}

TEST(CpuLatencyWorkPlanTest, CanonicalTargetOrderAndComponentIdentities) {
  CpuLatencyConfig cfg;
  cfg.buffer_size_mb = 4;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  cfg.output_file = "out.json";

  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.active_targets.size(), 3u);
  EXPECT_EQ(plan.active_targets[0], CpuLatencyTarget::L1);
  EXPECT_EQ(plan.active_targets[1], CpuLatencyTarget::L2);
  EXPECT_EQ(plan.active_targets[2], CpuLatencyTarget::MainMemory);

  EXPECT_EQ(plan.component_identities.version, "cpu-latency-components-v1");
  EXPECT_EQ(plan.component_identities.work_plan_version,
            "cpu-latency-work-plan-v1");
  EXPECT_EQ(plan.component_identities.chain_generator_version,
            "splitmix64-fisher-yates-rejection-v1");
  EXPECT_EQ(plan.component_identities.chain_version, "cpu-latency-chain-v1");
  EXPECT_EQ(plan.component_identities.run_policy_version,
            "bounded-loop-snapshots-v1");
  EXPECT_EQ(plan.plan_identity.rfind("cpu-latency-work-plan-v1|", 0), 0u);

  // Custom cache replaces L1/L2 in canonical position before MainMemory.
  cfg.custom_cache_size_kb = 64;
  const auto custom_plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(custom_plan.valid);
  ASSERT_EQ(custom_plan.active_targets.size(), 2u);
  EXPECT_EQ(custom_plan.active_targets[0], CpuLatencyTarget::Custom);
  EXPECT_EQ(custom_plan.active_targets[1], CpuLatencyTarget::MainMemory);
}

TEST(CpuLatencyWorkPlanTest, CompareLocalityBudgetAndWorkCapPreflight) {
  CpuLatencyConfig cfg;
  cfg.buffer_size_mb = 4;
  cfg.custom_cache_size_kb = 0;
  cfg.compare_locality = true;
  cfg.seed = 424242ULL;
  cfg.output_file = "loc.json";

  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_TRUE(plan.locality_16k_plan.has_value());
  EXPECT_TRUE(plan.locality_global_plan.has_value());
  // 4 MiB main buffer + 4 MiB compare buffer = 8 MiB committed.
  EXPECT_EQ(plan.memory_budget.committed_data_bytes, 8ULL * 1024ULL * 1024ULL);
  EXPECT_GT(plan.memory_budget.transient_peak_bytes, 0u);
  EXPECT_TRUE(plan.memory_budget.valid);

  // Tight memory budget rejects without silent capping.
  const auto over_plan = build_cpu_latency_work_plan(
      cfg, make_sys_info(2ULL * 1024ULL * 1024ULL));
  EXPECT_FALSE(over_plan.valid);
  EXPECT_FALSE(over_plan.memory_budget.valid);
  EXPECT_EQ(over_plan.reason_code,
            CpuLatencyWorkPlanReason::MEMORY_BUDGET_EXCEEDED);

  // Fallback budget activates when available_memory_bytes == 0.
  const auto fb_plan = build_cpu_latency_work_plan(cfg, make_sys_info(0));
  ASSERT_TRUE(fb_plan.valid);
  EXPECT_TRUE(fb_plan.memory_budget.used_fallback);
  EXPECT_EQ(fb_plan.memory_budget.allowed_memory_bytes,
            2048ULL * 1024ULL * 1024ULL);

  // Work cap exceeded when complete_chain_cycles * node_count > 4,000,000,000.
  const auto* main_tp = plan.find_target(CpuLatencyTarget::MainMemory);
  ASSERT_NE(main_tp, nullptr);
  const auto bad_exec = build_cpu_latency_target_execution_plan(
      *main_tp, kCpuLatencyMaxCliIterations, true);
  EXPECT_FALSE(bad_exec.valid);
  EXPECT_EQ(bad_exec.reason_code, CpuLatencyWorkPlanReason::WORK_CAP_EXCEEDED);
}

}  // namespace
