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

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <vector>

#include "asm/asm_functions.h"
#include "core/audit/checkpoint_policy.h"
#include "core/memory/memory_manager.h"
#include "core/timing/timer.h"
#include "cpu_latency/cpu_latency_executor.h"
#include "cpu_latency/cpu_latency_runner.h"

namespace {

CpuLatencyConfig make_small_config(std::optional<size_t> iterations = 2,
                                   size_t loops = 3,
                                   size_t samples = 4,
                                   bool compare_locality = false) {
  CpuLatencyConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.custom_cache_size_kb = 16;
  cfg.iterations = iterations;
  cfg.user_specified_iterations = iterations.has_value();
  cfg.loop_count = loops;
  cfg.latency_samples = samples;
  cfg.latency_stride_bytes = 256;
  cfg.latency_chain_mode = LatencyChainMode::Auto;
  cfg.latency_tlb_locality_kb = 16;
  cfg.compare_locality = compare_locality;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  return cfg;
}

CpuLatencySystemInfo make_sys_info() {
  CpuLatencySystemInfo info;
  info.l1_cache_bytes = 128ULL * 1024ULL;
  info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  info.page_size_bytes = 16384ULL;
  info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  return info;
}

int fail_madvise_einval(void*, size_t, int) {
  errno = EINVAL;
  return -1;
}

int fail_mprotect_eacces(void*, size_t, int) {
  errno = EACCES;
  return -1;
}

std::atomic<uint64_t> g_mock_ticks{1000000ULL};
uint64_t g_mock_step_ticks = 1000000ULL;

uint64_t mock_absolute_time() {
  return g_mock_ticks.fetch_add(g_mock_step_ticks, std::memory_order_relaxed);
}

kern_return_t mock_timebase_info(mach_timebase_info_t info) {
  info->numer = 125;
  info->denom = 3;
  return KERN_SUCCESS;
}

class CpuLatencyExecutorRunnerTest : public ::testing::Test {
 protected:
  void TearDown() override {
    reset_memory_system_calls_for_testing();
    reset_timer_system_calls_for_testing();
  }
};

TEST_F(CpuLatencyExecutorRunnerTest, MadviseNonFatalAndMprotectFailureDetected) {
  const auto cfg = make_small_config(2, 1, 4, false);
  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  // Non-fatal madvise failure is recorded in AllocatedBufferEvidence.
  MemorySystemCalls madvise_fail;
  madvise_fail.advise = fail_madvise_einval;
  set_memory_system_calls_for_testing(madvise_fail);

  CpuLatencyResourceSet res;
  ASSERT_TRUE(prepare_cpu_latency_resources(plan, true, res));
  const auto& custom_buf =
      res.targets[cpu_latency_target_index(CpuLatencyTarget::Custom)];
  EXPECT_FALSE(custom_buf.buffer.madvise_applied);
  EXPECT_EQ(custom_buf.buffer.madvise_errno, EINVAL);
  EXPECT_TRUE(custom_buf.buffer.read_only_protected);
  EXPECT_TRUE(custom_buf.hamiltonian_prevalidated);
  release_cpu_latency_resources(res);

  // mprotect failure fails resource preparation cleanly.
  MemorySystemCalls mprotect_fail;
  mprotect_fail.protect = fail_mprotect_eacces;
  set_memory_system_calls_for_testing(mprotect_fail);
  EXPECT_FALSE(prepare_cpu_latency_resources(plan, false, res));
  EXPECT_EQ(res.reason_code, CpuLatencyExecutorReason::CHAIN_READONLY_FAILED);
  release_cpu_latency_resources(res);
}

TEST_F(CpuLatencyExecutorRunnerTest, ExecutorFaultAndPointerCorruptionInjection) {
  const auto cfg = make_small_config(2, 1, 4, false);
  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  CpuLatencyResourceSet res;
  ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));
  const auto& custom_tplan = *plan.find_target(CpuLatencyTarget::Custom);
  const auto& custom_buf =
      res.targets[cpu_latency_target_index(CpuLatencyTarget::Custom)];
  const auto exec_plan =
      build_cpu_latency_target_execution_plan(custom_tplan, 2, true);
  ASSERT_TRUE(exec_plan.valid);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  // 1. Hamiltonian prevalidation failure injection.
  {
    CpuLatencyExecutorTestControl ctrl;
    ctrl.force_hamiltonian_fail = true;
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *timer_opt, 4, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code,
              CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED);
    EXPECT_EQ(ex.validation_checks[1].valid, std::optional<bool>(false));
  }

  // 2. Kernel failure injection.
  {
    CpuLatencyExecutorTestControl ctrl;
    ctrl.force_kernel_fail = true;
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *timer_opt, 4, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, CpuLatencyExecutorReason::KERNEL_FAILED);
  }

  // 3. Zero elapsed time injection.
  {
    CpuLatencyExecutorTestControl ctrl;
    ctrl.force_zero_elapsed = true;
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *timer_opt, 4, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, CpuLatencyExecutorReason::INVALID_ELAPSED_TIME);
  }

  // 4. Main final-pointer corruption detected cleanly.
  {
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t) -> uintptr_t* {
      return chain_node_to_pointer(ptr, custom_tplan.stride_bytes, 1);
    };
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *timer_opt, 4, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code,
              CpuLatencyExecutorReason::MAIN_FINAL_POINTER_FAILED);
    EXPECT_EQ(ex.final_pointer.actual_node, std::optional<size_t>(1u));
    EXPECT_EQ(ex.final_pointer.valid, std::optional<bool>(false));
    EXPECT_EQ(ex.validation_checks[3].valid, std::optional<bool>(false));
  }

  // 5. Sample-window end-pointer corruption detected cleanly while main passes.
  {
    g_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
    g_mock_step_ticks = 1000ULL;
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = mock_absolute_time;
    timer_calls.timebase_info = mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);
    auto mock_timer_opt = HighResTimer::create();
    ASSERT_TRUE(mock_timer_opt.has_value());

    int call_num = 0;
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      ++call_num;
      if (call_num == 1) {
        return ptr;  // Main pass returns node 0 as expected.
      }
      // First sample window returns corrupted node 0 when steps != full cycle.
      if (call_num == 2) {
        return static_cast<uintptr_t*>(custom_buf.buffer.ptr.get());
      }
      (void)steps;
      return ptr;
    };
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *mock_timer_opt, 3, &ctrl);
    reset_timer_system_calls_for_testing();
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code,
              CpuLatencyExecutorReason::SAMPLE_WINDOW_END_POINTER_FAILED);
    ASSERT_TRUE(ex.sample_windows.has_value());
    EXPECT_FALSE(ex.sample_windows->validation_valid);
    EXPECT_FALSE(ex.sample_windows->all_end_pointers_valid);
    EXPECT_EQ(ex.validation_checks[4].valid, std::optional<bool>(false));
  }

  // 6. Valid execution with -n 0 marks sample-window check not-applicable.
  {
    const auto ex = execute_cpu_latency_attempt(
        custom_tplan, exec_plan, custom_buf, *timer_opt, 0, nullptr);
    EXPECT_TRUE(ex.valid);
    EXPECT_FALSE(ex.sample_windows.has_value());
    EXPECT_FALSE(ex.validation_checks[4].applicable);
    EXPECT_FALSE(ex.validation_checks[4].evaluated.has_value());
  }

  release_cpu_latency_resources(res);
}

TEST_F(CpuLatencyExecutorRunnerTest, AutoCalibrationScalesAndFreezesBeforeLoopZero) {
  // Mock Mach timer so 1 pilot call = 0.04167s (1,000,000 ticks * 125 / 3 ns)
  // -> scales up to ~6x, then hits target window cleanly.
  g_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
  g_mock_step_ticks = 1000000ULL;
  TimerSystemCalls timer_calls;
  timer_calls.absolute_time = mock_absolute_time;
  timer_calls.timebase_info = mock_timebase_info;
  set_timer_system_calls_for_testing(timer_calls);

  const auto cfg = make_small_config(std::nullopt, 2, 4, false);
  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  CpuLatencyResourceSet res;
  ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

  CpuLatencyResult result;
  CpuLatencyRunnerHooks hooks;
  const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
  EXPECT_EQ(rc, EXIT_SUCCESS);
  EXPECT_EQ(result.status, CpuLatencyRunStatus::Complete);
  EXPECT_TRUE(result.run_accepted);
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.target_order_balance_complete);

  for (CpuLatencyTarget t : plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(t);
    ASSERT_TRUE(result.frozen_plan_refs[t_idx].has_value());
    const auto& attempts = result.calibration_attempts[t_idx];
    ASSERT_GE(attempts.size(), 3u);
    EXPECT_EQ(attempts[0].purpose, "calibration_shape_warmup");
    EXPECT_EQ(attempts[1].purpose, "pilot");
    EXPECT_EQ(attempts.back().purpose, "frozen_measurement_warmup");
  }

  // Loop 0 order: [custom, main_memory]; Loop 1 order: [main_memory, custom].
  ASSERT_EQ(result.loops.size(), 2u);
  EXPECT_EQ(result.loops[0].planned_order,
            (std::vector<CpuLatencyTarget>{CpuLatencyTarget::Custom,
                                           CpuLatencyTarget::MainMemory}));
  EXPECT_EQ(result.loops[1].planned_order,
            (std::vector<CpuLatencyTarget>{CpuLatencyTarget::MainMemory,
                                           CpuLatencyTarget::Custom}));

  release_cpu_latency_resources(res);
}

TEST_F(CpuLatencyExecutorRunnerTest, InterruptionAndBoundedCheckpointFailureBehavior) {
  // 1. Mid-run interruption cleanly stops prefix and marks remaining rows interrupted.
  {
    const auto cfg = make_small_config(2, 4, 2, false);
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);

    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> stop{false};
    CpuLatencyRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_measurement_hook = [&](size_t loop_idx, size_t pos) {
      if (loop_idx == 1 && pos == 1) {
        stop.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 3u);
    EXPECT_EQ(result.measurements[3].status,
              CpuLatencyMeasurementStatus::Interrupted);
    EXPECT_FALSE(result.measurements[3].attempted);
    release_cpu_latency_resources(res);
  }

  // 2. Bounded checkpoints (count = 10 -> interval = ceil(10/8) = 2) & terminal checkpoint failure.
  {
    const auto cfg = make_small_config(1, 10, 0, false);
    EXPECT_EQ(calculate_bounded_snapshot_interval_loops(cfg.loop_count), 2u);
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);

    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    int progress_calls = 0;
    int terminal_calls = 0;
    CpuLatencyRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.checkpoint = [&](const CpuLatencyResult&,
                           CpuLatencyCheckpointKind kind) {
      if (kind == CpuLatencyCheckpointKind::Progress) {
        ++progress_calls;
        return EXIT_SUCCESS;
      }
      ++terminal_calls;
      return EXIT_FAILURE;  // Terminal failure must NOT be retried.
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_FAILURE);
    // Intermediate loops 1,3,5,7 -> 4 progress snapshots (loop 9 is terminal).
    EXPECT_EQ(progress_calls, 4);
    EXPECT_EQ(terminal_calls, 1);
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Failed);
    EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
    EXPECT_FALSE(result.run_accepted);
    release_cpu_latency_resources(res);
  }
}

TEST_F(CpuLatencyExecutorRunnerTest, CompareLocalityRunsThreePairedRoundsPerLoop) {
  const auto cfg = make_small_config(2, 2, 4, true);
  const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  CpuLatencyResourceSet res;
  ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

  CpuLatencyResult result;
  CpuLatencyRunnerHooks hooks;
  const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
  EXPECT_EQ(rc, EXIT_SUCCESS);
  EXPECT_EQ(result.status, CpuLatencyRunStatus::Complete);
  EXPECT_TRUE(result.run_accepted);
  EXPECT_EQ(result.locality_rounds.size(), 6u);
  EXPECT_TRUE(result.locality_aggregate.locality_16k_latency_ns.has_value());
  EXPECT_TRUE(result.locality_aggregate.global_random_latency_ns.has_value());
  EXPECT_TRUE(result.locality_aggregate.locality_latency_delta_ns.has_value());
  release_cpu_latency_resources(res);
}

TEST_F(CpuLatencyExecutorRunnerTest,
       CompareLocalityLoopZeroInterruptionLeavesAllMeasurementsInterrupted) {
  auto cfg = make_small_config(2, 2, 2, true);
  cfg.custom_cache_size_kb = 0;

  // 1. Interrupt at before_locality_round_hook(0, 1) after round 0 completes.
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> stop{false};
    CpuLatencyRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 0 && r_idx == 1) {
        stop.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 0u);
    EXPECT_EQ(result.counters.attempted_loops, 0u);
    EXPECT_EQ(result.locality_rounds.size(), 1u);
    ASSERT_EQ(result.loops.size(), 2u);
    EXPECT_TRUE(result.loops[0].realized_order.empty());
    EXPECT_TRUE(result.loops[1].realized_order.empty());
    ASSERT_EQ(result.measurements.size(), 2u);
    for (const auto& row : result.measurements) {
      EXPECT_EQ(row.status, CpuLatencyMeasurementStatus::Interrupted);
      EXPECT_FALSE(row.attempted);
    }
    release_cpu_latency_resources(res);
  }

  // 2. Mid-round interruption between locality_16k and locality_global in loop 0 round 0.
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> armed{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (armed.exchange(false)) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };

    CpuLatencyRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 0 && r_idx == 0) {
        armed.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 0u);
    EXPECT_EQ(result.counters.attempted_loops, 0u);
    ASSERT_EQ(result.locality_rounds.size(), 1u);
    EXPECT_EQ(result.locality_rounds[0].locality_16k.status, "measured");
    EXPECT_EQ(result.locality_rounds[0].locality_global.status, "interrupted");
    EXPECT_FALSE(result.locality_rounds[0].locality_global.attempted);
    EXPECT_FALSE(
        result.locality_aggregate.locality_16k_latency_ns.has_value());
    EXPECT_FALSE(
        result.locality_aggregate.global_random_latency_ns.has_value());
    EXPECT_FALSE(
        result.locality_aggregate.locality_latency_delta_ns.has_value());
    EXPECT_TRUE(result.loops[0].realized_order.empty());
    for (const auto& row : result.measurements) {
      EXPECT_EQ(row.status, CpuLatencyMeasurementStatus::Interrupted);
      EXPECT_FALSE(row.attempted);
    }
    release_cpu_latency_resources(res);
  }
}

TEST_F(CpuLatencyExecutorRunnerTest,
       CompareLocalityFinalLoopInterruptionStopsBeforeMainMeasurements) {
  auto cfg = make_small_config(2, 2, 2, true);
  cfg.custom_cache_size_kb = 0;

  // 1. Challenger stop_mid_locality: stop at before_locality_round_hook(1, 1).
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> stop{false};
    CpuLatencyRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 1 && r_idx == 1) {
        stop.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.planned_measurements, 2u);
    EXPECT_EQ(result.counters.measured_measurements, 1u);
    EXPECT_EQ(result.counters.attempted_loops, 1u);
    EXPECT_EQ(result.locality_rounds.size(), 4u);
    ASSERT_EQ(result.loops.size(), 2u);
    EXPECT_EQ(result.loops[0].realized_order.size(), 1u);
    EXPECT_TRUE(result.loops[1].realized_order.empty());
    ASSERT_EQ(result.measurements.size(), 2u);
    EXPECT_EQ(result.measurements[0].status,
              CpuLatencyMeasurementStatus::Measured);
    EXPECT_EQ(result.measurements[1].status,
              CpuLatencyMeasurementStatus::Interrupted);
    EXPECT_FALSE(result.measurements[1].attempted);
    release_cpu_latency_resources(res);
  }

  // 2. Mid-round interruption between locality_16k and locality_global in loop 1 round 1.
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> armed{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (armed.exchange(false)) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };

    CpuLatencyRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 1 && r_idx == 1) {
        armed.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 1u);
    EXPECT_EQ(result.counters.attempted_loops, 1u);
    ASSERT_EQ(result.locality_rounds.size(), 5u);
    EXPECT_EQ(result.locality_rounds[4].locality_16k.status, "measured");
    EXPECT_EQ(result.locality_rounds[4].locality_global.status, "interrupted");
    EXPECT_FALSE(result.locality_rounds[4].locality_global.attempted);
    EXPECT_TRUE(result.loops[1].realized_order.empty());
    EXPECT_EQ(result.measurements[1].status,
              CpuLatencyMeasurementStatus::Interrupted);
    EXPECT_FALSE(result.measurements[1].attempted);
    release_cpu_latency_resources(res);
  }
}

TEST_F(CpuLatencyExecutorRunnerTest,
       CompareLocalityFinalLoopFailureLeavesFinalMeasurementFailed) {
  auto cfg = make_small_config(2, 2, 2, true);
  cfg.custom_cache_size_kb = 0;

  // 1. Hamiltonian prevalidation failure triggered at before_locality_round_hook(1, 1).
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    CpuLatencyExecutorTestControl ctrl;
    CpuLatencyRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 1 && r_idx == 1) {
        ctrl.force_hamiltonian_fail = true;
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_FAILURE);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Failed);
    EXPECT_EQ(result.reason_code,
              CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 1u);
    ASSERT_EQ(result.locality_rounds.size(), 5u);
    EXPECT_EQ(result.locality_rounds[4].locality_16k.status, "failed");
    EXPECT_EQ(result.measurements[0].status,
              CpuLatencyMeasurementStatus::Measured);
    EXPECT_EQ(result.measurements[1].status,
              CpuLatencyMeasurementStatus::Failed);
    EXPECT_FALSE(result.measurements[1].attempted);
    release_cpu_latency_resources(res);
  }

  // 2. Pointer corruption triggered at before_locality_round_hook(1, 1).
  {
    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    g_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
    g_mock_step_ticks = 1000ULL;
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = mock_absolute_time;
    timer_calls.timebase_info = mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    CpuLatencyResult result;
    std::atomic<bool> corrupt{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (corrupt.load()) {
        return chain_node_to_pointer(ptr, plan.locality_16k_plan->stride_bytes,
                                     1);
      }
      return memory_latency_chase_asm(ptr, steps);
    };

    CpuLatencyRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    hooks.before_locality_round_hook = [&](size_t loop_idx, size_t r_idx) {
      if (loop_idx == 1 && r_idx == 1) {
        corrupt.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(rc, EXIT_FAILURE);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Failed);
    EXPECT_EQ(result.reason_code,
              CpuLatencyExecutorReason::MAIN_FINAL_POINTER_FAILED);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 1u);
    ASSERT_EQ(result.locality_rounds.size(), 5u);
    EXPECT_EQ(result.locality_rounds[4].locality_16k.status, "invalid");
    EXPECT_EQ(result.measurements[1].status,
              CpuLatencyMeasurementStatus::Failed);
    EXPECT_FALSE(result.measurements[1].attempted);
    release_cpu_latency_resources(res);
  }
}

TEST_F(CpuLatencyExecutorRunnerTest,
       InterruptionDuringFinalMeasurementCompletesCleanly) {
  for (const bool compare_locality : {false, true}) {
    auto cfg = make_small_config(2, 2, 2, compare_locality);
    cfg.custom_cache_size_kb = 0;

    const auto plan = build_cpu_latency_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res));

    CpuLatencyResult result;
    std::atomic<bool> in_final_measurement{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (in_final_measurement.load()) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };

    CpuLatencyRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_measurement_hook = [&](size_t loop_idx, size_t pos) {
      if (loop_idx + 1 == cfg.loop_count &&
          pos + 1 == plan.active_targets.size()) {
        in_final_measurement.store(true);
      }
    };

    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuLatencyRunStatus::Complete);
    EXPECT_EQ(result.reason_code, "complete");
    EXPECT_TRUE(result.results_complete);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_FALSE(result.interruption_requested);
    EXPECT_EQ(result.counters.measured_measurements,
              result.counters.planned_measurements);
    if (compare_locality) {
      EXPECT_EQ(result.locality_rounds.size(), cfg.loop_count * 3);
    }
    release_cpu_latency_resources(res);
  }
}

}  // namespace
