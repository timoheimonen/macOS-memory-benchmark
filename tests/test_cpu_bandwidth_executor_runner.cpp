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

#include "core/audit/checkpoint_policy.h"
#include "core/config/constants.h"
#include "core/memory/memory_manager.h"
#include "core/timing/timer.h"
#include "cpu_bandwidth/cpu_bandwidth_executor.h"
#include "cpu_bandwidth/cpu_bandwidth_runner.h"

namespace {

CpuBandwidthConfig make_small_config(std::optional<size_t> iterations = 2,
                                     size_t loops = 3) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 2;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.iterations = iterations;
  cfg.user_specified_iterations = iterations.has_value();
  cfg.loop_count = loops;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  return cfg;
}

CpuBandwidthSystemInfo make_sys_info() {
  CpuBandwidthSystemInfo info;
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

int fail_worker_qos(void*, size_t) noexcept {
  return 5;
}

std::atomic<uint64_t> g_bw_mock_ticks{1000000ULL};
std::atomic<uint64_t> g_bw_mock_step_ticks{1000000ULL};
std::vector<uint64_t> g_bw_scripted_steps;
std::atomic<size_t> g_bw_script_index{0};

uint64_t bw_mock_absolute_time() {
  static std::atomic<bool> is_stop{false};
  const bool stop_call = is_stop.exchange(!is_stop.load(std::memory_order_relaxed),
                                          std::memory_order_relaxed);
  if (!stop_call) {
    return g_bw_mock_ticks.load(std::memory_order_relaxed);
  }
  uint64_t step = g_bw_mock_step_ticks.load(std::memory_order_relaxed);
  const size_t idx = g_bw_script_index.fetch_add(1, std::memory_order_relaxed);
  if (idx < g_bw_scripted_steps.size()) {
    step = g_bw_scripted_steps[idx];
  }
  return g_bw_mock_ticks.fetch_add(step, std::memory_order_relaxed) + step;
}

kern_return_t bw_mock_timebase_info(mach_timebase_info_t info) {
  info->numer = 125;
  info->denom = 3;
  return KERN_SUCCESS;
}

kern_return_t bw_fail_timebase_info(mach_timebase_info_t) {
  return KERN_FAILURE;
}

class CpuBandwidthExecutorRunnerTest : public ::testing::Test {
 protected:
  void TearDown() override {
    reset_memory_system_calls_for_testing();
    reset_timer_system_calls_for_testing();
    g_bw_scripted_steps.clear();
    g_bw_script_index.store(0, std::memory_order_relaxed);
  }
};

TEST_F(CpuBandwidthExecutorRunnerTest, MadviseNonFatalAndMprotectFailureDetected) {
  const auto cfg = make_small_config(2, 1);
  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  // Non-fatal madvise failure is recorded in AllocatedBufferEvidence.
  MemorySystemCalls madvise_fail;
  madvise_fail.advise = fail_madvise_einval;
  set_memory_system_calls_for_testing(madvise_fail);

  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, true, res));
  const auto& main_buf =
      res.targets[cpu_bandwidth_target_index(CpuBandwidthTarget::MainMemory)];
  EXPECT_FALSE(main_buf.src.madvise_applied);
  EXPECT_EQ(main_buf.src.madvise_errno, EINVAL);
  EXPECT_TRUE(main_buf.src.read_only_protected);
  release_cpu_bandwidth_resources(res);

  // mprotect failure fails resource preparation cleanly.
  MemorySystemCalls mprotect_fail;
  mprotect_fail.protect = fail_mprotect_eacces;
  set_memory_system_calls_for_testing(mprotect_fail);
  EXPECT_FALSE(prepare_cpu_bandwidth_resources(plan, false, res));
  EXPECT_EQ(res.reason_code, CpuBandwidthExecutorReason::SRC_READONLY_FAILED);
  release_cpu_bandwidth_resources(res);
}

TEST_F(CpuBandwidthExecutorRunnerTest, ExecutorFaultAndCorruptionInjectionSeams) {
  const auto cfg = make_small_config(2, 1);
  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, false, res));
  const auto& main_tplan = *plan.find_target(CpuBandwidthTarget::MainMemory);
  const auto& main_buf =
      res.targets[cpu_bandwidth_target_index(CpuBandwidthTarget::MainMemory)];
  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  // 1. Worker startup failure cancels partial worker team cleanly.
  {
    const auto op_plan = build_cpu_bandwidth_operation_plan(
        main_tplan, CpuBandwidthOperation::Read, 2, true);
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.fail_before_worker_index = 1;
    const auto ex = execute_cpu_bandwidth_operation(
        cfg, main_tplan, op_plan, main_buf, *timer_opt, std::nullopt,
        std::nullopt, 0ULL, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_TRUE(ex.worker_startup_failed);
    EXPECT_EQ(ex.reason_code, CpuBandwidthExecutorReason::WORKER_STARTUP_FAILED);
  }

  // 2. Worker QoS failure recorded without aborting run.
  {
    auto read_plan = build_cpu_bandwidth_operation_plan(
        main_tplan, CpuBandwidthOperation::Read, 2, true);
    read_plan.expected_checksum = compute_cpu_bandwidth_expected_read_checksum(
        main_tplan.src_seed, read_plan.op_seed, 2, main_tplan.worker_spans);
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.set_worker_qos = fail_worker_qos;
    const auto ex = execute_cpu_bandwidth_operation(
        cfg, main_tplan, read_plan, main_buf, *timer_opt, std::nullopt,
        std::nullopt, 0ULL, &ctrl);
    EXPECT_TRUE(ex.valid) << ex.reason_code;
    EXPECT_EQ(ex.qos_failed_workers, 2u);
  }

  // 3. Write 1-bit corruption caught by write-final-full-buffer check.
  {
    const auto write_plan = build_cpu_bandwidth_operation_plan(
        main_tplan, CpuBandwidthOperation::Write, 2, true);
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.corrupt_dst_after_kernel = [](void* dst, size_t bytes) {
      static_cast<uint8_t*>(dst)[bytes / 2] ^= 0x04U;
    };
    const auto ex = execute_cpu_bandwidth_operation(
        cfg, main_tplan, write_plan, main_buf, *timer_opt, 100ULL,
        derive_cpu_bandwidth_write_pattern(100ULL), 0ULL, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code,
              CpuBandwidthExecutorReason::WRITE_POST_VALIDATION_FAILED);
    EXPECT_FALSE(ex.validation_checks[2].valid.value_or(true));
  }

  // 4. Copy 1-bit corruption caught by copy-final-full-buffer check.
  {
    const auto copy_plan = build_cpu_bandwidth_operation_plan(
        main_tplan, CpuBandwidthOperation::Copy, 2, true);
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
      static_cast<uint8_t*>(dst)[17] ^= 0x01U;
    };
    const auto ex = execute_cpu_bandwidth_operation(
        cfg, main_tplan, copy_plan, main_buf, *timer_opt, std::nullopt,
        std::nullopt, 0x5555555555555555ULL, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code,
              CpuBandwidthExecutorReason::COPY_POST_VALIDATION_FAILED);
    EXPECT_FALSE(ex.validation_checks[3].valid.value_or(true));
  }

  // 5. Zero elapsed time rejected cleanly.
  {
    const auto write_plan = build_cpu_bandwidth_operation_plan(
        main_tplan, CpuBandwidthOperation::Write, 2, true);
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.force_zero_elapsed = true;
    const auto ex = execute_cpu_bandwidth_operation(
        cfg, main_tplan, write_plan, main_buf, *timer_opt, 101ULL,
        derive_cpu_bandwidth_write_pattern(101ULL), 0ULL, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, CpuBandwidthExecutorReason::INVALID_ELAPSED_TIME);
  }

  release_cpu_bandwidth_resources(res);
}

TEST_F(CpuBandwidthExecutorRunnerTest, RunnerLatinSquareCheckpointsAndInterruption) {
  const auto cfg = make_small_config(2, 3);
  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, false, res));

  // Complete 3-loop run with progress checkpoints (interval = max(1, ceil(3/8)) = 1).
  {
    CpuBandwidthResult result;
    size_t progress_calls = 0;
    size_t terminal_calls = 0;
    CpuBandwidthRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.observe_file_writes = [&]() {
      return std::make_pair(progress_calls, progress_calls);
    };
    hooks.checkpoint = [&](const CpuBandwidthResult&,
                           CpuBandwidthCheckpointKind kind) {
      if (kind == CpuBandwidthCheckpointKind::Progress) {
        ++progress_calls;
      } else {
        ++terminal_calls;
      }
      return EXIT_SUCCESS;
    };
    ASSERT_EQ(run_cpu_bandwidth_suite(cfg, plan, res, result, hooks),
              EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Complete);
    EXPECT_TRUE(result.results_complete);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_TRUE(result.operation_order_balance_complete);
    EXPECT_EQ(result.counters.measured_measurements, 18u);
    EXPECT_EQ(progress_calls, 2u);
    EXPECT_EQ(terminal_calls, 1u);
    EXPECT_EQ(result.snapshot_interval_loops, 1u);
  }

  // Mid-run interruption after measurement #2.
  {
    CpuBandwidthResult result;
    std::atomic<bool> stop{false};
    size_t executed = 0;
    CpuBandwidthRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_measurement_hook = [&](size_t, size_t) {
      if (executed >= 2) {
        stop.store(true);
      }
      ++executed;
    };
    EXPECT_EQ(run_cpu_bandwidth_suite(cfg, plan, res, result, hooks),
              EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 2u);
    EXPECT_EQ(result.measurements[2].status,
              CpuBandwidthMeasurementStatus::Interrupted);
  }

  // Checkpoint failure is terminal and not retried.
  {
    CpuBandwidthResult result;
    size_t attempts = 0;
    CpuBandwidthRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.checkpoint = [&](const CpuBandwidthResult&,
                           CpuBandwidthCheckpointKind) {
      ++attempts;
      return EXIT_FAILURE;
    };
    EXPECT_EQ(run_cpu_bandwidth_suite(cfg, plan, res, result, hooks),
              EXIT_FAILURE);
    EXPECT_EQ(attempts, 1u);
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Failed);
    EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
    EXPECT_FALSE(result.run_accepted);
  }

  release_cpu_bandwidth_resources(res);
}

TEST_F(CpuBandwidthExecutorRunnerTest, BoundedSnapshotIntervalMatchesPolicy) {
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(1), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(8), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(9), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(16), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(17), 3u);
}

TEST_F(CpuBandwidthExecutorRunnerTest,
       InterruptionDuringFinalMeasurementCompletesCleanly) {
  auto cfg = make_small_config(1, 2);
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 1;

  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, false, res));

  std::atomic<bool> armed{false};
  std::atomic<int> post_arm_checks{0};
  CpuBandwidthRunnerHooks hooks;
  hooks.stop_requested = [&]() {
    if (!armed.load()) {
      return false;
    }
    // First check is immediately after before_measurement_hook (before the final measurement runs);
    // any subsequent check (i.e. after the final measurement finishes) returns true.
    return post_arm_checks.fetch_add(1) >= 1;
  };
  hooks.before_measurement_hook = [&](size_t loop_idx, size_t pos) {
    if (loop_idx + 1 == cfg.loop_count &&
        pos + 1 == plan.active_targets.size() * kCpuBandwidthOperationCount) {
      armed.store(true);
    }
  };

  CpuBandwidthResult result;
  const int rc = run_cpu_bandwidth_suite(cfg, plan, res, result, hooks);
  EXPECT_EQ(rc, EXIT_SUCCESS);
  EXPECT_EQ(result.status, CpuBandwidthRunStatus::Complete);
  EXPECT_EQ(result.reason_code, "complete");
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.run_accepted);
  EXPECT_FALSE(result.interruption_requested);
  EXPECT_EQ(result.counters.measured_measurements,
            result.counters.planned_measurements);
  release_cpu_bandwidth_resources(res);
}

TEST_F(CpuBandwidthExecutorRunnerTest,
       AutoCalibrationScalesUpDownSingleUnitAndFreezes) {
  auto cfg = make_small_config(std::nullopt, 2);
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 1;

  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, false, res));

  // 1. Scale up from below-target-window (0.025s -> 600,000 ticks) into [0.100s, 0.250s] (3,600,000 ticks).
  {
    g_bw_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
    g_bw_mock_step_ticks.store(3600000ULL, std::memory_order_relaxed);
    g_bw_script_index.store(0, std::memory_order_relaxed);
    // Per op (Read, Write, Copy): shape_warmup, pilot (0.025s), correction (0.15s) -> then warmup + measurements use default 3,600,000 ticks.
    g_bw_scripted_steps = {
        3600000ULL, 600000ULL, 3600000ULL,  // Read
        3600000ULL, 600000ULL, 3600000ULL,  // Write
        3600000ULL, 600000ULL, 3600000ULL,  // Copy
    };
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = bw_mock_absolute_time;
    timer_calls.timebase_info = bw_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    CpuBandwidthResult result;
    CpuBandwidthRunnerHooks hooks;
    const int rc = run_cpu_bandwidth_suite(cfg, plan, res, result, hooks);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Complete);
    EXPECT_TRUE(result.run_accepted);
    const size_t t_idx = cpu_bandwidth_target_index(CpuBandwidthTarget::Custom);
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      ASSERT_TRUE(result.frozen_plan_refs[t_idx][op_i].has_value());
      const auto& atts = result.calibration_attempts[t_idx][op_i];
      ASSERT_GE(atts.size(), 4u);
      EXPECT_EQ(atts[0].purpose, "calibration_shape_warmup");
      EXPECT_EQ(atts[1].purpose, "pilot");
      EXPECT_EQ(atts[2].purpose, "correction");
      EXPECT_EQ(atts.back().purpose, "frozen_measurement_warmup");
    }
  }

  // 2. Scale down to 1 pass (single_unit_confirmation_warmup + single_unit_confirmation) and clamp at passes==1 (>0.250s).
  {
    g_bw_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
    g_bw_mock_step_ticks.store(3600000ULL, std::memory_order_relaxed);
    g_bw_script_index.store(0, std::memory_order_relaxed);
    // Pilot elapsed = 100s (2,400,000,000 ticks) -> scales down passes to 1 -> runs single_unit_confirmation_warmup + single_unit_confirmation (0.30s = 7,200,000 ticks -> breaks at passes==1 && elapsed > MAX_SECONDS).
    g_bw_scripted_steps = {
        3600000ULL, 2400000000ULL, 7200000ULL, 7200000ULL,  // Read
        3600000ULL, 2400000000ULL, 7200000ULL, 7200000ULL,  // Write
        3600000ULL, 2400000000ULL, 7200000ULL, 7200000ULL,  // Copy
    };
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = bw_mock_absolute_time;
    timer_calls.timebase_info = bw_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    CpuBandwidthResult result;
    CpuBandwidthRunnerHooks hooks;
    const int rc = run_cpu_bandwidth_suite(cfg, plan, res, result, hooks);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Complete);
    const size_t t_idx = cpu_bandwidth_target_index(CpuBandwidthTarget::Custom);
    const auto& atts = result.calibration_attempts[t_idx][0];
    ASSERT_GE(atts.size(), 5u);
    EXPECT_EQ(atts[2].purpose, "single_unit_confirmation_warmup");
    EXPECT_EQ(atts[3].purpose, "single_unit_confirmation");
    EXPECT_EQ(result.operation_plans[*result.frozen_plan_refs[t_idx][0]].passes,
              1u);
  }

  release_cpu_bandwidth_resources(res);
}

TEST_F(CpuBandwidthExecutorRunnerTest,
       AutoCalibrationInterruptionAndSetupFailurePaths) {
  auto cfg = make_small_config(std::nullopt, 2);
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 1;

  const auto plan = build_cpu_bandwidth_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  CpuBandwidthResourceSet res;
  ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, false, res));

  // 1. Setup failure initializer & invalid work_plan / resources / timer failure.
  {
    CpuBandwidthResult fail_res;
    initialize_cpu_bandwidth_setup_failure_result(
        cfg, plan, CpuBandwidthRunStatus::Interrupted, "interruption-requested",
        true, fail_res);
    EXPECT_EQ(fail_res.status, CpuBandwidthRunStatus::Interrupted);
    EXPECT_TRUE(fail_res.interruption_requested);
    EXPECT_FALSE(fail_res.run_accepted);
    EXPECT_EQ(fail_res.measurements.size(), 6u);
    EXPECT_EQ(fail_res.measurements[0].status,
              CpuBandwidthMeasurementStatus::Interrupted);

    CpuBandwidthWorkPlan invalid_plan;
    invalid_plan.valid = false;
    invalid_plan.reason_code = "memory-budget-exceeded";
    CpuBandwidthResult runner_fail;
    EXPECT_EQ(run_cpu_bandwidth_suite(cfg, invalid_plan, res, runner_fail, {}),
              EXIT_FAILURE);
    EXPECT_EQ(runner_fail.status, CpuBandwidthRunStatus::Failed);
    EXPECT_EQ(runner_fail.reason_code, "memory-budget-exceeded");

    TimerSystemCalls bad_timer;
    bad_timer.timebase_info = bw_fail_timebase_info;
    set_timer_system_calls_for_testing(bad_timer);
    CpuBandwidthResult timer_fail;
    EXPECT_EQ(run_cpu_bandwidth_suite(cfg, plan, res, timer_fail, {}),
              EXIT_FAILURE);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(timer_fail.reason_code, "timer-creation-failed");
  }

  // 2. Interruption during auto-calibration at stop_check calls 1, 2, 3, and during frozen warmup.
  for (size_t trip_on : {1u, 2u, 3u, 10u}) {
    g_bw_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
    g_bw_mock_step_ticks.store(600000ULL, std::memory_order_relaxed);
    g_bw_script_index.store(0, std::memory_order_relaxed);
    g_bw_scripted_steps.clear();
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = bw_mock_absolute_time;
    timer_calls.timebase_info = bw_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    size_t stop_calls = 0;
    CpuBandwidthRunnerHooks hooks;
    hooks.stop_requested = [&]() { return ++stop_calls >= trip_on; };
    CpuBandwidthResult result;
    const int rc = run_cpu_bandwidth_suite(cfg, plan, res, result, hooks);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.run_accepted);
    if (trip_on <= 3u) {
      EXPECT_EQ(result.counters.planned_passes, 0u);
    }
  }

  // 3. Kernel failure during calibration_shape_warmup & frozen warmup.
  {
    CpuBandwidthExecutorTestControl ctrl;
    ctrl.force_kernel_fail = true;
    CpuBandwidthRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    CpuBandwidthResult result;
    EXPECT_EQ(run_cpu_bandwidth_suite(cfg, plan, res, result, hooks),
              EXIT_FAILURE);
    EXPECT_EQ(result.status, CpuBandwidthRunStatus::Failed);
    EXPECT_EQ(result.reason_code, CpuBandwidthExecutorReason::KERNEL_FAILED);
  }

  release_cpu_bandwidth_resources(res);
}

}  // namespace
