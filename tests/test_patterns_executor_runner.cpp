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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/audit/checkpoint_policy.h"
#include "core/config/constants.h"
#include "core/memory/memory_manager.h"
#include "core/timing/timer.h"
#include "pattern_benchmark/patterns_executor.h"
#include "pattern_benchmark/patterns_oracle.h"
#include "pattern_benchmark/patterns_output.h"
#include "pattern_benchmark/patterns_runner.h"
#include "pattern_benchmark/patterns_work_plan.h"
#include "utils/cyclic_order.h"

namespace {

PatternsWorkPlanConfig make_config(size_t buffer_mb = 1,
                                   size_t workers = 2,
                                   std::optional<size_t> iterations = 2,
                                   size_t loops = 3,
                                   bool non_cacheable = false) {
  PatternsWorkPlanConfig cfg;
  cfg.buffer_size_mb = buffer_mb;
  cfg.requested_workers = workers;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.iterations = iterations;
  cfg.loop_count = loops;
  cfg.non_cacheable = non_cacheable;
  cfg.seed = 424242ULL;
  return cfg;
}

PatternsSystemInfo make_sys_info() {
  PatternsSystemInfo info;
  info.page_size_bytes = 16384ULL;
  info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  return info;
}

PatternsResultMetadata make_metadata(bool user_seed = true) {
  PatternsResultMetadata meta;
  meta.processor_name = "Apple Silicon Test";
  meta.macos_version = "15.0";
  meta.performance_core_count = 4;
  meta.efficiency_core_count = 4;
  meta.logical_core_count = 8;
  meta.page_size_bytes = 16384ULL;
  meta.l1_data_cache_bytes = 128ULL * 1024ULL;
  meta.l2_data_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  meta.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  meta.user_specified_seed = user_seed;
  meta.main_thread_qos.requested = true;
  meta.main_thread_qos.applied = true;
  meta.environment_start.thermal_state = "nominal";
  meta.environment_start.low_power_mode_available = true;
  meta.environment_start.low_power_mode_enabled = false;
  meta.environment_end = meta.environment_start;
  return meta;
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

std::atomic<uint64_t> g_pat_mock_ticks{1000000ULL};
std::atomic<uint64_t> g_pat_mock_step_ticks{3600000ULL};
std::vector<uint64_t> g_pat_scripted_steps;
std::atomic<size_t> g_pat_script_index{0};
std::atomic<bool> g_pat_is_stop{false};

void reset_mock_timer_state(uint64_t default_step_ticks,
                            std::vector<uint64_t> scripted = {}) {
  g_pat_mock_ticks.store(1000000ULL, std::memory_order_relaxed);
  g_pat_mock_step_ticks.store(default_step_ticks, std::memory_order_relaxed);
  g_pat_scripted_steps = std::move(scripted);
  g_pat_script_index.store(0, std::memory_order_relaxed);
  g_pat_is_stop.store(false, std::memory_order_relaxed);
}

uint64_t pat_mock_absolute_time() {
  const bool stop_call = g_pat_is_stop.exchange(
      !g_pat_is_stop.load(std::memory_order_relaxed), std::memory_order_relaxed);
  if (!stop_call) {
    return g_pat_mock_ticks.load(std::memory_order_relaxed);
  }
  uint64_t step = g_pat_mock_step_ticks.load(std::memory_order_relaxed);
  const size_t idx = g_pat_script_index.fetch_add(1, std::memory_order_relaxed);
  if (idx < g_pat_scripted_steps.size()) {
    step = g_pat_scripted_steps[idx];
  }
  return g_pat_mock_ticks.fetch_add(step, std::memory_order_relaxed) + step;
}

kern_return_t pat_mock_timebase_info(mach_timebase_info_t info) {
  info->numer = 125;
  info->denom = 3;
  return KERN_SUCCESS;
}

kern_return_t pat_fail_timebase_info(mach_timebase_info_t) {
  return KERN_FAILURE;
}

PatternsOperationPlan make_op_plan_with_checksum(
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    PatternKind kind,
    PatternOperation op,
    size_t passes) {
  const auto& pplan = work_plan.patterns[patterns_kind_index(kind)];
  PatternsOperationPlan op_plan = build_patterns_operation_plan(
      pplan, work_plan.src_seed, op, passes, true);
  if (op == PatternOperation::Read && op_plan.valid) {
    op_plan.expected_checksum = compute_patterns_expected_read_checksum(
        work_plan.src_seed, op_plan.op_seed, kind, pplan.stride_bytes,
        static_cast<uint64_t>(passes), pplan.worker_spans,
        kind == PatternKind::Random ? &resources.random_workload : nullptr);
  }
  return op_plan;
}

struct StreamCapture {
  std::ostream& target;
  std::ostringstream oss;
  std::streambuf* old_buf;

  explicit StreamCapture(std::ostream& os)
      : target(os), old_buf(os.rdbuf(oss.rdbuf())) {}
  ~StreamCapture() { target.rdbuf(old_buf); }
};

class PatternsExecutorRunnerTest : public ::testing::Test {
 protected:
  void TearDown() override {
    reset_memory_system_calls_for_testing();
    reset_timer_system_calls_for_testing();
    reset_mock_timer_state(3600000ULL);
  }
};

TEST_F(PatternsExecutorRunnerTest,
       ResourcePrepMadviseNonFatalMprotectFailureStopAndRandomPropagation) {
  const auto cfg = make_config(1, 2, 2, 1, true);
  PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.patterns[patterns_kind_index(PatternKind::Random)]
                .random_workload.logical_working_set_bytes,
            0u);
  EXPECT_TRUE(plan.patterns[patterns_kind_index(PatternKind::Random)]
                  .random_workload.random_indices_sha256.empty());

  // 1. Non-fatal madvise failure recorded while PROT_READ and random propagation succeed.
  {
    MemorySystemCalls madvise_fail;
    madvise_fail.advise = fail_madvise_einval;
    set_memory_system_calls_for_testing(madvise_fail);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan, true, res));
    EXPECT_TRUE(res.valid);
    EXPECT_FALSE(res.src.madvise_applied);
    EXPECT_EQ(res.src.madvise_errno, EINVAL);
    EXPECT_TRUE(res.src.read_only_protected);
    EXPECT_EQ(res.src.mprotect_errno, 0);

    const auto& rand_entry =
        plan.patterns[patterns_kind_index(PatternKind::Random)];
    EXPECT_GT(rand_entry.random_workload.logical_working_set_bytes, 0u);
    EXPECT_EQ(rand_entry.random_workload.random_indices_sha256.size(), 64u);
    EXPECT_EQ(rand_entry.random_workload.random_indices_sha256,
              res.random_workload.random_indices_sha256);
    EXPECT_LE(res.random_validation_index_bytes, 8ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(res.random_index_table_bytes,
              rand_entry.random_workload.selected_accesses_per_pass *
                  sizeof(size_t));

    const auto rand_op = build_patterns_operation_plan(
        rand_entry, plan.src_seed, PatternOperation::Read, 2, true);
    EXPECT_TRUE(rand_op.valid);
    EXPECT_EQ(rand_op.logical_working_set_bytes,
              rand_entry.random_workload.logical_working_set_bytes);

    release_patterns_resources(res);
    reset_memory_system_calls_for_testing();
  }

  // 2. mprotect failure fails resource prep cleanly and preserves src evidence.
  {
    MemorySystemCalls mprotect_fail;
    mprotect_fail.protect = fail_mprotect_eacces;
    set_memory_system_calls_for_testing(mprotect_fail);

    PatternsResourceSet res;
    EXPECT_FALSE(prepare_patterns_resources(plan, false, res));
    EXPECT_FALSE(res.valid);
    EXPECT_EQ(res.reason_code, PatternsExecutorReason::SRC_READONLY_FAILED);
    EXPECT_FALSE(res.src.read_only_protected);
    EXPECT_EQ(res.src.mprotect_errno, EACCES);
    release_patterns_resources(res);
    reset_memory_system_calls_for_testing();
  }

  // 3. stop_requested interrupts resource prep cleanly.
  {
    for (int trip_call : {1, 2, 3, 4}) {
      int calls = 0;
      PatternsResourceSet res;
      EXPECT_FALSE(prepare_patterns_resources(
          plan, false, res, [&]() { return ++calls >= trip_call; }));
      EXPECT_FALSE(res.valid);
      EXPECT_EQ(res.reason_code, PatternsExecutorReason::PREPARATION_INTERRUPTED);
      EXPECT_EQ(res.src.ptr, nullptr);
      EXPECT_EQ(res.dst.ptr, nullptr);
      release_patterns_resources(res);
    }
  }
}

TEST_F(PatternsExecutorRunnerTest, ValidationChecksOrderAndApplicabilityMatrix) {
  const auto read_checks =
      make_default_patterns_checks(PatternOperation::Read, false);
  const auto write_dense =
      make_default_patterns_checks(PatternOperation::Write, false);
  const auto write_sparse =
      make_default_patterns_checks(PatternOperation::Write, true);
  const auto copy_dense =
      make_default_patterns_checks(PatternOperation::Copy, false);
  const auto copy_sparse =
      make_default_patterns_checks(PatternOperation::Copy, true);

  const std::array<const char*, kPatternsValidationCheckCount> kExpectedKinds = {
      "post-validation-structure",
      "read-checksum",
      "write-final-touched-bytes",
      "copy-final-touched-bytes",
      "untouched-sentinel-isolation",
      "src-readonly-protection",
  };
  for (size_t i = 0; i < kPatternsValidationCheckCount; ++i) {
    EXPECT_EQ(read_checks[i].kind, kExpectedKinds[i]);
    EXPECT_EQ(write_sparse[i].kind, kExpectedKinds[i]);
    EXPECT_EQ(copy_sparse[i].kind, kExpectedKinds[i]);
  }

  // Read: 0, 1, 5 applicable
  EXPECT_TRUE(read_checks[0].applicable);
  EXPECT_TRUE(read_checks[1].applicable);
  EXPECT_FALSE(read_checks[2].applicable);
  EXPECT_FALSE(read_checks[3].applicable);
  EXPECT_FALSE(read_checks[4].applicable);
  EXPECT_TRUE(read_checks[5].applicable);

  // Write dense: 0, 2 applicable
  EXPECT_TRUE(write_dense[0].applicable);
  EXPECT_FALSE(write_dense[1].applicable);
  EXPECT_TRUE(write_dense[2].applicable);
  EXPECT_FALSE(write_dense[3].applicable);
  EXPECT_FALSE(write_dense[4].applicable);
  EXPECT_FALSE(write_dense[5].applicable);

  // Write sparse: 0, 2, 4 applicable
  EXPECT_TRUE(write_sparse[0].applicable);
  EXPECT_FALSE(write_sparse[1].applicable);
  EXPECT_TRUE(write_sparse[2].applicable);
  EXPECT_FALSE(write_sparse[3].applicable);
  EXPECT_TRUE(write_sparse[4].applicable);
  EXPECT_FALSE(write_sparse[5].applicable);

  // Copy dense: 0, 3, 5 applicable
  EXPECT_TRUE(copy_dense[0].applicable);
  EXPECT_FALSE(copy_dense[1].applicable);
  EXPECT_FALSE(copy_dense[2].applicable);
  EXPECT_TRUE(copy_dense[3].applicable);
  EXPECT_FALSE(copy_dense[4].applicable);
  EXPECT_TRUE(copy_dense[5].applicable);

  // Copy sparse: 0, 3, 4, 5 applicable
  EXPECT_TRUE(copy_sparse[0].applicable);
  EXPECT_FALSE(copy_sparse[1].applicable);
  EXPECT_FALSE(copy_sparse[2].applicable);
  EXPECT_TRUE(copy_sparse[3].applicable);
  EXPECT_TRUE(copy_sparse[4].applicable);
  EXPECT_TRUE(copy_sparse[5].applicable);
}

TEST_F(PatternsExecutorRunnerTest, ExecutorFaultChecksumAndCorruptionInjection) {
  const auto cfg = make_config(1, 2, 2, 1);
  PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);

  PatternsResourceSet res;
  ASSERT_TRUE(prepare_patterns_resources(plan, false, res));
  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  const auto& seq_fwd_plan =
      plan.patterns[patterns_kind_index(PatternKind::SequentialForward)];
  const auto& seq_rev_plan =
      plan.patterns[patterns_kind_index(PatternKind::SequentialReverse)];
  const auto& s64_plan =
      plan.patterns[patterns_kind_index(PatternKind::Strided64)];
  const auto& s4096_plan =
      plan.patterns[patterns_kind_index(PatternKind::Strided4096)];
  const auto& rand_plan =
      plan.patterns[patterns_kind_index(PatternKind::Random)];

  // 1. Worker startup failure cancels worker team cleanly.
  {
    const auto op_plan = make_op_plan_with_checksum(
        plan, res, PatternKind::SequentialForward, PatternOperation::Read, 2);
    PatternsExecutorTestControl ctrl;
    ctrl.fail_before_worker_index = 1;
    const auto ex = execute_patterns_operation(
        cfg, seq_fwd_plan, op_plan, res, *timer_opt, std::nullopt,
        std::nullopt, std::nullopt, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_TRUE(ex.worker_startup_failed);
    EXPECT_EQ(ex.reason_code, PatternsExecutorReason::WORKER_STARTUP_FAILED);
    EXPECT_FALSE(ex.validation_checks[0].valid.value_or(true));
  }

  // 2. Worker QoS failure recorded without failing run.
  {
    const auto op_plan = make_op_plan_with_checksum(
        plan, res, PatternKind::SequentialForward, PatternOperation::Read, 2);
    PatternsExecutorTestControl ctrl;
    ctrl.set_worker_qos = fail_worker_qos;
    const auto ex = execute_patterns_operation(
        cfg, seq_fwd_plan, op_plan, res, *timer_opt, std::nullopt,
        std::nullopt, std::nullopt, &ctrl);
    EXPECT_TRUE(ex.valid) << ex.reason_code;
    EXPECT_EQ(ex.qos_failed_workers, 2u);
    EXPECT_EQ(ex.qos_successful_workers, 0u);
  }

  // 3. Kernel failure still evaluates src-readonly-protection for Read and Copy.
  {
    const auto op_plan = make_op_plan_with_checksum(
        plan, res, PatternKind::SequentialForward, PatternOperation::Read, 2);
    PatternsExecutorTestControl ctrl;
    ctrl.force_kernel_fail = true;
    const auto ex = execute_patterns_operation(
        cfg, seq_fwd_plan, op_plan, res, *timer_opt, std::nullopt,
        std::nullopt, std::nullopt, &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, PatternsExecutorReason::KERNEL_FAILED);
    EXPECT_FALSE(ex.validation_checks[0].valid.value_or(true));
    EXPECT_TRUE(ex.validation_checks[5].evaluated.value_or(false));
    EXPECT_TRUE(ex.validation_checks[5].valid.value_or(false));
  }

  // 4. Zero elapsed time rejected cleanly with cpu_raw cleared.
  {
    const auto op_plan = make_op_plan_with_checksum(
        plan, res, PatternKind::SequentialForward, PatternOperation::Write, 2);
    PatternsExecutorTestControl ctrl;
    ctrl.force_zero_elapsed = true;
    const uint64_t pat = derive_patterns_write_pattern(101ULL);
    const auto ex = execute_patterns_operation(
        cfg, seq_fwd_plan, op_plan, res, *timer_opt, 101ULL, pat, std::nullopt,
        &ctrl);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, PatternsExecutorReason::INVALID_ELAPSED_TIME);
    EXPECT_FALSE(ex.timing_valid);
    EXPECT_FALSE(ex.cpu_raw.has_value());
  }

  // 5. Read checksum mismatch caught by read-checksum check.
  {
    auto op_plan = make_op_plan_with_checksum(
        plan, res, PatternKind::Strided64, PatternOperation::Read, 3);
    op_plan.expected_checksum.run_checksum.state_a ^= 1ULL;
    const auto ex = execute_patterns_operation(
        cfg, s64_plan, op_plan, res, *timer_opt, std::nullopt, std::nullopt,
        std::nullopt, nullptr);
    EXPECT_FALSE(ex.valid);
    EXPECT_EQ(ex.reason_code, PatternsExecutorReason::CHECKSUM_MISMATCH);
    EXPECT_TRUE(ex.checksum_evaluated);
    EXPECT_FALSE(ex.checksum_valid);
    EXPECT_FALSE(ex.validation_checks[1].valid.value_or(true));
  }

  // 6. Write touched-byte corruption across SequentialForward, SequentialReverse, partial Strided64, Random.
  {
    const uint64_t pat = derive_patterns_write_pattern(201ULL);
    const uint64_t sent =
        derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed, 1ULL);
    ASSERT_NE(pat, sent);

    // SequentialForward
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::SequentialForward, PatternOperation::Write, 1);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t bytes) {
        static_cast<uint8_t*>(dst)[bytes / 2] ^= 0x04U;
      };
      const auto ex = execute_patterns_operation(
          cfg, seq_fwd_plan, op_plan, res, *timer_opt, 201ULL, pat,
          std::nullopt, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[2].valid.value_or(true));
    }

    // SequentialReverse
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::SequentialReverse, PatternOperation::Write, 1);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[33] ^= 0x80U;
      };
      const auto ex = execute_patterns_operation(
          cfg, seq_rev_plan, op_plan, res, *timer_opt, 201ULL, pat,
          std::nullopt, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[2].valid.value_or(true));
    }

    // Partial Strided64 (passes=1 < Q=2) -> touched byte at offset 0 corrupted
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Strided64, PatternOperation::Write, 1);
      ASSERT_TRUE(op_plan.has_untouched_dst_bytes);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[0] ^= 0x01U;
      };
      const auto ex = execute_patterns_operation(
          cfg, s64_plan, op_plan, res, *timer_opt, 201ULL, pat, sent, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[2].valid.value_or(true));
    }

    // Random touched slot corrupted (dense 1 MiB)
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Random, PatternOperation::Write, 1);
      EXPECT_FALSE(op_plan.has_untouched_dst_bytes);
      const size_t touched_offset =
          res.random_workload.workers[0].offset_bytes +
          res.random_workload.sorted_validation_indices[0][0];
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [touched_offset](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[touched_offset + 5] ^= 0x08U;
      };
      const auto ex = execute_patterns_operation(
          cfg, rand_plan, op_plan, res, *timer_opt, 201ULL, pat, std::nullopt,
          &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[2].valid.value_or(true));
    }
  }

  // 7. Copy touched-byte corruption (Strided4096 partial, Random).
  {
    const uint64_t sent =
        derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed, 2ULL);

    // Strided4096 partial (passes=1 < Q=128) -> offset 0 is touched
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Strided4096, PatternOperation::Copy, 1);
      ASSERT_TRUE(op_plan.has_untouched_dst_bytes);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[17] ^= 0x02U;
      };
      const auto ex = execute_patterns_operation(
          cfg, s4096_plan, op_plan, res, *timer_opt, std::nullopt, std::nullopt,
          sent, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::COPY_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[3].valid.value_or(true));
    }

    // Random Copy touched slot corrupted
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Random, PatternOperation::Copy, 1);
      const size_t touched_offset =
          res.random_workload.workers[0].offset_bytes +
          res.random_workload.sorted_validation_indices[0][0];
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [touched_offset](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[touched_offset + 11] ^= 0x10U;
      };
      const auto ex = execute_patterns_operation(
          cfg, rand_plan, op_plan, res, *timer_opt, std::nullopt, std::nullopt,
          sent, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::COPY_POST_VALIDATION_FAILED);
      EXPECT_FALSE(ex.validation_checks[3].valid.value_or(true));
    }
  }

  // 8. Untouched sentinel corruption caught cleanly while touched check remains true.
  {
    const uint64_t pat = derive_patterns_write_pattern(301ULL);
    const uint64_t sent =
        derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed, 3ULL);
    ASSERT_NE(pat, sent);

    // Strided4096 passes=1 < Q=128: slot 1 (offset 32) is untouched.
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Strided4096, PatternOperation::Write, 1);
      ASSERT_TRUE(op_plan.has_untouched_dst_bytes);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[32 + 7] ^= 0x40U;
      };
      const auto ex = execute_patterns_operation(
          cfg, s4096_plan, op_plan, res, *timer_opt, 301ULL, pat, sent, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED);
      EXPECT_TRUE(ex.validation_checks[2].valid.value_or(false));
      EXPECT_FALSE(ex.validation_checks[4].valid.value_or(true));
    }

    // Strided4096 Copy passes=1 < Q=128: slot 1 (offset 32) untouched corrupted.
    {
      const auto op_plan = make_op_plan_with_checksum(
          plan, res, PatternKind::Strided4096, PatternOperation::Copy, 1);
      PatternsExecutorTestControl ctrl;
      ctrl.corrupt_dst_after_kernel = [](void* dst, size_t) {
        static_cast<uint8_t*>(dst)[32 + 12] ^= 0x20U;
      };
      const auto ex = execute_patterns_operation(
          cfg, s4096_plan, op_plan, res, *timer_opt, std::nullopt, std::nullopt,
          sent, &ctrl);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code,
                PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED);
      EXPECT_TRUE(ex.validation_checks[3].valid.value_or(false));
      EXPECT_FALSE(ex.validation_checks[4].valid.value_or(true));
    }

    // Random Write & Copy untouched slot corrupted (sparse 32 MiB buffer).
    {
      const auto cfg32 = make_config(32, 2, 1, 1);
      PatternsWorkPlan plan32 = build_patterns_work_plan(cfg32, make_sys_info());
      ASSERT_TRUE(plan32.valid);
      PatternsResourceSet res32;
      ASSERT_TRUE(prepare_patterns_resources(plan32, false, res32));
      const auto& rand_plan32 =
          plan32.patterns[patterns_kind_index(PatternKind::Random)];

      // Find first untouched 32B slot in worker 0.
      const auto& sorted0 = res32.random_workload.sorted_validation_indices[0];
      std::unordered_set<size_t> touched_set(sorted0.begin(), sorted0.end());
      size_t untouched_rel = 0;
      while (touched_set.count(untouched_rel) > 0) {
        untouched_rel += kPatternsAccessSizeBytes;
      }
      const size_t untouched_global =
          res32.random_workload.workers[0].offset_bytes + untouched_rel;

      for (PatternOperation op :
           {PatternOperation::Write, PatternOperation::Copy}) {
        const auto op_plan = make_op_plan_with_checksum(
            plan32, res32, PatternKind::Random, op, 1);
        ASSERT_TRUE(op_plan.has_untouched_dst_bytes);
        PatternsExecutorTestControl ctrl;
        ctrl.corrupt_dst_after_kernel = [untouched_global](void* dst, size_t) {
          static_cast<uint8_t*>(dst)[untouched_global + 3] ^= 0x04U;
        };
        const auto ex = execute_patterns_operation(
            cfg32, rand_plan32, op_plan, res32, *timer_opt,
            op == PatternOperation::Write ? std::optional<uint64_t>(302ULL)
                                          : std::nullopt,
            op == PatternOperation::Write ? std::optional<uint64_t>(pat)
                                          : std::nullopt,
            sent, &ctrl);
        EXPECT_FALSE(ex.valid);
        EXPECT_EQ(ex.reason_code,
                  PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED);
        EXPECT_TRUE(ex.validation_checks[op == PatternOperation::Write ? 2 : 3]
                        .valid.value_or(false));
        EXPECT_FALSE(ex.validation_checks[4].valid.value_or(true));
      }
      release_patterns_resources(res32);
    }
  }

  // 9. src-readonly-protection check failure detected for Read and Copy.
  {
    PatternsResourceSet bad_ro_res = {};
    ASSERT_TRUE(prepare_patterns_resources(plan, false, bad_ro_res));
    bad_ro_res.src.read_only_protected = false;
    bad_ro_res.src.mprotect_errno = EACCES;

    for (PatternOperation op :
         {PatternOperation::Read, PatternOperation::Copy}) {
      const auto op_plan = make_op_plan_with_checksum(
          plan, bad_ro_res, PatternKind::SequentialForward, op, 1);
      const std::optional<uint64_t> sent =
          op == PatternOperation::Copy
              ? std::optional<uint64_t>(derive_patterns_pre_op_sentinel(
                    plan.pre_op_sentinel_seed, 99ULL))
              : std::nullopt;
      const auto ex = execute_patterns_operation(
          cfg, seq_fwd_plan, op_plan, bad_ro_res, *timer_opt, std::nullopt,
          std::nullopt, sent, nullptr);
      EXPECT_FALSE(ex.valid);
      EXPECT_EQ(ex.reason_code, PatternsExecutorReason::SRC_READONLY_FAILED);
      EXPECT_FALSE(ex.validation_checks[5].valid.value_or(true));
    }
    release_patterns_resources(bad_ro_res);
  }

  release_patterns_resources(res);
}

TEST_F(PatternsExecutorRunnerTest, BoundedSnapshotIntervalAndSetupFailures) {
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(1), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(8), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(9), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(16), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(17), 3u);

  const auto cfg = make_config(1, 2, 2, 2);
  PatternsWorkPlan valid_plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(valid_plan.valid);

  // Interrupted setup failure preserves 7x3 aggregates (6 unavailable, strided_2mb skipped).
  {
    PatternsResult fail_res;
    initialize_patterns_setup_failure_result(
        cfg, valid_plan, PatternsRunStatus::Interrupted,
        "interruption-requested", true, fail_res);
    EXPECT_EQ(fail_res.status, PatternsRunStatus::Interrupted);
    EXPECT_TRUE(fail_res.interruption_requested);
    EXPECT_FALSE(fail_res.run_accepted);
    EXPECT_EQ(fail_res.measurements.size(), 36u);
    EXPECT_EQ(fail_res.measurements[0].status,
              PatternsMeasurementStatus::Interrupted);
    EXPECT_EQ(fail_res.aggregates[patterns_kind_index(PatternKind::Strided2MiB)]
                                 [0]
                                     .status,
              PatternsAggregateStatus::Skipped);
    EXPECT_EQ(fail_res.aggregates[patterns_kind_index(PatternKind::Strided2MiB)]
                                 [0]
                                     .reason_code,
              PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);
    EXPECT_EQ(fail_res.aggregates[0][0].status,
              PatternsAggregateStatus::Unavailable);
  }

  // Pre-freeze 32 MiB interrupted setup preserves sparse Random & Strided2MiB applicability.
  {
    const auto cfg32 = make_config(32, 2, std::nullopt, 1);
    PatternsWorkPlan plan32 = build_patterns_work_plan(cfg32, make_sys_info());
    ASSERT_TRUE(plan32.valid);
    ASSERT_EQ(plan32.active_patterns.size(), 7u);

    PatternsResult fail_res32;
    initialize_patterns_setup_failure_result(
        cfg32, plan32, PatternsRunStatus::Interrupted,
        "interruption-requested", true, fail_res32);
    ASSERT_EQ(fail_res32.measurements.size(), 21u);
    for (const auto& row : fail_res32.measurements) {
      EXPECT_FALSE(row.plan_ref.has_value());
      const bool expected_untouched =
          (row.operation != PatternOperation::Read) &&
          (patterns_kind_is_strided(row.pattern) ||
           row.pattern == PatternKind::Random);
      const auto& ck4 = row.execution.validation_checks[4];
      EXPECT_EQ(ck4.kind, "untouched-sentinel-isolation");
      EXPECT_EQ(ck4.applicable, expected_untouched);
      EXPECT_EQ(ck4.evaluated,
                expected_untouched ? std::optional<bool>(false) : std::nullopt);
      EXPECT_FALSE(ck4.valid.has_value());
      EXPECT_EQ(ck4.reason_code,
                expected_untouched ? "not-evaluated" : "not-applicable");
    }
  }

  // Invalid work_plan -> all 7x3 aggregates Unavailable with budget reason.
  {
    PatternsWorkPlan invalid_plan;
    invalid_plan.valid = false;
    invalid_plan.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    PatternsResourceSet dummy_res;
    PatternsResult runner_fail;
    EXPECT_EQ(run_patterns_suite(cfg, invalid_plan, dummy_res, runner_fail, {}),
              EXIT_FAILURE);
    EXPECT_EQ(runner_fail.status, PatternsRunStatus::Failed);
    EXPECT_EQ(runner_fail.reason_code,
              PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
    for (size_t p = 0; p < kPatternsMaxKinds; ++p) {
      for (size_t op = 0; op < kPatternsOperationCount; ++op) {
        EXPECT_EQ(runner_fail.aggregates[p][op].status,
                  PatternsAggregateStatus::Unavailable);
        EXPECT_EQ(runner_fail.aggregates[p][op].reason_code,
                  PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
      }
    }
  }

  // Timer creation failure -> timer-creation-failed.
  {
    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(valid_plan, false, res));
    TimerSystemCalls bad_timer;
    bad_timer.timebase_info = pat_fail_timebase_info;
    set_timer_system_calls_for_testing(bad_timer);

    PatternsResult timer_fail;
    EXPECT_EQ(run_patterns_suite(cfg, valid_plan, res, timer_fail, {}),
              EXIT_FAILURE);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(timer_fail.reason_code, "timer-creation-failed");
    EXPECT_EQ(timer_fail.aggregates[patterns_kind_index(PatternKind::Strided2MiB)]
                                   [0]
                                       .status,
              PatternsAggregateStatus::Skipped);
    release_patterns_resources(res);
  }
}

TEST_F(PatternsExecutorRunnerTest,
       TwoAxisLatinSquareRotationAndStrided2MiBSkippedVsActive) {
  reset_mock_timer_state(3600000ULL);
  TimerSystemCalls timer_calls;
  timer_calls.absolute_time = pat_mock_absolute_time;
  timer_calls.timebase_info = pat_mock_timebase_info;
  set_timer_system_calls_for_testing(timer_calls);

  // 1. 1 MiB, 6 loops -> 6 active patterns, strided_2mb skipped, both balances complete.
  {
    const auto cfg = make_config(1, 2, 2, 6);
    PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    ASSERT_EQ(plan.active_patterns.size(), 6u);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan, false, res));
    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_EQ(result.counters.planned_measurements, 108u);
    EXPECT_EQ(result.counters.measured_measurements, 108u);
    EXPECT_TRUE(result.operation_order_balance_complete);
    EXPECT_TRUE(result.pattern_order_balance_complete);

    const size_t s2m_idx = patterns_kind_index(PatternKind::Strided2MiB);
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      EXPECT_TRUE(result.calibration_attempts[s2m_idx][op_i].empty());
      EXPECT_FALSE(result.frozen_plan_refs[s2m_idx][op_i].has_value());
      EXPECT_EQ(result.aggregates[s2m_idx][op_i].status,
                PatternsAggregateStatus::Skipped);
      EXPECT_EQ(result.aggregates[s2m_idx][op_i].reason_code,
                PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);
    }

    for (size_t loop_idx = 0; loop_idx < 6; ++loop_idx) {
      const auto p_order = build_cyclic_order(6, loop_idx);
      const auto o_order = build_cyclic_order(3, loop_idx);
      const auto& loop = result.loops[loop_idx];
      ASSERT_EQ(loop.planned_order.size(), 18u);
      ASSERT_EQ(loop.realized_order.size(), 18u);
      size_t k = 0;
      for (size_t p_pos = 0; p_pos < 6; ++p_pos) {
        const PatternKind expected_pat = plan.active_patterns[p_order[p_pos]];
        for (size_t o_pos = 0; o_pos < 3; ++o_pos) {
          const auto expected_op = static_cast<PatternOperation>(o_order[o_pos]);
          EXPECT_EQ(loop.planned_order[k].pattern, expected_pat);
          EXPECT_EQ(loop.planned_order[k].operation, expected_op);
          EXPECT_EQ(loop.realized_order[k].pattern, expected_pat);
          EXPECT_EQ(loop.realized_order[k].operation, expected_op);
          ++k;
        }
      }
    }
    release_patterns_resources(res);
  }

  // 2. 3 MiB, 3 loops -> 7 active patterns, strided_2mb measured, pattern balance false (3 % 7 != 0).
  {
    const auto cfg = make_config(3, 2, 2, 3);
    PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);
    ASSERT_EQ(plan.active_patterns.size(), 7u);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan, false, res));
    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_EQ(result.counters.planned_measurements, 63u);
    EXPECT_EQ(result.counters.measured_measurements, 63u);
    EXPECT_TRUE(result.operation_order_balance_complete);
    EXPECT_FALSE(result.pattern_order_balance_complete);

    const size_t s2m_idx = patterns_kind_index(PatternKind::Strided2MiB);
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      EXPECT_TRUE(result.frozen_plan_refs[s2m_idx][op_i].has_value());
      EXPECT_EQ(result.aggregates[s2m_idx][op_i].status,
                PatternsAggregateStatus::Measured);
    }

    for (size_t loop_idx = 0; loop_idx < 3; ++loop_idx) {
      const auto p_order = build_cyclic_order(7, loop_idx);
      const auto o_order = build_cyclic_order(3, loop_idx);
      const auto& loop = result.loops[loop_idx];
      ASSERT_EQ(loop.planned_order.size(), 21u);
      size_t k = 0;
      for (size_t p_pos = 0; p_pos < 7; ++p_pos) {
        const PatternKind expected_pat = plan.active_patterns[p_order[p_pos]];
        for (size_t o_pos = 0; o_pos < 3; ++o_pos) {
          const auto expected_op = static_cast<PatternOperation>(o_order[o_pos]);
          EXPECT_EQ(loop.planned_order[k].pattern, expected_pat);
          EXPECT_EQ(loop.planned_order[k].operation, expected_op);
          ++k;
        }
      }
    }
    release_patterns_resources(res);
  }

  reset_timer_system_calls_for_testing();
}

TEST_F(PatternsExecutorRunnerTest,
       ExcludedAutoCalibrationScaleUpDownSingleUnitClampAndInterrupt) {
  const auto cfg = make_config(1, 2, std::nullopt, 2);
  PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  ASSERT_EQ(plan.active_patterns.size(), 6u);

  PatternsResourceSet res;
  ASSERT_TRUE(prepare_patterns_resources(plan, false, res));

  // 1. Scale up from 0.025s (600,000 ticks) into 0.150s (3,600,000 ticks).
  {
    std::vector<uint64_t> scripted;
    for (size_t i = 0; i < 6 * 3; ++i) {
      scripted.push_back(3600000ULL);  // calibration_shape_warmup
      scripted.push_back(600000ULL);   // pilot (0.025s -> scales up 6x)
      scripted.push_back(3600000ULL);  // correction (0.150s)
    }
    reset_mock_timer_state(3600000ULL, scripted);
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = pat_mock_absolute_time;
    timer_calls.timebase_info = pat_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_TRUE(result.run_accepted);
    for (PatternKind kind : plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        ASSERT_TRUE(result.frozen_plan_refs[p_idx][op_i].has_value());
        const auto& atts = result.calibration_attempts[p_idx][op_i];
        ASSERT_EQ(atts.size(), 4u);
        EXPECT_EQ(atts[0].purpose, "calibration_shape_warmup");
        EXPECT_EQ(atts[1].purpose, "pilot");
        EXPECT_EQ(atts[2].purpose, "correction");
        EXPECT_EQ(atts[3].purpose, "frozen_measurement_warmup");
        EXPECT_EQ(result.operation_plans[*result.frozen_plan_refs[p_idx][op_i]]
                      .passes,
                  atts[2].plan_ref.has_value()
                      ? result.operation_plans[*atts[2].plan_ref].passes
                      : 0u);
      }
    }
    // Check that strided_2mb was NOT calibrated.
    EXPECT_TRUE(
        result.calibration_attempts[patterns_kind_index(PatternKind::Strided2MiB)]
                                   [0]
                                       .empty());
  }

  // 2. Scale down to passes == 1 (single_unit_confirmation_warmup + single_unit_confirmation)
  //    and clamp at passes == 1 when > 0.250s.
  {
    std::vector<uint64_t> scripted;
    for (size_t i = 0; i < 6 * 3; ++i) {
      scripted.push_back(3600000ULL);     // calibration_shape_warmup
      scripted.push_back(2400000000ULL);  // pilot = 100s -> scales down to passes=1
      scripted.push_back(7200000ULL);     // single_unit_confirmation_warmup (0.30s)
      scripted.push_back(7200000ULL);     // single_unit_confirmation (0.30s -> clamps!)
    }
    reset_mock_timer_state(3600000ULL, scripted);
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = pat_mock_absolute_time;
    timer_calls.timebase_info = pat_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    const size_t p_idx = patterns_kind_index(PatternKind::SequentialForward);
    const auto& atts = result.calibration_attempts[p_idx][0];
    ASSERT_EQ(atts.size(), 5u);
    EXPECT_EQ(atts[2].purpose, "single_unit_confirmation_warmup");
    EXPECT_EQ(atts[3].purpose, "single_unit_confirmation");
    EXPECT_EQ(atts[4].purpose, "frozen_measurement_warmup");
    EXPECT_EQ(result.operation_plans[*result.frozen_plan_refs[p_idx][0]].passes,
              1u);
  }

  // 3. Interruption during auto-calibration (stop_calls 1, 2, 3) and during frozen warmup.
  for (size_t trip_on : {1u, 2u, 3u, 40u}) {
    reset_mock_timer_state(600000ULL);
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = pat_mock_absolute_time;
    timer_calls.timebase_info = pat_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    size_t stop_calls = 0;
    PatternsRunnerHooks hooks;
    hooks.stop_requested = [&]() { return ++stop_calls >= trip_on; };
    PatternsResult result;
    const int rc = run_patterns_suite(cfg, plan, res, result, hooks);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(rc, EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.run_accepted);
    if (trip_on <= 3u) {
      EXPECT_EQ(result.counters.planned_passes, 0u);
    }
    if (trip_on == 1u) {
      ASSERT_FALSE(result.measurements.empty());
      for (const auto& row : result.measurements) {
        EXPECT_FALSE(row.plan_ref.has_value());
        const bool expected_untouched =
            (row.operation != PatternOperation::Read) &&
            patterns_kind_is_strided(row.pattern);
        const auto& ck4 = row.execution.validation_checks[4];
        EXPECT_EQ(ck4.kind, "untouched-sentinel-isolation");
        EXPECT_EQ(ck4.applicable, expected_untouched);
        EXPECT_EQ(ck4.evaluated,
                  expected_untouched ? std::optional<bool>(false) : std::nullopt);
        EXPECT_FALSE(ck4.valid.has_value());
        EXPECT_EQ(ck4.reason_code,
                  expected_untouched ? "not-evaluated" : "not-applicable");
      }
    }
  }

  // 4. Kernel failure during calibration_shape_warmup fails cleanly.
  {
    PatternsExecutorTestControl ctrl;
    ctrl.force_kernel_fail = true;
    PatternsRunnerHooks hooks;
    hooks.executor_test_control = &ctrl;
    PatternsResult result;
    EXPECT_EQ(run_patterns_suite(cfg, plan, res, result, hooks), EXIT_FAILURE);
    EXPECT_EQ(result.status, PatternsRunStatus::Failed);
    EXPECT_EQ(result.reason_code, PatternsExecutorReason::KERNEL_FAILED);
  }

  // 5. Exact target-window calibration boundaries:
  //    0.100s (2,400,000 ticks) & 0.250s (6,000,000 ticks) -> 0 corrections;
  //    0.099999s (2,399,976 ticks) & 0.250001s (6,000,024 ticks) -> 1 correction;
  //    persistent out-of-window (240,000 ticks) -> capped at 2 corrections (5 attempts).
  {
    const size_t fwd_idx = patterns_kind_index(PatternKind::SequentialForward);
    const size_t rev_idx = patterns_kind_index(PatternKind::SequentialReverse);
    const size_t s64_idx = patterns_kind_index(PatternKind::Strided64);
    const size_t s4k_idx = patterns_kind_index(PatternKind::Strided4096);
    const size_t s16k_idx = patterns_kind_index(PatternKind::Strided16384);

    // Script pilots per pattern/op (18 total pairs in 1 MiB active_patterns):
    // - fwd/read (pair 0): 2,400,000 ticks (0.100s) -> 0 corrections (3 attempts)
    // - fwd/write (pair 1): 6,000,000 ticks (0.250s) -> 0 corrections (3 attempts)
    // - fwd/copy (pair 2): 2,399,976 ticks (0.099999s) then 3,600,000 -> 1 correction (4 attempts)
    // - rev/read (pair 3): 6,000,024 ticks (0.250001s) then 3,600,000 -> 1 correction (4 attempts)
    // - rev/write (pair 4): 240,000 ticks repeatedly -> 2 corrections cap (5 attempts)
    // - rev/copy (pair 5): 24,000,000 ticks (1.0s -> scales to passes=1), then
    //   single_unit_confirmation_warmup (3,600,000), single_unit_confirmation (240,000 = 0.010s < 0.100s
    //   -> scales back up to 15 passes), correction (3,600,000) -> 6 attempts!
    std::vector<uint64_t> scripted;
    // pair 0 (fwd/read): 0.100s exact
    scripted.push_back(3600000ULL);
    scripted.push_back(2400000ULL);
    // pair 1 (fwd/write): 0.250s exact
    scripted.push_back(3600000ULL);
    scripted.push_back(6000000ULL);
    // pair 2 (fwd/copy): 0.099999s -> 1 correction
    scripted.push_back(3600000ULL);
    scripted.push_back(2399976ULL);
    scripted.push_back(3600000ULL);
    // pair 3 (rev/read): 0.250001s -> 1 correction
    scripted.push_back(3600000ULL);
    scripted.push_back(6000024ULL);
    scripted.push_back(3600000ULL);
    // pair 4 (rev/write): persistent 0.010s -> 2-correction cap
    scripted.push_back(3600000ULL);
    scripted.push_back(240000ULL);
    scripted.push_back(240000ULL);
    scripted.push_back(240000ULL);
    // pair 5 (rev/copy): 1.0s -> single_unit (0.010s < 0.100s) -> scales back up!
    scripted.push_back(3600000ULL);
    scripted.push_back(24000000ULL);
    scripted.push_back(3600000ULL);
    scripted.push_back(240000ULL);
    scripted.push_back(3600000ULL);
    // Remaining 12 pairs (s64..random) stay in-window on pilot (default 3,600,000 ticks).
    reset_mock_timer_state(3600000ULL, scripted);
    TimerSystemCalls timer_calls;
    timer_calls.absolute_time = pat_mock_absolute_time;
    timer_calls.timebase_info = pat_mock_timebase_info;
    set_timer_system_calls_for_testing(timer_calls);

    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    reset_timer_system_calls_for_testing();
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_TRUE(result.run_accepted);

    const auto& fwd_r = result.calibration_attempts[fwd_idx][0];
    ASSERT_EQ(fwd_r.size(), 3u);
    EXPECT_EQ(fwd_r[1].purpose, "pilot");
    EXPECT_EQ(fwd_r[1].duration_quality, "within-target-window");
    EXPECT_EQ(fwd_r[2].purpose, "frozen_measurement_warmup");

    const auto& fwd_w = result.calibration_attempts[fwd_idx][1];
    ASSERT_EQ(fwd_w.size(), 3u);
    EXPECT_EQ(fwd_w[1].purpose, "pilot");
    EXPECT_EQ(fwd_w[1].duration_quality, "within-target-window");
    EXPECT_EQ(fwd_w[2].purpose, "frozen_measurement_warmup");

    const auto& fwd_c = result.calibration_attempts[fwd_idx][2];
    ASSERT_EQ(fwd_c.size(), 4u);
    EXPECT_EQ(fwd_c[1].duration_quality, "below-target-window");
    EXPECT_EQ(fwd_c[2].purpose, "correction");

    const auto& rev_r = result.calibration_attempts[rev_idx][0];
    ASSERT_EQ(rev_r.size(), 4u);
    EXPECT_EQ(rev_r[1].duration_quality, "above-target-window");
    EXPECT_EQ(rev_r[2].purpose, "correction");

    const auto& rev_w = result.calibration_attempts[rev_idx][1];
    ASSERT_EQ(rev_w.size(), 5u);
    EXPECT_EQ(rev_w[1].purpose, "pilot");
    EXPECT_EQ(rev_w[2].purpose, "correction");
    EXPECT_EQ(rev_w[3].purpose, "correction");
    EXPECT_EQ(rev_w[4].purpose, "frozen_measurement_warmup");

    const auto& rev_c = result.calibration_attempts[rev_idx][2];
    ASSERT_EQ(rev_c.size(), 6u);
    EXPECT_EQ(rev_c[2].purpose, "single_unit_confirmation_warmup");
    EXPECT_EQ(rev_c[3].purpose, "single_unit_confirmation");
    EXPECT_EQ(rev_c[3].duration_quality, "below-target-window");
    EXPECT_EQ(result.operation_plans[*rev_c[3].plan_ref].passes, 1u);
    EXPECT_EQ(rev_c[4].purpose, "correction");
    EXPECT_GT(result.operation_plans[*rev_c[4].plan_ref].passes, 1u);
    EXPECT_EQ(rev_c[5].purpose, "frozen_measurement_warmup");
    EXPECT_EQ(result.operation_plans[*result.frozen_plan_refs[rev_idx][2]].passes,
              result.operation_plans[*rev_c[4].plan_ref].passes);
    (void)s64_idx;
    (void)s4k_idx;
    (void)s16k_idx;
  }

  release_patterns_resources(res);
}

TEST_F(PatternsExecutorRunnerTest,
       CheckpointsSignalCompletionWinsSentinelsAndUniqueWritePatterns) {
  reset_mock_timer_state(3600000ULL);
  TimerSystemCalls timer_calls;
  timer_calls.absolute_time = pat_mock_absolute_time;
  timer_calls.timebase_info = pat_mock_timebase_info;
  set_timer_system_calls_for_testing(timer_calls);

  const auto cfg9 = make_config(1, 2, 2, 9);
  PatternsWorkPlan plan9 = build_patterns_work_plan(cfg9, make_sys_info());
  ASSERT_TRUE(plan9.valid);
  PatternsResourceSet res;
  ASSERT_TRUE(prepare_patterns_resources(plan9, false, res));

  // 1. Bounded checkpoints for loop_count=9 -> K=2: Progress after loops 1,3,5,7 (4) + 1 terminal.
  {
    PatternsResult result;
    std::vector<size_t> progress_loops;
    size_t terminal_calls = 0;
    PatternsRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.observe_file_writes = [&]() {
      return std::make_pair(progress_loops.size(), progress_loops.size());
    };
    hooks.checkpoint = [&](const PatternsResult& r,
                           PatternsCheckpointKind kind) {
      if (kind == PatternsCheckpointKind::Progress) {
        progress_loops.push_back(r.counters.completed_loops);
      } else {
        ++terminal_calls;
      }
      return EXIT_SUCCESS;
    };
    ASSERT_EQ(run_patterns_suite(cfg9, plan9, res, result, hooks), EXIT_SUCCESS);
    EXPECT_EQ(result.snapshot_interval_loops, 2u);
    EXPECT_EQ(progress_loops, (std::vector<size_t>{2u, 4u, 6u, 8u}));
    EXPECT_EQ(terminal_calls, 1u);
    EXPECT_EQ(result.prior_file_writer_attempts, 4u);
    EXPECT_EQ(result.prior_successful_file_writes, 4u);
    EXPECT_TRUE(result.terminal_checkpoint_attempted);
    EXPECT_TRUE(result.terminal_checkpoint_completed);

    // Verify all write_pattern values across calibration + measurements are odd, non-zero, distinct,
    // and sparse Write / all Copy have valid pre_op_sentinel != write_pattern.
    std::unordered_set<uint64_t> seen_writes;
    size_t write_count = 0;
    size_t sparse_write_sentinel_count = 0;
    size_t copy_sentinel_count = 0;
    const auto check_exec = [&](const PatternsOperationPlan& op_plan,
                                const PatternsOperationExecution& ex) {
      if (op_plan.operation == PatternOperation::Write) {
        ASSERT_TRUE(ex.write_pattern.has_value());
        EXPECT_NE(*ex.write_pattern, 0ULL);
        EXPECT_EQ((*ex.write_pattern & 1ULL), 1ULL);
        EXPECT_TRUE(seen_writes.insert(*ex.write_pattern).second);
        ++write_count;
        if (op_plan.has_untouched_dst_bytes) {
          ASSERT_TRUE(ex.pre_op_sentinel.has_value());
          EXPECT_NE(*ex.pre_op_sentinel, 0ULL);
          EXPECT_EQ((*ex.pre_op_sentinel & 1ULL), 1ULL);
          EXPECT_NE(*ex.pre_op_sentinel, *ex.write_pattern);
          ++sparse_write_sentinel_count;
        } else {
          EXPECT_FALSE(ex.pre_op_sentinel.has_value());
        }
      } else if (op_plan.operation == PatternOperation::Copy) {
        ASSERT_TRUE(ex.pre_op_sentinel.has_value());
        EXPECT_NE(*ex.pre_op_sentinel, 0ULL);
        EXPECT_EQ((*ex.pre_op_sentinel & 1ULL), 1ULL);
        ++copy_sentinel_count;
      } else {
        EXPECT_FALSE(ex.pre_op_sentinel.has_value());
      }
    };
    for (const auto& pat_atts : result.calibration_attempts) {
      for (const auto& op_atts : pat_atts) {
        for (const auto& att : op_atts) {
          check_exec(result.operation_plans.at(*att.plan_ref), att.execution);
        }
      }
    }
    for (const auto& row : result.measurements) {
      check_exec(result.operation_plans.at(*row.plan_ref), row.execution);
    }
    EXPECT_GT(write_count, 0u);
    EXPECT_GT(sparse_write_sentinel_count, 0u);
    EXPECT_GT(copy_sentinel_count, 0u);
  }

  // 2. Checkpoint failure is terminal and never retried.
  {
    PatternsResult result;
    size_t attempts = 0;
    PatternsRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.checkpoint = [&](const PatternsResult&, PatternsCheckpointKind) {
      ++attempts;
      return EXIT_FAILURE;
    };
    EXPECT_EQ(run_patterns_suite(cfg9, plan9, res, result, hooks), EXIT_FAILURE);
    EXPECT_EQ(attempts, 1u);
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_EQ(result.status, PatternsRunStatus::Failed);
    EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
    EXPECT_FALSE(result.run_accepted);
  }

  // 2b. Mid-run Progress checkpoint failure after loop 0 (`loop_count = 3`).
  {
    const auto cfg3 = make_config(1, 2, 2, 3);
    PatternsWorkPlan plan3 = build_patterns_work_plan(cfg3, make_sys_info());
    ASSERT_TRUE(plan3.valid);
    PatternsResult result;
    size_t progress_calls = 0;
    size_t terminal_calls = 0;
    PatternsRunnerHooks hooks;
    hooks.progress_snapshots = true;
    hooks.checkpoint = [&](const PatternsResult&, PatternsCheckpointKind kind) {
      if (kind == PatternsCheckpointKind::Progress) {
        ++progress_calls;
        return EXIT_FAILURE;
      }
      ++terminal_calls;
      return EXIT_SUCCESS;
    };
    EXPECT_EQ(run_patterns_suite(cfg3, plan3, res, result, hooks), EXIT_FAILURE);
    EXPECT_EQ(progress_calls, 1u);
    EXPECT_EQ(terminal_calls, 0u);
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_FALSE(result.terminal_checkpoint_attempted);
    EXPECT_EQ(result.snapshot_request, "progress");
    EXPECT_EQ(result.status, PatternsRunStatus::Failed);
    EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 18u);
    EXPECT_EQ(result.counters.terminal_measurements, 54u);
    ASSERT_EQ(result.measurements.size(), 54u);
    for (size_t i = 0; i < 18u; ++i) {
      EXPECT_EQ(result.measurements[i].status, PatternsMeasurementStatus::Measured);
    }
    for (size_t i = 18u; i < 54u; ++i) {
      EXPECT_EQ(result.measurements[i].status, PatternsMeasurementStatus::Failed);
      EXPECT_EQ(result.measurements[i].reason_code,
                "not-run-after-runtime-failure");
    }
  }

  // 3. Mid-run interruption after measurement #2.
  {
    PatternsResult result;
    std::atomic<bool> stop{false};
    size_t executed = 0;
    PatternsRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop.load(); };
    hooks.before_measurement_hook = [&](size_t, size_t) {
      if (executed >= 2) {
        stop.store(true);
      }
      ++executed;
    };
    EXPECT_EQ(run_patterns_suite(cfg9, plan9, res, result, hooks), EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Interrupted);
    EXPECT_TRUE(result.interruption_requested);
    EXPECT_FALSE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 2u);
    EXPECT_EQ(result.measurements[2].status,
              PatternsMeasurementStatus::Interrupted);
  }

  // 4. Completion-wins when signal is armed during the FINAL measurement.
  {
    const auto cfg2 = make_config(1, 2, 2, 2);
    std::atomic<bool> armed{false};
    std::atomic<int> post_arm_checks{0};
    PatternsRunnerHooks hooks;
    hooks.stop_requested = [&]() {
      if (!armed.load()) {
        return false;
      }
      return post_arm_checks.fetch_add(1) >= 1;
    };
    hooks.before_measurement_hook = [&](size_t loop_idx, size_t pos) {
      if (loop_idx + 1 == cfg2.loop_count &&
          pos + 1 == plan9.active_patterns.size() * kPatternsOperationCount) {
        armed.store(true);
      }
    };
    PatternsResult result;
    EXPECT_EQ(run_patterns_suite(cfg2, plan9, res, result, hooks),
              EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_EQ(result.reason_code, "complete");
    EXPECT_TRUE(result.results_complete);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_FALSE(result.interruption_requested);
  }

  reset_timer_system_calls_for_testing();
  release_patterns_resources(res);
}

TEST_F(PatternsExecutorRunnerTest,
       RelativeToForwardCvThresholdsQualityWarningsAndConsoleOutput) {
  const auto cfg = make_config(1, 2, 2, 3, true);
  PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  PatternsResourceSet res;
  ASSERT_TRUE(prepare_patterns_resources(plan, true, res));

  // Script loop durations per pattern:
  // For streaming patterns (SequentialForward, SequentialReverse, Strided64) and sparse (Strided4096):
  // Give Read op ~7% CV across 3 loops (e.g. ticks 3600000, 3900000, 4200000) so streaming triggers cv_warning (>5%)
  // and sparse does NOT trigger cv_warning (<=10%).
  // Order in loop 0: pat 0..5, op 0..2; loop 1: rotated; loop 2: rotated.
  // Actually simplest: script ALL measurements with alternating steps across loops:
  // Loop 0 -> 3600000, Loop 1 -> 3900000, Loop 2 -> 4200000 (produces ~7.7% CV on EVERY pattern/op!).
  std::vector<uint64_t> scripted;
  // 18 frozen_measurement_warmup attempts (6 active patterns * 3 ops):
  for (size_t i = 0; i < 18; ++i) {
    scripted.push_back(3600000ULL);
  }
  // Loop 0 (18 measurements): 3,600,000 ticks (0.150 s)
  for (size_t i = 0; i < 18; ++i) {
    scripted.push_back(3600000ULL);
  }
  // Loop 1 (18 measurements): 3,900,000 ticks (0.1625 s)
  for (size_t i = 0; i < 18; ++i) {
    scripted.push_back(3900000ULL);
  }
  // Loop 2 (18 measurements): 4,200,000 ticks (0.175 s)
  for (size_t i = 0; i < 18; ++i) {
    scripted.push_back(4200000ULL);
  }

  reset_mock_timer_state(3600000ULL, scripted);
  TimerSystemCalls timer_calls;
  timer_calls.absolute_time = pat_mock_absolute_time;
  timer_calls.timebase_info = pat_mock_timebase_info;
  set_timer_system_calls_for_testing(timer_calls);

  PatternsExecutorTestControl ctrl;
  ctrl.set_worker_qos = fail_worker_qos;
  PatternsRunnerHooks hooks;
  hooks.executor_test_control = &ctrl;

  PatternsResult result;
  ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, hooks), EXIT_SUCCESS);
  reset_timer_system_calls_for_testing();

  // Check relative_to_forward_pct: sequential_forward == 100.0, others == 100.0 * val / fwd.
  const size_t fwd_idx = patterns_kind_index(PatternKind::SequentialForward);
  for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
    const double fwd_val = *result.aggregates[fwd_idx][op_i].value;
    EXPECT_NEAR(*result.aggregates[fwd_idx][op_i].relative_to_forward_pct,
                100.0, 1e-9);
    for (PatternKind kind : plan.active_patterns) {
      const auto& agg = result.aggregates[patterns_kind_index(kind)][op_i];
      ASSERT_TRUE(agg.relative_to_forward_pct.has_value());
      EXPECT_NEAR(*agg.relative_to_forward_pct, 100.0 * (*agg.value) / fwd_val,
                  1e-9);
    }
  }

  // Check CV thresholds: 5.0% streaming (triggers) vs 10.0% sparse (does not trigger).
  EXPECT_DOUBLE_EQ(
      result.aggregates[patterns_kind_index(PatternKind::SequentialForward)][0]
          .cv_warning_threshold_pct,
      5.0);
  EXPECT_TRUE(
      result.aggregates[patterns_kind_index(PatternKind::SequentialForward)][0]
          .cv_warning);
  EXPECT_TRUE(
      result.aggregates[patterns_kind_index(PatternKind::SequentialReverse)][0]
          .cv_warning);
  EXPECT_TRUE(result.aggregates[patterns_kind_index(PatternKind::Strided64)][0]
                  .cv_warning);

  EXPECT_DOUBLE_EQ(
      result.aggregates[patterns_kind_index(PatternKind::Strided4096)][0]
          .cv_warning_threshold_pct,
      10.0);
  EXPECT_FALSE(
      result.aggregates[patterns_kind_index(PatternKind::Strided4096)][0]
          .cv_warning);
  EXPECT_FALSE(
      result.aggregates[patterns_kind_index(PatternKind::Strided16384)][0]
          .cv_warning);
  EXPECT_FALSE(
      result.aggregates[patterns_kind_index(PatternKind::Random)][0].cv_warning);

  // Quality warnings collector (with non-nominal env, main-thread QoS fail, madvise fail, duration warning, high-CV).
  PatternsResultMetadata meta = make_metadata(false);
  meta.environment_start.thermal_state = "serious";
  meta.environment_end.low_power_mode_available = true;
  meta.environment_end.low_power_mode_enabled = true;
  meta.main_thread_qos.requested = true;
  meta.main_thread_qos.applied = false;
  res.src.madvise_applied = false;
  result.quality_warnings = {"custom-warn", "custom-warn"};

  // Test auto-calibration duration warning token too.
  auto auto_cfg = cfg;
  auto_cfg.iterations = std::nullopt;
  result.measurements[0].duration_quality = "below-target-window";

  const auto tokens =
      collect_patterns_quality_warning_tokens(auto_cfg, plan, res, meta, result);
  auto has_token = [&](const std::string& tok) {
    return std::find(tokens.begin(), tokens.end(), tok) != tokens.end();
  };
  EXPECT_EQ(std::count(tokens.begin(), tokens.end(), "custom-warn"), 1);
  EXPECT_TRUE(has_token("environment-not-nominal"));
  EXPECT_TRUE(has_token("main-thread-qos-not-applied"));
  EXPECT_TRUE(has_token("worker-qos-not-applied"));
  EXPECT_TRUE(has_token("madvise-random-not-applied"));
  EXPECT_TRUE(has_token("sequential_forward-read-high-cv"));
  EXPECT_FALSE(has_token("strided_4096-read-high-cv"));
  EXPECT_TRUE(has_token(
      std::string(patterns_kind_to_string(result.measurements[0].pattern)) +
      "-" + patterns_operation_to_string(result.measurements[0].operation) +
      "-duration-below-target-window"));

  // Console output checks: help, configuration, summary.
  {
    std::ostringstream help_oss;
    print_patterns_help("memory_benchmark", help_oss);
    const std::string help_text = help_oss.str();
    EXPECT_NE(help_text.find("Usage: memory_benchmark -P"), std::string::npos);
    for (const char* opt : {"-P, --patterns", "-b, --buffer-size", "-t, --threads",
                            "-i, --iterations", "-r, --count", "--seed",
                            "-u, --non-cacheable", "-o, --output", "-S, --sweep",
                            "-X, --sweep-max-runs", "-h, --help"}) {
      EXPECT_NE(help_text.find(opt), std::string::npos) << opt;
    }
    EXPECT_EQ(help_text.find("--cache-size"), std::string::npos);
    EXPECT_EQ(help_text.find("-k,"), std::string::npos);
  }

  {
    StreamCapture cap_out(std::cout);
    print_patterns_configuration(cfg, plan, meta);
    const std::string conf_text = cap_out.oss.str();
    EXPECT_NE(conf_text.find("(generated)"), std::string::npos);
    EXPECT_NE(conf_text.find("Strided 2MiB]: skipped (buffer-too-small-for-stride-transition)"),
              std::string::npos);
  }

  {
    StreamCapture cap_out(std::cout);
    StreamCapture cap_err(std::cerr);
    print_patterns_console_summary(auto_cfg, plan, res, meta, result);
    const std::string out_text = cap_out.oss.str();
    const std::string err_text = cap_err.oss.str();
    EXPECT_NE(out_text.find("P50="), std::string::npos);
    EXPECT_NE(out_text.find("% vs forward)"), std::string::npos);
    EXPECT_NE(out_text.find("skipped (buffer-too-small-for-stride-transition)"),
              std::string::npos);
    EXPECT_NE(err_text.find("environment-not-nominal"), std::string::npos);
  }

  release_patterns_resources(res);
}

class PatternsExecutorRunnerIntegrationTest : public ::testing::Test {};

TEST_F(PatternsExecutorRunnerIntegrationTest,
       RealAll7Patterns3OpsWithStridedFullAndPartialPhases) {
  const auto cfg = make_config(3, 2, 2, 1);
  PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid);
  ASSERT_EQ(plan.active_patterns.size(), 7u);

  PatternsResourceSet res;
  ASSERT_TRUE(prepare_patterns_resources(plan, false, res));
  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  // Execute all 7 patterns x 3 ops with passes=3 on Strided64 (full_cycles=1, rem_phases=1)
  // and passes=2 on all other patterns.
  for (PatternKind kind : plan.active_patterns) {
    const auto& pplan = plan.patterns[patterns_kind_index(kind)];
    const size_t passes = (kind == PatternKind::Strided64) ? 3u : 2u;
    for (PatternOperation op :
         {PatternOperation::Read, PatternOperation::Write,
          PatternOperation::Copy}) {
      const auto op_plan =
          make_op_plan_with_checksum(plan, res, kind, op, passes);
      ASSERT_TRUE(op_plan.valid);
      std::optional<uint64_t> write_seed;
      std::optional<uint64_t> write_pat;
      if (op == PatternOperation::Write) {
        write_seed = derive_patterns_write_attempt_seed(op_plan.op_seed, 1ULL);
        write_pat = derive_patterns_write_pattern(*write_seed);
      }
      std::optional<uint64_t> sentinel;
      if (op == PatternOperation::Copy ||
          (op == PatternOperation::Write && op_plan.has_untouched_dst_bytes)) {
        sentinel = derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed,
                                                   10ULL);
        if (write_pat.has_value() && *sentinel == *write_pat) {
          sentinel = derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed,
                                                     11ULL);
        }
      }

      const auto ex = execute_patterns_operation(
          cfg, pplan, op_plan, res, *timer_opt, write_seed, write_pat, sentinel,
          nullptr);
      EXPECT_TRUE(ex.valid) << patterns_kind_to_string(kind) << " "
                            << patterns_operation_to_string(op) << ": "
                            << ex.reason_code;
      EXPECT_TRUE(ex.cpu_raw.has_value());
      EXPECT_GT(ex.elapsed_seconds, 0.0);
      for (const auto& chk : ex.validation_checks) {
        if (chk.applicable) {
          EXPECT_TRUE(chk.evaluated.value_or(false)) << chk.kind;
          EXPECT_TRUE(chk.valid.value_or(false)) << chk.kind;
        }
      }
    }
  }

  release_patterns_resources(res);
}

TEST_F(PatternsExecutorRunnerIntegrationTest,
       RealRunner3MiBActiveAnd1MiBSkipped2MiB) {
  for (size_t buf_mb : {size_t{3}, size_t{1}}) {
    const auto cfg = make_config(buf_mb, 2, 2, 3);
    PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
    ASSERT_TRUE(plan.valid);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan, false, res));

    PatternsResult result;
    ASSERT_EQ(run_patterns_suite(cfg, plan, res, result, {}), EXIT_SUCCESS);
    EXPECT_EQ(result.status, PatternsRunStatus::Complete);
    EXPECT_TRUE(result.results_complete);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_TRUE(result.operation_order_balance_complete);

    const size_t expected_active = (buf_mb == 3) ? 7u : 6u;
    EXPECT_EQ(result.counters.planned_measurements, expected_active * 3u * 3u);
    EXPECT_EQ(result.counters.measured_measurements, expected_active * 3u * 3u);

    const size_t s2m_idx = patterns_kind_index(PatternKind::Strided2MiB);
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      if (buf_mb == 3) {
        EXPECT_EQ(result.aggregates[s2m_idx][op_i].status,
                  PatternsAggregateStatus::Measured);
      } else {
        EXPECT_EQ(result.aggregates[s2m_idx][op_i].status,
                  PatternsAggregateStatus::Skipped);
        EXPECT_EQ(result.aggregates[s2m_idx][op_i].reason_code,
                  PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);
      }
    }

    const PatternsResultMetadata meta = make_metadata(true);
    StreamCapture cap_out(std::cout);
    StreamCapture cap_err(std::cerr);
    print_patterns_configuration(cfg, plan, meta);
    print_patterns_console_summary(cfg, plan, res, meta, result);
    EXPECT_NE(cap_out.oss.str().find("Sequential Forward"), std::string::npos);

    release_patterns_resources(res);
  }
}

}  // namespace
