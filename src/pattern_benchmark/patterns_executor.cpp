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

/**
 * @file patterns_executor.cpp
 * @brief Persistent buffer preparation, synchronized ARM64 worker gate, and cold checks
 */

#include "pattern_benchmark/patterns_executor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <system_error>
#include <thread>

#include <mach/mach.h>
#include <pthread/qos.h>

#include "asm/asm_functions.h"
#include "core/config/constants.h"

namespace {

int apply_worker_qos(const PatternsExecutorTestControl* test_control,
                     size_t worker_index) noexcept {
  if (test_control != nullptr && test_control->set_worker_qos != nullptr) {
    return test_control->set_worker_qos(test_control->qos_context,
                                        worker_index);
  }
  return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

void join_threads(std::vector<std::thread>& threads) noexcept {
  for (std::thread& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

bool equal_worker_checksum(
    const PatternsExpectedWorkerChecksum& lhs,
    const PatternsExpectedWorkerChecksum& rhs) noexcept {
  return lhs.worker_index == rhs.worker_index &&
         lhs.state_a == rhs.state_a &&
         lhs.state_b == rhs.state_b &&
         lhs.exact_bytes_read == rhs.exact_bytes_read &&
         lhs.pass_count == rhs.pass_count;
}

bool random_workload_matches_plan(const PatternPlanEntry& pattern_plan,
                                  const PatternsResourceSet& resources) noexcept {
  if (pattern_plan.kind != PatternKind::Random) {
    return true;
  }
  const size_t workers = pattern_plan.effective_workers;
  const auto& rw = resources.random_workload;
  if (rw.workers.size() != workers ||
      rw.sorted_validation_indices.size() != workers ||
      rw.worker_pass_xors.size() != workers) {
    return false;
  }
  for (size_t w = 0; w < workers; ++w) {
    const auto& span = pattern_plan.worker_spans[w];
    const auto& rw_worker = rw.workers[w];
    const auto& sorted_vec = rw.sorted_validation_indices[w];
    if (rw_worker.offset_bytes != span.offset_bytes ||
        rw_worker.span_bytes != span.span_bytes ||
        rw_worker.indices.size() != span.random_access_count ||
        sorted_vec.size() != span.random_access_count ||
        rw_worker.indices.empty() ||
        !std::is_sorted(sorted_vec.begin(), sorted_vec.end())) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::array<PatternsValidationCheck, kPatternsValidationCheckCount>
make_default_patterns_checks(PatternOperation op,
                             bool has_untouched_dst_bytes) {
  std::array<PatternsValidationCheck, kPatternsValidationCheckCount> checks;
  checks[0] = {"post-validation-structure", true, false, std::nullopt,
               "not-evaluated"};

  const bool is_read = (op == PatternOperation::Read);
  const bool is_write = (op == PatternOperation::Write);
  const bool is_copy = (op == PatternOperation::Copy);
  const bool check_untouched = (is_write || is_copy) && has_untouched_dst_bytes;
  const bool uses_src = (is_read || is_copy);

  checks[1] = {"read-checksum", is_read,
               is_read ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_read ? "not-evaluated" : "not-applicable"};
  checks[2] = {"write-final-touched-bytes", is_write,
               is_write ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_write ? "not-evaluated" : "not-applicable"};
  checks[3] = {"copy-final-touched-bytes", is_copy,
               is_copy ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_copy ? "not-evaluated" : "not-applicable"};
  checks[4] = {"untouched-sentinel-isolation", check_untouched,
               check_untouched ? std::optional<bool>(false) : std::nullopt,
               std::nullopt,
               check_untouched ? "not-evaluated" : "not-applicable"};
  checks[5] = {"src-readonly-protection", uses_src,
               uses_src ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, uses_src ? "not-evaluated" : "not-applicable"};
  return checks;
}

bool prepare_patterns_resources(
    PatternsWorkPlan& plan,
    bool non_cacheable,
    PatternsResourceSet& resources,
    const std::function<bool()>& stop_requested) {
  release_patterns_resources(resources);
  resources = PatternsResourceSet{};

  if (!plan.valid || !plan.memory_budget.valid || plan.buffer_bytes == 0 ||
      plan.buffer_worker_spans.empty()) {
    resources.reason_code = PatternsExecutorReason::INVALID_RESOURCES;
    return false;
  }

  if (stop_requested && stop_requested()) {
    resources.reason_code = PatternsExecutorReason::PREPARATION_INTERRUPTED;
    return false;
  }

  resources.src = allocate_buffer_with_evidence(plan.buffer_bytes, non_cacheable,
                                                "patterns_src");
  resources.dst = allocate_buffer_with_evidence(plan.buffer_bytes, non_cacheable,
                                                "patterns_dst");
  if (resources.src.ptr == nullptr || resources.dst.ptr == nullptr) {
    release_patterns_resources(resources);
    resources.reason_code = PatternsExecutorReason::BUFFER_ALLOCATION_FAILED;
    return false;
  }

  if (stop_requested && stop_requested()) {
    release_patterns_resources(resources);
    resources.reason_code = PatternsExecutorReason::PREPARATION_INTERRUPTED;
    return false;
  }

  initialize_patterns_src_buffer(resources.src.ptr.get(), plan.buffer_bytes,
                                 plan.src_seed, plan.buffer_worker_spans);

  const size_t protect_size = resources.src.mapped_bytes > 0
                                  ? resources.src.mapped_bytes
                                  : plan.buffer_bytes;
  if (!protect_buffer_read_only_with_evidence(resources.src, protect_size)) {
    // Preserve `resources.src` evidence (`read_only_protected`, `mprotect_errno`).
    resources.valid = false;
    resources.reason_code = PatternsExecutorReason::SRC_READONLY_FAILED;
    return false;
  }

  const uint64_t initial_sentinel =
      derive_patterns_pre_op_sentinel(plan.pre_op_sentinel_seed, 0ULL);
  initialize_patterns_dst_sentinel(resources.dst.ptr.get(), plan.buffer_bytes,
                                   initial_sentinel, plan.buffer_worker_spans);

  if (stop_requested && stop_requested()) {
    release_patterns_resources(resources);
    resources.reason_code = PatternsExecutorReason::PREPARATION_INTERRUPTED;
    return false;
  }

  const PatternPlanEntry* rand_plan = plan.find_pattern(PatternKind::Random);
  if (rand_plan != nullptr && rand_plan->enabled) {
    std::vector<size_t> worker_access_counts;
    worker_access_counts.reserve(rand_plan->worker_spans.size());
    for (const auto& span : rand_plan->worker_spans) {
      worker_access_counts.push_back(span.random_access_count);
    }

    resources.random_workload = generate_patterns_random_indices(
        plan.random_domain_seed, plan.src_seed, rand_plan->worker_spans,
        worker_access_counts);

    const size_t selected =
        resources.random_workload.selected_accesses_per_pass;
    if (resources.random_workload.workers.size() !=
            rand_plan->worker_spans.size() ||
        selected == 0 ||
        selected != rand_plan->random_workload.selected_accesses_per_pass ||
        selected > kPatternsMaxRandomAccesses ||
        resources.random_workload.random_indices_sha256.empty()) {
      release_patterns_resources(resources);
      resources.reason_code = PatternsExecutorReason::INVALID_RESOURCES;
      return false;
    }

    auto& plan_random =
        plan.patterns[patterns_kind_index(PatternKind::Random)].random_workload;
    plan_random.selected_accesses_per_pass = selected;
    plan_random.distinct_address_count =
        resources.random_workload.distinct_address_count;
    plan_random.logical_working_set_bytes =
        resources.random_workload.logical_working_set_bytes;
    plan_random.random_indices_sha256 =
        resources.random_workload.random_indices_sha256;

    resources.random_index_table_bytes = selected * sizeof(size_t);
    resources.random_validation_index_bytes = selected * sizeof(size_t);
  }

  if (stop_requested && stop_requested()) {
    release_patterns_resources(resources);
    resources.reason_code = PatternsExecutorReason::PREPARATION_INTERRUPTED;
    return false;
  }

  resources.valid = true;
  resources.reason_code = PatternsExecutorReason::VALID;
  return true;
}

void release_patterns_resources(PatternsResourceSet& resources) noexcept {
  resources.src.ptr.reset();
  resources.dst.ptr.reset();
  resources.random_workload = PatternsRandomWorkload{};
  resources.random_index_table_bytes = 0;
  resources.random_validation_index_bytes = 0;
  resources.valid = false;
}

PatternsOperationExecution execute_patterns_operation(
    const PatternsWorkPlanConfig& config,
    const PatternPlanEntry& pattern_plan,
    const PatternsOperationPlan& op_plan,
    const PatternsResourceSet& resources,
    HighResTimer& timer,
    std::optional<uint64_t> write_attempt_seed,
    std::optional<uint64_t> write_pattern,
    std::optional<uint64_t> pre_op_sentinel,
    const PatternsExecutorTestControl* test_control) noexcept {
  PatternsOperationExecution result;
  result.requested_workers = config.requested_workers;
  result.validation_checks = make_default_patterns_checks(
      op_plan.operation, op_plan.has_untouched_dst_bytes);
  result.checksum_applicable = (op_plan.operation == PatternOperation::Read);

  try {
    if (!pattern_plan.enabled || !op_plan.valid ||
        op_plan.pattern != pattern_plan.kind || !resources.valid ||
        resources.src.ptr == nullptr || resources.dst.ptr == nullptr ||
        pattern_plan.buffer_bytes == 0 ||
        pattern_plan.effective_workers == 0 ||
        pattern_plan.worker_spans.size() != pattern_plan.effective_workers) {
      result.reason_code = PatternsExecutorReason::INVALID_RESOURCES;
      return result;
    }

    if (patterns_kind_is_strided(pattern_plan.kind) &&
        (pattern_plan.stride_bytes < 64 ||
         (pattern_plan.stride_bytes % kPatternsAccessSizeBytes) != 0)) {
      result.reason_code = PatternsExecutorReason::INVALID_OPERATION_PLAN;
      return result;
    }

    if (pattern_plan.kind == PatternKind::Random &&
        !random_workload_matches_plan(pattern_plan, resources)) {
      result.reason_code = PatternsExecutorReason::INVALID_RESOURCES;
      return result;
    }

    if (op_plan.operation == PatternOperation::Write &&
        (!write_pattern.has_value() || *write_pattern == 0 ||
         (*write_pattern & 1ULL) == 0)) {
      result.reason_code = PatternsExecutorReason::INVALID_OPERATION_PLAN;
      return result;
    }

    const bool needs_sentinel =
        (op_plan.operation == PatternOperation::Copy) ||
        (op_plan.operation == PatternOperation::Write &&
         op_plan.has_untouched_dst_bytes);
    if (needs_sentinel) {
      if (!pre_op_sentinel.has_value() || *pre_op_sentinel == 0 ||
          (*pre_op_sentinel & 1ULL) == 0) {
        result.reason_code = PatternsExecutorReason::INVALID_OPERATION_PLAN;
        return result;
      }
      if (op_plan.operation == PatternOperation::Write &&
          *pre_op_sentinel == *write_pattern) {
        result.reason_code = PatternsExecutorReason::INVALID_OPERATION_PLAN;
        return result;
      }
    }

    result.available = true;
    result.write_pattern =
        (op_plan.operation == PatternOperation::Write) ? write_pattern
                                                       : std::nullopt;
    result.attempt_seed =
        (op_plan.operation == PatternOperation::Write) ? write_attempt_seed
                                                       : std::nullopt;
    result.pre_op_sentinel = needs_sentinel ? pre_op_sentinel : std::nullopt;

    if (needs_sentinel) {
      initialize_patterns_dst_sentinel(
          resources.dst.ptr.get(), pattern_plan.buffer_bytes, *pre_op_sentinel,
          pattern_plan.worker_spans);
    }

    const size_t worker_count = pattern_plan.effective_workers;
    if (op_plan.operation == PatternOperation::Read) {
      result.actual_worker_checksums.resize(worker_count);
    }

    std::vector<uint8_t> worker_succeeded(worker_count, 0);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    std::mutex state_mutex;
    std::condition_variable state_cv;
    size_t ready_workers = 0;
    bool start_workers = false;
    bool cancel_workers = false;
    bool measurement_complete = false;
    bool timer_stop_succeeded = false;
    double measured_duration = 0.0;
    std::atomic<size_t> remaining_workers{worker_count};
    std::atomic<size_t> completed_workers{0};
    std::atomic<size_t> qos_successful_workers{0};
    std::atomic<size_t> qos_failed_workers{0};

    uint8_t* const dst_base = static_cast<uint8_t*>(resources.dst.ptr.get());
    const uint8_t* const src_base =
        static_cast<const uint8_t*>(resources.src.ptr.get());
    const uint64_t passes = static_cast<uint64_t>(op_plan.passes);
    const size_t stride_bytes = pattern_plan.stride_bytes;

    try {
      for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        if (test_control != nullptr &&
            test_control->fail_before_worker_index >= 0 &&
            worker_index ==
                static_cast<size_t>(test_control->fail_before_worker_index)) {
          throw std::system_error(
              std::make_error_code(std::errc::resource_unavailable_try_again));
        }

        const PatternsWorkerSpan span = pattern_plan.worker_spans[worker_index];
        threads.emplace_back([&, worker_index, span]() {
          const int qos_ret = apply_worker_qos(test_control, worker_index);
          if (qos_ret == KERN_SUCCESS) {
            qos_successful_workers.fetch_add(1, std::memory_order_relaxed);
          } else {
            qos_failed_workers.fetch_add(1, std::memory_order_relaxed);
          }

          {
            std::unique_lock<std::mutex> lock(state_mutex);
            ++ready_workers;
            state_cv.notify_all();
            state_cv.wait(lock,
                          [&]() { return start_workers || cancel_workers; });
            if (cancel_workers) {
              return;
            }
          }

          bool ok = true;
          if (test_control != nullptr && test_control->force_kernel_fail) {
            ok = false;
          } else if (op_plan.operation == PatternOperation::Read) {
            uint64_t state_a = 0;
            uint64_t state_b = 0;
            initial_patterns_worker_checksum_state(op_plan.op_seed, worker_index,
                                                   state_a, state_b);
            const uint8_t* const chunk_src = src_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            uint64_t exact_bytes_read = 0;

            if (pattern_plan.kind == PatternKind::SequentialForward) {
              for (uint64_t p = 1; p <= passes; ++p) {
                const uint64_t pass_xor =
                    memory_read_loop_asm(chunk_src, chunk_bytes);
                state_a += pass_xor + kPatternsChecksumA * p;
                state_b += (pass_xor ^ kPatternsChecksumB) * p +
                           kPatternsChecksumC;
              }
              exact_bytes_read = passes * static_cast<uint64_t>(chunk_bytes);
            } else if (pattern_plan.kind == PatternKind::SequentialReverse) {
              for (uint64_t p = 1; p <= passes; ++p) {
                const uint64_t pass_xor =
                    memory_read_reverse_loop_asm(chunk_src, chunk_bytes);
                state_a += pass_xor + kPatternsChecksumA * p;
                state_b += (pass_xor ^ kPatternsChecksumB) * p +
                           kPatternsChecksumC;
              }
              exact_bytes_read = passes * static_cast<uint64_t>(chunk_bytes);
            } else if (patterns_kind_is_strided(pattern_plan.kind)) {
              const uint64_t Q =
                  static_cast<uint64_t>(stride_bytes / kPatternsAccessSizeBytes);
              const uint64_t full_cycles = passes / Q;
              const uint64_t rem_phases = passes % Q;
              for (uint64_t c = 1; c <= full_cycles; ++c) {
                const uint64_t cycle_xor = memory_read_strided_phased_loop_asm(
                    chunk_src, chunk_bytes, stride_bytes, static_cast<size_t>(Q),
                    0);
                state_a += cycle_xor + kPatternsChecksumA * c;
                state_b += (cycle_xor ^ kPatternsChecksumB) * c +
                           kPatternsChecksumC;
              }
              if (rem_phases > 0ULL) {
                const uint64_t partial_xor = memory_read_strided_phased_loop_asm(
                    chunk_src, chunk_bytes, stride_bytes,
                    static_cast<size_t>(rem_phases), 0);
                const uint64_t step = full_cycles + 1ULL;
                state_a += partial_xor + kPatternsChecksumA * step;
                state_b += (partial_xor ^ kPatternsChecksumB) * step +
                           kPatternsChecksumC;
              }
              const uint64_t num_slots =
                  static_cast<uint64_t>(chunk_bytes / kPatternsAccessSizeBytes);
              const uint64_t full_blocks = num_slots / Q;
              const uint64_t tail_slots = num_slots % Q;
              const uint64_t total_slots_read =
                  full_cycles * num_slots + full_blocks * rem_phases +
                  std::min(tail_slots, rem_phases);
              exact_bytes_read = total_slots_read * kPatternsAccessSizeBytes;
            } else {
              // PatternKind::Random
              const auto& w_indices =
                  resources.random_workload.workers[worker_index].indices;
              const size_t* const idx_ptr = w_indices.data();
              const size_t idx_count = w_indices.size();
              for (uint64_t p = 1; p <= passes; ++p) {
                const uint64_t pass_xor =
                    memory_read_random_loop_asm(chunk_src, idx_ptr, idx_count);
                state_a += pass_xor + kPatternsChecksumA * p;
                state_b += (pass_xor ^ kPatternsChecksumB) * p +
                           kPatternsChecksumC;
              }
              exact_bytes_read = passes * static_cast<uint64_t>(idx_count) *
                                 kPatternsAccessSizeBytes;
            }

            PatternsExpectedWorkerChecksum& wout =
                result.actual_worker_checksums[worker_index];
            wout.worker_index = worker_index;
            wout.state_a = state_a;
            wout.state_b = state_b;
            wout.exact_bytes_read = exact_bytes_read;
            wout.pass_count = passes;
          } else if (op_plan.operation == PatternOperation::Write) {
            uint8_t* const chunk_dst = dst_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            const uint64_t pat = *write_pattern;
            if (pattern_plan.kind == PatternKind::SequentialForward) {
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_write_pattern_loop_asm(chunk_dst, chunk_bytes, pat);
              }
            } else if (pattern_plan.kind == PatternKind::SequentialReverse) {
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_write_reverse_pattern_loop_asm(chunk_dst, chunk_bytes,
                                                      pat);
              }
            } else if (patterns_kind_is_strided(pattern_plan.kind)) {
              memory_write_strided_phased_pattern_loop_asm(
                  chunk_dst, chunk_bytes, stride_bytes, op_plan.passes, 0, pat);
            } else {
              // PatternKind::Random
              const auto& w_indices =
                  resources.random_workload.workers[worker_index].indices;
              const size_t* const idx_ptr = w_indices.data();
              const size_t idx_count = w_indices.size();
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_write_random_pattern_loop_asm(chunk_dst, idx_ptr,
                                                     idx_count, pat);
              }
            }
          } else {
            // PatternOperation::Copy
            uint8_t* const chunk_dst = dst_base + span.offset_bytes;
            const uint8_t* const chunk_src = src_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            if (pattern_plan.kind == PatternKind::SequentialForward) {
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_copy_loop_asm(chunk_dst, chunk_src, chunk_bytes);
              }
            } else if (pattern_plan.kind == PatternKind::SequentialReverse) {
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_copy_reverse_loop_asm(chunk_dst, chunk_src, chunk_bytes);
              }
            } else if (patterns_kind_is_strided(pattern_plan.kind)) {
              memory_copy_strided_phased_loop_asm(chunk_dst, chunk_src,
                                                  chunk_bytes, stride_bytes,
                                                  op_plan.passes, 0);
            } else {
              // PatternKind::Random
              const auto& w_indices =
                  resources.random_workload.workers[worker_index].indices;
              const size_t* const idx_ptr = w_indices.data();
              const size_t idx_count = w_indices.size();
              for (uint64_t p = 1; p <= passes; ++p) {
                memory_copy_random_loop_asm(chunk_dst, chunk_src, idx_ptr,
                                            idx_count);
              }
            }
          }

          worker_succeeded[worker_index] = ok ? 1 : 0;
          asm volatile("dsb ish" ::: "memory");
          completed_workers.fetch_add(1, std::memory_order_relaxed);
          if (remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            double duration = 0.0;
            bool stopped = false;
            try {
              duration = timer.stop();
              stopped = true;
            } catch (...) {
              stopped = false;
            }
            {
              std::lock_guard<std::mutex> lock(state_mutex);
              measured_duration = duration;
              timer_stop_succeeded = stopped;
              measurement_complete = true;
            }
            state_cv.notify_one();
          }
        });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        cancel_workers = true;
      }
      state_cv.notify_all();
      join_threads(threads);
      result.created_workers = threads.size();
      result.qos_successful_workers =
          qos_successful_workers.load(std::memory_order_relaxed);
      result.qos_failed_workers =
          qos_failed_workers.load(std::memory_order_relaxed);
      result.worker_startup_failed = true;
      result.reason_code = PatternsExecutorReason::WORKER_STARTUP_FAILED;
      result.validation_checks[0].evaluated = true;
      result.validation_checks[0].valid = false;
      result.validation_checks[0].reason_code =
          PatternsExecutorReason::WORKER_STARTUP_FAILED;
      return result;
    }

    {
      std::unique_lock<std::mutex> lock(state_mutex);
      state_cv.wait(lock, [&]() { return ready_workers == worker_count; });
      timer.start();
      result.timer_started = true;
      start_workers = true;
    }
    state_cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(state_mutex);
      state_cv.wait(lock, [&]() { return measurement_complete; });
    }
    join_threads(threads);

    result.created_workers = threads.size();
    result.completed_workers =
        completed_workers.load(std::memory_order_relaxed);
    result.qos_successful_workers =
        qos_successful_workers.load(std::memory_order_relaxed);
    result.qos_failed_workers =
        qos_failed_workers.load(std::memory_order_relaxed);
    result.timer_stopped = timer_stop_succeeded;

    if (test_control != nullptr && test_control->force_zero_elapsed) {
      measured_duration = 0.0;
      result.cpu_raw = std::nullopt;
    } else if (timer_stop_succeeded) {
      result.cpu_raw = timer.last_snapshot;
    }
    result.elapsed_seconds = measured_duration;
    result.timing_evaluated = result.timer_started && result.timer_stopped;
    result.timing_valid = result.timing_evaluated &&
                          std::isfinite(result.elapsed_seconds) &&
                          result.elapsed_seconds > 0.0;

    result.kernel_succeeded = std::all_of(
        worker_succeeded.begin(), worker_succeeded.end(),
        [](uint8_t s) { return s != 0; });

    if (test_control != nullptr && test_control->corrupt_dst_after_kernel) {
      test_control->corrupt_dst_after_kernel(resources.dst.ptr.get(),
                                             pattern_plan.buffer_bytes);
    }

    // Parallel cold post-validation (outside timer).
    const bool random_ok = random_workload_matches_plan(pattern_plan, resources);
    const bool structure_ok =
        result.created_workers == worker_count &&
        result.completed_workers == worker_count &&
        !result.worker_startup_failed &&
        result.kernel_succeeded &&
        resources.src.mapped_bytes >= pattern_plan.buffer_bytes &&
        resources.dst.mapped_bytes >= pattern_plan.buffer_bytes &&
        random_ok;

    result.validation_checks[0].evaluated = true;
    result.validation_checks[0].valid = structure_ok;
    result.validation_checks[0].reason_code =
        structure_ok ? PatternsExecutorReason::VALID
                     : PatternsExecutorReason::STRUCTURE_VALIDATION_FAILED;

    const auto* sorted_random_ptr =
        (pattern_plan.kind == PatternKind::Random)
            ? &resources.random_workload.sorted_validation_indices
            : nullptr;

    if (op_plan.operation == PatternOperation::Read) {
      if (result.kernel_succeeded) {
        result.actual_run_checksum =
            fold_patterns_worker_checksums(result.actual_worker_checksums);
        bool matches = op_plan.expected_checksum.available &&
                       result.actual_run_checksum.state_a ==
                           op_plan.expected_checksum.run_checksum.state_a &&
                       result.actual_run_checksum.state_b ==
                           op_plan.expected_checksum.run_checksum.state_b &&
                       result.actual_worker_checksums.size() ==
                           op_plan.expected_checksum.workers.size();
        for (size_t w = 0; matches && w < result.actual_worker_checksums.size();
             ++w) {
          matches = equal_worker_checksum(result.actual_worker_checksums[w],
                                          op_plan.expected_checksum.workers[w]);
        }
        result.checksum_evaluated = true;
        result.checksum_valid = matches;
        result.validation_checks[1].evaluated = true;
        result.validation_checks[1].valid = matches;
        result.validation_checks[1].reason_code =
            matches ? PatternsExecutorReason::VALID
                    : PatternsExecutorReason::CHECKSUM_MISMATCH;
      }
    } else if (op_plan.operation == PatternOperation::Write) {
      if (result.kernel_succeeded) {
        const PatternsPostValidationResult pv = validate_patterns_write_buffer(
            pattern_plan.kind, resources.dst.ptr.get(),
            pattern_plan.buffer_bytes, *write_pattern,
            op_plan.has_untouched_dst_bytes,
            op_plan.has_untouched_dst_bytes ? *pre_op_sentinel : 0ULL,
            pattern_plan.stride_bytes, op_plan.passes,
            pattern_plan.worker_spans, sorted_random_ptr);
        result.validation_checks[2].evaluated = true;
        result.validation_checks[2].valid = pv.touched_ok;
        result.validation_checks[2].reason_code =
            pv.touched_ok
                ? PatternsExecutorReason::VALID
                : PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED;
        if (op_plan.has_untouched_dst_bytes) {
          result.validation_checks[4].evaluated = true;
          result.validation_checks[4].valid = pv.untouched_ok;
          result.validation_checks[4].reason_code =
              pv.untouched_ok
                  ? PatternsExecutorReason::VALID
                  : PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED;
        }
      }
    } else {
      // PatternOperation::Copy
      if (result.kernel_succeeded) {
        const PatternsPostValidationResult pv = validate_patterns_copy_buffer(
            pattern_plan.kind, resources.dst.ptr.get(), resources.src.ptr.get(),
            pattern_plan.buffer_bytes, op_plan.has_untouched_dst_bytes,
            *pre_op_sentinel, pattern_plan.stride_bytes, op_plan.passes,
            pattern_plan.worker_spans, sorted_random_ptr);
        result.validation_checks[3].evaluated = true;
        result.validation_checks[3].valid = pv.touched_ok;
        result.validation_checks[3].reason_code =
            pv.touched_ok
                ? PatternsExecutorReason::VALID
                : PatternsExecutorReason::COPY_POST_VALIDATION_FAILED;
        if (op_plan.has_untouched_dst_bytes) {
          result.validation_checks[4].evaluated = true;
          result.validation_checks[4].valid = pv.untouched_ok;
          result.validation_checks[4].reason_code =
              pv.untouched_ok
                  ? PatternsExecutorReason::VALID
                  : PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED;
        }
      }
    }

    if (op_plan.operation == PatternOperation::Read ||
        op_plan.operation == PatternOperation::Copy) {
      const bool ro_ok = resources.src.read_only_protected &&
                         resources.src.mprotect_errno == 0;
      result.validation_checks[5].evaluated = true;
      result.validation_checks[5].valid = ro_ok;
      result.validation_checks[5].reason_code =
          ro_ok ? PatternsExecutorReason::VALID
                : PatternsExecutorReason::SRC_READONLY_FAILED;
    }

    if (!result.kernel_succeeded) {
      result.reason_code = PatternsExecutorReason::KERNEL_FAILED;
      return result;
    }
    if (!result.timing_valid) {
      result.reason_code = PatternsExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }
    if (!structure_ok) {
      result.reason_code = PatternsExecutorReason::STRUCTURE_VALIDATION_FAILED;
      return result;
    }
    if (op_plan.operation == PatternOperation::Read &&
        (!result.checksum_evaluated || !result.checksum_valid)) {
      result.reason_code = PatternsExecutorReason::CHECKSUM_MISMATCH;
      return result;
    }
    if (op_plan.operation == PatternOperation::Write &&
        (!result.validation_checks[2].evaluated.value_or(false) ||
         !result.validation_checks[2].valid.value_or(false))) {
      result.reason_code = PatternsExecutorReason::WRITE_POST_VALIDATION_FAILED;
      return result;
    }
    if (op_plan.operation == PatternOperation::Copy &&
        (!result.validation_checks[3].evaluated.value_or(false) ||
         !result.validation_checks[3].valid.value_or(false))) {
      result.reason_code = PatternsExecutorReason::COPY_POST_VALIDATION_FAILED;
      return result;
    }
    if ((op_plan.operation == PatternOperation::Write ||
         op_plan.operation == PatternOperation::Copy) &&
        op_plan.has_untouched_dst_bytes &&
        (!result.validation_checks[4].evaluated.value_or(false) ||
         !result.validation_checks[4].valid.value_or(false))) {
      result.reason_code = PatternsExecutorReason::UNTOUCHED_SENTINEL_FAILED;
      return result;
    }
    if ((op_plan.operation == PatternOperation::Read ||
         op_plan.operation == PatternOperation::Copy) &&
        !result.validation_checks[5].valid.value_or(false)) {
      result.reason_code = PatternsExecutorReason::SRC_READONLY_FAILED;
      return result;
    }

    result.valid = true;
    result.reason_code = PatternsExecutorReason::VALID;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = PatternsExecutorReason::KERNEL_FAILED;
    return result;
  }
}
