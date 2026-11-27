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
 * @file cpu_bandwidth_executor.cpp
 * @brief Persistent buffer preparation, synchronized ARM64 worker gate, and cold checks
 */

#include "cpu_bandwidth/cpu_bandwidth_executor.h"

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

int apply_worker_qos(const CpuBandwidthExecutorTestControl* test_control,
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
    const CpuBandwidthExpectedWorkerChecksum& lhs,
    const CpuBandwidthExpectedWorkerChecksum& rhs) noexcept {
  return lhs.worker_index == rhs.worker_index &&
         lhs.state_a == rhs.state_a &&
         lhs.state_b == rhs.state_b &&
         lhs.exact_bytes_read == rhs.exact_bytes_read &&
         lhs.pass_count == rhs.pass_count;
}

}  // namespace

std::array<CpuBandwidthValidationCheck, 5> make_default_cpu_bandwidth_checks(
    CpuBandwidthOperation op) {
  std::array<CpuBandwidthValidationCheck, 5> checks;
  checks[0] = {"post-validation-structure", true, false, std::nullopt,
               "not-evaluated"};
  const bool is_read = (op == CpuBandwidthOperation::Read);
  const bool is_write = (op == CpuBandwidthOperation::Write);
  const bool is_copy = (op == CpuBandwidthOperation::Copy);
  const bool uses_src = (is_read || is_copy);

  checks[1] = {"read-checksum", is_read,
               is_read ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_read ? "not-evaluated" : "not-applicable"};
  checks[2] = {"write-final-full-buffer", is_write,
               is_write ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_write ? "not-evaluated" : "not-applicable"};
  checks[3] = {"copy-final-full-buffer", is_copy,
               is_copy ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, is_copy ? "not-evaluated" : "not-applicable"};
  checks[4] = {"src-readonly-protection", uses_src,
               uses_src ? std::optional<bool>(false) : std::nullopt,
               std::nullopt, uses_src ? "not-evaluated" : "not-applicable"};
  return checks;
}

bool prepare_cpu_bandwidth_resources(
    const CpuBandwidthWorkPlan& plan,
    bool non_cacheable,
    CpuBandwidthResourceSet& resources,
    const std::function<bool()>& stop_requested) {
  release_cpu_bandwidth_resources(resources);
  resources = CpuBandwidthResourceSet{};

  if (!plan.valid || !plan.memory_budget.valid) {
    resources.reason_code = CpuBandwidthExecutorReason::INVALID_RESOURCES;
    return false;
  }

  for (CpuBandwidthTarget target : plan.active_targets) {
    if (stop_requested && stop_requested()) {
      release_cpu_bandwidth_resources(resources);
      resources.reason_code = CpuBandwidthExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    const size_t idx = cpu_bandwidth_target_index(target);
    const CpuBandwidthTargetPlan& tplan = plan.targets[idx];
    CpuBandwidthTargetBuffers& tbuf = resources.targets[idx];
    tbuf.target = target;
    tbuf.active = true;

    tbuf.src = allocate_buffer_with_evidence(tplan.bytes, non_cacheable,
                                             "cpu_bandwidth_src");
    tbuf.dst = allocate_buffer_with_evidence(tplan.bytes, non_cacheable,
                                             "cpu_bandwidth_dst");
    if (tbuf.src.ptr == nullptr || tbuf.dst.ptr == nullptr) {
      release_cpu_bandwidth_resources(resources);
      resources.reason_code =
          CpuBandwidthExecutorReason::BUFFER_ALLOCATION_FAILED;
      return false;
    }

    if (stop_requested && stop_requested()) {
      release_cpu_bandwidth_resources(resources);
      resources.reason_code = CpuBandwidthExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    initialize_cpu_bandwidth_src_buffer(tbuf.src.ptr.get(), tplan.bytes,
                                        tplan.src_seed, tplan.worker_spans);
    const size_t protect_size =
        tbuf.src.mapped_bytes > 0 ? tbuf.src.mapped_bytes : tplan.bytes;
    if (!protect_buffer_read_only_with_evidence(tbuf.src, protect_size)) {
      // Keep resource evidence intact on target if needed, or mark failure.
      tbuf.valid = false;
      resources.valid = false;
      resources.reason_code = CpuBandwidthExecutorReason::SRC_READONLY_FAILED;
      return false;
    }

    const bool cache_target = (target != CpuBandwidthTarget::MainMemory);
    const uint64_t initial_sentinel =
        derive_cpu_bandwidth_pre_copy_sentinel_pattern(
            plan.pre_copy_sentinel_seed, 0ULL);
    initialize_cpu_bandwidth_dst_sentinel(tbuf.dst.ptr.get(), tplan.bytes,
                                          initial_sentinel, cache_target,
                                          tplan.worker_spans);
    tbuf.valid = true;
  }

  resources.valid = true;
  resources.reason_code = CpuBandwidthExecutorReason::VALID;
  return true;
}

void release_cpu_bandwidth_resources(
    CpuBandwidthResourceSet& resources) noexcept {
  for (auto& tbuf : resources.targets) {
    tbuf.src.ptr.reset();
    tbuf.dst.ptr.reset();
    tbuf.valid = false;
    tbuf.active = false;
  }
  resources.valid = false;
}

CpuBandwidthOperationExecution execute_cpu_bandwidth_operation(
    const CpuBandwidthConfig& config,
    const CpuBandwidthTargetPlan& target_plan,
    const CpuBandwidthOperationPlan& op_plan,
    const CpuBandwidthTargetBuffers& buffers,
    HighResTimer& timer,
    std::optional<uint64_t> write_attempt_seed,
    std::optional<uint64_t> write_pattern,
    uint64_t pre_copy_sentinel_pattern,
    const CpuBandwidthExecutorTestControl* test_control) noexcept {
  CpuBandwidthOperationExecution result;
  result.requested_workers = config.requested_workers;
  result.validation_checks = make_default_cpu_bandwidth_checks(op_plan.operation);
  result.checksum_applicable =
      (op_plan.operation == CpuBandwidthOperation::Read);

  try {
    if (!target_plan.enabled || !op_plan.valid || !buffers.valid ||
        buffers.src.ptr == nullptr || buffers.dst.ptr == nullptr ||
        target_plan.effective_workers == 0 ||
        target_plan.worker_spans.size() != target_plan.effective_workers) {
      result.reason_code = CpuBandwidthExecutorReason::INVALID_RESOURCES;
      return result;
    }
    if (op_plan.operation == CpuBandwidthOperation::Write &&
        (!write_pattern.has_value() || *write_pattern == 0 ||
         (*write_pattern & 1ULL) == 0)) {
      result.reason_code = CpuBandwidthExecutorReason::INVALID_OPERATION_PLAN;
      return result;
    }

    result.available = true;
    result.write_pattern =
        op_plan.operation == CpuBandwidthOperation::Write ? write_pattern
                                                          : std::nullopt;
    result.attempt_seed =
        op_plan.operation == CpuBandwidthOperation::Write ? write_attempt_seed
                                                          : std::nullopt;

    const bool cache_target =
        (target_plan.target != CpuBandwidthTarget::MainMemory);
    if (op_plan.operation == CpuBandwidthOperation::Copy) {
      initialize_cpu_bandwidth_dst_sentinel(
          buffers.dst.ptr.get(), target_plan.bytes, pre_copy_sentinel_pattern,
          cache_target, target_plan.worker_spans);
    }

    const size_t worker_count = target_plan.effective_workers;
    if (op_plan.operation == CpuBandwidthOperation::Read) {
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

    uint8_t* const dst_base = static_cast<uint8_t*>(buffers.dst.ptr.get());
    const uint8_t* const src_base =
        static_cast<const uint8_t*>(buffers.src.ptr.get());
    const uint64_t passes = static_cast<uint64_t>(op_plan.passes);

    try {
      for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        if (test_control != nullptr &&
            test_control->fail_before_worker_index >= 0 &&
            worker_index ==
                static_cast<size_t>(test_control->fail_before_worker_index)) {
          throw std::system_error(
              std::make_error_code(std::errc::resource_unavailable_try_again));
        }

        const CpuBandwidthWorkerSpan span = target_plan.worker_spans[worker_index];
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
            state_cv.wait(lock, [&]() { return start_workers || cancel_workers; });
            if (cancel_workers) {
              return;
            }
          }

          bool ok = true;
          if (test_control != nullptr && test_control->force_kernel_fail) {
            ok = false;
          } else if (op_plan.operation == CpuBandwidthOperation::Read) {
            uint64_t state_a = 0;
            uint64_t state_b = 0;
            initial_cpu_bandwidth_worker_checksum_state(
                op_plan.op_seed, worker_index, state_a, state_b);
            const uint8_t* const chunk_src = src_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            for (uint64_t p = 1; p <= passes; ++p) {
              const uint64_t pass_xor =
                  cache_target
                      ? memory_read_cache_loop_asm(chunk_src, chunk_bytes)
                      : memory_read_loop_asm(chunk_src, chunk_bytes);
              state_a += pass_xor + kCpuBandwidthChecksumA * p;
              state_b += (pass_xor ^ kCpuBandwidthChecksumB) * p +
                         kCpuBandwidthChecksumC;
            }
            CpuBandwidthExpectedWorkerChecksum& wout =
                result.actual_worker_checksums[worker_index];
            wout.worker_index = worker_index;
            wout.state_a = state_a;
            wout.state_b = state_b;
            wout.exact_bytes_read = passes * static_cast<uint64_t>(chunk_bytes);
            wout.pass_count = passes;
          } else if (op_plan.operation == CpuBandwidthOperation::Write) {
            uint8_t* const chunk_dst = dst_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            const uint64_t pat = *write_pattern;
            for (uint64_t p = 1; p <= passes; ++p) {
              if (cache_target) {
                memory_write_cache_pattern_loop_asm(chunk_dst, chunk_bytes, pat);
              } else {
                memory_write_pattern_loop_asm(chunk_dst, chunk_bytes, pat);
              }
            }
          } else {
            uint8_t* const chunk_dst = dst_base + span.offset_bytes;
            const uint8_t* const chunk_src = src_base + span.offset_bytes;
            const size_t chunk_bytes = span.span_bytes;
            for (uint64_t p = 1; p <= passes; ++p) {
              if (cache_target) {
                memory_copy_cache_loop_asm(chunk_dst, chunk_src, chunk_bytes);
              } else {
                memory_copy_loop_asm(chunk_dst, chunk_src, chunk_bytes);
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
      result.reason_code = CpuBandwidthExecutorReason::WORKER_STARTUP_FAILED;
      result.validation_checks[0].evaluated = true;
      result.validation_checks[0].valid = false;
      result.validation_checks[0].reason_code =
          CpuBandwidthExecutorReason::WORKER_STARTUP_FAILED;
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
      test_control->corrupt_dst_after_kernel(buffers.dst.ptr.get(),
                                             target_plan.bytes);
    }

    // Cold post-validation (outside timer).
    const bool structure_ok =
        result.created_workers == worker_count &&
        result.completed_workers == worker_count &&
        !result.worker_startup_failed &&
        result.kernel_succeeded &&
        buffers.src.mapped_bytes >= target_plan.bytes &&
        buffers.dst.mapped_bytes >= target_plan.bytes;

    result.validation_checks[0].evaluated = true;
    result.validation_checks[0].valid = structure_ok;
    result.validation_checks[0].reason_code =
        structure_ok ? CpuBandwidthExecutorReason::VALID
                     : CpuBandwidthExecutorReason::STRUCTURE_VALIDATION_FAILED;

    if (op_plan.operation == CpuBandwidthOperation::Read) {
      if (result.kernel_succeeded) {
        result.actual_run_checksum = fold_cpu_bandwidth_worker_checksums(
            result.actual_worker_checksums);
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
            matches ? CpuBandwidthExecutorReason::VALID
                    : CpuBandwidthExecutorReason::CHECKSUM_MISMATCH;
      }
    } else if (op_plan.operation == CpuBandwidthOperation::Write) {
      if (result.kernel_succeeded) {
        const bool write_ok = validate_cpu_bandwidth_write_buffer(
            buffers.dst.ptr.get(), target_plan.bytes, *write_pattern,
            target_plan.worker_spans);
        result.validation_checks[2].evaluated = true;
        result.validation_checks[2].valid = write_ok;
        result.validation_checks[2].reason_code =
            write_ok ? CpuBandwidthExecutorReason::VALID
                     : CpuBandwidthExecutorReason::WRITE_POST_VALIDATION_FAILED;
      }
    } else {
      if (result.kernel_succeeded) {
        const bool copy_ok = validate_cpu_bandwidth_copy_buffer(
            buffers.dst.ptr.get(), buffers.src.ptr.get(), target_plan.bytes,
            target_plan.worker_spans);
        result.validation_checks[3].evaluated = true;
        result.validation_checks[3].valid = copy_ok;
        result.validation_checks[3].reason_code =
            copy_ok ? CpuBandwidthExecutorReason::VALID
                    : CpuBandwidthExecutorReason::COPY_POST_VALIDATION_FAILED;
      }
    }

    if (op_plan.operation == CpuBandwidthOperation::Read ||
        op_plan.operation == CpuBandwidthOperation::Copy) {
      const bool ro_ok =
          buffers.src.read_only_protected && buffers.src.mprotect_errno == 0;
      result.validation_checks[4].evaluated = true;
      result.validation_checks[4].valid = ro_ok;
      result.validation_checks[4].reason_code =
          ro_ok ? CpuBandwidthExecutorReason::VALID
                : CpuBandwidthExecutorReason::SRC_READONLY_FAILED;
    }

    if (!result.kernel_succeeded) {
      result.reason_code = CpuBandwidthExecutorReason::KERNEL_FAILED;
      return result;
    }
    if (!result.timing_valid) {
      result.reason_code = CpuBandwidthExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }
    if (!structure_ok) {
      result.reason_code =
          CpuBandwidthExecutorReason::STRUCTURE_VALIDATION_FAILED;
      return result;
    }
    if (op_plan.operation == CpuBandwidthOperation::Read &&
        (!result.checksum_evaluated || !result.checksum_valid)) {
      result.reason_code = CpuBandwidthExecutorReason::CHECKSUM_MISMATCH;
      return result;
    }
    if (op_plan.operation == CpuBandwidthOperation::Write &&
        (!result.validation_checks[2].evaluated.value_or(false) ||
         !result.validation_checks[2].valid.value_or(false))) {
      result.reason_code =
          CpuBandwidthExecutorReason::WRITE_POST_VALIDATION_FAILED;
      return result;
    }
    if (op_plan.operation == CpuBandwidthOperation::Copy &&
        (!result.validation_checks[3].evaluated.value_or(false) ||
         !result.validation_checks[3].valid.value_or(false))) {
      result.reason_code =
          CpuBandwidthExecutorReason::COPY_POST_VALIDATION_FAILED;
      return result;
    }
    if ((op_plan.operation == CpuBandwidthOperation::Read ||
         op_plan.operation == CpuBandwidthOperation::Copy) &&
        !result.validation_checks[4].valid.value_or(false)) {
      result.reason_code = CpuBandwidthExecutorReason::SRC_READONLY_FAILED;
      return result;
    }

    result.valid = true;
    result.reason_code = CpuBandwidthExecutorReason::VALID;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = CpuBandwidthExecutorReason::KERNEL_FAILED;
    return result;
  }
}
