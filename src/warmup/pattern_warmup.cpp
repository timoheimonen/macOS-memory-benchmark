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
/**
 * @file pattern_warmup.cpp
 * @brief Pattern benchmark warmup functions
 */

#include "warmup/warmup.h"

#include <atomic>
#include <iostream>
#include <system_error>
#include <thread>
#include <vector>

// macOS specific QoS
#include <mach/mach.h>
#include <pthread/qos.h>

#include "asm/asm_functions.h"
#include "output/console/messages/messages_api.h"
#include "pattern_benchmark/pattern_work_plan.h"
#include "warmup/warmup_internal.h"

namespace {

/**
 * @brief Execute one warmup operation per finalized worker partition.
 *
 * Multi-worker warmups retain the worker QoS behavior used by the previous
 * random warmup implementation. A single worker runs on the calling thread,
 * matching the existing one-worker behavior.
 */
template <typename Worker, typename WorkerOperation>
void run_pattern_warmup_workers(const std::vector<Worker>& workers,
                                WorkerOperation operation) {
  if (workers.empty()) {
    return;
  }
  if (workers.size() == 1) {
    operation(workers.front());
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(workers.size());
  WarmupThreadJoinGuard thread_join_guard(threads);
  for (const Worker& worker : workers) {
    const Worker* worker_ptr = &worker;
    try {
      threads.emplace_back([worker_ptr, operation]() {
        kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qos_ret != KERN_SUCCESS) {
          std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
        }
        operation(*worker_ptr);
      });
    } catch (const std::system_error&) {
      return;
    }
  }
}

}  // namespace

void warmup_read_strided(void* buffer, const PatternWorkPlan& plan,
                         std::atomic<uint64_t>& dummy_checksum) {
  if (buffer == nullptr || plan.phase_period_passes == 0) {
    return;
  }

  char* buffer_start = static_cast<char*>(buffer);
  run_pattern_warmup_workers(
      plan.workers, [buffer_start, &plan, &dummy_checksum](const PatternWorkerRange& worker) {
        const uint64_t result = memory_read_strided_phased_loop_asm(
            buffer_start + worker.offset_bytes, worker.span_bytes,
            plan.stride_bytes, plan.phase_period_passes, 0);
        dummy_checksum.fetch_xor(result, std::memory_order_release);
      });
}

void warmup_write_strided(void* buffer, const PatternWorkPlan& plan) {
  if (buffer == nullptr || plan.phase_period_passes == 0) {
    return;
  }

  char* buffer_start = static_cast<char*>(buffer);
  run_pattern_warmup_workers(
      plan.workers, [buffer_start, &plan](const PatternWorkerRange& worker) {
        memory_write_strided_phased_loop_asm(
            buffer_start + worker.offset_bytes, worker.span_bytes,
            plan.stride_bytes, plan.phase_period_passes, 0);
      });
}

void warmup_copy_strided(void* dst, void* src, const PatternWorkPlan& plan) {
  if (dst == nullptr || src == nullptr || plan.phase_period_passes == 0) {
    return;
  }

  char* dst_start = static_cast<char*>(dst);
  const char* src_start = static_cast<const char*>(src);
  run_pattern_warmup_workers(
      plan.workers, [dst_start, src_start, &plan](const PatternWorkerRange& worker) {
        memory_copy_strided_phased_loop_asm(
            dst_start + worker.offset_bytes, src_start + worker.offset_bytes,
            worker.span_bytes, plan.stride_bytes, plan.phase_period_passes, 0);
      });
}

void warmup_read_random(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    std::atomic<uint64_t>& dummy_checksum) {
  if (buffer == nullptr) {
    return;
  }

  char* buffer_start = static_cast<char*>(buffer);
  run_pattern_warmup_workers(
      worker_indices,
      [buffer_start, &dummy_checksum](const PatternRandomWorkerIndices& worker) {
        if (worker.indices.empty()) {
          return;
        }
        const uint64_t result = memory_read_random_loop_asm(
            buffer_start + worker.offset_bytes, worker.indices.data(),
            worker.indices.size());
        dummy_checksum.fetch_xor(result, std::memory_order_release);
      });
}

void warmup_write_random(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices) {
  if (buffer == nullptr) {
    return;
  }

  char* buffer_start = static_cast<char*>(buffer);
  run_pattern_warmup_workers(
      worker_indices, [buffer_start](const PatternRandomWorkerIndices& worker) {
        if (worker.indices.empty()) {
          return;
        }
        memory_write_random_loop_asm(buffer_start + worker.offset_bytes,
                                     worker.indices.data(),
                                     worker.indices.size());
      });
}

void warmup_copy_random(
    void* dst, void* src,
    const std::vector<PatternRandomWorkerIndices>& worker_indices) {
  if (dst == nullptr || src == nullptr) {
    return;
  }

  char* dst_start = static_cast<char*>(dst);
  const char* src_start = static_cast<const char*>(src);
  run_pattern_warmup_workers(
      worker_indices,
      [dst_start, src_start](const PatternRandomWorkerIndices& worker) {
        if (worker.indices.empty()) {
          return;
        }
        memory_copy_random_loop_asm(
            dst_start + worker.offset_bytes, src_start + worker.offset_bytes,
            worker.indices.data(), worker.indices.size());
      });
}
