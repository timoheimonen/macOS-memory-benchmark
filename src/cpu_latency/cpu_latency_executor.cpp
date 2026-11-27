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
 * @file cpu_latency_executor.cpp
 * @brief Persistent chain buffers, ARM64 chase execution, sample windows, and checks
 */

#include "cpu_latency/cpu_latency_executor.h"

#include <algorithm>
#include <cmath>
#include <sys/mman.h>

#include "asm/asm_functions.h"
#include "utils/numeric_utils.h"

namespace {

CpuLatencyChainIdentity make_chain_identity(
    const CpuLatencyChainPermutation& perm) {
  CpuLatencyChainIdentity id;
  id.active = perm.valid;
  id.chain_seed = perm.chain_seed;
  id.node_count = perm.node_count;
  id.stride_bytes = perm.stride_bytes;
  id.chain_mode = perm.effective_mode;
  id.tlb_locality_kb = perm.tlb_locality_kb;
  id.permutation_sha256 = perm.permutation_sha256;
  return id;
}

bool prepare_single_chain_identity(
    const CpuLatencyTargetPlan& tplan,
    uint64_t base_seed,
    uint64_t domain_id,
    CpuLatencyChainIdentity& out_identity,
    std::vector<uint64_t>& out_indices,
    std::string& out_reason) {
  CpuLatencyChainPermutation perm = generate_cpu_latency_chain_permutation(
      base_seed, domain_id, tplan.bytes, tplan.stride_bytes,
      tplan.effective_mode, tplan.tlb_locality_kb);
  if (!perm.valid) {
    out_reason = perm.reason_code;
    return false;
  }
  out_identity = make_chain_identity(perm);
  out_indices = std::move(perm.indices);
  return true;
}

bool apply_rw_protection(AllocatedBufferEvidence& buf,
                         size_t protect_size,
                         const CpuLatencyExecutorTestControl* test_control) {
  if (!buf.read_only_protected) {
    return true;
  }
  bool ok = false;
  if (test_control != nullptr && test_control->mprotect_override) {
    ok = test_control->mprotect_override(buf.ptr.get(), protect_size,
                                         PROT_READ | PROT_WRITE);
  } else {
    ok = (::mprotect(buf.ptr.get(), protect_size, PROT_READ | PROT_WRITE) == 0);
  }
  if (ok) {
    buf.read_only_protected = false;
    buf.mprotect_errno = 0;
  }
  return ok;
}

}  // namespace

std::array<CpuLatencyValidationCheck, 5> make_default_cpu_latency_checks(
    bool sample_windows_applicable) {
  std::array<CpuLatencyValidationCheck, 5> checks;
  checks[0] = {"post-validation-structure", true, false, std::nullopt,
               "not-evaluated"};
  checks[1] = {"chain-hamiltonian-prevalidation", true, false, std::nullopt,
               "not-evaluated"};
  checks[2] = {"chain-readonly-protection", true, false, std::nullopt,
               "not-evaluated"};
  checks[3] = {"main-final-pointer", true, false, std::nullopt,
               "not-evaluated"};
  checks[4] = {"sample-window-end-pointers", sample_windows_applicable,
               sample_windows_applicable ? std::optional<bool>(false)
                                         : std::nullopt,
               std::nullopt,
               sample_windows_applicable ? "not-evaluated" : "not-applicable"};
  return checks;
}

bool prepare_cpu_latency_resources(
    const CpuLatencyWorkPlan& plan,
    bool non_cacheable,
    CpuLatencyResourceSet& resources,
    const std::function<bool()>& stop_requested) {
  release_cpu_latency_resources(resources);
  resources = CpuLatencyResourceSet{};

  if (!plan.valid || !plan.memory_budget.valid) {
    resources.reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
    return false;
  }

  for (CpuLatencyTarget target : plan.active_targets) {
    if (stop_requested && stop_requested()) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    const size_t idx = cpu_latency_target_index(target);
    const CpuLatencyTargetPlan& tplan = plan.targets[idx];
    CpuLatencyTargetBuffer& tbuf = resources.targets[idx];
    tbuf.target = target;
    tbuf.active = true;

    tbuf.buffer = allocate_buffer_with_evidence(tplan.bytes, non_cacheable,
                                                "cpu_latency_chain");
    if (tbuf.buffer.ptr == nullptr) {
      release_cpu_latency_resources(resources);
      resources.reason_code =
          CpuLatencyExecutorReason::BUFFER_ALLOCATION_FAILED;
      return false;
    }

    if (stop_requested && stop_requested()) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    CpuLatencyChainPermutation perm = generate_cpu_latency_chain_permutation(
        plan.base_seed, cpu_latency_target_domain_id(target), tplan.bytes,
        tplan.stride_bytes, tplan.effective_mode, tplan.tlb_locality_kb);
    if (!perm.valid) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::CHAIN_GENERATION_FAILED;
      return false;
    }

    std::string pop_reason;
    if (!populate_cpu_latency_chain_buffer(tbuf.buffer.ptr.get(), tplan.bytes,
                                           perm, pop_reason)) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::CHAIN_GENERATION_FAILED;
      return false;
    }

    tbuf.chain_identity = make_chain_identity(perm);
    tbuf.indices = std::move(perm.indices);

    tbuf.hamiltonian_prevalidated = validate_cpu_latency_hamiltonian_cycle(
        tbuf.buffer.ptr.get(), tplan.bytes, tplan.stride_bytes,
        tplan.node_count);
    if (!tbuf.hamiltonian_prevalidated) {
      tbuf.valid = false;
      resources.valid = false;
      resources.reason_code =
          CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED;
      return false;
    }

    const size_t protect_size =
        tbuf.buffer.mapped_bytes > 0 ? tbuf.buffer.mapped_bytes : tplan.bytes;
    if (!protect_buffer_read_only_with_evidence(tbuf.buffer, protect_size)) {
      tbuf.valid = false;
      resources.valid = false;
      resources.reason_code = CpuLatencyExecutorReason::CHAIN_READONLY_FAILED;
      return false;
    }

    tbuf.valid = true;
  }

  if (plan.compare_locality) {
    if (!plan.locality_16k_plan.has_value() ||
        !plan.locality_global_plan.has_value()) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
      return false;
    }
    if (stop_requested && stop_requested()) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    resources.compare_locality_active = true;
    resources.compare_buffer = allocate_buffer_with_evidence(
        plan.locality_16k_plan->bytes, non_cacheable,
        "cpu_latency_compare_locality");
    if (resources.compare_buffer.ptr == nullptr) {
      release_cpu_latency_resources(resources);
      resources.reason_code =
          CpuLatencyExecutorReason::BUFFER_ALLOCATION_FAILED;
      return false;
    }

    if (stop_requested && stop_requested()) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::PREPARATION_INTERRUPTED;
      return false;
    }

    std::string reason;
    if (!prepare_single_chain_identity(
            *plan.locality_16k_plan, plan.base_seed, kCpuLatencyDomainLocality16k,
            resources.locality_16k_identity, resources.locality_16k_indices,
            reason) ||
        !prepare_single_chain_identity(
            *plan.locality_global_plan, plan.base_seed,
            kCpuLatencyDomainLocalityGlobal, resources.locality_global_identity,
            resources.locality_global_indices, reason)) {
      release_cpu_latency_resources(resources);
      resources.reason_code = CpuLatencyExecutorReason::CHAIN_GENERATION_FAILED;
      return false;
    }
  }

  resources.valid = true;
  resources.reason_code = CpuLatencyExecutorReason::VALID;
  return true;
}

void release_cpu_latency_resources(CpuLatencyResourceSet& resources) noexcept {
  for (auto& tbuf : resources.targets) {
    tbuf.buffer.ptr.reset();
    tbuf.indices.clear();
    tbuf.valid = false;
    tbuf.active = false;
    tbuf.hamiltonian_prevalidated = false;
  }
  resources.compare_buffer.ptr.reset();
  resources.locality_16k_indices.clear();
  resources.locality_global_indices.clear();
  resources.compare_locality_active = false;
  resources.valid = false;
}

CpuLatencyAttemptExecution execute_cpu_latency_attempt(
    const CpuLatencyTargetPlan& target_plan,
    const CpuLatencyTargetExecutionPlan& exec_plan,
    const CpuLatencyTargetBuffer& buffer,
    HighResTimer& timer,
    size_t requested_samples,
    const CpuLatencyExecutorTestControl* test_control) noexcept {
  CpuLatencyAttemptExecution result;
  const bool samples_applicable = requested_samples > 0;
  result.validation_checks = make_default_cpu_latency_checks(samples_applicable);

  try {
    const bool structure_ok =
        target_plan.enabled && exec_plan.valid && buffer.valid &&
        buffer.buffer.ptr != nullptr &&
        buffer.buffer.mapped_bytes >= target_plan.bytes &&
        target_plan.node_count >= 2 &&
        buffer.indices.size() == target_plan.node_count &&
        exec_plan.access_count > 0;

    result.validation_checks[0].evaluated = true;
    result.validation_checks[0].valid = structure_ok;
    result.validation_checks[0].reason_code =
        structure_ok ? CpuLatencyExecutorReason::VALID
                     : CpuLatencyExecutorReason::STRUCTURE_VALIDATION_FAILED;

    if (!structure_ok) {
      result.reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
      return result;
    }

    result.available = true;

    const bool hamiltonian_ok =
        buffer.hamiltonian_prevalidated &&
        !(test_control != nullptr && test_control->force_hamiltonian_fail);
    result.validation_checks[1].evaluated = true;
    result.validation_checks[1].valid = hamiltonian_ok;
    result.validation_checks[1].reason_code =
        hamiltonian_ok ? CpuLatencyExecutorReason::VALID
                       : CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED;

    const bool ro_ok = buffer.buffer.read_only_protected &&
                       buffer.buffer.mprotect_errno == 0;
    result.validation_checks[2].evaluated = true;
    result.validation_checks[2].valid = ro_ok;
    result.validation_checks[2].reason_code =
        ro_ok ? CpuLatencyExecutorReason::VALID
              : CpuLatencyExecutorReason::CHAIN_READONLY_FAILED;

    if (!hamiltonian_ok) {
      result.reason_code = CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED;
      return result;
    }
    if (!ro_ok) {
      result.reason_code = CpuLatencyExecutorReason::CHAIN_READONLY_FAILED;
      return result;
    }

    uintptr_t* const base_ptr =
        static_cast<uintptr_t*>(buffer.buffer.ptr.get());
    uintptr_t* actual_final_ptr = nullptr;
    double elapsed_seconds = 0.0;
    bool kernel_ok = true;

    timer.start();
    result.timer_started = true;
    if (test_control != nullptr && test_control->force_kernel_fail) {
      kernel_ok = false;
    } else if (test_control != nullptr && test_control->chase_override) {
      actual_final_ptr =
          test_control->chase_override(base_ptr, exec_plan.access_count);
    } else {
      actual_final_ptr =
          memory_latency_chase_asm(base_ptr, exec_plan.access_count);
    }
    elapsed_seconds = timer.stop();
    result.timer_stopped = true;
    result.kernel_succeeded = kernel_ok;

    if (test_control != nullptr && test_control->force_zero_elapsed) {
      elapsed_seconds = 0.0;
      result.cpu_raw = std::nullopt;
    } else {
      result.cpu_raw = timer.last_snapshot;
    }
    result.elapsed_seconds = elapsed_seconds;
    result.timing_evaluated = result.timer_started && result.timer_stopped;
    result.timing_valid = result.timing_evaluated &&
                          std::isfinite(result.elapsed_seconds) &&
                          result.elapsed_seconds > 0.0 &&
                          result.cpu_raw.has_value() &&
                          result.cpu_raw->numer > 0 &&
                          result.cpu_raw->denom > 0;

    if (!result.kernel_succeeded) {
      result.reason_code = CpuLatencyExecutorReason::KERNEL_FAILED;
      return result;
    }

    result.final_pointer.expected_node = 0;
    result.final_pointer.actual_node = pointer_to_chain_node_index(
        base_ptr, target_plan.bytes, target_plan.stride_bytes,
        target_plan.node_count, actual_final_ptr);
    const bool main_ptr_ok =
        (actual_final_ptr == base_ptr) &&
        result.final_pointer.actual_node.has_value() &&
        *result.final_pointer.actual_node == 0;
    result.final_pointer.valid = main_ptr_ok;

    result.validation_checks[3].evaluated = true;
    result.validation_checks[3].valid = main_ptr_ok;
    result.validation_checks[3].reason_code =
        main_ptr_ok ? CpuLatencyExecutorReason::VALID
                    : CpuLatencyExecutorReason::MAIN_FINAL_POINTER_FAILED;

    if (!result.timing_valid) {
      result.reason_code = CpuLatencyExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }
    if (!main_ptr_ok) {
      result.reason_code = CpuLatencyExecutorReason::MAIN_FINAL_POINTER_FAILED;
      return result;
    }

    if (samples_applicable) {
      const size_t effective_samples =
          std::min(requested_samples, exec_plan.access_count);
      const size_t base_step = exec_plan.access_count / effective_samples;
      const size_t rem = exec_plan.access_count % effective_samples;

      CpuLatencySampleWindows sw;
      sw.timebase_numer = result.cpu_raw->numer;
      sw.timebase_denom = result.cpu_raw->denom;
      sw.sample_count = effective_samples;
      sw.sample_delta_ticks.reserve(effective_samples);
      sw.samples_ns.reserve(effective_samples);
      sw.end_node_indices.reserve(effective_samples);

      uintptr_t* current_ptr = base_ptr;
      size_t cum_steps = 0;
      bool all_end_ptrs_ok = true;
      bool all_sample_timings_ok = true;

      for (size_t idx = 0; idx < effective_samples; ++idx) {
        const size_t accesses_i = base_step + (idx < rem ? 1 : 0);
        cum_steps += accesses_i;

        timer.start();
        if (test_control != nullptr && test_control->chase_override) {
          current_ptr = test_control->chase_override(current_ptr, accesses_i);
        } else {
          current_ptr = memory_latency_chase_asm(current_ptr, accesses_i);
        }
        (void)timer.stop();

        const auto& snap = timer.last_snapshot;
        if (snap.has_value() && snap->denom > 0 && snap->numer > 0 &&
            snap->denom == sw.timebase_denom &&
            snap->numer == sw.timebase_numer) {
          sw.sample_delta_ticks.push_back(snap->delta_ticks);
          const double win_ns =
              (static_cast<double>(snap->delta_ticks) *
               static_cast<double>(snap->numer) /
               static_cast<double>(snap->denom)) /
              static_cast<double>(accesses_i);
          if (!std::isfinite(win_ns)) {
            all_sample_timings_ok = false;
          }
          sw.samples_ns.push_back(win_ns);
        } else {
          all_sample_timings_ok = false;
          sw.sample_delta_ticks.push_back(0);
          sw.samples_ns.push_back(0.0);
        }

        const size_t expected_node = static_cast<size_t>(
            buffer.indices[cum_steps % target_plan.node_count]);
        const std::optional<size_t> actual_node = pointer_to_chain_node_index(
            base_ptr, target_plan.bytes, target_plan.stride_bytes,
            target_plan.node_count, current_ptr);
        sw.end_node_indices.push_back(actual_node);
        if (!actual_node.has_value() || *actual_node != expected_node) {
          all_end_ptrs_ok = false;
        }
      }

      sw.all_end_pointers_valid = all_end_ptrs_ok;
      sw.validation_valid = all_end_ptrs_ok && all_sample_timings_ok;
      result.sample_windows = std::move(sw);

      result.validation_checks[4].evaluated = true;
      result.validation_checks[4].valid =
          result.sample_windows->validation_valid;
      result.validation_checks[4].reason_code =
          result.sample_windows->validation_valid
              ? CpuLatencyExecutorReason::VALID
              : CpuLatencyExecutorReason::SAMPLE_WINDOW_END_POINTER_FAILED;

      if (!result.sample_windows->validation_valid) {
        result.reason_code =
            CpuLatencyExecutorReason::SAMPLE_WINDOW_END_POINTER_FAILED;
        return result;
      }
    }

    result.valid = true;
    result.reason_code = CpuLatencyExecutorReason::VALID;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = CpuLatencyExecutorReason::KERNEL_FAILED;
    return result;
  }
}

CpuLatencyLocalitySubrunResult execute_cpu_latency_locality_subrun(
    const CpuLatencyTargetPlan& sub_plan,
    const CpuLatencyChainIdentity& identity,
    const std::vector<uint64_t>& indices,
    size_t complete_chain_cycles,
    CpuLatencyResourceSet& resources,
    HighResTimer& timer,
    const CpuLatencyExecutorTestControl* test_control) noexcept {
  CpuLatencyLocalitySubrunResult sub;
  try {
    if (!resources.compare_locality_active ||
        resources.compare_buffer.ptr == nullptr || !sub_plan.enabled ||
        !identity.active || indices.size() != sub_plan.node_count ||
        complete_chain_cycles == 0) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
      return sub;
    }

    size_t total_accesses = 0;
    if (!NumericUtils::checked_multiply(complete_chain_cycles,
                                        sub_plan.node_count, total_accesses) ||
        total_accesses == 0 || total_accesses > kCpuLatencyMaxAccesses) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::INVALID_EXECUTION_PLAN;
      return sub;
    }

    const size_t protect_size = resources.compare_buffer.mapped_bytes > 0
                                    ? resources.compare_buffer.mapped_bytes
                                    : sub_plan.bytes;
    if (!apply_rw_protection(resources.compare_buffer, protect_size,
                             test_control)) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::CHAIN_READONLY_FAILED;
      return sub;
    }

    CpuLatencyChainPermutation perm;
    perm.valid = true;
    perm.chain_seed = identity.chain_seed;
    perm.node_count = identity.node_count;
    perm.stride_bytes = identity.stride_bytes;
    perm.effective_mode = identity.chain_mode;
    perm.tlb_locality_kb = identity.tlb_locality_kb;
    perm.indices = indices;
    perm.permutation_sha256 = identity.permutation_sha256;

    std::string pop_reason;
    if (!populate_cpu_latency_chain_buffer(resources.compare_buffer.ptr.get(),
                                           sub_plan.bytes, perm, pop_reason)) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::CHAIN_GENERATION_FAILED;
      return sub;
    }

    if (!validate_cpu_latency_hamiltonian_cycle(
            resources.compare_buffer.ptr.get(), sub_plan.bytes,
            sub_plan.stride_bytes, sub_plan.node_count) ||
        (test_control != nullptr && test_control->force_hamiltonian_fail)) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::CHAIN_HAMILTONIAN_FAILED;
      return sub;
    }

    if (!protect_buffer_read_only_with_evidence(resources.compare_buffer,
                                                protect_size)) {
      sub.status = "failed";
      sub.reason_code = CpuLatencyExecutorReason::CHAIN_READONLY_FAILED;
      return sub;
    }

    sub.attempted = true;
    uintptr_t* const base_ptr =
        static_cast<uintptr_t*>(resources.compare_buffer.ptr.get());
    uintptr_t* actual_final_ptr = nullptr;

    timer.start();
    if (test_control != nullptr && test_control->force_kernel_fail) {
      (void)timer.stop();
      sub.cpu_raw = timer.last_snapshot;
      sub.elapsed_seconds = 0.0;
      sub.status = "invalid";
      sub.reason_code = CpuLatencyExecutorReason::KERNEL_FAILED;
      return sub;
    } else if (test_control != nullptr && test_control->chase_override) {
      actual_final_ptr =
          test_control->chase_override(base_ptr, total_accesses);
    } else {
      actual_final_ptr = memory_latency_chase_asm(base_ptr, total_accesses);
    }
    double elapsed_seconds = timer.stop();
    if (test_control != nullptr && test_control->force_zero_elapsed) {
      elapsed_seconds = 0.0;
      sub.cpu_raw = std::nullopt;
    } else {
      sub.cpu_raw = timer.last_snapshot;
    }
    sub.elapsed_seconds = elapsed_seconds;
    sub.timing_evaluated = true;
    sub.timing_valid = std::isfinite(elapsed_seconds) &&
                       elapsed_seconds > 0.0 && sub.cpu_raw.has_value() &&
                       sub.cpu_raw->numer > 0 && sub.cpu_raw->denom > 0;

    sub.final_pointer.expected_node = 0;
    sub.final_pointer.actual_node = pointer_to_chain_node_index(
        base_ptr, sub_plan.bytes, sub_plan.stride_bytes, sub_plan.node_count,
        actual_final_ptr);
    const bool ptr_ok = (actual_final_ptr == base_ptr) &&
                        sub.final_pointer.actual_node.has_value() &&
                        *sub.final_pointer.actual_node == 0;
    sub.final_pointer.valid = ptr_ok;

    if (!sub.timing_valid) {
      sub.status = "invalid";
      sub.reason_code = CpuLatencyExecutorReason::INVALID_ELAPSED_TIME;
      return sub;
    }
    if (!ptr_ok) {
      sub.status = "invalid";
      sub.reason_code = CpuLatencyExecutorReason::MAIN_FINAL_POINTER_FAILED;
      return sub;
    }

    const double lat_ns =
        (elapsed_seconds * 1.0e9) / static_cast<double>(total_accesses);
    if (!std::isfinite(lat_ns) || lat_ns <= 0.0) {
      sub.status = "invalid";
      sub.reason_code = "invalid-derived-metric";
      return sub;
    }

    sub.status = "measured";
    sub.reason_code = "measured";
    sub.completed_accesses = total_accesses;
    sub.latency_ns = lat_ns;
    return sub;
  } catch (...) {
    sub.status = "failed";
    sub.reason_code = CpuLatencyExecutorReason::KERNEL_FAILED;
    return sub;
  }
}
