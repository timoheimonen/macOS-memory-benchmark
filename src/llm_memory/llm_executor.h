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
 * @file llm_executor.h
 * @brief Mapping, initialization, descriptor, checksum, and CPU execution boundary
 */

#ifndef LLM_EXECUTOR_H
#define LLM_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "core/memory/memory_manager.h"
#include "llm_memory/llm_work_plan.h"

struct HighResTimer;

/** Stable machine-readable reasons returned by the LLM executor boundary. */
namespace LlmExecutorReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* INVALID_WORK_PLAN = "invalid-work-plan";
inline constexpr const char* INVALID_DESCRIPTOR_LAYOUT = "invalid-descriptor-layout";
inline constexpr const char* AUXILIARY_BYTES_OVERFLOW = "auxiliary-bytes-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED = "memory-budget-exceeded";
inline constexpr const char* OUTPUT_NOT_EMPTY = "output-not-empty";
inline constexpr const char* WEIGHT_MAPPING_FAILED = "weight-mapping-failed";
inline constexpr const char* K_MAPPING_FAILED = "k-mapping-failed";
inline constexpr const char* V_MAPPING_FAILED = "v-mapping-failed";
inline constexpr const char* DESCRIPTOR_ALLOCATION_FAILED = "descriptor-allocation-failed";
inline constexpr const char* INITIALIZATION_FAILED = "initialization-failed";
inline constexpr const char* INVALID_RESOURCES = "invalid-resources";
inline constexpr const char* INVALID_SCENARIO_PLAN = "invalid-scenario-plan";
inline constexpr const char* SCENARIO_PLAN_MISMATCH = "scenario-plan-mismatch";
inline constexpr const char* EXPECTED_CHECKSUM_OVERFLOW = "expected-checksum-overflow";
inline constexpr const char* EXPECTED_CHECKSUM_ALLOCATION_FAILED = "expected-checksum-allocation-failed";
inline constexpr const char* WORKER_STARTUP_FAILED = "worker-startup-failed";
inline constexpr const char* KERNEL_FAILED = "kernel-failed";
inline constexpr const char* INVALID_ELAPSED_TIME = "invalid-elapsed-time";
inline constexpr const char* CHECKSUM_MISMATCH = "checksum-mismatch";
}  // namespace LlmExecutorReason

inline constexpr uint64_t kLlmScenarioFlagWeight = 1ULL;
inline constexpr uint64_t kLlmScenarioFlagKv = 2ULL;
inline constexpr uint64_t kLlmScenarioFlagMixed = kLlmScenarioFlagWeight | kLlmScenarioFlagKv;

/** Logical read-checksum component domains in canonical fold order. */
enum class LlmChecksumComponent : uint8_t {
  Weight = 0,
  K,
  V,
};

/** Frozen `llm-read-checksum-v1` state for one worker/component. */
struct alignas(16) LlmReadChecksumComponent {
  uint64_t state_a;
  uint64_t state_b;
  uint64_t exact_bytes_read;
  uint64_t span_count;
};

/** Frozen assembly output for one worker, ordered weight, K, then V. */
struct alignas(16) LlmWorkerChecksum {
  LlmReadChecksumComponent weight;
  LlmReadChecksumComponent k;
  LlmReadChecksumComponent v;
};

/** Canonical worker/component fold stored outside the measured interval. */
struct LlmRunChecksum {
  uint64_t state_a;
  uint64_t state_b;
};

/**
 * Static sums captured while a finalized read span is initialized.
 *
 * Word parity restarts at each span, and the final partial word is
 * little-endian and zero-padded. Empty spans use an all-zero reference.
 */
struct LlmStaticSpanReference {
  uint64_t span_even;
  uint64_t span_odd;
  uint64_t span_bytes;
};

static_assert(std::is_standard_layout_v<LlmReadChecksumComponent>);
static_assert(alignof(LlmReadChecksumComponent) == 16);
static_assert(sizeof(LlmReadChecksumComponent) == 32);
static_assert(offsetof(LlmReadChecksumComponent, state_a) == 0);
static_assert(offsetof(LlmReadChecksumComponent, state_b) == 8);
static_assert(offsetof(LlmReadChecksumComponent, exact_bytes_read) == 16);
static_assert(offsetof(LlmReadChecksumComponent, span_count) == 24);
static_assert(std::is_standard_layout_v<LlmWorkerChecksum>);
static_assert(alignof(LlmWorkerChecksum) == 16);
static_assert(sizeof(LlmWorkerChecksum) == 96);
static_assert(offsetof(LlmWorkerChecksum, weight) == 0);
static_assert(offsetof(LlmWorkerChecksum, k) == 32);
static_assert(offsetof(LlmWorkerChecksum, v) == 64);
static_assert(sizeof(LlmRunChecksum) == 16);
static_assert(sizeof(LlmStaticSpanReference) == 24);

/**
 * Exact executor-owned backing-byte estimate used for budget re-admission.
 * `reason_code` references static storage, so copies remain self-contained.
 */
struct LlmExecutorAuxiliaryEstimate {
  bool valid = false;
  std::string_view reason_code = LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
  size_t static_reference_bytes = 0;
  size_t expected_checksum_bytes = 0;
  size_t actual_checksum_bytes = 0;
  size_t run_checksum_bytes = 0;
  size_t worker_status_bytes = 0;
  size_t thread_handle_bytes = 0;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  size_t total_auxiliary_bytes = 0;
};

/** Three regular cached mappings published only after all allocations succeed. */
struct LlmBufferSet {
  MmapPtr weight{nullptr, MmapDeleter{0}};
  MmapPtr k{nullptr, MmapDeleter{0}};
  MmapPtr v{nullptr, MmapDeleter{0}};

  bool complete() const noexcept;
};

/** Result and peak-memory evidence from atomic three-mapping allocation. */
struct LlmBufferAllocationResult {
  bool valid = false;
  std::string reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
  LlmExecutorAuxiliaryEstimate auxiliary;
  LlmMemoryBudget memory_budget;
};

/** Proof that every requested mapping byte was initialized and pre-touched. */
struct LlmInitializationEvidence {
  bool complete = false;
  size_t weight_bytes = 0;
  size_t k_bytes = 0;
  size_t v_bytes = 0;
  size_t total_bytes = 0;
  size_t non_empty_weight_spans = 0;
  size_t non_empty_k_spans = 0;
  size_t non_empty_v_spans = 0;
};

/**
 * Move-only ownership for mappings, materialized ABI arrays, and static sums.
 *
 * Descriptor and reference arrays are flat worker-major arrays. Worker `w`
 * starts at `w * layer_descriptors_per_worker` and
 * `w * sequence_descriptors_per_worker`, respectively. Their addresses remain
 * stable for the lifetime of this object. No method is thread-safe against a
 * concurrent move or destruction; immutable access from worker threads is safe.
 */
struct LlmExecutionResources {
  LlmExecutionResources() = default;
  LlmExecutionResources(const LlmExecutionResources&) = delete;
  LlmExecutionResources& operator=(const LlmExecutionResources&) = delete;
  LlmExecutionResources(LlmExecutionResources&&) noexcept = default;
  LlmExecutionResources& operator=(LlmExecutionResources&&) noexcept = default;

  bool valid = false;
  std::string model_plan_identity;
  LlmBufferSet buffers;
  std::unique_ptr<LlmLayerDescriptor[]> layer_descriptors;
  std::unique_ptr<LlmKvSequenceDescriptor[]> sequence_descriptors;
  std::unique_ptr<LlmStaticSpanReference[]> weight_references;
  std::unique_ptr<LlmStaticSpanReference[]> k_references;
  std::unique_ptr<LlmStaticSpanReference[]> v_references;
  size_t worker_count = 0;
  size_t layer_descriptors_per_worker = 0;
  size_t sequence_descriptors_per_worker = 0;
  size_t total_layer_descriptors = 0;
  size_t total_sequence_descriptors = 0;
  LlmExecutorAuxiliaryEstimate auxiliary;
  LlmMemoryBudget memory_budget;
  LlmInitializationEvidence initialization;

  const LlmLayerDescriptor* worker_layers(size_t worker_index) const noexcept;
  const LlmKvSequenceDescriptor* worker_sequences(size_t worker_index) const noexcept;
  const LlmStaticSpanReference* worker_weight_references(size_t worker_index) const noexcept;
  const LlmStaticSpanReference* worker_k_references(size_t worker_index) const noexcept;
  const LlmStaticSpanReference* worker_v_references(size_t worker_index) const noexcept;
};

/** Result of atomic allocation, ABI materialization, and deterministic init. */
struct LlmResourcePreparationResult {
  bool valid = false;
  std::string reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
  LlmExecutorAuxiliaryEstimate auxiliary;
  LlmMemoryBudget memory_budget;
  LlmInitializationEvidence initialization;
};

/** Exact seven-argument invocation presented to a production or fake kernel. */
struct LlmKernelInvocation {
  const LlmLayerDescriptor* layers = nullptr;
  const LlmKvSequenceDescriptor* sequences = nullptr;
  uint64_t layer_count = 0;
  uint64_t step_count = 0;
  uint64_t scenario_flags = 0;
  uint64_t scenario_seed = 0;
  LlmWorkerChecksum* output = nullptr;
  size_t worker_index = 0;
};

/**
 * Kernel adapter seam. Production calls the dedicated assembly once per
 * worker. A fake may inspect the immutable invocation and return false to
 * inject failure. The executor catches exceptions from test adapters.
 */
using LlmKernelAdapterFunction = bool (*)(void* context, const LlmKernelInvocation& invocation);

struct LlmKernelAdapter {
  LlmKernelAdapterFunction invoke = nullptr;
  void* context = nullptr;
};

/** Test-only lifecycle observations; callbacks run on their named threads. */
enum class LlmExecutorEvent : uint8_t {
  WorkerReady = 0,
  TimerStarted,
  KernelStarted,
  KernelCompleted,
  TimerStopped,
  ChecksumValidationStarted,
  WorkerCancelled,
};

using LlmExecutorEventObserver = void (*)(void* context, LlmExecutorEvent event, size_t worker_index) noexcept;
using LlmWorkerQosFunction = int (*)(void* context, size_t worker_index) noexcept;

/** Deterministic worker startup, QoS, and event seams used only by tests. */
struct LlmExecutorTestControl {
  int fail_before_worker_index = -1;
  LlmWorkerQosFunction set_worker_qos = nullptr;
  void* qos_context = nullptr;
  LlmExecutorEventObserver observe_event = nullptr;
  void* event_context = nullptr;
};

/** Expected worker array and canonical fold for one frozen scenario task. */
struct LlmExpectedChecksumResult {
  bool valid = false;
  std::string reason_code = LlmExecutorReason::INVALID_SCENARIO_PLAN;
  std::vector<LlmWorkerChecksum> workers;
  LlmRunChecksum run_checksum{0, 0};
};

/** One synchronized task result; checksum vectors are worker-index ordered. */
struct LlmExecutorResult {
  bool valid = false;
  std::string reason_code = LlmExecutorReason::INVALID_RESOURCES;
  double elapsed_seconds = 0.0;
  size_t requested_workers = 0;
  size_t created_workers = 0;
  size_t completed_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
  bool kernel_succeeded = false;
  bool timer_started = false;
  bool timer_stopped = false;
  bool checksum_evaluated = false;  ///< True only after expected and actual checksums were compared.
  bool checksum_valid = false;
  std::vector<LlmWorkerChecksum> expected_checksums;
  std::vector<LlmWorkerChecksum> actual_checksums;
  LlmRunChecksum expected_run_checksum{0, 0};
  LlmRunChecksum actual_run_checksum{0, 0};
};

/** Dedicated ARM64 implementation; @p output must be non-null. */
extern "C" void llm_decode_memory_asm(const LlmLayerDescriptor* layers, const LlmKvSequenceDescriptor* sequences,
                                      uint64_t layer_count, uint64_t step_count, uint64_t scenario_flags,
                                      uint64_t scenario_seed, LlmWorkerChecksum* output) noexcept;

/** Return the frozen assembly flag for a valid scenario, or zero. */
uint64_t llm_scenario_flags(LlmScenario scenario) noexcept;

/** Return one deterministic `llm-buffer-pattern-v1` word. */
uint64_t llm_buffer_pattern_word(uint64_t buffer_domain_seed, uint64_t absolute_mapping_word_index) noexcept;

/** Return one frozen affine K/V append word. */
uint64_t llm_append_word(uint64_t scenario_seed, uint64_t task_local_step, uint64_t layer_index,
                         uint64_t batch_sequence_index, uint64_t record_word_index,
                         LlmChecksumComponent component) noexcept;

/** Return the canonical initial state for one checksum component. */
LlmReadChecksumComponent initial_llm_read_checksum(LlmChecksumComponent component) noexcept;

/** Fold every worker's weight/K/V tuples in canonical order. */
LlmRunChecksum fold_llm_worker_checksums(const LlmWorkerChecksum* workers, size_t worker_count) noexcept;

/** Calculate exact executor-owned backing bytes with checked arithmetic. */
LlmExecutorAuxiliaryEstimate calculate_llm_executor_auxiliary_estimate(const LlmMemoryWorkPlan& plan) noexcept;

/**
 * Allocate all three regular cacheable mappings atomically.
 *
 * The finalized plan is structurally validated and re-admitted with the exact
 * executor auxiliary estimate before the first `mmap`. @p output must own no
 * mapping on entry. A populated or partial output is rejected before `mmap`,
 * so steady-workload budget evidence never omits a replacement peak. On any
 * failure, @p output is unchanged and candidate mappings are released by RAII.
 */
LlmBufferAllocationResult allocate_llm_buffers(const LlmMemoryWorkPlan& plan, LlmBufferSet& output) noexcept;

/**
 * Build all execution resources into an output that owns no prior resources.
 *
 * A populated or partial output is rejected before allocation. This keeps the
 * admitted memory peak exact and avoids an unbudgeted replacement contract.
 *
 * Each physical mapping byte is written exactly once using
 * `llm-buffer-pattern-v1`; finalized span sums are accumulated during those
 * writes, so preparation performs no separate reference-reading pass.
 */
LlmResourcePreparationResult prepare_llm_execution_resources(const LlmMemoryWorkPlan& plan,
                                                             LlmExecutionResources& output) noexcept;

/** Derive exact worker/component checksums without invoking or rereading ASM. */
LlmExpectedChecksumResult calculate_llm_expected_checksums(const LlmMemoryWorkPlan& model_plan,
                                                           const LlmScenarioWorkPlan& scenario_plan,
                                                           const LlmExecutionResources& resources) noexcept;

/** Return the production adapter that invokes `llm_decode_memory_asm`. */
LlmKernelAdapter production_llm_kernel_adapter() noexcept;

/**
 * Execute one frozen scenario with an all-or-cancel synchronized start gate.
 *
 * Descriptor validation, expected generation, worker/QoS preparation, and
 * thread creation finish before `timer.start()`. The last completing worker
 * stops the timer; joins, exact per-component validation, and run folds happen
 * afterwards. A partial worker team is cancelled before any kernel invocation.
 */
LlmExecutorResult execute_llm_scenario(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                       const LlmExecutionResources& resources, HighResTimer& timer,
                                       LlmKernelAdapter kernel = production_llm_kernel_adapter(),
                                       const LlmExecutorTestControl* test_control = nullptr) noexcept;

#endif  // LLM_EXECUTOR_H
