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
 * @file llm_backend.h
 * @brief Objective-C-free backend boundary for LLM memory workloads
 */

#ifndef LLM_BACKEND_H
#define LLM_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "llm_memory/llm_executor.h"

/** Stable backend-boundary reasons shared by production and fake backends. */
namespace LlmBackendReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* NOT_INITIALIZED = "backend-not-initialized";
inline constexpr const char* BACKEND_MISMATCH = "backend-mismatch";
inline constexpr const char* BACKEND_NOT_ACTIVATED = "backend-not-activated";
inline constexpr const char* BACKEND_INITIALIZATION_FAILED = "backend-initialization-failed";
inline constexpr const char* TIMER_UNAVAILABLE = "backend-timer-unavailable";
inline constexpr const char* EXECUTION_PLAN_MISMATCH = "backend-execution-plan-mismatch";
inline constexpr const char* RESOURCES_NOT_PREPARED = "backend-resources-not-prepared";
inline constexpr const char* RESOURCE_RELEASE_FAILED = "backend-resource-release-failed";
inline constexpr const char* TASK_IDENTITY_MISMATCH = "task-identity-mismatch";
inline constexpr const char* TASK_COMPLETION_MISMATCH = "task-completion-mismatch";
inline constexpr const char* VALIDATION_NOT_EVALUATED = "validation-not-evaluated";
inline constexpr const char* VALIDATION_FAILED = "validation-failed";
inline constexpr const char* INVALID_AUTHORITATIVE_ELAPSED = "invalid-authoritative-elapsed";
inline constexpr const char* TASK_UNSUPPORTED = "task-unsupported";
inline constexpr const char* METAL_DEVICE_UNAVAILABLE =
    "metal-device-unavailable";
inline constexpr const char* UNIFIED_MEMORY_REQUIRED =
    "unified-memory-required";
inline constexpr const char* APPLE7_FAMILY_REQUIRED =
    "apple7-family-required";
inline constexpr const char* ARGUMENT_BUFFER_TIER2_REQUIRED =
    "argument-buffer-tier2-required";
inline constexpr const char* METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY =
    "metal-max-buffer-length-below-segment-capacity";
inline constexpr const char* METAL_COMMAND_QUEUE_CREATION_FAILED =
    "metal-command-queue-creation-failed";
inline constexpr const char* METAL_KERNEL_COMPILATION_FAILED =
    "metal-kernel-compilation-failed";
inline constexpr const char* METAL_PIPELINE_CREATION_FAILED =
    "metal-pipeline-creation-failed";
inline constexpr const char* METAL_ARGUMENT_ENCODER_CREATION_FAILED =
    "metal-argument-encoder-creation-failed";
inline constexpr const char* METAL_ARGUMENT_BUFFER_LAYOUT_INVALID =
    "metal-argument-buffer-layout-invalid";
inline constexpr const char* SEGMENT_COUNT_CAP_EXCEEDED =
    "segment-count-cap-exceeded";
inline constexpr const char* PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY =
    "paged-block-exceeds-segment-capacity";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED =
    "memory-budget-exceeded";
inline constexpr const char* METAL_RESOURCE_ALLOCATION_FAILED =
    "metal-resource-allocation-failed";
inline constexpr const char* METAL_RESOURCE_INITIALIZATION_FAILED =
    "metal-resource-initialization-failed";
inline constexpr const char* PREPARATION_INTERRUPTED =
    "preparation-interrupted";
inline constexpr const char* PLAN_RESOURCE_IDENTITY_MISMATCH =
    "plan-resource-identity-mismatch";
inline constexpr const char* POST_VALIDATION_COMMAND_FAILED =
    "post-validation-command-failed";
inline constexpr const char* PADDING_CANARY_MISMATCH =
    "padding-canary-mismatch";
}  // namespace LlmBackendReason

/** Backend lifecycle outcome; unsupported is distinct from runtime failure. */
enum class LlmBackendStatus : uint8_t {
  NotStarted = 0,
  Ready,
  Unsupported,
  Failed,
};

/** Atomic task outcome before the common runner applies acceptance rules. */
enum class LlmTaskExecutionStatus : uint8_t {
  NotStarted = 0,
  Complete,
  Invalid,
  Unsupported,
  Failed,
};

/** Excluded or measured task category exposed to every backend. */
enum class LlmRunnerTaskKind : uint8_t {
  Warmup = 0,
  Calibration,
  Measurement,
};

inline constexpr size_t kLlmNoTaskIndex = std::numeric_limits<size_t>::max();

/** Cold-path identity for one synchronous backend invocation. */
struct LlmRunnerTaskContext {
  LlmRunnerTaskKind kind = LlmRunnerTaskKind::Warmup;
  std::string_view purpose;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  size_t attempt_index = kLlmNoTaskIndex;
  size_t loop_index = kLlmNoTaskIndex;
  size_t order_position = kLlmNoTaskIndex;
};

/**
 * Status and stable reason for one backend lifecycle transition.
 *
 * The reason view refers either to static storage or to the command-owned
 * backend evidence and remains valid until the next lifecycle reset.
 */
struct LlmBackendLifecycleResult {
  LlmBackendStatus status = LlmBackendStatus::NotStarted;
  std::string_view reason_code = LlmBackendReason::NOT_INITIALIZED;
};

/** Backend-owned memory needed concurrently with the common runner. */
struct LlmBackendAuxiliaryEstimate {
  bool valid = false;
  std::string_view reason_code = LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  size_t total_auxiliary_bytes = 0;
  std::variant<std::monostate, LlmExecutorAuxiliaryEstimate> backend_evidence;
};

/** Transient exact identity copied from the frozen task inputs. */
struct LlmTaskIdentity {
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  LlmPhase phase = LlmPhase::Decode;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  LlmKvWriteKind kv_write_kind = LlmKvWriteKind::None;
  LlmRunnerTaskKind task_kind = LlmRunnerTaskKind::Warmup;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  size_t attempt_index = kLlmNoTaskIndex;
  size_t loop_index = kLlmNoTaskIndex;
  size_t order_position = kLlmNoTaskIndex;
  std::string_view purpose;
  std::string_view model_plan_identity;
  std::string_view scenario_plan_identity;
};

/** Backend-authoritative duration; diagnostic timers remain tagged evidence. */
struct LlmAuthoritativeTiming {
  bool evaluated = false;
  bool valid = false;
  double elapsed_seconds = 0.0;
};

/** Exact planned and completed logical work reported by one task. */
struct LlmTaskCompletion {
  size_t planned_work_units = 0;
  size_t completed_work_units = 0;
  size_t completed_effective_model_payload_bytes = 0;
  size_t completed_layout_metadata_lookup_count = 0;
  size_t completed_layout_metadata_read_bytes = 0;
  size_t completed_task_accounted_bytes = 0;
};

/** Backend-independent post-validation outcome. */
struct LlmTaskValidation {
  bool evaluated = false;
  bool valid = false;
};

/** Unchanged legacy CPU executor evidence retained behind the adapter. */
struct LlmCpuTaskEvidence {
  LlmExecutorResult executor;
};

/** Phase-8 Metal tasks are deliberately unsupported until timed kernels land. */
struct LlmMetalTaskEvidence {
  bool timed_pipeline_available = false;
  bool timing_evaluated = false;
};

using LlmTaggedTaskEvidence = std::variant<std::monostate, LlmCpuTaskEvidence, LlmMetalTaskEvidence>;

/** Generic result consumed synchronously by the logical runner. */
struct LlmTaskExecutionResult {
  LlmTaskExecutionStatus status = LlmTaskExecutionStatus::NotStarted;
  std::string reason_code = LlmBackendReason::NOT_INITIALIZED;
  LlmTaskIdentity identity;
  LlmAuthoritativeTiming timing;
  LlmTaskCompletion completion;
  LlmTaskValidation validation;
  LlmTaggedTaskEvidence backend_evidence;
};

/** CPU allocation/preparation evidence exposed through the generic snapshot. */
struct LlmCpuBackendEvidence {
  LlmResourcePreparationResult preparation;
};

/** Bounded Objective-C/Metal error diagnostics behind the pure C++ boundary. */
struct LlmMetalErrorDiagnostic {
  std::string domain;
  long long code = 0;
  std::string description;
};

/** Runtime properties for one compiled Phase-8 foundation pipeline. */
struct LlmMetalPipelineEvidence {
  std::string label;
  size_t thread_execution_width = 0;
  size_t max_total_threads_per_threadgroup = 0;
};

/** Capability, compiler, source, and argument-encoder evidence. */
struct LlmMetalCapabilityEvidence {
  std::string device_name;
  uint64_t registry_id = 0;
  bool has_unified_memory = false;
  bool required_apple7_family_supported = false;
  bool argument_buffers_tier2_supported = false;
  std::vector<std::string> supported_families;
  size_t max_buffer_length = 0;
  uint64_t recommended_max_working_set_size = 0;
  std::string compilation_mode = "runtime-source";
  std::string msl_language_version = "2.3";
  std::string kernel_revision;
  std::string kernel_source_sha256;
  std::string compiler_diagnostics;
  std::string compiler_identifier;
  std::string build_sdk;
  std::string deployment_target;
  size_t argument_buffer_encoded_length = 0;
  size_t argument_buffer_alignment = 0;
  bool layout_probe_evaluated = false;
  bool layout_probe_valid = false;
  size_t layout_probe_resource_count = 0;
  std::vector<LlmMetalPipelineEvidence> foundation_pipelines;
  LlmMetalErrorDiagnostic error;
};

/** Actual Metal allocation metadata for one owned candidate resource. */
struct LlmMetalResourceMetadata {
  std::string label;
  std::string pool;
  size_t pool_index = 0;
  std::string storage_mode;
  std::string cpu_cache_mode;
  std::string hazard_tracking_mode;
  uint64_t resource_options = 0;
  size_t length_bytes = 0;
  bool allocated_size_available = false;
  size_t allocated_size_bytes = 0;
  size_t committed_bytes = 0;
  size_t allocation_rounding_bytes = 0;
};

/** Atomic candidate-allocation, initialization, and cleanup evidence. */
struct LlmMetalResourceEvidence {
  bool allocation_attempted = false;
  bool allocation_completed = false;
  bool initialization_completed = false;
  bool table_upload_completed = false;
  bool table_validation_completed = false;
  std::optional<LlmKvPermutationIdentity> table_permutation;
  bool post_validation_completed = false;
  bool cpu_sample_readback_validation_completed = false;
  bool candidate_cleanup_completed = false;
  bool resources_published = false;
  size_t persistent_resource_length_bytes = 0;
  size_t committed_resource_bytes = 0;
  size_t resource_rounding_bytes = 0;
  size_t layout_padding_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t additional_owned_bytes = 0;
  size_t known_owned_peak_bytes = 0;
  size_t admitted_budget_bytes = 0;
  uint64_t current_allocated_size_before = 0;
  uint64_t current_allocated_size_peak = 0;
  uint64_t current_allocated_size_after_release = 0;
  bool recommended_working_set_available = false;
  int64_t recommended_working_set_headroom_bytes = 0;
  double recommended_working_set_headroom_fraction = 0.0;
  bool exceeds_recommended_working_set = false;
  std::vector<LlmMetalResourceMetadata> resources;
  LlmMetalErrorDiagnostic error;
};

/** Inactive Metal foundation snapshot; it contains no timed result. */
struct LlmMetalBackendEvidence {
  LlmMetalCapabilityEvidence capability;
  LlmMetalResourceEvidence resources;
  bool timed_results_available = false;
};

using LlmTaggedBackendEvidence = std::variant<std::monostate, LlmCpuBackendEvidence, LlmMetalBackendEvidence>;

/** Complete command-scoped lifecycle snapshot owned by a backend instance. */
struct LlmBackendEvidence {
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  LlmBackendLifecycleResult initialization;
  LlmBackendLifecycleResult plan_resolution;
  LlmBackendLifecycleResult preparation;
  LlmBackendLifecycleResult release;
  LlmTaggedBackendEvidence backend_evidence;
};

/**
 * Synchronous Objective-C-free execution boundary used by the common runner.
 *
 * One command owns one instance. Calls occur in initialize, plan resolution,
 * preparation, zero or more whole tasks, and release order. `execute_task()`
 * contains the complete reset, timed execution, checksum comparison, and
 * post-validation lifecycle; the runner never polls stop state inside it.
 * Implementations retain resources until `release_resources()` or destruction
 * and are not safe for concurrent calls. Expected capability and execution
 * failures use status results. Allocation failures or unexpected implementation
 * exceptions may escape initialization, plan resolution, preparation, or task
 * execution; the runner converts them to terminal reason evidence and still
 * calls the non-throwing release boundary.
 */
class LlmBackend {
 public:
  virtual ~LlmBackend() = default;

  virtual LlmMemoryBackend kind() const noexcept = 0;
  virtual LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmAuxiliaryPreflightView& preflight) const noexcept;
  virtual LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept = 0;
  virtual LlmBackendLifecycleResult initialize(const LlmMemoryConfig& config) = 0;
  virtual LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan& model_plan) = 0;
  virtual LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model_plan) = 0;
  virtual LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan,
                                              const LlmScenarioWorkPlan& scenario_plan,
                                              const LlmRunnerTaskContext& context) = 0;
  virtual const LlmBackendEvidence& evidence() const noexcept = 0;
  virtual LlmBackendLifecycleResult release_resources() noexcept = 0;
};

/** Create the selected backend without probing capabilities or allocating. */
std::unique_ptr<LlmBackend> create_llm_backend(LlmMemoryBackend backend);

/** Return CPU preparation evidence when the snapshot carries the CPU tag. */
const LlmResourcePreparationResult* get_llm_cpu_preparation(const LlmBackendEvidence& evidence) noexcept;

/** Return legacy CPU evidence when a generic task carries the CPU tag. */
const LlmExecutorResult* get_llm_cpu_task_evidence(const LlmTaskExecutionResult& result) noexcept;
LlmExecutorResult* get_llm_cpu_task_evidence(LlmTaskExecutionResult& result) noexcept;

/** Return Metal evidence only when the snapshot carries the Metal tag. */
const LlmMetalBackendEvidence* get_llm_metal_backend_evidence(
    const LlmBackendEvidence& evidence) noexcept;

/** Return Metal task evidence only when the result carries the Metal tag. */
const LlmMetalTaskEvidence* get_llm_metal_task_evidence(
    const LlmTaskExecutionResult& result) noexcept;

/** Return stable diagnostic tokens for backend and task statuses. */
const char* llm_backend_status_to_string(LlmBackendStatus status) noexcept;
const char* llm_task_execution_status_to_string(LlmTaskExecutionStatus status) noexcept;

#endif  // LLM_BACKEND_H
