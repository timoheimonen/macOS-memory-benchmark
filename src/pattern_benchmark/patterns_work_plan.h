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
 * @file patterns_work_plan.h
 * @brief Deterministic 7-pattern geometry, 64B worker spans, seeds, and budget preflight
 */

#ifndef PATTERNS_WORK_PLAN_H
#define PATTERNS_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/config/constants.h"

/** Canonical pattern kinds in schema v1 order. */
enum class PatternKind {
  SequentialForward = 0,
  SequentialReverse,
  Strided64,
  Strided4096,
  Strided16384,
  Strided2MiB,
  Random,
};

inline constexpr size_t kPatternsMaxKinds = 7;

/** Canonical pattern operations in schema v1 order. */
enum class PatternOperation {
  Read = 0,
  Write,
  Copy,
};

inline constexpr size_t kPatternsOperationCount = 3;

inline constexpr int kPatternsSchemaVersion = 1;
inline constexpr const char* kPatternsModeName = "patterns";
inline constexpr const char* kPatternsMethodologyVersion =
    "patterns-v1-arm64-phase-calibrated-seeded-balanced";
inline constexpr const char* kPatternsComponentsVersion =
    "patterns-components-v1";
inline constexpr const char* kPatternsWorkPlanVersion =
    "patterns-work-plan-v1";
inline constexpr const char* kPatternsBufferPatternVersion =
    "patterns-buffer-pattern-v1";
inline constexpr const char* kPatternsReadChecksumVersion =
    "patterns-read-checksum-v1";
inline constexpr const char* kPatternsRandomIndexGeneratorVersion =
    "splitmix64-coprime-fisher-yates-v1";
inline constexpr const char* kPatternsRunPolicyVersion =
    "bounded-loop-snapshots-v1";

inline constexpr uint64_t kPatternDomainSrc = 0x5041545F5352435FULL;
inline constexpr uint64_t kPatternDomainRandom = 0x5041545F52414E44ULL;
inline constexpr uint64_t kPatternDomainPreOpSentinel = 0x5041545F53454E54ULL;

inline constexpr size_t kPatternMaxPasses =
    Constants::BANDWIDTH_CALIBRATION_MAX_PASSES;
inline constexpr size_t kPatternMaxExactPayloadBytes =
    64ULL * 1024ULL * Constants::BYTES_PER_MB;
inline constexpr size_t kPatternMaxMeasurements = 100000;
inline constexpr size_t kPatternsAccessSizeBytes =
    Constants::PATTERN_ACCESS_SIZE_BYTES;
inline constexpr size_t kPatternsMinRandomAccesses =
    Constants::PATTERN_RANDOM_ACCESS_MIN;
inline constexpr size_t kPatternsMaxRandomAccesses =
    Constants::PATTERN_RANDOM_ACCESS_MAX;
inline constexpr double kPatternsStreamingCvWarningPct =
    Constants::PATTERN_STREAMING_CV_WARNING_PCT;
inline constexpr double kPatternsSparseCvWarningPct =
    Constants::PATTERN_SPARSE_CV_WARNING_PCT;

namespace PatternsWorkPlanReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* ACTIVE = "active";
inline constexpr const char* SKIPPED_BUFFER_TOO_SMALL =
    "buffer-too-small-for-stride-transition";
inline constexpr const char* INVALID_BUFFER_SIZE = "invalid-buffer-size";
inline constexpr const char* INVALID_WORKER_COUNT = "invalid-worker-count";
inline constexpr const char* INVALID_LOOP_COUNT = "invalid-loop-count";
inline constexpr const char* INVALID_ITERATIONS = "invalid-iterations";
inline constexpr const char* INVALID_PAGE_SIZE = "invalid-page-size";
inline constexpr const char* BUFFER_BYTES_OVERFLOW = "buffer-bytes-overflow";
inline constexpr const char* WORK_CAP_EXCEEDED = "work-cap-exceeded";
inline constexpr const char* MEASUREMENT_LIMIT_EXCEEDED =
    "measurement-limit-exceeded";
inline constexpr const char* JSON_INTEGER_OUT_OF_RANGE =
    "json-integer-out-of-range";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW = "memory-budget-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED = "memory-budget-exceeded";
inline constexpr const char* WITHIN_MEMORY_BUDGET = "within-memory-budget";
}  // namespace PatternsWorkPlanReason

/** Self-contained planning configuration consumed by `build_patterns_work_plan`. */
struct PatternsWorkPlanConfig {
  size_t buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
  size_t requested_workers = 1;
  size_t available_workers = 1;
  bool user_specified_workers = false;
  std::optional<size_t> iterations;
  size_t loop_count = 3;
  bool non_cacheable = false;
  uint64_t seed = 0;
  bool collect_json_for_sweep = false;
  std::string output_file;
  std::vector<std::string> argv;
};

/** System hardware inputs consumed by pure work planning. */
struct PatternsSystemInfo {
  size_t page_size_bytes = 0;
  size_t available_memory_bytes = 0;
};

/** Contiguous byte span assigned to one worker thread. */
struct PatternsWorkerSpan {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
  size_t accesses_per_pass = 0;
  size_t random_access_count = 0;
};

/** Expected checksum entry for one worker in a read operation plan. */
struct PatternsExpectedWorkerChecksum {
  size_t worker_index = 0;
  uint64_t state_a = 0;
  uint64_t state_b = 0;
  uint64_t exact_bytes_read = 0;
  uint64_t pass_count = 0;
};

/** Folded run checksum pair `(state_a, state_b)`. */
struct PatternsRunChecksum {
  uint64_t state_a = 0;
  uint64_t state_b = 0;
};

/** Expected checksum structure stored once per `operation_plans[]` entry. */
struct PatternsExpectedChecksum {
  bool applicable = false;
  bool available = false;
  std::string reason_code = "not-applicable";
  std::vector<PatternsExpectedWorkerChecksum> workers;
  PatternsRunChecksum run_checksum{};
};

/** Metadata for the `random` pattern workload. */
struct PatternsRandomWorkloadPlan {
  uint64_t random_seed = 0;
  size_t selected_accesses_per_pass = 0;
  size_t distinct_address_count = 0;
  size_t logical_working_set_bytes = 0;
  std::string random_indices_sha256;
};

/** Resolved geometry and worker plan for one of the 7 canonical patterns. */
struct PatternPlanEntry {
  PatternKind kind = PatternKind::SequentialForward;
  bool enabled = false;
  std::string status = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;
  std::string reason_code = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;
  size_t access_size_bytes = kPatternsAccessSizeBytes;
  size_t stride_bytes = 0;
  size_t phase_period_passes = 0;
  size_t native_page_size_bytes = 0;
  bool stride_equals_native_page_size = false;
  bool large_page_backing_verified = false;
  std::string large_page_backing_status = "not-applicable";
  size_t buffer_bytes = 0;
  size_t effective_workers = 0;
  size_t accesses_per_pass = 0;
  std::vector<PatternsWorkerSpan> worker_spans;
  std::vector<size_t> boundaries;  ///< Size effective_workers + 1, empty if skipped.
  PatternsRandomWorkloadPlan random_workload;
};

/** Canonical operation work plan referenced by calibration attempts & measurements. */
struct PatternsOperationPlan {
  bool valid = false;
  std::string reason_code = PatternsWorkPlanReason::VALID;
  size_t plan_ref = 0;
  PatternKind pattern = PatternKind::SequentialForward;
  PatternOperation operation = PatternOperation::Read;
  size_t passes = 0;
  bool explicit_iterations = false;
  size_t total_accesses = 0;
  size_t exact_payload_bytes = 0;
  size_t accesses_per_pass = 0;
  size_t min_accesses_per_pass = 0;
  size_t max_accesses_per_pass = 0;
  size_t distinct_address_count = 0;
  size_t logical_working_set_bytes = 0;
  size_t completed_phase_cycles = 0;
  bool has_untouched_dst_bytes = false;
  uint64_t op_seed = 0;
  PatternsExpectedChecksum expected_checksum;
};

/** Conservative JSON DOM peak reserve estimate. */
struct PatternsJsonPeakEstimate {
  bool valid = false;
  std::string reason_code = "json-output-peak-bytes-overflow";
  size_t fixed_schema_bytes = 0;
  size_t input_string_bytes = 0;
  size_t measurement_record_bytes = 0;
  size_t worker_checksum_bytes = 0;
  size_t total_bytes = 0;
};

/** Strict preflight memory budget record. */
struct PatternsMemoryBudget {
  size_t requested_data_bytes = 0;
  size_t committed_data_bytes = 0;
  size_t resource_rounding_bytes = 0;
  size_t index_structure_bytes = 0;
  size_t setup_peak_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t runtime_peak_bytes = 0;
  size_t known_owned_peak_bytes = 0;
  size_t available_memory_bytes = 0;
  size_t allowed_memory_bytes = 0;
  size_t admitted_budget_bytes = 0;
  bool used_fallback = false;
  bool valid = false;
  std::string reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
};

/** Component identity strings emitted under `resolved_plan.component_identities`. */
struct PatternsComponentIdentities {
  std::string version = kPatternsComponentsVersion;
  std::string work_plan_version = kPatternsWorkPlanVersion;
  std::string buffer_pattern_version = kPatternsBufferPatternVersion;
  std::string read_checksum_version = kPatternsReadChecksumVersion;
  std::string random_index_generator_version = kPatternsRandomIndexGeneratorVersion;
  std::string run_policy_version = kPatternsRunPolicyVersion;
};

/** Complete pure work plan built prior to buffer allocation. */
struct PatternsWorkPlan {
  bool valid = false;
  std::string reason_code = PatternsWorkPlanReason::INVALID_BUFFER_SIZE;
  std::string plan_identity;
  PatternsComponentIdentities component_identities;
  size_t buffer_bytes = 0;
  size_t page_rounded_bytes = 0;
  uint64_t base_seed = 0;
  uint64_t src_seed = 0;
  uint64_t random_domain_seed = 0;
  uint64_t pre_op_sentinel_seed = 0;
  std::array<PatternPlanEntry, kPatternsMaxKinds> patterns;
  std::vector<PatternKind> active_patterns;
  std::vector<PatternsWorkerSpan> buffer_worker_spans;
  PatternsJsonPeakEstimate json_peak_estimate;
  PatternsMemoryBudget memory_budget;

  const PatternPlanEntry* find_pattern(PatternKind kind) const noexcept;
  const PatternPlanEntry* entry(PatternKind kind) const noexcept;
};

/** Return canonical lowercase token for a pattern kind. */
const char* patterns_kind_to_string(PatternKind kind) noexcept;

/** Return canonical lowercase token for an operation. */
const char* patterns_operation_to_string(PatternOperation op) noexcept;

/** Return canonical array index `0..6` for a pattern kind. */
size_t patterns_kind_index(PatternKind kind) noexcept;

/** Return canonical array index `0..2` for an operation. */
size_t patterns_operation_index(PatternOperation op) noexcept;

/** Return true if `kind` is one of the 4 strided patterns. */
bool patterns_kind_is_strided(PatternKind kind) noexcept;

/** Return stride in bytes (`0` for sequential/random; `64`, `4096`, `16384`, `2097152` for strided). */
size_t patterns_kind_stride_bytes(PatternKind kind) noexcept;

/** Return true if `kind` is streaming (`SequentialForward`, `SequentialReverse`, `Strided64`). */
bool patterns_kind_is_streaming(PatternKind kind) noexcept;

/** Return CV warning threshold (`5.0` streaming, `10.0` sparse). */
double patterns_cv_warning_threshold_pct(PatternKind kind) noexcept;

/** Derive `src_seed = splitmix64_mix(base_seed ^ kPatternDomainSrc)`. */
uint64_t derive_patterns_src_seed(uint64_t base_seed) noexcept;

/** Derive `random_domain_seed = splitmix64_mix(base_seed ^ kPatternDomainRandom)`. */
uint64_t derive_patterns_random_domain_seed(uint64_t base_seed) noexcept;

/** Derive `pre_op_sentinel_seed = splitmix64_mix(base_seed ^ kPatternDomainPreOpSentinel)`. */
uint64_t derive_patterns_pre_op_sentinel_seed(uint64_t base_seed) noexcept;

/** Derive deterministic `op_seed` for `(src_seed, kind, op, passes)`. */
uint64_t derive_patterns_op_seed(uint64_t src_seed,
                                 PatternKind kind,
                                 PatternOperation op,
                                 size_t passes) noexcept;

/**
 * Partition `total_bytes` across `workers` matching `partition_64b`,
 * guaranteeing `offset_bytes % 64 == 0` for `idx > 0` whenever possible.
 */
std::vector<PatternsWorkerSpan> partition_patterns_64b(
    size_t total_bytes, size_t workers);

/** Build a deterministic work plan and perform strict memory budget preflight. */
PatternsWorkPlan build_patterns_work_plan(
    const PatternsWorkPlanConfig& config,
    const PatternsSystemInfo& sys_info);

/** Build and validate one operation plan (checking 64 GiB and pass caps). */
PatternsOperationPlan build_patterns_operation_plan(
    const PatternPlanEntry& pattern_plan,
    uint64_t src_seed,
    PatternOperation op,
    size_t passes,
    bool explicit_iterations);

/** Compute maximum allowed passes for `(pattern_plan, op)` respecting caps. */
size_t calculate_patterns_max_passes(
    const PatternPlanEntry& pattern_plan, PatternOperation op) noexcept;

#endif  // PATTERNS_WORK_PLAN_H
