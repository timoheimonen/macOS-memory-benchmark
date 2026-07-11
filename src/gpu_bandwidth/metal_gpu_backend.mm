// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file metal_gpu_backend.mm
 * @brief Private Objective-C++ implementation of the Metal GPU backend
 *
 * Metal and Objective-C ownership stay in this translation unit. Every public
 * call is synchronous, uses a bounded autorelease pool, and converts Metal or
 * C++ failures into the pure C++ backend contract.
 */

#include "gpu_bandwidth/gpu_backend.h"

#include "core/config/constants.h"
#include "core/system/system_info.h"
#include "gpu_bandwidth/gpu_kernels_source.h"
#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace {

constexpr size_t kStatusWordCount = 4;
constexpr size_t kStatusBytes = kStatusWordCount * sizeof(uint32_t);

struct alignas(8) KernelParams {
  uint64_t vector_count = 0;
  uint32_t seed_low = 0;
  uint32_t seed_high = 0;
  uint32_t pattern_tag = 0;
  uint32_t pattern_pass = 0;
  uint32_t pass_index = 0;
  uint32_t tail_bytes = 0;
  uint32_t timed_element_weight_first = 0;
  uint32_t timed_element_weight_second = 0;
  uint32_t timed_dispatch_token_first = 0;
  uint32_t timed_dispatch_token_second = 0;
};

static_assert(alignof(KernelParams) == 8,
              "KernelParams alignment must match the canonical MSL layout");
static_assert(sizeof(KernelParams) == 48,
              "KernelParams must match the canonical MSL layout");
static_assert(offsetof(KernelParams, vector_count) == 0);
static_assert(offsetof(KernelParams, seed_low) == 8);
static_assert(offsetof(KernelParams, seed_high) == 12);
static_assert(offsetof(KernelParams, pattern_tag) == 16);
static_assert(offsetof(KernelParams, pattern_pass) == 20);
static_assert(offsetof(KernelParams, pass_index) == 24);
static_assert(offsetof(KernelParams, tail_bytes) == 28);
static_assert(offsetof(KernelParams, timed_element_weight_first) == 32);
static_assert(offsetof(KernelParams, timed_element_weight_second) == 36);
static_assert(offsetof(KernelParams, timed_dispatch_token_first) == 40);
static_assert(offsetof(KernelParams, timed_dispatch_token_second) == 44);

std::string ns_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* utf8 = value.UTF8String;
  return utf8 != nullptr ? std::string(utf8) : std::string();
}

GpuErrorDiagnostic error_diagnostic(NSError* error) {
  if (error == nil) {
    return {};
  }
  return {ns_string(error.domain), static_cast<long long>(error.code),
          ns_string(error.localizedDescription)};
}

GpuErrorDiagnostic internal_error(const std::string& description) {
  return {"macos-memory-benchmark.gpu", 0, description};
}

std::string read_sysctl_string(const char* key) {
  size_t length = 0;
  if (sysctlbyname(key, nullptr, &length, nullptr, 0) != 0 || length == 0) {
    return {};
  }
  std::string value(length, '\0');
  if (sysctlbyname(key, value.data(), &length, nullptr, 0) != 0) {
    return {};
  }
  if (length != 0 && value[length - 1] == '\0') {
    --length;
  }
  value.resize(length);
  return value;
}

std::string version_macro_string(long value) {
  const long major = value / 10000;
  const long minor = (value / 100) % 100;
  const long patch = value % 100;
  std::string result = std::to_string(major) + "." + std::to_string(minor);
  if (patch != 0) {
    result += "." + std::to_string(patch);
  }
  return result;
}

std::string storage_mode_string(MTLStorageMode mode) {
  switch (mode) {
    case MTLStorageModeShared:
      return "shared";
    case MTLStorageModeManaged:
      return "managed";
    case MTLStorageModePrivate:
      return "private";
    case MTLStorageModeMemoryless:
      return "memoryless";
  }
  return "unknown";
}

std::string cpu_cache_mode_string(MTLCPUCacheMode mode) {
  switch (mode) {
    case MTLCPUCacheModeDefaultCache:
      return "default-cache";
    case MTLCPUCacheModeWriteCombined:
      return "write-combined";
  }
  return "unknown";
}

std::string hazard_mode_string(MTLHazardTrackingMode mode) {
  switch (mode) {
    case MTLHazardTrackingModeDefault:
      return "default";
    case MTLHazardTrackingModeUntracked:
      return "untracked";
    case MTLHazardTrackingModeTracked:
      return "tracked";
  }
  return "unknown";
}

GpuResourceMetadata resource_metadata(id<MTLBuffer> buffer,
                                      const std::string& label) {
  GpuResourceMetadata metadata;
  metadata.label = label;
  if (buffer == nil) {
    return metadata;
  }
  metadata.storage_mode = storage_mode_string(buffer.storageMode);
  metadata.cpu_cache_mode = cpu_cache_mode_string(buffer.cpuCacheMode);
  metadata.hazard_tracking_mode =
      hazard_mode_string(buffer.hazardTrackingMode);
  metadata.resource_options = static_cast<uint64_t>(buffer.resourceOptions);
  metadata.length_bytes = static_cast<size_t>(buffer.length);
  return metadata;
}

GpuPipelineMetadata pipeline_metadata(id<MTLComputePipelineState> pipeline,
                                      const std::string& label) {
  GpuPipelineMetadata metadata;
  metadata.label = label;
  if (pipeline != nil) {
    metadata.thread_execution_width =
        static_cast<size_t>(pipeline.threadExecutionWidth);
    metadata.max_total_threads_per_threadgroup =
        static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
  }
  return metadata;
}

std::string thermal_state_string(NSProcessInfoThermalState state) {
  switch (state) {
    case NSProcessInfoThermalStateNominal:
      return "nominal";
    case NSProcessInfoThermalStateFair:
      return "fair";
    case NSProcessInfoThermalStateSerious:
      return "serious";
    case NSProcessInfoThermalStateCritical:
      return "critical";
  }
  return "unavailable";
}

double steady_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint32_t low32(uint64_t value) {
  return static_cast<uint32_t>(value);
}

uint32_t multiply_mod32(uint32_t left, uint32_t right) {
  return static_cast<uint32_t>(static_cast<uint64_t>(left) * right);
}

uint32_t avalanche_mod32(uint32_t value) {
  value ^= value >> 16U;
  value = multiply_mod32(value, 0x7feb352dU);
  value ^= value >> 15U;
  value = multiply_mod32(value, 0x846ca68bU);
  value ^= value >> 16U;
  return value;
}

uint32_t triangular_mod32(uint64_t count) {
  if ((count & 1U) == 0U) {
    return multiply_mod32(low32(count / 2U), low32(count - 1U));
  }
  return multiply_mod32(low32(count), low32((count - 1U) / 2U));
}

uint32_t mask_for_byte_count(size_t byte_count) {
  if (byte_count >= sizeof(uint32_t)) {
    return std::numeric_limits<uint32_t>::max();
  }
  return (uint32_t{1} << (byte_count * 8U)) - 1U;
}

uint32_t pattern_word(uint64_t word_index, uint64_t seed,
                      uint32_t pattern_tag, uint32_t pattern_pass) {
  const uint32_t seed_low = low32(seed);
  const uint32_t seed_high = low32(seed >> 32U);
  return seed_low +
         multiply_mod32(seed_high,
                        GpuKernelContract::kPatternSeedHighMultiplier) +
         multiply_mod32(low32(word_index),
                        GpuKernelContract::kPatternIndexMultiplier) +
         multiply_mod32(pattern_tag,
                        GpuKernelContract::kPatternTagMultiplier) +
         multiply_mod32(pattern_pass,
                        GpuKernelContract::kPatternPassMultiplier);
}

struct PatternSummary {
  uint32_t value_sum = 0;
  uint32_t index_sum = 0;
};

/**
 * Compute the exact affine pattern population in O(1).
 *
 * Full words use arithmetic-series sums. At most one final partial word is
 * masked to the bytes stored by the MSL tail path. All operations intentionally
 * wrap modulo 2^32, matching Metal `uint` behavior.
 */
PatternSummary summarize_pattern(size_t byte_count, uint64_t seed,
                                 uint32_t pattern_tag,
                                 uint32_t pattern_pass) {
  const uint64_t full_words = byte_count / sizeof(uint32_t);
  const size_t partial_bytes = byte_count % sizeof(uint32_t);
  const uint32_t base = pattern_word(0, seed, pattern_tag, pattern_pass);
  const uint32_t index_sum = triangular_mod32(full_words);

  PatternSummary summary;
  summary.value_sum =
      multiply_mod32(low32(full_words), base) +
      multiply_mod32(GpuKernelContract::kPatternIndexMultiplier,
                     index_sum);
  summary.index_sum = index_sum;
  if (partial_bytes != 0) {
    summary.value_sum +=
        pattern_word(full_words, seed, pattern_tag, pattern_pass) &
        mask_for_byte_count(partial_bytes);
    summary.index_sum += low32(full_words);
  }
  return summary;
}

uint32_t operation_tag(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return GpuKernelContract::kReadOperationTag;
    case GpuOperation::Write:
      return GpuKernelContract::kWriteOperationTag;
    case GpuOperation::Copy:
      return GpuKernelContract::kCopyOperationTag;
  }
  return 0;
}

struct TimedAccumulatorDomains {
  uint32_t element_weight_first = 0;
  uint32_t element_weight_second = 0;
  uint32_t dispatch_token_first = 0;
  uint32_t dispatch_token_second = 0;
};

/**
 * Derive pass-specific odd weights and nonzero dispatch tokens in O(1).
 *
 * The two keys use separate multiplication domains and include every frozen
 * dispatch discriminator. Avalanche mixing makes adjacent pass indices and
 * power-of-two buffer sizes affect all output bits. Setting the low bit keeps
 * element multiplication invertible modulo 2^32 and guarantees that an
 * encoded dispatch contributes a nonzero token in both lanes.
 */
TimedAccumulatorDomains derive_timed_accumulator_domains(
    const GpuBackendAttemptRequest& request, uint32_t pass_index,
    uint32_t direction) {
  const uint32_t seed_low = low32(request.operation_seed);
  const uint32_t seed_high = low32(request.operation_seed >> 32U);
  const uint32_t byte_count_low = low32(request.buffer_size_bytes);
  const uint32_t byte_count_high =
      low32(static_cast<uint64_t>(request.buffer_size_bytes) >> 32U);
  const uint32_t op_tag = operation_tag(request.operation);

  const uint32_t first_key =
      seed_low +
      multiply_mod32(seed_high,
                     GpuKernelContract::kTimedKeySeedHighFirst) +
      multiply_mod32(byte_count_low,
                     GpuKernelContract::kTimedKeyBufferLowFirst) +
      multiply_mod32(byte_count_high,
                     GpuKernelContract::kTimedKeyBufferHighFirst) +
      multiply_mod32(pass_index,
                     GpuKernelContract::kTimedKeyPassFirst) +
      multiply_mod32(op_tag,
                     GpuKernelContract::kTimedKeyOperationFirst) +
      multiply_mod32(direction,
                     GpuKernelContract::kTimedKeyDirectionFirst);
  const uint32_t second_key =
      seed_high +
      multiply_mod32(seed_low,
                     GpuKernelContract::kTimedKeySeedLowSecond) +
      multiply_mod32(byte_count_low,
                     GpuKernelContract::kTimedKeyBufferLowSecond) +
      multiply_mod32(byte_count_high,
                     GpuKernelContract::kTimedKeyBufferHighSecond) +
      multiply_mod32(pass_index,
                     GpuKernelContract::kTimedKeyPassSecond) +
      multiply_mod32(op_tag,
                     GpuKernelContract::kTimedKeyOperationSecond) +
      multiply_mod32(direction,
                     GpuKernelContract::kTimedKeyDirectionSecond);

  return {
      avalanche_mod32(first_key ^
                      GpuKernelContract::kTimedElementDomainFirst) |
          1U,
      avalanche_mod32(second_key ^
                      GpuKernelContract::kTimedElementDomainSecond) |
          1U,
      avalanche_mod32(first_key ^
                      GpuKernelContract::kTimedDispatchDomainFirst) |
          1U,
      avalanche_mod32(second_key ^
                      GpuKernelContract::kTimedDispatchDomainSecond) |
          1U,
  };
}

uint32_t source_pattern_tag(GpuOperation operation) {
  return operation == GpuOperation::Copy
             ? GpuKernelContract::kCopySourcePatternTag
             : GpuKernelContract::kReadSourcePatternTag;
}

GpuDualChecksum expected_timed_accumulator(
    const GpuBackendAttemptRequest& request) {
  GpuDualChecksum expected;
  for (size_t pass = 0; pass < request.passes; ++pass) {
    const uint32_t pattern_tag =
        request.operation == GpuOperation::Write
            ? GpuKernelContract::kWritePatternTag
            : source_pattern_tag(request.operation);
    const uint32_t pattern_pass =
        request.operation == GpuOperation::Write ? low32(pass) : 0U;
    const uint32_t direction =
        request.operation == GpuOperation::Copy ? low32(pass & 1U) : 0U;
    const PatternSummary summary = summarize_pattern(
        request.buffer_size_bytes, request.operation_seed, pattern_tag,
        pattern_pass);
    const TimedAccumulatorDomains domains =
        derive_timed_accumulator_domains(request, low32(pass), direction);

    expected.first +=
        multiply_mod32(
            summary.value_sum +
                multiply_mod32(GpuKernelContract::kTimedIndexFirst,
                               summary.index_sum),
            domains.element_weight_first) +
        domains.dispatch_token_first;
    expected.second +=
        multiply_mod32(
            multiply_mod32(summary.value_sum,
                           GpuKernelContract::kTimedValueSecond) +
                multiply_mod32(summary.index_sum,
                               GpuKernelContract::kTimedIndexSecond),
            domains.element_weight_second) +
        domains.dispatch_token_second;
  }
  return expected;
}

GpuDualChecksum expected_final_checksum(
    const GpuBackendAttemptRequest& request) {
  const uint32_t pattern_tag =
      request.operation == GpuOperation::Write
          ? GpuKernelContract::kWritePatternTag
          : source_pattern_tag(request.operation);
  const uint32_t pattern_pass =
      request.operation == GpuOperation::Write
          ? low32(request.passes - 1U)
          : 0U;
  const PatternSummary summary = summarize_pattern(
      request.buffer_size_bytes, request.operation_seed, pattern_tag,
      pattern_pass);
  return {
      summary.value_sum +
          multiply_mod32(summary.index_sum,
                         GpuKernelContract::kChecksumIndexFirst),
      multiply_mod32(summary.value_sum,
                     GpuKernelContract::kChecksumValueSecond) +
          multiply_mod32(summary.index_sum,
                         GpuKernelContract::kChecksumIndexSecond)};
}

KernelParams make_params(const GpuBackendAttemptRequest& request,
                         uint32_t pattern_tag, uint32_t pattern_pass,
                         uint32_t pass_index, uint32_t direction) {
  KernelParams params;
  params.vector_count = request.vector_count;
  params.seed_low = low32(request.operation_seed);
  params.seed_high = low32(request.operation_seed >> 32U);
  params.pattern_tag = pattern_tag;
  params.pattern_pass = pattern_pass;
  params.pass_index = pass_index;
  params.tail_bytes = static_cast<uint32_t>(request.tail_bytes);
  const TimedAccumulatorDomains domains =
      derive_timed_accumulator_domains(request, pass_index, direction);
  params.timed_element_weight_first = domains.element_weight_first;
  params.timed_element_weight_second = domains.element_weight_second;
  params.timed_dispatch_token_first = domains.dispatch_token_first;
  params.timed_dispatch_token_second = domains.dispatch_token_second;
  return params;
}

class MetalGpuBackend final : public GpuBackend {
 public:
  ~MetalGpuBackend() override {
    @autoreleasepool {
      buffer_a_ = nil;
      buffer_b_ = nil;
      status_buffer_ = nil;
      last_output_ = nil;
    }
  }

  GpuBackendInitialization initialize() noexcept override {
    @autoreleasepool {
      try {
        return initialize_impl();
      } catch (const std::exception& exception) {
        GpuBackendInitialization result;
        result.reason_code = "backend-initialization-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuBackendInitialization result;
        result.reason_code = "backend-initialization-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuAllocationResult allocate_resources(
      const GpuAllocationRequest& request) noexcept override {
    @autoreleasepool {
      try {
        return allocate_resources_impl(request);
      } catch (const std::exception& exception) {
        GpuAllocationResult result;
        result.reason_code = "resource-allocation-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuAllocationResult result;
        result.reason_code = "resource-allocation-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuEnvironmentSnapshot snapshot_environment() noexcept override {
    @autoreleasepool {
      try {
        GpuEnvironmentSnapshot snapshot;
        NSProcessInfo* process_info = NSProcessInfo.processInfo;
        snapshot.thermal_state =
            thermal_state_string(process_info.thermalState);
        if (@available(macOS 12.0, *)) {
          snapshot.low_power_mode_available = true;
          snapshot.low_power_mode_enabled = process_info.lowPowerModeEnabled;
        }
        snapshot.current_allocated_size =
            device_ != nil
                ? static_cast<uint64_t>(device_.currentAllocatedSize)
                : 0;
        return snapshot;
      } catch (...) {
        return GpuEnvironmentSnapshot{};
      }
    }
  }

  GpuBackendPhaseResult run_warmup(
      const GpuBackendAttemptRequest& request) noexcept override {
    @autoreleasepool {
      try {
        return run_setup_phase(request, true);
      } catch (const std::exception& exception) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "warmup-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "warmup-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuBackendPhaseResult run_precondition(
      const GpuBackendAttemptRequest& request) noexcept override {
    @autoreleasepool {
      try {
        return run_setup_phase(request, false);
      } catch (const std::exception& exception) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "precondition-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "precondition-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuTimedResult run_timed(
      const GpuBackendAttemptRequest& request) noexcept override {
    @autoreleasepool {
      try {
        return run_timed_impl(request);
      } catch (const std::exception& exception) {
        GpuTimedResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "timed-command-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuTimedResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "timed-command-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuValidationResult run_validation(
      const GpuBackendAttemptRequest& request,
      const GpuTimedResult& timed_result) noexcept override {
    @autoreleasepool {
      try {
        return run_validation_impl(request, timed_result);
      } catch (const std::exception& exception) {
        GpuValidationResult result;
        if (request.operation != GpuOperation::Read) {
          result.final_checksum_algorithm = "gpu-dual-mod32-v1";
        }
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "validation-exception";
        result.validation_status = GpuValidationStatus::Error;
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuValidationResult result;
        if (request.operation != GpuOperation::Read) {
          result.final_checksum_algorithm = "gpu-dual-mod32-v1";
        }
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "validation-unknown-exception";
        result.validation_status = GpuValidationStatus::Error;
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuBackendPhaseResult readback_last_output(
      std::vector<uint8_t>& output) noexcept override {
    @autoreleasepool {
      try {
        return readback_last_output_impl(output);
      } catch (const std::exception& exception) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "readback-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuBackendPhaseResult result;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "readback-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

  GpuAllocationResult release_resources() noexcept override {
    @autoreleasepool {
      try {
        return release_resources_impl();
      } catch (const std::exception& exception) {
        GpuAllocationResult result = allocation_result_;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "resource-release-exception";
        result.error = internal_error(exception.what());
        return result;
      } catch (...) {
        GpuAllocationResult result = allocation_result_;
        result.status = GpuBackendStatus::Failed;
        result.reason_code = "resource-release-unknown-exception";
        result.error = internal_error("unknown C++ exception");
        return result;
      }
    }
  }

 private:
  GpuBackendInitialization initialize_impl() {
    if (initialized_) {
      return initialization_result_;
    }

    GpuBackendInitialization result;
    result.compilation.kernel_revision = GpuKernelContract::kRevision;
    result.compilation.kernel_source_sha256 =
        canonical_gpu_kernel_source_sha256();
    result.compilation.compiler_identifier = __clang_version__;
#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
    result.compilation.build_sdk =
        version_macro_string(__MAC_OS_X_VERSION_MAX_ALLOWED);
#else
    result.compilation.build_sdk = "unknown";
#endif
#ifdef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    result.compilation.deployment_target = version_macro_string(
        __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__);
#else
    result.compilation.deployment_target = "unknown";
#endif

    // Host provenance is meaningful even when Metal has no default device, so
    // populate it before the capability boundary for unsupported checkpoints.
    result.device.macos_product_version = get_macos_version();
    result.device.macos_build = read_sysctl_string("kern.osversion");
    result.device.hardware_model = read_sysctl_string("hw.model");
    result.device.physical_memory_bytes =
        static_cast<uint64_t>(NSProcessInfo.processInfo.physicalMemory);
    const unsigned long available_memory_mb = get_available_memory_mb();
    result.device.available_memory_bytes =
        static_cast<uint64_t>(available_memory_mb) * Constants::BYTES_PER_MB;
    result.device.available_memory_source =
        "project-mach-free-plus-inactive-mib";

    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      result.status = GpuBackendStatus::Unsupported;
      result.reason_code = "metal-device-unavailable";
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }

    result.device.device_name = ns_string(device_.name);
    result.device.registry_id = device_.registryID;
    result.device.has_unified_memory = device_.hasUnifiedMemory;
    result.device.required_apple7_family_supported =
        [device_ supportsFamily:MTLGPUFamilyApple7];
    result.device.max_buffer_length =
        static_cast<size_t>(device_.maxBufferLength);
    result.device.recommended_max_working_set_size =
        device_.recommendedMaxWorkingSetSize;
    result.device.current_allocated_size = device_.currentAllocatedSize;
    append_supported_families(result.device.supported_families);
    if (!result.device.has_unified_memory) {
      result.status = GpuBackendStatus::Unsupported;
      result.reason_code = "unified-memory-required";
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    if (!result.device.required_apple7_family_supported) {
      result.status = GpuBackendStatus::Unsupported;
      result.reason_code = "apple7-family-required";
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }

    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      result.reason_code = "metal-command-queue-creation-failed";
      result.error = internal_error("newCommandQueue returned nil");
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    queue_.label = @"membenchmark.gpu.command-queue";

    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion2_3;
    options.preprocessorMacros = @{};
    NSString* source = [[NSString alloc]
        initWithBytes:GpuKernelContract::kSource.data()
               length:GpuKernelContract::kSource.size()
             encoding:NSUTF8StringEncoding];
    if (source == nil) {
      result.reason_code = "kernel-source-utf8-conversion-failed";
      result.error = internal_error("canonical MSL source is not valid UTF-8");
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }

    NSError* compile_error = nil;
    library_ = [device_ newLibraryWithSource:source
                                    options:options
                                      error:&compile_error];
    if (compile_error != nil) {
      result.compilation.compiler_diagnostics =
          ns_string(compile_error.localizedDescription);
    }
    if (library_ == nil) {
      result.reason_code = "metal-runtime-compilation-failed";
      result.error = error_diagnostic(compile_error);
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    library_.label = @"membenchmark.gpu.runtime-library";

    NSError* pipeline_error = nil;
    initialization_pipeline_ = create_pipeline(
        @"gpu_fill_pattern", @"membenchmark.gpu.pipeline.initialization",
        &pipeline_error);
    if (!record_pipeline_failure(initialization_pipeline_, pipeline_error,
                                 result)) {
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    read_pipeline_ = create_pipeline(
        @"gpu_read_bandwidth", @"membenchmark.gpu.pipeline.read",
        &pipeline_error);
    if (!record_pipeline_failure(read_pipeline_, pipeline_error, result)) {
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    write_pipeline_ = create_pipeline(
        @"gpu_write_bandwidth", @"membenchmark.gpu.pipeline.write",
        &pipeline_error);
    if (!record_pipeline_failure(write_pipeline_, pipeline_error, result)) {
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    copy_pipeline_ = create_pipeline(
        @"gpu_copy_bandwidth", @"membenchmark.gpu.pipeline.copy",
        &pipeline_error);
    if (!record_pipeline_failure(copy_pipeline_, pipeline_error, result)) {
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }
    validation_pipeline_ = create_pipeline(
        @"gpu_checksum", @"membenchmark.gpu.pipeline.validation",
        &pipeline_error);
    if (!record_pipeline_failure(validation_pipeline_, pipeline_error,
                                 result)) {
      initialization_result_ = result;
      initialized_ = true;
      return result;
    }

    result.device.initialization_pipeline = pipeline_metadata(
        initialization_pipeline_, "membenchmark.gpu.pipeline.initialization");
    result.device.read_pipeline =
        pipeline_metadata(read_pipeline_, "membenchmark.gpu.pipeline.read");
    result.device.write_pipeline = pipeline_metadata(
        write_pipeline_, "membenchmark.gpu.pipeline.write");
    result.device.copy_pipeline =
        pipeline_metadata(copy_pipeline_, "membenchmark.gpu.pipeline.copy");
    result.device.validation_pipeline = pipeline_metadata(
        validation_pipeline_, "membenchmark.gpu.pipeline.validation");
    result.status = GpuBackendStatus::Success;
    result.reason_code = "success";
    initialization_result_ = result;
    initialized_ = true;
    return result;
  }

  id<MTLComputePipelineState> create_pipeline(NSString* function_name,
                                              NSString* label,
                                              NSError** error) {
    *error = nil;
    id<MTLFunction> function = [library_ newFunctionWithName:function_name];
    if (function == nil) {
      *error = [NSError
          errorWithDomain:@"macos-memory-benchmark.gpu"
                     code:1
                 userInfo:@{
                   NSLocalizedDescriptionKey :
                       [NSString stringWithFormat:@"Missing function %@",
                                                  function_name]
                 }];
      return nil;
    }
    function.label = label;
    MTLComputePipelineDescriptor* descriptor =
        [MTLComputePipelineDescriptor new];
    descriptor.label = label;
    descriptor.computeFunction = function;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    return [device_ newComputePipelineStateWithDescriptor:descriptor
                                                   options:MTLPipelineOptionNone
                                                reflection:nil
                                                     error:error];
  }

  bool record_pipeline_failure(id<MTLComputePipelineState> pipeline,
                               NSError* error,
                               GpuBackendInitialization& result) {
    if (pipeline != nil) {
      return true;
    }
    result.reason_code = "metal-pipeline-creation-failed";
    result.error = error_diagnostic(error);
    return false;
  }

  void append_supported_families(std::vector<std::string>& families) {
    struct FamilyEntry {
      NSInteger raw_value;
      const char* name;
    };
    constexpr FamilyEntry kBaselineFamilies[] = {
        {1001, "apple1"}, {1002, "apple2"}, {1003, "apple3"},
        {1004, "apple4"}, {1005, "apple5"}, {1006, "apple6"},
        {1007, "apple7"}, {1008, "apple8"}, {1009, "apple9"},
        {1010, "apple10"}, {2001, "mac1"}, {2002, "mac2"},
        {3001, "common1"}, {3002, "common2"}, {3003, "common3"},
        {4001, "mac-catalyst1"}, {4002, "mac-catalyst2"}};
    for (const FamilyEntry& entry : kBaselineFamilies) {
      if ([device_ supportsFamily:static_cast<MTLGPUFamily>(entry.raw_value)]) {
        families.emplace_back(entry.name);
      }
    }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
    if (@available(macOS 13.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal3]) {
        families.emplace_back("metal3");
      }
    }
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal4]) {
        families.emplace_back("metal4");
      }
    }
#endif
  }

  GpuAllocationResult allocate_resources_impl(
      const GpuAllocationRequest& request) {
    GpuAllocationResult result;
    result.requested_buffer_size_bytes = request.buffer_size_bytes;
    result.auxiliary_bytes = request.auxiliary_bytes;
    result.memory_budget_bytes = request.memory_budget_bytes;
    if (!initialized_ ||
        initialization_result_.status != GpuBackendStatus::Success) {
      result.reason_code = "backend-not-ready";
      return result;
    }
    if (request.buffer_size_bytes == 0) {
      result.reason_code = "buffer-size-zero";
      return result;
    }
    if (request.auxiliary_bytes < kStatusBytes) {
      result.reason_code = "auxiliary-buffer-too-small";
      return result;
    }
    if (request.buffer_size_bytes > device_.maxBufferLength) {
      result.reason_code = "buffer-exceeds-metal-max-buffer-length";
      return result;
    }

    size_t data_bytes = 0;
    if (!NumericUtils::checked_multiply(request.buffer_size_bytes, 2,
                                        data_bytes) ||
        !NumericUtils::checked_add(data_bytes, request.auxiliary_bytes,
                                   result.required_total_bytes)) {
      result.reason_code = "allocation-size-overflow";
      return result;
    }
    if (request.memory_budget_bytes == 0) {
      result.reason_code = "memory-budget-zero";
      return result;
    }
    if (result.required_total_bytes > request.memory_budget_bytes) {
      result.reason_code = "memory-budget-exceeded";
      return result;
    }

    release_buffer_references();
    result.current_allocated_size_before = device_.currentAllocatedSize;
    const uint64_t recommended = device_.recommendedMaxWorkingSetSize;
    if (recommended != 0) {
      result.recommended_working_set_available = true;
      const long double headroom =
          static_cast<long double>(recommended) -
          static_cast<long double>(result.current_allocated_size_before) -
          static_cast<long double>(result.required_total_bytes);
      const long double clamped_headroom = std::max(
          static_cast<long double>(std::numeric_limits<int64_t>::min()),
          std::min(headroom, static_cast<long double>(
                                  std::numeric_limits<int64_t>::max())));
      result.recommended_working_set_headroom_bytes =
          static_cast<int64_t>(clamped_headroom);
      result.recommended_working_set_headroom_fraction =
          static_cast<double>(headroom /
                              static_cast<long double>(recommended));
      result.exceeds_recommended_working_set = headroom < 0.0L;
    }

    const MTLResourceOptions private_options =
        MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
    const MTLResourceOptions shared_options =
        MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    buffer_a_ = [device_ newBufferWithLength:request.buffer_size_bytes
                                     options:private_options];
    if (buffer_a_ != nil) {
      buffer_a_.label = @"membenchmark.gpu.buffer-a";
    }
    buffer_b_ = [device_ newBufferWithLength:request.buffer_size_bytes
                                     options:private_options];
    if (buffer_b_ != nil) {
      buffer_b_.label = @"membenchmark.gpu.buffer-b";
    }
    status_buffer_ = [device_ newBufferWithLength:request.auxiliary_bytes
                                          options:shared_options];
    if (status_buffer_ != nil) {
      status_buffer_.label = @"membenchmark.gpu.status";
    }
    if (buffer_a_ == nil || buffer_b_ == nil || status_buffer_ == nil ||
        status_buffer_.contents == nullptr) {
      release_buffer_references();
      result.current_allocated_size_after_release =
          device_.currentAllocatedSize;
      result.reason_code = "metal-resource-allocation-failed";
      result.error = internal_error("newBufferWithLength returned nil");
      allocation_result_ = result;
      return result;
    }

    buffer_size_bytes_ = request.buffer_size_bytes;
    reset_all_status_words();
    result.current_allocated_size_peak = device_.currentAllocatedSize;
    result.buffer_a =
        resource_metadata(buffer_a_, "membenchmark.gpu.buffer-a");
    result.buffer_b =
        resource_metadata(buffer_b_, "membenchmark.gpu.buffer-b");
    result.status_buffer =
        resource_metadata(status_buffer_, "membenchmark.gpu.status");
    result.status = GpuBackendStatus::Success;
    result.reason_code = "success";
    allocation_result_ = result;
    return result;
  }

  bool validate_attempt_request(const GpuBackendAttemptRequest& request,
                                std::string& reason_code) const {
    if (buffer_a_ == nil || buffer_b_ == nil || status_buffer_ == nil) {
      reason_code = "resources-not-allocated";
      return false;
    }
    if (request.buffer_size_bytes == 0 ||
        request.buffer_size_bytes != buffer_size_bytes_) {
      reason_code = "attempt-buffer-size-mismatch";
      return false;
    }
    if (request.passes == 0) {
      reason_code = "attempt-pass-count-zero";
      return false;
    }
    if (operation_tag(request.operation) == 0) {
      reason_code = "attempt-operation-invalid";
      return false;
    }
    if (request.passes > Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT) {
      reason_code = "attempt-dispatch-cap-exceeded";
      return false;
    }
    const size_t payload_multiplier =
        request.operation == GpuOperation::Copy
            ? static_cast<size_t>(Constants::COPY_OPERATION_MULTIPLIER)
            : 1U;
    size_t bytes_per_pass = 0;
    size_t exact_payload_bytes = 0;
    if (!NumericUtils::checked_multiply(request.buffer_size_bytes,
                                        payload_multiplier,
                                        bytes_per_pass) ||
        !NumericUtils::checked_multiply(bytes_per_pass, request.passes,
                                        exact_payload_bytes)) {
      reason_code = "attempt-payload-overflow";
      return false;
    }
    if (exact_payload_bytes > Constants::GPU_MAX_EXACT_PAYLOAD_BYTES) {
      reason_code = "attempt-payload-cap-exceeded";
      return false;
    }
    if (request.vector_count !=
            request.buffer_size_bytes / Constants::GPU_VECTOR_WIDTH_BYTES ||
        request.tail_bytes !=
            request.buffer_size_bytes % Constants::GPU_VECTOR_WIDTH_BYTES) {
      reason_code = "attempt-vector-tail-mismatch";
      return false;
    }
    if (request.threads_per_threadgroup == 0 ||
        request.threadgroups_per_grid == 0 || request.grid_threads == 0) {
      reason_code = "attempt-grid-zero";
      return false;
    }
    size_t expected_grid_threads = 0;
    if (!NumericUtils::checked_multiply(request.threads_per_threadgroup,
                                        request.threadgroups_per_grid,
                                        expected_grid_threads) ||
        expected_grid_threads != request.grid_threads) {
      reason_code = "attempt-grid-size-mismatch";
      return false;
    }
    if (!pipeline_supports_geometry(pipeline_for(request.operation),
                                    request.threads_per_threadgroup) ||
        !pipeline_supports_geometry(initialization_pipeline_,
                                    request.threads_per_threadgroup) ||
        !pipeline_supports_geometry(validation_pipeline_,
                                    request.threads_per_threadgroup)) {
      reason_code = "attempt-threadgroup-geometry-invalid";
      return false;
    }
    return true;
  }

  bool pipeline_supports_geometry(id<MTLComputePipelineState> pipeline,
                                  size_t threads_per_threadgroup) const {
    if (pipeline == nil || pipeline.threadExecutionWidth == 0) {
      return false;
    }
    const size_t execution_width =
        static_cast<size_t>(pipeline.threadExecutionWidth);
    return threads_per_threadgroup <=
               static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup) &&
           threads_per_threadgroup % execution_width == 0 &&
           threads_per_threadgroup / execution_width <=
               GpuKernelContract::kMaxSimdgroupsPerThreadgroup;
  }

  id<MTLComputePipelineState> pipeline_for(GpuOperation operation) const {
    switch (operation) {
      case GpuOperation::Read:
        return read_pipeline_;
      case GpuOperation::Write:
        return write_pipeline_;
      case GpuOperation::Copy:
        return copy_pipeline_;
    }
    return nil;
  }

  void dispatch(id<MTLComputeCommandEncoder> encoder,
                const GpuBackendAttemptRequest& request) {
    [encoder
        dispatchThreadgroups:MTLSizeMake(request.threadgroups_per_grid, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(request.threads_per_threadgroup, 1,
                                         1)];
  }

  void encode_fill(id<MTLComputeCommandEncoder> encoder,
                   id<MTLBuffer> destination,
                   const GpuBackendAttemptRequest& request,
                   uint32_t pattern_tag, uint32_t pattern_pass) {
    const KernelParams params =
        make_params(request, pattern_tag, pattern_pass, 0, 0);
    [encoder setComputePipelineState:initialization_pipeline_];
    [encoder setBuffer:destination offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    dispatch(encoder, request);
  }

  size_t encode_precondition(id<MTLComputeCommandEncoder> encoder,
                             const GpuBackendAttemptRequest& request) {
    switch (request.operation) {
      case GpuOperation::Read:
        encode_fill(encoder, buffer_a_, request,
                    GpuKernelContract::kReadSourcePatternTag, 0);
        return 1;
      case GpuOperation::Write:
        encode_fill(encoder, buffer_a_, request,
                    GpuKernelContract::kPoisonPatternTag, 0);
        return 1;
      case GpuOperation::Copy:
        encode_fill(encoder, buffer_a_, request,
                    GpuKernelContract::kCopySourcePatternTag, 0);
        encode_fill(encoder, buffer_b_, request,
                    GpuKernelContract::kPoisonPatternTag, 0);
        return 2;
    }
    return 0;
  }

  void encode_operation_pass(id<MTLComputeCommandEncoder> encoder,
                             const GpuBackendAttemptRequest& request,
                             size_t pass) {
    const uint32_t direction =
        request.operation == GpuOperation::Copy ? low32(pass & 1U) : 0U;
    const uint32_t pattern_tag =
        request.operation == GpuOperation::Write
            ? GpuKernelContract::kWritePatternTag
            : source_pattern_tag(request.operation);
    const KernelParams params = make_params(
        request, pattern_tag,
        request.operation == GpuOperation::Write ? low32(pass) : 0U,
        low32(pass), direction);
    [encoder setComputePipelineState:pipeline_for(request.operation)];
    switch (request.operation) {
      case GpuOperation::Read:
        [encoder setBuffer:buffer_a_ offset:0 atIndex:0];
        [encoder setBuffer:status_buffer_ offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        break;
      case GpuOperation::Write:
        [encoder setBuffer:buffer_a_ offset:0 atIndex:0];
        [encoder setBuffer:status_buffer_ offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        break;
      case GpuOperation::Copy: {
        id<MTLBuffer> source = (pass & 1U) == 0U ? buffer_a_ : buffer_b_;
        id<MTLBuffer> destination =
            (pass & 1U) == 0U ? buffer_b_ : buffer_a_;
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:destination offset:0 atIndex:1];
        [encoder setBuffer:status_buffer_ offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        break;
      }
    }
    dispatch(encoder, request);
  }

  GpuBackendPhaseResult run_setup_phase(
      const GpuBackendAttemptRequest& request, bool include_warmup_pass) {
    GpuBackendPhaseResult result;
    result.status = GpuBackendStatus::Failed;
    std::string validation_reason;
    if (!validate_attempt_request(request, validation_reason)) {
      result.reason_code = validation_reason;
      return result;
    }

    reset_all_status_words();
    last_output_ = nil;
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      result.reason_code = "setup-command-buffer-creation-failed";
      return result;
    }
    command_buffer.label = include_warmup_pass
                               ? @"membenchmark.gpu.command.warmup"
                               : @"membenchmark.gpu.command.precondition";
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder == nil) {
      result.reason_code = "setup-encoder-creation-failed";
      return result;
    }
    encoder.label = include_warmup_pass
                        ? @"membenchmark.gpu.encoder.warmup"
                        : @"membenchmark.gpu.encoder.precondition";
    result.compute_encoder_count = 1;
    result.data_initialization_dispatch_count =
        encode_precondition(encoder, request);
    result.dispatch_count = result.data_initialization_dispatch_count;
    result.status_reset_count = 1;
    if (include_warmup_pass) {
      encode_operation_pass(encoder, request, 0);
      ++result.dispatch_count;
      result.benchmark_operation_dispatch_count = 1;
    }
    [encoder endEncoding];
    return commit_and_wait(command_buffer, result,
                           include_warmup_pass ? "warmup" : "precondition");
  }

  template <typename Result>
  Result commit_and_wait(id<MTLCommandBuffer> command_buffer, Result result,
                         const char* phase) {
    result.command_buffer_count = 1;
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      result.status = GpuBackendStatus::Failed;
      result.command_status = GpuCommandStatus::Error;
      result.reason_code = std::string(phase) + "-command-buffer-error";
      result.error = error_diagnostic(command_buffer.error);
      return result;
    }
    result.status = GpuBackendStatus::Success;
    result.command_status = GpuCommandStatus::Completed;
    result.reason_code = "success";
    return result;
  }

  GpuTimedResult run_timed_impl(const GpuBackendAttemptRequest& request) {
    GpuTimedResult result;
    result.status = GpuBackendStatus::Failed;
    std::string validation_reason;
    if (!validate_attempt_request(request, validation_reason)) {
      result.reason_code = validation_reason;
      return result;
    }

    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      result.reason_code = "timed-command-buffer-creation-failed";
      return result;
    }
    command_buffer.label = @"membenchmark.gpu.command.timed";
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder == nil) {
      result.reason_code = "timed-encoder-creation-failed";
      return result;
    }
    encoder.label = @"membenchmark.gpu.encoder.timed.serial";
    result.compute_encoder_count = 1;
    result.dispatch_count = request.passes;
    result.benchmark_operation_dispatch_count = request.passes;
    for (size_t pass = 0; pass < request.passes; ++pass) {
      encode_operation_pass(encoder, request, pass);
    }
    [encoder endEncoding];

    result.command_buffer_count = 1;
    result.host_submit_seconds = steady_seconds();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    result.host_wait_end_seconds = steady_seconds();
    result.host_wall_seconds =
        result.host_wait_end_seconds - result.host_submit_seconds;
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      result.status = GpuBackendStatus::Failed;
      result.command_status = GpuCommandStatus::Error;
      result.reason_code = "timed-command-buffer-error";
      result.error = error_diagnostic(command_buffer.error);
      return result;
    }

    result.status = GpuBackendStatus::Success;
    result.command_status = GpuCommandStatus::Completed;
    result.reason_code = "success";
    result.gpu_start_seconds = command_buffer.GPUStartTime;
    result.gpu_end_seconds = command_buffer.GPUEndTime;
    result.gpu_elapsed_seconds =
        result.gpu_end_seconds - result.gpu_start_seconds;
    result.expected_accumulator = expected_timed_accumulator(request);
    const uint32_t* words = status_words();
    result.actual_accumulator = {words[0], words[1]};
    switch (request.operation) {
      case GpuOperation::Read:
      case GpuOperation::Write:
        last_output_ = buffer_a_;
        break;
      case GpuOperation::Copy:
        last_output_ = (request.passes & 1U) == 0U ? buffer_a_ : buffer_b_;
        break;
    }
    last_output_size_ = request.buffer_size_bytes;
    return result;
  }

  GpuValidationResult run_validation_impl(
      const GpuBackendAttemptRequest& request,
      const GpuTimedResult& timed_result) {
    GpuValidationResult result;
    result.status = GpuBackendStatus::Failed;
    if (request.operation != GpuOperation::Read) {
      result.final_checksum_algorithm = "gpu-dual-mod32-v1";
    }
    std::string validation_reason;
    if (!validate_attempt_request(request, validation_reason)) {
      result.reason_code = validation_reason;
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }
    if (timed_result.command_status != GpuCommandStatus::Completed) {
      result.reason_code = "timed-command-not-completed";
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }
    if (!std::isfinite(timed_result.gpu_start_seconds) ||
        !std::isfinite(timed_result.gpu_end_seconds) ||
        !std::isfinite(timed_result.gpu_elapsed_seconds) ||
        timed_result.gpu_start_seconds <= 0.0 ||
        timed_result.gpu_end_seconds <= timed_result.gpu_start_seconds ||
        timed_result.gpu_elapsed_seconds <= 0.0) {
      result.status = GpuBackendStatus::Success;
      result.reason_code = "gpu-timestamp-invalid";
      result.validation_status = GpuValidationStatus::NotRunTimerInvalid;
      return result;
    }
    if (timed_result.expected_accumulator !=
        timed_result.actual_accumulator) {
      result.status = GpuBackendStatus::Success;
      result.reason_code = "timed-accumulator-mismatch";
      result.validation_status = GpuValidationStatus::Mismatch;
      return result;
    }
    if (request.operation == GpuOperation::Read) {
      result.status = GpuBackendStatus::Success;
      result.reason_code = "success";
      result.validation_status = GpuValidationStatus::Passed;
      return result;
    }
    if (last_output_ == nil || last_output_size_ != request.buffer_size_bytes) {
      result.reason_code = "validation-output-unavailable";
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }

    uint32_t* words = status_words();
    words[2] = 0;
    words[3] = 0;
    result.status_reset_count = 1;
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      result.reason_code = "validation-command-buffer-creation-failed";
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }
    command_buffer.label = @"membenchmark.gpu.command.validation";
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder == nil) {
      result.reason_code = "validation-encoder-creation-failed";
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }
    encoder.label = @"membenchmark.gpu.encoder.validation";
    const KernelParams params = make_params(request, 0, 0, 0, 0);
    [encoder setComputePipelineState:validation_pipeline_];
    [encoder setBuffer:last_output_ offset:0 atIndex:0];
    [encoder setBuffer:status_buffer_ offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    dispatch(encoder, request);
    [encoder endEncoding];
    result.compute_encoder_count = 1;
    result.dispatch_count = 1;
    result.validation_dispatch_count = 1;
    result = commit_and_wait(command_buffer, result, "validation");
    if (result.status != GpuBackendStatus::Success) {
      result.validation_status = GpuValidationStatus::Error;
      return result;
    }

    result.expected_final_checksum = expected_final_checksum(request);
    result.actual_final_checksum = {words[2], words[3]};
    if (result.expected_final_checksum != result.actual_final_checksum) {
      result.reason_code = "final-checksum-mismatch";
      result.validation_status = GpuValidationStatus::Mismatch;
      return result;
    }
    result.validation_status = GpuValidationStatus::Passed;
    return result;
  }

  GpuBackendPhaseResult readback_last_output_impl(
      std::vector<uint8_t>& output) {
    GpuBackendPhaseResult result;
    result.status = GpuBackendStatus::Failed;
    output.clear();
    if (last_output_ == nil || last_output_size_ == 0) {
      result.reason_code = "readback-output-unavailable";
      return result;
    }
    const MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    id<MTLBuffer> staging =
        [device_ newBufferWithLength:last_output_size_ options:options];
    if (staging == nil || staging.contents == nullptr) {
      result.reason_code = "readback-staging-allocation-failed";
      return result;
    }
    staging.label = @"membenchmark.gpu.readback-staging";
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      result.reason_code = "readback-command-buffer-creation-failed";
      return result;
    }
    command_buffer.label = @"membenchmark.gpu.command.readback";
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    if (blit == nil) {
      result.reason_code = "readback-blit-encoder-creation-failed";
      return result;
    }
    blit.label = @"membenchmark.gpu.encoder.readback";
    [blit copyFromBuffer:last_output_
            sourceOffset:0
                toBuffer:staging
       destinationOffset:0
                    size:last_output_size_];
    [blit endEncoding];
    result = commit_and_wait(command_buffer, result, "readback");
    if (result.status != GpuBackendStatus::Success) {
      return result;
    }
    const uint8_t* bytes =
        static_cast<const uint8_t*>(staging.contents);
    output.assign(bytes, bytes + last_output_size_);
    return result;
  }

  GpuAllocationResult release_resources_impl() {
    GpuAllocationResult result = allocation_result_;
    const bool had_resources =
        buffer_a_ != nil || buffer_b_ != nil || status_buffer_ != nil;
    release_buffer_references();
    result.current_allocated_size_after_release =
        device_ != nil ? device_.currentAllocatedSize : 0;
    result.status = GpuBackendStatus::Success;
    result.reason_code = had_resources ? "resources-released"
                                       : "resources-already-released";
    allocation_result_ = result;
    return result;
  }

  void release_buffer_references() {
    last_output_ = nil;
    last_output_size_ = 0;
    buffer_a_ = nil;
    buffer_b_ = nil;
    status_buffer_ = nil;
    buffer_size_bytes_ = 0;
  }

  uint32_t* status_words() const {
    return static_cast<uint32_t*>(status_buffer_.contents);
  }

  void reset_all_status_words() {
    std::memset(status_buffer_.contents, 0,
                static_cast<size_t>(status_buffer_.length));
  }

  bool initialized_ = false;
  GpuBackendInitialization initialization_result_;
  GpuAllocationResult allocation_result_;
  size_t buffer_size_bytes_ = 0;
  size_t last_output_size_ = 0;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> initialization_pipeline_ = nil;
  id<MTLComputePipelineState> read_pipeline_ = nil;
  id<MTLComputePipelineState> write_pipeline_ = nil;
  id<MTLComputePipelineState> copy_pipeline_ = nil;
  id<MTLComputePipelineState> validation_pipeline_ = nil;
  id<MTLBuffer> buffer_a_ = nil;
  id<MTLBuffer> buffer_b_ = nil;
  id<MTLBuffer> status_buffer_ = nil;
  id<MTLBuffer> last_output_ = nil;
};

}  // namespace

std::unique_ptr<GpuBackend> create_metal_gpu_backend() {
  return std::make_unique<MetalGpuBackend>();
}

std::string canonical_gpu_kernel_source_sha256() {
  return HashUtils::sha256_hex(GpuKernelContract::kSource);
}

GpuDualChecksum calculate_expected_gpu_timed_accumulator(
    const GpuBackendAttemptRequest& request) {
  return expected_timed_accumulator(request);
}
