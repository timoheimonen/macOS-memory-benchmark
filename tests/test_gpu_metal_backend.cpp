// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file test_gpu_metal_backend.cpp
 * @brief Real-Metal integration coverage for the private GPU backend
 */

#include <gtest/gtest.h>

#include "core/config/constants.h"
#include "gpu_bandwidth/gpu_backend.h"
#include "gpu_bandwidth/gpu_runner.h"
#include "utils/cyclic_order.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

// Independent CPU readback oracle for gpu-linear-word-mod32-v2. These literals
// intentionally do not include the private kernel-source header.
constexpr uint32_t kPatternSeedHighMultiplier = 0x9e3779b9U;
constexpr uint32_t kPatternIndexMultiplier = 0x85ebca6bU;
constexpr uint32_t kPatternTagMultiplier = 0xc2b2ae35U;
constexpr uint32_t kPatternPassMultiplier = 0x27d4eb2fU;
constexpr uint32_t kReadSourcePatternTag = 0x11U;
constexpr uint32_t kCopySourcePatternTag = 0x22U;
constexpr uint32_t kWritePatternTag = 0x44U;
constexpr double kGridSelectionRelativeFloor = 0.98;

bool grid_candidate_passes_selection_rule(double candidate_median,
                                          double best_median) {
  return std::isfinite(candidate_median) &&
         std::isfinite(best_median) && candidate_median > 0.0 &&
         best_median > 0.0 &&
         candidate_median >= best_median * kGridSelectionRelativeFloor;
}

uint32_t multiply_mod32(uint32_t left, uint32_t right) {
  return static_cast<uint32_t>(static_cast<uint64_t>(left) * right);
}

uint32_t oracle_pattern_word(uint64_t word_index, uint64_t seed,
                             uint32_t pattern_tag,
                             uint32_t pattern_pass) {
  return static_cast<uint32_t>(seed) +
         multiply_mod32(static_cast<uint32_t>(seed >> 32U),
                        kPatternSeedHighMultiplier) +
         multiply_mod32(static_cast<uint32_t>(word_index),
                        kPatternIndexMultiplier) +
         multiply_mod32(pattern_tag, kPatternTagMultiplier) +
         multiply_mod32(pattern_pass, kPatternPassMultiplier);
}

std::vector<uint8_t> expected_output_bytes(
    const GpuBackendAttemptRequest& request) {
  uint32_t pattern_tag = kReadSourcePatternTag;
  uint32_t pattern_pass = 0;
  if (request.operation == GpuOperation::Write) {
    pattern_tag = kWritePatternTag;
    pattern_pass = static_cast<uint32_t>(request.passes - 1U);
  } else if (request.operation == GpuOperation::Copy) {
    pattern_tag = kCopySourcePatternTag;
  }

  std::vector<uint8_t> expected(request.buffer_size_bytes);
  for (size_t byte_index = 0; byte_index < expected.size(); ++byte_index) {
    const uint32_t value = oracle_pattern_word(
        byte_index / sizeof(uint32_t), request.operation_seed, pattern_tag,
        pattern_pass);
    expected[byte_index] = static_cast<uint8_t>(
        value >> ((byte_index % sizeof(uint32_t)) * 8U));
  }
  return expected;
}

const GpuPipelineMetadata& pipeline_for(
    const GpuBackendInitialization& initialization,
    GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return initialization.device.read_pipeline;
    case GpuOperation::Write:
      return initialization.device.write_pipeline;
    case GpuOperation::Copy:
      return initialization.device.copy_pipeline;
  }
  return initialization.device.read_pipeline;
}

GpuBackendAttemptRequest build_attempt_request(
    const GpuBackendInitialization& initialization, GpuOperation operation,
    size_t buffer_size_bytes, size_t passes, uint64_t operation_seed) {
  const GpuPipelineMetadata& pipeline =
      pipeline_for(initialization, operation);
  const size_t thread_cap = std::min(
      Constants::GPU_THREADS_PER_THREADGROUP_CAP,
      pipeline.max_total_threads_per_threadgroup);
  const size_t threads_per_threadgroup =
      (thread_cap / pipeline.thread_execution_width) *
      pipeline.thread_execution_width;
  const size_t vector_count =
      buffer_size_bytes / Constants::GPU_VECTOR_WIDTH_BYTES;
  const size_t required_threadgroups =
      vector_count / threads_per_threadgroup +
      (vector_count % threads_per_threadgroup != 0 ? 1U : 0U);
  const size_t threadgroups = std::max<size_t>(
      1, std::min(required_threadgroups,
                  Constants::GPU_MAX_THREADGROUPS_PER_GRID));

  GpuBackendAttemptRequest request;
  request.operation = operation;
  request.buffer_size_bytes = buffer_size_bytes;
  request.passes = passes;
  request.operation_seed = operation_seed;
  request.vector_count = vector_count;
  request.tail_bytes =
      buffer_size_bytes % Constants::GPU_VECTOR_WIDTH_BYTES;
  request.threads_per_threadgroup = threads_per_threadgroup;
  request.threadgroups_per_grid = threadgroups;
  request.grid_threads = threadgroups * threads_per_threadgroup;
  return request;
}

class GpuMetalBackendIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = create_metal_gpu_backend();
    ASSERT_NE(backend_, nullptr);
    initialization_ = backend_->initialize();
    if (initialization_.status == GpuBackendStatus::Unsupported) {
      GTEST_SKIP() << "Metal backend unsupported: "
                   << initialization_.reason_code;
    }
    ASSERT_EQ(initialization_.status, GpuBackendStatus::Success)
        << initialization_.reason_code << ": "
        << initialization_.error.description;
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      backend_->release_resources();
    }
  }

  void allocate(size_t buffer_size_bytes) {
    backend_->release_resources();
    GpuAllocationRequest request;
    request.buffer_size_bytes = buffer_size_bytes;
    request.auxiliary_bytes = Constants::GPU_AUXILIARY_BUFFER_BYTES;
    request.memory_budget_bytes = 128U * Constants::BYTES_PER_MB;
    const GpuAllocationResult result = backend_->allocate_resources(request);
    ASSERT_EQ(result.status, GpuBackendStatus::Success)
        << result.reason_code << ": " << result.error.description;
    ASSERT_EQ(result.requested_buffer_size_bytes, buffer_size_bytes);
    ASSERT_EQ(result.auxiliary_bytes,
              Constants::GPU_AUXILIARY_BUFFER_BYTES);
    ASSERT_EQ(result.required_total_bytes,
              buffer_size_bytes * 2U +
                  Constants::GPU_AUXILIARY_BUFFER_BYTES);
  }

  void execute_and_verify(GpuOperation operation, size_t buffer_size_bytes,
                          size_t passes, uint64_t operation_seed) {
    allocate(buffer_size_bytes);
    const GpuBackendAttemptRequest request = build_attempt_request(
        initialization_, operation, buffer_size_bytes, passes,
        operation_seed);

    const GpuBackendPhaseResult warmup = backend_->run_warmup(request);
    ASSERT_EQ(warmup.status, GpuBackendStatus::Success)
        << warmup.reason_code;
    EXPECT_EQ(warmup.command_status, GpuCommandStatus::Completed);
    EXPECT_EQ(warmup.command_buffer_count, 1U);
    EXPECT_EQ(warmup.compute_encoder_count, 1U);
    const size_t precondition_dispatches =
        operation == GpuOperation::Copy ? 2U : 1U;
    EXPECT_EQ(warmup.dispatch_count, precondition_dispatches + 1U);
    EXPECT_EQ(warmup.data_initialization_dispatch_count,
              precondition_dispatches);
    EXPECT_EQ(warmup.benchmark_operation_dispatch_count, 1U);
    EXPECT_EQ(warmup.validation_dispatch_count, 0U);
    EXPECT_EQ(warmup.status_reset_count, 1U);

    const GpuBackendPhaseResult precondition =
        backend_->run_precondition(request);
    ASSERT_EQ(precondition.status, GpuBackendStatus::Success)
        << precondition.reason_code;
    EXPECT_EQ(precondition.command_status, GpuCommandStatus::Completed);
    EXPECT_EQ(precondition.command_buffer_count, 1U);
    EXPECT_EQ(precondition.compute_encoder_count, 1U);
    EXPECT_EQ(precondition.dispatch_count, precondition_dispatches);
    EXPECT_EQ(precondition.data_initialization_dispatch_count,
              precondition_dispatches);
    EXPECT_EQ(precondition.benchmark_operation_dispatch_count, 0U);
    EXPECT_EQ(precondition.status_reset_count, 1U);

    const GpuTimedResult timed = backend_->run_timed(request);
    ASSERT_EQ(timed.status, GpuBackendStatus::Success)
        << timed.reason_code << ": " << timed.error.description;
    EXPECT_EQ(timed.command_status, GpuCommandStatus::Completed);
    EXPECT_EQ(timed.command_buffer_count, 1U);
    EXPECT_EQ(timed.compute_encoder_count, 1U);
    EXPECT_EQ(timed.dispatch_count, passes);
    EXPECT_EQ(timed.benchmark_operation_dispatch_count, passes);
    EXPECT_EQ(timed.data_initialization_dispatch_count, 0U);
    EXPECT_GT(timed.gpu_start_seconds, 0.0);
    EXPECT_GT(timed.gpu_end_seconds, timed.gpu_start_seconds);
    EXPECT_GT(timed.gpu_elapsed_seconds, 0.0);
    EXPECT_GT(timed.host_wall_seconds, 0.0);
    EXPECT_EQ(timed.actual_accumulator, timed.expected_accumulator);

    const GpuValidationResult validation =
        backend_->run_validation(request, timed);
    ASSERT_EQ(validation.status, GpuBackendStatus::Success)
        << validation.reason_code << ": "
        << validation.error.description;
    EXPECT_EQ(validation.validation_status, GpuValidationStatus::Passed)
        << validation.reason_code;
    EXPECT_EQ(validation.timed_accumulator_algorithm,
              "gpu-dual-mod32-v2");
    if (operation == GpuOperation::Read) {
      EXPECT_EQ(validation.final_checksum_algorithm, "not-applicable");
      EXPECT_EQ(validation.command_buffer_count, 0U);
      EXPECT_EQ(validation.compute_encoder_count, 0U);
      EXPECT_EQ(validation.dispatch_count, 0U);
      EXPECT_EQ(validation.validation_dispatch_count, 0U);
      EXPECT_EQ(validation.status_reset_count, 0U);
    } else {
      EXPECT_EQ(validation.final_checksum_algorithm,
                "gpu-dual-mod32-v1");
      EXPECT_EQ(validation.command_status, GpuCommandStatus::Completed);
      EXPECT_EQ(validation.command_buffer_count, 1U);
      EXPECT_EQ(validation.compute_encoder_count, 1U);
      EXPECT_EQ(validation.dispatch_count, 1U);
      EXPECT_EQ(validation.validation_dispatch_count, 1U);
      EXPECT_EQ(validation.status_reset_count, 1U);
      EXPECT_EQ(validation.actual_final_checksum,
                validation.expected_final_checksum);
    }

    std::vector<uint8_t> output;
    const GpuBackendPhaseResult readback =
        backend_->readback_last_output(output);
    ASSERT_EQ(readback.status, GpuBackendStatus::Success)
        << readback.reason_code;
    EXPECT_EQ(readback.command_status, GpuCommandStatus::Completed);
    EXPECT_EQ(readback.command_buffer_count, 1U);
    EXPECT_EQ(readback.compute_encoder_count, 0U);
    EXPECT_EQ(readback.dispatch_count, 0U);
    EXPECT_EQ(output, expected_output_bytes(request));
  }

  std::unique_ptr<GpuBackend> backend_;
  GpuBackendInitialization initialization_;
};

/**
 * Opt-in release-reference selection gate for the frozen 8192-group grid cap.
 * Run only on an Apple M4-family GPU with:
 * `MEMBENCHMARK_RUN_GPU_GRID_VALIDATION=1 ./test_runner
 * --gtest_filter=GpuMetalGridSelectionIntegrationTest.*`
 */
class GpuMetalGridSelectionIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* enabled =
        std::getenv("MEMBENCHMARK_RUN_GPU_GRID_VALIDATION");
    if (enabled == nullptr || std::string(enabled) != "1") {
      GTEST_SKIP()
          << "set MEMBENCHMARK_RUN_GPU_GRID_VALIDATION=1 to run the "
             "count-9 M4 grid selection gate";
    }

    backend_ = create_metal_gpu_backend();
    ASSERT_NE(backend_, nullptr);
    initialization_ = backend_->initialize();
    if (initialization_.status == GpuBackendStatus::Unsupported) {
      GTEST_SKIP() << "Metal backend unsupported: "
                   << initialization_.reason_code;
    }
    ASSERT_EQ(initialization_.status, GpuBackendStatus::Success)
        << initialization_.reason_code << ": "
        << initialization_.error.description;
    if (initialization_.device.device_name.rfind("Apple M4", 0) != 0) {
      GTEST_SKIP() << "grid selection gate requires an Apple M4-family GPU; "
                   << "found " << initialization_.device.device_name;
    }
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      backend_->release_resources();
    }
  }

  std::unique_ptr<GpuBackend> backend_;
  GpuBackendInitialization initialization_;
};

TEST(GpuGridSelectionTest, Production8192PassesFrozenM4WriteGate) {
  constexpr std::array<double, 3> kWriteMedians = {
      77.879509, 78.703126, 82.914226};
  const double best_median =
      *std::max_element(kWriteMedians.begin(), kWriteMedians.end());

  static_assert(Constants::GPU_MAX_THREADGROUPS_PER_GRID == 8192,
                "M4 gate selected the 8192-group production cap");
  EXPECT_FALSE(
      grid_candidate_passes_selection_rule(kWriteMedians[1], best_median));
  EXPECT_TRUE(
      grid_candidate_passes_selection_rule(kWriteMedians[2], best_median));
}

TEST_F(GpuMetalBackendIntegrationTest,
       InitializationAndRuntimeCompilationIntegration) {
  const GpuDeviceMetadata& device = initialization_.device;
  EXPECT_FALSE(device.device_name.empty());
  EXPECT_TRUE(device.has_unified_memory);
  EXPECT_TRUE(device.required_apple7_family_supported);
  EXPECT_NE(std::find(device.supported_families.begin(),
                      device.supported_families.end(), "apple7"),
            device.supported_families.end());
  EXPECT_GT(device.max_buffer_length, 0U);
  EXPECT_GT(device.recommended_max_working_set_size, 0U);
  EXPECT_FALSE(device.macos_product_version.empty());
  EXPECT_FALSE(device.macos_build.empty());
  EXPECT_FALSE(device.hardware_model.empty());
  EXPECT_GT(device.physical_memory_bytes, 0U);
  EXPECT_GT(device.available_memory_bytes, 0U);
  EXPECT_EQ(device.available_memory_source,
            "project-mach-free-plus-inactive-mib");

  EXPECT_EQ(initialization_.compilation.compilation_mode,
            "runtime-source");
  EXPECT_EQ(initialization_.compilation.msl_language_version, "2.3");
  EXPECT_EQ(initialization_.compilation.floating_point_math,
            "not_applicable_integer_only");
  EXPECT_EQ(initialization_.compilation.preprocessor_macros, "none");
  EXPECT_EQ(initialization_.compilation.kernel_revision,
            "gpu-linear-word-mod32-tg-reduce-v2");
  EXPECT_EQ(initialization_.compilation.kernel_source_sha256,
            "b9a242d2b959c9c11f6f130a52afd66f111d6761be2193beec1f051baa094296");
  for (char character :
       initialization_.compilation.kernel_source_sha256) {
    EXPECT_TRUE((character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f'));
  }
  // Warning text is retained rather than treated as runtime-compile failure.
  // The release-reference M4 cohort additionally requires warning-free source.
  if (device.device_name.find("M4") != std::string::npos) {
    EXPECT_TRUE(initialization_.compilation.compiler_diagnostics.empty());
  }
  EXPECT_FALSE(initialization_.compilation.compiler_identifier.empty());
  EXPECT_FALSE(initialization_.compilation.build_sdk.empty());
  EXPECT_EQ(initialization_.compilation.deployment_target, "11.0");

  const std::array<const GpuPipelineMetadata*, 5> pipelines = {
      &device.initialization_pipeline, &device.read_pipeline,
      &device.write_pipeline, &device.copy_pipeline,
      &device.validation_pipeline};
  for (const GpuPipelineMetadata* pipeline : pipelines) {
    EXPECT_FALSE(pipeline->label.empty());
    EXPECT_GT(pipeline->thread_execution_width, 0U);
    EXPECT_GT(pipeline->max_total_threads_per_threadgroup, 0U);
  }
}

TEST_F(GpuMetalBackendIntegrationTest,
       PrivateTrackedAndSharedTrackedResourcesIntegration) {
  constexpr size_t kBufferBytes = 4097;
  allocate(kBufferBytes);
  GpuAllocationRequest request;
  request.buffer_size_bytes = kBufferBytes;
  request.auxiliary_bytes = Constants::GPU_AUXILIARY_BUFFER_BYTES;
  request.memory_budget_bytes = 64U * Constants::BYTES_PER_MB;

  // Re-allocation also verifies deterministic replacement of suite resources.
  const GpuAllocationResult allocation =
      backend_->allocate_resources(request);
  ASSERT_EQ(allocation.status, GpuBackendStatus::Success)
      << allocation.reason_code;
  for (const GpuResourceMetadata* resource :
       {&allocation.buffer_a, &allocation.buffer_b}) {
    EXPECT_EQ(resource->storage_mode, "private");
    EXPECT_EQ(resource->hazard_tracking_mode, "tracked");
    EXPECT_EQ(resource->length_bytes, kBufferBytes);
    EXPECT_NE(resource->resource_options, 0U);
  }
  EXPECT_EQ(allocation.status_buffer.storage_mode, "shared");
  EXPECT_EQ(allocation.status_buffer.hazard_tracking_mode, "tracked");
  EXPECT_EQ(allocation.status_buffer.length_bytes,
            Constants::GPU_AUXILIARY_BUFFER_BYTES);
  EXPECT_NE(allocation.status_buffer.resource_options, 0U);

  const GpuEnvironmentSnapshot snapshot =
      backend_->snapshot_environment();
  EXPECT_NE(snapshot.thermal_state, "unavailable");
  EXPECT_GT(snapshot.current_allocated_size, 0U);

  const GpuAllocationResult released = backend_->release_resources();
  EXPECT_EQ(released.status, GpuBackendStatus::Success);
  EXPECT_EQ(released.reason_code, "resources-released");
  std::vector<uint8_t> unavailable_output;
  const GpuBackendPhaseResult readback_after_release =
      backend_->readback_last_output(unavailable_output);
  EXPECT_EQ(readback_after_release.status, GpuBackendStatus::Failed);
  EXPECT_EQ(readback_after_release.reason_code,
            "readback-output-unavailable");
  EXPECT_TRUE(unavailable_output.empty());
}

TEST_F(GpuMetalBackendIntegrationTest,
       ReadWriteCopyMultiPassAndReadbackIntegration) {
  // More than one 256-thread group verifies the threadgroup-leader global
  // reduction path, while the final byte keeps the tail path active.
  constexpr size_t kBufferBytes = 65553;
  execute_and_verify(GpuOperation::Read, kBufferBytes, 3,
                     0x0123456789abcdefULL);
  execute_and_verify(GpuOperation::Write, kBufferBytes, 4,
                     0xfedcba9876543210ULL);
  execute_and_verify(GpuOperation::Copy, kBufferBytes, 3,
                     0x0f1e2d3c4b5a6978ULL);
  execute_and_verify(GpuOperation::Copy, kBufferBytes, 4,
                     0x0f1e2d3c4b5a6978ULL);
}

TEST_F(GpuMetalBackendIntegrationTest,
       AccumulatorV2PowerOfTwoPassCountsIntegration) {
  constexpr size_t kBufferBytes = 4096;
  constexpr std::array<GpuOperation, 3> kOperations = {
      GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};
  uint64_t seed = 6102026ULL;
  for (GpuOperation operation : kOperations) {
    execute_and_verify(operation, kBufferBytes, 24, seed++);
    execute_and_verify(operation, kBufferBytes, 64, seed++);
  }
}

TEST_F(GpuMetalBackendIntegrationTest,
       CappedGridStrideCoversEveryVectorIntegration) {
  const size_t vector_count =
      Constants::GPU_MAX_THREADGROUPS_PER_GRID *
          Constants::GPU_THREADS_PER_THREADGROUP_CAP +
      17U;
  const size_t buffer_size =
      vector_count * Constants::GPU_VECTOR_WIDTH_BYTES + 15U;
  const GpuBackendAttemptRequest request = build_attempt_request(
      initialization_, GpuOperation::Read, buffer_size, 2,
      0x5566778899aabbccULL);
  ASSERT_EQ(request.threadgroups_per_grid,
            Constants::GPU_MAX_THREADGROUPS_PER_GRID);
  ASSERT_GT(request.vector_count, request.grid_threads);
  execute_and_verify(GpuOperation::Read, buffer_size, 2,
                     request.operation_seed);
}

TEST_F(GpuMetalBackendIntegrationTest,
       ExactTailReadWriteCopyIntegration) {
  constexpr std::array<size_t, 4> kTailSizes = {1, 15, 17, 31};
  constexpr std::array<GpuOperation, 3> kOperations = {
      GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};
  uint64_t seed = 0x1020304050607080ULL;
  for (size_t size : kTailSizes) {
    for (GpuOperation operation : kOperations) {
      SCOPED_TRACE(::testing::Message()
                   << "size=" << size
                   << " operation=" << static_cast<int>(operation));
      execute_and_verify(operation, size, 3, seed++);
    }
  }
}

TEST_F(GpuMetalGridSelectionIntegrationTest,
       FixedWorkCountNineCandidateMediansIntegration) {
  static_assert(Constants::GPU_MAX_THREADGROUPS_PER_GRID == 8192,
                "grid selection gate must track the production cap");
  constexpr size_t kBufferSizeBytes =
      512U * Constants::BYTES_PER_MB;
  constexpr size_t kPasses = 24;
  constexpr size_t kSampleCount = 9;
  constexpr uint64_t kBaseSeed = 6102026ULL;
  constexpr std::array<size_t, 3> kCandidateThreadgroups = {
      2048, 4096, Constants::GPU_MAX_THREADGROUPS_PER_GRID};
  constexpr std::array<GpuOperation, kGpuOperationCount> kOperations = {
      GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};

  ASSERT_LE(kBufferSizeBytes, initialization_.device.max_buffer_length);
  GpuAllocationRequest allocation_request;
  allocation_request.buffer_size_bytes = kBufferSizeBytes;
  allocation_request.auxiliary_bytes =
      Constants::GPU_AUXILIARY_BUFFER_BYTES;
  allocation_request.memory_budget_bytes =
      kBufferSizeBytes * 2U + Constants::GPU_AUXILIARY_BUFFER_BYTES;
  const GpuAllocationResult allocation =
      backend_->allocate_resources(allocation_request);
  ASSERT_EQ(allocation.status, GpuBackendStatus::Success)
      << allocation.reason_code << ": " << allocation.error.description;
  const GpuEnvironmentSnapshot environment_start =
      backend_->snapshot_environment();

  std::array<std::array<GpuWorkPlan, kCandidateThreadgroups.size()>,
             kGpuOperationCount>
      plans;
  for (size_t operation_index = 0; operation_index < kOperations.size();
       ++operation_index) {
    const GpuOperation operation = kOperations[operation_index];
    const GpuPipelineMetadata& pipeline =
        pipeline_for(initialization_, operation);
    for (size_t candidate_index = 0;
         candidate_index < kCandidateThreadgroups.size();
         ++candidate_index) {
      GpuWorkPlanRequest request;
      request.operation = operation;
      request.requested_buffer_bytes = kBufferSizeBytes;
      request.passes = kPasses;
      request.base_seed = kBaseSeed;
      request.thread_execution_width = pipeline.thread_execution_width;
      request.max_total_threads_per_threadgroup =
          pipeline.max_total_threads_per_threadgroup;
      request.explicit_iterations = true;
      request.maximum_threadgroups_per_grid =
          kCandidateThreadgroups[candidate_index];
      plans[operation_index][candidate_index] =
          build_gpu_work_plan(request);
      ASSERT_TRUE(plans[operation_index][candidate_index].valid)
          << gpu_operation_to_string(operation) << " candidate="
          << kCandidateThreadgroups[candidate_index] << " reason="
          << plans[operation_index][candidate_index].reason_code;
      ASSERT_EQ(
          plans[operation_index][candidate_index].threadgroups_per_grid,
          kCandidateThreadgroups[candidate_index]);
    }
  }

  std::array<std::array<std::vector<double>,
                        kCandidateThreadgroups.size()>,
             kGpuOperationCount>
      candidate_values;
  for (size_t operation_index = 0; operation_index < kOperations.size();
       ++operation_index) {
    for (size_t sample_index = 0; sample_index < kSampleCount;
         ++sample_index) {
      // Rotate candidate order so every grid appears equally often in every
      // order position instead of conflating a candidate with thermal drift.
      const std::vector<size_t> candidate_order =
          build_cyclic_order(kCandidateThreadgroups.size(), sample_index);
      for (size_t candidate_index : candidate_order) {
        const GpuWorkPlan& plan =
            plans[operation_index][candidate_index];
        const GpuBackendAttemptRequest attempt_request =
            build_gpu_backend_attempt_request(plan);
        SCOPED_TRACE(::testing::Message()
                     << "operation="
                     << gpu_operation_to_string(plan.operation)
                     << " candidate_threadgroups="
                     << kCandidateThreadgroups[candidate_index]
                     << " sample=" << sample_index);

        const GpuBackendPhaseResult warmup =
            backend_->run_warmup(attempt_request);
        ASSERT_EQ(warmup.status, GpuBackendStatus::Success)
            << warmup.reason_code << ": " << warmup.error.description;
        ASSERT_EQ(warmup.command_status, GpuCommandStatus::Completed);

        const GpuBackendPhaseResult precondition =
            backend_->run_precondition(attempt_request);
        ASSERT_EQ(precondition.status, GpuBackendStatus::Success)
            << precondition.reason_code << ": "
            << precondition.error.description;
        ASSERT_EQ(precondition.command_status,
                  GpuCommandStatus::Completed);

        const GpuTimedResult timed =
            backend_->run_timed(attempt_request);
        ASSERT_EQ(timed.status, GpuBackendStatus::Success)
            << timed.reason_code << ": " << timed.error.description;
        ASSERT_EQ(timed.command_status, GpuCommandStatus::Completed);
        ASSERT_EQ(timed.command_buffer_count, 1U);
        ASSERT_EQ(timed.compute_encoder_count, 1U);
        ASSERT_EQ(timed.dispatch_count, kPasses);
        ASSERT_TRUE(std::isfinite(timed.gpu_elapsed_seconds));
        ASSERT_GT(timed.gpu_elapsed_seconds, 0.0);
        ASSERT_EQ(timed.actual_accumulator, timed.expected_accumulator);

        const GpuValidationResult validation =
            backend_->run_validation(attempt_request, timed);
        ASSERT_EQ(validation.status, GpuBackendStatus::Success)
            << validation.reason_code << ": "
            << validation.error.description;
        ASSERT_EQ(validation.validation_status,
                  GpuValidationStatus::Passed)
            << validation.reason_code;
        ASSERT_EQ(validation.actual_final_checksum,
                  validation.expected_final_checksum);

        const long double bandwidth =
            static_cast<long double>(plan.exact_payload_bytes) /
            static_cast<long double>(timed.gpu_elapsed_seconds) /
            1.0e9L;
        const double value_gb_s = static_cast<double>(bandwidth);
        ASSERT_TRUE(std::isfinite(value_gb_s));
        ASSERT_GT(value_gb_s, 0.0);
        candidate_values[operation_index][candidate_index].push_back(
            value_gb_s);
      }
    }
  }

  const GpuEnvironmentSnapshot environment_end =
      backend_->snapshot_environment();
  const std::ios::fmtflags original_flags = std::cout.flags();
  const std::streamsize original_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(6)
            << "[ M4 GRID ] environment_start thermal="
            << environment_start.thermal_state
            << " low_power_mode_available="
            << environment_start.low_power_mode_available
            << " low_power_mode_enabled="
            << environment_start.low_power_mode_enabled << '\n'
            << "[ M4 GRID ] environment_end thermal="
            << environment_end.thermal_state
            << " low_power_mode_available="
            << environment_end.low_power_mode_available
            << " low_power_mode_enabled="
            << environment_end.low_power_mode_enabled << '\n';
  for (size_t operation_index = 0; operation_index < kOperations.size();
       ++operation_index) {
    std::array<double, kCandidateThreadgroups.size()> medians{};
    for (size_t candidate_index = 0;
         candidate_index < kCandidateThreadgroups.size();
         ++candidate_index) {
      ASSERT_EQ(candidate_values[operation_index][candidate_index].size(),
                kSampleCount);
      const DescriptiveStatistics statistics =
          calculate_descriptive_statistics(
              candidate_values[operation_index][candidate_index]);
      ASSERT_TRUE(statistics.coefficient_of_variation_defined);
      medians[candidate_index] = statistics.median;
      std::cout << "[ M4 GRID ] operation="
                << gpu_operation_to_string(kOperations[operation_index])
                << " threadgroups="
                << kCandidateThreadgroups[candidate_index]
                << " count=" << statistics.sample_count
                << " median_gb_s=" << statistics.median
                << " cv_pct="
                << statistics.coefficient_of_variation_pct << '\n';
    }

    const double best_median =
        *std::max_element(medians.begin(), medians.end());
    const auto production_candidate = std::find(
        kCandidateThreadgroups.begin(), kCandidateThreadgroups.end(),
        Constants::GPU_MAX_THREADGROUPS_PER_GRID);
    ASSERT_NE(production_candidate, kCandidateThreadgroups.end());
    const size_t production_index = static_cast<size_t>(std::distance(
        kCandidateThreadgroups.begin(), production_candidate));
    const double production_median = medians[production_index];
    EXPECT_TRUE(grid_candidate_passes_selection_rule(production_median,
                                                     best_median))
        << "operation="
        << gpu_operation_to_string(kOperations[operation_index])
        << " medians(2048,4096,8192)=" << medians[0] << ","
        << medians[1] << "," << medians[2];
  }
  std::cout.flags(original_flags);
  std::cout.precision(original_precision);
}

}  // namespace
