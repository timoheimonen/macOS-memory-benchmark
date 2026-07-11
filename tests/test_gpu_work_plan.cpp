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

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "benchmark/benchmark_work_plan.h"
#include "core/config/constants.h"
#include "gpu_bandwidth/gpu_backend.h"
#include "gpu_bandwidth/gpu_work_plan.h"
#include "utils/cyclic_order.h"

namespace {

GpuWorkPlanRequest make_request(GpuOperation operation, size_t buffer_bytes,
                                size_t passes) {
  GpuWorkPlanRequest request;
  request.operation = operation;
  request.requested_buffer_bytes = buffer_bytes;
  request.passes = passes;
  request.base_seed = 42;
  request.thread_execution_width = 32;
  request.max_total_threads_per_threadgroup = 1024;
  request.data_resource_options = 33;
  request.status_resource_options = 17;
  request.kernel_revision = "test-kernel-v1";
  request.kernel_source_sha256 = "test-source-sha256";
  return request;
}

}  // namespace

TEST(GpuWorkPlanTest, ConstantsMatchLockedGpuMethodology) {
  EXPECT_EQ(Constants::GPU_DEFAULT_BUFFER_SIZE_MB, 512u);
  EXPECT_EQ(Constants::GPU_MIN_BUFFER_SIZE_MB, 64u);
  EXPECT_EQ(Constants::GPU_DEFAULT_LOOP_COUNT, 3u);
  EXPECT_DOUBLE_EQ(Constants::GPU_CALIBRATION_TARGET_SECONDS,
                   Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS);
  EXPECT_DOUBLE_EQ(Constants::GPU_CALIBRATION_MIN_SECONDS, 0.100);
  EXPECT_DOUBLE_EQ(Constants::GPU_CALIBRATION_MAX_SECONDS, 0.250);
  EXPECT_EQ(Constants::GPU_CALIBRATION_MAX_CORRECTIONS, 2u);
  EXPECT_EQ(Constants::GPU_CALIBRATION_MIN_PILOT_BYTES,
            8 * Constants::BYTES_PER_MB);
  EXPECT_EQ(Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT, 16384u);
  EXPECT_EQ(Constants::GPU_MAX_EXACT_PAYLOAD_BYTES,
            64ULL * 1024ULL * Constants::BYTES_PER_MB);
  EXPECT_EQ(Constants::GPU_VECTOR_WIDTH_BYTES, 16u);
  EXPECT_EQ(Constants::GPU_THREADS_PER_THREADGROUP_CAP, 256u);
  EXPECT_EQ(Constants::GPU_MAX_THREADGROUPS_PER_GRID, 8192u);
  EXPECT_DOUBLE_EQ(Constants::GPU_STREAMING_CV_WARNING_PCT, 5.0);
}

TEST(GpuWorkPlanTest, SharedCyclicOrderPreservesBenchmarkCompatibility) {
  EXPECT_EQ(build_cyclic_order(4, 0),
            (std::vector<size_t>{0, 1, 2, 3}));
  EXPECT_EQ(build_cyclic_order(4, 1),
            (std::vector<size_t>{1, 2, 3, 0}));
  EXPECT_EQ(build_cyclic_order(4, std::numeric_limits<size_t>::max()),
            build_benchmark_cyclic_order(
                4, std::numeric_limits<size_t>::max()));
  EXPECT_TRUE(build_cyclic_order(0, 7).empty());
}

TEST(GpuWorkPlanTest, OperationOrderRotatesReadWriteCopyAcrossLoops) {
  EXPECT_EQ(build_gpu_operation_order(0),
            (std::array<GpuOperation, 3>{GpuOperation::Read,
                                         GpuOperation::Write,
                                         GpuOperation::Copy}));
  EXPECT_EQ(build_gpu_operation_order(1),
            (std::array<GpuOperation, 3>{GpuOperation::Write,
                                         GpuOperation::Copy,
                                         GpuOperation::Read}));
  EXPECT_EQ(build_gpu_operation_order(2),
            (std::array<GpuOperation, 3>{GpuOperation::Copy,
                                         GpuOperation::Read,
                                         GpuOperation::Write}));
  EXPECT_EQ(build_gpu_operation_order(3), build_gpu_operation_order(0));
}

TEST(GpuWorkPlanTest, OperationSeedsHaveStableSplitMixDomains) {
  EXPECT_EQ(gpu_operation_seed_domain(GpuOperation::Read),
            0x4750555f52454144ULL);
  EXPECT_EQ(gpu_operation_seed_domain(GpuOperation::Write),
            0x4750555752495445ULL);
  EXPECT_EQ(gpu_operation_seed_domain(GpuOperation::Copy),
            0x4750555f434f5059ULL);
  EXPECT_EQ(derive_gpu_operation_seed(42, GpuOperation::Read),
            0x2d62513ce5b9b955ULL);
  EXPECT_EQ(derive_gpu_operation_seed(42, GpuOperation::Write),
            0x11dbcad689d7ff1bULL);
  EXPECT_EQ(derive_gpu_operation_seed(42, GpuOperation::Copy),
            0x283298849b447452ULL);
  EXPECT_NE(derive_gpu_operation_seed(42, GpuOperation::Read),
            derive_gpu_operation_seed(43, GpuOperation::Read));
  EXPECT_EQ(derive_gpu_operation_seed(
                42, static_cast<GpuOperation>(99)),
            0u);
}

TEST(GpuWorkPlanTest, PassLimitsUseExactPayloadsAndEffectiveCaps) {
  const size_t buffer_bytes = 64 * Constants::BYTES_PER_MB;
  const GpuPassLimits read =
      calculate_gpu_pass_limits(buffer_bytes, GpuOperation::Read);
  const GpuPassLimits write =
      calculate_gpu_pass_limits(buffer_bytes, GpuOperation::Write);
  const GpuPassLimits copy =
      calculate_gpu_pass_limits(buffer_bytes, GpuOperation::Copy);

  ASSERT_TRUE(read.valid);
  ASSERT_TRUE(write.valid);
  ASSERT_TRUE(copy.valid);
  EXPECT_EQ(read.bytes_per_pass, buffer_bytes);
  EXPECT_EQ(write.bytes_per_pass, buffer_bytes);
  EXPECT_EQ(copy.bytes_per_pass, 2 * buffer_bytes);
  EXPECT_EQ(read.maximum_passes_by_payload, 1024u);
  EXPECT_EQ(read.effective_maximum_passes, 1024u);
  EXPECT_EQ(copy.maximum_passes_by_payload, 512u);
  EXPECT_EQ(copy.effective_maximum_passes, 512u);
  EXPECT_TRUE(read.payload_cap_is_limiting);
  EXPECT_FALSE(read.dispatch_cap_is_limiting);

  const GpuPassLimits tiny =
      calculate_gpu_pass_limits(1, GpuOperation::Read);
  ASSERT_TRUE(tiny.valid);
  EXPECT_EQ(tiny.effective_maximum_passes,
            Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT);
  EXPECT_TRUE(tiny.dispatch_cap_is_limiting);
  EXPECT_FALSE(tiny.payload_cap_is_limiting);
}

TEST(GpuWorkPlanTest, PassLimitsRejectZeroOverflowAndOverCapSinglePass) {
  const GpuPassLimits zero =
      calculate_gpu_pass_limits(0, GpuOperation::Read);
  EXPECT_FALSE(zero.valid);
  EXPECT_EQ(zero.reason_code, GpuWorkPlanReason::BUFFER_SIZE_ZERO);

  const GpuPassLimits invalid = calculate_gpu_pass_limits(
      4096, static_cast<GpuOperation>(99));
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, GpuWorkPlanReason::INVALID_OPERATION);

  const GpuPassLimits copy_overflow = calculate_gpu_pass_limits(
      std::numeric_limits<size_t>::max(), GpuOperation::Copy);
  EXPECT_FALSE(copy_overflow.valid);
  EXPECT_EQ(copy_overflow.reason_code,
            GpuWorkPlanReason::PAYLOAD_PER_PASS_OVERFLOW);

  const GpuPassLimits too_large = calculate_gpu_pass_limits(
      Constants::GPU_MAX_EXACT_PAYLOAD_BYTES + 1, GpuOperation::Read);
  EXPECT_FALSE(too_large.valid);
  EXPECT_EQ(too_large.reason_code,
            GpuWorkPlanReason::PAYLOAD_CAP_BELOW_ONE_PASS);
}

TEST(GpuWorkPlanTest, PlannerUsesDeterministicVectorTailAndGridGeometry) {
  GpuWorkPlanRequest request = make_request(GpuOperation::Read, 31, 2);
  request.max_total_threads_per_threadgroup = 200;
  const GpuWorkPlan plan = build_gpu_work_plan(request);

  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.vector_width_bytes, 16u);
  EXPECT_EQ(plan.vector_count, 1u);
  EXPECT_EQ(plan.tail_bytes, 15u);
  EXPECT_EQ(plan.threads_per_threadgroup, 192u);
  EXPECT_EQ(plan.required_threadgroups_per_grid, 1u);
  EXPECT_EQ(plan.threadgroups_per_grid, 1u);
  EXPECT_EQ(plan.grid_threads, 192u);
  EXPECT_EQ(plan.dispatch_count, 2u);
  EXPECT_EQ(plan.measured_command_buffer_count, 1u);
  EXPECT_EQ(plan.measured_compute_encoder_count, 1u);
  EXPECT_EQ(plan.data_resource_options, 33u);
  EXPECT_EQ(plan.status_resource_options, 17u);
  EXPECT_EQ(plan.data_storage_mode, "private");
  EXPECT_EQ(plan.data_hazard_tracking_mode, "tracked");
  EXPECT_EQ(plan.status_storage_mode, "shared");
  EXPECT_EQ(plan.status_hazard_tracking_mode, "tracked");
  EXPECT_EQ(plan.kernel_revision, "test-kernel-v1");
  EXPECT_EQ(plan.kernel_source_sha256, "test-source-sha256");
  EXPECT_EQ(plan.msl_language_version, "2.3");
  EXPECT_EQ(plan.methodology_version, Constants::GPU_METHODOLOGY_VERSION);
  EXPECT_EQ(plan.timing_policy,
            "one-command-buffer-one-serial-encoder");
  EXPECT_EQ(plan.pass_mapping,
            "one-full-buffer-pass-per-dispatch");
  EXPECT_EQ(plan.warmup_policy,
            "excluded-full-buffer-warmup-before-each-attempt");
  EXPECT_EQ(plan.calibration_policy,
            "automatic-150ms-100-250ms-max-two-corrections");
  EXPECT_EQ(plan.duration_quality_policy,
            "report-without-performance-retry-after-freeze");
  EXPECT_EQ(plan.timed_accumulator_algorithm, "gpu-dual-mod32-v2");
  EXPECT_EQ(plan.final_checksum_algorithm, "not-applicable");

  struct TailCase {
    size_t buffer_bytes;
    size_t vector_count;
    size_t tail_bytes;
  };
  for (const TailCase& tail_case :
       {TailCase{1, 0, 1}, TailCase{15, 0, 15}, TailCase{17, 1, 1},
        TailCase{31, 1, 15}}) {
    const GpuWorkPlan tail_plan = build_gpu_work_plan(
        make_request(GpuOperation::Write, tail_case.buffer_bytes, 1));
    ASSERT_TRUE(tail_plan.valid) << tail_case.buffer_bytes;
    EXPECT_EQ(tail_plan.vector_count, tail_case.vector_count);
    EXPECT_EQ(tail_plan.tail_bytes, tail_case.tail_bytes);
    EXPECT_EQ(tail_plan.threadgroups_per_grid, 1u);
  }
}

TEST(GpuWorkPlanTest, PlannerCapsGridAndSupportsValidationOverrides) {
  const size_t vector_count =
      Constants::GPU_MAX_THREADGROUPS_PER_GRID *
          Constants::GPU_THREADS_PER_THREADGROUP_CAP +
      1;
  const size_t buffer_bytes =
      vector_count * Constants::GPU_VECTOR_WIDTH_BYTES;
  GpuWorkPlanRequest request =
      make_request(GpuOperation::Read, buffer_bytes, 1);

  const GpuWorkPlan production = build_gpu_work_plan(request);
  ASSERT_TRUE(production.valid) << production.reason_code;
  EXPECT_EQ(production.threads_per_threadgroup, 256u);
  EXPECT_EQ(production.required_threadgroups_per_grid, 8193u);
  EXPECT_EQ(production.threadgroups_per_grid, 8192u);
  EXPECT_EQ(production.grid_threads, 8192u * 256u);

  request.maximum_threadgroups_per_grid = 2048;
  const GpuWorkPlan validation_override = build_gpu_work_plan(request);
  ASSERT_TRUE(validation_override.valid);
  EXPECT_EQ(validation_override.required_threadgroups_per_grid, 8193u);
  EXPECT_EQ(validation_override.threadgroups_per_grid, 2048u);
  EXPECT_NE(validation_override.plan_identity, production.plan_identity);
}

TEST(GpuWorkPlanTest, PlannerRejectsInvalidPipelineGeometry) {
  GpuWorkPlanRequest request = make_request(GpuOperation::Read, 4096, 1);
  request.thread_execution_width = 0;
  EXPECT_EQ(build_gpu_work_plan(request).reason_code,
            GpuWorkPlanReason::THREAD_EXECUTION_WIDTH_ZERO);

  request = make_request(GpuOperation::Read, 4096, 1);
  request.max_total_threads_per_threadgroup = 0;
  EXPECT_EQ(build_gpu_work_plan(request).reason_code,
            GpuWorkPlanReason::MAX_THREADS_PER_THREADGROUP_ZERO);

  request = make_request(GpuOperation::Read, 4096, 1);
  request.thread_execution_width = 257;
  EXPECT_EQ(build_gpu_work_plan(request).reason_code,
            GpuWorkPlanReason::THREAD_EXECUTION_WIDTH_EXCEEDS_CAP);

  request = make_request(GpuOperation::Read, 4096, 1);
  request.maximum_threadgroups_per_grid = 0;
  EXPECT_EQ(build_gpu_work_plan(request).reason_code,
            GpuWorkPlanReason::THREADGROUP_LIMIT_ZERO);
}

TEST(GpuWorkPlanTest, PlannerCalculatesExactPayloadAndRejectsExplicitCaps) {
  const GpuWorkPlan read =
      build_gpu_work_plan(make_request(GpuOperation::Read, 4096, 5));
  const GpuWorkPlan copy =
      build_gpu_work_plan(make_request(GpuOperation::Copy, 4096, 5));
  ASSERT_TRUE(read.valid);
  ASSERT_TRUE(copy.valid);
  EXPECT_EQ(read.bytes_per_pass, 4096u);
  EXPECT_EQ(read.exact_payload_bytes, 20480u);
  EXPECT_EQ(copy.bytes_per_pass, 8192u);
  EXPECT_EQ(copy.exact_payload_bytes, 40960u);

  GpuWorkPlanRequest dispatch_exceeded =
      make_request(GpuOperation::Read, 1,
                   Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT + 1);
  dispatch_exceeded.explicit_iterations = true;
  EXPECT_EQ(build_gpu_work_plan(dispatch_exceeded).reason_code,
            GpuWorkPlanReason::EXPLICIT_DISPATCH_CAP_EXCEEDED);

  GpuWorkPlanRequest payload_exceeded = make_request(
      GpuOperation::Copy, 64 * Constants::BYTES_PER_MB, 513);
  payload_exceeded.explicit_iterations = true;
  EXPECT_EQ(build_gpu_work_plan(payload_exceeded).reason_code,
            GpuWorkPlanReason::EXPLICIT_PAYLOAD_CAP_EXCEEDED);

  payload_exceeded.passes = 512;
  const GpuWorkPlan at_cap = build_gpu_work_plan(payload_exceeded);
  ASSERT_TRUE(at_cap.valid) << at_cap.reason_code;
  EXPECT_EQ(at_cap.exact_payload_bytes,
            Constants::GPU_MAX_EXACT_PAYLOAD_BYTES);
}

TEST(GpuWorkPlanTest, InvalidPlansRetainStableReasonsWithoutIdentity) {
  GpuWorkPlanRequest zero_passes =
      make_request(GpuOperation::Read, 4096, 0);
  const GpuWorkPlan zero = build_gpu_work_plan(zero_passes);
  EXPECT_FALSE(zero.valid);
  EXPECT_EQ(zero.reason_code, GpuWorkPlanReason::PASS_COUNT_ZERO);
  EXPECT_TRUE(zero.plan_identity.empty());
  EXPECT_EQ(zero.exact_payload_bytes, 0u);

  GpuWorkPlanRequest automatic_dispatch_exceeded = make_request(
      GpuOperation::Read, 1,
      Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT + 1);
  EXPECT_EQ(build_gpu_work_plan(automatic_dispatch_exceeded).reason_code,
            GpuWorkPlanReason::DISPATCH_CAP_EXCEEDED);

  GpuWorkPlanRequest automatic_payload_exceeded = make_request(
      GpuOperation::Copy, 64 * Constants::BYTES_PER_MB, 513);
  EXPECT_EQ(build_gpu_work_plan(automatic_payload_exceeded).reason_code,
            GpuWorkPlanReason::PAYLOAD_CAP_EXCEEDED);
}

TEST(GpuWorkPlanTest, CalibrationHelpersScaleClampAndClassifyGuardrails) {
  const GpuPassLimits small =
      calculate_gpu_pass_limits(1, GpuOperation::Read);
  const GpuPassLimits payload_limited = calculate_gpu_pass_limits(
      64 * Constants::BYTES_PER_MB, GpuOperation::Read);
  ASSERT_TRUE(small.valid);
  ASSERT_TRUE(payload_limited.valid);

  EXPECT_EQ(calculate_gpu_pilot_passes(small),
            Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT);
  EXPECT_EQ(calculate_gpu_pilot_passes(payload_limited), 1u);
  EXPECT_EQ(calculate_gpu_calibrated_passes(0.010, 100, small), 1500u);
  EXPECT_EQ(calculate_gpu_calibrated_passes(0.0, 100, small), 0u);

  EXPECT_FALSE(gpu_duration_in_target_window(
      std::nextafter(Constants::GPU_CALIBRATION_MIN_SECONDS, 0.0)));
  EXPECT_TRUE(gpu_duration_in_target_window(
      Constants::GPU_CALIBRATION_MIN_SECONDS));
  EXPECT_TRUE(gpu_duration_in_target_window(
      Constants::GPU_CALIBRATION_MAX_SECONDS));
  EXPECT_FALSE(gpu_duration_in_target_window(
      std::nextafter(Constants::GPU_CALIBRATION_MAX_SECONDS,
                     std::numeric_limits<double>::infinity())));

  EXPECT_EQ(classify_gpu_duration_quality(0.150, 10, small),
            "within-target-window");
  EXPECT_EQ(classify_gpu_duration_quality(
                0.050, small.effective_maximum_passes, small),
            "dispatch-cap-below-target");
  EXPECT_EQ(classify_gpu_duration_quality(
                0.050, payload_limited.effective_maximum_passes,
                payload_limited),
            "payload-cap-below-target");
  EXPECT_EQ(classify_gpu_duration_quality(0.300, 1, small),
            "single-pass-exceeds-window");
  EXPECT_EQ(classify_gpu_duration_quality(0.300, 2, small),
            "above-target-window");
  EXPECT_EQ(classify_gpu_duration_quality(
                std::numeric_limits<double>::quiet_NaN(), 1, small),
            "invalid-duration");
}

TEST(GpuWorkPlanTest, FrozenPlanIdentityIsCanonicalAndSensitiveToWork) {
  GpuWorkPlanRequest request = make_request(GpuOperation::Read, 31, 2);
  request.max_total_threads_per_threadgroup = 200;
  const GpuWorkPlan first = build_gpu_work_plan(request);
  const GpuWorkPlan repeated = build_gpu_work_plan(request);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(repeated.valid);
  EXPECT_EQ(first.plan_identity, repeated.plan_identity);
  EXPECT_EQ(
      first.plan_identity,
      "gpu-work-plan-v1|operation=read|requested_buffer_bytes=31|"
      "effective_buffer_bytes=31|passes=2|bytes_per_pass=31|"
      "exact_payload_bytes=62|base_seed=42|"
      "operation_seed=3270265601418443093|vector_width_bytes=16|"
      "vector_count=1|tail_bytes=15|thread_execution_width=32|"
      "max_total_threads_per_threadgroup=200|threads_per_threadgroup=192|"
      "maximum_threadgroups_per_grid=8192|threadgroups_per_grid=1|"
      "grid_threads=192|dispatch_count=2|data_resource_options=33|"
      "status_resource_options=17|data_storage_mode=private|"
      "data_hazard_tracking_mode=tracked|status_storage_mode=shared|"
      "status_hazard_tracking_mode=tracked|kernel_revision=test-kernel-v1|"
      "kernel_source_sha256=test-source-sha256|msl_language_version=2.3|"
      "methodology_version=gpu-bandwidth-v1-private-runtime-single-cmdbuf-"
      "calibrated-balanced|timing_policy=one-command-buffer-one-serial-encoder|"
      "pass_mapping=one-full-buffer-pass-per-dispatch|warmup_policy=excluded-"
      "full-buffer-warmup-before-each-attempt|calibration_policy=automatic-"
      "150ms-100-250ms-max-two-corrections|duration_quality_policy=report-"
      "without-performance-retry-after-freeze|timed_accumulator_algorithm="
      "gpu-dual-mod32-v2|final_checksum_algorithm=not-applicable");

  request.passes = 3;
  EXPECT_NE(build_gpu_work_plan(request).plan_identity,
            first.plan_identity);
  request.passes = 2;
  request.base_seed = 43;
  EXPECT_NE(build_gpu_work_plan(request).plan_identity,
            first.plan_identity);
  request.base_seed = 42;
  request.kernel_source_sha256 = "different-source-sha256";
  EXPECT_NE(build_gpu_work_plan(request).plan_identity,
            first.plan_identity);
}

TEST(GpuTimedAccumulatorOracleTest,
     PowerOfTwoReleaseWorkIsNonzeroAndPassSensitive) {
  constexpr size_t kBufferSizeBytes =
      512U * Constants::BYTES_PER_MB;
  constexpr uint64_t kBaseSeed = 6102026ULL;
  constexpr std::array<GpuOperation, 3> kOperations = {
      GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};
  constexpr std::array<GpuDualChecksum, 3> kExpectedPass24 = {
      GpuDualChecksum{1292850922U, 1629732278U},
      GpuDualChecksum{1684113256U, 3742197060U},
      GpuDualChecksum{1151773566U, 3961788034U}};
  constexpr std::array<GpuDualChecksum, 3> kExpectedPass64 = {
      GpuDualChecksum{2454305124U, 2861025294U},
      GpuDualChecksum{1933397850U, 2368268164U},
      GpuDualChecksum{1074271602U, 2541544478U}};
  const GpuDualChecksum zero;

  for (size_t operation_index = 0; operation_index < kOperations.size();
       ++operation_index) {
    const GpuOperation operation = kOperations[operation_index];
    GpuBackendAttemptRequest request;
    request.operation = operation;
    request.buffer_size_bytes = kBufferSizeBytes;
    request.operation_seed =
        derive_gpu_operation_seed(kBaseSeed, operation);
    request.vector_count =
        kBufferSizeBytes / Constants::GPU_VECTOR_WIDTH_BYTES;

    request.passes = 23;
    const GpuDualChecksum pass_23 =
        calculate_expected_gpu_timed_accumulator(request);
    request.passes = 24;
    const GpuDualChecksum pass_24 =
        calculate_expected_gpu_timed_accumulator(request);
    request.passes = 63;
    const GpuDualChecksum pass_63 =
        calculate_expected_gpu_timed_accumulator(request);
    request.passes = 64;
    const GpuDualChecksum pass_64 =
        calculate_expected_gpu_timed_accumulator(request);

    SCOPED_TRACE(::testing::Message()
                 << "operation=" << gpu_operation_to_string(operation));
    EXPECT_NE(pass_24, zero);
    EXPECT_NE(pass_64, zero);
    EXPECT_EQ(pass_24, kExpectedPass24[operation_index]);
    EXPECT_EQ(pass_64, kExpectedPass64[operation_index]);
    EXPECT_NE(pass_23, pass_24);
    EXPECT_NE(pass_24, pass_64);
    EXPECT_NE(pass_63, pass_64);
    EXPECT_NE(pass_24.first, 0U);
    EXPECT_NE(pass_24.second, 0U);
    EXPECT_NE(pass_64.first, 0U);
    EXPECT_NE(pass_64.second, 0U);
  }
}
