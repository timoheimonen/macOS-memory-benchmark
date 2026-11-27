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
#include <set>
#include <string>
#include <vector>

#include "core/audit/deterministic_rng.h"
#include "pattern_benchmark/patterns_work_plan.h"

namespace {

PatternsSystemInfo make_sys_info(size_t avail_gb = 16,
                                 size_t page_size_bytes = 16384ULL) {
  PatternsSystemInfo info;
  info.page_size_bytes = page_size_bytes;
  info.available_memory_bytes = avail_gb * 1024ULL * 1024ULL * 1024ULL;
  return info;
}

PatternsWorkPlanConfig make_config(size_t buffer_size_mb = 4,
                                   size_t workers = 2,
                                   size_t loop_count = 3,
                                   uint64_t seed = 424242ULL) {
  PatternsWorkPlanConfig cfg;
  cfg.buffer_size_mb = buffer_size_mb;
  cfg.requested_workers = workers;
  cfg.available_workers = workers;
  cfg.user_specified_workers = true;
  cfg.loop_count = loop_count;
  cfg.seed = seed;
  cfg.output_file = "out.json";
  return cfg;
}

TEST(PatternsWorkPlanTest, CanonicalEnumStringsIndicesAndCvThresholds) {
  const std::array<PatternKind, kPatternsMaxKinds> kinds = {
      PatternKind::SequentialForward, PatternKind::SequentialReverse,
      PatternKind::Strided64,         PatternKind::Strided4096,
      PatternKind::Strided16384,      PatternKind::Strided2MiB,
      PatternKind::Random,
  };
  const std::array<const char*, kPatternsMaxKinds> names = {
      "sequential_forward", "sequential_reverse", "strided_64", "strided_4096",
      "strided_16384",      "strided_2mb",        "random",
  };
  const std::array<size_t, kPatternsMaxKinds> strides = {
      0, 0, 64, 4096, 16384, 2097152, 0,
  };
  const std::array<bool, kPatternsMaxKinds> is_strided = {
      false, false, true, true, true, true, false,
  };
  const std::array<bool, kPatternsMaxKinds> is_streaming = {
      true, true, true, false, false, false, false,
  };

  for (size_t i = 0; i < kPatternsMaxKinds; ++i) {
    EXPECT_EQ(patterns_kind_index(kinds[i]), i);
    EXPECT_STREQ(patterns_kind_to_string(kinds[i]), names[i]);
    EXPECT_EQ(patterns_kind_stride_bytes(kinds[i]), strides[i]);
    EXPECT_EQ(patterns_kind_is_strided(kinds[i]), is_strided[i]);
    EXPECT_EQ(patterns_kind_is_streaming(kinds[i]), is_streaming[i]);
    EXPECT_DOUBLE_EQ(patterns_cv_warning_threshold_pct(kinds[i]),
                     is_streaming[i] ? 5.0 : 10.0);
  }

  EXPECT_EQ(patterns_operation_index(PatternOperation::Read), 0u);
  EXPECT_EQ(patterns_operation_index(PatternOperation::Write), 1u);
  EXPECT_EQ(patterns_operation_index(PatternOperation::Copy), 2u);
  EXPECT_STREQ(patterns_operation_to_string(PatternOperation::Read), "read");
  EXPECT_STREQ(patterns_operation_to_string(PatternOperation::Write), "write");
  EXPECT_STREQ(patterns_operation_to_string(PatternOperation::Copy), "copy");
}

TEST(PatternsWorkPlanTest, ComponentIdentitiesAndSplitMix64SeedHierarchy) {
  const auto cfg = make_config(3, 2, 3, 424242ULL);
  const PatternsWorkPlan plan = build_patterns_work_plan(cfg, make_sys_info());
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  EXPECT_EQ(kPatternsSchemaVersion, 1);
  EXPECT_STREQ(kPatternsModeName, "patterns");
  EXPECT_STREQ(kPatternsMethodologyVersion,
               "patterns-v1-arm64-phase-calibrated-seeded-balanced");
  EXPECT_EQ(plan.component_identities.version, "patterns-components-v1");
  EXPECT_EQ(plan.component_identities.work_plan_version,
            "patterns-work-plan-v1");
  EXPECT_EQ(plan.component_identities.buffer_pattern_version,
            "patterns-buffer-pattern-v1");
  EXPECT_EQ(plan.component_identities.read_checksum_version,
            "patterns-read-checksum-v1");
  EXPECT_EQ(plan.component_identities.random_index_generator_version,
            "splitmix64-coprime-fisher-yates-v1");
  EXPECT_EQ(plan.component_identities.run_policy_version,
            "bounded-loop-snapshots-v1");

  EXPECT_EQ(plan.src_seed, splitmix64_mix(424242ULL ^ kPatternDomainSrc));
  EXPECT_EQ(plan.random_domain_seed,
            splitmix64_mix(424242ULL ^ kPatternDomainRandom));
  EXPECT_EQ(plan.pre_op_sentinel_seed,
            splitmix64_mix(424242ULL ^ kPatternDomainPreOpSentinel));

  std::set<uint64_t> op_seeds;
  for (size_t k = 0; k < kPatternsMaxKinds; ++k) {
    for (size_t o = 0; o < kPatternsOperationCount; ++o) {
      const auto kind = static_cast<PatternKind>(k);
      const auto op = static_cast<PatternOperation>(o);
      const uint64_t expected_op_seed =
          splitmix64_mix(plan.src_seed ^ ((uint64_t{k} + 1ULL) << 48U) ^
                         ((uint64_t{o} + 1ULL) << 56U) ^ 7ULL);
      const uint64_t actual =
          derive_patterns_op_seed(plan.src_seed, kind, op, 7);
      EXPECT_EQ(actual, expected_op_seed);
      op_seeds.insert(actual);
    }
  }
  EXPECT_EQ(op_seeds.size(), kPatternsMaxKinds * kPatternsOperationCount);
}

TEST(PatternsWorkPlanTest, Partition64bPreservesAlignmentAndContiguity) {
  const std::vector<size_t> sizes = {63, 64, 65, 127, 128, 193, 16384, 524288};
  const std::vector<size_t> workers_list = {1, 2, 3, 4, 10};
  for (size_t sz : sizes) {
    for (size_t w : workers_list) {
      const auto spans = partition_patterns_64b(sz, w);
      ASSERT_EQ(spans.size(), w);
      size_t cursor = 0;
      for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i].span_bytes > 0) {
          EXPECT_EQ(spans[i].offset_bytes, cursor);
          if (i > 0 && sz >= w * 64) {
            EXPECT_EQ(spans[i].offset_bytes % 64u, 0u);
          }
          cursor += spans[i].span_bytes;
        }
      }
      EXPECT_EQ(cursor, sz);
    }
  }
}

TEST(PatternsWorkPlanTest, SevenPatternGeometryAndStrided2MiBSkipVsActive) {
  // 1 MiB and 2 MiB buffers skip strided_2mb because buffer_bytes < 2 MiB + 32 B.
  for (size_t mb : {size_t{1}, size_t{2}}) {
    const PatternsWorkPlan plan =
        build_patterns_work_plan(make_config(mb, 4), make_sys_info(16, 16384));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.active_patterns.size(), 6u);
    EXPECT_EQ(plan.find_pattern(PatternKind::Strided2MiB), nullptr);
    const PatternPlanEntry* s2m = plan.entry(PatternKind::Strided2MiB);
    ASSERT_NE(s2m, nullptr);
    EXPECT_FALSE(s2m->enabled);
    EXPECT_EQ(s2m->status, "skipped");
    EXPECT_EQ(s2m->reason_code,
              PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);
    EXPECT_EQ(s2m->phase_period_passes, 65536u);
    EXPECT_FALSE(s2m->large_page_backing_verified);
    EXPECT_EQ(s2m->large_page_backing_status, "not-verified");
    EXPECT_EQ(s2m->effective_workers, 0u);
    EXPECT_TRUE(s2m->boundaries.empty());

    const PatternPlanEntry* s16k = plan.find_pattern(PatternKind::Strided16384);
    ASSERT_NE(s16k, nullptr);
    EXPECT_EQ(s16k->native_page_size_bytes, 16384u);
    EXPECT_TRUE(s16k->stride_equals_native_page_size);
    EXPECT_FALSE(plan.find_pattern(PatternKind::Strided4096)
                     ->stride_equals_native_page_size);
  }

  // 4 KiB native page size switches stride_equals_native_page_size to Strided4096.
  {
    const PatternsWorkPlan plan_4k =
        build_patterns_work_plan(make_config(1, 2), make_sys_info(16, 4096));
    ASSERT_TRUE(plan_4k.valid);
    EXPECT_TRUE(plan_4k.find_pattern(PatternKind::Strided4096)
                    ->stride_equals_native_page_size);
    EXPECT_FALSE(plan_4k.find_pattern(PatternKind::Strided16384)
                     ->stride_equals_native_page_size);
  }

  // 3 MiB buffer: all 7 patterns active, strided_2mb reduces workers to 1.
  {
    const PatternsWorkPlan plan_3m =
        build_patterns_work_plan(make_config(3, 4), make_sys_info());
    ASSERT_TRUE(plan_3m.valid) << plan_3m.reason_code;
    EXPECT_EQ(plan_3m.active_patterns.size(), 7u);
    const PatternPlanEntry* s2m = plan_3m.find_pattern(PatternKind::Strided2MiB);
    ASSERT_NE(s2m, nullptr);
    EXPECT_TRUE(s2m->enabled);
    EXPECT_EQ(s2m->status, "active");
    EXPECT_EQ(s2m->effective_workers, 1u);
    EXPECT_EQ(s2m->large_page_backing_status, "not-verified");
  }

  // 8 MiB buffer with -t 10: 8 MiB / (2 MiB + 32 B) = 3 workers max.
  {
    const PatternsWorkPlan plan_8m =
        build_patterns_work_plan(make_config(8, 10), make_sys_info());
    ASSERT_TRUE(plan_8m.valid) << plan_8m.reason_code;
    const PatternPlanEntry* s2m = plan_8m.find_pattern(PatternKind::Strided2MiB);
    ASSERT_NE(s2m, nullptr);
    EXPECT_EQ(s2m->effective_workers, 3u);
    for (const auto& sp : s2m->worker_spans) {
      EXPECT_GE(sp.span_bytes, s2m->stride_bytes + 32u);
      EXPECT_GE(sp.accesses_per_pass, 2u);
    }
  }
}

TEST(PatternsWorkPlanTest, PhaseRotatedStridedOperationPlanAccounting) {
  const PatternsWorkPlan plan =
      build_patterns_work_plan(make_config(4, 2), make_sys_info());
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  for (PatternKind kind : {PatternKind::Strided64, PatternKind::Strided4096,
                           PatternKind::Strided16384,
                           PatternKind::Strided2MiB}) {
    const PatternPlanEntry* entry = plan.find_pattern(kind);
    ASSERT_NE(entry, nullptr);
    const size_t Q = entry->phase_period_passes;

    // Partial cycle (1 pass < Q): untouched bytes exist.
    const auto op_partial = build_patterns_operation_plan(
        *entry, plan.src_seed, PatternOperation::Write, 1, true);
    ASSERT_TRUE(op_partial.valid);
    EXPECT_EQ(op_partial.completed_phase_cycles, 0u);
    EXPECT_EQ(op_partial.total_accesses, entry->accesses_per_pass);
    EXPECT_EQ(op_partial.exact_payload_bytes, entry->accesses_per_pass * 32u);
    EXPECT_EQ(op_partial.distinct_address_count, entry->accesses_per_pass);
    EXPECT_TRUE(op_partial.has_untouched_dst_bytes);
    EXPECT_LE(op_partial.min_accesses_per_pass, op_partial.max_accesses_per_pass);

    // Exact full cycle (passes == Q): all slots touched, Copy has x2 payload.
    const auto op_full = build_patterns_operation_plan(
        *entry, plan.src_seed, PatternOperation::Copy, Q, true);
    ASSERT_TRUE(op_full.valid);
    EXPECT_EQ(op_full.completed_phase_cycles, 1u);
    EXPECT_EQ(op_full.distinct_address_count, entry->buffer_bytes / 32u);
    EXPECT_EQ(op_full.logical_working_set_bytes, entry->buffer_bytes);
    EXPECT_FALSE(op_full.has_untouched_dst_bytes);
    EXPECT_EQ(op_full.exact_payload_bytes, op_full.total_accesses * 32u * 2u);

    // Multi-cycle + remainder (Q + 3 passes for Q >= 4, else Q + 1).
    const size_t rem = (Q >= 4) ? 3u : 1u;
    const auto op_multi = build_patterns_operation_plan(
        *entry, plan.src_seed, PatternOperation::Read, Q + rem, true);
    ASSERT_TRUE(op_multi.valid);
    EXPECT_EQ(op_multi.completed_phase_cycles, 1u);
    EXPECT_EQ(op_multi.distinct_address_count, entry->buffer_bytes / 32u);
    EXPECT_FALSE(op_multi.has_untouched_dst_bytes);
    EXPECT_GT(op_multi.total_accesses, op_full.total_accesses);
  }
}

TEST(PatternsWorkPlanTest, RandomWorkerDistributionAndUntouchedFlag) {
  // 1 MiB: total_slots = 32768 -> all 32768 slots selected -> no untouched bytes.
  {
    const PatternsWorkPlan plan =
        build_patterns_work_plan(make_config(1, 3), make_sys_info());
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const PatternPlanEntry* rnd = plan.find_pattern(PatternKind::Random);
    ASSERT_NE(rnd, nullptr);
    EXPECT_EQ(rnd->random_workload.selected_accesses_per_pass, 32768u);
    size_t sum = 0;
    for (const auto& sp : rnd->worker_spans) {
      EXPECT_GE(sp.random_access_count, 1u);
      EXPECT_LE(sp.random_access_count, sp.span_bytes / 32u);
      sum += sp.random_access_count;
    }
    EXPECT_EQ(sum, 32768u);

    const auto op = build_patterns_operation_plan(
        *rnd, plan.src_seed, PatternOperation::Write, 2, true);
    ASSERT_TRUE(op.valid);
    EXPECT_FALSE(op.has_untouched_dst_bytes);
    EXPECT_EQ(op.total_accesses, 2u * 32768u);
  }

  // 512 MiB: total_slots = 16,777,216 -> clamped to 1,000,000 -> untouched bytes exist.
  {
    const PatternsWorkPlan plan =
        build_patterns_work_plan(make_config(512, 8), make_sys_info(16));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const PatternPlanEntry* rnd = plan.find_pattern(PatternKind::Random);
    ASSERT_NE(rnd, nullptr);
    EXPECT_EQ(rnd->random_workload.selected_accesses_per_pass, 1000000u);
    size_t sum = 0;
    for (const auto& sp : rnd->worker_spans) {
      EXPECT_GE(sp.random_access_count, 1u);
      EXPECT_LE(sp.random_access_count, sp.span_bytes / 32u);
      sum += sp.random_access_count;
    }
    EXPECT_EQ(sum, 1000000u);

    const auto op = build_patterns_operation_plan(
        *rnd, plan.src_seed, PatternOperation::Copy, 1, true);
    ASSERT_TRUE(op.valid);
    EXPECT_TRUE(op.has_untouched_dst_bytes);
    EXPECT_EQ(op.exact_payload_bytes, 1000000u * 32u * 2u);
  }
}

TEST(PatternsWorkPlanTest, StrictMemoryBudgetAndFallbackWithoutSilentCapping) {
  auto cfg = make_config(999999999ULL, 4, 1);
  cfg.iterations = 1;
  const PatternsWorkPlan plan =
      build_patterns_work_plan(cfg, make_sys_info(16));
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason_code, PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_FALSE(plan.memory_budget.valid);
  EXPECT_EQ(plan.memory_budget.runtime_peak_bytes,
            plan.memory_budget.known_owned_peak_bytes);
  // Buffer bytes MUST NOT be silently truncated!
  EXPECT_EQ(plan.buffer_bytes, 999999999ULL * 1024ULL * 1024ULL);

  // Fallback budget when available_memory_bytes == 0.
  auto fallback_cfg = make_config(16, 2, 1);
  const PatternsWorkPlan fallback =
      build_patterns_work_plan(fallback_cfg, make_sys_info(0));
  ASSERT_TRUE(fallback.valid) << fallback.reason_code;
  EXPECT_TRUE(fallback.memory_budget.used_fallback);
  EXPECT_EQ(fallback.memory_budget.allowed_memory_bytes,
            Constants::FALLBACK_TOTAL_LIMIT_MB * Constants::BYTES_PER_MB);
  const size_t expected_idx_bytes =
      2u *
      fallback.find_pattern(PatternKind::Random)
          ->random_workload.selected_accesses_per_pass *
      sizeof(size_t);
  EXPECT_EQ(fallback.memory_budget.index_structure_bytes, expected_idx_bytes);
  EXPECT_GT(fallback.memory_budget.transient_peak_bytes, expected_idx_bytes);
}

TEST(PatternsWorkPlanTest, GuardrailsAndCapsEnforced) {
  // buffer_size_mb == 0
  {
    const auto plan =
        build_patterns_work_plan(make_config(0, 2, 1), make_sys_info());
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, PatternsWorkPlanReason::INVALID_BUFFER_SIZE);
  }

  // buffer_size_mb > 2^53 - 1
  {
    const auto plan = build_patterns_work_plan(
        make_config(Constants::LLM_JSON_MAX_SAFE_INTEGER + 1ULL, 2, 1),
        make_sys_info());
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code,
              PatternsWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE);
  }

  // iterations > 10^9
  {
    auto cfg = make_config(1, 2, 1);
    cfg.iterations = 1000000001ULL;
    const auto plan = build_patterns_work_plan(cfg, make_sys_info());
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, PatternsWorkPlanReason::INVALID_ITERATIONS);
  }

  // planned_measurements > 100,000
  {
    const auto plan =
        build_patterns_work_plan(make_config(1, 2, 20000), make_sys_info());
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code,
              PatternsWorkPlanReason::MEASUREMENT_LIMIT_EXCEEDED);
  }

  // 64 GiB payload cap for Read (65 * 1 GiB = 65 GiB) and Copy (33 * 2 * 1 GiB = 66 GiB)
  {
    auto cfg = make_config(1024, 2, 1);
    cfg.iterations = 65;
    const auto plan_read_cap =
        build_patterns_work_plan(cfg, make_sys_info(16));
    EXPECT_FALSE(plan_read_cap.valid);
    EXPECT_EQ(plan_read_cap.reason_code,
              PatternsWorkPlanReason::WORK_CAP_EXCEEDED);

    cfg.iterations = 33;  // Read/Write ok (33 GiB), Copy fails (66 GiB > 64 GiB)
    const auto plan_copy_cap =
        build_patterns_work_plan(cfg, make_sys_info(16));
    EXPECT_FALSE(plan_copy_cap.valid);
    EXPECT_EQ(plan_copy_cap.reason_code,
              PatternsWorkPlanReason::WORK_CAP_EXCEEDED);

    const auto plan_ok_ref =
        build_patterns_work_plan(make_config(1024, 2, 1), make_sys_info(16));
    ASSERT_TRUE(plan_ok_ref.valid);
    const auto* fwd = plan_ok_ref.find_pattern(PatternKind::SequentialForward);
    ASSERT_NE(fwd, nullptr);
    EXPECT_EQ(calculate_patterns_max_passes(*fwd, PatternOperation::Read), 64u);
    EXPECT_EQ(calculate_patterns_max_passes(*fwd, PatternOperation::Copy), 32u);

    const auto* s2m = plan_ok_ref.find_pattern(PatternKind::Strided2MiB);
    ASSERT_NE(s2m, nullptr);
    EXPECT_GE(calculate_patterns_max_passes(*s2m, PatternOperation::Read), 64u);
    EXPECT_LE(calculate_patterns_max_passes(*s2m, PatternOperation::Read),
              kPatternMaxPasses);
  }

  // buffer_size_mb = 1ULL << 45 (<= 2^53 - 1) overflows byte conversion cleanly.
  {
    const auto plan = build_patterns_work_plan(
        make_config(1ULL << 45, 2, 1), make_sys_info());
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, PatternsWorkPlanReason::BUFFER_BYTES_OVERFLOW);
  }

  // Exact 2 MiB (2097152 B) vs 2 MiB + 32 B (2097184 B) strided_2mb transition boundary.
  {
    const auto plan_2m =
        build_patterns_work_plan(make_config(2, 1, 1), make_sys_info());
    ASSERT_TRUE(plan_2m.valid);
    const PatternPlanEntry* s2m_2m = plan_2m.entry(PatternKind::Strided2MiB);
    ASSERT_NE(s2m_2m, nullptr);
    EXPECT_FALSE(s2m_2m->enabled);
    EXPECT_EQ(s2m_2m->status, "skipped");
    EXPECT_EQ(s2m_2m->reason_code,
              PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);
    const auto skipped_op = build_patterns_operation_plan(
        *s2m_2m, plan_2m.src_seed, PatternOperation::Read, 1, true);
    EXPECT_FALSE(skipped_op.valid);
    EXPECT_EQ(skipped_op.reason_code,
              PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL);

    // Construct exact 2 MiB + 32 B entry: valid minimum 2 accesses per pass.
    PatternPlanEntry exact_2m_plus_32 = *s2m_2m;
    exact_2m_plus_32.enabled = true;
    exact_2m_plus_32.status = PatternsWorkPlanReason::ACTIVE;
    exact_2m_plus_32.reason_code = PatternsWorkPlanReason::ACTIVE;
    exact_2m_plus_32.buffer_bytes = 2097152ULL + 32ULL;
    exact_2m_plus_32.effective_workers = 1;
    exact_2m_plus_32.accesses_per_pass = 2;
    exact_2m_plus_32.worker_spans = {{0ULL, 2097152ULL + 32ULL, 2ULL, 0ULL}};

    const auto op_1 = build_patterns_operation_plan(
        exact_2m_plus_32, plan_2m.src_seed, PatternOperation::Write, 1, true);
    ASSERT_TRUE(op_1.valid);
    EXPECT_EQ(op_1.total_accesses, 2u);
    EXPECT_EQ(op_1.min_accesses_per_pass, 2u);
    EXPECT_EQ(op_1.max_accesses_per_pass, 2u);
    EXPECT_EQ(op_1.distinct_address_count, 2u);
    EXPECT_TRUE(op_1.has_untouched_dst_bytes);

    const auto op_2 = build_patterns_operation_plan(
        exact_2m_plus_32, plan_2m.src_seed, PatternOperation::Read, 2, true);
    ASSERT_TRUE(op_2.valid);
    EXPECT_EQ(op_2.total_accesses, 3u);
    EXPECT_EQ(op_2.min_accesses_per_pass, 1u);
    EXPECT_EQ(op_2.max_accesses_per_pass, 2u);
    EXPECT_EQ(op_2.distinct_address_count, 3u);
    EXPECT_TRUE(op_2.has_untouched_dst_bytes);

    const auto op_full = build_patterns_operation_plan(
        exact_2m_plus_32, plan_2m.src_seed, PatternOperation::Read, 65536, true);
    ASSERT_TRUE(op_full.valid);
    EXPECT_EQ(op_full.completed_phase_cycles, 1u);
    EXPECT_EQ(op_full.distinct_address_count, 65537u);
    EXPECT_FALSE(op_full.has_untouched_dst_bytes);

    // Exact 2 MiB span (2097152 B) has only 1 phase-0 access (< 2, no stride transition).
    PatternPlanEntry too_small_2m = exact_2m_plus_32;
    too_small_2m.buffer_bytes = 2097152ULL;
    too_small_2m.accesses_per_pass = 1;
    too_small_2m.worker_spans = {{0ULL, 2097152ULL, 1ULL, 0ULL}};
    const auto op_too_small = build_patterns_operation_plan(
        too_small_2m, plan_2m.src_seed, PatternOperation::Read, 1, true);
    EXPECT_TRUE(op_too_small.valid);
    EXPECT_EQ(op_too_small.min_accesses_per_pass, 1u);
    EXPECT_EQ(op_too_small.max_accesses_per_pass, 1u);
  }
}

}  // namespace
