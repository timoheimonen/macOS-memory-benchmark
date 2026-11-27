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
 * @file patterns_output.cpp
 * @brief Console help and human-readable summary output for `-P` / `--patterns`
 */

#include "pattern_benchmark/patterns_output.h"

#include <iomanip>
#include <iostream>
#include <string>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/console/messages/messages_api.h"

namespace {

const char* pattern_display_name(PatternKind kind) noexcept {
  switch (kind) {
    case PatternKind::SequentialForward:
      return "Sequential Forward";
    case PatternKind::SequentialReverse:
      return "Sequential Reverse";
    case PatternKind::Strided64:
      return "Strided 64B";
    case PatternKind::Strided4096:
      return "Strided 4096B";
    case PatternKind::Strided16384:
      return "Strided 16384B";
    case PatternKind::Strided2MiB:
      return "Strided 2MiB";
    case PatternKind::Random:
      return "Random";
  }
  return "Unknown";
}

const char* operation_display_name(PatternOperation op) noexcept {
  switch (op) {
    case PatternOperation::Read:
      return "Read";
    case PatternOperation::Write:
      return "Write";
    case PatternOperation::Copy:
      return "Copy";
  }
  return "Unknown";
}

}  // namespace

void print_patterns_help(const char* prog_name, std::ostream& os) {
  const char* program = (prog_name != nullptr && prog_name[0] != '\0')
                            ? prog_name
                            : "memory_benchmark";
  os << "macOS Memory Benchmark v" << SOFTVERSION << "\n\n"
     << "Usage: " << program << " -P [options]\n"
     << "       " << program << " --patterns [options]\n\n"
     << "Patterns Mode Options:\n"
     << "  -P, --patterns                Run standalone audited memory access pattern benchmark\n"
     << "  -b, --buffer-size <MB>        Buffer size in MB (1-9007199254740991, default: "
     << Constants::DEFAULT_BUFFER_SIZE_MB << ")\n"
     << "  -t, --threads <n>             Requested worker thread count (default: logical cores)\n"
     << "  -i, --iterations <n>          Fixed pass count per operation (1-"
     << Constants::PATTERN_CALIBRATION_MAX_PASSES << "; default: auto-calibrate)\n"
     << "  -r, --count <n>               Number of benchmark loops (default: 3)\n"
     << "      --seed <uint64>           Explicit 64-bit base seed (0 allowed)\n"
     << "  -u, --non-cacheable           Best-effort madvise(MADV_RANDOM) allocation hint\n"
     << "  -o, --output <target>         Write schema-1 JSON to file, or '-' for stdout JSON\n"
     << "  -S, --sweep <key=v1,v2,...>   Parameter sweep (keys: buffer-size, threads)\n"
     << "  -X, --sweep-max-runs <n>      Maximum generated sweep runs (default: "
     << Constants::DEFAULT_SWEEP_MAX_RUNS << ")\n"
     << "  -h, --help                    Display this help message and exit\n";
}

void print_patterns_configuration(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResultMetadata& metadata) {
  std::cout << "Patterns Mode (schema v" << kPatternsSchemaVersion << ")\n"
            << "  CPU: " << metadata.processor_name
            << " (P=" << metadata.performance_core_count
            << ", E=" << metadata.efficiency_core_count
            << ", Logical=" << metadata.logical_core_count << ")\n"
            << "  Buffer: " << config.buffer_size_mb << " MiB ("
            << work_plan.buffer_bytes << " bytes)\n"
            << "  Work Policy: "
            << (config.iterations.has_value()
                    ? "explicit_fixed_work (passes=" +
                          std::to_string(*config.iterations) + ")"
                    : "automatic_calibration (target=0.150s)")
            << ", Loops: " << config.loop_count
            << ", Requested Workers: " << config.requested_workers
            << " (" << (config.user_specified_workers ? "user" : "auto") << ")\n"
            << "  Base Seed: " << config.seed
            << " (" << (metadata.user_specified_seed ? "user" : "generated")
            << "), Non-cacheable hint: "
            << (config.non_cacheable ? "enabled" : "disabled") << "\n";

  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    const auto& pplan = work_plan.patterns[p_idx];
    std::cout << "  Pattern [" << pattern_display_name(kind) << "]: ";
    if (!pplan.enabled) {
      std::cout << "skipped (" << pplan.reason_code << ")\n";
      continue;
    }
    std::cout << "active, effective_workers=" << pplan.effective_workers;
    if (patterns_kind_is_strided(kind)) {
      std::cout << ", stride=" << pplan.stride_bytes
                << "B, phase_period_passes=" << pplan.phase_period_passes;
    } else if (kind == PatternKind::Random) {
      std::cout << ", selected_accesses_per_pass="
                << pplan.random_workload.selected_accesses_per_pass;
    }
    std::cout << "\n";
  }
}

void print_patterns_console_summary(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result) {
  std::cout << "\n--- Patterns Benchmark Results ("
            << patterns_run_status_to_string(result.status) << ", "
            << result.counters.measured_measurements << "/"
            << result.counters.planned_measurements << " measured) ---\n";

  const auto prev_flags = std::cout.flags();
  const auto prev_prec = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);

  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    std::cout << "  " << pattern_display_name(kind) << ":\n";
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto op = static_cast<PatternOperation>(op_i);
      const auto& agg = result.aggregates[p_idx][op_i];
      std::cout << "    " << std::setw(5) << std::left
                << operation_display_name(op) << ": ";
      if (agg.status == PatternsAggregateStatus::Skipped) {
        std::cout << "skipped (" << agg.reason_code << ")\n";
      } else if (agg.status == PatternsAggregateStatus::Measured &&
                 agg.value.has_value()) {
        std::cout << "P50=" << *agg.value << " GB/s";
        if (agg.relative_to_forward_pct.has_value()) {
          std::cout << " (" << std::setprecision(1)
                    << *agg.relative_to_forward_pct << "% vs forward)"
                    << std::setprecision(2);
        }
        std::cout << " (min=" << agg.statistics.min
                  << ", max=" << agg.statistics.max;
        if (agg.statistics.coefficient_of_variation_defined) {
          std::cout << ", CV=" << std::setprecision(1)
                    << agg.statistics.coefficient_of_variation_pct << "%"
                    << std::setprecision(2);
        }
        std::cout << ", MAD=" << agg.statistics.median_absolute_deviation
                  << ", n=" << agg.statistics.sample_count << ")\n";
      } else {
        std::cout << "unavailable (" << agg.reason_code << ")\n";
      }
    }
  }

  std::cout.flags(prev_flags);
  std::cout.precision(prev_prec);

  for (const auto& warning : collect_patterns_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    std::cerr << Messages::warning_prefix() << warning << std::endl;
  }
}
