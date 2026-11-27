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
 * @file cpu_bandwidth_output.cpp
 * @brief Console help and human-readable summary output for `--cpu-bandwidth`
 */

#include "cpu_bandwidth/cpu_bandwidth_output.h"

#include <iomanip>
#include <iostream>
#include <string>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "cpu_bandwidth/cpu_bandwidth_json.h"
#include "output/console/messages/messages_api.h"

namespace {

const char* target_display_name(CpuBandwidthTarget target) noexcept {
  switch (target) {
    case CpuBandwidthTarget::MainMemory:
      return "Main Memory";
    case CpuBandwidthTarget::L1:
      return "L1 Cache";
    case CpuBandwidthTarget::L2:
      return "L2 Cache";
    case CpuBandwidthTarget::Custom:
      return "Custom Cache";
  }
  return "Unknown";
}

const char* operation_display_name(CpuBandwidthOperation op) noexcept {
  switch (op) {
    case CpuBandwidthOperation::Read:
      return "Read";
    case CpuBandwidthOperation::Write:
      return "Write";
    case CpuBandwidthOperation::Copy:
      return "Copy";
  }
  return "Unknown";
}

}  // namespace

void print_cpu_bandwidth_help(const char* prog_name, std::ostream& os) {
  const char* program = (prog_name != nullptr && prog_name[0] != '\0')
                            ? prog_name
                            : "memory_benchmark";
  os << "macOS Memory Benchmark v" << SOFTVERSION << "\n\n"
     << "Usage: " << program << " -B [options]\n"
     << "       " << program << " --cpu-bandwidth [options]\n\n"
     << "CPU Bandwidth Mode Options:\n"
     << "  -B, --cpu-bandwidth           Run standalone audited CPU memory bandwidth benchmark\n"
     << "  -b, --buffer-size <MB>        Main memory buffer size in MB (default: "
     << Constants::DEFAULT_BUFFER_SIZE_MB << ", 0 to omit)\n"
     << "  -k, --cache-size <KB>         Custom cache buffer size in KB ("
     << Constants::MIN_CACHE_SIZE_KB << "-" << Constants::MAX_CACHE_SIZE_KB
     << ", 0 to omit caches)\n"
     << "  -t, --threads <n>             Requested worker thread count (default: logical cores for main, 1 for cache)\n"
     << "  -i, --iterations <n>          Fixed pass count per operation (1-1000000000; default: auto-calibrate)\n"
     << "  -r, --count <n>               Number of benchmark loops (default: 3)\n"
     << "      --seed <uint64>           Explicit 64-bit base seed (0 allowed)\n"
     << "  -u, --non-cacheable           Best-effort madvise(MADV_RANDOM) allocation hint\n"
     << "  -o, --output <target>         Write schema-1 JSON to file, or '-' for stdout JSON\n"
     << "  -S, --sweep <key=v1,v2,...>   Parameter sweep (keys: buffer-size, cache-size, threads)\n"
     << "  -X, --sweep-max-runs <n>      Maximum generated sweep runs (default: "
     << Constants::DEFAULT_SWEEP_MAX_RUNS << ")\n"
     << "  -h, --help                    Display this help message and exit\n";
}

void print_cpu_bandwidth_configuration(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResultMetadata& metadata) {
  std::cout << "CPU Bandwidth Mode (schema v" << kCpuBandwidthJsonSchemaVersion
            << ")\n"
            << "  CPU: " << metadata.processor_name
            << " (P=" << metadata.performance_core_count
            << ", E=" << metadata.efficiency_core_count
            << ", Logical=" << metadata.logical_core_count << ")\n"
            << "  Work Policy: "
            << (config.iterations.has_value()
                    ? "explicit_fixed_work (passes=" +
                          std::to_string(*config.iterations) + ")"
                    : "automatic_calibration (target=0.150s)")
            << ", Loops: " << config.loop_count
            << ", Requested Workers: " << config.requested_workers
            << " (" << (config.user_specified_workers ? "user" : "auto") << ")\n"
            << "  Base Seed: " << config.seed
            << " (" << (config.user_specified_seed ? "user" : "generated")
            << "), Non-cacheable hint: "
            << (config.non_cacheable ? "enabled" : "disabled") << "\n";
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const auto& tplan = work_plan.targets[cpu_bandwidth_target_index(target)];
    std::cout << "  Target [" << target_display_name(target)
              << "]: " << tplan.bytes << " bytes, effective_workers="
              << tplan.effective_workers << "\n";
  }
}

void print_cpu_bandwidth_console_summary(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result) {
  std::cout << "\n--- CPU Bandwidth Results ("
            << cpu_bandwidth_run_status_to_string(result.status) << ", "
            << result.counters.measured_measurements << "/"
            << result.counters.planned_measurements << " measured) ---\n";

  const auto prev_flags = std::cout.flags();
  const auto prev_prec = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);

  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    std::cout << "  " << target_display_name(target) << ":\n";
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto op = static_cast<CpuBandwidthOperation>(op_i);
      const auto& agg = result.aggregates[t_idx][op_i];
      if (agg.value.has_value()) {
        std::cout << "    " << std::setw(5) << std::left
                  << operation_display_name(op) << ": P50=" << *agg.value
                  << " GB/s (min=" << agg.statistics.min
                  << ", max=" << agg.statistics.max;
        if (agg.statistics.coefficient_of_variation_defined) {
          std::cout << ", CV=" << std::setprecision(1)
                    << agg.statistics.coefficient_of_variation_pct << "%"
                    << std::setprecision(2);
        }
        std::cout << ", MAD=" << agg.statistics.median_absolute_deviation
                  << ", n=" << agg.statistics.sample_count << ")\n";
      } else {
        std::cout << "    " << std::setw(5) << std::left
                  << operation_display_name(op) << ": unavailable\n";
      }
    }
  }

  std::cout.flags(prev_flags);
  std::cout.precision(prev_prec);

  for (const auto& warning : collect_cpu_bandwidth_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    std::cerr << Messages::warning_prefix() << warning << std::endl;
  }
}
