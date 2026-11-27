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
 * @file cpu_latency_output.cpp
 * @brief Console help and human-readable summary output for `--cpu-latency`
 */

#include "cpu_latency/cpu_latency_output.h"

#include <iomanip>
#include <iostream>
#include <string>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "cpu_latency/cpu_latency_json.h"
#include "output/console/messages/messages_api.h"

namespace {

const char* target_display_name(CpuLatencyTarget target) noexcept {
  switch (target) {
    case CpuLatencyTarget::L1:
      return "L1 Cache";
    case CpuLatencyTarget::L2:
      return "L2 Cache";
    case CpuLatencyTarget::Custom:
      return "Custom Cache";
    case CpuLatencyTarget::MainMemory:
      return "Main Memory";
  }
  return "Unknown";
}

}  // namespace

void print_cpu_latency_help(const char* prog_name, std::ostream& os) {
  const char* program = (prog_name != nullptr && prog_name[0] != '\0')
                            ? prog_name
                            : "memory_benchmark";
  os << "macOS Memory Benchmark v" << SOFTVERSION << "\n\n"
     << "Usage: " << program << " -L [options]\n"
     << "       " << program << " --cpu-latency [options]\n\n"
     << "CPU Latency Mode Options:\n"
     << "  -L, --cpu-latency                   Run standalone audited CPU memory latency benchmark\n"
     << "  -b, --buffer-size <MB>              Main memory buffer size in MB (default: "
     << Constants::DEFAULT_BUFFER_SIZE_MB << ", 0 to omit)\n"
     << "  -k, --cache-size <KB>               Custom cache buffer size in KB ("
     << Constants::MIN_CACHE_SIZE_KB << "-" << Constants::MAX_CACHE_SIZE_KB
     << ", 0 to omit caches)\n"
     << "  -i, --iterations <n>                Fixed complete chain cycles (1-1000000000; default: auto-calibrate)\n"
     << "  -r, --count <n>                     Number of benchmark loops (default: 3)\n"
     << "  -n, --latency-samples <n>           Sample windows per measurement (default: "
     << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ", 0 to disable)\n"
     << "  -s, --latency-stride-bytes <B>      Pointer stride in bytes, 8-byte aligned (default: "
     << Constants::LATENCY_STRIDE_BYTES << ")\n"
     << "  -m, --latency-chain-mode <mode>     Chain mode: auto, global-random, random-box, same-random-in-box, diff-random-in-box\n"
     << "  -l, --latency-tlb-locality-kb <KB>  TLB locality window in KB, page-aligned or 0 (default: "
     << Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB << ")\n"
     << "      --compare-locality              Run paired 16 KiB random-box vs global-random comparison rounds\n"
     << "      --seed <uint64>                 Explicit 64-bit base seed (0 allowed)\n"
     << "  -u, --non-cacheable                 Best-effort madvise(MADV_RANDOM) allocation hint\n"
     << "  -o, --output <target>               Write schema-1 JSON to file, or '-' for stdout JSON\n"
     << "  -S, --sweep <key=v1,v2,...>         Parameter sweep (keys: buffer-size, cache-size, latency-tlb-locality-kb, latency-stride-bytes, latency-chain-mode)\n"
     << "  -X, --sweep-max-runs <n>            Maximum generated sweep runs (default: "
     << Constants::DEFAULT_SWEEP_MAX_RUNS << ")\n"
     << "  -h, --help                          Display this help message and exit\n";
}

void print_cpu_latency_configuration(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResultMetadata& metadata) {
  std::cout << "CPU Latency Mode (schema v" << kCpuLatencyJsonSchemaVersion
            << ")\n"
            << "  CPU: " << metadata.processor_name
            << " (P=" << metadata.performance_core_count
            << ", E=" << metadata.efficiency_core_count
            << ", Logical=" << metadata.logical_core_count
            << ", Page=" << metadata.page_size_bytes << " B)\n"
            << "  Work Policy: "
            << (config.iterations.has_value()
                    ? "explicit_fixed_work (cycles=" +
                          std::to_string(*config.iterations) + ")"
                    : "automatic_calibration (target=0.250s)")
            << ", Loops: " << config.loop_count
            << ", Samples: " << config.latency_samples << "\n"
            << "  Chain: mode="
            << latency_chain_mode_to_string(work_plan.effective_chain_mode)
            << " (requested="
            << latency_chain_mode_to_string(config.latency_chain_mode)
            << "), stride=" << config.latency_stride_bytes
            << " B, locality=" << config.latency_tlb_locality_kb
            << " KB, compare_locality="
            << (config.compare_locality ? "enabled" : "disabled") << "\n"
            << "  Base Seed: " << config.seed << " ("
            << (config.user_specified_seed ? "user" : "generated")
            << "), Non-cacheable hint: "
            << (config.non_cacheable ? "enabled" : "disabled") << "\n";
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const auto& tplan = work_plan.targets[cpu_latency_target_index(target)];
    std::cout << "  Target [" << target_display_name(target)
              << "]: " << tplan.bytes << " bytes (" << tplan.node_count
              << " nodes)\n";
  }
}

void print_cpu_latency_console_summary(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResourceSet& resources,
    const CpuLatencyResultMetadata& metadata,
    const CpuLatencyResult& result) {
  std::cout << "\n--- CPU Latency Results ("
            << cpu_latency_run_status_to_string(result.status) << ", "
            << result.counters.measured_measurements << "/"
            << result.counters.planned_measurements << " measured) ---\n";

  const auto prev_flags = std::cout.flags();
  const auto prev_prec = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);

  for (CpuLatencyTarget target : work_plan.active_targets) {
    const auto& agg = result.aggregates[cpu_latency_target_index(target)];
    if (agg.value.has_value()) {
      std::cout << "  " << std::setw(12) << std::left
                << target_display_name(target) << ": P50=" << *agg.value
                << " ns (min=" << agg.statistics.min
                << ", max=" << agg.statistics.max;
      if (agg.statistics.coefficient_of_variation_defined) {
        std::cout << ", CV=" << std::setprecision(1)
                  << agg.statistics.coefficient_of_variation_pct << "%"
                  << std::setprecision(2);
      }
      std::cout << ", MAD=" << agg.statistics.median_absolute_deviation
                << ", n=" << agg.statistics.sample_count << ")";
      if (agg.pooled_applicable && !agg.pooled_samples_ns.empty()) {
        std::cout << " | PooledSamples P50=" << agg.pooled_statistics.median
                  << " ns (P90=" << agg.pooled_statistics.p90
                  << ", P99=" << agg.pooled_statistics.p99
                  << ", n=" << agg.pooled_statistics.sample_count << ")";
      }
      std::cout << "\n";
    } else {
      std::cout << "  " << std::setw(12) << std::left
                << target_display_name(target) << ": unavailable\n";
    }
  }

  if (config.compare_locality) {
    if (result.locality_aggregate.locality_16k_latency_ns.has_value() &&
        result.locality_aggregate.global_random_latency_ns.has_value() &&
        result.locality_aggregate.locality_latency_delta_ns.has_value()) {
      std::cout << "  Locality Comparison: 16KB P50="
                << *result.locality_aggregate.locality_16k_latency_ns
                << " ns, Global P50="
                << *result.locality_aggregate.global_random_latency_ns
                << " ns, Delta="
                << *result.locality_aggregate.locality_latency_delta_ns
                << " ns\n";
    } else {
      std::cout << "  Locality Comparison: unavailable\n";
    }
  }

  std::cout.flags(prev_flags);
  std::cout.precision(prev_prec);

  for (const auto& warning : collect_cpu_latency_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    std::cerr << Messages::warning_prefix() << warning << std::endl;
  }
}
