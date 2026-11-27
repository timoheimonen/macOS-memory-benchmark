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
//

/**
 * @file main.cpp
 * @brief Main entry point for the memory benchmark application
 *
 * This file contains the main program logic that selects and dispatches memory
 * benchmark modes. Result-producing modes handle configuration parsing,
 * mode-specific buffer preparation, benchmark execution, and console/JSON
 * output here. The LLM-memory path owns its dedicated production runner and
 * schema-v1 transport behind the standalone command boundary.
 *
 * The program supports seven primary benchmark modes:
 * - CPU bandwidth: Standalone audited CPU read/write/copy measurements (-B/--cpu-bandwidth)
 * - CPU latency: Standalone audited CPU pointer-chasing latency measurements (-L/--cpu-latency)
 * - Pattern benchmarks: Access pattern-specific tests (forward, reverse, strided, random)
 * - TLB analysis: Page-native paired locality measurements and boundary analysis
 * - Core-to-core analysis: Best-effort inter-core round-trip latency measurements
 * - GPU bandwidth: Standalone Metal GPU memory read/write/copy measurements
 * - LLM memory profile: Standalone CPU/Metal synthetic decode/prefill memory workloads with contiguous or paged KV
 *
 * CPU bandwidth, CPU latency, pattern, TLB, core-to-core, and LLM-memory modes also support validated parameter sweeps.
 * GPU bandwidth is intentionally standalone and does not participate in sweeps.
 *
 * @author Timo Heimonen
 */

#include <cstdlib>  // Exit codes
#include <exception>
#include <iostream>
#include <optional>
#include <string>

#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/mode_selector.h"
#include "benchmark/core_to_core_latency.h"
#include "benchmark/sweep_runner.h"
#include "benchmark/tlb_analysis.h"
#include "cpu_bandwidth/cpu_bandwidth.h"
#include "cpu_latency/cpu_latency.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_session.h"
#include "pattern_benchmark/patterns.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "gpu_bandwidth/gpu_bandwidth.h"
#include "llm_memory/llm_memory.h"

namespace {

void set_benchmark_qos(BenchmarkConfig& config) {
  const MainThreadQosResult qos_result = prepare_main_thread_benchmark_qos();
  config.main_thread_qos_requested = qos_result.requested;
  config.main_thread_qos_applied = qos_result.applied;
  config.main_thread_qos_code = qos_result.code;
}

template <typename Fn>
int run_with_benchmark_preparation(BenchmarkConfig& config, Fn&& fn) {
  set_benchmark_qos(config);
  BenchmarkSignalMaskGuard signal_guard;
  return fn();
}

void report_json_output_boundary_failure(const std::string& raw_output,
                                         const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix();
    if (raw_output == "-") {
      std::cerr << Messages::error_json_stdout_write_failed(details);
    } else {
      std::cerr << Messages::error_file_write_failed(raw_output, details);
    }
    std::cerr << std::endl;
  } catch (...) {
    // A secondary diagnostic failure must not escape the command boundary.
  }
}

}  // namespace

/**
 * @brief Main entry point for the memory benchmark application
 *
 * This function selects a mode and parses and validates its command-line
 * arguments. For result-producing modes, it then configures system settings,
 * prepares any required buffers, executes the requested benchmark, and emits
 * human and optional JSON results. The LLM-memory path performs its own
 * preflight, full-size resource preparation, execution, console rendering,
 * and JSON transport.
 *
 * The program supports seven mutually exclusive primary modes:
 * - CPU bandwidth measurements (--cpu-bandwidth)
 * - CPU latency measurements (--cpu-latency)
 * - Pattern-specific benchmarks (--patterns)
 * - Standalone TLB analysis (--analyze-tlb)
 * - Standalone core-to-core analysis (--analyze-core2core)
 * - Standalone GPU memory bandwidth (--gpu-bandwidth)
 * - Standalone LLM decode/prefill memory profile (--llm-memory)
 * - Validated multi-configuration runs (--sweep)
 * - Multiple loop iterations for statistical analysis (--count)
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return `EXIT_SUCCESS` for help, complete execution, and the established
 *         graceful-interruption paths. Success alone does not prove that a
 *         JSON result is complete; callers must apply the mode-specific
 *         predicate documented in `documents/API.md`.
 * @return `EXIT_FAILURE` on configuration, allocation, benchmark, or output
 *         failure. An initialized mode may still emit inspectable terminal
 *         evidence before returning failure.
 *
 * @note CPU bandwidth, CPU latency, pattern, TLB, sweep, GPU, and LLM-memory
 *       entry paths make a best-effort QOS_CLASS_USER_INTERACTIVE request for
 *       the command thread. Core-to-core mode instead requests QoS independently
 *       for its workers.
 * @note All allocated buffers are automatically freed when going out of scope
 * @note Standalone modes manage their own configuration, buffers, and JSON output.
 *
 * @see parse_arguments() for TLB/general argument parsing
 * @see run_patterns_mode() for standalone pattern benchmark execution
 */
int main(int argc, char *argv[]) {
  // Install signal handlers early (before any benchmark logic)
  install_signal_handlers();

  // Bare invocation with no arguments prints help without starting timers or mode scanners.
  if (argc <= 1) {
    print_help(cli_program_name(argc, argv));
    return EXIT_SUCCESS;
  }

  PrimaryModeSelection mode_selection;
  try {
    mode_selection = select_primary_benchmark_mode(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", error.what())
              << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", "")
              << std::endl;
    return EXIT_FAILURE;
  }

  if (mode_selection.mode == PrimaryBenchmarkMode::Conflict) {
    std::cerr << Messages::error_prefix()
              << Messages::error_mutually_exclusive_modes(
                     mode_selection.selected_options[0],
                     mode_selection.selected_options[1])
              << std::endl;
    return EXIT_FAILURE;
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::CpuBandwidth) {
    return run_cpu_bandwidth_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::CpuLatency) {
    return run_cpu_latency_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::Patterns) {
    return run_patterns_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::LlmMemory) {
    return run_llm_memory_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::GpuBandwidth) {
    return run_gpu_bandwidth_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::AnalyzeCoreToCore) {
    return run_core_to_core_latency_mode(argc, argv);
  }

  // --- Parse and Validate Configuration ---
  BenchmarkConfig config;
  int parse_result = EXIT_FAILURE;
  try {
    parse_result = parse_arguments(argc, argv, config);
  } catch (const std::exception& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", error.what())
              << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", "")
              << std::endl;
    return EXIT_FAILURE;
  }
  if (parse_result != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  // If -h/--help was handled, help text was already printed — exit now.
  if (config.help_printed) {
    return EXIT_SUCCESS;
  }

  // Reject modifier-only invocations without a primary mode before opening any JSON output session.
  if (!config.analyze_tlb) {
    validate_config(config);
    return EXIT_FAILURE;
  }

  // Every general CPU command shares one target owner. Install stdout routing
  // before sweep validation and before any benchmark worker can start.
  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory));
  } catch (const std::exception& error) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(error.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    if (validate_config(config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    try {
      const SweepExecutionResult execution =
          run_sweep_mode(config, *output_session);
      if (output_session->kind() == JsonOutputKind::Stdout &&
          !execution.output_json.empty() &&
          output_session->write_final(execution.output_json) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      return execution.exit_code;
    } catch (const std::exception& error) {
      std::cerr << Messages::error_prefix()
                << Messages::error_command_execution_exception(
                       "Sweep", error.what())
                << std::endl;
    } catch (...) {
      std::cerr << Messages::error_prefix()
                << Messages::error_command_execution_exception("Sweep", "")
                << std::endl;
    }
    return EXIT_FAILURE;
  }

  if (config.analyze_tlb) {
    if (validate_config(config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    print_runtime_banner();
    return run_with_benchmark_preparation(config, [&]() {
      try {
        if (output_session->kind() == JsonOutputKind::Disabled) {
          return run_tlb_analysis(config);
        }

        nlohmann::ordered_json result_json;
        const int run_status = run_tlb_analysis_collect(config, result_json);
        if (!result_json.empty() &&
            output_session->write_final(result_json) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
        return run_status;
      } catch (const std::exception& error) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_command_execution_exception(
                         "TLB analysis", error.what())
                  << std::endl;
      } catch (...) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_command_execution_exception(
                         "TLB analysis", "")
                  << std::endl;
      }
      return EXIT_FAILURE;
    });
  }

  return EXIT_FAILURE;
}
