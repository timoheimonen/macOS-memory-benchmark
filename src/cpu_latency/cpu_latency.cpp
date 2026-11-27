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
 * @file cpu_latency.cpp
 * @brief Standalone `-L` / `--cpu-latency` CLI, configuration, and entrypoint
 */

#include "cpu_latency/cpu_latency.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "cpu_latency/cpu_latency_chain.h"
#include "cpu_latency/cpu_latency_executor.h"
#include "cpu_latency/cpu_latency_json.h"
#include "cpu_latency/cpu_latency_output.h"
#include "cpu_latency/cpu_latency_runner.h"
#include "cpu_latency/cpu_latency_sweep_runner.h"
#include "cpu_latency/cpu_latency_work_plan.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_session.h"
#include "utils/json_utils.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

CpuLatencyParserTestHooks active_parser_hooks;
bool parser_hooks_active = false;
CpuLatencyCommandTestHooks active_command_hooks;
bool command_hooks_active = false;

enum CpuLatencyOptionId {
  OPT_CPU_LATENCY = 1,
  OPT_HELP,
  OPT_BUFFER_SIZE,
  OPT_CACHE_SIZE,
  OPT_ITERATIONS,
  OPT_COUNT,
  OPT_LATENCY_SAMPLES,
  OPT_LATENCY_STRIDE_BYTES,
  OPT_LATENCY_CHAIN_MODE,
  OPT_LATENCY_TLB_LOCALITY_KB,
  OPT_COMPARE_LOCALITY,
  OPT_SEED,
  OPT_NON_CACHEABLE,
  OPT_OUTPUT,
  OPT_SWEEP,
  OPT_SWEEP_MAX_RUNS,
};

constexpr std::array<CliOptionSpec, 16> kCpuLatencyOptionSpecs{{
    {OPT_CPU_LATENCY, "-L", "--cpu-latency", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_HELP, "-h", "--help", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_BUFFER_SIZE, "-b", "--buffer-size", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_CACHE_SIZE, "-k", "--cache-size", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ITERATIONS, "-i", "--iterations", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_COUNT, "-r", "--count", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_LATENCY_SAMPLES, "-n", "--latency-samples",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_LATENCY_STRIDE_BYTES, "-s", "--latency-stride-bytes",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_LATENCY_CHAIN_MODE, "-m", "--latency-chain-mode",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_LATENCY_TLB_LOCALITY_KB, "-l", "--latency-tlb-locality-kb",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_COMPARE_LOCALITY, nullptr, "--compare-locality",
     CliOptionValueKind::None, CliOptionArity::Single},
    {OPT_SEED, nullptr, "--seed", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_NON_CACHEABLE, "-u", "--non-cacheable", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_OUTPUT, "-o", "--output", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_SWEEP, "-S", "--sweep", CliOptionValueKind::RequiredValue,
     CliOptionArity::Repeatable},
    {OPT_SWEEP_MAX_RUNS, "-X", "--sweep-max-runs",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

const std::string& allowed_options_hint() {
  static const std::string hint =
      "-L/--cpu-latency, -b/--buffer-size, -k/--cache-size, -i/--iterations, "
      "-r/--count, -n/--latency-samples, -s/--latency-stride-bytes, "
      "-m/--latency-chain-mode, -l/--latency-tlb-locality-kb, "
      "--compare-locality, --seed, -u/--non-cacheable, -o/--output, "
      "-S/--sweep, -X/--sweep-max-runs, -h/--help";
  return hint;
}

bool parse_strict_uint64_bounded(const std::string& option,
                                 const std::string& token,
                                 uint64_t min_val,
                                 uint64_t max_val,
                                 const char* range_reason,
                                 uint64_t& out_value) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_unsigned_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option, token,
                     strict_unsigned_decimal_error_reason(status))
              << std::endl;
    return false;
  }
  if (parsed < min_val || parsed > max_val) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, token, range_reason)
              << std::endl;
    return false;
  }
  out_value = parsed;
  return true;
}

bool parse_cache_size_kb_value(const std::string& option,
                               const std::string& token,
                               size_t& out_kb) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_unsigned_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option, token,
                     strict_unsigned_decimal_error_reason(status))
              << std::endl;
    return false;
  }
  if (parsed != 0 &&
      (parsed < static_cast<uint64_t>(Constants::MIN_CACHE_SIZE_KB) ||
       parsed > static_cast<uint64_t>(Constants::MAX_CACHE_SIZE_KB))) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option, token,
                     "must be 0 or between 16 and 1048576 KB")
              << std::endl;
    return false;
  }
  out_kb = static_cast<size_t>(parsed);
  return true;
}

bool parse_stride_bytes_value(const std::string& option,
                              const std::string& token,
                              size_t& out_stride) {
  uint64_t parsed = 0;
  if (!parse_strict_uint64_bounded(
          option, token, sizeof(uintptr_t),
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
          "must be between 8 and 4294967295", parsed)) {
    return false;
  }
  if ((parsed % sizeof(uintptr_t)) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_alignment(
                     static_cast<size_t>(parsed), sizeof(uintptr_t))
              << std::endl;
    return false;
  }
  out_stride = static_cast<size_t>(parsed);
  return true;
}

bool parse_sweep_spec(const std::string& raw_spec,
                      CpuLatencySweepSpec& out_spec) {
  ParsedSweepText parsed;
  try {
    parsed = parse_sweep_text(raw_spec);
  } catch (const std::invalid_argument& err) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("--sweep", raw_spec, err.what())
              << std::endl;
    return false;
  }

  out_spec = CpuLatencySweepSpec{};
  out_spec.parameter_name = parsed.key;
  if (parsed.key == "buffer-size") {
    out_spec.parameter = CpuLatencySweepParameter::BufferSizeMb;
  } else if (parsed.key == "cache-size") {
    out_spec.parameter = CpuLatencySweepParameter::CacheSizeKb;
  } else if (parsed.key == "latency-tlb-locality-kb") {
    out_spec.parameter = CpuLatencySweepParameter::LatencyTlbLocalityKb;
  } else if (parsed.key == "latency-stride-bytes") {
    out_spec.parameter = CpuLatencySweepParameter::LatencyStrideBytes;
  } else if (parsed.key == "latency-chain-mode") {
    out_spec.parameter = CpuLatencySweepParameter::LatencyChainMode;
  } else {
    std::cerr << Messages::error_prefix()
              << Messages::error_sweep_parameter_not_allowed(
                     parsed.key, kCpuLatencyJsonModeName)
              << std::endl;
    return false;
  }

  out_spec.values.reserve(parsed.values.size());
  for (const std::string& raw_val : parsed.values) {
    CpuLatencySweepValue typed;
    typed.raw_value = raw_val;
    switch (out_spec.parameter) {
      case CpuLatencySweepParameter::BufferSizeMb: {
        uint64_t v = 0;
        if (!parse_strict_uint64_bounded(
                "--sweep", raw_val, 0, Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 0 and 9007199254740991", v)) {
          return false;
        }
        typed.integer_value = static_cast<size_t>(v);
        break;
      }
      case CpuLatencySweepParameter::CacheSizeKb: {
        size_t v = 0;
        if (!parse_cache_size_kb_value("--sweep", raw_val, v)) {
          return false;
        }
        typed.integer_value = v;
        break;
      }
      case CpuLatencySweepParameter::LatencyTlbLocalityKb: {
        uint64_t v = 0;
        if (!parse_strict_uint64_bounded(
                "--sweep", raw_val, 0, Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 0 and 9007199254740991", v)) {
          return false;
        }
        typed.integer_value = static_cast<size_t>(v);
        break;
      }
      case CpuLatencySweepParameter::LatencyStrideBytes: {
        size_t v = 0;
        if (!parse_stride_bytes_value("--sweep", raw_val, v)) {
          return false;
        }
        typed.integer_value = v;
        break;
      }
      case CpuLatencySweepParameter::LatencyChainMode: {
        LatencyChainMode m = LatencyChainMode::Auto;
        if (!latency_chain_mode_from_string(raw_val, m)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           "--sweep", raw_val,
                           "invalid latency chain mode")
                    << std::endl;
          return false;
        }
        typed.chain_mode = m;
        break;
      }
    }
    out_spec.values.push_back(std::move(typed));
  }
  return true;
}

uint64_t generate_base_seed() {
  if (parser_hooks_active) {
    const uint64_t s = active_parser_hooks.generated_seed;
    return SeedUtils::generate_seed([s]() { return s; });
  }
  return SeedUtils::generate_seed();
}

CpuLatencySystemInfo capture_system_info(CpuLatencyResultMetadata& metadata) {
  CpuLatencySystemInfo info;
  if (parser_hooks_active) {
    metadata.processor_name = "Test Apple Silicon";
    metadata.macos_version = "26.0";
    metadata.performance_core_count =
        static_cast<int>(active_parser_hooks.detected_workers);
    metadata.efficiency_core_count = 0;
    metadata.logical_core_count =
        static_cast<int>(active_parser_hooks.detected_workers);
    metadata.page_size_bytes = active_parser_hooks.page_size_bytes;
    metadata.l1_data_cache_bytes = active_parser_hooks.l1_cache_bytes;
    metadata.l2_data_cache_bytes = active_parser_hooks.l2_cache_bytes;
    if (active_parser_hooks.use_memory_override) {
      metadata.available_memory_bytes =
          active_parser_hooks.available_memory_bytes;
      metadata.available_memory_source =
          metadata.available_memory_bytes == 0
              ? "unavailable"
              : "mach-free-plus-inactive-rounded-mib";
    } else {
      const unsigned long avail_mb = get_available_memory_mb();
      if (avail_mb > 0) {
        NumericUtils::checked_multiply(static_cast<size_t>(avail_mb),
                                       Constants::BYTES_PER_MB,
                                       metadata.available_memory_bytes);
      } else {
        metadata.available_memory_source = "unavailable";
      }
    }
  } else {
    metadata.processor_name = get_processor_name();
    metadata.macos_version = get_macos_version();
    metadata.performance_core_count = get_performance_cores();
    metadata.efficiency_core_count = get_efficiency_cores();
    metadata.logical_core_count = get_total_logical_cores();
    metadata.page_size_bytes = get_system_page_size_bytes();
    metadata.l1_data_cache_bytes = get_l1_cache_size();
    metadata.l2_data_cache_bytes = get_l2_cache_size();
    const unsigned long avail_mb = get_available_memory_mb();
    if (avail_mb == 0) {
      metadata.available_memory_source = "unavailable";
    } else if (!NumericUtils::checked_multiply(static_cast<size_t>(avail_mb),
                                               Constants::BYTES_PER_MB,
                                               metadata.available_memory_bytes)) {
      metadata.available_memory_bytes = 0;
      metadata.available_memory_source = "unavailable";
    }
  }
  info.l1_cache_bytes = metadata.l1_data_cache_bytes;
  info.l2_cache_bytes = metadata.l2_data_cache_bytes;
  info.page_size_bytes = metadata.page_size_bytes;
  info.available_memory_bytes = metadata.available_memory_bytes;
  return info;
}

void populate_missing_chain_identities(const CpuLatencyWorkPlan& work_plan,
                                       CpuLatencyResourceSet& resources) {
  if (!work_plan.valid) {
    return;
  }
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t idx = cpu_latency_target_index(target);
    const CpuLatencyTargetPlan& tplan = work_plan.targets[idx];
    CpuLatencyTargetBuffer& tbuf = resources.targets[idx];
    tbuf.target = target;
    tbuf.active = true;
    if (!tbuf.chain_identity.active) {
      const CpuLatencyChainPermutation perm =
          generate_cpu_latency_chain_permutation(
              work_plan.base_seed, cpu_latency_target_domain_id(target),
              tplan.bytes, tplan.stride_bytes, tplan.effective_mode,
              tplan.tlb_locality_kb);
      if (perm.valid) {
        tbuf.chain_identity.active = true;
        tbuf.chain_identity.chain_seed = perm.chain_seed;
        tbuf.chain_identity.node_count = perm.node_count;
        tbuf.chain_identity.stride_bytes = perm.stride_bytes;
        tbuf.chain_identity.chain_mode = perm.effective_mode;
        tbuf.chain_identity.tlb_locality_kb = perm.tlb_locality_kb;
        tbuf.chain_identity.permutation_sha256 = perm.permutation_sha256;
      }
    }
  }
  if (work_plan.compare_locality) {
    if (!resources.locality_16k_identity.active &&
        work_plan.locality_16k_plan.has_value()) {
      const auto& p = *work_plan.locality_16k_plan;
      const CpuLatencyChainPermutation perm =
          generate_cpu_latency_chain_permutation(
              work_plan.base_seed, kCpuLatencyDomainLocality16k, p.bytes,
              p.stride_bytes, p.effective_mode, p.tlb_locality_kb);
      if (perm.valid) {
        resources.locality_16k_identity.active = true;
        resources.locality_16k_identity.chain_seed = perm.chain_seed;
        resources.locality_16k_identity.node_count = perm.node_count;
        resources.locality_16k_identity.stride_bytes = perm.stride_bytes;
        resources.locality_16k_identity.chain_mode = perm.effective_mode;
        resources.locality_16k_identity.tlb_locality_kb = perm.tlb_locality_kb;
        resources.locality_16k_identity.permutation_sha256 =
            perm.permutation_sha256;
      }
    }
    if (!resources.locality_global_identity.active &&
        work_plan.locality_global_plan.has_value()) {
      const auto& p = *work_plan.locality_global_plan;
      const CpuLatencyChainPermutation perm =
          generate_cpu_latency_chain_permutation(
              work_plan.base_seed, kCpuLatencyDomainLocalityGlobal, p.bytes,
              p.stride_bytes, p.effective_mode, p.tlb_locality_kb);
      if (perm.valid) {
        resources.locality_global_identity.active = true;
        resources.locality_global_identity.chain_seed = perm.chain_seed;
        resources.locality_global_identity.node_count = perm.node_count;
        resources.locality_global_identity.stride_bytes = perm.stride_bytes;
        resources.locality_global_identity.chain_mode = perm.effective_mode;
        resources.locality_global_identity.tlb_locality_kb = perm.tlb_locality_kb;
        resources.locality_global_identity.permutation_sha256 =
            perm.permutation_sha256;
      }
    }
  }
}

void report_json_boundary_failure(const std::string& raw_output,
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
  }
}

void report_command_exception(const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("cpu_latency",
                                                             details)
              << std::endl;
  } catch (...) {
  }
}

}  // namespace

void set_cpu_latency_parser_test_hooks(const CpuLatencyParserTestHooks* hooks) {
  if (hooks == nullptr) {
    active_parser_hooks = CpuLatencyParserTestHooks{};
    parser_hooks_active = false;
  } else {
    active_parser_hooks = *hooks;
    parser_hooks_active = true;
  }
}

void set_cpu_latency_command_test_hooks(
    const CpuLatencyCommandTestHooks* hooks) {
  if (hooks == nullptr) {
    active_command_hooks = CpuLatencyCommandTestHooks{};
    command_hooks_active = false;
  } else {
    active_command_hooks = *hooks;
    command_hooks_active = true;
  }
}

int parse_cpu_latency_cli(int argc, char* argv[],
                          CpuLatencyCliOptions& options) {
  options = CpuLatencyCliOptions{};
  if (argc > 0 && argv != nullptr) {
    options.argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      if (argv[i] != nullptr) {
        options.argv.emplace_back(argv[i]);
      }
    }
  }

  // Stage 1: Exact-whitelist syntax parsing.
  const CliParseResult parsed =
      parse_cli_options(argc, argv, kCpuLatencyOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix()
              << format_cli_parse_error(parsed, &allowed_options_hint())
              << std::endl;
    return EXIT_FAILURE;
  }

  // Stage 2: Value parsing & range bounds before help check.
  for (const CliParsedOption& opt : parsed.options) {
    switch (opt.id) {
      case OPT_CPU_LATENCY:
        break;
      case OPT_HELP:
        options.help_requested = true;
        break;
      case OPT_BUFFER_SIZE: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 0,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 0 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.buffer_size_mb = static_cast<size_t>(val);
        options.user_specified_buffer_size = true;
        break;
      }
      case OPT_CACHE_SIZE: {
        size_t val = 0;
        if (!parse_cache_size_kb_value(opt.canonical_name, opt.value, val)) {
          return EXIT_FAILURE;
        }
        options.custom_cache_size_kb = val;
        options.user_specified_cache_size = true;
        break;
      }
      case OPT_ITERATIONS: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                static_cast<uint64_t>(kCpuLatencyMaxCliIterations),
                "must be between 1 and 1000000000", val)) {
          return EXIT_FAILURE;
        }
        options.iterations = static_cast<size_t>(val);
        options.user_specified_iterations = true;
        break;
      }
      case OPT_COUNT: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 1 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.loop_count = static_cast<size_t>(val);
        break;
      }
      case OPT_LATENCY_SAMPLES: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 0,
                static_cast<uint64_t>(std::numeric_limits<int>::max()),
                "must be between 0 and 2147483647", val)) {
          return EXIT_FAILURE;
        }
        options.latency_samples = static_cast<size_t>(val);
        options.user_specified_latency_samples = true;
        break;
      }
      case OPT_LATENCY_STRIDE_BYTES: {
        size_t val = 0;
        if (!parse_stride_bytes_value(opt.canonical_name, opt.value, val)) {
          return EXIT_FAILURE;
        }
        options.latency_stride_bytes = val;
        options.user_specified_stride_bytes = true;
        break;
      }
      case OPT_LATENCY_CHAIN_MODE: {
        LatencyChainMode mode = LatencyChainMode::Auto;
        if (!latency_chain_mode_from_string(opt.value, mode)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_latency_chain_mode_invalid()
                    << std::endl;
          return EXIT_FAILURE;
        }
        options.latency_chain_mode = mode;
        options.user_specified_chain_mode = true;
        break;
      }
      case OPT_LATENCY_TLB_LOCALITY_KB: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 0,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 0 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.latency_tlb_locality_kb = static_cast<size_t>(val);
        options.user_specified_tlb_locality_kb = true;
        break;
      }
      case OPT_COMPARE_LOCALITY:
        options.compare_locality = true;
        break;
      case OPT_SEED: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 0,
                std::numeric_limits<uint64_t>::max(),
                "must be a valid unsigned 64-bit integer", val)) {
          return EXIT_FAILURE;
        }
        options.seed = val;
        options.user_specified_seed = true;
        break;
      }
      case OPT_NON_CACHEABLE:
        options.non_cacheable = true;
        break;
      case OPT_OUTPUT:
        options.output_file = opt.value;
        break;
      case OPT_SWEEP: {
        CpuLatencySweepSpec spec;
        if (!parse_sweep_spec(opt.value, spec)) {
          return EXIT_FAILURE;
        }
        for (const auto& existing : options.sweep_specs) {
          if (existing.parameter == spec.parameter) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_duplicate_sweep_parameter(
                             spec.parameter_name)
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        options.sweep_specs.push_back(std::move(spec));
        options.run_sweep = true;
        break;
      }
      case OPT_SWEEP_MAX_RUNS: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 1 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.sweep_max_runs = static_cast<size_t>(val);
        break;
      }
    }
  }

  // Stage 3: Isolated help (before OS queries, target omission, or file I/O).
  if (options.help_requested) {
    print_cpu_latency_help(cli_program_name(argc, argv), std::cout);
    return EXIT_SUCCESS;
  }

  // Stage 4: Sweep validation & semantic checks.
  if (options.run_sweep) {
    if (options.output_file.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_output() << std::endl;
      return EXIT_FAILURE;
    }
    for (const auto& spec : options.sweep_specs) {
      if (spec.parameter == CpuLatencySweepParameter::BufferSizeMb &&
          options.user_specified_buffer_size) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict("--buffer-size",
                                                           spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == CpuLatencySweepParameter::CacheSizeKb &&
          options.user_specified_cache_size) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict("--cache-size",
                                                           spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == CpuLatencySweepParameter::LatencyTlbLocalityKb &&
          options.user_specified_tlb_locality_kb) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(
                         "--latency-tlb-locality-kb", spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == CpuLatencySweepParameter::LatencyStrideBytes &&
          options.user_specified_stride_bytes) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(
                         "--latency-stride-bytes", spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == CpuLatencySweepParameter::LatencyChainMode &&
          options.user_specified_chain_mode) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(
                         "--latency-chain-mode", spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
    const size_t run_count =
        calculate_sweep_run_count_from_specs(options.sweep_specs);
    if (run_count == 0) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_parameter() << std::endl;
      return EXIT_FAILURE;
    }
    if (run_count > options.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(run_count,
                                                       options.sweep_max_runs)
                << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    if (options.buffer_size_mb == 0 &&
        options.custom_cache_size_kb.has_value() &&
        *options.custom_cache_size_kb == 0) {
      std::cerr << Messages::error_prefix()
                << "At least one target is required: --buffer-size 0 and --cache-size 0 cannot both omit all targets."
                << std::endl;
      return EXIT_FAILURE;
    }
    if (options.compare_locality) {
      if (options.buffer_size_mb == 0) {
        std::cerr << Messages::error_prefix()
                  << "--compare-locality requires main memory (--buffer-size > 0)."
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (options.latency_stride_bytes > (kCpuLatencyLocality16kBytes / 2)) {
        std::cerr << Messages::error_prefix()
                  << "--compare-locality requires --latency-stride-bytes <= 8192."
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
    size_t locality_bytes = 0;
    if (!NumericUtils::checked_multiply(options.latency_tlb_locality_kb,
                                        Constants::BYTES_PER_KB,
                                        locality_bytes)) {
      std::cerr << Messages::error_prefix()
                << "TLB locality size overflows addressable bytes."
                << std::endl;
      return EXIT_FAILURE;
    }
    const LatencyChainMode effective_mode =
        resolve_latency_chain_mode(options.latency_chain_mode, locality_bytes);
    if (effective_mode != LatencyChainMode::GlobalRandom) {
      if (options.latency_tlb_locality_kb == 0) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_latency_chain_mode_requires_locality(
                         latency_chain_mode_to_string(effective_mode))
                  << std::endl;
        return EXIT_FAILURE;
      }
      if ((locality_bytes / options.latency_stride_bytes) < 2) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_latency_tlb_locality_too_small_for_stride(
                         locality_bytes, options.latency_stride_bytes)
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}

int build_cpu_latency_config(const CpuLatencyCliOptions& options,
                             CpuLatencyConfig& config) {
  config = CpuLatencyConfig{};
  config.help_printed = options.help_requested;
  config.argv = options.argv;
  if (options.help_requested) {
    return EXIT_SUCCESS;
  }

  config.buffer_size_mb = options.buffer_size_mb;
  config.custom_cache_size_kb = options.custom_cache_size_kb;
  config.iterations = options.iterations;
  config.loop_count = options.loop_count;
  config.latency_samples = options.latency_samples;
  config.latency_stride_bytes = options.latency_stride_bytes;
  config.latency_chain_mode = options.latency_chain_mode;
  config.latency_tlb_locality_kb = options.latency_tlb_locality_kb;
  config.compare_locality = options.compare_locality;
  config.non_cacheable = options.non_cacheable;
  config.output_file = options.output_file;
  config.run_sweep = options.run_sweep;
  config.sweep_max_runs = options.sweep_max_runs;
  config.sweep_specs = options.sweep_specs;
  config.user_specified_buffer_size = options.user_specified_buffer_size;
  config.user_specified_cache_size = options.user_specified_cache_size;
  config.user_specified_iterations = options.user_specified_iterations;
  config.user_specified_latency_samples =
      options.user_specified_latency_samples;
  config.user_specified_stride_bytes = options.user_specified_stride_bytes;
  config.user_specified_chain_mode = options.user_specified_chain_mode;
  config.user_specified_tlb_locality_kb =
      options.user_specified_tlb_locality_kb;
  config.user_specified_seed = options.user_specified_seed;
  config.seed = options.user_specified_seed ? *options.seed : generate_base_seed();

  for (const auto& spec : config.sweep_specs) {
    switch (spec.parameter) {
      case CpuLatencySweepParameter::BufferSizeMb:
        config.swept_buffer_size = true;
        break;
      case CpuLatencySweepParameter::CacheSizeKb:
        config.swept_cache_size = true;
        break;
      case CpuLatencySweepParameter::LatencyTlbLocalityKb:
        config.swept_tlb_locality_kb = true;
        break;
      case CpuLatencySweepParameter::LatencyStrideBytes:
        config.swept_stride_bytes = true;
        break;
      case CpuLatencySweepParameter::LatencyChainMode:
        config.swept_chain_mode = true;
        break;
    }
  }

  CpuLatencyResultMetadata dummy_meta;
  const CpuLatencySystemInfo sys_info = capture_system_info(dummy_meta);
  if (sys_info.page_size_bytes == 0) {
    std::cerr << Messages::error_prefix()
              << "System page size could not be determined." << std::endl;
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    if (!preflight_cpu_latency_sweep(config, sys_info)) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }

  size_t locality_bytes = 0;
  if (!NumericUtils::checked_multiply(config.latency_tlb_locality_kb,
                                      Constants::BYTES_PER_KB,
                                      locality_bytes)) {
    std::cerr << Messages::error_prefix()
              << "TLB locality size overflows addressable bytes." << std::endl;
    return EXIT_FAILURE;
  }
  const LatencyChainMode effective_mode =
      resolve_latency_chain_mode(config.latency_chain_mode, locality_bytes);
  if (effective_mode != LatencyChainMode::GlobalRandom) {
    if ((locality_bytes % sys_info.page_size_bytes) != 0) {
      std::cerr << Messages::error_prefix()
                << Messages::error_latency_tlb_locality_page_multiple(
                       config.latency_tlb_locality_kb,
                       sys_info.page_size_bytes / Constants::BYTES_PER_KB)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int execute_cpu_latency_run(const CpuLatencyConfig& config,
                            JsonOutputSession* output_session,
                            bool install_signal_mask,
                            bool print_banner,
                            nlohmann::ordered_json* collected_json,
                            std::string* failure_reason,
                            bool* preparation_interrupted) noexcept {
  const auto stop_check = []() {
    if (command_hooks_active && active_command_hooks.stop_requested) {
      return active_command_hooks.stop_requested();
    }
    return signal_received();
  };
  const auto record_failure = [&](std::string_view reason) {
    if (failure_reason != nullptr && !reason.empty()) {
      *failure_reason = std::string(reason);
    }
  };
  if (collected_json != nullptr) {
    *collected_json = nlohmann::ordered_json();
  }
  if (failure_reason != nullptr) {
    failure_reason->clear();
  }
  if (preparation_interrupted != nullptr) {
    *preparation_interrupted = false;
  }

  int run_status = EXIT_FAILURE;
  CpuLatencyWorkPlan work_plan;
  CpuLatencyResourceSet resources;
  CpuLatencyResultMetadata metadata;
  CpuLatencyResult result;

  try {
    if (print_banner) {
      print_runtime_banner();
    }
    const std::string binary_sha256 = capture_binary_sha256();
    metadata.build_manifest =
        build_runtime_provenance_manifest_json(binary_sha256);
    metadata.timestamp = build_utc_timestamp();
    metadata.main_thread_qos =
        prepare_main_thread_benchmark_qos(MainThreadQosSetter{}, false);

    std::optional<BenchmarkSignalMaskGuard> signal_guard;
    if (install_signal_mask) {
      signal_guard.emplace();
    }

    metadata.environment_start = capture_host_environment();
    metadata.environment_end = metadata.environment_start;
    const CpuLatencySystemInfo sys_info = capture_system_info(metadata);

    work_plan = build_cpu_latency_work_plan(config, sys_info);
    if (!work_plan.valid) {
      record_failure(work_plan.reason_code);
      std::cerr << Messages::error_prefix()
                << "CPU latency work plan rejected: " << work_plan.reason_code
                << std::endl;
      initialize_cpu_latency_setup_failure_result(
          config, work_plan, CpuLatencyRunStatus::Failed, work_plan.reason_code,
          false, result);
      populate_missing_chain_identities(work_plan, resources);
      metadata.environment_end = capture_host_environment();
      if (output_session != nullptr &&
          output_session->kind() != JsonOutputKind::Disabled) {
        const nlohmann::ordered_json payload = build_cpu_latency_json(
            config, work_plan, resources, metadata, result);
        if (output_session->write_final(payload, false) != EXIT_SUCCESS) {
          release_cpu_latency_resources(resources);
          return EXIT_FAILURE;
        }
      } else if (collected_json != nullptr) {
        *collected_json = build_cpu_latency_json(config, work_plan, resources,
                                                 metadata, result);
      }
      release_cpu_latency_resources(resources);
      return EXIT_FAILURE;
    }

    if (!prepare_cpu_latency_resources(work_plan, config.non_cacheable,
                                       resources, stop_check)) {
      record_failure(resources.reason_code);
      const bool interrupted_in_prep =
          resources.reason_code ==
          CpuLatencyExecutorReason::PREPARATION_INTERRUPTED;
      if (collected_json != nullptr && interrupted_in_prep) {
        if (preparation_interrupted != nullptr) {
          *preparation_interrupted = true;
        }
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        return EXIT_SUCCESS;
      }

      if (interrupted_in_prep) {
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        initialize_cpu_latency_setup_failure_result(
            config, work_plan, CpuLatencyRunStatus::Interrupted,
            "interruption-requested", true, result);
      } else {
        std::cerr << Messages::error_prefix()
                  << "CPU latency resource preparation failed: "
                  << resources.reason_code << std::endl;
        initialize_cpu_latency_setup_failure_result(
            config, work_plan, CpuLatencyRunStatus::Failed,
            resources.reason_code, false, result);
      }
      populate_missing_chain_identities(work_plan, resources);
      metadata.environment_end = capture_host_environment();
      if (output_session != nullptr &&
          output_session->kind() != JsonOutputKind::Disabled) {
        const nlohmann::ordered_json payload = build_cpu_latency_json(
            config, work_plan, resources, metadata, result);
        if (output_session->write_final(payload, false) != EXIT_SUCCESS) {
          release_cpu_latency_resources(resources);
          return EXIT_FAILURE;
        }
      } else if (collected_json != nullptr) {
        *collected_json = build_cpu_latency_json(config, work_plan, resources,
                                                 metadata, result);
      }
      release_cpu_latency_resources(resources);
      return interrupted_in_prep ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (print_banner) {
      print_cpu_latency_configuration(config, work_plan, metadata);
    }

    CpuLatencyRunnerHooks hooks;
    hooks.stop_requested = stop_check;
    if (output_session != nullptr) {
      hooks.progress_snapshots = output_session->persists_checkpoints();
      hooks.observe_file_writes = [&]() {
        return std::make_pair(output_session->file_writer_attempts(),
                              output_session->successful_file_writes());
      };
      hooks.checkpoint = [&](const CpuLatencyResult& snapshot,
                             CpuLatencyCheckpointKind kind) {
        if (kind == CpuLatencyCheckpointKind::Progress) {
          return output_session->checkpoint([&]() {
            metadata.environment_end = capture_host_environment();
            return build_cpu_latency_json(config, work_plan, resources,
                                          metadata, snapshot);
          });
        }
        if (output_session->kind() == JsonOutputKind::File) {
          return output_session->checkpoint([&]() {
            metadata.environment_end = capture_host_environment();
            return build_cpu_latency_json(config, work_plan, resources,
                                          metadata, snapshot);
          });
        }
        return EXIT_SUCCESS;
      };
    }

    if (command_hooks_active && active_command_hooks.before_runner) {
      active_command_hooks.before_runner(work_plan);
    }

    run_status =
        run_cpu_latency_suite(config, work_plan, resources, result, hooks);
    if (command_hooks_active && active_command_hooks.after_initialized_runner) {
      active_command_hooks.after_initialized_runner();
    }

    if (output_session == nullptr ||
        output_session->kind() != JsonOutputKind::File ||
        !result.terminal_checkpoint_completed) {
      metadata.environment_end = capture_host_environment();
    }

    if (print_banner || output_session != nullptr) {
      print_cpu_latency_console_summary(config, work_plan, resources, metadata,
                                        result);
    }

    if (output_session != nullptr &&
        output_session->kind() == JsonOutputKind::File &&
        result.terminal_checkpoint_completed && !result.checkpoint_failed) {
      std::cout << Messages::msg_results_saved_to(config.output_file)
                << std::endl;
    }

    if (result.status == CpuLatencyRunStatus::Interrupted) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    } else if (run_status != EXIT_SUCCESS) {
      record_failure(result.reason_code);
      std::cerr << Messages::error_prefix() << "CPU latency run failed ("
                << result.reason_code << ")." << std::endl;
    }
  } catch (const std::exception& err) {
    record_failure("runner-exception");
    report_command_exception(err.what());
    release_cpu_latency_resources(resources);
    return EXIT_FAILURE;
  } catch (...) {
    record_failure("runner-unknown-exception");
    report_command_exception("");
    release_cpu_latency_resources(resources);
    return EXIT_FAILURE;
  }

  nlohmann::ordered_json built_json;
  bool json_built = false;
  if (collected_json != nullptr ||
      (output_session != nullptr &&
       output_session->kind() == JsonOutputKind::Stdout &&
       result.initialized)) {
    try {
      built_json = build_cpu_latency_json(config, work_plan, resources,
                                          metadata, result);
      json_built = true;
    } catch (const std::exception& err) {
      report_json_boundary_failure(
          config.output_file,
          Messages::error_json_payload_construction_failed(err.what()));
      release_cpu_latency_resources(resources);
      return EXIT_FAILURE;
    } catch (...) {
      report_json_boundary_failure(
          config.output_file,
          Messages::error_json_payload_construction_failed(""));
      release_cpu_latency_resources(resources);
      return EXIT_FAILURE;
    }
  }

  release_cpu_latency_resources(resources);

  if (collected_json != nullptr && json_built) {
    *collected_json = std::move(built_json);
  }
  if (output_session != nullptr &&
      output_session->kind() == JsonOutputKind::Stdout && json_built) {
    if (output_session->write_final(built_json) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }
  return run_status;
}

int run_cpu_latency_collect(const CpuLatencyConfig& config,
                            nlohmann::ordered_json& result_json,
                            std::string* failure_reason,
                            bool* preparation_interrupted) {
  return execute_cpu_latency_run(config, /*output_session=*/nullptr,
                                 /*install_signal_mask=*/false,
                                 /*print_banner=*/false, &result_json,
                                 failure_reason, preparation_interrupted);
}

int run_cpu_latency_mode(int argc, char* argv[]) {
  install_signal_handlers();

  CpuLatencyCliOptions options;
  if (parse_cpu_latency_cli(argc, argv, options) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (options.help_requested) {
    return EXIT_SUCCESS;
  }

  CpuLatencyConfig config;
  if (build_cpu_latency_config(options, config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  JsonOutputTarget output_target;
  try {
    output_target = make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory);
  } catch (const std::exception& err) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(err.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(std::move(output_target));
  } catch (const std::exception& err) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(err.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    try {
      const SweepExecutionResult execution =
          run_cpu_latency_sweep(config, *output_session);
      if (output_session->kind() == JsonOutputKind::Stdout &&
          !execution.output_json.empty() &&
          output_session->write_final(execution.output_json) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      return execution.exit_code;
    } catch (const std::exception& err) {
      report_command_exception(err.what());
      return EXIT_FAILURE;
    } catch (...) {
      report_command_exception("");
      return EXIT_FAILURE;
    }
  }

  return execute_cpu_latency_run(config, &*output_session,
                                 /*install_signal_mask=*/true,
                                 /*print_banner=*/true);
}
