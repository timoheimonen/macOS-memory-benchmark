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
#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

namespace {

struct MessageCase {
  const char* name;
  std::string actual;
  std::string expected;
};

void expect_exact_messages(const std::vector<MessageCase>& cases) {
  for (const MessageCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(test_case.actual, test_case.expected);
  }
}

void expect_no_model_specific_llm_metal_status(const std::string& usage) {
  constexpr const char* forbidden_phrases[] = {
      "Apple7/M1", "M4 evidence", "M5", "validation remains pending",
      "baseline smoke", "cross-family", "device matrix"};
  for (const char* phrase : forbidden_phrases) {
    SCOPED_TRACE(phrase);
    EXPECT_EQ(usage.find(phrase), std::string::npos);
  }
}

}  // namespace

TEST(MessagesFormattingTest, LinearHelpersHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"missing iterations", Messages::error_missing_value("--iterations"), "Missing value for --iterations"},
      {"missing buffer size", Messages::error_missing_value("--buffer-size"), "Missing value for --buffer-size"},
      {"missing count", Messages::error_missing_value("--count"), "Missing value for --count"},
      {"missing cache size", Messages::error_missing_value("--cache-size"), "Missing value for --cache-size"},
      {"unknown short", Messages::error_unknown_option("-unknown"), "Unknown option: -unknown"},
      {"unknown long", Messages::error_unknown_option("--invalid"), "Unknown option: --invalid"},
      {"unknown flag", Messages::error_unknown_option("-bad-flag"), "Unknown option: -bad-flag"},
      {"invalid iterations", Messages::error_invalid_value("--iterations", "abc", "must be a number"),
       "Invalid value for --iterations: abc (must be a number)"},
      {"invalid cache size", Messages::error_invalid_value("--cache-size", "-1", "must be positive"),
       "Invalid value for --cache-size: -1 (must be positive)"},
      {"invalid buffer size", Messages::error_invalid_value("--buffer-size", "0", "must be greater than zero"),
       "Invalid value for --buffer-size: 0 (must be greater than zero)"},
      {"mmap source", Messages::error_mmap_failed("src_buffer"), "mmap failed for src_buffer"},
      {"mmap destination", Messages::error_mmap_failed("dst_buffer"), "mmap failed for dst_buffer"},
      {"mmap latency", Messages::error_mmap_failed("lat_buffer"), "mmap failed for lat_buffer"},
      {"benchmark loop 1", Messages::error_benchmark_loop(1, "memory error"),
       "Error during benchmark loop 1: memory error"},
      {"benchmark loop 5", Messages::error_benchmark_loop(5, "timeout"), "Error during benchmark loop 5: timeout"},
      {"benchmark loop 10", Messages::error_benchmark_loop(10, "allocation failed"),
       "Error during benchmark loop 10: allocation failed"},
      {"qos code 1", Messages::warning_qos_failed(1), "Failed to set QoS class for main thread (code: 1)"},
      {"qos code 42", Messages::warning_qos_failed(42), "Failed to set QoS class for main thread (code: 42)"},
      {"qos code 100", Messages::warning_qos_failed(100), "Failed to set QoS class for main thread (code: 100)"},
      {"qos code -1", Messages::warning_qos_failed(-1), "Failed to set QoS class for main thread (code: -1)"},
  };

  expect_exact_messages(cases);
}

// ============================================================================
// Error Messages Tests
// ============================================================================

TEST(MessagesErrorTest, NumericValidationErrorsHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"buffer calculation", Messages::error_buffer_size_calculation(1024), "Buffer size calculation error (1024 MB)."},
      {"buffer too small", Messages::error_buffer_size_too_small(1024), "Final buffer size (1024 bytes) is too small."},
      {"cache size", Messages::error_cache_size_invalid(16, 524288, 512),
       "cache-size invalid (must be between 16 KB and 524288 KB (512 MB))"},
      {"iterations negative", Messages::error_iterations_invalid(-5, 1, 2147483647),
       "iterations invalid (must be between 1 and 2147483647, got -5)"},
      {"iterations zero", Messages::error_iterations_invalid(0, 1, 100),
       "iterations invalid (must be between 1 and 100, got 0)"},
      {"buffer size negative", Messages::error_buffersize_invalid(-100, 18446744073709551615UL),
       "buffer-size invalid (must be >= 0 and <= 18446744073709551615, got -100)"},
      {"buffer size zero", Messages::error_buffersize_invalid(0, 1000),
       "buffer-size invalid (must be >= 0 and <= 1000, got 0)"},
      {"count zero", Messages::error_count_invalid(0, 1, 2147483647),
       "count invalid (must be between 1 and 2147483647, got 0)"},
      {"count negative", Messages::error_count_invalid(-10, 1, 1000),
       "count invalid (must be between 1 and 1000, got -10)"},
      {"latency samples zero", Messages::error_latency_samples_invalid(0, 1, 2147483647),
       "latency-samples invalid (must be between 1 and 2147483647, got 0)"},
      {"latency samples negative", Messages::error_latency_samples_invalid(-1, 1, 5000),
       "latency-samples invalid (must be between 1 and 5000, got -1)"},
      {"latency stride range", Messages::error_latency_stride_invalid(0, 1, 9223372036854775807LL),
       "latency-stride-bytes invalid (must be between 1 and 9223372036854775807, got 0)"},
      {"latency stride alignment", Messages::error_latency_stride_alignment(65, 8),
       "latency-stride-bytes must be a multiple of 8 bytes, got 65"},
      {"latency locality range", Messages::error_latency_tlb_locality_invalid(-1, 1024),
       "latency-tlb-locality-kb invalid (must be >= 0 and <= 1024, got -1)"},
      {"latency locality page multiple", Messages::error_latency_tlb_locality_page_multiple(10, 16),
       "latency-tlb-locality-kb must be a multiple of system page size (16 KB), got 10 KB"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesErrorTest, JsonStdoutWriteFailureHasExactOutput) {
  EXPECT_EQ(Messages::json_stdout_reason_stream_unavailable(),
            "stdout stream is unavailable");
  EXPECT_EQ(Messages::json_stdout_reason_size_limit_exceeded(),
            "serialized JSON exceeds stream size limits");
  EXPECT_EQ(Messages::json_stdout_reason_write_failed(),
            "write operation failed");
  EXPECT_EQ(Messages::json_stdout_reason_flush_failed(),
            "flush operation failed");
  EXPECT_EQ(Messages::error_json_stdout_write_failed(
                Messages::json_stdout_reason_flush_failed()),
            "Failed to write JSON to stdout: flush operation failed");
  EXPECT_EQ(Messages::error_json_stdout_write_failed(""),
            "Failed to write JSON to stdout: unknown exception");
}

TEST(MessagesErrorTest, JsonCommandBoundaryFailuresHaveExactOutput) {
  EXPECT_EQ(
      Messages::error_json_output_initialization_failed("cwd unavailable"),
      "JSON output initialization failed: cwd unavailable");
  EXPECT_EQ(Messages::error_json_output_initialization_failed(""),
            "JSON output initialization failed: unknown exception");
  EXPECT_EQ(
      Messages::error_json_payload_construction_failed("allocation failed"),
      "JSON payload construction failed: allocation failed");
  EXPECT_EQ(Messages::error_json_payload_construction_failed(""),
            "JSON payload construction failed: unknown exception");
  EXPECT_EQ(
      Messages::error_command_execution_exception("TLB analysis",
                                                  "allocation failed"),
      "TLB analysis failed with unexpected exception: allocation failed");
  EXPECT_EQ(
      Messages::error_command_execution_exception("Core-to-core analysis", ""),
      "Core-to-core analysis failed with unexpected exception: unknown exception");
  EXPECT_EQ(
      Messages::error_sweep_nested_run_exception("allocator failed"),
      "Sweep nested run failed with unexpected exception: allocator failed");
  EXPECT_EQ(
      Messages::error_sweep_nested_run_exception(""),
      "Sweep nested run failed with unexpected exception: unknown exception");
}

TEST(MessagesErrorTest, ErrorLatencyChainModeInvalid) {
  std::string msg = Messages::error_latency_chain_mode_invalid();
  EXPECT_NE(msg.find("latency-chain-mode invalid"), std::string::npos);
  EXPECT_NE(msg.find("random-box"), std::string::npos);
}

TEST(MessagesErrorTest, ErrorLatencyChainModeRequiresLocality) {
  const std::string mode = "same-random-in-box-increasing-box";
  EXPECT_EQ(Messages::error_latency_chain_mode_requires_locality(mode),
            "latency-chain-mode 'same-random-in-box-increasing-box' requires --latency-tlb-locality-kb > 0");
}

TEST(MessagesErrorTest, ErrorAnalyzeTlbGlobalRandomUnsupported) {
  const std::string& msg = Messages::error_analyze_tlb_global_random_unsupported();
  EXPECT_NE(msg.find("--analyze-tlb"), std::string::npos);
  EXPECT_NE(msg.find("global-random"), std::string::npos);
}

TEST(MessagesErrorTest, ErrorLatencyTlbLocalityTooSmallForStride) {
  std::string msg = Messages::error_latency_tlb_locality_too_small_for_stride(4096, 4096);
  EXPECT_NE(msg.find("latency-tlb-locality-kb too small for latency-stride-bytes"), std::string::npos);
  EXPECT_NE(msg.find("requires at least 8192 bytes"), std::string::npos);
}

TEST(MessagesErrorTest, ErrorAnalyzeTlbMustBeUsedAlone) {
  const std::string& msg = Messages::error_analyze_tlb_must_be_used_alone();
  EXPECT_NE(msg.find("--analyze-tlb"), std::string::npos);
  EXPECT_NE(msg.find("--output <target>"), std::string::npos);
  EXPECT_NE(msg.find("--latency-stride-bytes"), std::string::npos);
  EXPECT_NE(msg.find("--latency-chain-mode"), std::string::npos);
  EXPECT_NE(msg.find("--tlb-density"), std::string::npos);
  EXPECT_NE(msg.find("--seed"), std::string::npos);
  EXPECT_NE(msg.find("-S/--sweep <key=...>"), std::string::npos);
  EXPECT_NE(msg.find("-X/--sweep-max-runs <count>"), std::string::npos);
}

TEST(MessagesErrorTest, ErrorSeedRequiresEverySupportedMode) {
  EXPECT_EQ(Messages::error_seed_requires_supported_mode(),
            "--seed requires --benchmark, --patterns, --analyze-tlb, "
            "--gpu-bandwidth, or --llm-memory");
}

TEST(MessagesTest, LlmMemoryCliMessagesHaveExactOutput) {
  const std::string expected_usage =
      "Usage: memory_benchmark --llm-memory [options]\n"
      "Options for standalone CPU/Metal synthetic LLM memory mode:\n"
      "  -M, --llm-memory       Select the memory-only LLM profile.\n"
      "      --llm-memory-backend <cpu|metal>\n"
      "                          Execution backend (default: cpu). The experimental Metal preview\n"
      "                          accepts decode with contiguous or paged KV and contiguous prefill;\n"
      "                          no fallback is performed.\n"
      "      --phase <decode|prefill>\n"
      "                          Workload phase (default: decode). CPU supports both phases;\n"
      "                          the Metal preview accepts decode and contiguous prefill.\n"
      "      --weight-size-mb <MiB>\n"
      "                          Required active weight bytes per work unit, in MiB.\n"
      "      --layers <count>    Required transformer layer count.\n"
      "      --query-heads <count>\n"
      "                          Required query-head count; must be at least as large as KV heads\n"
      "                          and divisible by them.\n"
      "      --kv-heads <count> Required physical KV-head count.\n"
      "      --head-dim <count> Required elements per K or V head vector.\n"
      "      --kv-element-bytes <1|2|4>\n"
      "                          KV element width (default: " +
      std::to_string(Constants::LLM_DEFAULT_KV_ELEMENT_BYTES) +
      " bytes).\n"
      "      --context-tokens <count>\n"
      "                          Required only for decode; fixed visible context including\n"
      "                          the current token. Rejected for prefill.\n"
      "      --prompt-tokens <count>\n"
      "                          Required only for prefill; full prompt length P, P >= 1.\n"
      "                          Rejected for decode.\n"
      "      --attention-query-tile-tokens <count>\n"
      "                          Required only for prefill; query tile Q, 1 <= Q <= P.\n"
      "                          Rejected for decode.\n"
      "      --kv-layout <contiguous|paged>\n"
      "                          KV storage layout (default: contiguous). CPU supports both layouts;\n"
      "                          Metal supports both for decode and contiguous for prefill.\n"
      "      --kv-block-tokens <count>\n"
      "                          Required only for paged KV; must be a positive power of two\n"
      "                          no greater than UINT32_MAX; it may exceed the phase sequence length.\n"
      "                          Rejected for contiguous KV.\n"
      "      --batch-size <count>\n"
      "                          Batch sequences per work unit (default: " +
      std::to_string(Constants::LLM_DEFAULT_BATCH_SIZE) +
      ").\n"
      "  -t, --threads <count>  Requested CPU workers; detected workers are used when omitted.\n"
      "                          Rejected for Metal.\n"
      "  -i, --iterations <count>\n"
      "                          Exact work units per scenario measurement. A work unit is one\n"
      "                          decode step or full-prompt prefill operation. When omitted, each\n"
      "                          scenario calibrates toward 150 ms in a\n"
      "                          100-250 ms window.\n"
      "  -r, --count <count>    Cyclic weights/KV/mixed loops (default: " +
      std::to_string(Constants::LLM_DEFAULT_LOOP_COUNT) +
      ").\n"
      "      --seed <uint64>    Reproducible base seed; generated once when omitted.\n"
      "  -o, --output <target>  JSON schema 1 target; exact - writes one final document to\n"
      "                          stdout and routes human output to stderr. Every other non-empty\n"
      "                          target is a file with atomic scenario and terminal checkpoints.\n"
      "                          An empty value disables JSON for this direct command.\n"
      "  -h, --help             Show this LLM-mode help and exit.\n"
      "This profile models CPU or Metal memory traffic only: it performs no Transformer math and\n"
      "does not report inference tokens/s. Effective model payload is not physical DRAM traffic.\n";

  const std::vector<MessageCase> cases = {
      {"mode isolation", Messages::error_llm_memory_must_be_used_alone(),
       "--llm-memory requires --weight-size-mb <MiB>, --layers <count>, "
       "--query-heads <count>, --kv-heads <count>, --head-dim <count>, and "
       "phase-specific token geometry; it allows only optional "
       "--llm-memory-backend <cpu|metal>, --phase <decode|prefill>, "
       "--context-tokens <count>, "
       "--prompt-tokens <count>, --attention-query-tile-tokens <count>, "
       "--kv-element-bytes <1|2|4>, --kv-layout <contiguous|paged>, "
       "--kv-block-tokens <count>, --batch-size <count>, "
       "-t/--threads <count>, -i/--iterations <count>, "
       "-r/--count <count>, --seed <uint64>, -o/--output <target>, and "
       "-h/--help (no other options allowed)"},
      {"required option",
       Messages::error_llm_memory_missing_required_option("--layers"),
       "Missing required --llm-memory option: --layers"},
      {"config reason",
       Messages::error_llm_memory_config_invalid(
           "query-heads-not-divisible-by-kv-heads"),
       "Invalid --llm-memory configuration "
       "(reason_code=query-heads-not-divisible-by-kv-heads)"},
      {"iteration limit",
       Messages::error_llm_memory_iterations_exceed_limit(5, 4),
       "LLM memory iterations exceed the exact-work guardrail "
       "(requested 5, maximum 4)"},
      {"runtime failure",
       Messages::error_llm_memory_run_failed("checksum-mismatch"),
       "Synthetic LLM memory profile failed "
       "(reason_code=checksum-mismatch)"},
      {"paged table protection",
       Messages::error_llm_paged_table_protection_failed(),
       "Failed to make the paged KV block table read-only"},
      {"positive integer", Messages::llm_memory_reason_positive_integer(),
       "must be a positive integer"},
      {"backend", Messages::llm_memory_reason_backend(),
       "must be exactly cpu or metal"},
      {"KV width", Messages::llm_memory_reason_kv_element_bytes(),
       "must be exactly 1, 2, or 4"},
      {"phase", Messages::llm_memory_reason_phase(),
       "must be exactly decode or prefill"},
      {"KV layout", Messages::llm_memory_reason_kv_layout(),
       "must be exactly contiguous or paged"},
      {"platform size", Messages::llm_memory_reason_platform_size_range(),
       "out of range for a platform size"},
      {"usage", Messages::llm_memory_usage_options("memory_benchmark"),
       expected_usage},
      {"command name", Messages::llm_memory_command_name(),
       "LLM memory profile"},
      {"report header",
       Messages::report_llm_memory_header(
           "cpu", "decode", "decode_step", "contiguous"),
       "Synthetic LLM memory profile (backend=cpu, phase=decode, "
       "work_unit=decode_step, kv_layout=contiguous, warm/cacheable)"},
      {"Metal backend",
       Messages::report_llm_memory_metal_backend(
           "Fake Apple GPU", std::numeric_limits<uint64_t>::max(), true,
           false, true, 4294967296ULL, 3221225472ULL),
       "  Metal device: name=Fake Apple GPU, "
       "registry_id=18446744073709551615\n"
       "  Metal capabilities: apple7=true, unified=false, tier2=true\n"
       "  Metal limits: max_buffer_length=4294967296 bytes, "
       "recommended_working_set=3221225472 bytes"},
      {"Metal resources",
       Messages::report_llm_memory_metal_resources(
           2, 3, 4, 268435456, 8192, 805306368, 1073741824,
           2147483648ULL),
       "  Metal segments: weights=2, K=3, V=4, capacity=268435456 bytes\n"
       "  Metal argument buffer: encoded_length=8192 bytes\n"
       "  Metal memory: committed=805306368 bytes, "
       "known_peak=1073741824 bytes, admitted_budget=2147483648 bytes"},
      {"Metal valid task",
       Messages::report_llm_memory_metal_task(
           "mixed", "llm_decode_contiguous_v1", 8, 64, true, true,
           0.00125, true, true, true, true, true, true, true, true),
       "  Metal task: scenario=mixed, pipeline=llm_decode_contiguous_v1, "
       "threadgroups=8, threads_per_threadgroup=64\n"
       "  Metal timing: gpu_elapsed_seconds=0.001250000\n"
       "  Metal validation: checksum=valid, kv_write=valid, canary=valid"},
      {"Metal invalid task without canary",
       Messages::report_llm_memory_metal_task(
           "kv_only", "llm_decode_contiguous_v1", 1, 32, true, true, 0.5,
           true, false, true, true, false, false, false, true),
       "  Metal task: scenario=kv_only, pipeline=llm_decode_contiguous_v1, "
       "threadgroups=1, threads_per_threadgroup=32\n"
       "  Metal timing: gpu_elapsed_seconds=0.500000000\n"
       "  Metal validation: checksum=invalid, kv_write=invalid, "
       "canary=not-applicable"},
      {"Metal weights task without KV write",
       Messages::report_llm_memory_metal_task(
           "weights_only", "llm_decode_contiguous_v1", 1, 32, true, true,
           0.25, false, false, false, false, true, false, false, false),
       "  Metal task: scenario=weights_only, "
       "pipeline=llm_decode_contiguous_v1, threadgroups=1, "
       "threads_per_threadgroup=32\n"
       "  Metal timing: gpu_elapsed_seconds=0.250000000\n"
       "  Metal validation: checksum=not-evaluated, "
       "kv_write=not-applicable, canary=not-applicable"},
      {"Metal invalid timing has no numeric observation",
       Messages::report_llm_memory_metal_task(
           "mixed", "llm_decode_contiguous_v1", 1, 32, true, false, 0.0,
           false, false, true, false, false, false, false, false),
       "  Metal task: scenario=mixed, pipeline=llm_decode_contiguous_v1, "
       "threadgroups=1, threads_per_threadgroup=32\n"
       "  Metal timing: gpu_elapsed_seconds=invalid\n"
       "  Metal validation: checksum=not-evaluated, kv_write=not-evaluated, "
       "canary=not-applicable"},
      {"decode unit",
       Messages::report_llm_memory_work_unit_name("decode_step", false),
       "decode step"},
      {"decode units",
       Messages::report_llm_memory_work_unit_name("decode_step", true),
       "decode steps"},
      {"prefill unit",
       Messages::report_llm_memory_work_unit_name("prefill_operation", false),
       "prefill operation"},
      {"prefill units",
       Messages::report_llm_memory_work_unit_name("prefill_operation", true),
       "prefill operations"},
      {"unknown unit",
       Messages::report_llm_memory_work_unit_name("future", false),
       "work unit"},
      {"unknown units",
       Messages::report_llm_memory_work_unit_name("future", true),
       "work units"},
      {"payload",
       Messages::report_llm_memory_payload("decode step", 1024, 768, 256),
       "  Active weight bytes / decode step: 1024\n"
       "  KV read bytes / decode step:       768\n"
       "  KV write bytes / decode step:      256"},
      {"decode geometry", Messages::report_llm_memory_decode_geometry(8, 4.0),
       "  Visible context tokens:     8\n"
       "  Traffic crossover:          4.00 visible context tokens"},
      {"prefill geometry",
       Messages::report_llm_memory_prefill_geometry(
           5, 2, 3, 11, 15, 60, 480),
       "  Prompt tokens (P):                 5\n"
       "  Attention query tile tokens (Q):  2\n"
       "  Attention query tiles (C):        3\n"
       "  Prefix token visits / sequence:   11\n"
       "  Causal token pairs / sequence:    15\n"
       "  Logical attention pairs:          60\n"
       "  Logical attention FMA terms:      480"},
      {"paged layout",
       Messages::report_llm_memory_paged_layout(
           {4, 2, 2, 4, 128, 4, 128, 384, 512, 128, 384, 512, 128,
            2, 8, 4096, "permutation-v1", 99, "0123456789abcdef",
            "paged-permutation-identity", 20, 80, 1104, "decode step"}),
       "  Paged KV block tokens (G): 4\n"
       "  Blocks per sequence (N):   2\n"
       "  Physical blocks/layer (P_b): 2\n"
       "  Physical block geometry: total_blocks=4, block_bytes=128\n"
       "  Terminal block: tokens=4, valid_bytes=128\n"
       "  K bytes (logical/physical/padding): 384/512/128\n"
       "  V bytes (logical/physical/padding): 384/512/128\n"
       "  Block table: 2 uint32 entries, 8 bytes, 4096 page-rounded bytes\n"
       "  Permutation: version=permutation-v1, seed=99, "
       "sha256=0123456789abcdef\n"
       "  Permutation identity: paged-permutation-identity\n"
       "  Timed block-table metadata / KV-active decode step: 20 lookups, 80 bytes\n"
       "  Accounted bytes / KV-active decode step: 1104\n"
       "  Effective model payload excludes timed block-table metadata bytes."},
      {"weights headline",
       Messages::report_llm_memory_scenario_headline(
           "Weights only", "decode step", "decode steps", 1.25, 800.0,
           100.5, false),
       "  Weights only: 1.250 ms/decode step, "
       "100.50 GB/s effective model payload"},
      {"mixed headline",
       Messages::report_llm_memory_scenario_headline(
           "Mixed", "decode step", "decode steps", 3.75, 266.666, 80.25,
           true),
       "  Mixed:        3.750 ms/decode step, 266.67 synthetic decode "
       "steps/s, 80.25 GB/s effective model payload"},
      {"weights label",
       Messages::report_llm_memory_scenario_name("weights_only"),
       "Weights only"},
      {"KV label", Messages::report_llm_memory_scenario_name("kv_only"),
       "KV only"},
      {"mixed label", Messages::report_llm_memory_scenario_name("mixed"),
       "Mixed"},
      {"unknown label", Messages::report_llm_memory_scenario_name("future"),
       "Unknown"},
      {"interpretation",
       Messages::report_llm_memory_interpretation_note(
           "decode", "contiguous", "decode step"),
       "  Interpretation: each decode step is synthetic memory-only work, not an inference "
       "token; effective model payload is logical, not physical DRAM-counter traffic.\n"
       "  Phase/layout: phase=decode, kv_layout=contiguous; the visible context includes the "
       "current-token slot; KV uses layer/batch/token/head/dimension order.\n"
       "  Crossover: logical weight/KV-read payload equality is not a proven hardware "
       "bottleneck transition.\n"
       "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
       "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence."},
      {"prefill interpretation",
       Messages::report_llm_memory_interpretation_note("prefill", "contiguous", "prefill operation"),
       "  Interpretation: each prefill operation is synthetic memory-only work, not an inference "
       "token; effective model payload is logical, not physical DRAM-counter traffic.\n"
       "  Phase/layout: phase=prefill, kv_layout=contiguous; prefill performs no Transformer "
       "compute and does not predict TTFT; KV uses layer/batch/token/head/dimension order.\n"
       "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
       "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence."},
      {"high CV", Messages::warning_llm_memory_high_cv("kv_only", 6.25, 5.0),
       "LLM KV only repeatability CV 6.25% exceeds 5.00%"},
      {"unbalanced order", Messages::warning_llm_memory_order_not_balanced(),
       "LLM scenario order is not fully balanced across completed loops"},
      {"duration",
       Messages::warning_llm_memory_duration_quality(
           "mixed", "above-target-single-work-unit"),
       "LLM Mixed duration quality is above-target-single-work-unit"},
      {"environment", Messages::warning_llm_memory_environment_not_nominal(),
       "LLM result environment is not reference-eligible (thermal state or Low Power Mode)"},
      {"main QoS",
       Messages::warning_llm_memory_main_thread_qos_not_applied(7),
       "LLM main-thread QoS request was not applied (code: 7)"},
      {"worker QoS", Messages::warning_llm_memory_worker_qos_not_applied(),
       "One or more LLM worker QoS requests were not applied"},
      {"cache",
       Messages::warning_llm_memory_weight_cache_dominant(1024, 4096),
       "LLM weight working set (1024 bytes) does not exceed reported L2 cache "
       "(4096 bytes); the result may be cache-dominant"},
      {"KV cache", Messages::warning_llm_memory_kv_cache_dominant(2048, 4096),
       "LLM KV working set (2048 bytes) does not exceed reported L2 cache "
       "(4096 bytes); the result may be cache-dominant"},
  };
  expect_exact_messages(cases);
  const std::string usage =
      Messages::llm_memory_usage_options("memory_benchmark");
  expect_no_model_specific_llm_metal_status(usage);
}

TEST(MessagesTest, GeneralHelpAdvertisesTheLlmBoundaryExactlyOnce) {
  const std::string usage = Messages::usage_options("memory_benchmark");
  EXPECT_NE(usage.find("-M, --llm-memory"), std::string::npos);
  EXPECT_EQ(usage.find("-M, --llm-memory"),
            usage.rfind("-M, --llm-memory"));
  EXPECT_NE(
      usage.find("standalone CPU/Metal synthetic LLM memory profile"),
      std::string::npos);
  EXPECT_NE(usage.find("experimental Metal preview accepts decode"),
            std::string::npos);
  EXPECT_NE(
      usage.find("with contiguous or paged KV and contiguous prefill, and never"),
      std::string::npos);
  expect_no_model_specific_llm_metal_status(usage);
  EXPECT_NE(usage.find("--weight-size-mb"), std::string::npos);
  EXPECT_NE(usage.find("--query-heads"), std::string::npos);
  EXPECT_NE(usage.find("--context-tokens"), std::string::npos);
  EXPECT_NE(usage.find("memory-only interpretation"), std::string::npos);
  EXPECT_NE(usage.find("JSON uses schema 1 methodologies "),
            std::string::npos);
  EXPECT_NE(usage.find(
                Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION),
            std::string::npos);
  EXPECT_NE(usage.find(Constants::LLM_CPU_DECODE_PAGED_METHODOLOGY_VERSION),
            std::string::npos);
  EXPECT_NE(usage.find("llm-memory-v1-metal-decode-contiguous"),
            std::string::npos);
  EXPECT_NE(usage.find("llm-memory-v1-metal-decode-paged"),
            std::string::npos);
  EXPECT_NE(usage.find("Metal LLM-memory rejects --threads"),
            std::string::npos);
  EXPECT_NE(usage.find("checkpoints each terminal scenario"),
            std::string::npos);
  EXPECT_EQ(usage.find("execution is unavailable"), std::string::npos);
  EXPECT_EQ(usage.find("future LLM"), std::string::npos);
}

TEST(MessagesErrorTest, GpuMessagesHaveExactMethodologyOutput) {
  const std::string expected_usage =
      "Usage: memory_benchmark --gpu-bandwidth [options]\n"
      "Options for standalone GPU memory bandwidth mode:\n"
      "  -G, --gpu-bandwidth   Measure Metal GPU memory read/write/copy bandwidth.\n"
      "  -b, --buffer-size <MB>\n"
      "                        Size of each private GPU buffer (default: " +
      std::to_string(Constants::GPU_DEFAULT_BUFFER_SIZE_MB) +
      " MB; minimum: " + std::to_string(Constants::GPU_MIN_BUFFER_SIZE_MB) +
      " MB).\n"
      "  -i, --iterations <count>\n"
      "                        Exact full-buffer pass count. When omitted, each operation\n"
      "                        calibrates toward 150 ms in a 100-250 ms window.\n"
      "  -r, --count <count>   Number of balanced read/write/copy loops (default: " +
      std::to_string(Constants::GPU_DEFAULT_LOOP_COUNT) +
      ").\n"
      "      --seed <uint64>   Reproducible base seed; generated once when omitted.\n"
      "  -o, --output <target> JSON output target; exact - writes one final schema 1 document\n"
      "                        to stdout and routes human output to stderr; exact ./- and\n"
      "                        every other non-empty target are files with atomic checkpoints.\n"
      "                        An empty value disables JSON for this direct command.\n"
      "  -h, --help            Show this GPU-mode help and exit\n";
  const std::vector<MessageCase> cases = {
      {"mode isolation", Messages::error_gpu_bandwidth_must_be_used_alone(),
       "--gpu-bandwidth allows only optional -b/--buffer-size <MB>, "
       "-i/--iterations <count>, -r/--count <count>, -o/--output <target>, "
       "--seed <uint64>, and -h/--help (no other options allowed)"},
      {"minimum buffer", Messages::error_gpu_buffer_size_below_minimum(32, 64),
       "GPU buffer-size must be at least 64 MB (got 32 MB)"},
      {"iteration guard", Messages::error_gpu_iterations_exceed_limit(513, 512),
       "GPU iterations exceed the exact-work guardrail (requested 513, maximum 512)"},
      {"run failure", Messages::error_gpu_run_failed("test-reason"),
       "GPU memory bandwidth benchmark failed (reason_code=test-reason)"},
      {"unknown device", Messages::gpu_unknown_device_name(), "unknown Apple GPU"},
      {"usage", Messages::gpu_usage_options("memory_benchmark"), expected_usage},
      {"report header", Messages::report_gpu_bandwidth_header("Apple M4", 3, true),
       "GPU memory bandwidth (Apple M4, private/tracked, 3 loops; headline: median)"},
      {"copy payload", Messages::report_gpu_bandwidth_value("Copy", 123.456, true),
       "  Copy:  123.46 GB/s  (aggregate read + write payload)"},
      {"repeatability", Messages::report_gpu_bandwidth_repeatability(1.0, 2.0, 3.0, true),
       "  Repeatability: read CV 1.00%, write CV 2.00%, copy CV 3.00%"},
      {"interpretation", Messages::report_gpu_bandwidth_interpretation_note(),
       "  Note: copy is aggregate read+write throughput; DRAM residency is unverified. "
       "Results can remain cache/dispatch-dominant even at the 64 MB methodology minimum."},
      {"high CV", Messages::warning_gpu_high_cv("read", 5.1, 5.0), "GPU read repeatability CV 5.10% exceeds 5.00%"},
      {"unbalanced order", Messages::warning_gpu_order_not_balanced(),
       "GPU operation order is not fully balanced across completed loops"},
      {"duration quality", Messages::warning_gpu_duration_quality("write", "payload-cap-below-target"),
       "GPU write duration quality is payload-cap-below-target"},
      {"environment", Messages::warning_gpu_environment_not_nominal(),
       "GPU result environment is not reference-eligible (thermal state or Low Power Mode)"},
      {"working set", Messages::warning_gpu_recommended_working_set_exceeded(),
       "GPU allocation exceeds Metal's advisory recommended working-set size"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesErrorTest, ErrorMadviseFailed) {
  std::string msg = Messages::error_madvise_failed("lat_buffer");
  EXPECT_EQ(msg, "madvise failed for lat_buffer");
}

TEST(MessagesErrorTest, ErrorBenchmarkTests) {
  std::string msg = Messages::error_benchmark_tests("test failure");
  EXPECT_EQ(msg, "Error during benchmark tests: test failure");
}

TEST(MessagesErrorTest, ErrorOnlyFlagsRequireBenchmark) {
  const std::string& msg = Messages::error_only_flags_require_benchmark();
  EXPECT_NE(msg.find("--only-bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("--only-latency"), std::string::npos);
  EXPECT_NE(msg.find("--benchmark"), std::string::npos);
}

TEST(MessagesErrorTest, ErrorSweepMessages) {
  const std::vector<MessageCase> cases = {
      {"missing parameter", Messages::error_sweep_requires_parameter(),
       "--sweep requires at least one parameter specification"},
      {"missing output", Messages::error_sweep_requires_output(),
       "--sweep requires --output <target> for the combined JSON result"},
      {"run cap", Messages::error_sweep_too_many_runs(12, 10),
       "Sweep would generate 12 runs, exceeding --sweep-max-runs 10"},
      {"parameter not allowed", Messages::error_sweep_parameter_not_allowed("cache-size", "--patterns"),
       "Sweep parameter 'cache-size' is not allowed with --patterns"},
  };

  expect_exact_messages(cases);
}

// ============================================================================
// Warning Messages Tests
// ============================================================================

TEST(MessagesWarningTest, WarningBufferSizeExceedsLimit) {
  std::string msg = Messages::warning_buffer_size_exceeds_limit(2048, 1024);
  EXPECT_NE(msg.find("2048"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("Requested buffer size"), std::string::npos);
}

TEST(MessagesWarningTest, WarningTlbMlockFailureIsBestEffort) {
  const std::string msg = Messages::warning_tlb_mlock_failed(12, "Cannot allocate memory");
  EXPECT_NE(msg.find("mlock()"), std::string::npos);
  EXPECT_NE(msg.find("errno 12"), std::string::npos);
  EXPECT_NE(msg.find("continuing"), std::string::npos);
}

// ============================================================================
// Info Messages Tests (using fixture)
// ============================================================================

TEST(MessagesInfoTest, InfoSettingMaxFallback) {
  std::string msg = Messages::info_setting_max_fallback(2048);
  EXPECT_NE(msg.find("2048"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
  EXPECT_NE(msg.find("Info"), std::string::npos);
}

TEST(MessagesInfoTest, InfoCalculatedMaxLessThanMin) {
  std::string msg = Messages::info_calculated_max_less_than_min(512, 1024);
  EXPECT_NE(msg.find("512"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
}

TEST(MessagesInfoTest, InfoCustomCacheRoundedUp) {
  std::string msg = Messages::info_custom_cache_rounded_up(250, 256);
  EXPECT_NE(msg.find("250"), std::string::npos);
  EXPECT_NE(msg.find("256"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
}

// ============================================================================
// Main Program Messages Tests
// ============================================================================

TEST(MessagesFormattingTest, LiteralMessagesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"timer creation error", Messages::error_timer_creation_failed(),
       "Failed to create high-resolution timer. Exiting."},
      {"memory fallback warning", Messages::warning_cannot_get_memory(),
       "Cannot get available memory. Using fallback limit."},
      {"interrupted", Messages::msg_interrupted_by_user(), "\nInterrupted by user. Partial results shown."},
      {"copyright", Messages::config_copyright(), "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>"},
      {"processor error", Messages::config_processor_name_error(), "Could not retrieve processor name."},
      {"cache header", Messages::cache_info_header(), "\nDetected Cache Sizes:"},
      {"main latency methodology", Messages::results_main_memory_latency(),
       "\nMain Memory Latency Test (single-threaded, pointer chase):"},
      {"cache latency methodology", Messages::results_cache_latency(),
       "\nCache Latency Tests (single-threaded, pointer chase):"},
      {"custom cache", Messages::results_custom_cache(), "  Custom Cache:"},
      {"L1 cache", Messages::results_l1_cache(), "  L1 Cache:"},
      {"L2 cache", Messages::results_l2_cache(), "  L2 Cache:"},
      {"results separator", Messages::results_separator(), "--------------"},
      {"statistics cache read", Messages::statistics_cache_read(), "  Read:"},
      {"statistics cache write", Messages::statistics_cache_write(), "  Write:"},
      {"statistics cache copy", Messages::statistics_cache_copy(), "  Copy:"},
      {"statistics cache latency header", Messages::statistics_cache_latency_header(), "\nCache Latency (ns):"},
      {"statistics main latency header", Messages::statistics_main_memory_latency_header(),
       "\nMain Memory Latency (ns):"},
      {"statistics footer", Messages::statistics_footer(), "----------------------------------"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, MsgSweepProgress) {
  std::string running = Messages::msg_running_sweep(3);
  EXPECT_NE(running.find("Running sweep"), std::string::npos);
  EXPECT_NE(running.find("3"), std::string::npos);

  std::string progress = Messages::msg_sweep_run_progress(2, 5);
  EXPECT_NE(progress.find("2/5"), std::string::npos);
}

TEST(MessagesFormattingTest, MsgDoneTotalTime) {
  const std::vector<MessageCase> cases = {
      {"ordinary duration", Messages::msg_done_total_time(123.456), "\nDone. Total execution time: 123.45600 s"},
      {"short duration", Messages::msg_done_total_time(0.001), "\nDone. Total execution time: 0.00100 s"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, MsgResultsSavedTo) {
  std::string msg = Messages::msg_results_saved_to("results.json");
  EXPECT_NE(msg.find("Results saved to"), std::string::npos);
  EXPECT_NE(msg.find("results.json"), std::string::npos);

  msg = Messages::msg_results_saved_to("/tmp/out/bench.json");
  EXPECT_NE(msg.find("/tmp/out/bench.json"), std::string::npos);
}

TEST(MessagesFormattingTest, MsgPatternBenchmarkLoopCompleted) {
  std::string msg = Messages::msg_pattern_benchmark_loop_completed(3, 10);
  EXPECT_NE(msg.find("3"), std::string::npos);
  EXPECT_NE(msg.find("10"), std::string::npos);
  EXPECT_NE(msg.find("Pattern benchmarks"), std::string::npos);
  EXPECT_NE(msg.find("completed"), std::string::npos);

  // Verify the loop separator is present
  EXPECT_NE(msg.find("3/10"), std::string::npos);
}

TEST(MessagesFormattingTest, MsgTlbAnalysisRefinementStart) {
  std::string msg = Messages::msg_tlb_analysis_refinement_start(20);
  EXPECT_NE(msg.find("Starting refinement sweep"), std::string::npos);
  EXPECT_NE(msg.find("20"), std::string::npos);
  EXPECT_NE(msg.find("points"), std::string::npos);
}

TEST(MessagesFormattingTest, MsgTlbAnalysisValidationStart) {
  const std::string msg = Messages::msg_tlb_analysis_validation_start(8);
  EXPECT_NE(msg.find("validation"), std::string::npos);
  EXPECT_NE(msg.find("8"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbStatisticalConfidence) {
  const std::string msg = Messages::report_tlb_statistical_confidence("High", 2.5, 2.1, 2.9, 2.0, 2.8);
  EXPECT_NE(msg.find("paired effect"), std::string::npos);
  EXPECT_NE(msg.find("discovery 95% CI"), std::string::npos);
  EXPECT_NE(msg.find("validation 95% CI"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbPairedLocalityIsCompactAndNamesEverySignal) {
  const std::string latency = Messages::report_tlb_paired_locality_progress(
      29, 29, 256 * Constants::BYTES_PER_MB, 12.43, 3.61, 8.82, Constants::BYTES_PER_MB, true);
  EXPECT_NE(latency.find("[29/29] 256 MiB"), std::string::npos);
  EXPECT_NE(latency.find("delta 8.82 ns"), std::string::npos);
  EXPECT_NE(latency.find("spread 12.43, packed 3.61"), std::string::npos);
  EXPECT_NE(latency.find("active 1 MiB"), std::string::npos);
  EXPECT_EQ(latency.back(), '*');
  EXPECT_EQ(latency.find("pages"), std::string::npos);
  EXPECT_EQ(latency.find('\n'), std::string::npos);

  const std::string legend = Messages::report_tlb_sweep_legend();
  EXPECT_NE(legend.find("cache-line footprint"), std::string::npos);
  EXPECT_NE(legend.find("<64-node short-cycle diagnostic"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbLatencyNormalizesNegativeZero) {
  const std::string latency = Messages::report_tlb_paired_locality_progress(
      1, 1, 16 * Constants::BYTES_PER_KB, 0.08, 0.08, -0.001, Constants::CACHE_LINE_SIZE_BYTES, false);
  EXPECT_NE(latency.find("delta 0.00 ns"), std::string::npos);
  EXPECT_EQ(latency.find("-0.00"), std::string::npos);
  EXPECT_NE(latency.find("16 KiB"), std::string::npos);
}

TEST(MessagesFormattingTest, AnalyzeTlbStrideGuardMessages) {
  const std::string exceeds = Messages::error_analyze_tlb_stride_exceeds_page(32768, 16384);
  EXPECT_NE(exceeds.find("--analyze-tlb"), std::string::npos);
  EXPECT_NE(exceeds.find("32768"), std::string::npos);
  EXPECT_NE(exceeds.find("16384"), std::string::npos);
}

TEST(MessagesFormattingTest, TlbChainSetupFailureMessage) {
  const std::string msg = Messages::error_tlb_chain_setup_failed(256, "packed", "integrity-failure", "early-cycle");
  EXPECT_NE(msg.find("packed"), std::string::npos);
  EXPECT_NE(msg.find("256"), std::string::npos);
  EXPECT_NE(msg.find("integrity-failure"), std::string::npos);
  EXPECT_NE(msg.find("early-cycle"), std::string::npos);
}

TEST(MessagesFormattingTest, DuplicateSweepParameterMessage) {
  const std::string msg = Messages::error_duplicate_sweep_parameter("latency-stride-bytes");
  EXPECT_NE(msg.find("more than once"), std::string::npos);
  EXPECT_NE(msg.find("latency-stride-bytes"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbAnalysisStatusAndSuppressedConclusions) {
  const std::string status = Messages::report_tlb_analysis_status("interrupted", 29, 7, false);
  EXPECT_NE(status.find("interrupted"), std::string::npos);
  EXPECT_NE(status.find("7/29"), std::string::npos);
  EXPECT_NE(status.find("suppressed"), std::string::npos);

  const std::string unavailable = Messages::report_tlb_conclusions_unavailable("interrupted");
  EXPECT_NE(unavailable.find("Suppressed"), std::string::npos);
  EXPECT_NE(unavailable.find("interrupted"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbRunAndResourceSummariesAreCompact) {
  const std::string run = Messages::report_tlb_run_summary("Apple M5", 16 * Constants::BYTES_PER_KB, 256, "quick",
                                                           "auto", "random-box", 570001, true);
  EXPECT_NE(run.find("Apple M5"), std::string::npos);
  EXPECT_NE(run.find("page 16 KiB"), std::string::npos);
  EXPECT_NE(run.find("stride 256 B | quick"), std::string::npos);
  EXPECT_NE(run.find("mode auto->random-box"), std::string::npos);
  EXPECT_NE(run.find("seed 570001 (user)"), std::string::npos);
  EXPECT_EQ(run.find('\n'), std::string::npos);

  const std::string resources =
      Messages::report_tlb_resource_summary(1024, true, true, true, 0, 2048, 1041 * Constants::BYTES_PER_MB);
  EXPECT_NE(resources.find("1024 MiB buffer (locked)"), std::string::npos);
  EXPECT_NE(resources.find("QoS applied"), std::string::npos);
  EXPECT_NE(resources.find("estimated peak/budget 1041.0/2048 MiB"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbResourceSummaryRetainsFailures) {
  const std::string failed =
      Messages::report_tlb_resource_summary(256, false, true, false, 6, 512, 300 * Constants::BYTES_PER_MB);
  EXPECT_NE(failed.find("unlocked"), std::string::npos);
  EXPECT_NE(failed.find("failed (code 6; best-effort)"), std::string::npos);

  const std::string not_requested =
      Messages::report_tlb_resource_summary(256, false, false, false, 0, 512, 300 * Constants::BYTES_PER_MB);
  EXPECT_NE(not_requested.find("QoS not requested"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbWorkEstimateIsConcise) {
  const std::string work = Messages::report_tlb_work_estimate("base", 15, 10, 20, 3.75, 7.5);
  EXPECT_NE(work.find("Work Estimate [base]"), std::string::npos);
  EXPECT_EQ(work.find("pointer accesses"), std::string::npos);
  EXPECT_EQ(work.find("peak"), std::string::npos);
  EXPECT_NE(work.find("3.75-7.50 s"), std::string::npos);

  const std::string completion = Messages::report_tlb_pass_completion("base", 12, "CI target reached");
  EXPECT_NE(completion.find("12 rounds"), std::string::npos);
  EXPECT_NE(completion.find("CI target reached"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbSweepPlanUsesIecUnits) {
  const std::string enabled = Messages::report_tlb_sweep_plan(
      16 * Constants::BYTES_PER_KB, 256 * Constants::BYTES_PER_MB, 15, true, 512 * Constants::BYTES_PER_MB, 512, 1024);
  EXPECT_NE(enabled.find("16 KiB -> 256 MiB"), std::string::npos);
  EXPECT_NE(enabled.find("15 points"), std::string::npos);
  EXPECT_NE(enabled.find("large comparison 512 MiB enabled"), std::string::npos);

  const std::string disabled = Messages::report_tlb_sweep_plan(
      16 * Constants::BYTES_PER_KB, 256 * Constants::BYTES_PER_MB, 15, false, 512 * Constants::BYTES_PER_MB, 512, 256);
  EXPECT_NE(disabled.find("unavailable"), std::string::npos);
  EXPECT_NE(disabled.find("requires 512 MiB"), std::string::npos);
  EXPECT_NE(disabled.find("selected 256 MiB"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbLargeLocalityUnavailable) {
  const std::string msg = Messages::report_tlb_large_locality_paired_unavailable(512, 256);
  EXPECT_NE(msg.find("Large-Locality Paired Comparison"), std::string::npos);
  EXPECT_NE(msg.find("N/A"), std::string::npos);
  EXPECT_NE(msg.find("requires 512 MiB"), std::string::npos);
  EXPECT_NE(msg.find("selected 256 MiB"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbLargeLocalityInterrupted) {
  const std::string msg = Messages::report_tlb_large_locality_paired_interrupted();
  EXPECT_NE(msg.find("N/A"), std::string::npos);
  EXPECT_NE(msg.find("did not complete"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbLargeLocalityPairedComparisonIsExplicit) {
  const std::string msg = Messages::report_tlb_large_locality_paired_comparison(
      512 * Constants::BYTES_PER_MB, 13.20, 6.40, 6.80, 32768, 128, 32768, 2 * Constants::BYTES_PER_MB);
  EXPECT_NE(msg.find("512 MiB"), std::string::npos);
  EXPECT_NE(msg.find("2 MiB"), std::string::npos);
  EXPECT_NE(msg.find("32768/128"), std::string::npos);
  EXPECT_NE(msg.find("P50: delta 6.80 ns/access"), std::string::npos);
  EXPECT_NE(msg.find("not DRAM latency"), std::string::npos);
  EXPECT_NE(msg.find("isolated page-table-walk cost"), std::string::npos);
  EXPECT_EQ(msg.find("Virtual Locality:"), std::string::npos);
}

TEST(MessagesFormattingTest, ReportTlbQuickProfileNoteRequiresConfirmation) {
  const std::string msg = Messages::report_tlb_quick_profile_note();
  EXPECT_NE(msg.find("screening estimates"), std::string::npos);
  EXPECT_NE(msg.find("confirm boundaries with medium or high"), std::string::npos);
}

TEST(MessagesFormattingTest, TlbPrivateCacheMessagesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"fine sweep", Messages::report_tlb_fine_sweep(6, 21), "Refinement: +6 points (21 total)"},
      {"private cache section", Messages::report_tlb_private_cache_section(), "[Private Cache Detection]"},
      {"strong candidate", Messages::report_tlb_private_cache_candidate(true),
       "  Candidate Type: Strong private-cache candidate"},
      {"early candidate", Messages::report_tlb_private_cache_candidate(false),
       "  Candidate Type: Early-cache candidate"},
      {"elevated risk", Messages::report_tlb_private_cache_interference(true, 512),
       "  TLB Interference Risk: Elevated near 512 KiB locality"},
      {"low risk", Messages::report_tlb_private_cache_interference(false, 512),
       "  TLB Interference Risk: Low near 512 KiB locality"},
      {"L1 distance", Messages::report_tlb_private_cache_l1_distance(4608, 288),
       "  Distance to L1 TLB Boundary: 4608 KiB (288 pages)"},
      {"boundary", Messages::report_tlb_boundary_kb(4096), "  Boundary: 4096 KiB"},
      {"size estimate", Messages::report_tlb_inferred_size_entries(248), "  Inferred Size Estimate: ~248 entries"},
      {"reach estimate", Messages::report_tlb_inferred_reach_entries(2000), "  Inferred Reach Estimate: ~2000 entries"},
      {"entry range", Messages::report_tlb_inferred_entries_range(240, 256), "  Inferred Entry Range: 240-256 entries"},
      {"private cache overlap", Messages::report_tlb_private_cache_overlap(),
       "  Private Cache Overlap: yes (kept as ambiguous L1 TLB candidate)"},
  };

  expect_exact_messages(cases);
}

// ============================================================================
// Usage/Help Messages Tests (using formatting fixture)
// ============================================================================

TEST(MessagesFormattingTest, UsageHeader) {
  std::string msg = Messages::usage_header("1.0.0");
  EXPECT_NE(msg.find("1.0.0"), std::string::npos);
  EXPECT_NE(msg.find("Timo Heimonen"), std::string::npos);
  EXPECT_NE(msg.find("GNU GPL"), std::string::npos);
  EXPECT_NE(msg.find("github.com"), std::string::npos);
}

TEST(MessagesFormattingTest, UsageOptions) {
  std::string msg = Messages::usage_options("memory_benchmark");
  const std::string standard_schema_contract =
      "                        JSON uses standard schema " +
      std::to_string(Constants::BENCHMARK_JSON_SCHEMA_VERSION) +
      " methodology " + Constants::BENCHMARK_METHODOLOGY_VERSION + ".\n";
  EXPECT_NE(msg.find("memory_benchmark"), std::string::npos);
  EXPECT_NE(msg.find("--benchmark"), std::string::npos);
  EXPECT_NE(msg.find(standard_schema_contract), std::string::npos);
  EXPECT_NE(msg.find("100-300 ms window"), std::string::npos);
  EXPECT_NE(msg.find("--iterations"), std::string::npos);
  EXPECT_NE(msg.find("--buffer-size"), std::string::npos);
  EXPECT_NE(msg.find("--count"), std::string::npos);
  EXPECT_NE(msg.find("--analyze-tlb"), std::string::npos);
  EXPECT_NE(msg.find("schema 4"), std::string::npos);
  EXPECT_NE(msg.find("exact string seeds"), std::string::npos);
  EXPECT_NE(msg.find("scoped counters"), std::string::npos);
  EXPECT_NE(msg.find("--tlb-density"), std::string::npos);
  EXPECT_NE(msg.find("default: medium"), std::string::npos);
  EXPECT_NE(msg.find("--analyze-tlb: 16"), std::string::npos);
  EXPECT_NE(msg.find("calibrate toward 150 ms"), std::string::npos);
  EXPECT_NE(msg.find("Reproducible workload/schedule seed for --benchmark, --patterns"), std::string::npos);
  EXPECT_NE(msg.find("--gpu-bandwidth"), std::string::npos);
  EXPECT_NE(msg.find(Constants::GPU_METHODOLOGY_VERSION), std::string::npos);
  EXPECT_NE(msg.find("minimum buffer size is 64 MB"), std::string::npos);
  EXPECT_NE(msg.find("standalone CPU/Metal synthetic LLM memory profile"),
            std::string::npos);
  EXPECT_NE(msg.find("prefill requires --phase prefill, --prompt-tokens"), std::string::npos);
  EXPECT_NE(msg.find("--attention-query-tile-tokens"), std::string::npos);
  EXPECT_NE(msg.find("Both phases support contiguous"), std::string::npos);
  EXPECT_NE(msg.find("llm-memory-v1-cpu-prefill-contiguous"), std::string::npos);
  EXPECT_NE(msg.find(Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION), std::string::npos);
  EXPECT_NE(msg.find("--analyze-core2core"), std::string::npos);
  EXPECT_NE(msg.find("acquire/release token-handoff"), std::string::npos);
  EXPECT_NE(msg.find("protocol, coherence, and scheduler effects"), std::string::npos);
  EXPECT_NE(msg.find("core-to-core schema 2"), std::string::npos);
  EXPECT_NE(msg.find("target 250 ms"), std::string::npos);
  EXPECT_NE(msg.find("Defaults to 3 loops"), std::string::npos);
  EXPECT_NE(msg.find("--latency-samples"), std::string::npos);
  EXPECT_NE(msg.find("--latency-stride-bytes"), std::string::npos);
  EXPECT_NE(msg.find("--latency-chain-mode"), std::string::npos);
  EXPECT_NE(msg.find("--latency-tlb-locality-kb"), std::string::npos);
  EXPECT_NE(msg.find("Locality-using modes require"), std::string::npos);
  EXPECT_NE(msg.find("explicit global-random ignores"), std::string::npos);
  EXPECT_NE(msg.find("cache bandwidth uses one"), std::string::npos);
  EXPECT_NE(msg.find("keeps an explicit request uncapped"),
            std::string::npos);
  EXPECT_NE(msg.find("latency remains"), std::string::npos);
  EXPECT_NE(msg.find("single-threaded"), std::string::npos);
  EXPECT_NE(msg.find("--cache-size"), std::string::npos);
  EXPECT_NE(msg.find("--output <target>"), std::string::npos);
  EXPECT_NE(msg.find("Exact - writes one final JSON document to stdout"), std::string::npos);
  EXPECT_NE(msg.find("one final"), std::string::npos);
  EXPECT_NE(msg.find("every result-producing direct mode and CPU sweep"),
            std::string::npos);
  EXPECT_NE(msg.find("routes human output to stderr"), std::string::npos);
  EXPECT_NE(msg.find("an empty value disables JSON for direct commands"),
            std::string::npos);
  EXPECT_NE(msg.find("invalid for sweeps"), std::string::npos);
  EXPECT_NE(msg.find("Every other non-empty value is a file"),
            std::string::npos);
  EXPECT_NE(msg.find("including ./- and names such as -G"),
            std::string::npos);
  EXPECT_NE(msg.find("Standard, GPU, and LLM-memory files retain"),
            std::string::npos);
  EXPECT_NE(msg.find("LLM-memory checkpoints each terminal scenario"),
            std::string::npos);
  EXPECT_NE(msg.find("sweep files checkpoint attempts"), std::string::npos);
  EXPECT_NE(msg.find("Requires --output <target>"), std::string::npos);
  EXPECT_NE(msg.find("-h"), std::string::npos);
  // Check that default values are included
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_ITERATIONS)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_BUFFER_SIZE_MB)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_LOOP_COUNT)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_LATENCY_SAMPLE_COUNT)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::MIN_CACHE_SIZE_KB)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::MAX_CACHE_SIZE_KB)), std::string::npos);
}

TEST(MessagesFormattingTest, UsageExample) {
  std::string msg = Messages::usage_example("memory_benchmark");
  EXPECT_NE(msg.find("memory_benchmark"), std::string::npos);
  EXPECT_NE(msg.find("--iterations"), std::string::npos);
  EXPECT_NE(msg.find("--buffer-size"), std::string::npos);
  EXPECT_NE(msg.find("--output"), std::string::npos);
  EXPECT_NE(msg.find("Machine JSON:"), std::string::npos);
  EXPECT_NE(msg.find("--only-bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("--output -"), std::string::npos);
}

// ============================================================================
// Configuration Output Messages Tests (using formatting fixture)
// ============================================================================

TEST(MessagesFormattingTest, ConfigHeader) {
  std::string msg = Messages::config_header("1.0.0");
  EXPECT_NE(msg.find("1.0.0"), std::string::npos);
  EXPECT_NE(msg.find("macOS-memory-benchmark"), std::string::npos);
}

TEST(MessagesFormattingTest, ConfigLicense) {
  std::string msg = Messages::config_license();
  EXPECT_NE(msg.find("free software"), std::string::npos);
  EXPECT_NE(msg.find("GNU General Public License"), std::string::npos);
  EXPECT_NE(msg.find("version 3"), std::string::npos);
}

TEST(MessagesFormattingTest, ConfigBufferSize) {
  std::string msg = Messages::config_buffer_size(1024.5, 1024);
  EXPECT_NE(msg.find("1024.50"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("MiB"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST(MessagesFormattingTest, ConfigTotalAllocation) {
  std::string msg = Messages::config_total_allocation(3072.75);
  EXPECT_NE(msg.find("3072.75"), std::string::npos);
  EXPECT_NE(msg.find("MiB"), std::string::npos);
  EXPECT_NE(msg.find("Peak Concurrent Allocation"), std::string::npos);
}

TEST(MessagesFormattingTest, ScalarConfigMessagesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"iterations", Messages::config_iterations(1000), "Iterations (per R/W/Copy test per loop): 1000"},
      {"loop count", Messages::config_loop_count(5), "Loop Count (total benchmark repetitions): 5"},
      {"latency stride", Messages::config_latency_stride(136), "Latency Stride: 136 B"},
      {"latency chain mode", Messages::config_latency_chain_mode("random-in-box-random-box"),
       "Latency Chain Mode: random-in-box-random-box"},
      {"processor name", Messages::config_processor_name("Apple M1"), "\nProcessor Name: Apple M1"},
      {"performance cores", Messages::config_performance_cores(8), "  Performance Cores: 8"},
      {"efficiency cores", Messages::config_efficiency_cores(2), "  Efficiency Cores: 2"},
      {"total cores", Messages::config_total_cores(10), "  Total CPU Cores Detected: 10"},
      {"benchmark threads", Messages::config_benchmark_threads(4), "  Benchmark Threads Requested: 4"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, ConfigNonCacheable) {
  // Test enabled
  std::string msg = Messages::config_non_cacheable(true);
  EXPECT_NE(msg.find("Non-Cacheable Memory Hints"), std::string::npos);
  EXPECT_NE(msg.find("Enabled"), std::string::npos);

  // Test disabled
  msg = Messages::config_non_cacheable(false);
  EXPECT_NE(msg.find("Non-Cacheable Memory Hints"), std::string::npos);
  EXPECT_NE(msg.find("Disabled"), std::string::npos);
}

TEST(MessagesFormattingTest, ConfigLatencyTlbLocality) {
  std::string msg = Messages::config_latency_tlb_locality(0);
  EXPECT_NE(msg.find("Latency TLB Locality"), std::string::npos);
  EXPECT_NE(msg.find("Global random"), std::string::npos);

  msg = Messages::config_latency_tlb_locality(16 * 1024);
  EXPECT_NE(msg.find("16.00"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
}

// ============================================================================
// Cache Info Messages Tests (using formatting fixture)
// ============================================================================

TEST(MessagesFormattingTest, CacheSizeCustom) {
  // Test bytes
  std::string msg = Messages::cache_size_custom(512);
  EXPECT_NE(msg.find("512"), std::string::npos);
  EXPECT_NE(msg.find("B"), std::string::npos);

  // Test KB
  msg = Messages::cache_size_custom(256 * 1024);
  EXPECT_NE(msg.find("256"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);

  // Test MB
  msg = Messages::cache_size_custom(2 * 1024 * 1024);
  EXPECT_NE(msg.find("2"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST(MessagesFormattingTest, CacheSizeCustomDisabled) {
  std::string msg = Messages::cache_size_custom_disabled();
  EXPECT_NE(msg.find("Custom Cache Size"), std::string::npos);
  EXPECT_NE(msg.find("Disabled"), std::string::npos);
}

TEST(MessagesFormattingTest, CacheSizeL1) {
  // Test bytes
  std::string msg = Messages::cache_size_l1(128 * 1024);
  EXPECT_NE(msg.find("128"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
  EXPECT_NE(msg.find("per P-core"), std::string::npos);

  // Test MB
  msg = Messages::cache_size_l1(1 * 1024 * 1024);
  EXPECT_NE(msg.find("1"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST(MessagesFormattingTest, CacheSizeL2) {
  // Test KB
  std::string msg = Messages::cache_size_l2(4 * 1024 * 1024);
  EXPECT_NE(msg.find("4"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
  EXPECT_NE(msg.find("per P-core cluster"), std::string::npos);
}

// ============================================================================
// Results Output Messages Tests (using formatting fixture)
// ============================================================================

TEST(MessagesFormattingTest, ResultsLoopHeader) {
  const std::vector<MessageCase> cases = {
      {"first loop", Messages::results_loop_header(0), "\n--- Results (Loop 1) ---"},
      {"fifth loop", Messages::results_loop_header(4), "\n--- Results (Loop 5) ---"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, ResultsMainMemoryBandwidth) {
  std::string msg = Messages::results_main_memory_bandwidth(8);
  EXPECT_NE(msg.find("8"), std::string::npos);
  EXPECT_NE(msg.find("threads"), std::string::npos);
  EXPECT_NE(msg.find("Main Memory Bandwidth"), std::string::npos);
}

TEST(MessagesFormattingTest, BandwidthResultsHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"main read", Messages::results_read_bandwidth(25.123, 1.456), "  Read : 25.12300 GB/s (Total time: 1.45600 s)"},
      {"main write", Messages::results_write_bandwidth(30.789, 2.345),
       "  Write: 30.78900 GB/s (Total time: 2.34500 s)"},
      {"main copy", Messages::results_copy_bandwidth(20.456, 3.789), "  Copy : 20.45600 GB/s (Total time: 3.78900 s)"},
      {"cache read", Messages::results_cache_read_bandwidth(150.789), "    Read : 150.78900 GB/s"},
      {"cache write", Messages::results_cache_write_bandwidth(200.123), "    Write: 200.12300 GB/s"},
      {"cache copy", Messages::results_cache_copy_bandwidth(175.456), "    Copy : 175.45600 GB/s"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, ResultsLatencyTotalTime) {
  EXPECT_EQ(Messages::results_latency_total_time(5.678), "  Total time: 5.67800 s");
}

TEST(MessagesFormattingTest, ResultsLatencyAverage) {
  std::string msg = Messages::results_latency_average(123.45, 1024 * 1024);
  EXPECT_NE(msg.find("123.45"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
  EXPECT_NE(msg.find("Average latency"), std::string::npos);
  EXPECT_NE(msg.find("1.00 MB locality"), std::string::npos);

  msg = Messages::results_latency_average(86.70, 0);
  EXPECT_NE(msg.find("global random locality"), std::string::npos);
}

TEST(MessagesFormattingTest, ResultsLatencyTlbHit) {
  std::string msg = Messages::results_latency_tlb_hit(24.10);
  EXPECT_NE(msg.find("16 KiB locality latency"), std::string::npos);
  EXPECT_NE(msg.find("24.10"), std::string::npos);
}

TEST(MessagesFormattingTest, ResultsLatencyTlbMiss) {
  std::string msg = Messages::results_latency_tlb_miss(86.70);
  EXPECT_NE(msg.find("Global-random latency"), std::string::npos);
  EXPECT_NE(msg.find("86.70"), std::string::npos);
}

TEST(MessagesFormattingTest, ResultsLatencyPageWalkPenalty) {
  std::string msg = Messages::results_latency_page_walk_penalty(62.60);
  EXPECT_NE(msg.find("Locality latency delta"), std::string::npos);
  EXPECT_NE(msg.find("global - 16 KiB"), std::string::npos);
  EXPECT_NE(msg.find("62.60"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
}

TEST(MessagesFormattingTest, ResultsCacheBandwidth) {
  std::string msg = Messages::results_cache_bandwidth(1);
  EXPECT_NE(msg.find("Cache Bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("single-threaded"), std::string::npos);
}

TEST(MessagesFormattingTest, BufferSizesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"bytes", Messages::results_buffer_size_bytes(1024), " (Buffer size: 1024 B)"},
      {"kilobytes", Messages::results_buffer_size_kb(256.5), " (Buffer size: 256.50 KB)"},
      {"megabytes", Messages::results_buffer_size_mb(1.25), " (Buffer size: 1.25 MB)"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, FiniteCacheLatenciesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"custom bytes", Messages::results_cache_latency_custom_ns(2.5, 256 * 1024),
       "  Custom Cache: 2.50 ns (Buffer size: 262144 B)"},
      {"custom kilobytes", Messages::results_cache_latency_custom_ns_kb(3.75, 128.5),
       "  Custom Cache: 3.75 ns (Buffer size: 128.50 KB)"},
      {"custom megabytes", Messages::results_cache_latency_custom_ns_mb(4.25, 0.5),
       "  Custom Cache: 4.25 ns (Buffer size: 0.50 MB)"},
      {"L1 bytes", Messages::results_cache_latency_l1_ns(0.5, 64 * 1024), "  L1 Cache: 0.50 ns (Buffer size: 65536 B)"},
      {"L1 kilobytes", Messages::results_cache_latency_l1_ns_kb(0.75, 32.25),
       "  L1 Cache: 0.75 ns (Buffer size: 32.25 KB)"},
      {"L1 megabytes", Messages::results_cache_latency_l1_ns_mb(1.0, 0.064),
       "  L1 Cache: 1.00 ns (Buffer size: 0.06 MB)"},
      {"L2 bytes", Messages::results_cache_latency_l2_ns(2.5, 4 * 1024 * 1024),
       "  L2 Cache: 2.50 ns (Buffer size: 4194304 B)"},
      {"L2 kilobytes", Messages::results_cache_latency_l2_ns_kb(3.0, 4096.5),
       "  L2 Cache: 3.00 ns (Buffer size: 4096.50 KB)"},
      {"L2 megabytes", Messages::results_cache_latency_l2_ns_mb(4.5, 4.0),
       "  L2 Cache: 4.50 ns (Buffer size: 4.00 MB)"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, NonFiniteCacheLatenciesRenderUnavailable) {
  const std::vector<MessageCase> cases = {
      {"NaN", Messages::results_cache_latency_custom_ns(std::numeric_limits<double>::quiet_NaN(), 1024),
       "  Custom Cache: N/A ns (Buffer size: 1024 B)"},
      {"infinity", Messages::results_cache_latency_custom_ns(std::numeric_limits<double>::infinity(), 1024),
       "  Custom Cache: N/A ns (Buffer size: 1024 B)"},
  };

  expect_exact_messages(cases);
}

// ============================================================================
// Statistics Messages Tests (using formatting fixture)
// ============================================================================

TEST(MessagesFormattingTest, StatisticsHeader) {
  std::string msg = Messages::statistics_header(5);
  EXPECT_NE(msg.find("5"), std::string::npos);
  EXPECT_NE(msg.find("Loops"), std::string::npos);
  EXPECT_NE(msg.find("Statistics"), std::string::npos);
}

TEST(MessagesFormattingTest, StatisticsAverage) {
  std::string msg = Messages::statistics_average(25.123, 3);
  EXPECT_NE(msg.find("25.123"), std::string::npos);
  EXPECT_NE(msg.find("Average"), std::string::npos);

  msg = Messages::statistics_average(100.5, 1);
  EXPECT_NE(msg.find("100.5"), std::string::npos);
}

TEST(MessagesFormattingTest, StatisticsCacheBandwidthHeader) {
  std::string msg = Messages::statistics_cache_bandwidth_header("L1");
  EXPECT_NE(msg.find("L1"), std::string::npos);
  EXPECT_NE(msg.find("Cache Bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
}

TEST(MessagesFormattingTest, StatisticsCacheLatencyName) {
  std::string msg = Messages::statistics_cache_latency_name("L1");
  EXPECT_EQ(msg, "  L1 Cache:");

  msg = Messages::statistics_cache_latency_name("Custom");
  EXPECT_EQ(msg, "  Custom Cache:");
}

TEST(MessagesFormattingTest, WarningBenchmarkHighCv) {
  const std::string msg = Messages::warning_benchmark_high_cv("read bandwidth", 9.25, 7.5);
  EXPECT_NE(msg.find("read bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("9.2%"), std::string::npos);
  EXPECT_NE(msg.find("7.5%"), std::string::npos);
}

TEST(MessagesFormattingTest, WarningQosFailedBenchmarkWorker) {
  const std::string msg = Messages::warning_qos_failed_benchmark_worker("read", 5);
  EXPECT_NE(msg.find("read benchmark worker"), std::string::npos);
  EXPECT_NE(msg.find("5"), std::string::npos);
}

TEST(MessagesFormattingTest, BenchmarkStatusReasonsHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"before measurement", Messages::benchmark_reason_interrupted_before_measurement(),
       "interrupted before measurement"},
      {"user interruption", Messages::benchmark_reason_interrupted_by_user(), "interrupted by user"},
      {"planned measurements", Messages::benchmark_reason_planned_measurements_unavailable(),
       "one or more planned measurements unavailable"},
      {"locality work", Messages::benchmark_reason_invalid_locality_work(), "invalid locality-comparison work"},
      {"locality comparison", Messages::benchmark_reason_locality_comparison_unavailable(),
       "paired locality comparison unavailable"},
      {"calibration interruption", Messages::benchmark_reason_interrupted_calibration_pilot(),
       "interrupted during calibration pilot"},
      {"calibration duration", Messages::benchmark_reason_invalid_calibration_pilot(),
       "invalid calibration pilot duration"},
      {"operation interruption", Messages::benchmark_reason_interrupted_measured_operation(),
       "interrupted during measured operation"},
      {"bandwidth duration", Messages::benchmark_reason_invalid_bandwidth_duration(),
       "invalid measured bandwidth duration"},
      {"bandwidth value", Messages::benchmark_reason_invalid_bandwidth_value(), "invalid measured bandwidth value"},
      {"latency pilot", Messages::benchmark_reason_interrupted_latency_pilot(),
       "interrupted during latency calibration pilot"},
      {"latency interruption", Messages::benchmark_reason_interrupted_latency_measurement(),
       "interrupted during latency measurement"},
      {"latency measurement", Messages::benchmark_reason_invalid_latency_measurement(),
       "invalid latency duration or access count"},
      {"loops remain", Messages::benchmark_reason_loops_remain(), "benchmark loops remain"},
      {"checkpoint", Messages::benchmark_reason_checkpoint_failed(), "failed to checkpoint standard benchmark JSON"},
      {"unknown loop exception", Messages::benchmark_reason_unknown_loop_exception(),
       "standard benchmark loop threw an unknown exception"},
      {"coordinator exception", Messages::benchmark_reason_coordinator_exception("boom"),
       "standard benchmark coordinator exception: boom"},
      {"unknown coordinator exception", Messages::benchmark_reason_unknown_coordinator_exception(),
       "standard benchmark coordinator threw an unknown exception"},
      {"latency chain", Messages::benchmark_reason_latency_chain_setup_failed("main-latency"),
       "failed to construct main-latency latency chain"},
      {"preparation", Messages::benchmark_reason_prepare_failed("cache latency"),
       "failed to prepare cache latency buffers"},
      {"bandwidth plan", Messages::benchmark_reason_invalid_bandwidth_plan(), "invalid bandwidth work-plan parameters"},
      {"worker partition", Messages::benchmark_reason_no_worker_partition(), "no valid aligned worker partition"},
      {"copy overflow", Messages::benchmark_reason_copy_payload_overflow(), "copy payload overflow"},
      {"payload overflow", Messages::benchmark_reason_total_payload_overflow(), "total payload overflow or pass limit"},
      {"latency plan", Messages::benchmark_reason_invalid_latency_plan(), "invalid latency work-plan parameters"},
      {"short latency chain", Messages::benchmark_reason_latency_chain_too_short(),
       "latency chain requires at least two nodes"},
      {"cycle limit", Messages::benchmark_reason_minimum_cycles_exceed_limit(),
       "minimum complete-cycle access count exceeds limit"},
      {"rounded access limit", Messages::benchmark_reason_rounded_accesses_exceed_limit(),
       "rounded complete-cycle access count exceeds limit"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, PatternStrideLabels) {
  EXPECT_EQ(Messages::pattern_cache_line_64b(), "64 B stride");
  EXPECT_EQ(Messages::pattern_page_4096b(), "4096 B stride");
  EXPECT_EQ(Messages::pattern_page_16384b(), "16 KiB stride");
  EXPECT_EQ(Messages::pattern_superpage_2mb(), "2 MiB stride");
}

TEST(MessagesFormattingTest, PatternMessagesHaveExactOutput) {
  const std::vector<MessageCase> cases = {
      {"unavailable", Messages::pattern_measurement_unavailable("skipped", "buffer too small"),
       "N/A [skipped: buffer too small]"},
      {"statistics header", Messages::statistics_pattern_bandwidth_header("Random"),
       "\nRandom Pattern Bandwidth (GB/s):"},
      {"coefficient of variation", Messages::statistics_coefficient_of_variation(12.34, 1), "  CV:      12.3%"},
      {"noise warning", Messages::warning_pattern_measurement_noisy("Random read", 12.3, 10.0),
       "Noisy pattern measurement: Random read CV 12.3% exceeds 10.0%"},
      {"not completed", Messages::pattern_reason_measurement_not_completed(), "measurement not completed"},
      {"timer", Messages::pattern_reason_timer_creation_failed(), "Failed to create pattern benchmark timer."},
      {"calibration", Messages::pattern_reason_calibration_or_accounting_failed(),
       "pattern calibration or byte accounting failed"},
      {"random workload", Messages::pattern_reason_no_valid_random_workload(), "no valid random access workload"},
      {"stride transition", Messages::pattern_reason_stride_transition_unavailable(),
       "buffer cannot provide a valid stride transition"},
      {"copy overflow", Messages::pattern_reason_copy_accounting_overflow(), "copy payload byte accounting overflow"},
      {"strided timing", Messages::pattern_reason_invalid_strided_timing(), "invalid strided timing result"},
      {"work-plan byte overflow", Messages::pattern_reason_work_plan_byte_overflow(),
       "strided work-plan byte accounting overflow"},
      {"work-plan parameters", Messages::pattern_reason_invalid_work_plan_parameters(),
       "invalid strided work-plan parameters"},
      {"stride sum overflow", Messages::pattern_reason_stride_access_sum_overflow(),
       "stride and access-size sum overflows"},
      {"two accesses", Messages::pattern_reason_buffer_lacks_two_strided_accesses(),
       "buffer cannot provide two strided accesses"},
      {"worker partition", Messages::pattern_reason_no_valid_strided_worker_partition(),
       "no valid worker partition contains a stride transition"},
      {"pass limit", Messages::pattern_reason_work_plan_pass_limit(), "strided work plan exceeds executor pass limit"},
      {"total overflow", Messages::pattern_reason_work_plan_total_overflow(),
       "strided work-plan total accounting overflow"},
      {"allocation", Messages::pattern_reason_buffers_allocation_failed(), "pattern buffer allocation failed"},
      {"initialization", Messages::pattern_reason_buffers_initialization_failed(),
       "pattern buffer initialization failed"},
      {"loop execution", Messages::pattern_reason_loop_execution_failed(), "pattern loop execution failed"},
      {"loop interrupted", Messages::pattern_reason_loop_interrupted(), "pattern loop interrupted by user"},
      {"loop incomplete", Messages::pattern_reason_loop_incomplete(), "pattern loop has incomplete measurements"},
      {"invalid measurement", Messages::pattern_reason_invalid_measurement(),
       "pattern loop contains an invalid measurement"},
      {"loops remain", Messages::pattern_reason_loops_remain(), "pattern benchmark loops remain"},
      {"loop exception", Messages::pattern_reason_loop_exception("boom"), "pattern loop threw an exception: boom"},
      {"unknown loop exception", Messages::pattern_reason_unknown_loop_exception(),
       "pattern loop threw an unknown exception"},
      {"coordinator exception", Messages::pattern_reason_coordinator_exception("boom"),
       "pattern coordinator threw an exception: boom"},
      {"unknown coordinator exception", Messages::pattern_reason_unknown_coordinator_exception(),
       "pattern coordinator threw an unknown exception"},
  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, ConfigPatternAutomaticIterations) {
  EXPECT_EQ(Messages::config_pattern_iterations_auto(0.150, 0.100, 0.250),
            "Pattern Passes: automatic duration calibration (target 150 ms; intended window 100-250 ms)");
}

TEST(MessagesFormattingTest, ConfigBenchmarkAutomaticIterations) {
  EXPECT_EQ(Messages::config_benchmark_iterations_auto(0.150, 0.100, 0.250),
            "Bandwidth Passes: automatic duration calibration (target 150 ms; intended window 100-250 ms)");
}

TEST(MessagesFormattingTest, ConfigLatencyCalibration) {
  EXPECT_EQ(Messages::config_latency_calibration(0.250, 0.100, 0.300, 16),
            "Latency Headline: automatic continuous-pass calibration (target 250 ms; intended window 100-300 ms; "
            "minimum 16 complete cycles)");
}
