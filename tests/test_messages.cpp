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
#include <optional>
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

}  // namespace

TEST(MessagesErrorTest, ValidationErrorsHaveExactOutput) {
  expect_exact_messages({
      {"buffer calculation", Messages::error_buffer_size_calculation(1024), "Buffer size calculation error (1024 MB)."},
      {"cache size", Messages::error_cache_size_invalid(16, 524288, 512),
       "cache-size invalid (must be between 16 KB and 524288 KB (512 MB))"},
      {"iterations negative", Messages::error_iterations_invalid(-5, 1, 2147483647),
       "iterations invalid (must be between 1 and 2147483647, got -5)"},
      {"buffer size negative", Messages::error_buffersize_invalid(-100, 18446744073709551615UL),
       "buffer-size invalid (must be >= 0 and <= 18446744073709551615, got -100)"},
      {"count zero", Messages::error_count_invalid(0, 1, 2147483647),
       "count invalid (must be between 1 and 2147483647, got 0)"},
      {"latency samples zero", Messages::error_latency_samples_invalid(0, 1, 2147483647),
       "latency-samples invalid (must be between 1 and 2147483647, got 0)"},
      {"latency stride range", Messages::error_latency_stride_invalid(0, 1, 9223372036854775807LL),
       "latency-stride-bytes invalid (must be between 1 and 9223372036854775807, got 0)"},
      {"latency stride alignment", Messages::error_latency_stride_alignment(65, 8),
       "latency-stride-bytes must be a multiple of 8 bytes, got 65"},
      {"latency locality range", Messages::error_latency_tlb_locality_invalid(-1, 1024),
       "latency-tlb-locality-kb invalid (must be >= 0 and <= 1024, got -1)"},
      {"latency locality page multiple", Messages::error_latency_tlb_locality_page_multiple(10, 16),
       "latency-tlb-locality-kb must be a multiple of system page size (16 KB), got 10 KB"},
      {"missing iterations", Messages::error_missing_value("--iterations"), "Missing value for --iterations"},
      {"unknown short", Messages::error_unknown_option("-unknown"), "Unknown option: -unknown"},
      {"option does not take value", Messages::error_option_does_not_take_value("--patterns"),
       "Option --patterns does not take a value"},
      {"option not allowed in mode",
       Messages::error_option_not_allowed_in_mode("--threads", Messages::error_gpu_bandwidth_must_be_used_alone()),
       "Unsupported option '--threads': " + Messages::error_gpu_bandwidth_must_be_used_alone()},
      {"missing primary mode", Messages::error_missing_primary_mode(),
       "A primary mode option is required (--cpu-bandwidth, --cpu-latency, "
       "--patterns, --analyze-tlb, --analyze-core2core, --gpu-bandwidth, or "
       "--llm-memory)"},
      {"analyze tlb alone", Messages::error_analyze_tlb_must_be_used_alone(),
       "--analyze-tlb allows only optional -o/--output <target>, -s/--latency-stride-bytes <bytes>, "
       "-m/--latency-chain-mode <mode>, -D/--tlb-density <low|medium|high>, --seed <uint64>, "
       "-S/--sweep <key=...>, -X/--sweep-max-runs <count>, and -h/--help (no other options allowed)"},
      {"signed invalid reason", Messages::reason_signed_integer_invalid(),
       "must be an integer without whitespace, a plus sign, or trailing characters"},
      {"signed range reason", Messages::reason_signed_integer_out_of_range(), "out of range"},
      {"unsigned invalid reason", Messages::reason_unsigned_integer_invalid(),
       "must be an unsigned 64-bit integer without whitespace, a sign, or trailing characters"},
      {"unsigned range reason", Messages::reason_unsigned_integer_out_of_range(),
       "out of range for an unsigned 64-bit integer"},
      {"positive integer reason", Messages::reason_positive_integer(), "must be a positive integer"},
      {"positive int range reason", Messages::reason_positive_int_range(2147483647),
       "must be between 1 and 2147483647"},
      {"tlb density reason", Messages::reason_tlb_density_invalid(), "must be one of: low, medium, high"},
      {"sweep syntax reason", Messages::reason_sweep_syntax(), "sweep must use key=value1,value2 syntax"},
      {"sweep empty reason", Messages::reason_sweep_empty_value(), "sweep value list cannot contain empty values"},
      {"unsupported sweep parameter", Messages::reason_unsupported_sweep_parameter("foo"),
       "unsupported sweep parameter: foo"},
      {"unsupported core2core sweep parameter", Messages::reason_unsupported_core_to_core_sweep_parameter("bar"),
       "unsupported core-to-core sweep parameter: bar"},
      {"invalid iterations", Messages::error_invalid_value("--iterations", "abc", "must be a number"),
       "Invalid value for --iterations: abc (must be a number)"},
      {"mmap source", Messages::error_mmap_failed("src_buffer"), "mmap failed for src_buffer"},
      {"qos code 1", Messages::warning_qos_failed(1), "Failed to set QoS class for main thread (code: 1)"},
      {"missing parameter", Messages::error_sweep_requires_parameter(),
       "--sweep requires at least one parameter specification"},
      {"missing output", Messages::error_sweep_requires_output(),
       "--sweep requires --output <target> for the combined JSON result"},
      {"run cap", Messages::error_sweep_too_many_runs(12, 10),
       "Sweep would generate 12 runs, exceeding --sweep-max-runs 10"},
      {"parameter not allowed", Messages::error_sweep_parameter_not_allowed("cache-size", "--patterns"),
       "Sweep parameter 'cache-size' is not allowed with --patterns"},
      {"option conflict", Messages::error_sweep_option_conflict("--context-tokens", "context-tokens"),
       "Cannot specify both --context-tokens and --sweep context-tokens"},
      {"locality mode", Messages::error_latency_chain_mode_requires_locality("same-random-in-box-increasing-box"),
       "latency-chain-mode 'same-random-in-box-increasing-box' requires --latency-tlb-locality-kb > 0"},
      {"locality minimum", Messages::error_latency_tlb_locality_too_small_for_stride(4096, 4096),
       "latency-tlb-locality-kb too small for latency-stride-bytes (requires at least 8192 bytes, got 4096 bytes)"},
      {"TLB stride", Messages::error_analyze_tlb_stride_exceeds_page(32768, 16384),
       "--analyze-tlb latency-stride-bytes must not exceed the system page size (16384 bytes), got 32768 bytes"},
      {"duplicate sweep", Messages::error_duplicate_sweep_parameter("latency-stride-bytes"),
       "sweep parameter specified more than once: latency-stride-bytes"},
  });
}

TEST(MessagesErrorTest, JsonCommandBoundaryFailuresHaveExactOutput) {
  expect_exact_messages({
      {"error_json_output_initialization_failed", Messages::error_json_output_initialization_failed("cwd unavailable"),
       "JSON output initialization failed: cwd unavailable"},
      {"error_json_output_initialization_failed empty reason", Messages::error_json_output_initialization_failed(""),
       "JSON output initialization failed: unknown exception"},
      {"error_json_payload_construction_failed", Messages::error_json_payload_construction_failed("allocation failed"),
       "JSON payload construction failed: allocation failed"},
      {"error_json_payload_construction_failed empty reason", Messages::error_json_payload_construction_failed(""),
       "JSON payload construction failed: unknown exception"},
      {"error_command_execution_exception",
       Messages::error_command_execution_exception("TLB analysis", "allocation failed"),
       "TLB analysis failed with unexpected "
       "exception: allocation failed"},
      {"error_command_execution_exception empty reason",
       Messages::error_command_execution_exception("Core-to-core analysis", ""),
       "Core-to-core analysis failed with unexpected exception: unknown exception"},
      {"error_sweep_nested_run_exception", Messages::error_sweep_nested_run_exception("allocator failed"),
       "Sweep nested run failed with unexpected exception: allocator failed"},
      {"error_sweep_nested_run_exception empty reason", Messages::error_sweep_nested_run_exception(""),
       "Sweep nested run failed with unexpected exception: unknown exception"},
      {"json_stdout_reason_stream_unavailable", Messages::json_stdout_reason_stream_unavailable(),
       "stdout stream is unavailable"},
      {"json_stdout_reason_size_limit_exceeded", Messages::json_stdout_reason_size_limit_exceeded(),
       "serialized JSON exceeds stream size limits"},
      {"json_stdout_reason_write_failed", Messages::json_stdout_reason_write_failed(), "write operation failed"},
      {"json_stdout_reason_flush_failed", Messages::json_stdout_reason_flush_failed(), "flush operation failed"},
      {"error_json_stdout_write_failed",
       Messages::error_json_stdout_write_failed(Messages::json_stdout_reason_flush_failed()),
       "Failed to write JSON to stdout: flush operation failed"},
      {"error_json_stdout_write_failed empty reason", Messages::error_json_stdout_write_failed(""),
       "Failed to write JSON to stdout: unknown exception"},
  });
}

TEST(MessagesTest, LlmMemoryCliMessagesHaveExactOutput) {
  const std::string expected_usage =
      "Usage: memory_benchmark --llm-memory [options]\n"
      "Options for standalone CPU/Metal synthetic LLM memory mode:\n"
      "  -M, --llm-memory       Select the memory-only LLM profile.\n"
      "      --llm-memory-backend <cpu|metal>\n"
      "                          Execution backend (default: cpu). Metal accepts both phases\n"
      "                          with contiguous or paged KV. Capability admission requires a\n"
      "                          default unified-memory Apple7-or-later device, Tier 2 argument\n"
      "                          buffers, and maxBufferLength >= 256 MiB. The selected MSL 2.3\n"
      "                          source, pipelines, and layout probe must succeed. Capability\n"
      "                          absence is unsupported; compiler, pipeline, resource, or task\n"
      "                          failure is terminal failed/invalid; no CPU fallback is performed.\n"
      "      --phase <decode|prefill>\n"
      "                          Workload phase (default: decode). CPU and Metal support both phases.\n"
      "      --weight-size-mb <MiB>\n"
      "                          Required active weight bytes per work unit, in MiB.\n"
      "      --layers <count>    Required transformer layer count.\n"
      "      --query-heads <count>\n"
      "                          Required query-head count; must be at least as large as KV heads\n"
      "                          and divisible by them. Model classification, not executed attention.\n"
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
      "                          Defines synthetic prefix rereads, not inference-kernel tiling.\n"
      "                          Rejected for decode.\n"
      "      --kv-layout <contiguous|paged>\n"
      "                          KV storage layout (default: contiguous). CPU and Metal support both layouts.\n"
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
      "  -o, --output <target>  JSON schema 2 target (or schema 1 sweep envelope with --sweep);\n"
      "                          exact - writes one final document to stdout and routes human\n"
      "                          output to stderr. Direct file targets snapshot every\n"
      "                          K=max(1,ceil(count/8)) loops plus terminal (abrupt loss bound:\n"
      "                          3K completed attempts); sweep files checkpoint after each run.\n"
      "                          An empty value disables JSON for direct commands and is invalid for sweeps.\n"
      "  -S, --sweep <key=a,b>  Run a Cartesian sweep over one parameter. Repeat for multiple\n"
      "                          parameters. Supported keys: context-tokens (decode),\n"
      "                          prompt-tokens and attention-query-tile-tokens (prefill),\n"
      "                          and kv-block-tokens (paged KV). Cannot be combined with the\n"
      "                          matching direct flag; requires --output <target>.\n"
      "  -X, --sweep-max-runs <count>\n"
      "                          Maximum generated sweep runs (default: " +
      std::to_string(Constants::DEFAULT_SWEEP_MAX_RUNS) +
      ").\n"
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
       "-r/--count <count>, --seed <uint64>, -o/--output <target>, "
       "-S/--sweep <key=...>, -X/--sweep-max-runs <count>, and "
       "-h/--help (no other options allowed)"},
      {"required option", Messages::error_llm_memory_missing_required_option("--layers"),
       "Missing required --llm-memory option: --layers"},
      {"config reason", Messages::error_llm_memory_config_invalid("query-heads-not-divisible-by-kv-heads"),
       "Invalid --llm-memory configuration "
       "(reason_code=query-heads-not-divisible-by-kv-heads)"},
      {"sweep combination invalid",
       Messages::error_llm_memory_sweep_combination_invalid(2, 4, "prompt-tokens=32, attention-query-tile-tokens=48"),
       "LLM sweep combination 2/4 (prompt-tokens=32, attention-query-tile-tokens=48) failed preflight validation"},
      {"iteration limit", Messages::error_llm_memory_iterations_exceed_limit(5, 4),
       "LLM memory iterations exceed the exact-work guardrail "
       "(requested 5, maximum 4)"},
      {"runtime failure", Messages::error_llm_memory_run_failed("checksum-mismatch"),
       "Synthetic LLM memory profile failed "
       "(reason_code=checksum-mismatch)"},
      {"paged table protection", Messages::error_llm_paged_table_protection_failed(),
       "Failed to make the paged KV block table read-only"},
      {"positive integer", Messages::llm_memory_reason_positive_integer(), "must be a positive integer"},
      {"backend", Messages::llm_memory_reason_backend(), "must be exactly cpu or metal"},
      {"KV width", Messages::llm_memory_reason_kv_element_bytes(), "must be exactly 1, 2, or 4"},
      {"phase", Messages::llm_memory_reason_phase(), "must be exactly decode or prefill"},
      {"KV layout", Messages::llm_memory_reason_kv_layout(), "must be exactly contiguous or paged"},
      {"platform size", Messages::llm_memory_reason_platform_size_range(), "out of range for a platform size"},
      {"usage", Messages::llm_memory_usage_options("memory_benchmark"), expected_usage},
      {"command name", Messages::llm_memory_command_name(), "LLM memory profile"},

  };
  expect_exact_messages(cases);
}

TEST(MessagesTest, LlmDistributionReportsAcceptedPopulationAndDescriptiveSpread) {
  EXPECT_EQ(Messages::report_llm_memory_distribution(7, 123.456, 98.766, 150.126, 6.257, 3.141),
            "    n=7, median=123.46 GB/s, min=98.77, max=150.13, CV=6.26%, MAD=3.14 GB/s");
  EXPECT_EQ(Messages::report_llm_memory_distribution(1, 42.0, 42.0, 42.0, 0.0, 0.0),
            "    n=1, median=42.00 GB/s, min=42.00, max=42.00, CV=0.00%, MAD=0.00 GB/s");
}

TEST(MessagesTest, LlmPrefillModelContextDistinguishesUnavailableFromKnownZero) {
  const std::string prefix =
      "  Prompt tokens (P):                 5\n"
      "  Attention query tile tokens (Q):  2\n"
      "  Attention query tiles (C):        3\n"
      "  Prefix token visits / sequence:   11\n";
  EXPECT_EQ(Messages::report_llm_memory_prefill_geometry(5, 2, 3, 11, 15, 60, std::nullopt),
            prefix + "  Causal token pairs / sequence:    15\n"
                     "  Logical attention pairs:          60\n"
                     "  Logical attention FMA terms:      unavailable (arithmetic-overflow)");
  EXPECT_EQ(Messages::report_llm_memory_prefill_geometry(5, 2, 3, 11, std::nullopt, std::nullopt, std::nullopt),
            prefix + "  Causal token pairs / sequence:    unavailable (arithmetic-overflow)\n"
                     "  Logical attention pairs:          unavailable (arithmetic-overflow)\n"
                     "  Logical attention FMA terms:      unavailable (arithmetic-overflow)");
  // Formatting helper receives known zero separately from absent theoretical evidence.
  EXPECT_EQ(Messages::report_llm_memory_prefill_geometry(5, 2, 3, 11, 0, 0, 0),
            prefix + "  Causal token pairs / sequence:    0\n"
                     "  Logical attention pairs:          0\n"
                     "  Logical attention FMA terms:      0");
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
      {"minimum buffer", Messages::error_gpu_buffer_size_below_minimum(32, 64),
       "GPU buffer-size must be at least 64 MB (got 32 MB)"},
      {"iteration guard", Messages::error_gpu_iterations_exceed_limit(513, 512),
       "GPU iterations exceed the exact-work guardrail (requested 513, maximum 512)"},
      {"run failure", Messages::error_gpu_run_failed("test-reason"),
       "GPU memory bandwidth benchmark failed (reason_code=test-reason)"},
      {"usage", Messages::gpu_usage_options("memory_benchmark"), expected_usage},
      {"report header", Messages::report_gpu_bandwidth_header("Apple M4", 3, true),
       "GPU memory bandwidth (Apple M4, private/tracked, 3 loops; headline: median)"},
      {"copy payload", Messages::report_gpu_bandwidth_value("Copy", 123.456, true),
       "  Copy:  123.46 GB/s  (aggregate read + write payload)"},
      {"repeatability", Messages::report_gpu_bandwidth_repeatability(1.0, 2.0, 3.0, true),
       "  Repeatability: read CV 1.00%, write CV 2.00%, copy CV 3.00%"},
      {"high CV", Messages::warning_gpu_high_cv("read", 5.1, 5.0), "GPU read repeatability CV 5.10% exceeds 5.00%"},
      {"duration quality", Messages::warning_gpu_duration_quality("write", "payload-cap-below-target"),
       "GPU write duration quality is payload-cap-below-target"},

  };

  expect_exact_messages(cases);
}

TEST(MessagesFormattingTest, ProgressMessagesHaveExactOutput) {
  expect_exact_messages({
      {"ordinary duration", Messages::msg_done_total_time(123.456), "\nDone. Total execution time: 123.45600 s"},
      {"short duration", Messages::msg_done_total_time(0.001), "\nDone. Total execution time: 0.00100 s"},
      {"sweep start", Messages::msg_running_sweep(3), "\nRunning sweep with 3 runs..."},
      {"sweep progress", Messages::msg_sweep_run_progress(2, 5), "\nSweep run 2/5"},
  });
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

  struct ResourceCase {
    std::string report;
    std::vector<std::string> fragments;
  };
  const ResourceCase cases[] = {
      {Messages::report_tlb_resource_summary(1024, true, true, true, 0, 2048, 1041 * Constants::BYTES_PER_MB),
       {"1024 MiB buffer (locked)", "QoS applied", "estimated peak/budget 1041.0/2048 MiB"}},
      {Messages::report_tlb_resource_summary(256, false, true, false, 6, 512, 300 * Constants::BYTES_PER_MB),
       {"unlocked", "failed (code 6; best-effort)"}},
      {Messages::report_tlb_resource_summary(256, false, false, false, 0, 512, 300 * Constants::BYTES_PER_MB),
       {"QoS not requested"}},
  };
  for (const ResourceCase& test_case : cases) {
    for (const std::string& fragment : test_case.fragments) {
      SCOPED_TRACE(fragment);
      EXPECT_NE(test_case.report.find(fragment), std::string::npos);
    }
  }
}

TEST(MessagesFormattingTest, ReportTlbWorkEstimateAndCompletionHaveExpectedContent) {
  const std::string work = Messages::report_tlb_work_estimate("base", 15, 10, 20, 3.75, 7.5);
  EXPECT_NE(work.find("Work Estimate [base]"), std::string::npos);
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

TEST(MessagesFormattingTest, ReportTlbLargeLocalityUnavailableReasons) {
  expect_exact_messages({
      {"insufficient buffer", Messages::report_tlb_large_locality_paired_unavailable(512, 256),
       "[Large-Locality Paired Comparison]\n  Result: N/A (requires 512 MiB or larger analysis buffer, selected "
       "256 MiB)"},
      {"interrupted", Messages::report_tlb_large_locality_paired_interrupted(),
       "[Large-Locality Paired Comparison]\n  Result: N/A (analysis incomplete or comparison measurement did "
       "not complete)"},
  });
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

TEST(MessagesFormattingTest, UsageOptions) {
  struct HelpCase {
    std::string text;
    std::vector<std::string> fragments;
  };
  const HelpCase cases[] = {
      {Messages::usage_options("memory_benchmark"),
       {
           "memory_benchmark",
           "-B, --cpu-bandwidth",
           "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced",
           "-L, --cpu-latency",
           "cpu-latency-v1-arm64-calibrated-seeded-balanced",
           "100-300 ms window",
           "--iterations",
           "--buffer-size",
           "--count",
           "--analyze-tlb",
           "schema 4",
           "exact string seeds",
           "scoped counters",
           "--tlb-density",
           "default: medium",
           "--analyze-tlb: 16",
           "calibrate work toward 150 ms",
           "Reproducible workload/schedule seed for --cpu-bandwidth, --cpu-latency",
           "--gpu-bandwidth",
           Constants::GPU_METHODOLOGY_VERSION,
           "minimum buffer size is 64 MB",
           "standalone CPU/Metal synthetic LLM memory profile",
           "prefill requires --phase prefill, --prompt-tokens",
           "--attention-query-tile-tokens",
           "Both phases support contiguous",
           "llm-memory-v2-cpu-prefill-contiguous",
           Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION,
           "--analyze-core2core",
           "acquire/release token-handoff",
           "protocol, coherence, and scheduler effects",
           "core-to-core schema 2",
           "target 250 ms",
           "Defaults to 3 loops",
           "--latency-samples",
           "--latency-stride-bytes",
           "--latency-chain-mode",
           "--latency-tlb-locality-kb",
           "--compare-locality",
           "Locality-using modes require",
           "explicit global-random ignores",
           "in --patterns, requested",
           "clamped per pattern",
           "--cpu-latency and Metal --llm-memory reject --threads",
           "--cache-size",
           "--output <target>",
           "Exact - writes one final JSON document to stdout",
           "one final",
           "every result-producing direct mode and supported sweep",
           "routes human output to stderr",
           "an empty value disables JSON for direct commands",
           "invalid for sweeps",
           "Every other non-empty value is a file",
           "including ./- and names such as -G",
           "CPU bandwidth, CPU latency, patterns, GPU, and LLM-memory files retain",
           "snapshot every K=max(1,ceil(count/8))",
           "Sweep files checkpoint attempts",
           "Cannot be combined with the matching direct flag. Requires --output <target>.",
           "patterns-v1-arm64-phase-calibrated-seeded-balanced",
           "Use --patterns --help for exact rules and 11-option whitelist.",
           "-h",
           std::to_string(Constants::DEFAULT_BUFFER_SIZE_MB),
           std::to_string(Constants::DEFAULT_LOOP_COUNT),
           std::to_string(Constants::DEFAULT_LATENCY_SAMPLE_COUNT),
           std::to_string(Constants::MIN_CACHE_SIZE_KB),
           std::to_string(Constants::MAX_CACHE_SIZE_KB),
           "Platform: macOS 26 or later on Apple Silicon (ARM64).",
           "Long options that take a value accept both --option <value> and --option=<value>.",
           "and -h/--help only).",
           "-M, --llm-memory",
           "or paged KV on CPU and Metal. Metal is runtime-capability-gated",
           "and never falls back to CPU",
           "--weight-size-mb",
           "--query-heads",
           "--context-tokens",
           "memory-only interpretation",
           "JSON uses schema 2 methodologies ",
           "and at command terminal (abrupt loss bound: 3K completed attempts).",
           "llm-memory-v2-cpu-decode-contiguous",
           "llm-memory-v2-cpu-decode-paged",
           "llm-memory-v2-cpu-prefill-paged",
           "llm-memory-v2-metal-decode-contiguous",
           "llm-memory-v2-metal-decode-paged",
           "llm-memory-v2-metal-prefill-contiguous",
           "llm-memory-v2-metal-prefill-paged",
       }},
      {Messages::usage_header("1.0.0"), {"1.0.0", "Timo Heimonen", "GNU GPL", "github.com"}},
      {Messages::usage_example("memory_benchmark"),
       {"memory_benchmark", "--cpu-bandwidth", "--iterations", "--buffer-size", "--output", "Machine JSON:",
        "--cpu-latency", "--output -"}},
  };
  for (const HelpCase& test_case : cases) {
    for (const std::string& fragment : test_case.fragments) {
      SCOPED_TRACE(fragment);
      EXPECT_NE(test_case.text.find(fragment), std::string::npos);
    }
  }
  const std::string usage = Messages::usage_options("memory_benchmark");
  EXPECT_EQ(usage.find("-M, --llm-memory"), usage.rfind("-M, --llm-memory"));
}

TEST(MessagesFormattingTest, ConfigBannerHasExactOutput) {
  expect_exact_messages({
      {"header", Messages::config_header("1.0.0"), "----- macOS-memory-benchmark v1.0.0 -----"},
      {"copyright", Messages::config_copyright(), "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>"},
      {"license", Messages::config_license(),
       "This program is free software: you can redistribute it and/or modify\n"
       "it under the terms of the GNU General Public License as published by\n"
       "the Free Software Foundation, either version 3 of the License, or\n"
       "(at your option) any later version.\n"
       "This program is distributed in the hope that it will be useful,\n"
       "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
       "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
       "See <https://www.gnu.org/licenses/> for more details.\n"},
  });
}

TEST(MessagesFormattingTest, WarningBenchmarkHighCv) {
  const std::string msg = Messages::warning_benchmark_high_cv("read bandwidth", 9.25, 7.5);
  EXPECT_NE(msg.find("read bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("9.2%"), std::string::npos);
  EXPECT_NE(msg.find("7.5%"), std::string::npos);
}

TEST(MessagesFormattingTest, StatisticsCoefficientOfVariationHasExactOutput) {
  expect_exact_messages({
      {"coefficient of variation", Messages::statistics_coefficient_of_variation(12.34, 1), "  CV:      12.3%"},
  });
}
