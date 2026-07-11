// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_bandwidth_messages.cpp
 * @brief Centralized GPU memory-bandwidth CLI and report text
 */

#include "output/console/messages/messages_api.h"

#include <iomanip>
#include <sstream>

#include "core/config/constants.h"

namespace Messages {

const std::string& error_gpu_bandwidth_must_be_used_alone() {
  static const std::string message =
      "--gpu-bandwidth allows only optional -b/--buffer-size <MB>, "
      "-i/--iterations <count>, -r/--count <count>, -o/--output <file>, "
      "--seed <uint64>, and -h/--help (no other options allowed)";
  return message;
}

std::string error_gpu_buffer_size_below_minimum(unsigned long requested_mb,
                                                unsigned long minimum_mb) {
  std::ostringstream message;
  message << "GPU buffer-size must be at least " << minimum_mb
          << " MB (got " << requested_mb << " MB)";
  return message.str();
}

std::string error_gpu_iterations_exceed_limit(size_t requested,
                                              size_t maximum) {
  std::ostringstream message;
  message << "GPU iterations exceed the exact-work guardrail (requested "
          << requested << ", maximum " << maximum << ")";
  return message.str();
}

std::string error_gpu_run_failed(const std::string& reason_code) {
  return "GPU memory bandwidth benchmark failed (reason_code=" +
         reason_code + ")";
}

const std::string& gpu_reason_positive_integer() {
  static const std::string reason = "must be a positive integer";
  return reason;
}

const std::string& gpu_reason_nonnegative_unsigned_long() {
  static const std::string reason =
      "must be a non-negative integer representable as unsigned long";
  return reason;
}

const std::string& gpu_reason_loop_count_out_of_range() {
  static const std::string reason = "out of range for loop count";
  return reason;
}

const std::string& msg_running_gpu_bandwidth() {
  static const std::string message = "Running GPU memory bandwidth benchmark...";
  return message;
}

const std::string& gpu_unknown_device_name() {
  static const std::string name = "unknown Apple GPU";
  return name;
}

std::string gpu_usage_options(const std::string& prog_name) {
  std::ostringstream usage;
  usage << "Usage: " << prog_name << " --gpu-bandwidth [options]\n"
        << "Options for standalone GPU memory bandwidth mode:\n"
        << "  -G, --gpu-bandwidth   Measure Metal GPU memory read/write/copy bandwidth.\n"
        << "  -b, --buffer-size <MB>\n"
        << "                        Size of each private GPU buffer (default: "
        << Constants::GPU_DEFAULT_BUFFER_SIZE_MB << " MB; minimum: "
        << Constants::GPU_MIN_BUFFER_SIZE_MB << " MB).\n"
        << "  -i, --iterations <count>\n"
        << "                        Exact full-buffer pass count. When omitted, each operation\n"
        << "                        calibrates toward 150 ms in a 100-250 ms window.\n"
        << "  -r, --count <count>   Number of balanced read/write/copy loops (default: "
        << Constants::GPU_DEFAULT_LOOP_COUNT << ").\n"
        << "      --seed <uint64>   Reproducible base seed; generated once when omitted.\n"
        << "  -o, --output <file>   Atomically checkpoint GPU schema 1 JSON after each result.\n"
        << "  -h, --help            Show this GPU-mode help and exit\n";
  return usage.str();
}

std::string report_gpu_bandwidth_header(const std::string& device_name,
                                        size_t loop_count,
                                        bool median_headline) {
  std::ostringstream report;
  report << "GPU memory bandwidth (" << device_name
         << ", private/tracked, " << loop_count << " loop"
         << (loop_count == 1 ? "" : "s") << "; headline: "
         << (median_headline ? "median" : "single measurement") << ")";
  return report.str();
}

std::string report_gpu_bandwidth_value(const std::string& operation,
                                       double value_gb_s,
                                       bool aggregate_copy_payload) {
  std::ostringstream report;
  report << "  " << std::left << std::setw(7) << (operation + ":")
         << std::right << std::fixed << std::setprecision(2)
         << value_gb_s << " GB/s  ("
         << (aggregate_copy_payload ? "aggregate read + write payload"
                                    : "effective compute payload")
         << ")";
  return report.str();
}

std::string report_gpu_bandwidth_repeatability(double read_cv_pct,
                                               double write_cv_pct,
                                               double copy_cv_pct,
                                               bool available) {
  if (!available) {
    return "  Repeatability: insufficient samples (requires at least 3 per operation)";
  }
  std::ostringstream report;
  report << "  Repeatability: read CV " << std::fixed << std::setprecision(2)
         << read_cv_pct << "%, write CV " << write_cv_pct
         << "%, copy CV " << copy_cv_pct << "%";
  return report.str();
}

const std::string& report_gpu_bandwidth_interpretation_note() {
  static const std::string message =
      "  Note: copy is aggregate read+write throughput; DRAM residency is unverified. "
      "Results can remain cache/dispatch-dominant even at the 64 MB methodology minimum.";
  return message;
}

std::string warning_gpu_high_cv(const std::string& operation,
                                double cv_pct,
                                double threshold_pct) {
  std::ostringstream warning;
  warning << "GPU " << operation << " repeatability CV " << std::fixed
          << std::setprecision(2) << cv_pct << "% exceeds "
          << threshold_pct << "%";
  return warning.str();
}

const std::string& warning_gpu_order_not_balanced() {
  static const std::string message =
      "GPU operation order is not fully balanced across completed loops";
  return message;
}

std::string warning_gpu_duration_quality(const std::string& operation,
                                         const std::string& quality) {
  return "GPU " + operation + " duration quality is " + quality;
}

const std::string& warning_gpu_environment_not_nominal() {
  static const std::string message =
      "GPU result environment is not reference-eligible (thermal state or Low Power Mode)";
  return message;
}

const std::string& warning_gpu_recommended_working_set_exceeded() {
  static const std::string message =
      "GPU allocation exceeds Metal's advisory recommended working-set size";
  return message;
}

}  // namespace Messages
