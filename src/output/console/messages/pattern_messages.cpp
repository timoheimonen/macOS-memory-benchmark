// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file pattern_messages.cpp
 * @brief Pattern benchmark messages
 *
 * Provides implementations for pattern benchmark message generation functions.
 * Includes formatted messages for different memory access patterns (sequential,
 * strided, random) and pattern efficiency analysis metrics.
 */

#include "messages_api.h"

#include <iomanip>
#include <sstream>

namespace Messages {

// --- Pattern Benchmark Messages ---
const std::string& pattern_sequential_forward() {
  static const std::string msg = "Sequential Forward:";
  return msg;
}

const std::string& pattern_sequential_reverse() {
  static const std::string msg = "Sequential Reverse:";
  return msg;
}

std::string pattern_strided(const std::string& stride_name) {
  return "Strided (" + stride_name + "):";
}

const std::string& pattern_random_uniform() {
  static const std::string msg = "Random Uniform:";
  return msg;
}

const std::string& pattern_cache_line_64b() {
  static const std::string msg = "64 B stride";
  return msg;
}

const std::string& pattern_page_4096b() {
  static const std::string msg = "4096 B stride";
  return msg;
}

const std::string& pattern_page_16384b() {
  static const std::string msg = "16 KiB stride";
  return msg;
}

const std::string& pattern_superpage_2mb() {
  static const std::string msg = "2 MiB stride";
  return msg;
}

const std::string& pattern_separator() {
  static const std::string msg = "\n================================\n\n";
  return msg;
}

const std::string& pattern_read_label() {
  static const std::string msg = "  Read : ";
  return msg;
}

const std::string& pattern_write_label() {
  static const std::string msg = "  Write: ";
  return msg;
}

const std::string& pattern_copy_label() {
  static const std::string msg = "  Copy : ";
  return msg;
}

const std::string& pattern_bandwidth_unit() {
  static const std::string msg = " GB/s";
  return msg;
}

std::string pattern_measurement_unavailable(const std::string& status,
                                            const std::string& reason) {
  return "N/A [" + status + (reason.empty() ? "" : ": " + reason) + "]";
}

std::string warning_pattern_measurement_noisy(const std::string& metric,
                                              double cv_pct,
                                              double threshold_pct) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1)
      << "Noisy pattern measurement: " << metric << " CV " << cv_pct
      << "% exceeds " << threshold_pct << "%";
  return oss.str();
}

const std::string& pattern_reason_measurement_not_completed() {
  static const std::string msg = "measurement not completed";
  return msg;
}

const std::string& pattern_reason_timer_creation_failed() {
  static const std::string msg = "Failed to create pattern benchmark timer.";
  return msg;
}

const std::string& pattern_reason_calibration_or_accounting_failed() {
  static const std::string msg = "pattern calibration or byte accounting failed";
  return msg;
}

const std::string& pattern_reason_no_valid_random_workload() {
  static const std::string msg = "no valid random access workload";
  return msg;
}

const std::string& pattern_reason_stride_transition_unavailable() {
  static const std::string msg = "buffer cannot provide a valid stride transition";
  return msg;
}

const std::string& pattern_reason_copy_accounting_overflow() {
  static const std::string msg = "copy payload byte accounting overflow";
  return msg;
}

const std::string& pattern_reason_invalid_strided_timing() {
  static const std::string msg = "invalid strided timing result";
  return msg;
}

const std::string& pattern_reason_work_plan_byte_overflow() {
  static const std::string msg = "strided work-plan byte accounting overflow";
  return msg;
}

const std::string& pattern_reason_invalid_work_plan_parameters() {
  static const std::string msg = "invalid strided work-plan parameters";
  return msg;
}

const std::string& pattern_reason_stride_access_sum_overflow() {
  static const std::string msg = "stride and access-size sum overflows";
  return msg;
}

const std::string& pattern_reason_buffer_lacks_two_strided_accesses() {
  static const std::string msg = "buffer cannot provide two strided accesses";
  return msg;
}

const std::string& pattern_reason_no_valid_strided_worker_partition() {
  static const std::string msg =
      "no valid worker partition contains a stride transition";
  return msg;
}

const std::string& pattern_reason_work_plan_pass_limit() {
  static const std::string msg = "strided work plan exceeds executor pass limit";
  return msg;
}

const std::string& pattern_reason_work_plan_total_overflow() {
  static const std::string msg = "strided work-plan total accounting overflow";
  return msg;
}

} // namespace Messages
