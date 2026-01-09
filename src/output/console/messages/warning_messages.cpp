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
 * @file warning_messages.cpp
 * @brief Warning message implementations
 *
 * Provides implementations for warning message generation functions used throughout
 * the benchmark application. Includes formatted warning messages for memory limits,
 * system configuration, cache detection, and non-fatal runtime issues.
 */

#include "messages.h"
#include <sstream>

namespace Messages {

// --- Warning Messages ---
const std::string& warning_prefix() {
  static const std::string msg = "Warning: ";
  return msg;
}

const std::string& warning_cannot_get_memory() {
  static const std::string msg = "Cannot get available memory. Using fallback limit.";
  return msg;
}

std::string warning_buffer_size_exceeds_limit(unsigned long requested_mb, unsigned long limit_mb) {
  std::ostringstream oss;
  oss << "Requested buffer size (" << requested_mb << " MB) > limit ("
      << limit_mb << " MB). Using limit.";
  return oss.str();
}

std::string warning_qos_failed(int code) {
  std::ostringstream oss;
  oss << "Failed to set QoS class for main thread (code: " << code << ")";
  return oss.str();
}

std::string warning_stride_not_aligned(size_t stride) {
  std::ostringstream oss;
  oss << "stride (" << stride << ") is not 32-byte aligned, may cause issues";
  return oss.str();
}

std::string warning_qos_failed_worker_thread(int code) {
  std::ostringstream oss;
  oss << "Failed to set QoS class for warmup worker thread (code: " << code << ")";
  return oss.str();
}

std::string warning_madvise_random_failed(const std::string& buffer_name, const std::string& error_msg) {
  std::ostringstream oss;
  oss << "madvise(MADV_RANDOM) failed for " << buffer_name
      << " (non-fatal, continuing with regular allocation): " << error_msg;
  return oss.str();
}

const std::string& warning_core_count_detection_failed() {
  static const std::string msg = "Failed to detect core count, defaulting to 1.";
  return msg;
}

const std::string& warning_mach_host_self_failed() {
  static const std::string msg = "Failed to get mach_host_self(). Cannot determine available memory.";
  return msg;
}

std::string warning_host_page_size_failed(const std::string& error_details) {
  std::ostringstream oss;
  oss << "Failed to get host_page_size(): " << error_details
      << ". Cannot determine available memory.";
  return oss.str();
}

std::string warning_host_statistics64_failed(const std::string& error_details) {
  std::ostringstream oss;
  oss << "Failed to get host_statistics64(): " << error_details
      << ". Cannot determine available memory.";
  return oss.str();
}

const std::string& warning_l1_cache_size_detection_failed() {
  static const std::string msg = "Could not detect L1 cache size, using fallback: 128 KB";
  return msg;
}

const std::string& warning_l2_cache_size_detection_failed_m1() {
  static const std::string msg = "Could not detect L2 cache size, using M1 fallback: 12 MB";
  return msg;
}

const std::string& warning_l2_cache_size_detection_failed_m2_m3_m4_m5() {
  static const std::string msg = "Could not detect L2 cache size, using M2/M3/M4/M5 fallback: 16 MB";
  return msg;
}

const std::string& warning_l2_cache_size_detection_failed_generic() {
  static const std::string msg = "Could not detect L2 cache size, using generic fallback: 16 MB";
  return msg;
}

std::string warning_threads_capped(int requested, int max_cores) {
  std::ostringstream oss;
  oss << "Requested threads (" << requested << ") exceeds maximum available cores ("
      << max_cores << "). Using " << max_cores << " threads.";
  return oss.str();
}

} // namespace Messages

