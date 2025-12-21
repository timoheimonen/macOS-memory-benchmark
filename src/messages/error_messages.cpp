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
#include "messages/messages.h"
#include <sstream>

namespace Messages {

// --- Error Messages ---
const std::string& error_prefix() {
  static const std::string msg = "Error: ";
  return msg;
}

std::string error_missing_value(const std::string& option) {
  return "Missing value for " + option;
}

std::string error_invalid_value(const std::string& option, const std::string& value, const std::string& reason) {
  return "Invalid value for " + option + ": " + value + " (" + reason + ")";
}

std::string error_unknown_option(const std::string& option) {
  return "Unknown option: " + option;
}

std::string error_buffer_size_calculation(unsigned long size_mb) {
  std::ostringstream oss;
  oss << "Buffer size calculation error (" << size_mb << " MB).";
  return oss.str();
}

std::string error_buffer_size_too_small(size_t size_bytes) {
  std::ostringstream oss;
  oss << "Final buffer size (" << size_bytes << " bytes) is too small.";
  return oss.str();
}

std::string error_cache_size_invalid(long long min_kb, long long max_kb, long long max_mb) {
  std::ostringstream oss;
  oss << "cache-size invalid (must be between " << min_kb << " KB and " << max_kb 
      << " KB (" << max_mb << " MB))";
  return oss.str();
}

const std::string& error_iterations_invalid() {
  static const std::string msg = "iterations invalid";
  return msg;
}

const std::string& error_buffersize_invalid() {
  static const std::string msg = "buffersize invalid";
  return msg;
}

const std::string& error_count_invalid() {
  static const std::string msg = "count invalid";
  return msg;
}

const std::string& error_latency_samples_invalid() {
  static const std::string msg = "latency-samples invalid";
  return msg;
}

std::string error_mmap_failed(const std::string& buffer_name) {
  return "mmap failed for " + buffer_name;
}

std::string error_madvise_failed(const std::string& buffer_name) {
  return "madvise failed for " + buffer_name;
}

std::string error_munmap_failed() {
  return "munmap failed in MmapDeleter";
}

std::string error_sysctlbyname_failed(const std::string& operation, const std::string& key) {
  return "sysctlbyname (" + operation + ") failed for " + key;
}

std::string error_mach_timebase_info_failed(const std::string& error_details) {
  return "mach_timebase_info failed: " + error_details;
}

std::string error_benchmark_tests(const std::string& error) {
  return "Error during benchmark tests: " + error;
}

std::string error_benchmark_loop(int loop, const std::string& error) {
  std::ostringstream oss;
  oss << "Error during benchmark loop " << loop << ": " << error;
  return oss.str();
}

std::string error_json_parse_failed(const std::string& error_details) {
  return "JSON parsing failed: " + error_details;
}

std::string error_json_file_read_failed(const std::string& file_path, const std::string& error_details) {
  return "Failed to read JSON file \"" + file_path + "\": " + error_details;
}

std::string error_file_write_failed(const std::string& file_path, const std::string& error_details) {
  return "Failed to write file \"" + file_path + "\": " + error_details;
}

std::string error_file_permission_denied(const std::string& file_path) {
  return "Permission denied: cannot write to \"" + file_path + "\"";
}

std::string error_file_directory_creation_failed(const std::string& dir_path, const std::string& error_details) {
  return "Failed to create directory \"" + dir_path + "\": " + error_details;
}

std::string error_stride_too_small() {
  return "stride must be >= 32 bytes";
}

std::string error_stride_too_large(size_t stride, size_t buffer_size) {
  std::ostringstream oss;
  oss << "stride (" << stride << ") must be <= buffer size (" << buffer_size << ")";
  return oss.str();
}

std::string error_indices_empty() {
  return "indices vector is empty";
}

std::string error_index_out_of_bounds(size_t index, size_t index_value, size_t buffer_size) {
  std::ostringstream oss;
  oss << "index " << index << " (" << index_value << ") exceeds buffer size " << buffer_size;
  return oss.str();
}

std::string error_index_not_aligned(size_t index, size_t index_value) {
  std::ostringstream oss;
  oss << "index " << index << " (" << index_value << ") is not 32-byte aligned";
  return oss.str();
}

std::string error_buffer_too_small_strided(size_t min_bytes) {
  std::ostringstream oss;
  oss << "Buffer too small for strided access (minimum " << min_bytes << " bytes)";
  return oss.str();
}

std::string error_no_iterations_strided() {
  return "No iterations possible for strided pattern (buffer too small)";
}

} // namespace Messages

