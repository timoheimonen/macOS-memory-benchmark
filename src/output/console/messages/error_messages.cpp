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
#include "messages.h"
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

std::string error_duplicate_option(const std::string& option) {
  return "Duplicate option: " + option;
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

std::string error_iterations_invalid(long long value, long long min_val, long long max_val) {
  std::ostringstream oss;
  oss << "iterations invalid (must be between " << min_val << " and " << max_val
      << ", got " << value << ")";
  return oss.str();
}

std::string error_buffersize_invalid(long long value, unsigned long max_val) {
  std::ostringstream oss;
  oss << "buffersize invalid (must be > 0 and <= " << max_val << ", got " << value << ")";
  return oss.str();
}

std::string error_count_invalid(long long value, long long min_val, long long max_val) {
  std::ostringstream oss;
  oss << "count invalid (must be between " << min_val << " and " << max_val
      << ", got " << value << ")";
  return oss.str();
}

std::string error_latency_samples_invalid(long long value, long long min_val, long long max_val) {
  std::ostringstream oss;
  oss << "latency-samples invalid (must be between " << min_val << " and " << max_val
      << ", got " << value << ")";
  return oss.str();
}

std::string error_threads_invalid(long long value, long long min_val, long long max_val) {
  std::ostringstream oss;
  oss << "threads invalid (must be between " << min_val << " and " << max_val
      << ", got " << value << ")";
  return oss.str();
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

std::string error_buffer_size_zero(const std::string& buffer_name) {
  return "Buffer size is zero for " + buffer_name;
}

const std::string& error_main_buffer_size_zero() {
  static const std::string msg = "Main buffer size is zero";
  return msg;
}

const std::string& error_buffer_size_overflow_calculation() {
  static const std::string msg = "Buffer size too large, would overflow when calculating total memory";
  return msg;
}

const std::string& error_total_memory_overflow() {
  static const std::string msg = "Total memory requirement would overflow";
  return msg;
}

const std::string& error_main_buffers_not_allocated() {
  static const std::string msg = "Main buffers not allocated";
  return msg;
}

const std::string& error_custom_buffer_not_allocated() {
  static const std::string msg = "Custom buffer not allocated but size > 0";
  return msg;
}

const std::string& error_l1_buffer_not_allocated() {
  static const std::string msg = "L1 buffer not allocated but size > 0";
  return msg;
}

const std::string& error_l2_buffer_not_allocated() {
  static const std::string msg = "L2 buffer not allocated but size > 0";
  return msg;
}

const std::string& error_custom_bandwidth_buffers_not_allocated() {
  static const std::string msg = "Custom bandwidth buffers not allocated but size > 0";
  return msg;
}

const std::string& error_l1_bandwidth_buffers_not_allocated() {
  static const std::string msg = "L1 bandwidth buffers not allocated but size > 0";
  return msg;
}

const std::string& error_l2_bandwidth_buffers_not_allocated() {
  static const std::string msg = "L2 bandwidth buffers not allocated but size > 0";
  return msg;
}

const std::string& error_buffer_pointer_null_latency_chain() {
  static const std::string msg = "Buffer pointer is null for latency chain setup";
  return msg;
}

const std::string& error_stride_zero_latency_chain() {
  static const std::string msg = "Stride is zero for latency chain setup";
  return msg;
}

std::string error_buffer_stride_invalid_latency_chain(size_t num_pointers, size_t buffer_size, size_t stride) {
  std::ostringstream oss;
  oss << "Buffer/stride invalid for latency chain setup (num_pointers=" << num_pointers 
      << "). Buffer size: " << buffer_size << ", Stride: " << stride;
  return oss.str();
}

const std::string& error_buffer_too_small_for_pointers() {
  static const std::string msg = "Buffer size too small to store pointers";
  return msg;
}

std::string error_offset_exceeds_bounds(size_t offset, size_t max_offset) {
  std::ostringstream oss;
  oss << "Calculated offset exceeds buffer bounds (offset=" << offset << ", max=" << max_offset << ")";
  return oss.str();
}

std::string error_next_pointer_offset_exceeds_bounds(size_t offset, size_t max_offset) {
  std::ostringstream oss;
  oss << "Next pointer offset exceeds buffer bounds (offset=" << offset << ", max=" << max_offset << ")";
  return oss.str();
}

const std::string& error_source_buffer_null() {
  static const std::string msg = "Source buffer pointer is null";
  return msg;
}

const std::string& error_destination_buffer_null() {
  static const std::string msg = "Destination buffer pointer is null";
  return msg;
}

const std::string& error_buffer_size_zero_generic() {
  static const std::string msg = "Buffer size is zero";
  return msg;
}

const std::string& error_calculated_custom_buffer_size_zero() {
  static const std::string msg = "Calculated custom buffer size is zero";
  return msg;
}

const std::string& error_l1_cache_size_overflow() {
  static const std::string msg = "L1 cache size too large, would overflow when calculating buffer size";
  return msg;
}

const std::string& error_l2_cache_size_overflow() {
  static const std::string msg = "L2 cache size too large, would overflow when calculating buffer size";
  return msg;
}

const std::string& error_calculated_l1_buffer_size_zero() {
  static const std::string msg = "Calculated L1 buffer size is zero";
  return msg;
}

const std::string& error_calculated_l2_buffer_size_zero() {
  static const std::string msg = "Calculated L2 buffer size is zero";
  return msg;
}

const std::string& error_latency_access_count_overflow() {
  static const std::string msg = "Calculated latency access count would overflow";
  return msg;
}

const std::string& error_latency_access_count_negative() {
  static const std::string msg = "Calculated latency access count is negative";
  return msg;
}

const std::string& error_incompatible_flags() {
  static const std::string msg = "-only-bandwidth and -only-latency are mutually exclusive (cannot use both together)";
  return msg;
}

const std::string& error_only_flags_with_patterns() {
  static const std::string msg = "-only-bandwidth and -only-latency cannot be used with -patterns (pattern benchmarks are a separate execution mode)";
  return msg;
}

const std::string& error_only_bandwidth_with_cache_size() {
  static const std::string msg = "-only-bandwidth cannot be used with -cache-size (cache-size is only relevant for latency tests)";
  return msg;
}

const std::string& error_only_bandwidth_with_latency_samples() {
  static const std::string msg = "-only-bandwidth cannot be used with -latency-samples (latency-samples is only relevant for latency tests)";
  return msg;
}

const std::string& error_only_latency_with_buffersize() {
  static const std::string msg = "-only-latency cannot be used with -buffersize (buffersize is only relevant for bandwidth tests)";
  return msg;
}

const std::string& error_only_latency_with_iterations() {
  static const std::string msg = "-only-latency cannot be used with -iterations (iterations is only relevant for bandwidth tests)";
  return msg;
}

} // namespace Messages

