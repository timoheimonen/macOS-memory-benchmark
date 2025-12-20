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
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include <iomanip>    // Required for std::setprecision, std::setw
#include <iostream>   // Required for std::cout, std::cerr
#include <sstream>    // Required for std::ostringstream
#include <fstream>    // Required for std::ofstream
#include <filesystem> // Required for std::filesystem::path
#include <chrono>     // Required for std::chrono
#include <ctime>      // Required for std::time, std::localtime, std::strftime
#include <cerrno>     // Required for errno
#include <cstring>    // Required for std::strerror
#include <system_error> // Required for std::errc

#include "json_output.h"
#include "benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)
#include "messages.h"   // Include centralized messages
#include "config.h"     // For BenchmarkConfig
#include "benchmark_runner.h"  // For BenchmarkStatistics
#include "json_utils.h" // JSON utility functions
#include "nlohmann/json.hpp"   // JSON library

// Helper function to add bandwidth results to JSON
// Adds read, write, and copy bandwidth with values and statistics if applicable
static void add_bandwidth_results(nlohmann::json& json_obj, 
                                   const std::vector<double>& read_values,
                                   const std::vector<double>& write_values,
                                   const std::vector<double>& copy_values) {
  if (read_values.empty()) {
    return;
  }
  
  json_obj[JsonKeys::BANDWIDTH] = nlohmann::json::object();
  
  // Read bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S][JsonKeys::VALUES] = read_values;
  if (read_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(read_values);
  }
  
  // Write bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S][JsonKeys::VALUES] = write_values;
  if (write_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(write_values);
  }
  
  // Copy bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S][JsonKeys::VALUES] = copy_values;
  if (copy_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(copy_values);
  }
}

// Helper function to add latency results to JSON
// Adds average latency and optionally sample distribution with statistics
static void add_latency_results(nlohmann::json& json_obj,
                                const std::vector<double>& average_values,
                                const std::vector<double>& samples) {
  if (average_values.empty()) {
    return;
  }
  
  json_obj[JsonKeys::LATENCY] = nlohmann::json::object();
  json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS] = nlohmann::json::object();
  json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS][JsonKeys::VALUES] = average_values;
  if (average_values.size() > 1) {
    json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS][JsonKeys::STATISTICS] = 
        calculate_json_statistics(average_values);
  }
  
  // Add sample distribution if available
  if (!samples.empty()) {
    json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS] = samples;
    if (samples.size() > 1) {
      json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_STATISTICS] = 
          calculate_json_statistics(samples);
    }
  }
}

// Build configuration JSON object
static nlohmann::json build_config_json(const BenchmarkConfig& config) {
  nlohmann::json config_json;
  config_json[JsonKeys::BUFFER_SIZE_MB] = config.buffer_size_mb;
  config_json[JsonKeys::BUFFER_SIZE_BYTES] = config.buffer_size;
  config_json[JsonKeys::ITERATIONS] = config.iterations;
  config_json[JsonKeys::LOOP_COUNT] = config.loop_count;
  config_json[JsonKeys::LATENCY_SAMPLE_COUNT] = config.latency_sample_count;
  config_json[JsonKeys::CPU_NAME] = config.cpu_name;
  config_json[JsonKeys::PERFORMANCE_CORES] = config.perf_cores;
  config_json[JsonKeys::EFFICIENCY_CORES] = config.eff_cores;
  config_json[JsonKeys::TOTAL_THREADS] = config.num_threads;
  config_json[JsonKeys::USE_CUSTOM_CACHE_SIZE] = config.use_custom_cache_size;
  
  if (config.use_custom_cache_size) {
    config_json[JsonKeys::CUSTOM_CACHE_SIZE_BYTES] = config.custom_cache_size_bytes;
    config_json[JsonKeys::CUSTOM_CACHE_SIZE_KB] = config.custom_cache_size_bytes / 1024;
    config_json[JsonKeys::CUSTOM_BUFFER_SIZE_BYTES] = config.custom_buffer_size;
  } else {
    config_json[JsonKeys::L1_CACHE_SIZE_BYTES] = config.l1_cache_size;
    config_json[JsonKeys::L2_CACHE_SIZE_BYTES] = config.l2_cache_size;
    config_json[JsonKeys::L1_BUFFER_SIZE_BYTES] = config.l1_buffer_size;
    config_json[JsonKeys::L2_BUFFER_SIZE_BYTES] = config.l2_buffer_size;
  }
  
  return config_json;
}

// Build main memory results JSON object
static nlohmann::json build_main_memory_json(const BenchmarkStatistics& stats) {
  nlohmann::json main_memory;
  
  // Add bandwidth results
  add_bandwidth_results(main_memory, 
                        stats.all_read_bw_gb_s,
                        stats.all_write_bw_gb_s,
                        stats.all_copy_bw_gb_s);
  
  // Add latency results
  add_latency_results(main_memory,
                      stats.all_average_latency_ns,
                      stats.all_main_mem_latency_samples);
  
  return main_memory;
}

// Build cache bandwidth JSON for a specific cache level
// cache_key: "l1", "l2", or "custom"
static void add_cache_bandwidth_json(nlohmann::json& cache_json,
                                     const std::string& cache_key,
                                     const std::vector<double>& read_values,
                                     const std::vector<double>& write_values,
                                     const std::vector<double>& copy_values) {
  if (read_values.empty()) {
    return;
  }
  
  if (!cache_json.contains(cache_key)) {
    cache_json[cache_key] = nlohmann::json::object();
  }
  
  add_bandwidth_results(cache_json[cache_key], read_values, write_values, copy_values);
}

// Build cache latency JSON for a specific cache level
// cache_key: "l1", "l2", or "custom"
static void add_cache_latency_json(nlohmann::json& cache_json,
                                   const std::string& cache_key,
                                   const std::vector<double>& average_values,
                                   const std::vector<double>& samples) {
  if (average_values.empty()) {
    return;
  }
  
  if (!cache_json.contains(cache_key)) {
    cache_json[cache_key] = nlohmann::json::object();
  }
  
  add_latency_results(cache_json[cache_key], average_values, samples);
}

// Build cache results JSON object
static nlohmann::json build_cache_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats) {
  nlohmann::json cache;
  
  if (config.use_custom_cache_size) {
    // Custom cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::CUSTOM,
                             stats.all_custom_read_bw_gb_s,
                             stats.all_custom_write_bw_gb_s,
                             stats.all_custom_copy_bw_gb_s);
    
    // Custom cache latency
    add_cache_latency_json(cache,
                           JsonKeys::CUSTOM,
                           stats.all_custom_latency_ns,
                           stats.all_custom_latency_samples);
  } else {
    // L1 cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::L1,
                             stats.all_l1_read_bw_gb_s,
                             stats.all_l1_write_bw_gb_s,
                             stats.all_l1_copy_bw_gb_s);
    
    // L1 cache latency
    add_cache_latency_json(cache,
                          JsonKeys::L1,
                          stats.all_l1_latency_ns,
                          stats.all_l1_latency_samples);
    
    // L2 cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::L2,
                             stats.all_l2_read_bw_gb_s,
                             stats.all_l2_write_bw_gb_s,
                             stats.all_l2_copy_bw_gb_s);
    
    // L2 cache latency
    add_cache_latency_json(cache,
                          JsonKeys::L2,
                          stats.all_l2_latency_ns,
                          stats.all_l2_latency_samples);
  }
  
  return cache;
}

// Write JSON to file with proper error handling and atomic writes
// Uses atomic file writing: writes to a temporary file, then renames it to the final destination
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
static int write_json_to_file(const std::filesystem::path& file_path, const nlohmann::json& json_output) {
  // Ensure parent directory exists
  std::filesystem::path parent_dir = file_path.parent_path();
  if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
    try {
      std::filesystem::create_directories(parent_dir);
    } catch (const std::filesystem::filesystem_error& e) {
      // Check for permission errors
      if (e.code() == std::errc::permission_denied) {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_directory_creation_failed(parent_dir.string(), "Permission denied") 
                  << std::endl;
      } else {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_directory_creation_failed(parent_dir.string(), e.what()) 
                  << std::endl;
      }
      return EXIT_FAILURE;
    }
  }
  
  // Create temporary file path in the same directory as the target file
  std::filesystem::path temp_file_path = file_path;
  temp_file_path += ".tmp";
  
  // Write JSON to temporary file (atomic write)
  try {
    std::ofstream out_file(temp_file_path, std::ios::out | std::ios::trunc);
    if (!out_file.is_open()) {
      // Check for permission errors
      if (errno == EACCES || errno == EPERM) {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_permission_denied(file_path.string()) 
                  << std::endl;
      } else {
        std::ostringstream oss;
        oss << "Failed to open temporary file: " << std::strerror(errno);
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_write_failed(temp_file_path.string(), oss.str()) 
                  << std::endl;
      }
      return EXIT_FAILURE;
    }
    
    // Write JSON content
    out_file << std::setw(2) << json_output << std::endl;
    
    // Check if write was successful
    if (out_file.fail() || out_file.bad()) {
      out_file.close();
      std::filesystem::remove(temp_file_path);  // Clean up temp file
      std::cerr << Messages::error_prefix() 
                << Messages::error_file_write_failed(temp_file_path.string(), "Write operation failed") 
                << std::endl;
      return EXIT_FAILURE;
    }
    
    // Ensure all data is written to disk
    out_file.flush();
    if (out_file.fail() || out_file.bad()) {
      out_file.close();
      std::filesystem::remove(temp_file_path);  // Clean up temp file
      std::cerr << Messages::error_prefix() 
                << Messages::error_file_write_failed(temp_file_path.string(), "Flush operation failed") 
                << std::endl;
      return EXIT_FAILURE;
    }
    
    out_file.close();
    
    // Atomically rename temporary file to final destination
    try {
      std::filesystem::rename(temp_file_path, file_path);
    } catch (const std::filesystem::filesystem_error& e) {
      // Clean up temp file on rename failure
      std::filesystem::remove(temp_file_path);
      
      // Check for permission errors
      if (e.code() == std::errc::permission_denied) {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_permission_denied(file_path.string()) 
                  << std::endl;
      } else {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_write_failed(file_path.string(), 
                      "Failed to rename temporary file: " + std::string(e.what())) 
                  << std::endl;
      }
      return EXIT_FAILURE;
    }
    
    std::cout << "Results saved to: " << file_path << std::endl;
  } catch (const std::filesystem::filesystem_error& e) {
    // Clean up temp file if it exists
    std::error_code ec;
    std::filesystem::remove(temp_file_path, ec);
    
    // Check for permission errors
    if (e.code() == std::errc::permission_denied) {
      std::cerr << Messages::error_prefix() 
                << Messages::error_file_permission_denied(file_path.string()) 
                << std::endl;
    } else {
      std::cerr << Messages::error_prefix() 
                << Messages::error_file_write_failed(file_path.string(), e.what()) 
                << std::endl;
    }
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    // Clean up temp file if it exists
    std::error_code ec;
    std::filesystem::remove(temp_file_path, ec);
    
    std::cerr << Messages::error_prefix() 
              << Messages::error_file_write_failed(file_path.string(), e.what()) 
              << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}

// Save benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_results_to_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats, double total_execution_time_sec) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;  // No output file specified, nothing to do
  }
  
  nlohmann::json json_output;
  
  // Add version
  std::ostringstream version_str;
  version_str << SOFTVERSION;
  json_output[JsonKeys::VERSION] = version_str.str();
  
  // Add timestamp (ISO 8601 UTC)
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  json_output[JsonKeys::TIMESTAMP] = timestamp_str.str();
  
  // Add configuration
  json_output[JsonKeys::CONFIGURATION] = build_config_json(config);
  
  // Add main memory results
  json_output[JsonKeys::MAIN_MEMORY] = build_main_memory_json(stats);
  
  // Add cache results
  json_output[JsonKeys::CACHE] = build_cache_json(config, stats);
  
  // Add execution time
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;
  
  // Resolve file path (handle relative paths)
  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    // Relative path - use current working directory
    file_path = std::filesystem::current_path() / file_path;
  }
  
  // Write JSON to file
  return write_json_to_file(file_path, json_output);
}

