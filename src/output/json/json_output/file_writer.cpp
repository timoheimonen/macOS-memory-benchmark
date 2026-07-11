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
 * @file file_writer.cpp
 * @brief JSON file writing with error handling
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file provides the file writing functionality for JSON output with
 * comprehensive error handling for file system operations. Includes directory
 * creation, permission checking, and detailed error reporting.
 *
 * Error handling:
 * - Automatic parent directory creation
 * - Permission validation
 * - Detailed error messages with errno information
 * - Safe file overwriting
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include <iomanip>    // Required for std::setprecision, std::setw
#include <iostream>   // Required for std::cout, std::cerr
#include <sstream>    // Required for std::ostringstream
#include <fstream>    // Required for std::ofstream
#include <filesystem> // Required for std::filesystem::path
#include <cerrno>     // Required for errno
#include <cstring>    // Required for std::strerror
#include <system_error> // Required for std::errc

#include "output/json/json_output/json_output_api.h"
#include "output/console/messages/messages_api.h"   // Include centralized messages
#include "third_party/nlohmann/json.hpp"   // JSON library

namespace {

int write_json_to_file_impl(const std::filesystem::path& file_path,
                            const nlohmann::ordered_json& json_output,
                            bool announce_success) {
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
      const int open_errno = errno;
      // A stale path (including a directory at the temporary name) must not
      // survive a failed checkpoint attempt.
      std::error_code cleanup_error;
      std::filesystem::remove(temp_file_path, cleanup_error);
      // Check for permission errors
      if (open_errno == EACCES || open_errno == EPERM) {
        std::cerr << Messages::error_prefix() 
                  << Messages::error_file_permission_denied(file_path.string()) 
                  << std::endl;
      } else {
        std::ostringstream oss;
        oss << "Failed to open temporary file: " << std::strerror(open_errno);
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
    if (out_file.fail() || out_file.bad()) {
      std::error_code cleanup_error;
      std::filesystem::remove(temp_file_path, cleanup_error);
      std::cerr << Messages::error_prefix()
                << Messages::error_file_write_failed(
                       temp_file_path.string(), "Close operation failed")
                << std::endl;
      return EXIT_FAILURE;
    }
    
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
    
    if (announce_success) {
      std::cout << Messages::msg_results_saved_to(file_path.string()) << std::endl;
    }
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

}  // namespace

// Write JSON atomically and contain every filesystem/library exception at the
// output boundary so callers can reliably convert failure to a return code.
int write_json_to_file(const std::filesystem::path& file_path,
                       const nlohmann::ordered_json& json_output,
                       bool announce_success) {
  try {
    return write_json_to_file_impl(file_path, json_output, announce_success);
  } catch (const std::exception& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_file_write_failed(file_path.string(),
                                                    error.what())
              << std::endl;
  } catch (...) {
    std::cerr << Messages::error_prefix()
              << Messages::error_file_write_failed(
                     file_path.string(), "Unknown file writer exception")
              << std::endl;
  }
  return EXIT_FAILURE;
}
