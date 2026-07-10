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
 * @file system_info.cpp
 * @brief System information detection implementation for macOS
 *
 * This file implements functions to detect and query various system characteristics
 * on macOS using sysctl and Mach APIs. Detection includes:
 * - CPU core topology (performance cores, efficiency cores, total cores)
 * - Processor model identification
 * - Available system memory
 * - Cache hierarchy (L1, L2 cache sizes)
 * - macOS version information
 *
 * The implementation is Apple Silicon-aware, distinguishing between performance
 * and efficiency cores on heterogeneous CPU architectures. It provides robust
 * fallback mechanisms when direct detection fails, using conservative estimates
 * or standard library alternatives.
 *
 * Detection strategy:
 * - Primary: macOS-specific sysctl keys (hw.perflevel0.*, hw.perflevel1.*)
 * - Fallback 1: Generic sysctl keys (hw.logicalcpu_max, hw.l1dcachesize)
 * - Fallback 2: Standard library (std::thread::hardware_concurrency)
 * - Fallback 3: Conservative hardcoded defaults based on CPU model
 *
 * @note All functions handle detection failures gracefully with warnings
 * @note Cache size detection includes Apple Silicon-specific optimizations
 * @note Memory calculations account for reclaimable inactive pages
 */

#include "core/system/system_info.h"

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

#include <mach/mach.h>
#include <mach/mach_error.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

/**
 * @brief Production provider backed by macOS sysctl, Mach, and the C++ runtime.
 */
class MacOsSystemInfoProvider final : public SystemInfoProvider {
 public:
  int query_sysctl(const char* name, void* old_value, size_t* old_length) const override {
    return sysctlbyname(name, old_value, old_length, nullptr, 0);
  }

  unsigned int hardware_concurrency() const override {
    return std::thread::hardware_concurrency();
  }

  SystemMemoryQueryResult query_available_memory() const override {
    const mach_port_t host_port = mach_host_self();
    if (host_port == MACH_PORT_NULL) {
      return {SystemMemoryQueryStatus::HostPortUnavailable, 0, {}};
    }

    vm_size_t page_size = 0;
    kern_return_t result = host_page_size(host_port, &page_size);
    if (result != KERN_SUCCESS || page_size == 0) {
      const char* details = mach_error_string(result);
      mach_port_deallocate(mach_task_self(), host_port);
      return {SystemMemoryQueryStatus::PageSizeFailed, 0, details != nullptr ? details : "unknown Mach error"};
    }

    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
    result = host_statistics64(host_port, HOST_VM_INFO64,
                               reinterpret_cast<host_info64_t>(&statistics), &info_count);
    if (result != KERN_SUCCESS) {
      const char* details = mach_error_string(result);
      mach_port_deallocate(mach_task_self(), host_port);
      return {SystemMemoryQueryStatus::StatisticsFailed, 0, details != nullptr ? details : "unknown Mach error"};
    }

    const uint64_t available_pages =
        static_cast<uint64_t>(statistics.free_count) + static_cast<uint64_t>(statistics.inactive_count);
    const uint64_t available_bytes = available_pages * static_cast<uint64_t>(page_size);
    mach_port_deallocate(mach_task_self(), host_port);
    return {SystemMemoryQueryStatus::Success, available_bytes, {}};
  }

  int last_error_number() const override {
    return errno;
  }
};

const SystemInfoProvider& default_system_info_provider() {
  static const MacOsSystemInfoProvider provider;
  return provider;
}

/**
 * @brief Read a string sysctl while preserving the two-call size/data contract.
 *
 * @param provider OS-query provider.
 * @param key Sysctl key.
 * @return Queried string without a trailing NUL, or an empty string on failure.
 */
std::string read_sysctl_string(const SystemInfoProvider& provider, const char* key) {
  size_t length = 0;
  if (provider.query_sysctl(key, nullptr, &length) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_sysctlbyname_failed("get size", key)
              << ": " << std::strerror(provider.last_error_number()) << std::endl;
    return {};
  }
  if (length == 0) {
    return {};
  }

  std::vector<char> buffer(length);
  if (provider.query_sysctl(key, buffer.data(), &length) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_sysctlbyname_failed("get data", key)
              << ": " << std::strerror(provider.last_error_number()) << std::endl;
    return {};
  }

  if (length > 0 && buffer[length - 1] == '\0') {
    --length;
  }
  return std::string(buffer.data(), length);
}

}  // namespace

/**
 * @brief Get the number of logical performance cores
 *
 * Queries the macOS-specific sysctl key for performance cores (perflevel0).
 * On Apple Silicon systems, performance cores are the high-power, high-performance
 * cores in the hybrid architecture.
 *
 * @return Number of logical performance cores, or 0 if detection fails
 *
 * @note Uses hw.perflevel0.logicalcpu_max sysctl key
 * @note Returns 0 (not an error code) on detection failure
 */
int get_performance_cores(const SystemInfoProvider& provider) {
  int p_cores = 0;
  size_t len = sizeof(p_cores);
  // Try reading the performance core count sysctl key.
  if (provider.query_sysctl("hw.perflevel0.logicalcpu_max", &p_cores, &len) == 0 && p_cores > 0) {
    return p_cores;  // Return count if successful and positive.
  }
  return 0;
}

int get_performance_cores() {
  return get_performance_cores(default_system_info_provider());
}

// Gets the number of logical Efficiency cores using sysctl.
int get_efficiency_cores(const SystemInfoProvider& provider) {
  int e_cores = 0;
  size_t len = sizeof(e_cores);
  // Try reading the efficiency core count sysctl key.
  if (provider.query_sysctl("hw.perflevel1.logicalcpu_max", &e_cores, &len) == 0 && e_cores >= 0) {
    return e_cores;  // Return count if successful (can be 0).
  }
  return 0;
}

int get_efficiency_cores() {
  return get_efficiency_cores(default_system_info_provider());
}

// Gets the total number of logical cores (P + E) using sysctl or fallbacks.
int get_total_logical_cores(const SystemInfoProvider& provider) {
  int p_cores = 0;
  int e_cores = 0;
  size_t len = sizeof(int);
  bool p_core_ok = false;
  bool e_core_ok = false;

  // First, try getting P and E core counts individually.
  if (provider.query_sysctl("hw.perflevel0.logicalcpu_max", &p_cores, &len) == 0 && p_cores > 0)
    p_core_ok = true;
  else
    p_cores = 0;
  len = sizeof(int);  // Reset len as sysctl might change it.
  if (provider.query_sysctl("hw.perflevel1.logicalcpu_max", &e_cores, &len) == 0 && e_cores >= 0)
    e_core_ok = true;
  else
    e_cores = 0;

  // If both P and E core counts were retrieved, return their sum.
  if (p_core_ok && e_core_ok) return p_cores + e_cores;

  // Fallback 1: Try the general logical CPU count key.
  int total_cores = 0;
  len = sizeof(total_cores);
  if (provider.query_sysctl("hw.logicalcpu_max", &total_cores, &len) == 0 && total_cores > 0) {
    return total_cores;
  }

  // Fallback 2: Use C++ standard library hardware_concurrency.
  const unsigned int hc = provider.hardware_concurrency();
  if (hc > 0) return hc;

  // If all methods fail, print a warning and return 1.
  std::cerr << Messages::warning_prefix() << Messages::warning_core_count_detection_failed() << std::endl;
  return 1;
}

int get_total_logical_cores() {
  return get_total_logical_cores(default_system_info_provider());
}

// Gets the CPU model name string using sysctl.
std::string get_processor_name(const SystemInfoProvider& provider) {
  return read_sysctl_string(provider, "machdep.cpu.brand_string");
}

std::string get_processor_name() {
  return get_processor_name(default_system_info_provider());
}

// Gets the estimated available system memory in Megabytes (MB) using Mach APIs.
unsigned long get_available_memory_mb(const SystemInfoProvider& provider) {
  const SystemMemoryQueryResult result = provider.query_available_memory();
  switch (result.status) {
    case SystemMemoryQueryStatus::Success:
      return static_cast<unsigned long>(result.available_bytes / Constants::BYTES_PER_MB);
    case SystemMemoryQueryStatus::HostPortUnavailable:
      std::cerr << Messages::warning_prefix() << Messages::warning_mach_host_self_failed() << std::endl;
      break;
    case SystemMemoryQueryStatus::PageSizeFailed:
      std::cerr << Messages::warning_prefix()
                << Messages::warning_host_page_size_failed(result.error_details) << std::endl;
      break;
    case SystemMemoryQueryStatus::StatisticsFailed:
      std::cerr << Messages::warning_prefix()
                << Messages::warning_host_statistics64_failed(result.error_details) << std::endl;
      break;
  }
  return 0;
}

unsigned long get_available_memory_mb() {
  return get_available_memory_mb(default_system_info_provider());
}

// Gets the L1 data cache size for performance cores using sysctl.
// Returns size in bytes. Uses fallback if detection fails.
size_t get_l1_cache_size(const SystemInfoProvider& provider) {
  size_t l1_size = 0;
  size_t len = sizeof(l1_size);
  // Try reading L1 data cache size for performance cores (perflevel0).
  if (provider.query_sysctl("hw.perflevel0.l1dcachesize", &l1_size, &len) == 0 && l1_size > 0) {
    return l1_size;  // Return detected size.
  }
  // Fallback: Use typical Apple Silicon P-core L1 size (128 KB).
  std::cerr << Messages::warning_prefix() << Messages::warning_l1_cache_size_detection_failed() << std::endl;
  return Constants::L1_CACHE_FALLBACK_SIZE_BYTES;
}

size_t get_l1_cache_size() {
  return get_l1_cache_size(default_system_info_provider());
}

// Gets the L2 cache size for performance cores using sysctl.
// Returns size in bytes. Uses fallback if detection fails.
size_t get_l2_cache_size(const SystemInfoProvider& provider) {
  size_t l2_size = 0;
  size_t len = sizeof(l2_size);
  // Try reading L2 cache size for performance cores (perflevel0).
  if (provider.query_sysctl("hw.perflevel0.l2cachesize", &l2_size, &len) == 0 && l2_size > 0) {
    return l2_size;  // Return detected size.
  }
  // Fallback: Try to infer from processor name, otherwise use conservative estimate.
  const std::string cpu_name = get_processor_name(provider);
  if (cpu_name.find("M1") != std::string::npos) {
    std::cerr << Messages::warning_prefix() << Messages::warning_l2_cache_size_detection_failed_m1() << std::endl;
    return Constants::L2_CACHE_M1_FALLBACK_SIZE_BYTES;
  } else if (cpu_name.find("M2") != std::string::npos || cpu_name.find("M3") != std::string::npos ||
             cpu_name.find("M4") != std::string::npos || cpu_name.find("M5") != std::string::npos) {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_l2_cache_size_detection_failed_m2_m3_m4_m5()
              << std::endl;
    return Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES;
  }
  // Generic fallback.
  std::cerr << Messages::warning_prefix() << Messages::warning_l2_cache_size_detection_failed_generic() << std::endl;
  return Constants::L2_CACHE_GENERIC_FALLBACK_SIZE_BYTES;
}

size_t get_l2_cache_size() {
  return get_l2_cache_size(default_system_info_provider());
}

// Gets the macOS version string using sysctl.
std::string get_macos_version(const SystemInfoProvider& provider) {
  return read_sysctl_string(provider, "kern.osproductversion");
}

std::string get_macos_version() {
  return get_macos_version(default_system_info_provider());
}
