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
#include <mach/mach_error.h>  // For mach_error_string
#include <mach/mach_host.h>   // For host_statistics64, mach_host_self, host_page_size
#include <sys/sysctl.h>       // For sysctlbyname

#include <cstdio>  // For perror
#include <thread>  // For std::thread::hardware_concurrency
#include <vector>  // For std::vector

#include "benchmark.h"

// Gets the number of logical Performance cores using sysctl.
int get_performance_cores() {
  int p_cores = 0;
  size_t len = sizeof(p_cores);
  // Try reading the performance core count sysctl key.
  if (sysctlbyname("hw.perflevel0.logicalcpu_max", &p_cores, &len, NULL, 0) == 0 && p_cores > 0) {
    return p_cores;  // Return count if successful and positive.
  } else {
    // Return 0 if key doesn't exist or read fails.
    return 0;
  }
}

// Gets the number of logical Efficiency cores using sysctl.
int get_efficiency_cores() {
  int e_cores = 0;
  size_t len = sizeof(e_cores);
  // Try reading the efficiency core count sysctl key.
  if (sysctlbyname("hw.perflevel1.logicalcpu_max", &e_cores, &len, NULL, 0) == 0 && e_cores >= 0) {
    return e_cores;  // Return count if successful (can be 0).
  } else {
    // Return 0 if key doesn't exist or read fails.
    return 0;
  }
}

// Gets the total number of logical cores (P + E) using sysctl or fallbacks.
int get_total_logical_cores() {
  int p_cores = 0;
  int e_cores = 0;
  size_t len = sizeof(int);
  bool p_core_ok = false;
  bool e_core_ok = false;

  // First, try getting P and E core counts individually.
  if (sysctlbyname("hw.perflevel0.logicalcpu_max", &p_cores, &len, NULL, 0) == 0 && p_cores > 0)
    p_core_ok = true;
  else
    p_cores = 0;
  len = sizeof(int);  // Reset len as sysctl might change it.
  if (sysctlbyname("hw.perflevel1.logicalcpu_max", &e_cores, &len, NULL, 0) == 0 && e_cores >= 0)
    e_core_ok = true;
  else
    e_cores = 0;

  // If both P and E core counts were retrieved, return their sum.
  if (p_core_ok && e_core_ok) return p_cores + e_cores;

  // Fallback 1: Try the general logical CPU count key.
  int total_cores = 0;
  len = sizeof(total_cores);
  if (sysctlbyname("hw.logicalcpu_max", &total_cores, &len, NULL, 0) == 0 && total_cores > 0) return total_cores;

  // Fallback 2: Use C++ standard library hardware_concurrency.
  unsigned int hc = std::thread::hardware_concurrency();
  if (hc > 0) return hc;

  // If all methods fail, print a warning and return 1.
  std::cerr << "Warning: Failed to detect core count, defaulting to 1." << std::endl;
  return 1;
}

// Gets the CPU model name string using sysctl.
std::string get_processor_name() {
  size_t len = 0;
  // First call to get the size of the string.
  if (sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0) == -1) {
    perror("sysctlbyname (get size) failed for machdep.cpu.brand_string");
    return "";  // Return empty string on error.
  }

  if (len > 0) {
    std::vector<char> buffer(len);
    // Second call to get the actual string data.
    if (sysctlbyname("machdep.cpu.brand_string", buffer.data(), &len, NULL, 0) == -1) {
      perror("sysctlbyname (get data) failed for machdep.cpu.brand_string");
      return "";  // Return empty string on error.
    }
    // Create string, excluding potential null terminator if included in len.
    return std::string(buffer.data(), len > 0 ? len - 1 : 0);
  }
  return "";  // Return empty string if size is 0.
}

// Gets the estimated available system memory in Megabytes (MB) using Mach APIs.
unsigned long get_available_memory_mb() {
  mach_port_t host_port = mach_host_self();  // Get the host port.
  if (host_port == MACH_PORT_NULL) {
    std::cerr << "Warning: Failed to get mach_host_self(). Cannot determine available memory." << std::endl;
    return 0;
  }

  vm_size_t page_size_local = 0;
  // Get the system page size.
  kern_return_t kern_ret = host_page_size(host_port, &page_size_local);
  if (kern_ret != KERN_SUCCESS || page_size_local == 0) {
    std::cerr << "Warning: Failed to get host_page_size(): " << mach_error_string(kern_ret)
              << ". Cannot determine available memory." << std::endl;
    return 0;
  }

  vm_statistics64_data_t vm_stats;
  mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
  // Get virtual memory statistics.
  kern_ret = host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &info_count);
  if (kern_ret != KERN_SUCCESS) {
    std::cerr << "Warning: Failed to get host_statistics64(): " << mach_error_string(kern_ret)
              << ". Cannot determine available memory." << std::endl;
    return 0;
  }

  // Calculate available memory (free + inactive pages) in bytes.
  // Inactive pages can be reclaimed by the OS when needed.
  uint64_t available_bytes = static_cast<uint64_t>(vm_stats.free_count + vm_stats.inactive_count) * page_size_local;
  const uint64_t bytes_per_mb = 1024 * 1024;
  // Convert bytes to MB.
  unsigned long available_mb = static_cast<unsigned long>(available_bytes / bytes_per_mb);
  return available_mb;
}