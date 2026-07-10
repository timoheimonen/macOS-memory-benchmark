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
 * @file system_info.h
 * @brief System information query functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions to query system information such as CPU cores,
 * processor name, memory, and cache sizes.
 */
#ifndef SYSTEM_INFO_H
#define SYSTEM_INFO_H

#include <cstddef>  // size_t
#include <cstdint>
#include <string>

/**
 * @brief Result category for the Mach-backed available-memory query.
 */
enum class SystemMemoryQueryStatus {
  Success,
  HostPortUnavailable,
  PageSizeFailed,
  StatisticsFailed,
};

/**
 * @brief Provider result for reclaimable system memory.
 *
 * `available_bytes` is meaningful only for `Success`. `error_details` carries
 * the Mach error description for page-size and statistics failures.
 */
struct SystemMemoryQueryResult {
  SystemMemoryQueryStatus status = SystemMemoryQueryStatus::HostPortUnavailable;
  uint64_t available_bytes = 0;
  std::string error_details;
};

/**
 * @brief Injectable boundary for sysctl, Mach memory, and concurrency queries.
 *
 * Production getters use a macOS-backed implementation. Tests can provide a
 * deterministic implementation through the provider-taking overloads below.
 * Implementations must keep returned error state valid until
 * `last_error_number()` is called and must be safe for the caller's threading
 * model; the production provider is thread-safe.
 */
class SystemInfoProvider {
 public:
  virtual ~SystemInfoProvider() = default;

  /**
   * @brief Query a sysctl value using the read-only `sysctlbyname` contract.
   * @param name Sysctl key.
   * @param old_value Destination buffer, or `nullptr` for a size query.
   * @param old_length In/out buffer length.
   * @return Zero on success, otherwise the platform error return value.
   */
  virtual int query_sysctl(const char* name, void* old_value, size_t* old_length) const = 0;

  /** @return Standard-library hardware concurrency fallback, or zero if unknown. */
  virtual unsigned int hardware_concurrency() const = 0;

  /** @return Mach-backed available-memory result. */
  virtual SystemMemoryQueryResult query_available_memory() const = 0;

  /** @return Error number produced by the most recent failed sysctl query. */
  virtual int last_error_number() const = 0;
};

// --- System Info Functions ---
/**
 * @brief Get number of performance cores
 * @return Number of performance cores
 */
int get_performance_cores();

/** @brief Provider-injected overload of `get_performance_cores()`. */
int get_performance_cores(const SystemInfoProvider& provider);

/**
 * @brief Get number of efficiency cores
 * @return Number of efficiency cores
 */
int get_efficiency_cores();

/** @brief Provider-injected overload of `get_efficiency_cores()`. */
int get_efficiency_cores(const SystemInfoProvider& provider);

/**
 * @brief Get total logical core count
 * @return Total number of logical cores
 */
int get_total_logical_cores();

/** @brief Provider-injected overload of `get_total_logical_cores()`. */
int get_total_logical_cores(const SystemInfoProvider& provider);

/**
 * @brief Get processor model name
 * @return Processor model name as string
 */
std::string get_processor_name();

/** @brief Provider-injected overload of `get_processor_name()`. */
std::string get_processor_name(const SystemInfoProvider& provider);

/**
 * @brief Get available system memory in MB
 * @return Available system memory in megabytes
 */
unsigned long get_available_memory_mb();

/** @brief Provider-injected overload of `get_available_memory_mb()`. */
unsigned long get_available_memory_mb(const SystemInfoProvider& provider);

/**
 * @brief Get L1 data cache size for performance cores
 * @return L1 data cache size in bytes
 */
size_t get_l1_cache_size();

/** @brief Provider-injected overload of `get_l1_cache_size()`. */
size_t get_l1_cache_size(const SystemInfoProvider& provider);

/**
 * @brief Get L2 cache size for performance cores
 * @return L2 cache size in bytes
 */
size_t get_l2_cache_size();

/** @brief Provider-injected overload of `get_l2_cache_size()`. */
size_t get_l2_cache_size(const SystemInfoProvider& provider);

/**
 * @brief Get macOS version string
 * @return macOS version as string (e.g., "14.2.1")
 */
std::string get_macos_version();

/** @brief Provider-injected overload of `get_macos_version()`. */
std::string get_macos_version(const SystemInfoProvider& provider);

#endif  // SYSTEM_INFO_H
