// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file test_system_info.cpp
 * @brief Deterministic system-query tests plus Apple Silicon integration contracts
 */

#include <gtest/gtest.h>

#include "core/config/constants.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"

#include <cerrno>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

class FakeSystemInfoProvider final : public SystemInfoProvider {
 public:
  template <typename T>
  void set_sysctl_value(const std::string& key, const T& value) {
    const auto* begin = reinterpret_cast<const unsigned char*>(&value);
    sysctl_values_[key] = std::vector<unsigned char>(begin, begin + sizeof(value));
  }

  void set_sysctl_string(const std::string& key, const std::string& value) {
    std::vector<unsigned char> bytes(value.begin(), value.end());
    bytes.push_back('\0');
    sysctl_values_[key] = std::move(bytes);
  }

  void fail_size_query(const std::string& key) {
    size_query_failures_.insert(key);
  }

  void fail_data_query(const std::string& key) {
    data_query_failures_.insert(key);
  }

  int query_sysctl(const char* name, void* old_value, size_t* old_length) const override {
    const std::string key = name;
    if (old_length == nullptr || size_query_failures_.count(key) != 0) {
      last_error_number_ = ENOENT;
      return -1;
    }

    const auto value = sysctl_values_.find(key);
    if (value == sysctl_values_.end()) {
      last_error_number_ = ENOENT;
      return -1;
    }

    if (old_value == nullptr) {
      *old_length = value->second.size();
      return 0;
    }
    if (data_query_failures_.count(key) != 0) {
      last_error_number_ = EIO;
      return -1;
    }
    if (*old_length < value->second.size()) {
      last_error_number_ = ENOMEM;
      return -1;
    }

    std::memcpy(old_value, value->second.data(), value->second.size());
    *old_length = value->second.size();
    return 0;
  }

  unsigned int hardware_concurrency() const override {
    return hardware_concurrency_value;
  }

  SystemMemoryQueryResult query_available_memory() const override {
    return memory_result;
  }

  int last_error_number() const override {
    return last_error_number_;
  }

  unsigned int hardware_concurrency_value = 0;
  SystemMemoryQueryResult memory_result;

 private:
  std::map<std::string, std::vector<unsigned char>> sysctl_values_;
  std::set<std::string> size_query_failures_;
  std::set<std::string> data_query_failures_;
  mutable int last_error_number_ = ENOENT;
};

std::string expected_warning(const std::string& message) {
  return Messages::warning_prefix() + message;
}

std::string expected_sysctl_error(const std::string& operation,
                                  const std::string& key,
                                  int error_number) {
  return Messages::error_prefix() +
         Messages::error_sysctlbyname_failed(operation, key) + ": " +
         std::strerror(error_number);
}

}  // namespace

TEST(SystemInfoTest, CoreQueriesUseValidTopologyValuesAndRejectInvalidOnes) {
  FakeSystemInfoProvider provider;
  provider.set_sysctl_value<int>("hw.perflevel0.logicalcpu_max", 6);
  provider.set_sysctl_value<int>("hw.perflevel1.logicalcpu_max", 4);

  EXPECT_EQ(get_performance_cores(provider), 6);
  EXPECT_EQ(get_efficiency_cores(provider), 4);
  EXPECT_EQ(get_total_logical_cores(provider), 10);

  FakeSystemInfoProvider invalid_provider;
  invalid_provider.set_sysctl_value<int>("hw.perflevel0.logicalcpu_max", 0);
  invalid_provider.set_sysctl_value<int>("hw.perflevel1.logicalcpu_max", -1);
  EXPECT_EQ(get_performance_cores(invalid_provider), 0);
  EXPECT_EQ(get_efficiency_cores(invalid_provider), 0);
}

TEST(SystemInfoTest, TotalCoreCountUsesOrderedFallbacks) {
  FakeSystemInfoProvider generic_provider;
  generic_provider.set_sysctl_value<int>("hw.logicalcpu_max", 12);
  generic_provider.hardware_concurrency_value = 99;
  EXPECT_EQ(get_total_logical_cores(generic_provider), 12);

  FakeSystemInfoProvider concurrency_provider;
  concurrency_provider.hardware_concurrency_value = 8;
  EXPECT_EQ(get_total_logical_cores(concurrency_provider), 8);

  FakeSystemInfoProvider exhausted_provider;
  testing::internal::CaptureStderr();
  EXPECT_EQ(get_total_logical_cores(exhausted_provider), 1);
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(stderr_output.find(expected_warning(Messages::warning_core_count_detection_failed())),
            std::string::npos);
}

TEST(SystemInfoTest, StringQueriesReturnCompleteValues) {
  FakeSystemInfoProvider provider;
  provider.set_sysctl_string("machdep.cpu.brand_string", "Apple M4 Pro");
  provider.set_sysctl_string("kern.osproductversion", "15.5.1");

  EXPECT_EQ(get_processor_name(provider), "Apple M4 Pro");
  EXPECT_EQ(get_macos_version(provider), "15.5.1");
}

TEST(SystemInfoTest, StringQueriesReportSizeAndDataFailures) {
  struct FailureCase {
    const char* key;
    bool processor_name;
    bool data_failure;
  };
  const FailureCase cases[] = {
      {"machdep.cpu.brand_string", true, false},
      {"machdep.cpu.brand_string", true, true},
      {"kern.osproductversion", false, false},
      {"kern.osproductversion", false, true},
  };

  for (const FailureCase& test_case : cases) {
    SCOPED_TRACE(test_case.key);
    SCOPED_TRACE(test_case.data_failure ? "data" : "size");
    FakeSystemInfoProvider provider;
    if (test_case.data_failure) {
      provider.set_sysctl_string(test_case.key, "unused");
      provider.fail_data_query(test_case.key);
    } else {
      provider.fail_size_query(test_case.key);
    }

    testing::internal::CaptureStderr();
    const std::string value = test_case.processor_name
                                  ? get_processor_name(provider)
                                  : get_macos_version(provider);
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(value.empty());
    const int error_number = test_case.data_failure ? EIO : ENOENT;
    EXPECT_NE(stderr_output.find(expected_sysctl_error(
                  test_case.data_failure ? "get data" : "get size",
                  test_case.key, error_number)),
              std::string::npos);
  }
}

TEST(SystemInfoTest, CacheQueriesUseDetectedValues) {
  FakeSystemInfoProvider provider;
  provider.set_sysctl_value<size_t>("hw.perflevel0.l1dcachesize", 192 * 1024);
  provider.set_sysctl_value<size_t>("hw.perflevel0.l2cachesize", 24 * 1024 * 1024);

  EXPECT_EQ(get_l1_cache_size(provider), static_cast<size_t>(192 * 1024));
  EXPECT_EQ(get_l2_cache_size(provider), static_cast<size_t>(24 * 1024 * 1024));
}

TEST(SystemInfoTest, L1CacheUsesCentralizedFallback) {
  FakeSystemInfoProvider provider;

  testing::internal::CaptureStderr();
  EXPECT_EQ(get_l1_cache_size(provider), Constants::L1_CACHE_FALLBACK_SIZE_BYTES);
  const std::string stderr_output = testing::internal::GetCapturedStderr();

  EXPECT_NE(stderr_output.find(expected_warning(Messages::warning_l1_cache_size_detection_failed())),
            std::string::npos);
}

TEST(SystemInfoTest, L2CacheUsesModelSpecificAndGenericFallbacks) {
  struct FallbackCase {
    const char* processor_name;
    size_t expected_size;
    std::string expected_message;
  };
  const FallbackCase cases[] = {
      {"Apple M1 Max", Constants::L2_CACHE_M1_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_m1()},
      {"Apple M2 Pro", Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_m2_m3_m4_m5()},
      {"Apple M3 Max", Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_m2_m3_m4_m5()},
      {"Apple M4", Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_m2_m3_m4_m5()},
      {"Apple M5", Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_m2_m3_m4_m5()},
      {"Apple Future", Constants::L2_CACHE_GENERIC_FALLBACK_SIZE_BYTES,
       Messages::warning_l2_cache_size_detection_failed_generic()},
  };

  for (const FallbackCase& test_case : cases) {
    SCOPED_TRACE(test_case.processor_name);
    FakeSystemInfoProvider provider;
    provider.set_sysctl_string("machdep.cpu.brand_string", test_case.processor_name);

    testing::internal::CaptureStderr();
    EXPECT_EQ(get_l2_cache_size(provider), test_case.expected_size);
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find(expected_warning(test_case.expected_message)),
              std::string::npos);
  }
}

TEST(SystemInfoTest, AvailableMemoryUsesProviderBytes) {
  FakeSystemInfoProvider provider;
  provider.memory_result = {
      SystemMemoryQueryStatus::Success,
      17 * static_cast<uint64_t>(Constants::BYTES_PER_MB) + 123,
      {},
  };

  EXPECT_EQ(get_available_memory_mb(provider), 17UL);
}

TEST(SystemInfoTest, AvailableMemoryReportsProviderFailures) {
  struct FailureCase {
    SystemMemoryQueryStatus status;
    const char* details;
    std::string expected_message;
  };
  const FailureCase cases[] = {
      {SystemMemoryQueryStatus::HostPortUnavailable, "",
       Messages::warning_mach_host_self_failed()},
      {SystemMemoryQueryStatus::PageSizeFailed, "page failure",
       Messages::warning_host_page_size_failed("page failure")},
      {SystemMemoryQueryStatus::StatisticsFailed, "statistics failure",
       Messages::warning_host_statistics64_failed("statistics failure")},
  };

  for (const FailureCase& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.status));
    FakeSystemInfoProvider provider;
    provider.memory_result = {test_case.status, 0, test_case.details};

    testing::internal::CaptureStderr();
    EXPECT_EQ(get_available_memory_mb(provider), 0UL);
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find(expected_warning(test_case.expected_message)),
              std::string::npos);
  }
}

TEST(SystemInfoIntegrationTest, CoreTopologyContract) {
  const int performance_cores = get_performance_cores();
  const int efficiency_cores = get_efficiency_cores();
  const int total_cores = get_total_logical_cores();

  EXPECT_GT(performance_cores, 0);
  EXPECT_GE(efficiency_cores, 0);
  EXPECT_GT(total_cores, 0);
  EXPECT_GE(total_cores, performance_cores);
  EXPECT_LE(performance_cores + efficiency_cores, total_cores * 2)
      << "performance=" << performance_cores
      << " efficiency=" << efficiency_cores
      << " total=" << total_cores;
}

TEST(SystemInfoIntegrationTest, ProcessorIdentityContract) {
  const std::string processor_name = get_processor_name();
  EXPECT_FALSE(processor_name.empty());
  EXPECT_NE(processor_name.find("Apple"), std::string::npos)
      << "Processor name was: " << processor_name;
}

TEST(SystemInfoIntegrationTest, CacheHierarchyContract) {
  const size_t l1_size = get_l1_cache_size();
  const size_t l2_size = get_l2_cache_size();

  EXPECT_GE(l1_size, static_cast<size_t>(64 * 1024));
  EXPECT_GE(l2_size, static_cast<size_t>(4 * 1024 * 1024));
  EXPECT_GT(l2_size, l1_size) << "L2=" << l2_size << " L1=" << l1_size;
}

TEST(SystemInfoIntegrationTest, MacOSVersionContract) {
  const std::string version = get_macos_version();
  EXPECT_FALSE(version.empty());
  EXPECT_NE(version.find('.'), std::string::npos)
      << "macOS version was: " << version;
}
