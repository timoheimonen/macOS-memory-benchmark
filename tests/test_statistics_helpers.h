#ifndef TEST_STATISTICS_HELPERS_H
#define TEST_STATISTICS_HELPERS_H

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "output/console/statistics.h"

namespace test_statistics_helpers {

inline const std::vector<double>& empty_values() {
  static const std::vector<double> k_empty_values;
  return k_empty_values;
}

inline std::string capture_bw(const std::vector<double>& values) {
  const std::vector<double>& empty = empty_values();
  testing::internal::CaptureStdout();
  print_statistics(2,
                   values,
                   values,
                   values,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   false,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   false,
                   false);
  return testing::internal::GetCapturedStdout();
}

inline std::string capture_lat(const std::vector<double>& values) {
  const std::vector<double>& empty = empty_values();
  testing::internal::CaptureStdout();
  print_statistics(2,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   values,
                   empty,
                   empty,
                   empty,
                   false,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   false,
                   true);
  return testing::internal::GetCapturedStdout();
}

inline std::string capture_auto_tlb_breakdown(const std::vector<double>& all_main_mem_latency,
                                              const std::vector<double>& all_tlb_hit_latency,
                                              const std::vector<double>& all_tlb_miss_latency,
                                              const std::vector<double>& all_page_walk_penalty,
                                              int loop_count = 2) {
  const std::vector<double>& empty = empty_values();
  testing::internal::CaptureStdout();
  print_statistics(loop_count,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   all_main_mem_latency,
                   all_tlb_hit_latency,
                   all_tlb_miss_latency,
                   all_page_walk_penalty,
                   false,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   empty,
                   false,
                   true);
  return testing::internal::GetCapturedStdout();
}

}  // namespace test_statistics_helpers

#endif  // TEST_STATISTICS_HELPERS_H
