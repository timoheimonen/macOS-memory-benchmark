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
#include "pattern_benchmark/pattern_benchmark.h"
#include "constants.h"
#include "messages.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// ============================================================================
// Output Formatting Functions
// ============================================================================

// Format percentage difference from baseline
static std::string format_percentage(double baseline, double value) {
  using namespace Constants;
  if (baseline == 0.0) return Messages::pattern_na();
  double pct = ((value - baseline) / baseline) * 100.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(PATTERN_PERCENTAGE_PRECISION);
  if (pct >= 0) {
    oss << " (+" << pct << "%)";
  } else {
    oss << " (" << pct << "%)";
  }
  return oss.str();
}

// Print sequential pattern results
static void print_sequential_results(const PatternResults& results) {
  using namespace Constants;
  
  // Sequential Forward (baseline)
  std::cout << Messages::pattern_sequential_forward() << "\n";
  std::cout << Messages::pattern_read_label() << std::fixed << std::setprecision(PATTERN_BANDWIDTH_PRECISION) 
            << results.forward_read_bw << Messages::pattern_bandwidth_unit_newline();
  std::cout << Messages::pattern_write_label() << results.forward_write_bw << Messages::pattern_bandwidth_unit_newline();
  std::cout << Messages::pattern_copy_label() << results.forward_copy_bw << Messages::pattern_bandwidth_unit_newline() << "\n";
  
  // Sequential Reverse
  std::cout << Messages::pattern_sequential_reverse() << "\n";
  std::cout << Messages::pattern_read_label() << results.reverse_read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, results.reverse_read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << results.reverse_write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, results.reverse_write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << results.reverse_copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, results.reverse_copy_bw) << "\n\n";
}

// Print strided pattern results
static void print_strided_results(const PatternResults& results, const std::string& stride_name, 
                                  double read_bw, double write_bw, double copy_bw) {
  std::cout << Messages::pattern_strided(stride_name) << "\n";
  std::cout << Messages::pattern_read_label() << read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, copy_bw) << "\n\n";
}

// Print random pattern results
static void print_random_results(const PatternResults& results) {
  std::cout << Messages::pattern_random_uniform() << "\n";
  std::cout << Messages::pattern_read_label() << results.random_read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, results.random_read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << results.random_write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, results.random_write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << results.random_copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, results.random_copy_bw) << "\n\n";
}

// Calculate pattern efficiency metrics
static void calculate_efficiency_metrics(const PatternResults& results,
                                         double& seq_coherence,
                                         double& prefetch_effectiveness,
                                         double& cache_thrashing,
                                         double& tlb_pressure) {
  using namespace Constants;
  
  double forward_total = results.forward_read_bw + results.forward_write_bw + results.forward_copy_bw;
  double reverse_total = results.reverse_read_bw + results.reverse_write_bw + results.reverse_copy_bw;
  double strided_64_total = results.strided_64_read_bw + results.strided_64_write_bw + results.strided_64_copy_bw;
  double strided_4096_total = results.strided_4096_read_bw + results.strided_4096_write_bw + results.strided_4096_copy_bw;
  double random_total = results.random_read_bw + results.random_write_bw + results.random_copy_bw;
  
  // Sequential coherence: ratio of reverse to forward
  seq_coherence = (reverse_total / forward_total) * 100.0;
  
  // Prefetcher effectiveness: ratio of strided 64B to forward (cache line stride should be well-prefetched)
  prefetch_effectiveness = (strided_64_total / forward_total) * 100.0;
  
  // Cache thrashing potential: based on strided 4096B performance (page stride causes more misses)
  cache_thrashing = (strided_4096_total / forward_total) * 100.0;
  
  // TLB pressure: based on random vs strided 4096B (random has more TLB misses)
  tlb_pressure = (random_total / strided_4096_total) * 100.0;
}

// Get cache thrashing level string
static const std::string& get_cache_thrashing_level(double cache_thrashing) {
  using namespace Constants;
  if (cache_thrashing > PATTERN_CACHE_THRASHING_HIGH_THRESHOLD) {
    return Messages::pattern_cache_thrashing_low();
  } else if (cache_thrashing > PATTERN_CACHE_THRASHING_MEDIUM_THRESHOLD) {
    return Messages::pattern_cache_thrashing_medium();
  } else {
    return Messages::pattern_cache_thrashing_high();
  }
}

// Get TLB pressure level string
static const std::string& get_tlb_pressure_level(double tlb_pressure) {
  using namespace Constants;
  if (tlb_pressure > PATTERN_TLB_PRESSURE_MINIMAL_THRESHOLD) {
    return Messages::pattern_tlb_pressure_minimal();
  } else if (tlb_pressure > PATTERN_TLB_PRESSURE_MODERATE_THRESHOLD) {
    return Messages::pattern_tlb_pressure_moderate();
  } else {
    return Messages::pattern_tlb_pressure_high();
  }
}

// Print pattern efficiency analysis
static void print_efficiency_analysis(const PatternResults& results) {
  using namespace Constants;
  
  double seq_coherence, prefetch_effectiveness, cache_thrashing, tlb_pressure;
  calculate_efficiency_metrics(results, seq_coherence, prefetch_effectiveness, 
                                cache_thrashing, tlb_pressure);
  
  std::cout << Messages::pattern_efficiency_analysis() << "\n";
  std::cout << "- " << Messages::pattern_sequential_coherence() << " " << std::fixed 
            << std::setprecision(PATTERN_PERCENTAGE_PRECISION) << seq_coherence << "%\n";
  std::cout << "- " << Messages::pattern_prefetcher_effectiveness() << " " << prefetch_effectiveness << "%\n";
  std::cout << "- " << Messages::pattern_cache_thrashing_potential() << " " 
            << get_cache_thrashing_level(cache_thrashing) << "\n";
  std::cout << "- " << Messages::pattern_tlb_pressure() << " " 
            << get_tlb_pressure_level(tlb_pressure) << "\n";
  std::cout << "\n";
}

void print_pattern_results(const PatternResults& results) {
  using namespace Constants;
  
  std::cout << Messages::pattern_separator();
  
  // Print all pattern results
  print_sequential_results(results);
  print_strided_results(results, Messages::pattern_cache_line_64b(), 
                        results.strided_64_read_bw, 
                        results.strided_64_write_bw, 
                        results.strided_64_copy_bw);
  print_strided_results(results, Messages::pattern_page_4096b(), 
                        results.strided_4096_read_bw, 
                        results.strided_4096_write_bw, 
                        results.strided_4096_copy_bw);
  print_random_results(results);
  
  // Print efficiency analysis
  print_efficiency_analysis(results);
}

