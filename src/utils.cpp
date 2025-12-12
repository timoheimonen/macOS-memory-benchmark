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
#include <algorithm>  // Required for std::min_element, std::max_element (finding min/max)
#include <iomanip>    // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>   // Required for std::cout, std::cerr
#include <numeric>    // Required for std::accumulate (calculating sums)
#include <vector>     // Required for std::vector
#include <atomic>     // Required for std::atomic (progress indicator)

#include "benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)

// --- Progress Indicator ---
// Simple spinner for showing progress without affecting performance
// Uses a static counter to cycle through spinner characters
static std::atomic<int> spinner_counter{0};
static const char spinner_chars[] = {'|', '/', '-', '\\'};

void show_progress() {
  int idx = spinner_counter.fetch_add(1, std::memory_order_relaxed) % 4;
  std::cout << '\r' << spinner_chars[idx] << " Running tests... " << std::flush;
}

// --- Helper function to print usage instructions ---
// Displays how to use the program via command-line arguments.
// 'prog_name': The name of the executable (typically argv[0]).
void print_usage(const char *prog_name) {
  // Output usage syntax, version, options, and an example to standard error.
  std::cerr << "Version: " << SOFTVERSION << " by Timo Heimonen <timo.heimonen@proton.me>\n"  // SOFTVERSION defined in benchmark.h
            << "License: GNU GPL v3. See <https://www.gnu.org/licenses/>\n"
            << "Link: https://github.com/timoheimonen/macOS-memory-benchmark\n\n"
            << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: 1000)\n"
            << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: 512).\n"
            << "                        The maximum allowed <size_mb> is automatically determined such that\n"
            << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
            << "  -count <count>        Number of full loops (read/write/copy/latency) (default: 1)\n"
            << "  -cache-size <size_kb> Custom cache size in Kilobytes (KB) as integer (16 KB to 524288 KB).\n"
            << "                        Minimum is 16 KB (system page size). When set, skips automatic\n"
            << "                        L1/L2 cache size detection and only performs bandwidth and latency\n"
            << "                        tests for the custom cache size.\n"
            << "  -h, --help            Show this help message and exit\n\n"
            << "Example: " << prog_name << " -iterations 500 -buffersize 1024\n"
            << "Example: " << prog_name << " -cache-size 256\n";
}

// --- Print Configuration ---
// Outputs the configuration parameters the benchmark will run with.
// 'buffer_size': The final size (in bytes) of each test buffer after memory checks.
// 'buffer_size_mb': The requested/capped buffer size in Megabytes.
// 'iterations': Number of iterations per bandwidth test per loop.
// 'loop_count': How many times the entire set of tests will be repeated.
// 'cpu_name': Detected processor name string.
// 'perf_cores': Number of detected performance cores.
// 'eff_cores': Number of detected efficiency cores.
// 'num_threads': Total number of threads used for bandwidth tests (usually equals total cores).
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count,
                         const std::string &cpu_name, int perf_cores, int eff_cores, int num_threads) {
  // Print benchmark header and copyright/license info.
  std::cout << "----- macOS-memory-benchmark v" << SOFTVERSION << " -----" << std::endl;
  std::cout << "Program is licensed under GNU GPL v3. See <https://www.gnu.org/licenses/>" << std::endl;
  std::cout << "Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>\n" << std::endl;
  // Display buffer sizes (actual MiB and requested/capped MB).
  std::cout << "Buffer Size (per buffer): " << std::fixed << std::setprecision(2) << buffer_size / (1024.0 * 1024.0)
            << " MiB (" << buffer_size_mb << " MB requested/capped)" << std::endl;
  // Display total approximate memory allocated for the three buffers.
  std::cout << "Total Allocation Size: ~" << 3.0 * buffer_size / (1024.0 * 1024.0) << " MiB (for 3 buffers)"
            << std::endl;
  // Display test repetition counts.
  std::cout << "Iterations (per R/W/Copy test per loop): " << iterations << std::endl;
  std::cout << "Loop Count (total benchmark repetitions): " << loop_count << std::endl;
  // Display CPU information if successfully retrieved.
  if (!cpu_name.empty()) {
    std::cout << "\nProcessor Name: " << cpu_name << std::endl;
  } else {
    std::cout << "Could not retrieve processor name." << std::endl;
  }
  // Display P-core and E-core counts if available.
  if (perf_cores > 0 || eff_cores > 0) {
    std::cout << "  Performance Cores: " << perf_cores << std::endl;
    std::cout << "  Efficiency Cores: " << eff_cores << std::endl;
  }
  // Display total cores detected and threads used for bandwidth tests.
  std::cout << "  Total CPU Cores Detected: " << num_threads << std::endl;
}

// --- Print Results for a single loop ---
// Outputs the performance results measured during one complete benchmark loop.
// 'loop': Index of the loop being reported (0-based).
// 'buffer_size', 'buffer_size_mb', 'iterations', 'num_threads': Config parameters for context.
// 'read_bw_gb_s', 'total_read_time': Results from the read test.
// 'write_bw_gb_s', 'total_write_time': Results from the write test.
// 'copy_bw_gb_s', 'total_copy_time': Results from the copy test.
// 'l1_latency_ns', 'l2_latency_ns': Results from cache latency tests.
// 'l1_buffer_size', 'l2_buffer_size': Buffer sizes used for cache tests.
// 'l1_read_bw_gb_s', 'l1_write_bw_gb_s', 'l1_copy_bw_gb_s': L1 cache bandwidth results.
// 'l2_read_bw_gb_s', 'l2_write_bw_gb_s', 'l2_copy_bw_gb_s': L2 cache bandwidth results.
// 'average_latency_ns', 'total_lat_time_ns': Results from the main memory latency test.
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'custom_latency_ns', 'custom_buffer_size': Custom cache latency results and buffer size.
// 'custom_read_bw_gb_s', 'custom_write_bw_gb_s', 'custom_copy_bw_gb_s': Custom cache bandwidth results.
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
                   double read_bw_gb_s, double total_read_time, double write_bw_gb_s, double total_write_time,
                   double copy_bw_gb_s, double total_copy_time, 
                   double l1_latency_ns, double l2_latency_ns,
                   size_t l1_buffer_size, size_t l2_buffer_size,
                   double l1_read_bw_gb_s, double l1_write_bw_gb_s, double l1_copy_bw_gb_s,
                   double l2_read_bw_gb_s, double l2_write_bw_gb_s, double l2_copy_bw_gb_s,
                   double average_latency_ns, double total_lat_time_ns,
                   bool use_custom_cache_size, double custom_latency_ns, size_t custom_buffer_size,
                   double custom_read_bw_gb_s, double custom_write_bw_gb_s, double custom_copy_bw_gb_s) {
  // Print a header indicating the current loop number.
  std::cout << "\n--- Results (Loop " << loop + 1 << ") ---" << std::endl;
  // Set output to fixed-point notation for consistent formatting.
  std::cout << std::fixed;

  // Display Main Memory Bandwidth test results.
  std::cout << "Main Memory Bandwidth Tests (multi-threaded, " << num_threads << " threads):" << std::endl;
  std::cout << "  Read : " << std::setprecision(3) << read_bw_gb_s << " GB/s (Total time: " << total_read_time << " s)"
            << std::endl;
  std::cout << "  Write: " << std::setprecision(3) << write_bw_gb_s << " GB/s (Total time: " << total_write_time
            << " s)" << std::endl;
  std::cout << "  Copy : " << std::setprecision(3) << copy_bw_gb_s << " GB/s (Total time: " << total_copy_time << " s)"
            << std::endl;
  
  // Display main memory latency test results.
  std::cout << "\nMain Memory Latency Test (single-threaded, pointer chase):" << std::endl;
  std::cout << "  Total time: " << std::setprecision(3) << total_lat_time_ns / 1e9 << " s"
            << std::endl;  // Show latency test time in seconds
  std::cout << "  Average latency: " << std::setprecision(2) << average_latency_ns << " ns" << std::endl;

  // Display cache bandwidth test results.
  std::cout << "\nCache Bandwidth Tests (single-threaded):" << std::endl;
  if (use_custom_cache_size) {
    // Display custom cache bandwidth results
    if (custom_buffer_size > 0) {
      std::cout << "  Custom Cache:" << std::endl;
      std::cout << "    Read : " << std::setprecision(3) << custom_read_bw_gb_s << " GB/s";
      if (custom_buffer_size < 1024) {
        std::cout << " (Buffer size: " << custom_buffer_size << " B)" << std::endl;
      } else if (custom_buffer_size < 1024 * 1024) {
        std::cout << " (Buffer size: " << std::setprecision(2) << custom_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << " (Buffer size: " << std::setprecision(2) << custom_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
      std::cout << "    Write: " << std::setprecision(3) << custom_write_bw_gb_s << " GB/s" << std::endl;
      std::cout << "    Copy : " << std::setprecision(3) << custom_copy_bw_gb_s << " GB/s" << std::endl;
    }
  } else {
    // Display L1/L2 cache bandwidth results
    if (l1_buffer_size > 0) {
      std::cout << "  L1 Cache:" << std::endl;
      std::cout << "    Read : " << std::setprecision(3) << l1_read_bw_gb_s << " GB/s";
      if (l1_buffer_size < 1024) {
        std::cout << " (Buffer size: " << l1_buffer_size << " B)" << std::endl;
      } else if (l1_buffer_size < 1024 * 1024) {
        std::cout << " (Buffer size: " << std::setprecision(2) << l1_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << " (Buffer size: " << std::setprecision(2) << l1_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
      std::cout << "    Write: " << std::setprecision(3) << l1_write_bw_gb_s << " GB/s" << std::endl;
      std::cout << "    Copy : " << std::setprecision(3) << l1_copy_bw_gb_s << " GB/s" << std::endl;
    }
    if (l2_buffer_size > 0) {
      std::cout << "  L2 Cache:" << std::endl;
      std::cout << "    Read : " << std::setprecision(3) << l2_read_bw_gb_s << " GB/s";
      if (l2_buffer_size < 1024) {
        std::cout << " (Buffer size: " << l2_buffer_size << " B)" << std::endl;
      } else if (l2_buffer_size < 1024 * 1024) {
        std::cout << " (Buffer size: " << std::setprecision(2) << l2_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << " (Buffer size: " << std::setprecision(2) << l2_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
      std::cout << "    Write: " << std::setprecision(3) << l2_write_bw_gb_s << " GB/s" << std::endl;
      std::cout << "    Copy : " << std::setprecision(3) << l2_copy_bw_gb_s << " GB/s" << std::endl;
    }
  }

  // Display cache latency test results.
  std::cout << "\nCache Latency Tests (single-threaded, pointer chase):" << std::endl;
  if (use_custom_cache_size) {
    // Display custom cache latency results
    if (custom_buffer_size > 0) {
      if (custom_buffer_size < 1024) {
        std::cout << "  Custom Cache: " << std::setprecision(2) << custom_latency_ns << " ns (Buffer size: " << custom_buffer_size << " B)" << std::endl;
      } else if (custom_buffer_size < 1024 * 1024) {
        std::cout << "  Custom Cache: " << std::setprecision(2) << custom_latency_ns << " ns (Buffer size: " << std::setprecision(2) << custom_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << "  Custom Cache: " << std::setprecision(2) << custom_latency_ns << " ns (Buffer size: " << std::setprecision(2) << custom_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
    }
  } else {
    // Display L1/L2 cache latency results
    if (l1_buffer_size > 0) {
      if (l1_buffer_size < 1024) {
        std::cout << "  L1 Cache: " << std::setprecision(2) << l1_latency_ns << " ns (Buffer size: " << l1_buffer_size << " B)" << std::endl;
      } else if (l1_buffer_size < 1024 * 1024) {
        std::cout << "  L1 Cache: " << std::setprecision(2) << l1_latency_ns << " ns (Buffer size: " << std::setprecision(2) << l1_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << "  L1 Cache: " << std::setprecision(2) << l1_latency_ns << " ns (Buffer size: " << std::setprecision(2) << l1_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
    }
    if (l2_buffer_size > 0) {
      if (l2_buffer_size < 1024) {
        std::cout << "  L2 Cache: " << std::setprecision(2) << l2_latency_ns << " ns (Buffer size: " << l2_buffer_size << " B)" << std::endl;
      } else if (l2_buffer_size < 1024 * 1024) {
        std::cout << "  L2 Cache: " << std::setprecision(2) << l2_latency_ns << " ns (Buffer size: " << std::setprecision(2) << l2_buffer_size / 1024.0 << " KB)" << std::endl;
      } else {
        std::cout << "  L2 Cache: " << std::setprecision(2) << l2_latency_ns << " ns (Buffer size: " << std::setprecision(2) << l2_buffer_size / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
    }
  }
  // Print a separator after the loop results.
  std::cout << "--------------" << std::endl;
}

// --- Print Statistics across all loops ---
// Calculates and displays summary statistics (Avg/Min/Max) if more than one loop was run.
// 'loop_count': The total number of loops that were executed.
// 'all_read_bw', 'all_write_bw', 'all_copy_bw': Vectors holding bandwidth results from each individual loop.
// 'all_l1_latency', 'all_l2_latency': Vectors holding cache latency results from each loop.
// 'all_l1_read_bw', 'all_l1_write_bw', 'all_l1_copy_bw': Vectors holding L1 cache bandwidth results from each loop.
// 'all_l2_read_bw', 'all_l2_write_bw', 'all_l2_copy_bw': Vectors holding L2 cache bandwidth results from each loop.
// 'all_main_mem_latency': Vector holding main memory latency results from each loop.
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'all_custom_latency', 'all_custom_read_bw', 'all_custom_write_bw', 'all_custom_copy_bw': Custom cache results vectors.
void print_statistics(int loop_count, const std::vector<double> &all_read_bw, const std::vector<double> &all_write_bw,
                      const std::vector<double> &all_copy_bw, 
                      const std::vector<double> &all_l1_latency, const std::vector<double> &all_l2_latency,
                      const std::vector<double> &all_l1_read_bw, const std::vector<double> &all_l1_write_bw,
                      const std::vector<double> &all_l1_copy_bw,
                      const std::vector<double> &all_l2_read_bw, const std::vector<double> &all_l2_write_bw,
                      const std::vector<double> &all_l2_copy_bw,
                      const std::vector<double> &all_main_mem_latency,
                      bool use_custom_cache_size,
                      const std::vector<double> &all_custom_latency,
                      const std::vector<double> &all_custom_read_bw,
                      const std::vector<double> &all_custom_write_bw,
                      const std::vector<double> &all_custom_copy_bw) {
  // Don't print statistics if only one loop ran or if there's no data.
  if (loop_count <= 1 || all_read_bw.empty()) return;

  // Print statistics header.
  std::cout << "\n--- Statistics Across " << loop_count << " Loops ---" << std::endl;

  // --- Calculate Averages ---
  // Sum up all results for each test type and divide by the number of loops.
  double avg_read_bw = std::accumulate(all_read_bw.begin(), all_read_bw.end(), 0.0) / loop_count;
  double avg_write_bw = std::accumulate(all_write_bw.begin(), all_write_bw.end(), 0.0) / loop_count;
  double avg_copy_bw = std::accumulate(all_copy_bw.begin(), all_copy_bw.end(), 0.0) / loop_count;
  
  double avg_l1_latency = 0.0, avg_l2_latency = 0.0;
  if (!all_l1_latency.empty()) {
    avg_l1_latency = std::accumulate(all_l1_latency.begin(), all_l1_latency.end(), 0.0) / all_l1_latency.size();
  }
  if (!all_l2_latency.empty()) {
    avg_l2_latency = std::accumulate(all_l2_latency.begin(), all_l2_latency.end(), 0.0) / all_l2_latency.size();
  }
  double avg_main_mem_latency = std::accumulate(all_main_mem_latency.begin(), all_main_mem_latency.end(), 0.0) / loop_count;

  // --- Calculate Min/Max ---
  // Find the minimum and maximum values recorded across all loops for each test type.
  double min_read_bw = *std::min_element(all_read_bw.begin(), all_read_bw.end());
  double max_read_bw = *std::max_element(all_read_bw.begin(), all_read_bw.end());

  double min_write_bw = *std::min_element(all_write_bw.begin(), all_write_bw.end());
  double max_write_bw = *std::max_element(all_write_bw.begin(), all_write_bw.end());

  double min_copy_bw = *std::min_element(all_copy_bw.begin(), all_copy_bw.end());
  double max_copy_bw = *std::max_element(all_copy_bw.begin(), all_copy_bw.end());

  double min_l1_latency = 0.0, max_l1_latency = 0.0;
  double min_l2_latency = 0.0, max_l2_latency = 0.0;
  if (!all_l1_latency.empty()) {
    min_l1_latency = *std::min_element(all_l1_latency.begin(), all_l1_latency.end());
    max_l1_latency = *std::max_element(all_l1_latency.begin(), all_l1_latency.end());
  }
  if (!all_l2_latency.empty()) {
    min_l2_latency = *std::min_element(all_l2_latency.begin(), all_l2_latency.end());
    max_l2_latency = *std::max_element(all_l2_latency.begin(), all_l2_latency.end());
  }
  double min_main_mem_latency = *std::min_element(all_main_mem_latency.begin(), all_main_mem_latency.end());
  double max_main_mem_latency = *std::max_element(all_main_mem_latency.begin(), all_main_mem_latency.end());

  // --- Print Statistics ---
  // Set output to fixed-point notation.
  std::cout << std::fixed;
  // Display Read Bandwidth stats.
  std::cout << "Read Bandwidth (GB/s):" << std::endl;
  std::cout << "  Average: " << std::setprecision(3) << avg_read_bw << std::endl;
  std::cout << "  Min:     " << std::setprecision(3) << min_read_bw << std::endl;
  std::cout << "  Max:     " << std::setprecision(3) << max_read_bw << std::endl;

  // Display Write Bandwidth stats.
  std::cout << "\nWrite Bandwidth (GB/s):" << std::endl;
  std::cout << "  Average: " << std::setprecision(3) << avg_write_bw << std::endl;
  std::cout << "  Min:     " << std::setprecision(3) << min_write_bw << std::endl;
  std::cout << "  Max:     " << std::setprecision(3) << max_write_bw << std::endl;

  // Display Copy Bandwidth stats.
  std::cout << "\nCopy Bandwidth (GB/s):" << std::endl;
  std::cout << "  Average: " << std::setprecision(3) << avg_copy_bw << std::endl;
  std::cout << "  Min:     " << std::setprecision(3) << min_copy_bw << std::endl;
  std::cout << "  Max:     " << std::setprecision(3) << max_copy_bw << std::endl;

  // Display Cache Bandwidth stats.
  if (use_custom_cache_size) {
    // Display custom cache bandwidth statistics
    if (!all_custom_read_bw.empty()) {
      double avg_custom_read_bw = std::accumulate(all_custom_read_bw.begin(), all_custom_read_bw.end(), 0.0) / all_custom_read_bw.size();
      double avg_custom_write_bw = std::accumulate(all_custom_write_bw.begin(), all_custom_write_bw.end(), 0.0) / all_custom_write_bw.size();
      double avg_custom_copy_bw = std::accumulate(all_custom_copy_bw.begin(), all_custom_copy_bw.end(), 0.0) / all_custom_copy_bw.size();
      double min_custom_read_bw = *std::min_element(all_custom_read_bw.begin(), all_custom_read_bw.end());
      double max_custom_read_bw = *std::max_element(all_custom_read_bw.begin(), all_custom_read_bw.end());
      double min_custom_write_bw = *std::min_element(all_custom_write_bw.begin(), all_custom_write_bw.end());
      double max_custom_write_bw = *std::max_element(all_custom_write_bw.begin(), all_custom_write_bw.end());
      double min_custom_copy_bw = *std::min_element(all_custom_copy_bw.begin(), all_custom_copy_bw.end());
      double max_custom_copy_bw = *std::max_element(all_custom_copy_bw.begin(), all_custom_copy_bw.end());
      
      std::cout << "\nCustom Cache Bandwidth (GB/s):" << std::endl;
      std::cout << "  Read:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_custom_read_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_custom_read_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_custom_read_bw << std::endl;
      std::cout << "  Write:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_custom_write_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_custom_write_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_custom_write_bw << std::endl;
      std::cout << "  Copy:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_custom_copy_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_custom_copy_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_custom_copy_bw << std::endl;
    }
  } else {
    // Display L1/L2 cache bandwidth statistics
    if (!all_l1_read_bw.empty()) {
      double avg_l1_read_bw = std::accumulate(all_l1_read_bw.begin(), all_l1_read_bw.end(), 0.0) / all_l1_read_bw.size();
      double avg_l1_write_bw = std::accumulate(all_l1_write_bw.begin(), all_l1_write_bw.end(), 0.0) / all_l1_write_bw.size();
      double avg_l1_copy_bw = std::accumulate(all_l1_copy_bw.begin(), all_l1_copy_bw.end(), 0.0) / all_l1_copy_bw.size();
      double min_l1_read_bw = *std::min_element(all_l1_read_bw.begin(), all_l1_read_bw.end());
      double max_l1_read_bw = *std::max_element(all_l1_read_bw.begin(), all_l1_read_bw.end());
      double min_l1_write_bw = *std::min_element(all_l1_write_bw.begin(), all_l1_write_bw.end());
      double max_l1_write_bw = *std::max_element(all_l1_write_bw.begin(), all_l1_write_bw.end());
      double min_l1_copy_bw = *std::min_element(all_l1_copy_bw.begin(), all_l1_copy_bw.end());
      double max_l1_copy_bw = *std::max_element(all_l1_copy_bw.begin(), all_l1_copy_bw.end());
      
      std::cout << "\nL1 Cache Bandwidth (GB/s):" << std::endl;
      std::cout << "  Read:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l1_read_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l1_read_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l1_read_bw << std::endl;
      std::cout << "  Write:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l1_write_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l1_write_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l1_write_bw << std::endl;
      std::cout << "  Copy:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l1_copy_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l1_copy_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l1_copy_bw << std::endl;
    }
    
    if (!all_l2_read_bw.empty()) {
      double avg_l2_read_bw = std::accumulate(all_l2_read_bw.begin(), all_l2_read_bw.end(), 0.0) / all_l2_read_bw.size();
      double avg_l2_write_bw = std::accumulate(all_l2_write_bw.begin(), all_l2_write_bw.end(), 0.0) / all_l2_write_bw.size();
      double avg_l2_copy_bw = std::accumulate(all_l2_copy_bw.begin(), all_l2_copy_bw.end(), 0.0) / all_l2_copy_bw.size();
      double min_l2_read_bw = *std::min_element(all_l2_read_bw.begin(), all_l2_read_bw.end());
      double max_l2_read_bw = *std::max_element(all_l2_read_bw.begin(), all_l2_read_bw.end());
      double min_l2_write_bw = *std::min_element(all_l2_write_bw.begin(), all_l2_write_bw.end());
      double max_l2_write_bw = *std::max_element(all_l2_write_bw.begin(), all_l2_write_bw.end());
      double min_l2_copy_bw = *std::min_element(all_l2_copy_bw.begin(), all_l2_copy_bw.end());
      double max_l2_copy_bw = *std::max_element(all_l2_copy_bw.begin(), all_l2_copy_bw.end());
      
      std::cout << "\nL2 Cache Bandwidth (GB/s):" << std::endl;
      std::cout << "  Read:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l2_read_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l2_read_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l2_read_bw << std::endl;
      std::cout << "  Write:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l2_write_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l2_write_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l2_write_bw << std::endl;
      std::cout << "  Copy:" << std::endl;
      std::cout << "    Average: " << std::setprecision(3) << avg_l2_copy_bw << std::endl;
      std::cout << "    Min:     " << std::setprecision(3) << min_l2_copy_bw << std::endl;
      std::cout << "    Max:     " << std::setprecision(3) << max_l2_copy_bw << std::endl;
    }
  }

  // Display Cache Latency stats.
  std::cout << "\nCache Latency (ns):" << std::endl;
  if (use_custom_cache_size) {
    // Display custom cache latency statistics
    if (!all_custom_latency.empty()) {
      double avg_custom_latency = std::accumulate(all_custom_latency.begin(), all_custom_latency.end(), 0.0) / all_custom_latency.size();
      double min_custom_latency = *std::min_element(all_custom_latency.begin(), all_custom_latency.end());
      double max_custom_latency = *std::max_element(all_custom_latency.begin(), all_custom_latency.end());
      
      std::cout << "  Custom Cache:" << std::endl;
      std::cout << "    Average: " << std::setprecision(2) << avg_custom_latency << std::endl;
      std::cout << "    Min:     " << std::setprecision(2) << min_custom_latency << std::endl;
      std::cout << "    Max:     " << std::setprecision(2) << max_custom_latency << std::endl;
    }
  } else {
    // Display L1/L2 cache latency statistics
    if (!all_l1_latency.empty()) {
      std::cout << "  L1 Cache:" << std::endl;
      std::cout << "    Average: " << std::setprecision(2) << avg_l1_latency << std::endl;
      std::cout << "    Min:     " << std::setprecision(2) << min_l1_latency << std::endl;
      std::cout << "    Max:     " << std::setprecision(2) << max_l1_latency << std::endl;
    }
    if (!all_l2_latency.empty()) {
      std::cout << "  L2 Cache:" << std::endl;
      std::cout << "    Average: " << std::setprecision(2) << avg_l2_latency << std::endl;
      std::cout << "    Min:     " << std::setprecision(2) << min_l2_latency << std::endl;
      std::cout << "    Max:     " << std::setprecision(2) << max_l2_latency << std::endl;
    }
  }

  // Display Main Memory Latency stats.
  std::cout << "\nMain Memory Latency (ns):" << std::endl;
  std::cout << "  Average: " << std::setprecision(2) << avg_main_mem_latency << std::endl;
  std::cout << "  Min:     " << std::setprecision(2) << min_main_mem_latency << std::endl;
  std::cout << "  Max:     " << std::setprecision(2) << max_main_mem_latency << std::endl;
  // Print a final separator after statistics.
  std::cout << "----------------------------------" << std::endl;
}

// --- Print Cache Information ---
// Outputs the detected cache sizes for L1 and L2 cache levels, or custom cache size.
// 'l1_cache_size': L1 data cache size in bytes (per P-core).
// 'l2_cache_size': L2 cache size in bytes (per P-core cluster).
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'custom_cache_size_bytes': Custom cache size in bytes (only used if use_custom_cache_size is true).
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes) {
  std::cout << "\nDetected Cache Sizes:" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  
  if (use_custom_cache_size) {
    // Display custom cache size.
    if (custom_cache_size_bytes < 1024) {
      std::cout << "  Custom Cache Size: " << custom_cache_size_bytes << " B" << std::endl;
    } else if (custom_cache_size_bytes < 1024 * 1024) {
      std::cout << "  Custom Cache Size: " << custom_cache_size_bytes / 1024.0 << " KB" << std::endl;
    } else {
      std::cout << "  Custom Cache Size: " << custom_cache_size_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    }
  } else {
    // Display L1 cache size.
    if (l1_cache_size < 1024) {
      std::cout << "  L1 Cache Size: " << l1_cache_size << " B (per P-core)" << std::endl;
    } else if (l1_cache_size < 1024 * 1024) {
      std::cout << "  L1 Cache Size: " << l1_cache_size / 1024.0 << " KB (per P-core)" << std::endl;
    } else {
      std::cout << "  L1 Cache Size: " << l1_cache_size / (1024.0 * 1024.0) << " MB (per P-core)" << std::endl;
    }
    
    // Display L2 cache size.
    if (l2_cache_size < 1024) {
      std::cout << "  L2 Cache Size: " << l2_cache_size << " B (per P-core cluster)" << std::endl;
    } else if (l2_cache_size < 1024 * 1024) {
      std::cout << "  L2 Cache Size: " << l2_cache_size / 1024.0 << " KB (per P-core cluster)" << std::endl;
    } else {
      std::cout << "  L2 Cache Size: " << l2_cache_size / (1024.0 * 1024.0) << " MB (per P-core cluster)" << std::endl;
    }
  }
  
}