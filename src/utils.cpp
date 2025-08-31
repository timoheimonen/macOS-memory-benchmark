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

#include "benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)

// --- Helper function to print usage instructions ---
// Displays how to use the program via command-line arguments.
// 'prog_name': The name of the executable (typically argv[0]).
void print_usage(const char *prog_name) {
  // Output usage syntax, version, options, and an example to standard error.
  std::cerr << "Usage: " << prog_name << " [options]\n"
            << "Version: " << SOFTVERSION << "\n\n"  // SOFTVERSION defined in benchmark.h
            << "Options:\n"
            << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: 1000)\n"
            << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: 512).\n"
            << "                        The maximum allowed <size_mb> is automatically determined such that\n"
            << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
            << "  -count <count>        Number of full loops (read/write/copy/latency) (default: 1)\n"
            << "  -h, --help            Show this help message and exit\n\n"
            << "Example: " << prog_name << " -iterations 500 -buffersize 1024\n";
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
// 'average_latency_ns', 'total_lat_time_ns': Results from the latency test.
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
                   double read_bw_gb_s, double total_read_time, double write_bw_gb_s, double total_write_time,
                   double copy_bw_gb_s, double total_copy_time, double average_latency_ns, double total_lat_time_ns) {
  // Print a header indicating the current loop number.
  std::cout << "\n--- Results (Loop " << loop + 1 << ") ---" << std::endl;
  // Set output to fixed-point notation for consistent formatting.
  std::cout << std::fixed;

  // Display bandwidth test results.
  std::cout << "Bandwidth Tests (multi-threaded, " << num_threads << " threads):" << std::endl;
  std::cout << "  Read : " << std::setprecision(3) << read_bw_gb_s << " GB/s (Total time: " << total_read_time << " s)"
            << std::endl;
  std::cout << "  Write: " << std::setprecision(3) << write_bw_gb_s << " GB/s (Total time: " << total_write_time
            << " s)" << std::endl;
  std::cout << "  Copy : " << std::setprecision(3) << copy_bw_gb_s << " GB/s (Total time: " << total_copy_time << " s)"
            << std::endl;

  // Display latency test results.
  std::cout << "\nLatency Test (single-threaded, pointer chase):" << std::endl;
  std::cout << "  Total time: " << std::setprecision(3) << total_lat_time_ns / 1e9 << " s"
            << std::endl;  // Show latency test time in seconds
  std::cout << "  Average latency: " << std::setprecision(2) << average_latency_ns << " ns" << std::endl;
  // Print a separator after the loop results.
  std::cout << "--------------" << std::endl;
}

// --- Print Statistics across all loops ---
// Calculates and displays summary statistics (Avg/Min/Max) if more than one loop was run.
// 'loop_count': The total number of loops that were executed.
// 'all_read_bw', 'all_write_bw', 'all_copy_bw', 'all_latency': Vectors holding the results from each individual loop.
void print_statistics(int loop_count, const std::vector<double> &all_read_bw, const std::vector<double> &all_write_bw,
                      const std::vector<double> &all_copy_bw, const std::vector<double> &all_latency) {
  // Don't print statistics if only one loop ran or if there's no data.
  if (loop_count <= 1 || all_read_bw.empty()) return;

  // Print statistics header.
  std::cout << "\n--- Statistics Across " << loop_count << " Loops ---" << std::endl;

  // --- Calculate Averages ---
  // Sum up all results for each test type and divide by the number of loops.
  double avg_read_bw = std::accumulate(all_read_bw.begin(), all_read_bw.end(), 0.0) / loop_count;
  double avg_write_bw = std::accumulate(all_write_bw.begin(), all_write_bw.end(), 0.0) / loop_count;
  double avg_copy_bw = std::accumulate(all_copy_bw.begin(), all_copy_bw.end(), 0.0) / loop_count;
  double avg_latency = std::accumulate(all_latency.begin(), all_latency.end(), 0.0) / loop_count;

  // --- Calculate Min/Max ---
  // Find the minimum and maximum values recorded across all loops for each test type.
  double min_read_bw = *std::min_element(all_read_bw.begin(), all_read_bw.end());
  double max_read_bw = *std::max_element(all_read_bw.begin(), all_read_bw.end());

  double min_write_bw = *std::min_element(all_write_bw.begin(), all_write_bw.end());
  double max_write_bw = *std::max_element(all_write_bw.begin(), all_write_bw.end());

  double min_copy_bw = *std::min_element(all_copy_bw.begin(), all_copy_bw.end());
  double max_copy_bw = *std::max_element(all_copy_bw.begin(), all_copy_bw.end());

  double min_latency = *std::min_element(all_latency.begin(), all_latency.end());
  double max_latency = *std::max_element(all_latency.begin(), all_latency.end());

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

  // Display Latency stats.
  std::cout << "\nLatency (ns):" << std::endl;
  std::cout << "  Average: " << std::setprecision(2) << avg_latency << std::endl;
  std::cout << "  Min:     " << std::setprecision(2) << min_latency << std::endl;
  std::cout << "  Max:     " << std::setprecision(2) << max_latency << std::endl;
  // Print a final separator after statistics.
  std::cout << "----------------------------------" << std::endl;
}