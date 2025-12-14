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
#include <iomanip>    // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>   // Required for std::cout, std::cerr

#include "benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)
#include "constants.h"  // Include constants for default values

// --- Helper function to print usage instructions ---
// Displays how to use the program via command-line arguments.
// 'prog_name': The name of the executable (typically argv[0]).
void print_usage(const char *prog_name) {
  // Output usage syntax, version, options, and an example to standard error.
  std::cerr << "Version: " << SOFTVERSION << " by Timo Heimonen <timo.heimonen@proton.me>\n"  // SOFTVERSION defined in benchmark.h
            << "License: GNU GPL v3. See <https://www.gnu.org/licenses/>\n"
            << "This program is free software: you can redistribute it and/or modify\n"
            << "it under the terms of the GNU General Public License as published by\n"
            << "the Free Software Foundation, either version 3 of the License, or\n"
            << "(at your option) any later version.\n"
            << "This program is distributed in the hope that it will be useful,\n"
            << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
            << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
            << "Link: https://github.com/timoheimonen/macOS-memory-benchmark\n\n"
            << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: " << Constants::DEFAULT_ITERATIONS << ")\n"
            << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: " << Constants::DEFAULT_BUFFER_SIZE_MB << ").\n"
            << "                        The maximum allowed <size_mb> is automatically determined such that\n"
            << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
            << "  -count <count>        Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ")\n"
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
  std::cout << "Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>" << std::endl;
  std::cout << "This program is free software: you can redistribute it and/or modify" << std::endl;
  std::cout << "it under the terms of the GNU General Public License as published by" << std::endl;
  std::cout << "the Free Software Foundation, either version 3 of the License, or" << std::endl;
  std::cout << "(at your option) any later version." << std::endl;
  std::cout << "This program is distributed in the hope that it will be useful," << std::endl;
  std::cout << "but WITHOUT ANY WARRANTY; without even the implied warranty of" << std::endl;
  std::cout << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE." << std::endl;
  std::cout << "See <https://www.gnu.org/licenses/> for more details.\n" << std::endl;
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
