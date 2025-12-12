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
// Include benchmark utilities
#include <unistd.h>

#include <atomic>
#include <cstdio>   // perror
#include <cstdlib>  // Exit codes
#include <iomanip>  // Output formatting
#include <iostream>
#include <limits>
#include <memory>     // std::unique_ptr
#include <stdexcept>  // For argument errors
#include <string>
#include <vector>

#include "benchmark.h"

// macOS specific memory management
#include <mach/mach.h>  // kern_return_t
#include <pthread/qos.h>
#include <sys/mman.h>  // mmap / munmap
#include <unistd.h>    // getpagesize

// Custom deleter for memory allocated with mmap
struct MmapDeleter {
  size_t allocation_size;  // Store the size needed for munmap

  // This function call operator is invoked by unique_ptr upon destruction
  void operator()(void *ptr) const {
    if (ptr && ptr != MAP_FAILED) {
      if (munmap(ptr, allocation_size) == -1) {
        // Log error if munmap fails, but don't throw from destructor
        perror("munmap failed in MmapDeleter");
      }
    }
  }
};

// Define a type alias for convenience
using MmapPtr = std::unique_ptr<void, MmapDeleter>;

// Main program entry
int main(int argc, char *argv[]) {
  // Start total execution timer
  HighResTimer total_execution_timer;
  total_execution_timer.start();

  // --- Default Settings ---
  unsigned long buffer_size_mb = 512;           // Default buffer size (MB)
  int iterations = 1000;                        // Default test iterations
  int loop_count = 1;                           // Default benchmark loops
  const size_t lat_stride = 128;                // Latency test access stride (bytes)
  size_t lat_num_accesses = 200 * 1000 * 1000;  // Latency test access count (will scale with buffer)

  // --- Get System Info ---
  std::string cpu_name = get_processor_name();  // Get CPU model
  int perf_cores = get_performance_cores();     // Get P-core count
  int eff_cores = get_efficiency_cores();       // Get E-core count
  int num_threads = get_total_logical_cores();  // Get total threads for BW tests

  // --- Get Cache Sizes ---
  size_t l1_cache_size = get_l1_cache_size();  // Get L1 cache size
  size_t l2_cache_size = get_l2_cache_size();  // Get L2 cache size

  // --- Calculate Cache Buffer Sizes ---
  // Use 75% of cache size for L1 and 10% for L2 to ensure fits within target level
  size_t l1_buffer_size = static_cast<size_t>(l1_cache_size * 0.75);
  size_t l2_buffer_size = static_cast<size_t>(l2_cache_size * 0.10);
  
  // Ensure buffer sizes are multiples of stride (128 bytes) and at least page size
  size_t page_size_check = getpagesize();
  l1_buffer_size = ((l1_buffer_size / lat_stride) * lat_stride);
  l2_buffer_size = ((l2_buffer_size / lat_stride) * lat_stride);
  
  // Ensure minimum size (at least 2 pointers worth)
  if (l1_buffer_size < lat_stride * 2) l1_buffer_size = lat_stride * 2;
  if (l2_buffer_size < lat_stride * 2) l2_buffer_size = lat_stride * 2;
  
  // Ensure buffer sizes are at least page size aligned
  if (l1_buffer_size < page_size_check) l1_buffer_size = page_size_check;
  if (l2_buffer_size < page_size_check) l2_buffer_size = page_size_check;

  // --- Calculate Memory Limit ---
  unsigned long available_mem_mb = get_available_memory_mb();  // Get free RAM (MB)
  unsigned long max_allowed_mb_per_buffer = 0;                 // Max size per buffer allowed
  const double memory_limit_factor = 0.80;                     // Use 80% of free RAM total
  const unsigned long fallback_total_limit_mb = 2048;          // Fallback if detection fails
  const unsigned long minimum_limit_mb_per_buffer = 64;        // Min allowed size per buffer

  // Try calculating limit based on available RAM
  if (available_mem_mb > 0) {
    unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * memory_limit_factor);
    max_allowed_mb_per_buffer = max_total_allowed_mb / 3;  // Divide limit by 3 buffers
  } else {
    // Use fallback limit if RAM detection failed
    std::cerr << "Warning: Cannot get available memory. Using fallback limit." << std::endl;
    max_allowed_mb_per_buffer = fallback_total_limit_mb / 3;
    std::cout << "Info: Setting max per buffer to fallback: " << max_allowed_mb_per_buffer << " MB." << std::endl;
  }
  // Enforce minimum buffer size
  if (max_allowed_mb_per_buffer < minimum_limit_mb_per_buffer) {
    std::cout << "Info: Calculated max (" << max_allowed_mb_per_buffer << " MB) < min (" << minimum_limit_mb_per_buffer
              << " MB). Using min." << std::endl;
    max_allowed_mb_per_buffer = minimum_limit_mb_per_buffer;
  }

  // --- Parse Command Line Args ---
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)

  // Loop through arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      // Handle -iterations N
      if (arg == "-iterations") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);  // Read value
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) throw std::out_of_range("iterations invalid");
          iterations = static_cast<int>(val_ll);  // Store value
        } else
          throw std::invalid_argument("Missing value for -iterations");
        // Handle -buffersize N
      } else if (arg == "-buffersize") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);  // Read value
          if (val_ll <= 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range("buffersize invalid");
          requested_buffer_size_mb_ll = val_ll;  // Store requested size
        } else
          throw std::invalid_argument("Missing value for -buffersize");
        // Handle -count N
      } else if (arg == "-count") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);  // Read value
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) throw std::out_of_range("count invalid");
          loop_count = static_cast<int>(val_ll);  // Store value
        } else
          throw std::invalid_argument("Missing value for -count");
        // Handle -h or --help
      } else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);  // Show help message
        return EXIT_SUCCESS;   // Exit cleanly
      } else {
        throw std::invalid_argument("Unknown option: " + arg);  // Unknown flag
      }
      // Catch argument errors
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
      // Catch out-of-range errors
    } catch (const std::out_of_range &e) {
      std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // --- Validate Final Buffer Size ---
  if (requested_buffer_size_mb_ll != -1) {  // User specified a size
    unsigned long requested_ul = static_cast<unsigned long>(requested_buffer_size_mb_ll);
    // Check if requested size exceeds the limit
    if (requested_ul > max_allowed_mb_per_buffer) {
      std::cerr << "Warning: Requested buffer size (" << requested_buffer_size_mb_ll << " MB) > limit ("
                << max_allowed_mb_per_buffer << " MB). Using limit." << std::endl;
      buffer_size_mb = max_allowed_mb_per_buffer;  // Use the limit
    } else {
      buffer_size_mb = requested_ul;  // Use user's valid size
    }
  } else {  // User did not specify size, check default
    if (buffer_size_mb > max_allowed_mb_per_buffer) {
      std::cout << "Info: Default buffer size (" << buffer_size_mb << " MB) > limit (" << max_allowed_mb_per_buffer
                << " MB). Using limit." << std::endl;
      buffer_size_mb = max_allowed_mb_per_buffer;  // Use the limit
    }
    // Otherwise, default is fine
  }

  // --- Calculate final buffer size in bytes ---
  const size_t bytes_per_mb = 1024 * 1024;
  size_t buffer_size = static_cast<size_t>(buffer_size_mb) * bytes_per_mb;  // Final size in bytes

  // --- Sanity Checks ---
  size_t page_size = getpagesize();  // Get OS page size
  // Check for size calculation overflow or zero size
  if (buffer_size_mb > 0 && (buffer_size == 0 || buffer_size / bytes_per_mb != buffer_size_mb)) {
    std::cerr << "Error: Buffer size calculation error (" << buffer_size_mb << " MB)." << std::endl;
    return EXIT_FAILURE;
  }
  // Check if buffer is too small for tests or page size
  if (buffer_size < page_size || buffer_size < lat_stride * 2) {
    std::cerr << "Error: Final buffer size (" << buffer_size << " bytes) is too small." << std::endl;
    return EXIT_FAILURE;
  }

  // Scale latency accesses proportionally to buffer size (e.g., ~200M for 512MB default)
  lat_num_accesses = static_cast<size_t>(200 * 1000 * 1000 * (static_cast<double>(buffer_size_mb) / 512.0));
  
  // Scale cache latency test access counts based on buffer size
  // More accesses for smaller buffers to get accurate timing
  size_t l1_num_accesses = 100 * 1000 * 1000;  // 100M accesses
  size_t l2_num_accesses = 50 * 1000 * 1000;   // 50M accesses

  // --- Print Config ---
  // Show final settings being used
  print_configuration(buffer_size, buffer_size_mb, iterations, loop_count, cpu_name, perf_cores, eff_cores,
                      num_threads);
  // Print cache information right after processor information
  print_cache_info(l1_cache_size, l2_cache_size);

  // --- Set QoS for the main thread (affects latency tests) ---
  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    // Non-critical error, just print a warning
    fprintf(stderr, "Warning: Failed to set QoS class for main thread (code: %d)\n", qos_ret);
  }

  // --- Allocate Memory ---
  // Reuse global MmapPtr alias
  MmapPtr src_buffer_ptr(nullptr, MmapDeleter{0});  // Initialize with nullptr and size 0
  MmapPtr dst_buffer_ptr(nullptr, MmapDeleter{0});
  MmapPtr lat_buffer_ptr(nullptr, MmapDeleter{0});
  MmapPtr l1_buffer_ptr(nullptr, MmapDeleter{0});  // Cache latency test buffers
  MmapPtr l2_buffer_ptr(nullptr, MmapDeleter{0});
  MmapPtr l1_bw_src_ptr(nullptr, MmapDeleter{0});  // Cache bandwidth test buffers (source)
  MmapPtr l1_bw_dst_ptr(nullptr, MmapDeleter{0});  // Cache bandwidth test buffers (destination)
  MmapPtr l2_bw_src_ptr(nullptr, MmapDeleter{0});  // Cache bandwidth test buffers (source)
  MmapPtr l2_bw_dst_ptr(nullptr, MmapDeleter{0});  // Cache bandwidth test buffers (destination)

  // --- Allocate Memory ---
  // Allocate source buffer using mmap
  void *temp_src = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (temp_src == MAP_FAILED) {
    perror("mmap failed for src_buffer");
    return EXIT_FAILURE;
  }
  src_buffer_ptr = MmapPtr(temp_src, MmapDeleter{buffer_size});
  if (madvise(temp_src, buffer_size, MADV_WILLNEED) == -1) {
    perror("madvise failed for src_buffer");
  }

  // Allocate destination buffer
  void *temp_dst = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (temp_dst == MAP_FAILED) {
    perror("mmap failed for dst_buffer");
    // src_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }
  dst_buffer_ptr = MmapPtr(temp_dst, MmapDeleter{buffer_size});
  if (madvise(temp_dst, buffer_size, MADV_WILLNEED) == -1) {
    perror("madvise failed for dst_buffer");
  }

  // Allocate latency buffer
  void *temp_lat = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (temp_lat == MAP_FAILED) {
    perror("mmap failed for lat_buffer");
    // src_buffer_ptr and dst_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }
  lat_buffer_ptr = MmapPtr(temp_lat, MmapDeleter{buffer_size});
  if (madvise(temp_lat, buffer_size, MADV_WILLNEED) == -1) {
    perror("madvise failed for lat_buffer");
  }

  // Allocate cache latency test buffers
  if (l1_buffer_size > 0) {
    void *temp_l1 = mmap(nullptr, l1_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l1 == MAP_FAILED) {
      perror("mmap failed for l1_buffer");
      return EXIT_FAILURE;
    }
    l1_buffer_ptr = MmapPtr(temp_l1, MmapDeleter{l1_buffer_size});
    if (madvise(temp_l1, l1_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l1_buffer");
    }
  }

  if (l2_buffer_size > 0) {
    void *temp_l2 = mmap(nullptr, l2_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l2 == MAP_FAILED) {
      perror("mmap failed for l2_buffer");
      return EXIT_FAILURE;
    }
    l2_buffer_ptr = MmapPtr(temp_l2, MmapDeleter{l2_buffer_size});
    if (madvise(temp_l2, l2_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l2_buffer");
    }
  }

  // Allocate cache bandwidth test buffers
  if (l1_buffer_size > 0) {
    // Allocate source buffer for L1 bandwidth tests
    void *temp_l1_bw_src = mmap(nullptr, l1_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l1_bw_src == MAP_FAILED) {
      perror("mmap failed for l1_bw_src_buffer");
      return EXIT_FAILURE;
    }
    l1_bw_src_ptr = MmapPtr(temp_l1_bw_src, MmapDeleter{l1_buffer_size});
    if (madvise(temp_l1_bw_src, l1_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l1_bw_src_buffer");
    }
    // Allocate destination buffer for L1 bandwidth tests
    void *temp_l1_bw_dst = mmap(nullptr, l1_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l1_bw_dst == MAP_FAILED) {
      perror("mmap failed for l1_bw_dst_buffer");
      return EXIT_FAILURE;
    }
    l1_bw_dst_ptr = MmapPtr(temp_l1_bw_dst, MmapDeleter{l1_buffer_size});
    if (madvise(temp_l1_bw_dst, l1_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l1_bw_dst_buffer");
    }
  }

  if (l2_buffer_size > 0) {
    // Allocate source buffer for L2 bandwidth tests
    void *temp_l2_bw_src = mmap(nullptr, l2_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l2_bw_src == MAP_FAILED) {
      perror("mmap failed for l2_bw_src_buffer");
      return EXIT_FAILURE;
    }
    l2_bw_src_ptr = MmapPtr(temp_l2_bw_src, MmapDeleter{l2_buffer_size});
    if (madvise(temp_l2_bw_src, l2_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l2_bw_src_buffer");
    }
    // Allocate destination buffer for L2 bandwidth tests
    void *temp_l2_bw_dst = mmap(nullptr, l2_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_l2_bw_dst == MAP_FAILED) {
      perror("mmap failed for l2_bw_dst_buffer");
      return EXIT_FAILURE;
    }
    l2_bw_dst_ptr = MmapPtr(temp_l2_bw_dst, MmapDeleter{l2_buffer_size});
    if (madvise(temp_l2_bw_dst, l2_buffer_size, MADV_WILLNEED) == -1) {
      perror("madvise failed for l2_bw_dst_buffer");
    }
  }

  // --- Get raw pointers for functions that need them ---
  // Use .get() method to access the raw pointer managed by unique_ptr
  void *src_buffer = src_buffer_ptr.get();  // Get raw pointer from unique_ptr for function calls
  void *dst_buffer = dst_buffer_ptr.get();  // Get raw pointer from unique_ptr
  void *lat_buffer = lat_buffer_ptr.get();  // Get raw pointer from unique_ptr
  void *l1_buffer = l1_buffer_ptr.get();    // Get raw pointer for cache latency test buffers
  void *l2_buffer = l2_buffer_ptr.get();
  void *l1_bw_src = l1_bw_src_ptr.get();   // Get raw pointer for L1 cache bandwidth test buffers
  void *l1_bw_dst = l1_bw_dst_ptr.get();
  void *l2_bw_src = l2_bw_src_ptr.get();   // Get raw pointer for L2 cache bandwidth test buffers
  void *l2_bw_dst = l2_bw_dst_ptr.get();

  // --- Init Buffers & Latency Chain ---
  initialize_buffers(src_buffer, dst_buffer, buffer_size);   // Fill buffers
  setup_latency_chain(lat_buffer, buffer_size, lat_stride);  // Prepare latency test buffer
  
  // Setup cache latency chains
  if (l1_buffer_size > 0) {
    setup_latency_chain(l1_buffer, l1_buffer_size, lat_stride);
  }
  if (l2_buffer_size > 0) {
    setup_latency_chain(l2_buffer, l2_buffer_size, lat_stride);
  }
  
  // Initialize cache bandwidth test buffers (not pointer chains, regular data)
  if (l1_buffer_size > 0 && l1_bw_src != nullptr && l1_bw_dst != nullptr) {
    initialize_buffers(l1_bw_src, l1_bw_dst, l1_buffer_size);
  }
  if (l2_buffer_size > 0 && l2_bw_src != nullptr && l2_bw_dst != nullptr) {
    initialize_buffers(l2_bw_src, l2_bw_dst, l2_buffer_size);
  }

  // --- Measurement Loops ---
  std::cout << "\nRunning benchmarks..." << std::endl;

  // Vectors to store results from each loop
  std::vector<double> all_read_bw_gb_s;
  std::vector<double> all_write_bw_gb_s;
  std::vector<double> all_copy_bw_gb_s;
  std::vector<double> all_l1_latency_ns;
  std::vector<double> all_l2_latency_ns;
  std::vector<double> all_average_latency_ns;
  // Cache bandwidth result vectors
  std::vector<double> all_l1_read_bw_gb_s;
  std::vector<double> all_l1_write_bw_gb_s;
  std::vector<double> all_l1_copy_bw_gb_s;
  std::vector<double> all_l2_read_bw_gb_s;
  std::vector<double> all_l2_write_bw_gb_s;
  std::vector<double> all_l2_copy_bw_gb_s;

  // Pre-allocate vector space if needed
  if (loop_count > 0) {
    all_read_bw_gb_s.reserve(loop_count);
    all_write_bw_gb_s.reserve(loop_count);
    all_copy_bw_gb_s.reserve(loop_count);
    all_l1_latency_ns.reserve(loop_count);
    all_l2_latency_ns.reserve(loop_count);
    all_average_latency_ns.reserve(loop_count);
    if (l1_buffer_size > 0) {
      all_l1_read_bw_gb_s.reserve(loop_count);
      all_l1_write_bw_gb_s.reserve(loop_count);
      all_l1_copy_bw_gb_s.reserve(loop_count);
    }
    if (l2_buffer_size > 0) {
      all_l2_read_bw_gb_s.reserve(loop_count);
      all_l2_write_bw_gb_s.reserve(loop_count);
      all_l2_copy_bw_gb_s.reserve(loop_count);
    }
  }

  HighResTimer test_timer;  // Timer for individual tests

  // Main benchmark loop
  for (int loop = 0; loop < loop_count; ++loop) {
    show_progress();

    // Per-loop result variables
    double total_read_time = 0.0, read_bw_gb_s = 0.0;
    double total_write_time = 0.0, write_bw_gb_s = 0.0;
    double total_copy_time = 0.0, copy_bw_gb_s = 0.0;
    double l1_lat_time_ns = 0.0, l1_latency_ns = 0.0;
    double l2_lat_time_ns = 0.0, l2_latency_ns = 0.0;
    double total_lat_time_ns = 0.0, average_latency_ns = 0.0;
    std::atomic<uint64_t> total_read_checksum{0};  // Checksum for read test
    // Cache bandwidth result variables
    double l1_read_time = 0.0, l1_read_bw_gb_s = 0.0;
    double l1_write_time = 0.0, l1_write_bw_gb_s = 0.0;
    double l1_copy_time = 0.0, l1_copy_bw_gb_s = 0.0;
    double l2_read_time = 0.0, l2_read_bw_gb_s = 0.0;
    double l2_write_time = 0.0, l2_write_bw_gb_s = 0.0;
    double l2_copy_time = 0.0, l2_copy_bw_gb_s = 0.0;
    std::atomic<uint64_t> l1_read_checksum{0};
    std::atomic<uint64_t> l2_read_checksum{0};

    try {
      // --- Run tests - warmups are done before each test ---
      show_progress();
      std::atomic<uint64_t> warmup_read_checksum{0};
      warmup_read(src_buffer, buffer_size, num_threads, warmup_read_checksum);
      total_read_time =
          run_read_test(src_buffer, buffer_size, iterations, num_threads, total_read_checksum, test_timer);
      show_progress();
      warmup_write(dst_buffer, buffer_size, num_threads);
      total_write_time = run_write_test(dst_buffer, buffer_size, iterations, num_threads, test_timer);
      show_progress();
      warmup_copy(dst_buffer, src_buffer, buffer_size, num_threads);
      total_copy_time = run_copy_test(dst_buffer, src_buffer, buffer_size, iterations, num_threads, test_timer);

      // --- Cache Bandwidth Tests (single-threaded) ---
      // Use more iterations for cache tests since buffers are small (10x more for better timing accuracy)
      int cache_iterations = iterations * 10;
      const int single_thread = 1;  // Cache bandwidth tests use single thread
      
      if (l1_buffer_size > 0 && l1_bw_src != nullptr && l1_bw_dst != nullptr) {
        show_progress();
        std::atomic<uint64_t> l1_warmup_read_checksum{0};
        warmup_cache_read(l1_bw_src, l1_buffer_size, single_thread, l1_warmup_read_checksum);
        l1_read_time = run_read_test(l1_bw_src, l1_buffer_size, cache_iterations, single_thread, l1_read_checksum, test_timer);
        warmup_cache_write(l1_bw_dst, l1_buffer_size, single_thread);
        l1_write_time = run_write_test(l1_bw_dst, l1_buffer_size, cache_iterations, single_thread, test_timer);
        warmup_cache_copy(l1_bw_dst, l1_bw_src, l1_buffer_size, single_thread);
        l1_copy_time = run_copy_test(l1_bw_dst, l1_bw_src, l1_buffer_size, cache_iterations, single_thread, test_timer);
      }
      
      if (l2_buffer_size > 0 && l2_bw_src != nullptr && l2_bw_dst != nullptr) {
        show_progress();
        std::atomic<uint64_t> l2_warmup_read_checksum{0};
        warmup_cache_read(l2_bw_src, l2_buffer_size, single_thread, l2_warmup_read_checksum);
        l2_read_time = run_read_test(l2_bw_src, l2_buffer_size, cache_iterations, single_thread, l2_read_checksum, test_timer);
        warmup_cache_write(l2_bw_dst, l2_buffer_size, single_thread);
        l2_write_time = run_write_test(l2_bw_dst, l2_buffer_size, cache_iterations, single_thread, test_timer);
        warmup_cache_copy(l2_bw_dst, l2_bw_src, l2_buffer_size, single_thread);
        l2_copy_time = run_copy_test(l2_bw_dst, l2_bw_src, l2_buffer_size, cache_iterations, single_thread, test_timer);
      }

      // --- Cache Latency Tests (run before RAM latency test) ---
      if (l1_buffer_size > 0 && l1_buffer != nullptr) {
        show_progress();
        warmup_cache_latency(l1_buffer, l1_num_accesses);
        l1_lat_time_ns = run_cache_latency_test(l1_buffer, l1_buffer_size, l1_num_accesses, test_timer);
        l1_latency_ns = l1_lat_time_ns / static_cast<double>(l1_num_accesses);
      }
      
      if (l2_buffer_size > 0 && l2_buffer != nullptr) {
        show_progress();
        warmup_cache_latency(l2_buffer, l2_num_accesses);
        l2_lat_time_ns = run_cache_latency_test(l2_buffer, l2_buffer_size, l2_num_accesses, test_timer);
        l2_latency_ns = l2_lat_time_ns / static_cast<double>(l2_num_accesses);
      }

      // Warm latency immediately before measuring it to keep cache state representative
      show_progress();
      warmup_latency(lat_buffer, lat_num_accesses);
      total_lat_time_ns = run_latency_test(lat_buffer, lat_num_accesses, test_timer);
    } catch (const std::exception &e) {
      std::cerr << "Error during benchmark tests: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }

    // --- Calculate results ---
    size_t total_bytes_read = static_cast<size_t>(iterations) * buffer_size;
    size_t total_bytes_written = static_cast<size_t>(iterations) * buffer_size;
    size_t total_bytes_copied_op = static_cast<size_t>(iterations) * buffer_size;  // Data per copy op

    // Calculate Bandwidths (GB/s)
    if (total_read_time > 0) read_bw_gb_s = static_cast<double>(total_bytes_read) / total_read_time / 1e9;
    if (total_write_time > 0) write_bw_gb_s = static_cast<double>(total_bytes_written) / total_write_time / 1e9;
    if (total_copy_time > 0)
      copy_bw_gb_s =
          static_cast<double>(total_bytes_copied_op * 2) / total_copy_time / 1e9;  // Copy = read + write (documented)
    
    // Calculate Cache Bandwidths (GB/s)
    int cache_iterations = iterations * 10;
    if (l1_buffer_size > 0) {
      size_t l1_total_bytes_read = static_cast<size_t>(cache_iterations) * l1_buffer_size;
      size_t l1_total_bytes_written = static_cast<size_t>(cache_iterations) * l1_buffer_size;
      size_t l1_total_bytes_copied_op = static_cast<size_t>(cache_iterations) * l1_buffer_size;
      if (l1_read_time > 0) l1_read_bw_gb_s = static_cast<double>(l1_total_bytes_read) / l1_read_time / 1e9;
      if (l1_write_time > 0) l1_write_bw_gb_s = static_cast<double>(l1_total_bytes_written) / l1_write_time / 1e9;
      if (l1_copy_time > 0) l1_copy_bw_gb_s = static_cast<double>(l1_total_bytes_copied_op * 2) / l1_copy_time / 1e9;
    }
    if (l2_buffer_size > 0) {
      size_t l2_total_bytes_read = static_cast<size_t>(cache_iterations) * l2_buffer_size;
      size_t l2_total_bytes_written = static_cast<size_t>(cache_iterations) * l2_buffer_size;
      size_t l2_total_bytes_copied_op = static_cast<size_t>(cache_iterations) * l2_buffer_size;
      if (l2_read_time > 0) l2_read_bw_gb_s = static_cast<double>(l2_total_bytes_read) / l2_read_time / 1e9;
      if (l2_write_time > 0) l2_write_bw_gb_s = static_cast<double>(l2_total_bytes_written) / l2_write_time / 1e9;
      if (l2_copy_time > 0) l2_copy_bw_gb_s = static_cast<double>(l2_total_bytes_copied_op * 2) / l2_copy_time / 1e9;
    }
    
    // Calculate Latency (ns)
    if (lat_num_accesses > 0) average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);

    // Store results for this loop
    all_read_bw_gb_s.push_back(read_bw_gb_s);
    all_write_bw_gb_s.push_back(write_bw_gb_s);
    all_copy_bw_gb_s.push_back(copy_bw_gb_s);
    if (l1_buffer_size > 0) {
      all_l1_latency_ns.push_back(l1_latency_ns);
      all_l1_read_bw_gb_s.push_back(l1_read_bw_gb_s);
      all_l1_write_bw_gb_s.push_back(l1_write_bw_gb_s);
      all_l1_copy_bw_gb_s.push_back(l1_copy_bw_gb_s);
    }
    if (l2_buffer_size > 0) {
      all_l2_latency_ns.push_back(l2_latency_ns);
      all_l2_read_bw_gb_s.push_back(l2_read_bw_gb_s);
      all_l2_write_bw_gb_s.push_back(l2_write_bw_gb_s);
      all_l2_copy_bw_gb_s.push_back(l2_copy_bw_gb_s);
    }
    all_average_latency_ns.push_back(average_latency_ns);

    // Print results for this loop
    std::cout << '\r' << std::flush;  // Clear progress indicator
    print_results(loop, buffer_size, buffer_size_mb, iterations, num_threads, read_bw_gb_s, total_read_time,
                  write_bw_gb_s, total_write_time, copy_bw_gb_s, total_copy_time,
                  l1_latency_ns, l2_latency_ns,
                  l1_buffer_size, l2_buffer_size,
                  l1_read_bw_gb_s, l1_write_bw_gb_s, l1_copy_bw_gb_s,
                  l2_read_bw_gb_s, l2_write_bw_gb_s, l2_copy_bw_gb_s,
                  average_latency_ns, total_lat_time_ns);

  }  // End loop

  // --- Print Stats ---
  // Print summary statistics if more than one loop was run
  print_statistics(loop_count, all_read_bw_gb_s, all_write_bw_gb_s, all_copy_bw_gb_s,
                   all_l1_latency_ns, all_l2_latency_ns,
                   all_l1_read_bw_gb_s, all_l1_write_bw_gb_s, all_l1_copy_bw_gb_s,
                   all_l2_read_bw_gb_s, all_l2_write_bw_gb_s, all_l2_copy_bw_gb_s,
                   all_average_latency_ns);

  // --- Free Memory ---
  // std::cout << "\nFreeing memory..." << std::endl;
  // Memory is freed automatically when src_buffer_ptr, dst_buffer_ptr,
  // and lat_buffer_ptr go out of scope. No manual munmap needed.

  // --- Print Total Time ---
  double total_elapsed_time_sec = total_execution_timer.stop();                                  // Stop overall timer
  std::cout << std::fixed << std::setprecision(3);                                               // Set output precision
  std::cout << "\nDone. Total execution time: " << total_elapsed_time_sec << " s" << std::endl;  // Print duration

  return EXIT_SUCCESS;  // Indicate success
}