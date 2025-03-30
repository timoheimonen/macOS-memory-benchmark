// Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>
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
#include "benchmark.h"

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <limits>
#include <unistd.h>
#include <stdexcept> // For argument errors
#include <cstdlib>   // Exit codes
#include <cstdio>    // perror
#include <iomanip>   // Output formatting

// macOS specific memory management
#include <sys/mman.h>     // mmap / munmap
#include <unistd.h>       // getpagesize

// Custom deleter for memory allocated with mmap
struct MmapDeleter {
    size_t allocation_size; // Store the size needed for munmap

    // This function call operator is invoked by unique_ptr upon destruction
    void operator()(void* ptr) const {
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
    unsigned long buffer_size_mb = 512; // Default buffer size (MB)
    int iterations = 1000;              // Default test iterations
    int loop_count = 1;                 // Default benchmark loops
    const size_t lat_stride = 128;      // Latency test access stride (bytes)
    const size_t lat_num_accesses = 200 * 1000 * 1000; // Latency test access count

    // --- Get System Info ---
    std::string cpu_name = get_processor_name();    // Get CPU model
    int perf_cores = get_performance_cores();       // Get P-core count
    int eff_cores = get_efficiency_cores();         // Get E-core count
    int num_threads = get_total_logical_cores();    // Get total threads for BW tests

    // --- Calculate Memory Limit ---
    unsigned long available_mem_mb = get_available_memory_mb(); // Get free RAM (MB)
    unsigned long max_allowed_mb_per_buffer = 0;    // Max size per buffer allowed
    const double memory_limit_factor = 0.80;        // Use 80% of free RAM total
    const unsigned long fallback_total_limit_mb = 2048; // Fallback if detection fails
    const unsigned long minimum_limit_mb_per_buffer = 64; // Min allowed size per buffer

    // Try calculating limit based on available RAM
    if (available_mem_mb > 0) {
        unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * memory_limit_factor);
        max_allowed_mb_per_buffer = max_total_allowed_mb / 3; // Divide limit by 3 buffers
    } else {
        // Use fallback limit if RAM detection failed
        std::cerr << "Warning: Cannot get available memory. Using fallback limit." << std::endl;
        max_allowed_mb_per_buffer = fallback_total_limit_mb / 3;
        std::cout << "Info: Setting max per buffer to fallback: " << max_allowed_mb_per_buffer << " MB." << std::endl;
    }
    // Enforce minimum buffer size
    if (max_allowed_mb_per_buffer < minimum_limit_mb_per_buffer) {
        std::cout << "Info: Calculated max (" << max_allowed_mb_per_buffer
                  << " MB) < min (" << minimum_limit_mb_per_buffer
                  << " MB). Using min." << std::endl;
        max_allowed_mb_per_buffer = minimum_limit_mb_per_buffer;
    }

    // --- Parse Command Line Args ---
    long long requested_buffer_size_mb_ll = -1; // User requested size (-1 = none)

    // Loop through arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            // Handle -iterations N
            if (arg == "-iterations") {
                if (++i < argc) {
                    long long val_ll = std::stoll(argv[i]); // Read value
                    if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) throw std::out_of_range("iterations invalid");
                    iterations = static_cast<int>(val_ll); // Store value
                } else throw std::invalid_argument("Missing value for -iterations");
            // Handle -buffersize N
            } else if (arg == "-buffersize") {
                if (++i < argc) {
                    long long val_ll = std::stoll(argv[i]); // Read value
                    if (val_ll <= 0) throw std::out_of_range("buffersize must be > 0");
                    requested_buffer_size_mb_ll = val_ll; // Store requested size
                } else throw std::invalid_argument("Missing value for -buffersize");
            // Handle -count N
            } else if (arg == "-count") {
                 if (++i < argc) {
                    long long val_ll = std::stoll(argv[i]); // Read value
                    if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) throw std::out_of_range("count invalid");
                    loop_count = static_cast<int>(val_ll); // Store value
                 } else throw std::invalid_argument("Missing value for -count");
            // Handle -h or --help
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]); // Show help message
                return EXIT_SUCCESS;  // Exit cleanly
            } else {
                 throw std::invalid_argument("Unknown option: " + arg); // Unknown flag
            }
        // Catch argument errors
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        // Catch out-of-range errors
        } catch (const std::out_of_range& e) {
             std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
             print_usage(argv[0]);
             return EXIT_FAILURE;
        }
    }

    // --- Validate Final Buffer Size ---
    if (requested_buffer_size_mb_ll != -1) { // User specified a size
        unsigned long requested_ul = static_cast<unsigned long>(requested_buffer_size_mb_ll);
        // Check if requested size exceeds the limit
        if (requested_ul > max_allowed_mb_per_buffer) {
             std::cerr << "Warning: Requested buffer size (" << requested_buffer_size_mb_ll
                      << " MB) > limit (" << max_allowed_mb_per_buffer << " MB). Using limit." << std::endl;
             buffer_size_mb = max_allowed_mb_per_buffer; // Use the limit
        } else {
             buffer_size_mb = requested_ul; // Use user's valid size
        }
    } else { // User did not specify size, check default
         if (buffer_size_mb > max_allowed_mb_per_buffer) {
             std::cout << "Info: Default buffer size (" << buffer_size_mb
                       << " MB) > limit (" << max_allowed_mb_per_buffer << " MB). Using limit." << std::endl;
             buffer_size_mb = max_allowed_mb_per_buffer; // Use the limit
         }
         // Otherwise, default is fine
    }

    // --- Calculate final buffer size in bytes ---
    const size_t bytes_per_mb = 1024 * 1024;
    size_t buffer_size = static_cast<size_t>(buffer_size_mb) * bytes_per_mb; // Final size in bytes

    // --- Sanity Checks ---
    size_t page_size = getpagesize(); // Get OS page size
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

    // --- Print Config ---
    // Show final settings being used
    print_configuration(buffer_size, buffer_size_mb, iterations, loop_count, cpu_name, perf_cores, eff_cores, num_threads);

    // --- Allocate Memory ---
    using MmapPtr = std::unique_ptr<void, MmapDeleter>; // mapDeleter is defined above
    MmapPtr src_buffer_ptr(nullptr, MmapDeleter{0}); // Initialize with nullptr and size 0
    MmapPtr dst_buffer_ptr(nullptr, MmapDeleter{0});
    MmapPtr lat_buffer_ptr(nullptr, MmapDeleter{0});

    // --- Allocate Memory ---
    std::cout << "\n--- Allocating Buffers ---" << std::endl;

    // Allocate source buffer using mmap
    std::cout << "Allocating src buffer (" << std::fixed << std::setprecision(2) << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    void* temp_src = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_src == MAP_FAILED) {
        perror("mmap failed for src_buffer");
        // Cleanup of potentially successful previous allocations happens automatically via RAII on return.
        return EXIT_FAILURE;
    }
    // Assign the allocated memory and correct deleter info to the unique_ptr
    if (temp_src == MAP_FAILED) {
        perror("mmap failed for src_buffer");
        return EXIT_FAILURE;
    }
    src_buffer_ptr = MmapPtr(temp_src, MmapDeleter{buffer_size});


    // Allocate destination buffer
    std::cout << "Allocating dst buffer (" << std::fixed << std::setprecision(2) << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    void* temp_dst = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_dst == MAP_FAILED) {
        perror("mmap failed for dst_buffer");
        // src_buffer_ptr will be cleaned up automatically upon return
        return EXIT_FAILURE;
    }
    if (temp_dst == MAP_FAILED) {
        perror("mmap failed for dst_buffer");
        // src_buffer_ptr cleans up automatically
        return EXIT_FAILURE;
    }
    dst_buffer_ptr = MmapPtr(temp_dst, MmapDeleter{buffer_size});


    // Allocate latency buffer
    std::cout << "Allocating lat buffer (" << std::fixed << std::setprecision(2) << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    void* temp_lat = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (temp_lat == MAP_FAILED) {
        perror("mmap failed for lat_buffer");
        // src_buffer_ptr and dst_buffer_ptr will be cleaned up automatically upon return
        return EXIT_FAILURE;
    }
    if (temp_lat == MAP_FAILED) {
        perror("mmap failed for lat_buffer");
        // src_buffer_ptr and dst_buffer_ptr clean up automatically
        return EXIT_FAILURE;
    }
    lat_buffer_ptr = MmapPtr(temp_lat, MmapDeleter{buffer_size});

    std::cout << "Buffers allocated." << std::endl; 

    // --- Get raw pointers for functions that need them ---
    // Use .get() method to access the raw pointer managed by unique_ptr
    void* src_buffer = src_buffer_ptr.get(); // Get raw pointer from unique_ptr for function calls
    void* dst_buffer = dst_buffer_ptr.get(); // Get raw pointer from unique_ptr
    void* lat_buffer = lat_buffer_ptr.get(); // Get raw pointer from unique_ptr

    // --- Init Buffers & Latency Chain ---
    initialize_buffers(src_buffer, dst_buffer, buffer_size); // Fill buffers
    setup_latency_chain(lat_buffer, buffer_size, lat_stride); // Prepare latency test buffer

    // --- Warm-up Runs ---
    std::cout << "\nPerforming warm-up runs..." << std::endl;
    std::atomic<uint64_t> dummy_checksum_warmup_atomic = 0; // Dummy for read warmup
    warmup_read(src_buffer, buffer_size, num_threads, dummy_checksum_warmup_atomic); // Warmup read
    warmup_write(dst_buffer, buffer_size, num_threads);    // Warmup write
    warmup_copy(dst_buffer, src_buffer, buffer_size, num_threads); // Warmup copy
    warmup_latency(lat_buffer, lat_num_accesses);        // Warmup latency
    std::cout << "Warm-up complete." << std::endl;

    // --- Measurement Loops ---
    std::cout << "\n--- Starting Measurements (" << loop_count << " loops) ---" << std::endl;

    // Vectors to store results from each loop
    std::vector<double> all_read_bw_gb_s;
    std::vector<double> all_write_bw_gb_s;
    std::vector<double> all_copy_bw_gb_s;
    std::vector<double> all_average_latency_ns;

    // Pre-allocate vector space if needed
    if (loop_count > 0) {
        all_read_bw_gb_s.reserve(loop_count);
        all_write_bw_gb_s.reserve(loop_count);
        all_copy_bw_gb_s.reserve(loop_count);
        all_average_latency_ns.reserve(loop_count);
    }

    HighResTimer test_timer; // Timer for individual tests

    // Main benchmark loop
    for (int loop = 0; loop < loop_count; ++loop) {
        std::cout << "\nStarting Loop " << loop + 1 << " of " << loop_count << "..." << std::endl;

        // Per-loop result variables
        double total_read_time = 0.0, read_bw_gb_s = 0.0;
        double total_write_time = 0.0, write_bw_gb_s = 0.0;
        double total_copy_time = 0.0, copy_bw_gb_s = 0.0;
        double total_lat_time_ns = 0.0, average_latency_ns = 0.0;
        std::atomic<uint64_t> total_read_checksum = 0; // Checksum for read test

        // --- Run tests ---
        total_read_time = run_read_test(src_buffer, buffer_size, iterations, num_threads, total_read_checksum, test_timer);
        total_write_time = run_write_test(dst_buffer, buffer_size, iterations, num_threads, test_timer);
        total_copy_time = run_copy_test(dst_buffer, src_buffer, buffer_size, iterations, num_threads, test_timer);
        total_lat_time_ns = run_latency_test(lat_buffer, lat_num_accesses, test_timer);

        // --- Calculate results ---
        size_t total_bytes_read = static_cast<size_t>(iterations) * buffer_size;
        size_t total_bytes_written = static_cast<size_t>(iterations) * buffer_size;
        size_t total_bytes_copied_op = static_cast<size_t>(iterations) * buffer_size; // Data per copy op

        // Calculate Bandwidths (GB/s)
        if (total_read_time > 0) read_bw_gb_s = static_cast<double>(total_bytes_read) / total_read_time / 1e9;
        if (total_write_time > 0) write_bw_gb_s = static_cast<double>(total_bytes_written) / total_write_time / 1e9;
        if (total_copy_time > 0) copy_bw_gb_s = static_cast<double>(total_bytes_copied_op * 2) / total_copy_time / 1e9; // Copy = read + write
        // Calculate Latency (ns)
        if (lat_num_accesses > 0) average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);

        // Store results for this loop
        all_read_bw_gb_s.push_back(read_bw_gb_s);
        all_write_bw_gb_s.push_back(write_bw_gb_s);
        all_copy_bw_gb_s.push_back(copy_bw_gb_s);
        all_average_latency_ns.push_back(average_latency_ns);

        // Print results for this loop
        print_results(loop, buffer_size, buffer_size_mb, iterations, num_threads,
                      read_bw_gb_s, total_read_time,
                      write_bw_gb_s, total_write_time,
                      copy_bw_gb_s, total_copy_time,
                      average_latency_ns, total_lat_time_ns,
                      lat_num_accesses, lat_stride);

    } // End loop

    // --- Print Stats ---
    // Print summary statistics if more than one loop was run
    print_statistics(loop_count, all_read_bw_gb_s, all_write_bw_gb_s, all_copy_bw_gb_s, all_average_latency_ns);

    // --- Free Memory ---
    std::cout << "\nFreeing memory..." << std::endl;
    // Memory is freed automatically when src_buffer_ptr, dst_buffer_ptr,
    // and lat_buffer_ptr go out of scope. No manual munmap needed.
    std::cout << "Memory will be freed automatically." << std::endl;

    // --- Print Total Time ---
    double total_elapsed_time_sec = total_execution_timer.stop(); // Stop overall timer
    std::cout << std::fixed << std::setprecision(3); // Set output precision
    std::cout << "\nDone. Total execution time: " << total_elapsed_time_sec << " s" << std::endl; // Print duration

    return EXIT_SUCCESS; // Indicate success
}