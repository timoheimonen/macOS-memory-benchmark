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
#include "benchmark.h" // Include benchmark definitions (assembly funcs, HighResTimer)
#include <vector>      // For std::vector
#include <thread>      // For std::thread
#include <atomic>      // For std::atomic
#include <iostream>    // For std::cout

// --- Local Helper ---
// Unnamed namespace restricts helper visibility to this file.
namespace {
    // Joins all threads in the provided vector and clears the vector.
    auto join_threads = [](std::vector<std::thread>& threads) {
        for (auto& t : threads) {
            if (t.joinable()) { // Check if thread is joinable
                t.join();       // Wait for thread completion
            }
        }
        threads.clear(); // Remove thread objects after joining
    };
}

// --- Benchmark Execution Functions ---

// Executes the multi-threaded read bandwidth benchmark.
// 'buffer': Pointer to the memory buffer to read.
// 'size': Size of the buffer in bytes.
// 'iterations': How many times to read the entire buffer.
// 'num_threads': Number of threads to use.
// 'checksum': Atomic variable to accumulate checksums from threads (ensures work isn't optimized away).
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_read_test(void* buffer, size_t size, int iterations, int num_threads, std::atomic<uint64_t>& checksum, HighResTimer& timer) {
    std::cout << "Measuring Read Bandwidth..." << std::endl;
    checksum = 0; // Ensure checksum starts at 0 for the measurement pass.
    std::vector<std::thread> threads;
    threads.reserve(num_threads); // Pre-allocate vector space for threads.

    // Divide work among threads (chunks calculated once).
    size_t offset = 0;
    size_t chunk_base_size = size / num_threads;
    size_t chunk_remainder = size % num_threads;

    timer.start(); // Start timing before launching threads.

    // Launch threads once; each handles its chunk for all iterations.
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue; // Avoid launching threads for zero work.
        char* chunk_start = static_cast<char*>(buffer) + offset;

        // iterations loop inside thread lambda for re-use (avoids per-iteration creation).
        threads.emplace_back([chunk_start, current_chunk_size, &checksum, iterations]() {
            // Set QoS for this worker thread
            kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            if (qos_ret != KERN_SUCCESS) {
                 fprintf(stderr, "Warning: Failed to set QoS class for read worker thread (code: %d)\n", qos_ret);
            }
            for (int i = 0; i < iterations; ++i) {
                // Call external assembly function for reading.
                uint64_t thread_checksum = memory_read_loop_asm(chunk_start, current_chunk_size);
                // Atomically combine result (relaxed order is sufficient).
                checksum.fetch_xor(thread_checksum, std::memory_order_relaxed);
            }
        });
        offset += current_chunk_size; // Update offset for the next chunk.
    }
    join_threads(threads); // Wait for all threads to finish (joined once after all iterations).
    double duration = timer.stop(); // Stop timing after all work.
    std::cout << "Read complete." << std::endl;
    return duration; // Return total time elapsed.
}

// Executes the multi-threaded write bandwidth benchmark.
// 'buffer': Pointer to the memory buffer to write to.
// 'size': Size of the buffer in bytes.
// 'iterations': How many times to write the entire buffer.
// 'num_threads': Number of threads to use.
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_write_test(void* buffer, size_t size, int iterations, int num_threads, HighResTimer& timer) {
    std::cout << "Measuring Write Bandwidth..." << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(num_threads); // Pre-allocate vector space.

    // Divide work among threads (chunks calculated once).
    size_t offset = 0;
    size_t chunk_base_size = size / num_threads;
    size_t chunk_remainder = size % num_threads;

    timer.start(); // Start timing.

    // Launch threads once; each handles its chunk for all iterations.
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* chunk_start = static_cast<char*>(buffer) + offset;

        // iterations loop inside thread lambda for re-use.
        threads.emplace_back([chunk_start, current_chunk_size, iterations]() { // Note: Lambda doesn't need checksum
             // Set QoS for this worker thread
             kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
             if (qos_ret != KERN_SUCCESS) {
                  fprintf(stderr, "Warning: Failed to set QoS class for write worker thread (code: %d)\n", qos_ret);
             }
             for (int i = 0; i < iterations; ++i) {
                 memory_write_loop_asm(chunk_start, current_chunk_size);
             }
         });
        offset += current_chunk_size;
    }
    join_threads(threads); // Wait for all threads to finish (joined once).
    double duration = timer.stop(); // Stop timing.
    std::cout << "Write complete." << std::endl;
    return duration; // Return total time elapsed.
}

// Executes the multi-threaded copy bandwidth benchmark.
// 'dst': Pointer to the destination buffer.
// 'src': Pointer to the source buffer.
// 'size': Size of the data to copy in bytes.
// 'iterations': How many times to copy the data.
// 'num_threads': Number of threads to use.
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_copy_test(void* dst, void* src, size_t size, int iterations, int num_threads, HighResTimer& timer) {
    std::cout << "Measuring Copy Bandwidth..." << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(num_threads); // Pre-allocate vector space.

    // Divide work among threads (chunks calculated once).
    size_t offset = 0;
    size_t chunk_base_size = size / num_threads;
    size_t chunk_remainder = size % num_threads;

    timer.start(); // Start timing.

    // Launch threads once; each handles its chunk for all iterations.
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* src_chunk = static_cast<char*>(src) + offset;
        char* dst_chunk = static_cast<char*>(dst) + offset;

        // iterations loop inside thread lambda for re-use.
        threads.emplace_back([dst_chunk, src_chunk, current_chunk_size, iterations]() {
             // Set QoS for this worker thread
             kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
             if (qos_ret != KERN_SUCCESS) {
                  fprintf(stderr, "Warning: Failed to set QoS class for copy worker thread (code: %d)\n", qos_ret);
             }
             for (int i = 0; i < iterations; ++i) {
                 memory_copy_loop_asm(dst_chunk, src_chunk, current_chunk_size);
             }
        });
        offset += current_chunk_size;
    }
    join_threads(threads); // Wait for all threads to finish (joined once).
    double duration = timer.stop(); // Stop timing.
    std::cout << "Copy complete." << std::endl;
    return duration; // Return total time elapsed.
}

// Executes the single-threaded memory latency benchmark.
// 'buffer': Pointer to the buffer containing the pointer chain.
// 'num_accesses': Total number of pointer dereferences to perform.
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in nanoseconds.
double run_latency_test(void* buffer, size_t num_accesses, HighResTimer& timer) {
    std::cout << "Measuring Latency (single thread)..." << std::endl;
    // Get the starting address of the pointer chain.
    uintptr_t* lat_start_ptr = static_cast<uintptr_t*>(buffer);
    timer.start(); // Start timing.
    // Call external assembly function to chase the pointer chain.
    memory_latency_chase_asm(lat_start_ptr, num_accesses);
    // Stop timing, getting result in nanoseconds for latency.
    double duration_ns = timer.stop_ns();
    std::cout << "Latency complete." << std::endl;
    return duration_ns; // Return total time elapsed in nanoseconds.
}