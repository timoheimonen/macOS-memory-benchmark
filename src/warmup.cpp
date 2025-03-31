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
#include "benchmark.h" // Includes definitions for assembly loops etc.
#include <vector>      // For std::vector
#include <thread>      // For std::thread
#include <atomic>      // For std::atomic
#include <iostream>    // For std::cout

// Helper lambda to join all threads in a given vector.
auto join_threads = [](std::vector<std::thread>& threads) {
    // Iterate over each thread in the vector.
    for (auto& t : threads) {
        // Check if the thread is joinable (can be waited for).
        if (t.joinable()) {
            // Wait for the thread to complete its execution.
            t.join();
        }
    }
    // Remove all thread objects from the vector.
    threads.clear();
};

// Warms up memory by reading from the buffer using multiple threads.
// 'buffer': Memory region to read from.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
    std::cout << "  Read warm-up..." << std::endl;
    // Vector to store thread objects.
    std::vector<std::thread> threads;
    // Pre-allocate space for efficiency.
    threads.reserve(num_threads);
    // Tracks the current position in the buffer.
    size_t offset = 0;
    // Calculate the base size of the memory chunk per thread.
    size_t chunk_base_size = size / num_threads;
    // Calculate the remainder to distribute among threads.
    size_t chunk_remainder = size % num_threads;

    // Create and launch threads.
    for (int t = 0; t < num_threads; ++t) {
        // Calculate the exact size for this thread's chunk, adding remainder if needed.
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        // If chunk size is zero (e.g., size < num_threads), skip creating a thread.
        if (current_chunk_size == 0) continue;
        // Calculate the start address for this thread's chunk.
        char* chunk_start = static_cast<char*>(buffer) + offset;

        // Create a new thread to execute the read loop.
        threads.emplace_back([chunk_start, current_chunk_size, &dummy_checksum]() {
            // Call the assembly read loop function (defined elsewhere).
            uint64_t checksum = memory_read_loop_asm(chunk_start, current_chunk_size);
            // Atomically combine the local checksum with the global dummy value.
            // Using relaxed memory order as strict consistency isn't needed for warmup.
            dummy_checksum.fetch_xor(checksum, std::memory_order_relaxed);
        });
        // Move the offset for the next thread's chunk.
        offset += current_chunk_size;
    }
    // Wait for all worker threads to complete.
    join_threads(threads);
}

// Warms up memory by writing to the buffer using multiple threads.
// 'buffer': Memory region to write to.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_write(void* buffer, size_t size, int num_threads) {
    std::cout << "  Write warm-up..." << std::endl;
    // Vector to store thread objects.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t offset = 0;
    // Calculate chunk size and remainder (as in warmup_read).
    size_t chunk_base_size = size / num_threads;
    size_t chunk_remainder = size % num_threads;

    // Create and launch threads.
    for (int t = 0; t < num_threads; ++t) {
        // Calculate the exact size for this thread's chunk.
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        // Calculate the start address for this thread's chunk.
        char* chunk_start = static_cast<char*>(buffer) + offset;

        // Create a new thread to execute the assembly write loop function (defined elsewhere).
        threads.emplace_back([chunk_start, current_chunk_size]() {
            // Set QoS for this worker thread
            kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            if (qos_ret != KERN_SUCCESS) {
                 fprintf(stderr, "Warning: Failed to set QoS class for write warmup worker thread (code: %d)\n", qos_ret);
            }
            memory_write_loop_asm(chunk_start, current_chunk_size);
        });
        // Move the offset for the next thread's chunk.
        offset += current_chunk_size;
    }
    // Wait for all worker threads to complete.
    join_threads(threads);
}

// Warms up memory by copying data between buffers using multiple threads.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'size': Total size of data to copy in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_copy(void* dst, void* src, size_t size, int num_threads) {
    std::cout << "  Copy warm-up..." << std::endl;
    // Vector to store thread objects.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t offset = 0;
    // Calculate chunk size and remainder (as in warmup_read).
    size_t chunk_base_size = size / num_threads;
    size_t chunk_remainder = size % num_threads;

    // Create and launch threads.
    for (int t = 0; t < num_threads; ++t) {
        // Calculate the exact size for this thread's chunk.
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        // Calculate the source start address for this thread's chunk.
        char* src_chunk = static_cast<char*>(src) + offset;
        // Calculate the destination start address for this thread's chunk.
        char* dst_chunk = static_cast<char*>(dst) + offset;

        // Create a new thread to execute the assembly copy loop function (defined elsewhere).
        threads.emplace_back([dst_chunk, src_chunk, current_chunk_size]() {
             // Set QoS for this worker thread
             kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
             if (qos_ret != KERN_SUCCESS) {
                  fprintf(stderr, "Warning: Failed to set QoS class for copy warmup worker thread (code: %d)\n", qos_ret);
             }
            memory_copy_loop_asm(dst_chunk, src_chunk, current_chunk_size);
        });
        // Move the offset for the next thread's chunk.
        offset += current_chunk_size;
    }
    // Wait for all worker threads to complete.
    join_threads(threads);
}

// Warms up memory for latency test by chasing pointers (single thread).
// 'buffer': Memory region containing the pointer chain.
// 'num_accesses': Total number of pointer dereferences planned for the actual test.
void warmup_latency(void* buffer, size_t num_accesses) {
    std::cout << "  Latency warm-up (single thread)..." << std::endl;
    // Only perform warmup if there will be accesses in the main test.
    if (num_accesses > 0) {
       // Perform a small fraction of the total accesses for warmup (at least 1).
       // This helps bring relevant cache lines into the hierarchy.
       size_t warmup_accesses = std::max(static_cast<size_t>(1), num_accesses / 100);
       // Get the starting pointer from the buffer.
       uintptr_t* lat_warmup_ptr = static_cast<uintptr_t*>(buffer);
       // Call the assembly latency chase function (defined elsewhere).
       memory_latency_chase_asm(lat_warmup_ptr, warmup_accesses);
    }
}