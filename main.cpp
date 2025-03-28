// Version 0.19
//
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

#include <iostream>       // cout
#include <vector>         // std::vector
#include <chrono>         // Timers (fallback)
#include <numeric>        // std::iota, std::accumulate
#include <cstdlib>        // exit, EXIT_FAILURE, perror
#include <cstddef>        // size_t
#include <random>         // std::shuffle, std::random_device, std::mt19937_64
#include <algorithm>      // std::shuffle, std::min
#include <cstdint>        // uintptr_t
#include <thread>         // std::thread, std::thread::hardware_concurrency
#include <functional>     // std::ref
#include <atomic>         // std::atomic

// macOS-specific headers
#include <sys/mman.h>     // mmap / munmap
#include <unistd.h>       // getpagesize
#include <mach/mach_time.h> // mach_absolute_time (high-resolution timer)
#include <sys/sysctl.h>   // sysctlbyname (for core count)

// --- Version Information ---
#define SOFTVERSION 0.19f // Updated version number

// Get total logical core count (P+E) via sysctl.
int get_total_logical_cores() {
    int p_cores = 0;
    int e_cores = 0;
    size_t len = sizeof(int);
    bool p_core_ok = false;
    bool e_core_ok = false;

    // Try P-cores (perf level 0)
    if (sysctlbyname("hw.perflevel0.logicalcpu_max", &p_cores, &len, NULL, 0) == 0 && p_cores > 0) {
        p_core_ok = true;
    } else {
        p_cores = 0; // Ensure 0 on failure
    }

    // Try E-cores (perf level 1)
    len = sizeof(int); // Reset len for next call
    if (sysctlbyname("hw.perflevel1.logicalcpu_max", &e_cores, &len, NULL, 0) == 0 && e_cores >= 0) {
         e_core_ok = true;
    } else {
        e_cores = 0; // Ensure 0 on failure
    }

    // Return sum if P & E detection was successful
    if (p_core_ok && e_core_ok) {
        return p_cores + e_cores;
    }

    // Fallback 1: Total logical cores via sysctl
    int total_cores = 0;
    len = sizeof(total_cores);
     if (sysctlbyname("hw.logicalcpu_max", &total_cores, &len, NULL, 0) == 0 && total_cores > 0) {
         return total_cores;
     }

     // Fallback 2: C++ hardware_concurrency()
     unsigned int hc = std::thread::hardware_concurrency();
     if (hc > 0) {
        return hc;
     }

     // Fallback 3: Default to 1
     std::cerr << "Warning: Failed to detect core count, defaulting to 1." << std::endl;
    return 1;
}


// --- Assembly function declarations ---
extern "C" {
    // Copy test (128B NT stores)
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
    // Read test (128B loads + XOR sum) - returns dummy checksum
    uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
    // Write test (128B NT stores of zeros)
    void memory_write_loop_asm(void* dst, size_t byteCount);
    // Latency test (pointer chasing)
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
}

// --- Timer helper (using mach_absolute_time) ---
struct HighResTimer {
    uint64_t start_ticks = 0;
    mach_timebase_info_data_t timebase_info;

    HighResTimer() {
        // Get timer calibration info once.
        if (mach_timebase_info(&timebase_info) != KERN_SUCCESS) {
            perror("mach_timebase_info failed");
            exit(EXIT_FAILURE);
        }
    }
    // Start the timer
    void start() {
        start_ticks = mach_absolute_time();
    }
    // Return elapsed time in seconds.
    double stop() {
        uint64_t end = mach_absolute_time();
        uint64_t elapsed_ticks = end - start_ticks;
        double elapsed_nanos = static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
        return elapsed_nanos / 1e9;
    }
    // Return elapsed time in nanoseconds.
    double stop_ns() {
        uint64_t end = mach_absolute_time();
        uint64_t elapsed_ticks = end - start_ticks;
        return static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
    }
};


// --- Latency Test Helper Function ---
// Sets up a pseudo-random pointer chain for latency testing.
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride) {
    size_t num_pointers = buffer_size / stride;
    if (num_pointers < 2) {
        std::cerr << "Error: Buffer/stride invalid for latency chain." << std::endl;
        return;
    }
    std::vector<size_t> indices(num_pointers);
    std::iota(indices.begin(), indices.end(), 0); // 0, 1, 2...

    // Shuffle indices for random access pattern
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::cout << "Setting up pointer chain (stride " << stride << " bytes, " << num_pointers << " pointers)..." << std::endl;
    char* base_ptr = static_cast<char*>(buffer);
    // Write chain: element[i] points to element[i+1] in shuffled order
    for (size_t i = 0; i < num_pointers; ++i) {
        uintptr_t* current_loc = (uintptr_t*)(base_ptr + indices[i] * stride);
        uintptr_t next_addr = (uintptr_t)(base_ptr + indices[(i + 1) % num_pointers] * stride);
        *current_loc = next_addr; // Write pointer to next element
    }
     std::cout << "Pointer chain setup complete." << std::endl;
}


int main(int argc, char *argv[]) {
    // --- 1. Configuration ---
    size_t buffer_size = 512 * 1024 * 1024; // Common buffer size (src/dst/lat)
    int iterations = 1000;                   // Iterations for EACH R/W/Copy test
    int num_threads = get_total_logical_cores(); // Use all logical cores for BW tests

    size_t lat_stride = 128;                    // Latency test stride
    size_t lat_num_accesses = 200 * 1000 * 1000;// Latency test access count

    std::cout << "----- macOS-memory-benchmark v" << SOFTVERSION << " -----" << std::endl;
    std::cout << "Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>" << std::endl;
    std::cout << "Program is licensed under GNU GPL v3. See <https://www.gnu.org/licenses/>" << std::endl;
    std::cout << "Buffer Size: " << buffer_size / (1024.0*1024.0) << " MiB" << std::endl;
    std::cout << "Found CPU cores: " << num_threads << std::endl;
    

    // --- Test Variables ---
    void* src_buffer = MAP_FAILED;
    void* dst_buffer = MAP_FAILED;
    void* lat_buffer = MAP_FAILED;

    double total_read_time = 0.0, read_bw_gb_s = 0.0;
    double total_write_time = 0.0, write_bw_gb_s = 0.0;
    double total_copy_time = 0.0, copy_bw_gb_s = 0.0; // Copy BW = 2*N / time
    size_t total_bytes_read = 0, total_bytes_written = 0, total_bytes_copied_op = 0;

    double total_lat_time_ns = 0.0, average_latency_ns = 0.0;
    std::atomic<uint64_t> total_read_checksum = 0; // Dummy checksum for read test

    // --- 2. Memory Allocation ---
    std::cout << "\n--- Allocating Buffers ---" << std::endl;
    std::cout << "Allocating src buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    src_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::cout << "Allocating dst buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    dst_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::cout << "Allocating lat buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    lat_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (src_buffer == MAP_FAILED || dst_buffer == MAP_FAILED || lat_buffer == MAP_FAILED) {
        perror("mmap failed");
        if (src_buffer != MAP_FAILED) munmap(src_buffer, buffer_size);
        if (dst_buffer != MAP_FAILED) munmap(dst_buffer, buffer_size);
        if (lat_buffer != MAP_FAILED) munmap(lat_buffer, buffer_size);
        return EXIT_FAILURE;
    }
    std::cout << "Buffers allocated." << std::endl;

    // --- 3. Memory Initialization & Setup ---
    std::cout << "\nInitializing src/dst buffers..." << std::endl;
    char* src_init = static_cast<char*>(src_buffer);
    char* dst_init = static_cast<char*>(dst_buffer);
    size_t page_size = getpagesize();
    // Write to ensure physical allocation; src needs some data for read test
    for (size_t i = 0; i < buffer_size; i += page_size) {
        src_init[i] = (char)(i & 0xFF);
        dst_init[i] = 0;
    }
    std::cout << "Src/Dst buffers initialized." << std::endl;

    // Setup latency buffer (also touches pages)
    setup_latency_chain(lat_buffer, buffer_size, lat_stride);

    // --- 4. Warm-up Runs (Multi-threaded for BW, Single for Latency) ---
    HighResTimer timer; // Timer used for warm-up, though time isn't measured
    std::vector<std::thread> warmup_threads;
    warmup_threads.reserve(num_threads);
    std::atomic<uint64_t> dummy_checksum_warmup_atomic = 0; // Atomic checksum for warm-up

    std::cout << "\nPerforming warm-up runs (" << num_threads << " threads for bandwidth)..." << std::endl;

    // --- READ WARM-UP (Multi-threaded) ---
    std::cout << "  Read warm-up..." << std::endl;
    warmup_threads.clear();
    size_t offset = 0;
    size_t chunk_base_size = buffer_size / num_threads;
    size_t chunk_remainder = buffer_size % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* src_chunk = static_cast<char*>(src_buffer) + offset;
        // Launch read thread, accumulate checksum atomically (though not printed)
        warmup_threads.emplace_back([src_chunk, current_chunk_size, &dummy_checksum_warmup_atomic](){
            uint64_t checksum = memory_read_loop_asm(src_chunk, current_chunk_size);
            dummy_checksum_warmup_atomic.fetch_xor(checksum, std::memory_order_relaxed); // Use atomic sum
        });
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads) if (t.joinable()) t.join();

    // --- WRITE WARM-UP (Multi-threaded) ---
    std::cout << "  Write warm-up..." << std::endl;
    warmup_threads.clear();
    offset = 0; // Reset offset
    chunk_base_size = buffer_size / num_threads;
    chunk_remainder = buffer_size % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
        warmup_threads.emplace_back(memory_write_loop_asm, dst_chunk, current_chunk_size);
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads) if (t.joinable()) t.join();

    // --- COPY WARM-UP (Multi-threaded) ---
    std::cout << "  Copy warm-up..." << std::endl;
    warmup_threads.clear();
    offset = 0; // Reset offset
    chunk_base_size = buffer_size / num_threads;
    chunk_remainder = buffer_size % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* src_chunk = static_cast<char*>(src_buffer) + offset;
        char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
        warmup_threads.emplace_back(memory_copy_loop_asm, dst_chunk, src_chunk, current_chunk_size);
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads) if (t.joinable()) t.join();

    // --- LATENCY WARM-UP (Single Thread) ---
    std::cout << "  Latency warm-up (single thread)..." << std::endl;
    uintptr_t* lat_warmup_ptr = (uintptr_t*)lat_buffer;
    memory_latency_chase_asm(lat_warmup_ptr, lat_num_accesses / 100); // ~1% of accesses

    std::cout << "Warm-up complete." << std::endl;


    // --- 5. Measurements (Multi-threaded for BW) ---
    std::cout << "\n--- Starting Measurements (" << num_threads << " threads, " << iterations << " iterations each) ---" << std::endl;
    std::vector<std::thread> threads; // Use different vector for measurement threads
    threads.reserve(num_threads);

    // --- READ TEST ---
    std::cout << "Measuring Read Bandwidth..." << std::endl;
    total_read_checksum = 0; // Reset checksum
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        threads.clear();
        offset = 0; // Reset offset per iteration
        chunk_base_size = buffer_size / num_threads;
        chunk_remainder = buffer_size % num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* src_chunk = static_cast<char*>(src_buffer) + offset;
            // Launch read worker thread, accumulate checksum atomically
            threads.emplace_back([src_chunk, current_chunk_size, &total_read_checksum](){
                uint64_t checksum = memory_read_loop_asm(src_chunk, current_chunk_size);
                total_read_checksum.fetch_xor(checksum, std::memory_order_relaxed);
            });
            offset += current_chunk_size;
        }
        for (auto& t : threads) if (t.joinable()) t.join();
    }
    total_read_time = timer.stop();
    std::cout << "Read complete." << std::endl;

    // --- WRITE TEST ---
    std::cout << "Measuring Write Bandwidth..." << std::endl;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        threads.clear();
        offset = 0; // Reset offset per iteration
        chunk_base_size = buffer_size / num_threads;
        chunk_remainder = buffer_size % num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
            threads.emplace_back(memory_write_loop_asm, dst_chunk, current_chunk_size);
            offset += current_chunk_size;
        }
        for (auto& t : threads) if (t.joinable()) t.join();
    }
    total_write_time = timer.stop();
    std::cout << "Write complete." << std::endl;

    // --- COPY TEST ---
    std::cout << "Measuring Copy Bandwidth..." << std::endl;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        threads.clear();
        offset = 0; // Reset offset per iteration
        chunk_base_size = buffer_size / num_threads;
        chunk_remainder = buffer_size % num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* src_chunk = static_cast<char*>(src_buffer) + offset;
            char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
            threads.emplace_back(memory_copy_loop_asm, dst_chunk, src_chunk, current_chunk_size);
            offset += current_chunk_size;
        }
        for (auto& t : threads) if (t.joinable()) t.join();
    }
    total_copy_time = timer.stop();
    std::cout << "Copy complete." << std::endl;

    // --- LATENCY TEST ---
    // Latency test remains single-threaded
    std::cout << "Measuring Latency..." << std::endl;
    uintptr_t* lat_start_ptr = (uintptr_t*)lat_buffer;
    timer.start();
    memory_latency_chase_asm(lat_start_ptr, lat_num_accesses);
    total_lat_time_ns = timer.stop_ns();
    std::cout << "Latency complete." << std::endl;

    // --- 7. Calculate Results ---
    total_bytes_read = static_cast<size_t>(iterations) * buffer_size;        // N bytes read per iteration
    total_bytes_written = static_cast<size_t>(iterations) * buffer_size;     // N bytes written per iteration
    total_bytes_copied_op = static_cast<size_t>(iterations) * buffer_size;   // N bytes logically copied

    if (total_read_time > 0) {
        read_bw_gb_s = static_cast<double>(total_bytes_read) / total_read_time / 1e9; // BW = N / T
    }
    if (total_write_time > 0) {
        write_bw_gb_s = static_cast<double>(total_bytes_written) / total_write_time / 1e9; // BW = N / T
    }
    if (total_copy_time > 0) {
        // Copy BW reported as total bus traffic (Read N + Write N) / Time = 2N / T
        copy_bw_gb_s = static_cast<double>(total_bytes_copied_op * 2) / total_copy_time / 1e9;
    }
    if (lat_num_accesses > 0) {
        average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);
    }

    // --- 8. Print Results ---
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Buffer Size: " << buffer_size / (1024.0*1024.0) << " MiB" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;

    std::cout << "\nBandwidth Tests (multi-threaded):" << std::endl;
    std::cout << "  Read : " << read_bw_gb_s << " GB/s" << std::endl;
    std::cout << "         (Total time: " << total_read_time << " s)" << std::endl;
    std::cout << "  Write: " << write_bw_gb_s << " GB/s" << std::endl;
    std::cout << "         (Total time: " << total_write_time << " s)" << std::endl;
    std::cout << "  Copy : " << copy_bw_gb_s << " GB/s" << std::endl;
    std::cout << "         (Total time: " << total_copy_time << " s)" << std::endl;

    std::cout << "\nLatency Test (single-threaded):" << std::endl;
    std::cout << "  Total time: " << total_lat_time_ns / 1e9 << " s" << std::endl;
    std::cout << "  Total accesses: " << lat_num_accesses << std::endl;
    std::cout << "  Average latency: " << average_latency_ns << " ns" << std::endl;
    std::cout << "--------------" << std::endl;

    // --- 9. Free Memory ---
    std::cout << "\nFreeing memory..." << std::endl;
    if (src_buffer != MAP_FAILED) munmap(src_buffer, buffer_size);
    if (dst_buffer != MAP_FAILED) munmap(dst_buffer, buffer_size);
    if (lat_buffer != MAP_FAILED) munmap(lat_buffer, buffer_size);
    std::cout << "Memory freed." << std::endl;

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}