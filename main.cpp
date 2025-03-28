// Version 0.14
// by Timo Heimonen <timo.heimonen@gmail.com>

#include <iostream>       // cout
#include <vector>         // std::vector
#include <chrono>         // Timers (fallback)
#include <numeric>        // std::iota, std::accumulate
#include <cstdlib>        // exit, EXIT_FAILURE
#include <cstddef>        // size_t
#include <random>         // std::shuffle, std::random_device, std::mt19937_64
#include <algorithm>      // std::shuffle, std::min
#include <cstdint>        // uintptr_t
#include <thread>         // std::thread, std::thread::hardware_concurrency
#include <functional>     // std::ref

// macOS-specific headers
#include <sys/mman.h>     // mmap / munmap
#include <unistd.h>       // getpagesize
#include <mach/mach_time.h> // mach_absolute_time (high-resolution timer)
#include <sys/sysctl.h>   // sysctlbyname (for core count)

// --- Version Information ---
#define SOFTVERSION 0.14f

// --- Helper Function to Get Performance Core Count ---
// Returns P-core count, falls back to total logical cores, then to 1.
int get_p_core_count() {
    int cores = 0;
    size_t len = sizeof(cores);
    // Try hw.perflevel0.logicalcpu_max first (P-cores)
    if (sysctlbyname("hw.perflevel0.logicalcpu_max", &cores, &len, NULL, 0) == 0 && cores > 0) {
        return cores;
    }
    // Fallback: hw.logicalcpu_max (Total logical cores P+E)
    len = sizeof(cores);
    if (sysctlbyname("hw.logicalcpu_max", &cores, &len, NULL, 0) == 0 && cores > 0) {
        return cores;
    }
    return 1; // Default to 1 on error
}


// --- Assembly function declarations ---
// Use extern "C" to prevent C++ name mangling.
extern "C" {
    // Optimized Bandwidth test (e.g., 128B non-temporal copy)
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);

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
            perror("mach_timebase_info failed"); // Use perror for system errors
            exit(EXIT_FAILURE);
        }
    }
    void start() { start_ticks = mach_absolute_time(); }
    // Return elapsed time in seconds.
    double stop() {
        uint64_t end = mach_absolute_time();
        return static_cast<double>(end - start_ticks) * timebase_info.numer / timebase_info.denom / 1e9;
    }
    // Return elapsed time in nanoseconds.
    double stop_ns() {
        uint64_t end = mach_absolute_time();
        return static_cast<double>(end - start_ticks) * timebase_info.numer / timebase_info.denom;
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
    size_t bw_buffer_size = 512 * 1024 * 1024; // 512 MiB (src/dst) - large to exceed L3 cache
    int bw_iterations = 200;                   // Number of full buffer copies per measurement
    int num_threads = get_p_core_count();      // Threads for bandwidth test (P-cores or total)

    size_t lat_buffer_size = 512 * 1024 * 1024; // 512 MiB - large to exceed L3 cache
    size_t lat_stride = 128;                    // Stride - match cache line size (e.g., 128B on Apple Silicon)
    size_t lat_num_accesses = 200 * 1000 * 1000;// Number of dependent loads for latency average

    std::cout << "--- macOS-memory-benchmark v" << SOFTVERSION << " ---" << std::endl;
    std::cout << "by Timo Heimonen <timo.heimonen@gmail.com>" << std::endl;

    // --- Test Variables ---
    void* src_buffer = MAP_FAILED;
    void* dst_buffer = MAP_FAILED;
    void* lat_buffer = MAP_FAILED;
    double total_bw_time = 0.0, bandwidth_gb_per_sec = 0.0;
    double total_lat_time_ns = 0.0, average_latency_ns = 0.0;
    size_t total_bytes_copied = 0;

    // --- 2. Memory Allocation ---
    std::cout << "\n--- Bandwidth Test Setup ---" << std::endl;
    std::cout << "Buffer size: " << bw_buffer_size / (1024.0 * 1024.0) << " MiB" << std::endl;
    std::cout << "Iterations: " << bw_iterations << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Allocating bandwidth buffers (" << bw_buffer_size * 2 / (1024.0 * 1024.0) << " MiB total)..." << std::endl;
    src_buffer = mmap(nullptr, bw_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    dst_buffer = mmap(nullptr, bw_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (src_buffer == MAP_FAILED || dst_buffer == MAP_FAILED) {
        perror("mmap failed for bandwidth buffers");
        // Simple cleanup on failure
        if (src_buffer != MAP_FAILED) munmap(src_buffer, bw_buffer_size);
        return EXIT_FAILURE;
    }
    std::cout << "Bandwidth buffers allocated." << std::endl;

    std::cout << "\n--- Latency Test Setup ---" << std::endl;
    std::cout << "Buffer size: " << lat_buffer_size / (1024.0 * 1024.0) << " MiB" << std::endl;
    std::cout << "Stride: " << lat_stride << " bytes" << std::endl;
    std::cout << "Accesses: " << lat_num_accesses << std::endl;
    std::cout << "Allocating latency buffer (" << lat_buffer_size / (1024.0 * 1024.0) << " MiB)..." << std::endl;
    lat_buffer = mmap(nullptr, lat_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
     if (lat_buffer == MAP_FAILED) {
        perror("mmap failed for latency buffer");
        // Cleanup all allocated buffers
        if (src_buffer != MAP_FAILED) munmap(src_buffer, bw_buffer_size);
        if (dst_buffer != MAP_FAILED) munmap(dst_buffer, bw_buffer_size);
        return EXIT_FAILURE;
     }
    std::cout << "Latency buffer allocated." << std::endl;

    // --- 3. Memory Initialization & Setup ---
    std::cout << "\nInitializing bandwidth buffers (touching pages)..." << std::endl;
    char* src_init = static_cast<char*>(src_buffer);
    char* dst_init = static_cast<char*>(dst_buffer);
    size_t page_size = getpagesize();
    // Write to first byte of each page to ensure physical allocation
    for (size_t i = 0; i < bw_buffer_size; i += page_size) {
        src_init[i] = (char)(i & 0xFF); // Write some data
        dst_init[i] = 0;
    }
    std::cout << "Bandwidth buffers initialized." << std::endl;

    // Setup latency buffer (also touches pages)
    setup_latency_chain(lat_buffer, lat_buffer_size, lat_stride);

    // --- 4. Warm-up Runs ---
    HighResTimer timer;
    // Single-threaded warm-up is simpler and usually sufficient
    std::cout << "\nPerforming bandwidth warm-up run (single thread)..." << std::endl;
    memory_copy_loop_asm(dst_buffer, src_buffer, bw_buffer_size);
    std::cout << "Bandwidth warm-up complete." << std::endl;

    std::cout << "Performing latency warm-up run..." << std::endl;
    uintptr_t* lat_warmup_ptr = (uintptr_t*)lat_buffer;
    memory_latency_chase_asm(lat_warmup_ptr, lat_num_accesses / 100); // ~1% of accesses
    std::cout << "Latency warm-up complete." << std::endl;

    // --- 5. Bandwidth Measurement (Multi-threaded) ---
    std::cout << "\nStarting multi-threaded bandwidth measurement (" << num_threads << " threads)..." << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    timer.start(); // Start timer before loop

    for (int i = 0; i < bw_iterations; ++i) {
        threads.clear();
        size_t offset = 0;
        size_t chunk_base_size = bw_buffer_size / num_threads;
        size_t chunk_remainder = bw_buffer_size % num_threads;

        for (int t = 0; t < num_threads; ++t) {
            // Distribute remainder across first 'chunk_remainder' threads
            size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue; // Safety check

            char* src_chunk = static_cast<char*>(src_buffer) + offset;
            char* dst_chunk = static_cast<char*>(dst_buffer) + offset;

            // Launch thread for its chunk
            threads.emplace_back(memory_copy_loop_asm, dst_chunk, src_chunk, current_chunk_size);

            offset += current_chunk_size;
        }

        // Wait for all threads in this iteration
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        // Optional: Swap src/dst pointers here for next iteration if needed
        // std::swap(src_buffer, dst_buffer);
    }

    total_bw_time = timer.stop(); // Stop timer after loop
    std::cout << "Bandwidth measurement complete." << std::endl;

    // --- 6. Latency Measurement (Single-threaded) ---
    // Latency test inherently benefits less from parallelization
    std::cout << "\nStarting latency measurement..." << std::endl;
    uintptr_t* lat_start_ptr = (uintptr_t*)lat_buffer;
    timer.start();
    memory_latency_chase_asm(lat_start_ptr, lat_num_accesses);
    total_lat_time_ns = timer.stop_ns();
    std::cout << "Latency measurement complete." << std::endl;

    // --- 7. Calculate Results ---
    // Bandwidth: Copy is read + write, factor of 2
    total_bytes_copied = static_cast<size_t>(bw_iterations) * 2 * bw_buffer_size;
    if (total_bw_time > 0) {
        bandwidth_gb_per_sec = static_cast<double>(total_bytes_copied) / total_bw_time / 1e9; // GB/s (10^9)
    }

    // Latency: Average per access
    if (lat_num_accesses > 0) {
        average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);
    }

    // --- 8. Print Results ---
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Bandwidth Test (" << num_threads << " threads):" << std::endl;
    std::cout << "  Total time: " << total_bw_time << " s" << std::endl;
    std::cout << "  Data copied: " << static_cast<double>(total_bytes_copied) / 1e9 << " GB" << std::endl;
    // GiB/s = GB/s * 1e9 / (1024^3)
    // std::cout << "  Memory bandwidth (copy): " << bandwidth_gb_per_sec * 1e9 / (1024.0*1024.0*1024.0) << " GiB/s" << std::endl;
    std::cout << "  Memory bandwidth (copy): " << bandwidth_gb_per_sec << " GB/s" << std::endl;

    std::cout << "\nLatency Test:" << std::endl;
    std::cout << "  Total time: " << total_lat_time_ns / 1e9 << " s" << std::endl;
    std::cout << "  Total accesses: " << lat_num_accesses << std::endl;
    std::cout << "  Average latency: " << average_latency_ns << " ns" << std::endl;
    std::cout << "--------------" << std::endl;

    // --- 9. Free Memory ---
    std::cout << "\nFreeing memory..." << std::endl;
    if (src_buffer != MAP_FAILED) munmap(src_buffer, bw_buffer_size);
    if (dst_buffer != MAP_FAILED) munmap(dst_buffer, bw_buffer_size);
    if (lat_buffer != MAP_FAILED) munmap(lat_buffer, lat_buffer_size);
    std::cout << "Memory freed." << std::endl;

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}