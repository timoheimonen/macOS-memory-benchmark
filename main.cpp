// Version 0.13
// by Timo Heimonen <timo.heimonen@gmail.com>

#include <iostream>       // cout
#include <vector>         // std::vector (for latency setup)
#include <chrono>         // Timers (fallback)
#include <numeric>        // std::iota (for latency setup)
#include <cstdlib>        // exit, EXIT_FAILURE
#include <cstddef>        // size_t
#include <random>         // std::shuffle, std::random_device, std::mt19937_64
#include <algorithm>      // std::shuffle
#include <cstdint>        // uintptr_t

// macOS-specific headers
#include <sys/mman.h>     // mmap / munmap
#include <unistd.h>       // getpagesize
#include <mach/mach_time.h> // mach_absolute_time (high-resolution timer)

// --- Version Information ---
#define SOFTVERSION 0.13f

// --- Assembly function declarations ---
// Use extern "C" to prevent C++ name mangling.
extern "C" {
    // Bandwidth test: Copy 'byteCount' bytes from 'src' to 'dst'.
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);

    // Latency test: Chase 'count' pointers starting from '*start_pointer'.
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
}

// --- Timer helper (using mach_absolute_time) ---
struct HighResTimer {
    uint64_t start_ticks = 0;
    mach_timebase_info_data_t timebase_info;

    HighResTimer() {
        // Get timer calibration info once.
        if (mach_timebase_info(&timebase_info) != KERN_SUCCESS) {
            std::cerr << "Error: mach_timebase_info failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void start() {
        start_ticks = mach_absolute_time();
    }

    // Returns elapsed time in seconds.
    double stop() {
        uint64_t end_ticks = mach_absolute_time();
        uint64_t elapsed_ticks = end_ticks - start_ticks;
        double elapsed_nanos = static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
        return elapsed_nanos / 1e9;
    }

    // Returns elapsed time directly in nanoseconds.
    double stop_ns() {
         uint64_t end_ticks = mach_absolute_time();
         uint64_t elapsed_ticks = end_ticks - start_ticks;
         return static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
    }
};

// --- Latency Test Helper Function ---
// Sets up a pseudo-random pointer chain within the buffer for latency testing.
// Each element at `buffer + indices[i] * stride` will contain the address
// of the element at `buffer + indices[i+1] * stride`.
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride) {
    size_t num_pointers = buffer_size / stride;
    if (num_pointers < 2) {
        // Need at least two pointers to form a chain.
        std::cerr << "Error: Buffer too small or stride too large for latency test chain." << std::endl;
        return; // Or handle error more robustly
    }

    std::vector<size_t> indices(num_pointers);
    std::iota(indices.begin(), indices.end(), 0); // 0, 1, 2, ...

    // Shuffle indices to create a random access pattern.
    std::random_device rd;
    std::mt19937_64 g(rd()); // Use a 64-bit Mersenne Twister for good randomness.
    std::shuffle(indices.begin(), indices.end(), g);

    std::cout << "Setting up pointer chain (stride " << stride << " bytes, " << num_pointers << " pointers)..." << std::endl;
    char* base_ptr = static_cast<char*>(buffer);

    // Write the pointer chain into the buffer.
    for (size_t i = 0; i < num_pointers; ++i) {
        size_t current_index = indices[i];
        size_t next_index = indices[(i + 1) % num_pointers]; // Wrap around for the last element.

        // Location where the next pointer address will be stored.
        uintptr_t* current_addr_location = (uintptr_t*)(base_ptr + current_index * stride);
        // The actual address of the next element in the chain.
        uintptr_t next_element_addr_value = (uintptr_t)(base_ptr + next_index * stride);

        // Write the next element's address into the current element's location.
        *current_addr_location = next_element_addr_value;
    }
     std::cout << "Pointer chain setup complete." << std::endl;
}


int main(int argc, char *argv[]) {
    // --- 1. Configuration ---
    // Bandwidth test config
    size_t bw_buffer_size = 512 * 1024 * 1024; // 512 MiB
    int bw_iterations = 100;                   // Number of copy operations per measurement.

    // Latency test config
    size_t lat_buffer_size = 512 * 1024 * 1024; // 512 MiB, should exceed L3 cache size.
    size_t lat_stride = 128;                    // Stride between pointers. Try cache line size (e.g., 64, 128).
    size_t lat_num_accesses = 200 * 1000 * 1000; // Number of dependent loads (pointer chases).

    std::cout << "--- macOS-memory-benchmark v" << SOFTVERSION << " ---" << std::endl;
    std::cout << "by Timo Heimonen <timo.heimonen@gmail.com>" << std::endl;
    

    // --- Test Variables ---
    void* src_buffer = MAP_FAILED;
    void* dst_buffer = MAP_FAILED;
    void* lat_buffer = MAP_FAILED;

    double total_bw_time = 0.0;
    size_t total_bytes_copied = 0;
    double bandwidth_gb_per_sec = 0.0;

    double total_lat_time_ns = 0.0;
    double average_latency_ns = 0.0;

    // --- 2. Memory Allocation ---
    // Allocate separate buffers for bandwidth and latency tests.
    std::cout << "\n--- Bandwidth Test Setup ---" << std::endl;
    std::cout << "Buffer size: " << bw_buffer_size / (1024.0 * 1024.0) << " MiB" << std::endl;
    std::cout << "Iterations: " << bw_iterations << std::endl;
    std::cout << "Allocating bandwidth buffers (" << bw_buffer_size * 2 / (1024.0 * 1024.0) << " MiB total)..." << std::endl;
    src_buffer = mmap(nullptr, bw_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    dst_buffer = mmap(nullptr, bw_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (src_buffer == MAP_FAILED || dst_buffer == MAP_FAILED) {
        perror("mmap failed for bandwidth buffers");
        if (src_buffer != MAP_FAILED) munmap(src_buffer, bw_buffer_size);
        // dst_buffer cleanup not needed if it failed allocation
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
        // Clean up previously allocated buffers before exiting.
        munmap(src_buffer, bw_buffer_size);
        munmap(dst_buffer, bw_buffer_size);
        return EXIT_FAILURE;
    }
    std::cout << "Latency buffer allocated." << std::endl;


    // --- 3. Memory Initialization & Setup ---
    // Initialize bandwidth buffers to ensure pages are mapped.
    std::cout << "\nInitializing bandwidth buffers..." << std::endl;
    char* src_ptr = static_cast<char*>(src_buffer);
    char* dst_ptr = static_cast<char*>(dst_buffer);
    size_t page_size = getpagesize();
    for (size_t i = 0; i < bw_buffer_size; i += page_size) {
        src_ptr[i] = (char)(i & 0xFF); // Touch each page.
        dst_ptr[i] = 0;
    }
    std::cout << "Bandwidth buffers initialized." << std::endl;

    // Setup latency buffer pointer chain. This also ensures pages are mapped.
    setup_latency_chain(lat_buffer, lat_buffer_size, lat_stride);


    // --- 4. Warm-up Runs ---
    // Perform short runs to allow CPU frequency scaling, cache/TLB warm-up etc.
    HighResTimer timer; // Reuse timer for both tests.

    // Bandwidth warm-up
    std::cout << "\nPerforming bandwidth warm-up run..." << std::endl;
    memory_copy_loop_asm(dst_buffer, src_buffer, bw_buffer_size);
    std::cout << "Bandwidth warm-up complete." << std::endl;

    // Latency warm-up
    std::cout << "Performing latency warm-up run..." << std::endl;
    uintptr_t* start_ptr_addr_lat = (uintptr_t*)lat_buffer; // Start chasing from the beginning of the buffer.
    memory_latency_chase_asm(start_ptr_addr_lat, lat_num_accesses / 100); // Use ~1% of accesses for warm-up.
    std::cout << "Latency warm-up complete." << std::endl;


    // --- 5. Bandwidth Measurement ---
    std::cout << "\nStarting bandwidth measurement..." << std::endl;
    timer.start();

    for (int i = 0; i < bw_iterations; ++i) {
        memory_copy_loop_asm(dst_buffer, src_buffer, bw_buffer_size);
        // Optionally swap src/dst for potentially different cache effects:
        // std::swap(src_buffer, dst_buffer);
    }

    total_bw_time = timer.stop();
    std::cout << "Bandwidth measurement complete." << std::endl;

    // --- 6. Latency Measurement ---
    std::cout << "\nStarting latency measurement..." << std::endl;
    start_ptr_addr_lat = (uintptr_t*)lat_buffer; // Reset starting pointer.
    timer.start();

    memory_latency_chase_asm(start_ptr_addr_lat, lat_num_accesses);

    total_lat_time_ns = timer.stop_ns(); // Get time directly in nanoseconds.
    std::cout << "Latency measurement complete." << std::endl;


    // --- 7. Calculate Results ---
    // Bandwidth: Copy involves read + write, so x2 bytes transferred.
    total_bytes_copied = static_cast<size_t>(bw_iterations) * 2 * bw_buffer_size;
    if (total_bw_time > 0) {
        bandwidth_gb_per_sec = static_cast<double>(total_bytes_copied) / total_bw_time / 1e9; // GB/s (10^9 bytes/s)
    }

    // Latency: Average time per access.
    if (lat_num_accesses > 0) {
        average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);
    }

    // --- 8. Print Results ---
    std::cout << "\n--- Results ---" << std::endl;

    std::cout << "Bandwidth Test:" << std::endl;
    std::cout << "  Total time: " << total_bw_time << " s" << std::endl;
    std::cout << "  Data copied: " << static_cast<double>(total_bytes_copied) / 1e9 << " GB" << std::endl;
    // Use GiB/s (2^30 bytes/s) if preferred:
    // std::cout << "  Memory bandwidth (copy): " << static_cast<double>(total_bytes_copied) / total_bw_time / (1024.0*1024.0*1024.0) << " GiB/s" << std::endl;
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