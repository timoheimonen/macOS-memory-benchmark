#include <iostream>       // For printing (cout)
#include <vector>         // Easy way to manage smaller data sets (maybe not for main buffers)
#include <chrono>         // Alternative timing, but mach_absolute_time is more precise on macOS
#include <numeric>        // std::iota, if needed for buffer initialization etc.
#include <cstdlib>        // exit, EXIT_FAILURE
#include <cstddef>        // size_t

// macOS-specific headers
#include <sys/mman.h>     // mmap / munmap for memory allocation
#include <unistd.h>       // sysconf, getpagesize
#include <mach/mach_time.h> // mach_absolute_time for high-resolution timer

// --- Assembly function declarations ---
// Use extern "C" to prevent C++ name mangling,
// so the linker can find the symbols defined in assembly code.
extern "C" {
    // Example: Copy 'byteCount' bytes from 'src' to 'dst'
    // Return value can be void or, for instance, clock cycles used if asm measures itself.
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);

    // You can declare other similar functions for different operations (e.g., write, read)
    // void memory_write_loop_asm(void* dst, size_t byteCount, uint64_t pattern);
    // uint64_t memory_read_loop_asm(const void* src, size_t byteCount); // Returns e.g., sum of read values
}

// --- Timer helper (using mach_absolute_time) ---
struct HighResTimer {
    uint64_t start_ticks = 0;
    mach_timebase_info_data_t timebase_info;

    HighResTimer() {
        // Get the conversion factor between ticks and nanoseconds once.
        if (mach_timebase_info(&timebase_info) != KERN_SUCCESS) {
            std::cerr << "Error: mach_timebase_info failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void start() {
        start_ticks = mach_absolute_time(); // Read the clock tick value
    }

    // Returns elapsed time in seconds
    double stop() {
        uint64_t end_ticks = mach_absolute_time();
        uint64_t elapsed_ticks = end_ticks - start_ticks;

        // Convert ticks to nanoseconds using timebase_info
        double elapsed_nanos = static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;

        // Convert nanoseconds to seconds
        return elapsed_nanos / 1e9;
    }
};


int main(int argc, char *argv[]) {
    // --- 1. Configuration ---
    // Define buffer size (much larger than caches, e.g., 256MB+)
    // and the number of iterations. These could also be read from command line (argc, argv).
    size_t buffer_size = 512 * 1024 * 1024; // 512 MiB
    int iterations = 100; // How many times the core loop is run per measurement

    std::cout << "--- Memory Speed Test ---" << std::endl;
    std::cout << "Buffer size: " << buffer_size / (1024.0 * 1024.0) << " MiB" << std::endl;
    std::cout << "Iterations / measurement: " << iterations << std::endl;

    // --- 2. Memory Allocation (mmap) ---
    // mmap is often better for large allocations than new/malloc.
    // Request private, anonymous memory (RAM).
    std::cout << "Allocating memory (" << buffer_size * 2 / (1024.0 * 1024.0) << " MiB)..." << std::endl;
    void* src_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void* dst_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (src_buffer == MAP_FAILED || dst_buffer == MAP_FAILED) {
        perror("mmap failed");
        // Clean up if one succeeded
        if (src_buffer != MAP_FAILED) munmap(src_buffer, buffer_size);
        if (dst_buffer != MAP_FAILED) munmap(dst_buffer, buffer_size);
        return EXIT_FAILURE;
    }
    std::cout << "Memory allocated." << std::endl;

    // --- 3. Memory Initialization ("warming up") ---
    // IMPORTANT! The OS might not allocate physical memory pages
    // until they are first written to (demand paging).
    // Initializing ensures the memory is actually mapped and ready.
    std::cout << "Initializing memory (touching every page)..." << std::endl;
    char* src_ptr = static_cast<char*>(src_buffer);
    char* dst_ptr = static_cast<char*>(dst_buffer);
    size_t page_size = getpagesize(); // Get the system's page size
    for (size_t i = 0; i < buffer_size; i += page_size) {
        src_ptr[i] = (char)(i & 0xFF); // Write something to each page
        dst_ptr[i] = 0;
    }
    std::cout << "Memory initialized." << std::endl;

    // --- 4. Warm-up Run ---
    // Run the test once without timing to allow the CPU
    // to potentially ramp up clock speeds and fill caches/TLBs.
    std::cout << "Performing warm-up run..." << std::endl;
    memory_copy_loop_asm(dst_buffer, src_buffer, buffer_size);
    std::cout << "Warm-up complete." << std::endl;

    // --- 5. Timing and Measurement Loop ---
    HighResTimer timer;
    double total_time = 0.0;
    size_t total_bytes_copied = 0;

    std::cout << "Starting measurement..." << std::endl;
    timer.start(); // Start the timer

    for (int i = 0; i < iterations; ++i) {
        // Call the assembly function to perform the core operation
        memory_copy_loop_asm(dst_buffer, src_buffer, buffer_size);
        // Optionally, swap src/dst buffers each iteration
        // std::swap(src_buffer, dst_buffer);
    }

    total_time = timer.stop(); // Stop the timer
    std::cout << "Measurement complete." << std::endl;

    // --- 6. Calculate Results ---
    // For a copy operation, data is both read and written,
    // so the total transferred bytes = 2 * buffer size * iterations.
    total_bytes_copied = static_cast<size_t>(iterations) * 2 * buffer_size;
    double bandwidth_gb_per_sec = 0.0;
    if (total_time > 0) {
        // Calculate bandwidth in GB/s (Gigabytes / second - note: Giga = 10^9)
         bandwidth_gb_per_sec = static_cast<double>(total_bytes_copied) / total_time / 1e9;
        // Or GiB/s (Gibibytes / second - note: Gibi = 2^30)
        // bandwidth_gb_per_sec = static_cast<double>(total_bytes_copied) / total_time / (1024.0 * 1024.0 * 1024.0);

    }

    // --- 7. Print Results ---
    std::cout << "--- Results ---" << std::endl;
    std::cout << "Total time: " << total_time << " s" << std::endl;
    std::cout << "Data transferred: " << static_cast<double>(total_bytes_copied) / 1e9 << " GB" << std::endl;
    // Or GiB:
    // std::cout << "Data transferred: " << static_cast<double>(total_bytes_copied) / (1024.0*1024.0*1024.0) << " GiB" << std::endl;
    std::cout << "Memory bandwidth (copy): " << bandwidth_gb_per_sec << " GB/s" << std::endl; // Or GiB/s
    std::cout << "--------------" << std::endl;


    // --- 8. Free Memory ---
    std::cout << "Freeing memory..." << std::endl;
    munmap(src_buffer, buffer_size);
    munmap(dst_buffer, buffer_size);
    std::cout << "Done." << std::endl;

    return EXIT_SUCCESS;
}