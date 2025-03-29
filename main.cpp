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

#include <iostream>
#include <vector>
#include <chrono>         // Timers (fallback)
#include <numeric>
#include <cstdlib>
#include <cstddef>
#include <random>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <functional>
#include <atomic>
#include <string>
#include <stdexcept>
#include <limits>
#include <mach/mach_error.h> // For mach_error_string
#include <iomanip>

// macOS-specific headers
#include <sys/mman.h>     // mmap / munmap
#include <unistd.h>       // getpagesize
#include <mach/mach_time.h> // mach_absolute_time (high-resolution timer)
#include <sys/sysctl.h>   // sysctlbyname (for core count)
#include <mach/mach_host.h> // Added for host_statistics64

// --- Version Information ---
#define SOFTVERSION 0.24f

// --- Function Forward Declarations ---
int get_total_logical_cores();
unsigned long get_available_memory_mb(); 
void print_usage(const char* prog_name);
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride);
extern "C" {
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
    uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
    void memory_write_loop_asm(void* dst, size_t byteCount);
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
}

// --- Timer helper struct ---
struct HighResTimer {
    uint64_t start_ticks = 0;
    mach_timebase_info_data_t timebase_info;

    HighResTimer() {
        if (mach_timebase_info(&timebase_info) != KERN_SUCCESS) {
            perror("mach_timebase_info failed");
            exit(EXIT_FAILURE);
        }
    }
    void start() {
        start_ticks = mach_absolute_time();
    }
    double stop() {
        uint64_t end = mach_absolute_time();
        uint64_t elapsed_ticks = end - start_ticks;
        double elapsed_nanos = static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
        return elapsed_nanos / 1e9;
    }
    double stop_ns() {
        uint64_t end = mach_absolute_time();
        uint64_t elapsed_ticks = end - start_ticks;
        return static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
    }
};

// Get logical Performance core count
int get_performance_cores() {
    int p_cores = 0;
    size_t len = sizeof(p_cores);
    // hw.perflevel0.logicalcpu_max should return the number of P-cores
    if (sysctlbyname("hw.perflevel0.logicalcpu_max", &p_cores, &len, NULL, 0) == 0 && p_cores > 0) {
        return p_cores;
    } else {
        // Error occurred or value is not positive.
        // perror("sysctlbyname failed for hw.perflevel0.logicalcpu_max"); // Optional detailed error
        // Return 0 if P-cores not found or key not supported.
        return 0;
    }
}

// Get logical Efficiency core count
int get_efficiency_cores() {
    int e_cores = 0;
    size_t len = sizeof(e_cores);
    // hw.perflevel1.logicalcpu_max should return the number of E-cores.
    // Check only for successful call (rc=0) and non-negative value, as 0 E-cores is valid.
    if (sysctlbyname("hw.perflevel1.logicalcpu_max", &e_cores, &len, NULL, 0) == 0 && e_cores >= 0) {
         return e_cores;
    } else {
        // Error occurred or value is negative (unlikely).
        // perror("sysctlbyname failed for hw.perflevel1.logicalcpu_max"); // Optional detailed error
        // Return 0 if E-cores not found or key not supported.
        return 0;
    }
}

// Get total logical core count (P+E) via sysctl or fallbacks.
int get_total_logical_cores() {
    int p_cores = 0;
    int e_cores = 0;
    size_t len = sizeof(int);
    bool p_core_ok = false;
    bool e_core_ok = false;
    if (sysctlbyname("hw.perflevel0.logicalcpu_max", &p_cores, &len, NULL, 0) == 0 && p_cores > 0) p_core_ok = true; else p_cores = 0;
    len = sizeof(int);
    if (sysctlbyname("hw.perflevel1.logicalcpu_max", &e_cores, &len, NULL, 0) == 0 && e_cores >= 0) e_core_ok = true; else e_cores = 0;
    if (p_core_ok && e_core_ok) return p_cores + e_cores;
    int total_cores = 0;
    len = sizeof(total_cores);
    if (sysctlbyname("hw.logicalcpu_max", &total_cores, &len, NULL, 0) == 0 && total_cores > 0) return total_cores;
    unsigned int hc = std::thread::hardware_concurrency();
    if (hc > 0) return hc;
    std::cerr << "Warning: Failed to detect core count, defaulting to 1." << std::endl;
    return 1;
}

// Identify CPU Model
std::string get_processor_name() {
    size_t len = 0;
    if (sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0) == -1) {
        perror("sysctlbyname (get size) failed for machdep.cpu.brand_string");
        return ""; 
    }

    if (len > 0) {
        std::vector<char> buffer(len);
        // Hae varsinainen nimi puskuriin
        if (sysctlbyname("machdep.cpu.brand_string", buffer.data(), &len, NULL, 0) == -1) {
            perror("sysctlbyname (get data) failed for machdep.cpu.brand_string");
            return "";
        }

        return std::string(buffer.data(), len - 1); 
    }

    return "";
}

// --- Function to query available system memory ---
// Returns estimated available memory in Megabytes (MB).
// Uses (free + inactive) pages as an approximation for available memory.
unsigned long get_available_memory_mb() {
    mach_port_t host_port = mach_host_self();
    if (host_port == MACH_PORT_NULL) {
        std::cerr << "Warning: Failed to get mach_host_self(). Cannot determine available memory." << std::endl;
        return 0;
    }

    vm_size_t page_size_local = 0; // Use local variable to avoid conflict
    kern_return_t kern_ret = host_page_size(host_port, &page_size_local);
    if (kern_ret != KERN_SUCCESS || page_size_local == 0) {
        std::cerr << "Warning: Failed to get host_page_size(): " << mach_error_string(kern_ret) << ". Cannot determine available memory." << std::endl;
        return 0;
    }

    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
    kern_ret = host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &info_count);
    if (kern_ret != KERN_SUCCESS) {
        std::cerr << "Warning: Failed to get host_statistics64(): " << mach_error_string(kern_ret) << ". Cannot determine available memory." << std::endl;
        return 0;
    }

    uint64_t available_bytes = static_cast<uint64_t>(vm_stats.free_count + vm_stats.inactive_count) * page_size_local;
    const uint64_t bytes_per_mb = 1024 * 1024;
    unsigned long available_mb = static_cast<unsigned long>(available_bytes / bytes_per_mb);
    return available_mb;
}

// --- Latency Test Helper Function ---
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride) {
    size_t num_pointers = buffer_size / stride;
    if (num_pointers < 2) {
        std::cerr << "Error: Buffer/stride invalid for latency chain setup (num_pointers=" << num_pointers << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<size_t> indices(num_pointers);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    std::cout << "Setting up pointer chain (stride " << stride << " bytes, " << num_pointers << " pointers)..." << std::endl;
    char* base_ptr = static_cast<char*>(buffer);
    for (size_t i = 0; i < num_pointers; ++i) {
        uintptr_t* current_loc = (uintptr_t*)(base_ptr + indices[i] * stride);
        uintptr_t next_addr = (uintptr_t)(base_ptr + indices[(i + 1) % num_pointers] * stride);
        *current_loc = next_addr;
    }
     std::cout << "Pointer chain setup complete." << std::endl;
}

// --- Helper function to print usage instructions ---
void print_usage(const char* prog_name) {
     std::cerr << "Usage: " << prog_name << " [options]\n"
              << "Version: " << SOFTVERSION << "\n\n"
              << "Options:\n"
              << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: 1000)\n"
              << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: 512).\n"
              << "                        The maximum allowed <size_mb> is automatically determined such that\n"
              << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
              << "  -count <count>        Number of full loops (read/write/copy/latency) (default: 1)\n"
              << "  -h, --help            Show this help message and exit\n\n"
              << "Example: " << prog_name << " -iterations 500 -buffersize 1024\n";
}


// --- Main Function ---
int main(int argc, char *argv[]) {
    // Total Timer Start
    HighResTimer total_execution_timer;
    total_execution_timer.start();
    // --- 1. Configuration (Defaults) ---
    unsigned long buffer_size_mb = 512;
    int iterations = 1000;
    std::string cpu_name = get_processor_name();
    int perf_cores = get_performance_cores();
    int eff_cores = get_efficiency_cores();
    int num_threads = get_total_logical_cores();
    size_t lat_stride = 128;
    size_t lat_num_accesses = 200 * 1000 * 1000;
    int loop_count = 1;

    // --- Get Available Memory for Limit ---
    unsigned long available_mem_mb = get_available_memory_mb();
    unsigned long max_allowed_mb_per_buffer = 0;
    const double memory_limit_factor = 0.80;
    const unsigned long fallback_total_limit_mb = 2048;
    const unsigned long minimum_limit_mb_per_buffer = 64;

    if (available_mem_mb > 0) {
        unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * memory_limit_factor);
        max_allowed_mb_per_buffer = max_total_allowed_mb / 3;

    } else {
        std::cerr
            << "Warning: Could not determine available memory automatically. "
               "Using fallback limit."
            << std::endl;
        max_allowed_mb_per_buffer = fallback_total_limit_mb / 3;
        std::cout << "Info: Setting maximum allowed size PER buffer to fallback "
                     "limit of "
                  << max_allowed_mb_per_buffer << " MB (total fallback ~"
                  << fallback_total_limit_mb << " MB)." << std::endl;
      }

    if (max_allowed_mb_per_buffer < minimum_limit_mb_per_buffer) {
        max_allowed_mb_per_buffer = minimum_limit_mb_per_buffer;
    }

        // --- Parse Command Line Arguments ---

    // Temporarily store the buffer size argument before validation against available memory
    long long requested_buffer_size_mb_ll = -1; // Use long long, init to -1 (or other sentinel)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // --- Iterations ---
        if (arg == "-iterations") {
            if (i + 1 < argc) {
                try {
                    long long val_ll = std::stoll(argv[i + 1]);
                    if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) {
                         std::cerr << "Error: Invalid value for -iterations: "
                                   << argv[i+1]
                                   << ". Must be a positive integer within int range."
                                   << std::endl;
                         print_usage(argv[0]);
                         return EXIT_FAILURE;
                    }
                    iterations = static_cast<int>(val_ll);
                    i++; // Increment i because we consumed argv[i+1]
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid number format for -iterations: " << argv[i+1] << std::endl;
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                } catch (const std::out_of_range& e) {
                     std::cerr << "Error: Value out of range for -iterations: " << argv[i+1] << std::endl;
                     print_usage(argv[0]);
                     return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Error: Missing value after -iterations" << std::endl;
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        // --- Buffersize ---
        else if (arg == "-buffersize") {
            if (i + 1 < argc) {
                 try {
                    long long val_ll = std::stoll(argv[i+1]);
                     if (val_ll <= 0) { 
                          std::cerr << "Error: Invalid value for -buffersize: "
                                   << argv[i+1]
                                   << ". Must be a positive integer." << std::endl;
                         print_usage(argv[0]);
                         return EXIT_FAILURE;
                     }
                     requested_buffer_size_mb_ll = val_ll;
                    i++; 
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid number format for -buffersize: " << argv[i+1] << std::endl;
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                } catch (const std::out_of_range& e) {
                     std::cerr << "Error: Value out of range for -buffersize: " << argv[i+1] << std::endl;
                     print_usage(argv[0]);
                     return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Error: Missing value after -buffersize" << std::endl;
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        // --- Count ---
        else if (arg == "-count") {
            if (i + 1 < argc) {
                try {
                    long long val_ll = std::stoll(argv[i + 1]);
                    if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) { 
                        std::cerr << "Error: Invalid value for -count: "
                            << argv[i + 1]
                            << ". Must be a positive integer within int range."
                            << std::endl;
                         print_usage(argv[0]); 
                        return EXIT_FAILURE;
                    }
                    loop_count = static_cast<int>(val_ll);
                    i++;
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid number format for -count: "
                        << argv[i + 1] << std::endl;
                     print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Error: Value out of range for -count: "
                        << argv[i + 1] << std::endl;
                     print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Error: Missing value after -count" << std::endl;
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else { 
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } 
    // --- Validate and Finalize Buffer Size ---
    if (requested_buffer_size_mb_ll != -1) { // If -buffersize was provided by user
        unsigned long requested_ul = static_cast<unsigned long>(requested_buffer_size_mb_ll);

        if (requested_ul > max_allowed_mb_per_buffer) {
             std::cerr << "Warning: Requested buffer size (" << requested_buffer_size_mb_ll
                      << " MB) exceeds the calculated limit (" << max_allowed_mb_per_buffer
                      << " MB based on available memory)." << std::endl;
             std::cerr << "         Using the maximum allowed size instead." << std::endl;
             buffer_size_mb = max_allowed_mb_per_buffer; // Use max allowed
        } else {
             buffer_size_mb = requested_ul; // Use user requested size
        }
    } else { // If -buffersize was not provided, check if default exceeds limit
         if (buffer_size_mb > max_allowed_mb_per_buffer) {
             std::cout << "Info: Default buffer size (" << buffer_size_mb
                       << " MB) exceeds calculated limit (" << max_allowed_mb_per_buffer
                       << " MB). Using limit instead." << std::endl;
             buffer_size_mb = max_allowed_mb_per_buffer; // Cap default if necessary
         }
         // If default is within limits, buffer_size_mb retains its default value (512)
    }
    // --- Calculate final buffer size in bytes ---
    const size_t bytes_per_mb = 1024 * 1024;
    size_t buffer_size = static_cast<size_t>(buffer_size_mb) * bytes_per_mb;

    // --- Get Page Size ONCE ---
    size_t page_size = getpagesize();

    // --- Overflow and minimum size checks ---
    if (buffer_size_mb > 0 && (buffer_size == 0 || buffer_size / bytes_per_mb != buffer_size_mb)) {
        std::cerr << "Error: Buffer size calculation resulted in zero or overflowed size_t ("
                  << buffer_size_mb << " MB requested)." << std::endl;
        return EXIT_FAILURE;
    }
    // Use page_size variable here
    if (buffer_size < page_size * 2 || buffer_size < lat_stride * 2) {
        std::cerr << "Error: Calculated buffer size (" << buffer_size << " bytes = " << buffer_size_mb
                  << " MB) is too small for operations (min page("<< page_size <<")/stride("<< lat_stride <<") requirements)." << std::endl;
        return EXIT_FAILURE;
    }

    // --- Print Configuration ---
    std::cout << "----- macOS-memory-benchmark v" << SOFTVERSION << " -----" << std::endl;
    std::cout << "Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>" << std::endl;
    std::cout << "Program is licensed under GNU GPL v3. See <https://www.gnu.org/licenses/>" << std::endl;
    std::cout << "Buffer Size (per buffer): " << buffer_size / (1024.0*1024.0) << " MiB (" << buffer_size_mb << " MB requested)" << std::endl;
    std::cout << "Total Allocation Size: ~" << 3.0 * buffer_size / (1024.0*1024.0) << " MiB (for 3 buffers)" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Loop Count: " << loop_count << std::endl; 
    if (!cpu_name.empty()) {
        std::cout << "\nProcessor Name: " << cpu_name << std::endl;
    } else {
        std::cout << "Could not retrieve processor name." << std::endl;
    }
    if (perf_cores > 0 || eff_cores > 0) {
        std::cout << "  Performance Cores: " << perf_cores << std::endl;
        std::cout << "  Efficiency Cores: " << eff_cores << std::endl;
   }
    std::cout << "  Total CPU Cores Detected: " << num_threads << std::endl;
    

    // --- Test Variables ---
    void* src_buffer = MAP_FAILED;
    void* dst_buffer = MAP_FAILED;
    void* lat_buffer = MAP_FAILED;
    double total_read_time = 0.0, read_bw_gb_s = 0.0;
    double total_write_time = 0.0, write_bw_gb_s = 0.0;
    double total_copy_time = 0.0, copy_bw_gb_s = 0.0;
    size_t total_bytes_read = 0, total_bytes_written = 0, total_bytes_copied_op = 0;
    double total_lat_time_ns = 0.0, average_latency_ns = 0.0;
    std::atomic<uint64_t> total_read_checksum = 0;

    // --- For Averages ---
    std::vector<double> all_read_bw_gb_s;
    std::vector<double> all_write_bw_gb_s;
    std::vector<double> all_copy_bw_gb_s;
    std::vector<double> all_average_latency_ns;

    if (loop_count > 0) {
        all_read_bw_gb_s.reserve(loop_count);
        all_write_bw_gb_s.reserve(loop_count);
        all_copy_bw_gb_s.reserve(loop_count);
        all_average_latency_ns.reserve(loop_count);
    }

    // --- 2. Memory Allocation ---
    std::cout << "\n--- Allocating Buffers ---" << std::endl;
    std::cout << "Allocating src buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    src_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::cout << "Allocating dst buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    dst_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::cout << "Allocating lat buffer (" << buffer_size / (1024.0*1024.0) << " MiB)..." << std::endl;
    lat_buffer = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (src_buffer == MAP_FAILED || dst_buffer == MAP_FAILED || lat_buffer == MAP_FAILED) {
        perror("Memory allocation failed");
        if (src_buffer != MAP_FAILED) munmap(src_buffer, buffer_size);
        if (dst_buffer != MAP_FAILED) munmap(dst_buffer, buffer_size);
        if (lat_buffer != MAP_FAILED) munmap(lat_buffer, buffer_size);
        return EXIT_FAILURE;
    }
    std::cout << "Buffers allocated." << std::endl;

    // --- 3. Memory Initialization & Setup ---
    std::cout << "\nInitializing src/dst buffers..." << std::endl;
    for (size_t i = 0; i < buffer_size; ++i) {
        static_cast<char*>(src_buffer)[i] = (char)(i % 256); // Example pattern
    }
    // Initialize dst_buffer with zeros (using memset)
    memset(dst_buffer, 0, buffer_size);
    std::cout << "Src/Dst buffers initialized." << std::endl;

    setup_latency_chain(lat_buffer, buffer_size, lat_stride);

    // --- 4. Warm-up Runs ---
    std::cout << "\nPerforming warm-up runs (" << num_threads << " threads for bandwidth)..." << std::endl;
    std::vector<std::thread> warmup_threads;
    warmup_threads.reserve(num_threads);
    std::atomic<uint64_t> dummy_checksum_warmup_atomic = 0;
    HighResTimer timer_warmup; // Not timed, just for consistency if needed later

    // --- READ WARM-UP ---
    std::cout << "  Read warm-up..." << std::endl;
    warmup_threads.clear();
    size_t offset = 0;
    size_t chunk_base_size = buffer_size / num_threads;
    size_t chunk_remainder = buffer_size % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* src_chunk = static_cast<char*>(src_buffer) + offset;
        warmup_threads.emplace_back([src_chunk, current_chunk_size,
                                    &dummy_checksum_warmup_atomic]() {
        uint64_t checksum = memory_read_loop_asm(src_chunk, current_chunk_size);
        dummy_checksum_warmup_atomic.fetch_xor(checksum,
                                                std::memory_order_relaxed);
        });
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads)
        if (t.joinable()) t.join();

    // --- WRITE WARM-UP ---
    std::cout << "  Write warm-up..." << std::endl;
    warmup_threads.clear();
    offset = 0;
    chunk_base_size = buffer_size / num_threads;
    chunk_remainder = buffer_size % num_threads; // Recalculate needed if changed
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
        warmup_threads.emplace_back(memory_write_loop_asm, dst_chunk,
                                    current_chunk_size);
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads)
        if (t.joinable()) t.join();

    // --- COPY WARM-UP ---
    std::cout << "  Copy warm-up..." << std::endl;
    warmup_threads.clear();
    offset = 0;
    chunk_base_size = buffer_size / num_threads;
    chunk_remainder = buffer_size % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
        if (current_chunk_size == 0) continue;
        char* src_chunk = static_cast<char*>(src_buffer) + offset;
        char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
        warmup_threads.emplace_back(memory_copy_loop_asm, dst_chunk, src_chunk,
                                    current_chunk_size);
        offset += current_chunk_size;
    }
    for (auto& t : warmup_threads)
        if (t.joinable()) t.join();

    // --- LATENCY WARM-UP ---
    std::cout << "  Latency warm-up (single thread)..." << std::endl;
    uintptr_t* lat_warmup_ptr = (uintptr_t*)lat_buffer;
    memory_latency_chase_asm(lat_warmup_ptr, lat_num_accesses / 100);
    std::cout << "Warm-up complete." << std::endl;

    // --- 5. Measurements ---
    std::cout << "\n--- Starting Measurements (" << num_threads
    << " threads, " << iterations << " iterations each, " << loop_count << " loops) ---"
    << std::endl;

    for (int loop = 0; loop < loop_count; ++loop) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        HighResTimer timer;

        // Reset per-loop variables
        total_read_time = 0.0; read_bw_gb_s = 0.0;
        total_write_time = 0.0; write_bw_gb_s = 0.0;
        total_copy_time = 0.0; copy_bw_gb_s = 0.0;
        total_lat_time_ns = 0.0; average_latency_ns = 0.0;
        total_read_checksum = 0; 
        
        // --- READ TEST ---
        std::cout << "Measuring Read Bandwidth..." << std::endl;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            threads.clear();
            offset = 0;
            chunk_base_size = buffer_size / num_threads;
            chunk_remainder = buffer_size % num_threads;
            for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size =
                chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* src_chunk = static_cast<char*>(src_buffer) + offset;
            threads.emplace_back([src_chunk, current_chunk_size,
                                    &total_read_checksum]() {
                uint64_t checksum = memory_read_loop_asm(src_chunk, current_chunk_size);
                total_read_checksum.fetch_xor(checksum, std::memory_order_relaxed);
            });
            offset += current_chunk_size;
            }
            for (auto& t : threads)
            if (t.joinable()) t.join();
        }
        total_read_time = timer.stop();
        std::cout << "Read complete." << std::endl;

        // --- WRITE TEST ---
        std::cout << "Measuring Write Bandwidth..." << std::endl;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            threads.clear();
            offset = 0;
            chunk_base_size = buffer_size / num_threads;
            chunk_remainder = buffer_size % num_threads;
            for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size =
                chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
            threads.emplace_back(memory_write_loop_asm, dst_chunk,
                                current_chunk_size);
            offset += current_chunk_size;
            }
            for (auto& t : threads)
            if (t.joinable()) t.join();
        }
        total_write_time = timer.stop();
        std::cout << "Write complete." << std::endl;

        // --- COPY TEST ---
        std::cout << "Measuring Copy Bandwidth..." << std::endl;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            threads.clear();
            offset = 0;
            chunk_base_size = buffer_size / num_threads;
            chunk_remainder = buffer_size % num_threads;
            for (int t = 0; t < num_threads; ++t) {
            size_t current_chunk_size =
                chunk_base_size + (t < chunk_remainder ? 1 : 0);
            if (current_chunk_size == 0) continue;
            char* src_chunk = static_cast<char*>(src_buffer) + offset;
            char* dst_chunk = static_cast<char*>(dst_buffer) + offset;
            threads.emplace_back(memory_copy_loop_asm, dst_chunk, src_chunk,
                                current_chunk_size);
            offset += current_chunk_size;
            }
            for (auto& t : threads)
            if (t.joinable()) t.join();
        }
        total_copy_time = timer.stop();
        std::cout << "Copy complete." << std::endl;


        // --- LATENCY TEST ---
        std::cout << "Measuring Latency (single thread)..." << std::endl; uintptr_t* lat_start_ptr = (uintptr_t*)lat_buffer; timer.start();
        memory_latency_chase_asm(lat_start_ptr, lat_num_accesses); total_lat_time_ns = timer.stop_ns();
        std::cout << "Latency complete." << std::endl;

        // --- 7. Calculate Results ---
        total_bytes_read = static_cast<size_t>(iterations) * buffer_size;
        total_bytes_written = static_cast<size_t>(iterations) * buffer_size;
        total_bytes_copied_op = static_cast<size_t>(iterations) * buffer_size;
        if (total_read_time > 0) read_bw_gb_s = static_cast<double>(total_bytes_read) / total_read_time / 1e9;
        if (total_write_time > 0) write_bw_gb_s = static_cast<double>(total_bytes_written) / total_write_time / 1e9;
        if (total_copy_time > 0) copy_bw_gb_s = static_cast<double>(total_bytes_copied_op * 2) / total_copy_time / 1e9;
        if (lat_num_accesses > 0) average_latency_ns = total_lat_time_ns / static_cast<double>(lat_num_accesses);

        // --- Store results for this loop ---
        all_read_bw_gb_s.push_back(read_bw_gb_s);
        all_write_bw_gb_s.push_back(write_bw_gb_s);
        all_copy_bw_gb_s.push_back(copy_bw_gb_s);
        all_average_latency_ns.push_back(average_latency_ns);

        // --- 8. Print Results ---
        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Buffer Size (per buffer): " << buffer_size / (1024.0*1024.0) << " MiB (" << buffer_size_mb << " MB requested)" << std::endl;
        std::cout << "  Total Allocation Size: ~" << 3.0 * buffer_size / (1024.0*1024.0) << " MiB" << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Threads (Bandwidth Tests): " << num_threads << std::endl;
        std::cout << "\nBandwidth Tests (multi-threaded):" << std::endl;
        std::cout << "  Read : " << read_bw_gb_s << " GB/s (Total time: " << total_read_time << " s)" << std::endl;
        std::cout << "  Write: " << write_bw_gb_s << " GB/s (Total time: " << total_write_time << " s)" << std::endl;
        std::cout << "  Copy : " << copy_bw_gb_s << " GB/s (Total time: " << total_copy_time << " s)" << std::endl;
        std::cout << "\nLatency Test (single-threaded, pointer chase):" << std::endl;
        std::cout << "  Total time: " << total_lat_time_ns / 1e9 << " s" << std::endl;
        std::cout << "  Total accesses: " << lat_num_accesses << std::endl;
        std::cout << "  Stride: " << lat_stride << " bytes" << std::endl;
        std::cout << "  Average latency: " << average_latency_ns << " ns" << std::endl;
        std::cout << "--------------" << std::endl;
    }

    // Averages If Loop Count > 1
    if (loop_count > 1) {
        std::cout << "\n--- Statistics Across " << loop_count << " Loops ---" << std::endl;

        // Calculate Averages
        double avg_read_bw = std::accumulate(all_read_bw_gb_s.begin(), all_read_bw_gb_s.end(), 0.0) / loop_count;
        double avg_write_bw = std::accumulate(all_write_bw_gb_s.begin(), all_write_bw_gb_s.end(), 0.0) / loop_count;
        double avg_copy_bw = std::accumulate(all_copy_bw_gb_s.begin(), all_copy_bw_gb_s.end(), 0.0) / loop_count;
        double avg_latency = std::accumulate(all_average_latency_ns.begin(), all_average_latency_ns.end(), 0.0) / loop_count;

        // Calculate Min/Max
        double min_read_bw = *std::min_element(all_read_bw_gb_s.begin(), all_read_bw_gb_s.end());
        double max_read_bw = *std::max_element(all_read_bw_gb_s.begin(), all_read_bw_gb_s.end());

        double min_write_bw = *std::min_element(all_write_bw_gb_s.begin(), all_write_bw_gb_s.end());
        double max_write_bw = *std::max_element(all_write_bw_gb_s.begin(), all_write_bw_gb_s.end());

        double min_copy_bw = *std::min_element(all_copy_bw_gb_s.begin(), all_copy_bw_gb_s.end());
        double max_copy_bw = *std::max_element(all_copy_bw_gb_s.begin(), all_copy_bw_gb_s.end());

        double min_latency = *std::min_element(all_average_latency_ns.begin(), all_average_latency_ns.end());
        double max_latency = *std::max_element(all_average_latency_ns.begin(), all_average_latency_ns.end());

        // Print Statistics
        std::cout << std::fixed << std::setprecision(3); // Precision for GB/s
        std::cout << "Read Bandwidth (GB/s):" << std::endl;
        std::cout << "  Average: " << avg_read_bw << std::endl;
        std::cout << "  Min:     " << min_read_bw << std::endl;
        std::cout << "  Max:     " << max_read_bw << std::endl;

        std::cout << "\nWrite Bandwidth (GB/s):" << std::endl;
        std::cout << "  Average: " << avg_write_bw << std::endl;
        std::cout << "  Min:     " << min_write_bw << std::endl;
        std::cout << "  Max:     " << max_write_bw << std::endl;

        std::cout << "\nCopy Bandwidth (GB/s):" << std::endl;
        std::cout << "  Average: " << avg_copy_bw << std::endl;
        std::cout << "  Min:     " << min_copy_bw << std::endl;
        std::cout << "  Max:     " << max_copy_bw << std::endl;

        std::cout << std::fixed << std::setprecision(2); // Precision for ns
        std::cout << "\nLatency (ns):" << std::endl;
        std::cout << "  Average: " << avg_latency << std::endl;
        std::cout << "  Min:     " << min_latency << std::endl;
        std::cout << "  Max:     " << max_latency << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }

    // --- 9. Free Memory and Print Total Time---
    std::cout << "\nFreeing memory..." << std::endl;
    if (src_buffer != MAP_FAILED) munmap(src_buffer, buffer_size);
    if (dst_buffer != MAP_FAILED) munmap(dst_buffer, buffer_size);
    if (lat_buffer != MAP_FAILED) munmap(lat_buffer, buffer_size);
    std::cout << "Memory freed." << std::endl;

    double total_elapsed_time_sec = total_execution_timer.stop();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Done. Total time: " << total_elapsed_time_sec << " s" << std::endl;
 
    return EXIT_SUCCESS;
}
