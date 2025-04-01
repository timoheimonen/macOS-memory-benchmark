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
#include "benchmark.h"
#include <vector>
#include <numeric>  // Needed for std::iota
#include <random>   // Needed for std::random_device, std::mt19937_64
#include <algorithm> // Needed for std::shuffle
#include <cstring>   // Needed for memset
#include <iostream> // Needed for std::cout, std::cerr
#include <cstdlib>   // Needed for exit, EXIT_FAILURE

// --- Latency Test Helper ---
// Sets up a randomly shuffled pointer chain within the buffer for latency measurement.
// 'buffer': The memory area to set up the chain in.
// 'buffer_size': Total size of the 'buffer' in bytes.
// 'stride': The distance between consecutive pointer locations in bytes.
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride) {
    // Calculate how many pointers fit in the buffer with the given stride.
    size_t num_pointers = buffer_size / stride;
    // Need at least two pointers to form a chain.
    if (num_pointers < 2) {
        std::cerr << "Error: Buffer/stride invalid for latency chain setup (num_pointers="
                  << num_pointers << "). Buffer size: " << buffer_size << ", Stride: " << stride << std::endl;
        exit(EXIT_FAILURE); // Exit if parameters are invalid.
    }

    // Create a vector to store the indices of pointer locations.
    std::vector<size_t> indices(num_pointers);
    // Fill indices vector with 0, 1, 2, ..., num_pointers - 1.
    std::iota(indices.begin(), indices.end(), 0);

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937_64 g(rd());
    // Randomly shuffle the indices.
    std::shuffle(indices.begin(), indices.end(), g);

    std::cout << "Setting up pointer chain   " << std::endl; //(stride " << stride << " bytes, " << num_pointers << " pointers)..." << std::endl;
    // Get a base pointer to the buffer.
    char* base_ptr = static_cast<char*>(buffer);

    // Create the linked list by linking shuffled indices.
    for (size_t i = 0; i < num_pointers; ++i) {
        // Calculate the memory address where the current pointer value will be stored.
        uintptr_t* current_loc = (uintptr_t*)(base_ptr + indices[i] * stride);
        // Calculate the memory address of the *next* element in the shuffled sequence.
        // The modulo operator ensures the last element points back to the first.
        uintptr_t next_addr = (uintptr_t)(base_ptr + indices[(i + 1) % num_pointers] * stride);
        // Write the address of the next element into the current location.
        *current_loc = next_addr;
    }
    std::cout << "Pointer chain setup complete." << std::endl;
}

// --- Buffer Initialization ---
// Fills source and destination buffers with initial data.
// 'src_buffer': Pointer to the source buffer.
// 'dst_buffer': Pointer to the destination buffer.
// 'buffer_size': Size of each buffer in bytes.
void initialize_buffers(void* src_buffer, void* dst_buffer, size_t buffer_size) {
     std::cout << "Initializing src/dst buffers..." << std::endl;
    // Fill the source buffer with a repeating byte pattern (0-255).
    for (size_t i = 0; i < buffer_size; ++i) {
        static_cast<char*>(src_buffer)[i] = (char)(i % 256);
    }
    // Fill the destination buffer with zeros using memset.
    memset(dst_buffer, 0, buffer_size);
    std::cout << "Src/Dst buffers initialized." << std::endl;
}