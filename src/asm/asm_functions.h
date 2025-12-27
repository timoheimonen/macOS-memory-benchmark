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
/**
 * @file asm_functions.h
 * @brief Assembly function declarations for optimized memory operations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides declarations for optimized assembly functions used in
 * memory benchmarks. These functions are implemented in assembly for maximum
 * performance.
 */
#ifndef ASM_FUNCTIONS_H
#define ASM_FUNCTIONS_H

#include <cstddef>  // size_t
#include <cstdint>  // uint64_t, uintptr_t

/**
 * @defgroup asm_functions Assembly Functions
 * @brief Optimized assembly functions for memory operations
 * @{
 */

extern "C" {
    /**
     * @brief Optimized memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to copy
     */
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
    
    /**
     * @brief Optimized memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to read
     * @return Checksum of read data
     */
    uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
    
    /**
     * @brief Optimized memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Number of bytes to write
     */
    void memory_write_loop_asm(void* dst, size_t byteCount);
    
    /**
     * @brief Pointer chasing loop for latency measurement (assembly)
     * @param start_pointer Starting pointer for pointer-chasing chain
     * @param count Number of pointer-chasing accesses to perform
     */
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
    
    // Pattern-specific assembly functions
    // Reverse sequential
    /**
     * @brief Optimized reverse sequential memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to read
     * @return Checksum of read data
     */
    uint64_t memory_read_reverse_loop_asm(const void* src, size_t byteCount);
    
    /**
     * @brief Optimized reverse sequential memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Number of bytes to write
     */
    void memory_write_reverse_loop_asm(void* dst, size_t byteCount);
    
    /**
     * @brief Optimized reverse sequential memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to copy
     */
    void memory_copy_reverse_loop_asm(void* dst, const void* src, size_t byteCount);
    
    // Strided access
    /**
     * @brief Optimized strided memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     * @return Checksum of read data
     */
    uint64_t memory_read_strided_loop_asm(const void* src, size_t byteCount, size_t stride);
    
    /**
     * @brief Optimized strided memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     */
    void memory_write_strided_loop_asm(void* dst, size_t byteCount, size_t stride);
    
    /**
     * @brief Optimized strided memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     */
    void memory_copy_strided_loop_asm(void* dst, const void* src, size_t byteCount, size_t stride);
    
    // Random access
    /**
     * @brief Optimized random access memory read loop (assembly)
     * @param src Source buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     * @return Checksum of read data
     */
    uint64_t memory_read_random_loop_asm(const void* src, const size_t* indices, size_t num_accesses);
    
    /**
     * @brief Optimized random access memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     */
    void memory_write_random_loop_asm(void* dst, const size_t* indices, size_t num_accesses);
    
    /**
     * @brief Optimized random access memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     */
    void memory_copy_random_loop_asm(void* dst, const void* src, const size_t* indices, size_t num_accesses);
}
/** @} */

#endif // ASM_FUNCTIONS_H

