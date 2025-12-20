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
// -----------------------------------------------------------------------------
// memory_read_random_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_random_loop_asm(const void* src, const size_t* indices, size_t num_accesses);
// Purpose:
//   Read memory using random access pattern defined by indices array to exercise
//   memory read bandwidth with unpredictable access pattern (worst case for prefetch).
//   Accumulates an XOR checksum across all loaded data to keep loads architecturally visible.
// Arguments:
//   x0 = src (const void*)
//   x1 = indices (const size_t*) - array of byte offsets into src
//   x2 = num_accesses (size_t) - number of accesses to perform
// Returns:
//   x0 = 64‑bit XOR checksum
// Clobbers:
//   x3‑x7, x12‑x13, q0‑q7, q16‑q31 (data + accumulators, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Accesses memory in completely random order as defined by pre-generated indices.
//   * Loads 32 bytes (one cache line) per access to test random access behavior.
//   * Distributes XOR across two accumulators (v0-v1) to reduce dependency depth.
//   * Random pattern maximizes cache misses and TLB pressure, testing worst-case performance.
// -----------------------------------------------------------------------------

.global _memory_read_random_loop_asm
.align 4
_memory_read_random_loop_asm:
    mov x3, xzr             // i = 0 (loop counter)
    mov x12, xzr            // Zero vector checksum accumulator (lower)
    mov x13, xzr            // Zero vector checksum accumulator (upper)
    
    // Zero accumulators (v0-v1) using XOR self-operation.
    // Use caller-saved registers q0-q7,q16-q31 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    eor v0.16b, v0.16b, v0.16b   // Zero accumulator 0 (caller-saved, safe)
    eor v1.16b, v1.16b, v1.16b   // Zero accumulator 1 (caller-saved, safe)

read_random_loop:              // Main random access loop
    // Loop invariants: x1=indices array base, x2=num_accesses, accumulators v0-v1 accumulate XOR.
    // Only x3 (loop counter) changes per iteration.
    
    // Check if done
    cmp x3, x2              // i >= num_accesses?
    b.ge read_random_combine_sum // If done, combine checksums
    
    // Load index: indices[i]
    // Array indexing: indices[i] = *(indices + i * sizeof(size_t))
    // lsl #3 multiplies by 8 (size of size_t)
    ldr x4, [x1, x3, lsl #3]  // x4 = indices[i] (multiply by 8 for size_t)
    
    // Calculate address: src + indices[i]
    add x5, x0, x4          // addr = src + indices[i] (random memory address)
    
    // Load 32 bytes at random offset
    // Using caller-saved registers: q2-q3 (avoiding q8-q15 per AAPCS64)
    ldp q2, q3, [x5]        // Load pair (32B total) from random address
    
    // Accumulate into checksum
    // Distribute XOR across 2 accumulators to reduce dependency depth
    eor v0.16b, v0.16b, v2.16b    // Accumulate q2 into v0
    eor v1.16b, v1.16b, v3.16b    // Accumulate q3 into v1
    
    // Increment loop counter
    add x3, x3, #1          // i++
    b read_random_loop      // Loop again

read_random_combine_sum:         // Final reduction + result write-back
    // Combine checksums
    // Final reduction: combine v0 and v1 into v0, then extract to x0
    eor v0.16b, v0.16b, v1.16b   // Combine v0 and v1 into v0
    umov x12, v0.d[0]            // Extract lower 64 bits of combined vector sum
    umov x13, v0.d[1]            // Extract upper 64 bits of combined vector sum
    eor x0, x12, x13             // Combine both halves into final checksum
    
    ret                         // Return checksum in x0

