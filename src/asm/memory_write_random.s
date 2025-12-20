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
// memory_write_random_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_random_loop_asm(void* dst, const size_t* indices, size_t num_accesses);
// Purpose:
//   Write memory using random access pattern defined by indices array to exercise
//   memory write bandwidth with unpredictable access pattern (worst case for prefetch).
//   Uses STNP as a non‑temporal hint.
// Arguments:
//   x0 = dst (void*)
//   x1 = indices (const size_t*) - array of byte offsets into dst
//   x2 = num_accesses (size_t) - number of accesses to perform
// Returns:
//   (none)
// Clobbers:
//   x3‑x6, q0‑q1 (zero vectors, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Accesses memory in completely random order as defined by pre-generated indices.
//   * Stores 32 bytes (one cache line) per access to test random write behavior.
//   * Zero vectors are materialized once (movi) then reused.
//   * Random pattern maximizes cache misses and TLB pressure, testing worst-case performance.
// -----------------------------------------------------------------------------

.global _memory_write_random_loop_asm
.align 4
_memory_write_random_loop_asm:
    mov x3, xzr             // i = 0 (loop counter)
    
    // Zero out data registers (using only caller-saved: v0-v1, avoiding v8-v15 per AAPCS64).
    // Use caller-saved registers q0-q1 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    // Zero vectors are materialized once (movi) then reused throughout the loop.
    movi v0.16b, #0             // Zero vector 0
    movi v1.16b, #0             // Zero vector 1

write_random_loop:              // Main random access loop
    // Loop invariants: x1=indices array base, x2=num_accesses, vectors v0-v1 are zero.
    // Only x3 (loop counter) changes per iteration.
    
    // Check if done
    cmp x3, x2              // i >= num_accesses?
    b.ge write_random_end   // If done, exit
    
    // Load index: indices[i]
    // Array indexing: indices[i] = *(indices + i * sizeof(size_t))
    // lsl #3 multiplies by 8 (size of size_t)
    ldr x4, [x1, x3, lsl #3]  // x4 = indices[i] (multiply by 8 for size_t)
    
    // Calculate address: dst + indices[i]
    add x5, x0, x4          // addr = dst + indices[i] (random memory address)
    
    // Store 32 bytes at random offset
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0, q1, [x5]       // Store pair (32B total, non-temporal) to random address
    
    // Increment loop counter
    add x3, x3, #1          // i++
    b write_random_loop      // Loop again

write_random_end:             // Return to caller
    ret                     // Return

