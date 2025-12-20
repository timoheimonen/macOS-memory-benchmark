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
// memory_copy_random_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_random_loop_asm(void* dst, const void* src, const size_t* indices, size_t num_accesses);
// Purpose:
//   Copy memory using random access pattern defined by indices array to exercise
//   memory copy bandwidth with unpredictable access pattern (worst case for prefetch).
//   Uses wide (128‑bit) vector loads and pair stores, processing 32 bytes per iteration.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = indices (const size_t*) - array of byte offsets
//   x3 = num_accesses (size_t) - number of accesses to perform
// Returns:
//   (none)
// Clobbers:
//   x4‑x8 (temporaries), q0‑q1 (data vectors, avoiding q8‑q15 per AAPCS64)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
// Implementation Notes:
//   * Accesses memory in completely random order as defined by pre-generated indices.
//   * Copies 32 bytes (one cache line) per access to test random copy behavior.
//   * Random pattern maximizes cache misses and TLB pressure, testing worst-case performance.
// -----------------------------------------------------------------------------

.global _memory_copy_random_loop_asm
.align 4
_memory_copy_random_loop_asm:
    mov x4, xzr             // i = 0 (loop counter)

copy_random_loop:              // Main random access loop
    // Loop invariants: x2=indices array base, x3=num_accesses, x0/x1 are base pointers.
    // Only x4 (loop counter) changes per iteration.
    
    // Check if done
    cmp x4, x3              // i >= num_accesses?
    b.ge copy_random_end    // If done, exit
    
    // Load index: indices[i]
    // Array indexing: indices[i] = *(indices + i * sizeof(size_t))
    // lsl #3 multiplies by 8 (size of size_t)
    ldr x5, [x2, x4, lsl #3]  // x5 = indices[i] (multiply by 8 for size_t)
    
    // Calculate addresses: src + indices[i], dst + indices[i]
    add x6, x1, x5          // src_addr = src + indices[i] (random source address)
    add x7, x0, x5          // dst_addr = dst + indices[i] (random destination address)
    
    // Load 32 bytes from source
    // Use caller-saved registers q0-q1 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    ldp q0, q1, [x6]        // Load pair (32B total) from random source address
    
    // Store to destination
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0, q1, [x7]       // Store pair (32B total, non-temporal) to random destination
    
    // Increment loop counter
    add x4, x4, #1          // i++
    b copy_random_loop      // Loop again

copy_random_end:              // Return to caller
    ret                     // Return

