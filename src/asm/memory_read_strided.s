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
// memory_read_strided_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_strided_loop_asm(const void* src, size_t byteCount, size_t stride, size_t num_iterations);
// Purpose:
//   Read 'byteCount' bytes from 'src' using strided access pattern to exercise
//   memory read bandwidth with non-sequential access (e.g., cache line or page stride).
//   Accesses memory at offsets: 0, stride, 2*stride, 3*stride, ... wrapping around.
//   Accumulates an XOR checksum across all loaded data to keep loads architecturally visible.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t) - for modulo wrapping only
//   x2 = stride (size_t) - stride in bytes (e.g., 64 for cache line, 4096 for page)
//   x3 = num_iterations (size_t) - number of strided accesses to perform
// Returns:
//   x0 = 64‑bit XOR checksum
// Clobbers:
//   x4‑x8, x12‑x13, q0‑q7, q16‑q31 (data + accumulators, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Uses modulo arithmetic to wrap around buffer when stride exceeds buffer size.
//   * Loads 32 bytes (one cache line) per iteration to test strided prefetch behavior.
//   * Distributes XOR across two accumulators (v0-v1) to reduce dependency depth.
// -----------------------------------------------------------------------------

.global _memory_read_strided_loop_asm
.align 4
_memory_read_strided_loop_asm:
    mov x4, xzr             // iteration_count = 0
    mov x5, xzr             // offset = 0 (tracks virtual offset, wraps via modulo)
    mov x12, xzr            // Zero vector checksum accumulator (lower)
    mov x13, xzr            // Zero vector checksum accumulator (upper)

    // Zero accumulators (v0-v1) using XOR self-operation.
    // Use caller-saved registers q0-q7,q16-q31 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    eor v0.16b, v0.16b, v0.16b   // Zero accumulator 0 (caller-saved, safe)
    eor v1.16b, v1.16b, v1.16b   // Zero accumulator 1 (caller-saved, safe)

read_strided_loop:              // Main strided access loop
    // Loop invariants: x2=stride, x1=byteCount, x3=num_iterations, accumulators v0-v1 accumulate XOR.
    // Only x4 (iteration_count) and x5 (offset) change per iteration, wrapping via modulo.

    // Check termination: iteration_count >= num_iterations?
    cmp x4, x3              // iteration_count >= num_iterations?
    b.ge read_strided_combine_sum  // If done, combine checksums

    // Calculate current address: src + (offset % byteCount)
    // Use modulo operation: offset % byteCount = offset - (offset / byteCount) * byteCount
    // This wraps the offset around when it exceeds buffer size, creating strided pattern.
    udiv x6, x5, x1         // quotient = offset / byteCount
    msub x7, x6, x1, x5     // remainder = offset - quotient * byteCount (wraps around)
    add x8, x0, x7          // addr = src + remainder (actual memory address)

    // Load 32 bytes (one cache line worth) at strided offset
    // Using caller-saved registers: q2-q3 (avoiding q8-q15 per AAPCS64)
    ldp q2, q3, [x8]        // Load pair (32B total) from strided address

    // Accumulate into checksum
    // Distribute XOR across 2 accumulators to reduce dependency depth
    eor v0.16b, v0.16b, v2.16b    // Accumulate q2 into v0
    eor v1.16b, v1.16b, v3.16b    // Accumulate q3 into v1

    // Advance offset and iteration count
    add x5, x5, x2          // offset += stride
    add x4, x4, #1          // iteration_count++
    b read_strided_loop     // Loop again

read_strided_combine_sum:       // Combine checksums and return
    
    // Combine checksums
    // Final reduction: combine v0 and v1 into v0, then extract to x0
    eor v0.16b, v0.16b, v1.16b   // Combine v0 and v1 into v0
    umov x12, v0.d[0]            // Extract lower 64 bits of combined vector sum
    umov x13, v0.d[1]            // Extract upper 64 bits of combined vector sum
    eor x0, x12, x13             // Combine both halves into final checksum
    
    ret                         // Return checksum in x0

