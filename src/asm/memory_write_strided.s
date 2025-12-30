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
// memory_write_strided_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_strided_loop_asm(void* dst, size_t byteCount, size_t stride, size_t num_iterations);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' using strided access pattern to exercise
//   memory write bandwidth with non-sequential access (e.g., cache line or page stride).
//   Accesses memory at offsets: 0, stride, 2*stride, 3*stride, ... wrapping around.
//   Uses STNP as a non‑temporal hint.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t) - for modulo wrapping only
//   x2 = stride (size_t) - stride in bytes (e.g., 64 for cache line, 4096 for page)
//   x3 = num_iterations (size_t) - number of strided accesses to perform
// Returns:
//   (none)
// Clobbers:
//   x4‑x8, q0‑q1 (zero vectors, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Uses modulo arithmetic to wrap around buffer when stride exceeds buffer size.
//   * Stores 32 bytes (one cache line) per iteration to test strided write behavior.
//   * Zero vectors are materialized once (movi) then reused.
// -----------------------------------------------------------------------------

.global _memory_write_strided_loop_asm
.align 4
_memory_write_strided_loop_asm:
    mov x4, xzr             // iteration_count = 0
    mov x5, xzr             // offset = 0 (tracks virtual offset, wraps via modulo)

    // Zero out data registers (using only caller-saved: v0-v1, avoiding v8-v15 per AAPCS64).
    // Use caller-saved registers q0-q1 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    // Zero vectors are materialized once (movi) then reused throughout the loop.
    movi v0.16b, #0             // Zero vector 0
    movi v1.16b, #0             // Zero vector 1

write_strided_loop:              // Main strided access loop
    // Loop invariants: x2=stride, x1=byteCount, x3=num_iterations, vectors v0-v1 are zero.
    // Only x4 (iteration_count) and x5 (offset) change per iteration, wrapping via modulo.

    // Check termination: iteration_count >= num_iterations?
    cmp x4, x3              // iteration_count >= num_iterations?
    b.ge write_strided_done // If done, exit

    // Calculate current address: dst + (offset % byteCount)
    // Use modulo operation: offset % byteCount = offset - (offset / byteCount) * byteCount
    // This wraps the offset around when it exceeds buffer size, creating strided pattern.
    udiv x6, x5, x1         // quotient = offset / byteCount
    msub x7, x6, x1, x5     // remainder = offset - quotient * byteCount (wraps around)
    add x8, x0, x7          // addr = dst + remainder (actual memory address)

    // Store 32 bytes (one cache line worth) at strided offset
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0, q1, [x8]       // Store pair (32B total, non-temporal) to strided address

    // Advance offset and iteration count
    add x5, x5, x2          // offset += stride
    add x4, x4, #1          // iteration_count++
    b write_strided_loop    // Loop again

write_strided_done:         // Return to caller
    ret                     // Return

