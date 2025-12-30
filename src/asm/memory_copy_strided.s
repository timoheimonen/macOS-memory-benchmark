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
// memory_copy_strided_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_strided_loop_asm(void* dst, const void* src, size_t byteCount, size_t stride, size_t num_iterations);
// Purpose:
//   Copy 'byteCount' bytes from 'src' to 'dst' using strided access pattern to exercise
//   memory copy bandwidth with non-sequential access (e.g., cache line or page stride).
//   Accesses memory at offsets: 0, stride, 2*stride, 3*stride, ... wrapping around.
//   Uses wide (128‑bit) vector loads and pair stores, processing 32 bytes per iteration.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = byteCount (size_t) - for modulo wrapping only
//   x3 = stride (size_t) - stride in bytes (e.g., 64 for cache line, 4096 for page)
//   x4 = num_iterations (size_t) - number of strided accesses to perform
// Returns:
//   (none)
// Clobbers:
//   x5‑x10 (temporaries), q0‑q1 (data vectors, avoiding q8‑q15 per AAPCS64)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
// Implementation Notes:
//   * Uses modulo arithmetic to wrap around buffer when stride exceeds buffer size.
//   * Copies 32 bytes (one cache line) per iteration to test strided copy behavior.
// -----------------------------------------------------------------------------

.global _memory_copy_strided_loop_asm
.align 4
_memory_copy_strided_loop_asm:
    mov x5, xzr             // iteration_count = 0
    mov x6, xzr             // offset = 0 (tracks virtual offset, wraps via modulo)

copy_strided_loop:              // Main strided access loop
    // Loop invariants: x3=stride, x2=byteCount, x4=num_iterations, x0/x1 are base pointers.
    // Only x5 (iteration_count) and x6 (offset) change per iteration, wrapping via modulo.

    // Check termination: iteration_count >= num_iterations?
    cmp x5, x4              // iteration_count >= num_iterations?
    b.ge copy_strided_done  // If done, exit

    // Calculate current addresses: src + (offset % byteCount), dst + (offset % byteCount)
    // Use modulo operation: offset % byteCount = offset - (offset / byteCount) * byteCount
    // This wraps the offset around when it exceeds buffer size, creating strided pattern.
    udiv x7, x6, x2         // quotient = offset / byteCount
    msub x8, x7, x2, x6     // remainder = offset - quotient * byteCount (wraps around)
    add x9, x1, x8          // src_addr = src + remainder (actual source address)
    add x10, x0, x8         // dst_addr = dst + remainder (actual destination address)

    // Load 32 bytes from source
    // Use caller-saved registers q0-q1 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    ldp q0, q1, [x9]        // Load pair (32B total) from strided source address

    // Store to destination
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0, q1, [x10]      // Store pair (32B total, non-temporal) to strided destination

    // Advance offset and iteration count
    add x6, x6, x3          // offset += stride
    add x5, x5, #1          // iteration_count++
    b copy_strided_loop     // Loop again

copy_strided_done:          // Return to caller
    ret                     // Return

