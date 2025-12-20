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
//   extern "C" void memory_copy_strided_loop_asm(void* dst, const void* src, size_t byteCount, size_t stride);
// Purpose:
//   Copy 'byteCount' bytes from 'src' to 'dst' using strided access pattern to exercise
//   memory copy bandwidth with non-sequential access (e.g., cache line or page stride).
//   Accesses memory at offsets: 0, stride, 2*stride, 3*stride, ... wrapping around.
//   Uses wide (128‑bit) vector loads and pair stores, processing 32 bytes per iteration.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = byteCount (size_t)
//   x3 = stride (size_t) - stride in bytes (e.g., 64 for cache line, 4096 for page)
// Returns:
//   (none)
// Clobbers:
//   x4‑x8 (temporaries), q0‑q1 (data vectors, avoiding q8‑q15 per AAPCS64)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
// Implementation Notes:
//   * Uses modulo arithmetic to wrap around buffer when stride exceeds buffer size.
//   * Copies 32 bytes (one cache line) per iteration to test strided copy behavior.
// -----------------------------------------------------------------------------

.global _memory_copy_strided_loop_asm
.align 4
_memory_copy_strided_loop_asm:
    mov x4, xzr             // offset = 0 (tracks virtual offset, wraps via modulo)

copy_strided_loop:              // Main strided access loop
    // Loop invariants: x3=stride, x2=byteCount, x0/x1 are base pointers.
    // Only x4 (offset) changes per iteration, wrapping via modulo.
    
    // Calculate current addresses: src + (offset % byteCount), dst + (offset % byteCount)
    // Use modulo operation: offset % byteCount = offset - (offset / byteCount) * byteCount
    // This wraps the offset around when it exceeds buffer size, creating strided pattern.
    udiv x5, x4, x2         // quotient = offset / byteCount
    msub x6, x5, x2, x4     // remainder = offset - quotient * byteCount (wraps around)
    add x7, x1, x6          // src_addr = src + remainder (actual source address)
    add x8, x0, x6          // dst_addr = dst + remainder (actual destination address)
    
    // Load 32 bytes from source
    // Use caller-saved registers q0-q1 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    ldp q0, q1, [x7]        // Load pair (32B total) from strided source address
    
    // Store to destination
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0, q1, [x8]       // Store pair (32B total, non-temporal) to strided destination
    
    // Advance offset by stride
    add x4, x4, x3          // offset += stride
    
    // Check if we've processed enough bytes (process byteCount bytes total)
    // We process 32 bytes per iteration, so we need byteCount/32 iterations
    // But we track by offset, so we need offset to reach at least byteCount
    cmp x4, x2              // offset >= byteCount?
    b.lt copy_strided_loop  // Loop if more bytes to process
    
    ret                     // Return

