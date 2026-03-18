// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
// memory_read_strided_64_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_strided_64_loop_asm(const void* src,
//                                                        size_t byteCount,
//                                                        size_t num_iterations);
// Purpose:
//   Read memory using a fixed 64-byte stride with wraparound addressing.
//   Accumulates an XOR checksum across all 32-byte loads to keep accesses
//   architecturally visible.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t)
//   x2 = num_iterations (size_t)
// Returns:
//   x0 = 64-bit XOR checksum
// Clobbers:
//   x3-x5, x12-x13, q0-q3 (caller-saved only)
// Implementation Notes:
//   * Fixed stride immediate avoids passing/decoding a dynamic stride.
//   * Wraparound uses compare/subtract (no divide/modulo in hot loop).
// -----------------------------------------------------------------------------

.global _memory_read_strided_64_loop_asm
.align 4
_memory_read_strided_64_loop_asm:
    mov x3, xzr                 // iteration_count = 0
    mov x4, xzr                 // offset = 0
    mov x12, xzr                // checksum low-half scratch
    mov x13, xzr                // checksum high-half scratch
    eor v0.16b, v0.16b, v0.16b  // zero accumulator 0
    eor v1.16b, v1.16b, v1.16b  // zero accumulator 1

read_strided_64_loop:
    cmp x3, x2                  // iteration_count >= num_iterations?
    b.hs read_strided_64_done

    add x5, x0, x4              // addr = src + offset
    ldp q2, q3, [x5]            // load 32B at current strided address
    eor v0.16b, v0.16b, v2.16b  // fold first 16B
    eor v1.16b, v1.16b, v3.16b  // fold second 16B

    add x4, x4, #64             // offset += 64B stride
    cmp x4, x1                  // wrapped past byteCount?
    b.lo read_strided_64_next
    sub x4, x4, x1              // offset -= byteCount (single-wrap)

read_strided_64_next:
    add x3, x3, #1              // iteration_count++
    b read_strided_64_loop

read_strided_64_done:
    eor v0.16b, v0.16b, v1.16b  // combine vector accumulators
    umov x12, v0.d[0]           // extract low 64 bits
    umov x13, v0.d[1]           // extract high 64 bits
    eor x0, x12, x13            // collapse to final checksum
    ret
