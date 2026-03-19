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
// memory_write_strided_64_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_strided_64_loop_asm(void* dst,
//                                                     size_t byteCount,
//                                                     size_t num_iterations);
// Purpose:
//   Write zeros using a fixed 64-byte stride and wraparound addressing.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
//   x2 = num_iterations (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x5, q0-q1 (caller-saved only)
// -----------------------------------------------------------------------------

.global _memory_write_strided_64_loop_asm
.align 4
_memory_write_strided_64_loop_asm:
    mov x3, xzr                 // iteration_count = 0
    mov x4, xzr                 // offset = 0
    movi v0.16b, #0             // zero vector 0
    movi v1.16b, #0             // zero vector 1

write_strided_64_loop:
    cmp x3, x2
    b.hs write_strided_64_done

    add x5, x0, x4              // addr = dst + offset
    stnp q0, q1, [x5]           // store 32B

    add x4, x4, #64             // offset += 64B stride
    cmp x4, x1
    b.lo write_strided_64_next
    sub x4, x4, x1              // wrap

write_strided_64_next:
    add x3, x3, #1
    b write_strided_64_loop

write_strided_64_done:
    ret
