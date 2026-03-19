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
// memory_copy_strided_64_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_strided_64_loop_asm(void* dst,
//                                                    const void* src,
//                                                    size_t byteCount,
//                                                    size_t num_iterations);
// Purpose:
//   Copy 32-byte chunks using a fixed 64-byte stride with wraparound.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = byteCount (size_t)
//   x3 = num_iterations (size_t)
// Returns:
//   (none)
// Clobbers:
//   x4-x7, q0-q1 (caller-saved only)
// -----------------------------------------------------------------------------

.global _memory_copy_strided_64_loop_asm
.align 4
_memory_copy_strided_64_loop_asm:
    mov x4, xzr                 // iteration_count = 0
    mov x5, xzr                 // offset = 0

copy_strided_64_loop:
    cmp x4, x3
    b.hs copy_strided_64_done

    add x6, x1, x5              // src_addr = src + offset
    add x7, x0, x5              // dst_addr = dst + offset
    ldp q0, q1, [x6]            // load 32B
    stnp q0, q1, [x7]           // store 32B

    add x5, x5, #64             // offset += 64B stride
    cmp x5, x2
    b.lo copy_strided_64_next
    sub x5, x5, x2              // wrap

copy_strided_64_next:
    add x4, x4, #1
    b copy_strided_64_loop

copy_strided_64_done:
    ret
