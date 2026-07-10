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
// memory_copy_strided_phased_loop_asm
// -----------------------------------------------------------------------------
// Executes complete valid 32-byte strided passes and advances the starting
// phase by 32 bytes after each pass. Arguments: x0=dst, x1=src, x2=byteCount,
// x3=stride, x4=passes, x5=initial_phase. The caller guarantees byteCount >=
// stride + 32, stride is a multiple of 32, and initial_phase < stride.
// Clobbers caller-saved x6-x11 and q0-q1 only. Timing barriers remain the
// caller's responsibility.
// -----------------------------------------------------------------------------

.global _memory_copy_strided_phased_loop_asm
.align 4
_memory_copy_strided_phased_loop_asm:
    mov x6, xzr                         // completed passes
    mov x7, x5                          // current phase
    sub x8, x2, #32                     // last valid 32-byte access offset

copy_strided_phased_pass:
    cmp x6, x4
    b.hs copy_strided_phased_done
    mov x9, x7                          // offset = phase

copy_strided_phased_access:
    cmp x9, x8
    b.hi copy_strided_phased_next_pass
    add x10, x1, x9
    add x11, x0, x9
    ldp q0, q1, [x10]
    stnp q0, q1, [x11]
    add x9, x9, x3
    b copy_strided_phased_access

copy_strided_phased_next_pass:
    add x7, x7, #32
    cmp x7, x3
    b.lo copy_strided_phased_phase_ready
    sub x7, x7, x3
copy_strided_phased_phase_ready:
    add x6, x6, #1
    b copy_strided_phased_pass

copy_strided_phased_done:
    ret
