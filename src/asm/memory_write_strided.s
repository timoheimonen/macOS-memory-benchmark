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
// memory_write_strided_phased_loop_asm
// -----------------------------------------------------------------------------
// Executes complete valid 32-byte strided passes and advances the starting
// phase by 32 bytes after each pass. Arguments: x0=dst, x1=byteCount,
// x2=stride, x3=passes, x4=initial_phase. The caller guarantees byteCount >=
// stride + 32, stride is a multiple of 32, and initial_phase < stride.
// Clobbers caller-saved x5-x9 and q0-q1 only. Timing barriers remain the
// caller's responsibility.
// -----------------------------------------------------------------------------

.global _memory_write_strided_phased_loop_asm
.align 4
_memory_write_strided_phased_loop_asm:
    mov x5, xzr                         // completed passes
    mov x6, x4                          // current phase
    sub x7, x1, #32                     // last valid 32-byte access offset
    movi v0.16b, #0
    movi v1.16b, #0

write_strided_phased_pass:
    cmp x5, x3
    b.hs write_strided_phased_done
    mov x8, x6                          // offset = phase

write_strided_phased_access:
    cmp x8, x7
    b.hi write_strided_phased_next_pass
    add x9, x0, x8
    stnp q0, q1, [x9]
    add x8, x8, x2
    b write_strided_phased_access

write_strided_phased_next_pass:
    add x6, x6, #32
    cmp x6, x2
    b.lo write_strided_phased_phase_ready
    sub x6, x6, x2
write_strided_phased_phase_ready:
    add x5, x5, #1
    b write_strided_phased_pass

write_strided_phased_done:
    ret
