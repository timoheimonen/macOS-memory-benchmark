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
// memory_read_strided_phased_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   uint64_t memory_read_strided_phased_loop_asm(const void* src,
//       size_t byteCount, size_t stride, size_t passes, size_t initial_phase);
//
// Executes every valid 32-byte access in each pass. The starting phase advances
// by 32 bytes after a pass and wraps at stride, so sparse repetitions do not
// revisit only the phase-zero addresses. The caller guarantees byteCount >=
// stride + 32, stride is a multiple of 32, and initial_phase < stride.
// Arguments: x0=src, x1=byteCount, x2=stride, x3=passes, x4=initial_phase.
// Clobbers caller-saved x5-x13 and q0-q3; preserves all AAPCS64 callee-saved
// registers. Timing barriers remain the caller's responsibility.
// -----------------------------------------------------------------------------

.global _memory_read_strided_phased_loop_asm
.align 4
_memory_read_strided_phased_loop_asm:
    mov x5, xzr                         // completed passes
    mov x6, x4                          // current phase
    sub x7, x1, #32                     // last valid 32-byte access offset

    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b

read_strided_phased_pass:
    cmp x5, x3
    b.hs read_strided_phased_done
    mov x8, x6                          // offset = phase

read_strided_phased_access:
    cmp x8, x7
    b.hi read_strided_phased_next_pass
    add x9, x0, x8
    ldp q2, q3, [x9]
    eor v0.16b, v0.16b, v2.16b
    eor v1.16b, v1.16b, v3.16b
    add x8, x8, x2
    b read_strided_phased_access

read_strided_phased_next_pass:
    add x6, x6, #32
    cmp x6, x2
    b.lo read_strided_phased_phase_ready
    sub x6, x6, x2
read_strided_phased_phase_ready:
    add x5, x5, #1
    b read_strided_phased_pass

read_strided_phased_done:
    eor v0.16b, v0.16b, v1.16b
    umov x12, v0.d[0]
    umov x13, v0.d[1]
    eor x0, x12, x13
    ret
