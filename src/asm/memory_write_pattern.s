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
// memory_write_pattern_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_pattern_loop_asm(void* dst, size_t byteCount, uint64_t pattern64);
// Purpose:
//   Write 'byteCount' bytes repeating the little-endian 64-bit 'pattern64'
//   to 'dst' using 512B non-temporal block stores (STNP).
//   Each written byte at offset i equals (pattern64 >> ((i & 7) * 8)) & 0xFF.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
//   x2 = pattern64 (uint64_t)
// Returns:
//   (none)
// Clobbers:
//   x2, x3, x5, x7, q0-q7 (caller-saved only; preserves AAPCS64 callee-saved)
// -----------------------------------------------------------------------------

.global _memory_write_pattern_loop_asm
.align 4
_memory_write_pattern_loop_asm:
    cbz x1, write_pattern_loop_end
    mov x7, x0                  // dst_ptr = dst
    mov x5, x1                  // remaining = byteCount

    // Materialize 64-bit pattern across caller-saved SIMD registers v0-v7.
    dup v0.2d, x2
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b

    // Split remaining byteCount into full 512B blocks (x3) and tail bytes (x5).
    // x2 stays untouched with pattern64 for the 8/4/2/1 scalar tail.
    lsr x3, x5, #9
    and x5, x5, #0x1ff
    cbz x3, write_pattern_loop_cleanup

    .p2align 6
write_pattern_loop_start_nt512:
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q0,  q1,  [x7, #128]
    stnp q2,  q3,  [x7, #160]
    stnp q4,  q5,  [x7, #192]
    stnp q6,  q7,  [x7, #224]
    stnp q0,  q1,  [x7, #256]
    stnp q2,  q3,  [x7, #288]
    stnp q4,  q5,  [x7, #320]
    stnp q6,  q7,  [x7, #352]
    stnp q0,  q1,  [x7, #384]
    stnp q2,  q3,  [x7, #416]
    stnp q4,  q5,  [x7, #448]
    stnp q6,  q7,  [x7, #480]

    add x7, x7, #512
    subs x3, x3, #1
    b.ne write_pattern_loop_start_nt512

write_pattern_loop_cleanup:
    cbz x5, write_pattern_loop_end

    tbz x5, #8, write_pattern_cleanup_128
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q0, q1, [x7, #128]
    stnp q2, q3, [x7, #160]
    stnp q4, q5, [x7, #192]
    stnp q6, q7, [x7, #224]
    add x7, x7, #256
    sub x5, x5, #256

write_pattern_cleanup_128:
    tbz x5, #7, write_pattern_cleanup_64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_pattern_cleanup_64:
    tbz x5, #6, write_pattern_cleanup_32
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_pattern_cleanup_32:
    tbz x5, #5, write_pattern_cleanup_16
    stnp q0, q1, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_pattern_cleanup_16:
    tbz x5, #4, write_pattern_cleanup_8
    str q0, [x7], #16
    sub x5, x5, #16

write_pattern_cleanup_8:
    tbz x5, #3, write_pattern_cleanup_4
    str x2, [x7], #8
    sub x5, x5, #8

write_pattern_cleanup_4:
    tbz x5, #2, write_pattern_cleanup_2
    str w2, [x7], #4
    lsr x2, x2, #32
    sub x5, x5, #4

write_pattern_cleanup_2:
    tbz x5, #1, write_pattern_cleanup_1
    strh w2, [x7], #2
    lsr x2, x2, #16
    sub x5, x5, #2

write_pattern_cleanup_1:
    tbz x5, #0, write_pattern_loop_end
    strb w2, [x7]

write_pattern_loop_end:
    ret
