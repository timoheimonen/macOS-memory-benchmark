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
// memory_write_reverse_pattern_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_reverse_pattern_loop_asm(void* dst, size_t byteCount, uint64_t pattern64);
// Purpose:
//   Write 'byteCount' bytes repeating the little-endian 64-bit 'pattern64'
//   to 'dst' in reverse order (backwards) using 512B non-temporal block stores (STNP).
//   Each written byte at offset i equals (pattern64 >> ((i & 7) * 8)) & 0xFF.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
//   x2 = pattern64 (uint64_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x7, x9, q0-q7 (caller-saved only; preserves AAPCS64 callee-saved)
// Implementation Notes:
//   * Reverse SIMD chunks start at dst + byteCount - 32*k, so their base offset
//     modulo 8 equals (byteCount & 7). Rotating pattern64 right by (byteCount & 7) * 8
//     bits aligns every 32B/64B/128B/256B/512B store with dst[i] for any byteCount.
//   * Main loop label is 64-byte aligned to match memory_write_reverse_loop_asm.
//   * No internal barriers (dsb/isb); timing barriers remain caller's responsibility.
// -----------------------------------------------------------------------------

.global _memory_write_reverse_pattern_loop_asm
.align 4
_memory_write_reverse_pattern_loop_asm:
    cbz x1, write_reverse_pattern_loop_end

    add x3, x0, x1          // end_ptr = dst + byteCount
    mov x4, #512            // step = 512 bytes
    mov x5, x1              // remaining = byteCount

    // Rotate pattern64 right by (byteCount & 7) * 8 bits into x6 so SIMD chunks
    // ending at dst + byteCount store exact (pattern64 >> ((i & 7) * 8)) & 0xFF.
    and x9, x1, #7
    lsl x9, x9, #3
    ror x6, x2, x9

    cmp x1, #32
    b.lo write_reverse_pattern_cleanup_byte

    // Materialize rotated 64-bit pattern across caller-saved SIMD registers v0-v7.
    dup v0.2d, x6
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b

    .p2align 6
write_reverse_pattern_loop_start_nt512:
    cmp x3, x0
    b.ls write_reverse_pattern_loop_end

    sub x5, x3, x0          // remaining = end_ptr - dst
    cmp x5, x4              // remaining < 512?
    b.lo write_reverse_pattern_cleanup

    sub x7, x3, #512        // block_start = end_ptr - 512

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

    sub x3, x3, x4          // end_ptr -= 512
    b write_reverse_pattern_loop_start_nt512

write_reverse_pattern_cleanup:
    cmp x3, x0
    b.ls write_reverse_pattern_loop_end

    subs x5, x3, x0         // remaining = end_ptr - dst

    // Handle 256B chunk (8 stnp pairs)
    cmp x5, #256
    b.lo write_reverse_pattern_cleanup_128
    sub x7, x3, #256
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q0, q1, [x7, #128]
    stnp q2, q3, [x7, #160]
    stnp q4, q5, [x7, #192]
    stnp q6, q7, [x7, #224]
    sub x3, x3, #256
    sub x5, x5, #256

write_reverse_pattern_cleanup_128:
    cmp x5, #128
    b.lo write_reverse_pattern_cleanup_64
    sub x7, x3, #128
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    sub x3, x3, #128
    sub x5, x5, #128

write_reverse_pattern_cleanup_64:
    cmp x5, #64
    b.lo write_reverse_pattern_cleanup_32
    sub x7, x3, #64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    sub x3, x3, #64
    sub x5, x5, #64

write_reverse_pattern_cleanup_32:
    cmp x5, #32
    b.lo write_reverse_pattern_cleanup_byte
    sub x7, x3, #32
    stnp q0, q1, [x7, #0]
    sub x3, x3, #32
    sub x5, x5, #32

write_reverse_pattern_cleanup_byte:
    cbz x5, write_reverse_pattern_loop_end
write_reverse_pattern_byte_loop:
    ror x6, x6, #56         // rotate left by 8 bits to expose preceding byte
    strb w6, [x3, #-1]!
    subs x5, x5, #1
    b.ne write_reverse_pattern_byte_loop

write_reverse_pattern_loop_end:
    ret
