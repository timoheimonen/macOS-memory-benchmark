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
// memory_read_cache_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_cache_loop_asm(const void* src, size_t byteCount);
// Purpose:
//   Read 'byteCount' bytes from 'src' for cache-bandwidth paths (L1/L2/custom).
//   This kernel is intentionally independent from main-memory read kernel so cache
//   tuning can evolve without coupling to DRAM path behavior.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t)
// Returns:
//   x0 = 64-bit XOR checksum
// Clobbers:
//   x2-x7, x12-x13, q0-q7, q16-q31
// -----------------------------------------------------------------------------

.global _memory_read_cache_loop_asm
.align 4
_memory_read_cache_loop_asm:
    mov x3, xzr
    mov x4, #512
    mov x12, xzr

    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b

cache_read_loop_start_512:
    subs x5, x1, x3
    cmp x5, x4
    b.lo cache_read_loop_cleanup

    add x6, x0, x3

    ldp q4,  q5,  [x6, #0]
    ldp q6,  q7,  [x6, #32]
    ldp q16, q17, [x6, #64]
    ldp q18, q19, [x6, #96]
    ldp q20, q21, [x6, #128]
    ldp q22, q23, [x6, #160]
    ldp q24, q25, [x6, #192]
    ldp q26, q27, [x6, #224]

    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    eor v0.16b, v0.16b, v16.16b
    eor v1.16b, v1.16b, v17.16b
    eor v2.16b, v2.16b, v18.16b
    eor v3.16b, v3.16b, v19.16b
    eor v0.16b, v0.16b, v20.16b
    eor v1.16b, v1.16b, v21.16b
    eor v2.16b, v2.16b, v22.16b
    eor v3.16b, v3.16b, v23.16b
    eor v0.16b, v0.16b, v24.16b
    eor v1.16b, v1.16b, v25.16b
    eor v2.16b, v2.16b, v26.16b
    eor v3.16b, v3.16b, v27.16b

    ldp q4,  q5,  [x6, #256]
    ldp q6,  q7,  [x6, #288]
    ldp q16, q17, [x6, #320]
    ldp q18, q19, [x6, #352]
    ldp q20, q21, [x6, #384]
    ldp q22, q23, [x6, #416]
    ldp q24, q25, [x6, #448]
    ldp q26, q27, [x6, #480]

    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    eor v0.16b, v0.16b, v16.16b
    eor v1.16b, v1.16b, v17.16b
    eor v2.16b, v2.16b, v18.16b
    eor v3.16b, v3.16b, v19.16b
    eor v0.16b, v0.16b, v20.16b
    eor v1.16b, v1.16b, v21.16b
    eor v2.16b, v2.16b, v22.16b
    eor v3.16b, v3.16b, v23.16b
    eor v0.16b, v0.16b, v24.16b
    eor v1.16b, v1.16b, v25.16b
    eor v2.16b, v2.16b, v26.16b
    eor v3.16b, v3.16b, v27.16b

    add x3, x3, x4
    b cache_read_loop_start_512

cache_read_loop_cleanup:
    cmp x3, x1
    b.hs cache_read_loop_combine_sum

    subs x5, x1, x3
    add x6, x0, x3

    cmp x5, #256
    b.lo cache_read_cleanup_128
    ldp q4, q5, [x6, #0]
    ldp q6, q7, [x6, #32]
    ldp q16, q17, [x6, #64]
    ldp q18, q19, [x6, #96]
    ldp q20, q21, [x6, #128]
    ldp q22, q23, [x6, #160]
    ldp q24, q25, [x6, #192]
    ldp q26, q27, [x6, #224]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    eor v0.16b, v0.16b, v16.16b
    eor v1.16b, v1.16b, v17.16b
    eor v2.16b, v2.16b, v18.16b
    eor v3.16b, v3.16b, v19.16b
    eor v0.16b, v0.16b, v20.16b
    eor v1.16b, v1.16b, v21.16b
    eor v2.16b, v2.16b, v22.16b
    eor v3.16b, v3.16b, v23.16b
    eor v0.16b, v0.16b, v24.16b
    eor v1.16b, v1.16b, v25.16b
    eor v2.16b, v2.16b, v26.16b
    eor v3.16b, v3.16b, v27.16b
    add x6, x6, #256
    sub x5, x5, #256

cache_read_cleanup_128:
    cmp x5, #128
    b.lo cache_read_cleanup_64
    ldp q4, q5, [x6, #0]
    ldp q6, q7, [x6, #32]
    ldp q16, q17, [x6, #64]
    ldp q18, q19, [x6, #96]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    eor v0.16b, v0.16b, v16.16b
    eor v1.16b, v1.16b, v17.16b
    eor v2.16b, v2.16b, v18.16b
    eor v3.16b, v3.16b, v19.16b
    add x6, x6, #128
    sub x5, x5, #128

cache_read_cleanup_64:
    cmp x5, #64
    b.lo cache_read_cleanup_32
    ldp q4, q5, [x6, #0]
    ldp q6, q7, [x6, #32]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    add x6, x6, #64
    sub x5, x5, #64

cache_read_cleanup_32:
    cmp x5, #32
    b.lo cache_read_cleanup_byte
    ldp q4, q5, [x6, #0]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    add x6, x6, #32
    sub x5, x5, #32

cache_read_cleanup_byte:
    cbz x5, cache_read_loop_combine_sum
    ldrb w13, [x6], #1
    eor x12, x12, x13
    subs x5, x5, #1
    b.ne cache_read_cleanup_byte

cache_read_loop_combine_sum:
    eor v0.16b, v0.16b, v1.16b
    eor v2.16b, v2.16b, v3.16b
    eor v0.16b, v0.16b, v2.16b
    umov x0, v0.d[0]
    eor x0, x0, x12
    ret
