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
// memory_write_cache_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_cache_loop_asm(void* dst, size_t byteCount);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' for cache-bandwidth paths (L1/L2/custom).
//   This kernel is intentionally independent from main-memory write kernel so cache
//   tuning can evolve without coupling to DRAM path behavior.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x7, q0-q7, q16-q31 (caller-saved only)
// -----------------------------------------------------------------------------

.global _memory_write_cache_loop_asm
.align 4
_memory_write_cache_loop_asm:
    mov x3, xzr
    mov x4, #512

    movi v0.16b, #0
    movi v1.16b, #0
    movi v2.16b, #0
    movi v3.16b, #0
    movi v4.16b, #0
    movi v5.16b, #0
    movi v6.16b, #0
    movi v7.16b, #0
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
    movi v23.16b, #0
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0

write_cache_loop_start_nt512:
    subs x5, x1, x3
    cmp x5, x4
    b.lo write_cache_loop_cleanup

    add x7, x0, x3
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q16, q17, [x7, #128]
    stnp q18, q19, [x7, #160]
    stnp q20, q21, [x7, #192]
    stnp q22, q23, [x7, #224]
    stnp q24, q25, [x7, #256]
    stnp q26, q27, [x7, #288]
    stnp q28, q29, [x7, #320]
    stnp q30, q31, [x7, #352]
    stnp q0,  q1,  [x7, #384]
    stnp q2,  q3,  [x7, #416]
    stnp q4,  q5,  [x7, #448]
    stnp q6,  q7,  [x7, #480]

    add x3, x3, x4
    b write_cache_loop_start_nt512

write_cache_loop_cleanup:
    cmp x3, x1
    b.hs write_cache_loop_end

    subs x5, x1, x3
    add x7, x0, x3

    cmp x5, #256
    b.lo write_cache_cleanup_128
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q16, q17, [x7, #128]
    stnp q18, q19, [x7, #160]
    stnp q20, q21, [x7, #192]
    stnp q22, q23, [x7, #224]
    add x7, x7, #256
    sub x5, x5, #256

write_cache_cleanup_128:
    cmp x5, #128
    b.lo write_cache_cleanup_64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_cache_cleanup_64:
    cmp x5, #64
    b.lo write_cache_cleanup_32
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_cache_cleanup_32:
    cmp x5, #32
    b.lo write_cache_cleanup_byte
    stnp q0, q1, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_cache_cleanup_byte:
    cbz x5, write_cache_loop_end
    strb wzr, [x7], #1
    subs x5, x5, #1
    b.ne write_cache_cleanup_byte

write_cache_loop_end:
    ret
