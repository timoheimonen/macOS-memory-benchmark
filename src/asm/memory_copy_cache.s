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
// memory_copy_cache_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_cache_loop_asm(void* dst, const void* src, size_t byteCount);
// Purpose:
//   Copy 'byteCount' bytes from 'src' to 'dst' for cache-bandwidth paths (L1/L2/custom).
//   This kernel is intentionally independent from main-memory copy kernel so cache
//   tuning can evolve without coupling to DRAM path behavior.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x8, q0-q7, q16-q31 (caller-saved only)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
// -----------------------------------------------------------------------------

.global _memory_copy_cache_loop_asm
.align 4
_memory_copy_cache_loop_asm:
    mov x3, xzr
    mov x4, #512

copy_cache_loop_start_nt512:
    subs x5, x2, x3
    cmp x5, x4
    b.lo copy_cache_loop_cleanup

    add x6, x1, x3
    add x7, x0, x3

    ldp q0,  q1,  [x6, #0]
    ldp q2,  q3,  [x6, #32]
    ldp q4,  q5,  [x6, #64]
    ldp q6,  q7,  [x6, #96]
    ldp q16, q17, [x6, #128]
    ldp q18, q19, [x6, #160]
    ldp q20, q21, [x6, #192]
    ldp q22, q23, [x6, #224]
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q16, q17, [x7, #128]
    stnp q18, q19, [x7, #160]
    stnp q20, q21, [x7, #192]
    stnp q22, q23, [x7, #224]

    ldp q24, q25, [x6, #256]
    ldp q26, q27, [x6, #288]
    ldp q28, q29, [x6, #320]
    ldp q30, q31, [x6, #352]
    ldp q0,  q1,  [x6, #384]
    ldp q2,  q3,  [x6, #416]
    ldp q4,  q5,  [x6, #448]
    ldp q6,  q7,  [x6, #480]
    stnp q24, q25, [x7, #256]
    stnp q26, q27, [x7, #288]
    stnp q28, q29, [x7, #320]
    stnp q30, q31, [x7, #352]
    stnp q0,  q1,  [x7, #384]
    stnp q2,  q3,  [x7, #416]
    stnp q4,  q5,  [x7, #448]
    stnp q6,  q7,  [x7, #480]

    add x3, x3, x4
    b copy_cache_loop_start_nt512

copy_cache_loop_cleanup:
    cmp x3, x2
    b.hs copy_cache_loop_end

    subs x5, x2, x3
    add x6, x1, x3
    add x7, x0, x3

    cmp x5, #256
    b.lo copy_cache_cleanup_128
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    ldp q16, q17, [x6, #128]
    ldp q18, q19, [x6, #160]
    ldp q20, q21, [x6, #192]
    ldp q22, q23, [x6, #224]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q16, q17, [x7, #128]
    stnp q18, q19, [x7, #160]
    stnp q20, q21, [x7, #192]
    stnp q22, q23, [x7, #224]
    add x6, x6, #256
    add x7, x7, #256
    sub x5, x5, #256

copy_cache_cleanup_128:
    cmp x5, #128
    b.lo copy_cache_cleanup_64
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x6, x6, #128
    add x7, x7, #128
    sub x5, x5, #128

copy_cache_cleanup_64:
    cmp x5, #64
    b.lo copy_cache_cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x6, x6, #64
    add x7, x7, #64
    sub x5, x5, #64

copy_cache_cleanup_32:
    cmp x5, #32
    b.lo copy_cache_cleanup_byte
    ldp q0, q1, [x6, #0]
    stnp q0, q1, [x7, #0]
    add x6, x6, #32
    add x7, x7, #32
    sub x5, x5, #32

copy_cache_cleanup_byte:
    cbz x5, copy_cache_loop_end
    ldrb w8, [x6], #1
    strb w8, [x7], #1
    subs x5, x5, #1
    b.ne copy_cache_cleanup_byte

copy_cache_loop_end:
    ret
