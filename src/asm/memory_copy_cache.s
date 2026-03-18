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
// Implementation Notes:
//   * Uses pointer+remaining loop state (x6/x7/x5) to reduce loop-control overhead.
//   * Processes 512B per iteration with 16 load/store pairs.
//   * Main body interleaves load/store pairs to keep copy pipeline busy.
//   * Tail uses bit-tested chunk ladder plus 16/8/4/2/1 scalar finish.
// -----------------------------------------------------------------------------

.global _memory_copy_cache_loop_asm
.align 4
_memory_copy_cache_loop_asm:
    // x6 = current src, x7 = current dst, x5 = remaining bytes.
    mov x6, x1
    mov x7, x0
    mov x5, x2

copy_cache_loop_start_nt512:  // Main 512B loop
    // If <512B remain, switch to tail handling.
    cmp x5, #512
    b.lo copy_cache_loop_cleanup

    // Interleaved load/store pairs across one 512B block.
    ldp q0,  q1,  [x6, #0]
    stp q0,  q1,  [x7, #0]
    ldp q2,  q3,  [x6, #32]
    stp q2,  q3,  [x7, #32]
    ldp q4,  q5,  [x6, #64]
    stp q4,  q5,  [x7, #64]
    ldp q6,  q7,  [x6, #96]
    stp q6,  q7,  [x7, #96]
    ldp q16, q17, [x6, #128]
    stp q16, q17, [x7, #128]
    ldp q18, q19, [x6, #160]
    stp q18, q19, [x7, #160]
    ldp q20, q21, [x6, #192]
    stp q20, q21, [x7, #192]
    ldp q22, q23, [x6, #224]
    stp q22, q23, [x7, #224]
    ldp q24, q25, [x6, #256]
    stp q24, q25, [x7, #256]
    ldp q26, q27, [x6, #288]
    stp q26, q27, [x7, #288]
    ldp q28, q29, [x6, #320]
    stp q28, q29, [x7, #320]
    ldp q30, q31, [x6, #352]
    stp q30, q31, [x7, #352]
    ldp q0,  q1,  [x6, #384]
    stp q0,  q1,  [x7, #384]
    ldp q2,  q3,  [x6, #416]
    stp q2,  q3,  [x7, #416]
    ldp q4,  q5,  [x6, #448]
    stp q4,  q5,  [x7, #448]
    ldp q6,  q7,  [x6, #480]
    stp q6,  q7,  [x7, #480]

    // Advance both pointers and remaining-byte counter.
    add x6, x6, #512
    add x7, x7, #512
    sub x5, x5, #512
    b copy_cache_loop_start_nt512

copy_cache_loop_cleanup:      // Tail handling when <512B remain
    cbz x5, copy_cache_loop_end

    // Tiered tail: test size bits for 256/128/64/32 before scalar ladder.
    tbz x5, #8, copy_cache_cleanup_128
    // 256B chunk
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    ldp q16, q17, [x6, #128]
    ldp q18, q19, [x6, #160]
    ldp q20, q21, [x6, #192]
    ldp q22, q23, [x6, #224]
    stp q0, q1, [x7, #0]
    stp q2, q3, [x7, #32]
    stp q4, q5, [x7, #64]
    stp q6, q7, [x7, #96]
    stp q16, q17, [x7, #128]
    stp q18, q19, [x7, #160]
    stp q20, q21, [x7, #192]
    stp q22, q23, [x7, #224]
    add x6, x6, #256
    add x7, x7, #256
    sub x5, x5, #256

copy_cache_cleanup_128:       // Optional 128B chunk
    tbz x5, #7, copy_cache_cleanup_64
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    stp q0, q1, [x7, #0]
    stp q2, q3, [x7, #32]
    stp q4, q5, [x7, #64]
    stp q6, q7, [x7, #96]
    add x6, x6, #128
    add x7, x7, #128
    sub x5, x5, #128

copy_cache_cleanup_64:        // Optional 64B chunk
    tbz x5, #6, copy_cache_cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    stp q0, q1, [x7, #0]
    stp q2, q3, [x7, #32]
    add x6, x6, #64
    add x7, x7, #64
    sub x5, x5, #64

copy_cache_cleanup_32:        // Optional 32B chunk
    tbz x5, #5, copy_cache_cleanup_16
    ldp q0, q1, [x6, #0]
    stp q0, q1, [x7, #0]
    add x6, x6, #32
    add x7, x7, #32
    sub x5, x5, #32

copy_cache_cleanup_16:        // Optional 16B scalar copy
    tbz x5, #4, copy_cache_cleanup_8
    ldr q0, [x6], #16
    str q0, [x7], #16
    sub x5, x5, #16

copy_cache_cleanup_8:         // Optional 8B scalar copy
    tbz x5, #3, copy_cache_cleanup_4
    ldr x8, [x6], #8
    str x8, [x7], #8
    sub x5, x5, #8

copy_cache_cleanup_4:         // Optional 4B scalar copy
    tbz x5, #2, copy_cache_cleanup_2
    ldr w8, [x6], #4
    str w8, [x7], #4
    sub x5, x5, #4

copy_cache_cleanup_2:         // Optional 2B scalar copy
    tbz x5, #1, copy_cache_cleanup_1
    ldrh w8, [x6], #2
    strh w8, [x7], #2
    sub x5, x5, #2

copy_cache_cleanup_1:         // Optional final 1B copy
    tbz x5, #0, copy_cache_loop_end
    ldrb w8, [x6]
    strb w8, [x7]

copy_cache_loop_end:          // Return to caller
    ret
