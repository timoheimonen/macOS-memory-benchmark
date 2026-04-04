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
// memory_copy_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// Purpose:
//   Copy 'byteCount' bytes from 'src' to 'dst' using wide (128‑bit) vector loads
//   and pair stores, processing 512 bytes per iteration to maximize throughput.
// Arguments:
//   x0 = dst (void*)
//   x1 = src (const void*)
//   x2 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3‑x8 (temporaries), q0‑q7, q16‑q31 (data vectors, avoiding q8‑q15 per AAPCS64)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
//   * Tail handled with progressively smaller vector tiers then byte loop (<32B).
// Implementation Notes:
//   * Pointer-bump addressing (no per-iteration offset math).
// -----------------------------------------------------------------------------
.global _memory_copy_loop_asm
.align 4
_memory_copy_loop_asm:
    mov x6, x1              // src_ptr = src
    mov x7, x0              // dst_ptr = dst
    mov x5, x2              // remaining = byteCount

copy_loop_start_nt512:      // Main 512B block loop
    cmp x5, #512            // remaining < 512?
    b.lo copy_loop_cleanup

    // Load 512 bytes from source (16 ldp pairs, 32B each)
    // Use caller-saved registers q0-q7,q16-q31 only (avoid q8-q15 per AAPCS64).
    // Load first 8 pairs (0-7) into q0-q7 and q16-q23
    ldp q0,  q1,  [x6, #0]
    ldp q2,  q3,  [x6, #32]
    ldp q4,  q5,  [x6, #64]
    ldp q6,  q7,  [x6, #96]
    ldp q16, q17, [x6, #128]
    ldp q18, q19, [x6, #160]
    ldp q20, q21, [x6, #192]
    ldp q22, q23, [x6, #224]

    // Store first 8 pairs before loading next 8 (to avoid register pressure).
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q16, q17, [x7, #128]
    stnp q18, q19, [x7, #160]
    stnp q20, q21, [x7, #192]
    stnp q22, q23, [x7, #224]

    // Load next 8 pairs (8-15) into q24-q31 and q0-q7 (reusing q0-q7)
    ldp q24, q25, [x6, #256]
    ldp q26, q27, [x6, #288]
    ldp q28, q29, [x6, #320]
    ldp q30, q31, [x6, #352]
    ldp q0,  q1,  [x6, #384]
    ldp q2,  q3,  [x6, #416]
    ldp q4,  q5,  [x6, #448]
    ldp q6,  q7,  [x6, #480]

    // Store next 8 pairs (8-15)
    stnp q24, q25, [x7, #256]
    stnp q26, q27, [x7, #288]
    stnp q28, q29, [x7, #320]
    stnp q30, q31, [x7, #352]
    stnp q0,  q1,  [x7, #384]
    stnp q2,  q3,  [x7, #416]
    stnp q4,  q5,  [x7, #448]
    stnp q6,  q7,  [x7, #480]

    add x6, x6, #512        // src_ptr += 512
    add x7, x7, #512        // dst_ptr += 512
    sub x5, x5, #512        // remaining -= 512
    b copy_loop_start_nt512

copy_loop_cleanup:          // Tail handling when <512B remain
    cbz x5, copy_loop_end   // If none remain, exit

    // Handle 256B chunks (8 ldp/stnp pairs)
    cmp x5, #256
    b.lo cleanup_128
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

cleanup_128:                  // Handle 128B chunks (4 ldp + 4 stnp)
    cmp x5, #128
    b.lo cleanup_64
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

cleanup_64:                   // Handle 64B chunks (2 ldp + 2 stnp)
    cmp x5, #64
    b.lo cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x6, x6, #64
    add x7, x7, #64
    sub x5, x5, #64

cleanup_32:                   // Handle 32B chunk (1 ldp + 1 stnp)
    cmp x5, #32
    b.lo copy_cleanup_byte
    ldp q0, q1, [x6, #0]
    stnp q0, q1, [x7, #0]
    add x6, x6, #32
    add x7, x7, #32
    sub x5, x5, #32

copy_cleanup_byte:           // Byte tail for final <32B
    // Final byte-by-byte copy for <32B remainder. Expected to be rare in
    // bandwidth benchmarks but ensures correctness for any input size.
    cbz x5, copy_loop_end
    ldrb w8, [x6], #1
    strb w8, [x7], #1
    subs x5, x5, #1
    b.ne copy_cleanup_byte

copy_loop_end:              // Return to caller
    ret                     // Return
