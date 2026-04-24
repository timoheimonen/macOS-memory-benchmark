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
//   * Tail handled with progressively smaller vector tiers plus 16/8/4/2/1
//     scalar fall-through (no byte-by-byte loop).
// Implementation Notes:
//   * Pointer-bump addressing (no per-iteration offset math).
//   * Main loop uses a precomputed block count (x3) and counts down with
//     `subs + b.ne` at the bottom (3 control ops/iter) instead of the older
//     `cmp + b.lo` top guard plus `sub + b` unconditional back-edge
//     (5 control ops/iter). x5 holds tail bytes (remaining % 512) during the
//     block loop and feeds the tbz tail tiers afterwards.
//   * Tail uses `tbz` bit-tests for 256/128/64/32 tiers plus scalar 16/8/4/2/1
//     copy tiers, harmonized with memory_copy_cache.s.
//   * Main loop label is 64-byte aligned to keep the unrolled body on a single
//     I-cache line boundary for steady run-to-run timing on Apple Silicon.
// Timing Contract:
//   Caller must emit `dsb ish; isb` before reading the start-of-measurement
//   timestamp and another `dsb ish; isb` before reading the end-of-measurement
//   timestamp. This kernel emits no internal fences; barrier discipline is the
//   caller's responsibility for reproducible timing.
// -----------------------------------------------------------------------------
.global _memory_copy_loop_asm
.align 4
_memory_copy_loop_asm:
    mov x6, x1              // src_ptr = src
    mov x7, x0              // dst_ptr = dst
    mov x5, x2              // remaining = byteCount

    // Split remaining byteCount into full 512B blocks (x3) and tail bytes (x5).
    // The hot loop then becomes a counted subs+b.ne (3 control ops per 512B
    // iter) rather than a remainder compare+subtract+unconditional-branch
    // (5 control ops per iter). x5 carries the residual into the tail tiers.
    lsr x3, x5, #9              // x3 = full 512B block count (byteCount / 512)
    and x5, x5, #0x1ff          // x5 = tail bytes (byteCount % 512)
    cbz x3, copy_loop_cleanup   // No full block? Go straight to tail.

    // Align hot loop entry to 64B so the unrolled 512B body always lands on a
    // predictable I-cache line. Reduces first-iteration fetch-boundary jitter.
    .p2align 6
copy_loop_start_nt512:      // Main 512B block loop (count-down on x3)
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
    subs x3, x3, #1         // block_count -= 1, set flags
    b.ne copy_loop_start_nt512 // Loop while blocks remain

copy_loop_cleanup:          // Tail handling when <512B remain (x5 = tail bytes)
    cbz x5, copy_loop_end   // If none remain, exit

    // Tiered tail: bits in x5 encode presence of each power-of-two chunk.
    // bit8=256B, bit7=128B, bit6=64B, bit5=32B.
    tbz x5, #8, cleanup_128
    // 256B chunk (8 ldp/stnp pairs)
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

cleanup_128:                  // Optional 128B chunk (4 ldp + 4 stnp)
    tbz x5, #7, cleanup_64
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

cleanup_64:                   // Optional 64B chunk (2 ldp + 2 stnp)
    tbz x5, #6, cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x6, x6, #64
    add x7, x7, #64
    sub x5, x5, #64

cleanup_32:                   // Optional 32B chunk (1 ldp + 1 stnp)
    tbz x5, #5, cleanup_16
    ldp q0, q1, [x6, #0]
    stnp q0, q1, [x7, #0]
    add x6, x6, #32
    add x7, x7, #32
    sub x5, x5, #32

cleanup_16:                   // Optional 16B chunk
    tbz x5, #4, cleanup_8
    ldr q0, [x6], #16
    str q0, [x7], #16
    sub x5, x5, #16

cleanup_8:                    // Optional 8B chunk
    tbz x5, #3, cleanup_4
    ldr x8, [x6], #8
    str x8, [x7], #8
    sub x5, x5, #8

cleanup_4:                    // Optional 4B chunk
    tbz x5, #2, cleanup_2
    ldr w8, [x6], #4
    str w8, [x7], #4
    sub x5, x5, #4

cleanup_2:                    // Optional 2B chunk
    tbz x5, #1, cleanup_1
    ldrh w8, [x6], #2
    strh w8, [x7], #2
    sub x5, x5, #2

cleanup_1:                    // Optional final 1B chunk
    tbz x5, #0, copy_loop_end
    ldrb w8, [x6]
    strb w8, [x7]

copy_loop_end:              // Return to caller
    ret                     // Return
