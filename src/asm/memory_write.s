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
// memory_write_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_loop_asm(void* dst, size_t byteCount);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' using 512B block stores
//   to measure raw write bandwidth. Uses STNP as a non‑temporal hint.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x2‑x7, q0‑q7 (zero vectors + block counter, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Pointer-bump addressing (no per-iteration offset math).
//   * Zero vectors materialized once (8 regs), reused for all 16 stnp pairs.
//   * Main loop uses a precomputed block count (x2) and counts down with
//     `subs + b.ne` at the bottom (3 control ops/iter) instead of the older
//     `cmp + b.lo` top guard plus `sub + b` unconditional back-edge
//     (5 control ops/iter). x5 holds tail bytes (remaining % 512) during the
//     block loop and feeds the tbz tail tiers afterwards.
//   * Tail uses `tbz` bit-tests for 256/128/64/32 tiers plus scalar 16/8/4/2/1
//     stores, harmonized with memory_write_cache.s. No byte-by-byte loop.
//   * Main loop label is 64-byte aligned to keep the unrolled body on a single
//     I-cache line boundary for steady run-to-run timing on Apple Silicon.
// Timing Contract:
//   Caller must emit `dsb ish; isb` before reading the start-of-measurement
//   timestamp and another `dsb ish; isb` before reading the end-of-measurement
//   timestamp. This kernel emits no internal fences; barrier discipline is the
//   caller's responsibility for reproducible timing.
// -----------------------------------------------------------------------------

.global _memory_write_loop_asm
.align 4
_memory_write_loop_asm:
    mov x7, x0              // dst_ptr = dst
    mov x5, x1              // remaining = byteCount

    // Zero out 8 data registers (caller-saved: v0-v7, avoiding v8-v15 per AAPCS64).
    // All 16 stnp pairs reuse these 8 registers (zeros are identical).
    movi v0.16b, #0
    movi v1.16b, #0
    movi v2.16b, #0
    movi v3.16b, #0
    movi v4.16b, #0
    movi v5.16b, #0
    movi v6.16b, #0
    movi v7.16b, #0

    // Split remaining byteCount into full 512B blocks (x2) and tail bytes (x5).
    // The hot loop then becomes a counted subs+b.ne (3 control ops per 512B
    // iter) rather than a remainder compare+subtract+unconditional-branch
    // (5 control ops per iter). x5 carries the residual into the tail tiers.
    lsr x2, x5, #9               // x2 = full 512B block count (byteCount / 512)
    and x5, x5, #0x1ff           // x5 = tail bytes (byteCount % 512)
    cbz x2, write_loop_cleanup   // No full block? Go straight to tail.

    // Align hot loop entry to 64B so the unrolled 512B body always lands on a
    // predictable I-cache line. Reduces first-iteration fetch-boundary jitter.
    .p2align 6
write_loop_start_nt512:     // Main 512B block loop (count-down on x2)
    // Store 512B zeros (non-temporal) (16 stnp pairs).
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
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

    add x7, x7, #512        // dst_ptr += 512
    subs x2, x2, #1         // block_count -= 1, set flags
    b.ne write_loop_start_nt512 // Loop while blocks remain

write_loop_cleanup:         // Tail handling when <512B remain (x5 = tail bytes)
    cbz x5, write_loop_end  // If none remain, exit

    // Tiered tail: bits in x5 encode presence of each power-of-two chunk.
    // bit8=256B, bit7=128B, bit6=64B, bit5=32B.
    tbz x5, #8, write_cleanup_128
    // 256B chunk (8 stnp pairs)
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

write_cleanup_128:            // Optional 128B chunk
    tbz x5, #7, write_cleanup_64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_cleanup_64:             // Optional 64B chunk
    tbz x5, #6, write_cleanup_32
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_cleanup_32:             // Optional 32B chunk
    tbz x5, #5, write_cleanup_16
    stnp q0, q1, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_cleanup_16:             // Optional 16B scalar zero store
    tbz x5, #4, write_cleanup_8
    str q0, [x7], #16
    sub x5, x5, #16

write_cleanup_8:              // Optional 8B scalar zero store
    tbz x5, #3, write_cleanup_4
    str xzr, [x7], #8
    sub x5, x5, #8

write_cleanup_4:              // Optional 4B scalar zero store
    tbz x5, #2, write_cleanup_2
    str wzr, [x7], #4
    sub x5, x5, #4

write_cleanup_2:              // Optional 2B scalar zero store
    tbz x5, #1, write_cleanup_1
    strh wzr, [x7], #2
    sub x5, x5, #2

write_cleanup_1:              // Optional final 1B zero store
    tbz x5, #0, write_loop_end
    strb wzr, [x7]

write_loop_end:             // Return to caller
    ret                     // Return
