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
//   x3‑x7, q0‑q7 (zero vectors, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Pointer-bump addressing (no per-iteration offset math).
//   * Zero vectors materialized once (8 regs), reused for all 16 stnp pairs.
//   * Tiered tail mirrors copy/read for consistency.
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

    // Align hot loop entry to 64B so the unrolled 512B body always lands on a
    // predictable I-cache line. Reduces first-iteration fetch-boundary jitter.
    .p2align 6
write_loop_start_nt512:     // Main 512B block loop
    cmp x5, #512            // remaining < 512?
    b.lo write_loop_cleanup

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
    sub x5, x5, #512        // remaining -= 512
    b write_loop_start_nt512

write_loop_cleanup:         // Tail handling when <512B remain
    cbz x5, write_loop_end  // If none remain, exit

    // Handle 256B chunks (8 stnp pairs)
    cmp x5, #256
    b.lo write_cleanup_128
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

write_cleanup_128:            // 128B chunk
    cmp x5, #128
    b.lo write_cleanup_64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_cleanup_64:             // 64B chunk
    cmp x5, #64
    b.lo write_cleanup_32
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_cleanup_32:             // 32B chunk
    cmp x5, #32
    b.lo write_cleanup_byte
    stnp q0, q1, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_cleanup_byte:            // Byte tail (<32B)
    // Final byte-by-byte write for <32B remainder. Expected to be rare in
    // bandwidth benchmarks but ensures correctness for any input size.
    cbz x5, write_loop_end
    strb wzr, [x7], #1
    subs x5, x5, #1
    b.ne write_cleanup_byte

write_loop_end:             // Return to caller
    ret                     // Return
