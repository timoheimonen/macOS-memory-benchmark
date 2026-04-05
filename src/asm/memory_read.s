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
// memory_read_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
// Purpose:
//   Read 'byteCount' bytes from 'src' to exercise memory read bandwidth.
//   Accumulates an XOR checksum across all loaded data (and byte tail) to keep
//   the loads architecturally visible and prevent dead‑code elimination.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t)
// Returns:
//   x0 = 64‑bit XOR checksum
// Clobbers:
//   x2‑x7, x12‑x13, q0‑q7, q16‑q31 (data + accumulators, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * 512B main loop with pointer-bump addressing (no per-iteration offset math).
//   * Distributes XOR into four accumulators (v0‑v3) to reduce dependency depth.
//   * Tail path mirrors tiered size reductions (256/128/64/32/bytes).
// -----------------------------------------------------------------------------

.global _memory_read_loop_asm
.align 4
_memory_read_loop_asm:
    mov x6, x0              // src_ptr = src
    mov x5, x1              // remaining = byteCount
    mov x12, xzr            // Zero byte cleanup checksum accumulator

    // Zero accumulators (v0-v3) using XOR self-operation.
    // Use caller-saved registers q0-q7,q16-q31 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    eor v0.16b, v0.16b, v0.16b   // Zero accumulator 0 (caller-saved, safe)
    eor v1.16b, v1.16b, v1.16b   // Zero accumulator 1 (caller-saved, safe)
    eor v2.16b, v2.16b, v2.16b   // Zero accumulator 2 (caller-saved, safe)
    eor v3.16b, v3.16b, v3.16b   // Zero accumulator 3 (caller-saved, safe)

read_loop_start_512:        // Main 512B block loop
    cmp x5, #512            // remaining < 512?
    b.lo read_loop_cleanup  // If less, handle remaining bytes

    // Load 512 bytes from source (16 * 32B = 512B) using pair loads.
    // Using only caller-saved registers: q0-q7 and q16-q31 (avoiding q8-q15 per AAPCS64).
    // Accumulators are v0-v3 (q0-q3), so we load data into q4-q7 and q16-q31 first.
    // Process in two chunks: first 8 pairs, then next 8 pairs.

    // Load first 8 pairs (0-7) into q4-q7 and q16-q23
    ldp q4,  q5,  [x6, #0]        // Load pair 0 (offset 0)
    ldp q6,  q7,  [x6, #32]       // Load pair 1 (offset 32)
    ldp q16, q17, [x6, #64]       // Load pair 2 (offset 64)
    ldp q18, q19, [x6, #96]       // Load pair 3 (offset 96)
    ldp q20, q21, [x6, #128]      // Load pair 4 (offset 128)
    ldp q22, q23, [x6, #160]      // Load pair 5 (offset 160)
    ldp q24, q25, [x6, #192]      // Load pair 6 (offset 192)
    ldp q26, q27, [x6, #224]      // Load pair 7 (offset 224)

    // Accumulate first 8 pairs into v0-v3.
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

    // Load next 8 pairs (8-15) into q4-q7 and q16-q31
    ldp q4,  q5,  [x6, #256]
    ldp q6,  q7,  [x6, #288]
    ldp q16, q17, [x6, #320]
    ldp q18, q19, [x6, #352]
    ldp q20, q21, [x6, #384]
    ldp q22, q23, [x6, #416]
    ldp q24, q25, [x6, #448]
    ldp q26, q27, [x6, #480]

    // Accumulate next 8 pairs into v0-v3
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

    add x6, x6, #512        // src_ptr += 512
    sub x5, x5, #512        // remaining -= 512
    b read_loop_start_512

read_loop_cleanup:          // Tail handling when <512B remain
    cbz x5, read_loop_combine_sum // If no bytes remain, combine checksums

    // 256B chunk (8 pair loads + XOR fold)
    cmp x5, #256
    b.lo read_cleanup_128
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

read_cleanup_128:              // 128B chunk
    cmp x5, #128
    b.lo read_cleanup_64
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

read_cleanup_64:               // 64B chunk
    cmp x5, #64
    b.lo read_cleanup_32
    ldp q4, q5, [x6, #0]
    ldp q6, q7, [x6, #32]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    add x6, x6, #64
    sub x5, x5, #64

read_cleanup_32:               // 32B chunk
    cmp x5, #32
    b.lo read_cleanup_byte
    ldp q4, q5, [x6, #0]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    add x6, x6, #32
    sub x5, x5, #32

read_cleanup_byte:             // Byte tail (<32B)
    cbz x5, read_loop_combine_sum
    ldrb w13, [x6], #1
    eor x12, x12, x13
    subs x5, x5, #1
    b.ne read_cleanup_byte

read_loop_combine_sum:         // Final reduction + result write-back
    eor v0.16b, v0.16b, v1.16b
    eor v2.16b, v2.16b, v3.16b
    eor v0.16b, v0.16b, v2.16b

    umov x0, v0.d[0]
    eor x0, x0, x12

    ret                         // Return checksum in x0
