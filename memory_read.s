// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
//   * 512B main loop mirrors copy routine structure.
//   * Distributes XOR into four accumulators (v0‑v3) to reduce dependency depth.
//   * Tail path mirrors tiered size reductions (256/128/64/32/bytes).
// -----------------------------------------------------------------------------

.global _memory_read_loop_asm
.align 4
_memory_read_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes
    mov x12, xzr            // Zero byte cleanup checksum accumulator

    // Zero accumulators (v0-v3) using XOR self-operation (caller-saved, safe to use)
    eor v0.16b, v0.16b, v0.16b   // Zero accumulator 0
    eor v1.16b, v1.16b, v1.16b   // Zero accumulator 1
    eor v2.16b, v2.16b, v2.16b   // Zero accumulator 2
    eor v3.16b, v3.16b, v3.16b   // Zero accumulator 3

read_loop_start_512:        // Main 512B block loop
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt read_loop_cleanup  // If less, handle remaining bytes

    add x6, x0, x3          // src_addr = base + offset

    // Load 512 bytes from source (16 * 32B = 512B) using pair loads
    // Using only caller-saved registers: q0-q7 and q16-q31 (avoiding q8-q15 per AAPCS64)
    // Accumulators are v0-v3 (q0-q3), so we load data into q4-q7 and q16-q31 first
    // Process in two chunks: first 8 pairs, then next 8 pairs
    
    // Load first 8 pairs (0-7) into q4-q7 and q16-q23
    ldp q4,  q5,  [x6, #0]        // Load pair 0 (offset 0)
    ldp q6,  q7,  [x6, #32]       // Load pair 1 (offset 32)
    ldp q16, q17, [x6, #64]       // Load pair 2 (offset 64)
    ldp q18, q19, [x6, #96]       // Load pair 3 (offset 96)
    ldp q20, q21, [x6, #128]      // Load pair 4 (offset 128)
    ldp q22, q23, [x6, #160]      // Load pair 5 (offset 160)
    ldp q24, q25, [x6, #192]      // Load pair 6 (offset 192)
    ldp q26, q27, [x6, #224]      // Load pair 7 (offset 224)
    
    // Accumulate first 8 pairs into v0-v3
    eor v0.16b, v0.16b, v4.16b    // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b    // Accumulate q5 into v1
    eor v2.16b, v2.16b, v6.16b    // Accumulate q6 into v2
    eor v3.16b, v3.16b, v7.16b    // Accumulate q7 into v3
    eor v0.16b, v0.16b, v16.16b   // Accumulate q16 into v0
    eor v1.16b, v1.16b, v17.16b   // Accumulate q17 into v1
    eor v2.16b, v2.16b, v18.16b   // Accumulate q18 into v2
    eor v3.16b, v3.16b, v19.16b   // Accumulate q19 into v3
    eor v0.16b, v0.16b, v20.16b   // Accumulate q20 into v0
    eor v1.16b, v1.16b, v21.16b   // Accumulate q21 into v1
    eor v2.16b, v2.16b, v22.16b   // Accumulate q22 into v2
    eor v3.16b, v3.16b, v23.16b   // Accumulate q23 into v3
    eor v0.16b, v0.16b, v24.16b   // Accumulate q24 into v0
    eor v1.16b, v1.16b, v25.16b   // Accumulate q25 into v1
    eor v2.16b, v2.16b, v26.16b   // Accumulate q26 into v2
    eor v3.16b, v3.16b, v27.16b   // Accumulate q27 into v3
    
    // Load next 8 pairs (8-15) into q4-q7 and q16-q31
    ldp q4,  q5,  [x6, #256]      // Load pair 8 (offset 256, reuse q4-q5)
    ldp q6,  q7,  [x6, #288]      // Load pair 9 (offset 288, reuse q6-q7)
    ldp q16, q17, [x6, #320]      // Load pair 10 (offset 320, reuse q16-q17)
    ldp q18, q19, [x6, #352]      // Load pair 11 (offset 352, reuse q18-q19)
    ldp q20, q21, [x6, #384]      // Load pair 12 (offset 384, reuse q20-q21)
    ldp q22, q23, [x6, #416]      // Load pair 13 (offset 416, reuse q22-q23)
    ldp q24, q25, [x6, #448]      // Load pair 14 (offset 448, reuse q24-q25)
    ldp q26, q27, [x6, #480]      // Load pair 15 (offset 480, reuse q26-q27)
    
    // Accumulate next 8 pairs into v0-v3
    eor v0.16b, v0.16b, v4.16b    // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b    // Accumulate q5 into v1
    eor v2.16b, v2.16b, v6.16b    // Accumulate q6 into v2
    eor v3.16b, v3.16b, v7.16b    // Accumulate q7 into v3
    eor v0.16b, v0.16b, v16.16b   // Accumulate q16 into v0
    eor v1.16b, v1.16b, v17.16b   // Accumulate q17 into v1
    eor v2.16b, v2.16b, v18.16b   // Accumulate q18 into v2
    eor v3.16b, v3.16b, v19.16b   // Accumulate q19 into v3
    eor v0.16b, v0.16b, v20.16b   // Accumulate q20 into v0
    eor v1.16b, v1.16b, v21.16b   // Accumulate q21 into v1
    eor v2.16b, v2.16b, v22.16b   // Accumulate q22 into v2
    eor v3.16b, v3.16b, v23.16b   // Accumulate q23 into v3
    eor v0.16b, v0.16b, v24.16b   // Accumulate q24 into v0
    eor v1.16b, v1.16b, v25.16b   // Accumulate q25 into v1
    eor v2.16b, v2.16b, v26.16b   // Accumulate q26 into v2
    eor v3.16b, v3.16b, v27.16b   // Accumulate q27 into v3

    add x3, x3, x4          // offset += step
    b read_loop_start_512   // Loop again

read_loop_cleanup:          // Tail handling when <512B remain
    cmp x3, x1              // offset == count?
    b.ge read_loop_combine_sum // If done, combine sums

    subs x5, x1, x3         // Recalc remaining (x5)
    add x6, x0, x3          // src_addr = src + offset

    // 256B chunk (8 pair loads + XOR fold)
    cmp x5, #256              // Check if >= 256 bytes remain
    b.lt read_cleanup_128     // If less, handle smaller chunks
    ldp q4, q5, [x6, #0]      // Load first pair (32B)
    ldp q6, q7, [x6, #32]     // Load second pair (32B)
    ldp q16, q17, [x6, #64]   // Load third pair (32B)
    ldp q18, q19, [x6, #96]   // Load fourth pair (32B)
    ldp q20, q21, [x6, #128]  // Load fifth pair (32B)
    ldp q22, q23, [x6, #160]  // Load sixth pair (32B)
    ldp q24, q25, [x6, #192]  // Load seventh pair (32B)
    ldp q26, q27, [x6, #224]  // Load eighth pair (32B)
    // XOR fold into accumulators (v0-v3)
    eor v0.16b, v0.16b, v4.16b    // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b    // Accumulate q5 into v1
    eor v2.16b, v2.16b, v6.16b    // Accumulate q6 into v2
    eor v3.16b, v3.16b, v7.16b    // Accumulate q7 into v3
    eor v0.16b, v0.16b, v16.16b   // Accumulate q16 into v0
    eor v1.16b, v1.16b, v17.16b   // Accumulate q17 into v1
    eor v2.16b, v2.16b, v18.16b   // Accumulate q18 into v2
    eor v3.16b, v3.16b, v19.16b   // Accumulate q19 into v3
    eor v0.16b, v0.16b, v20.16b   // Accumulate q20 into v0
    eor v1.16b, v1.16b, v21.16b   // Accumulate q21 into v1
    eor v2.16b, v2.16b, v22.16b   // Accumulate q22 into v2
    eor v3.16b, v3.16b, v23.16b   // Accumulate q23 into v3
    eor v0.16b, v0.16b, v24.16b   // Accumulate q24 into v0
    eor v1.16b, v1.16b, v25.16b   // Accumulate q25 into v1
    eor v2.16b, v2.16b, v26.16b   // Accumulate q26 into v2
    eor v3.16b, v3.16b, v27.16b   // Accumulate q27 into v3
    add x6, x6, #256          // Advance source pointer by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

read_cleanup_128:              // 128B chunk
    cmp x5, #128               // Check if >= 128 bytes remain
    b.lt read_cleanup_64        // If less, handle smaller chunks
    ldp q4, q5, [x6, #0]        // Load first pair (32B)
    ldp q6, q7, [x6, #32]       // Load second pair (32B)
    ldp q16, q17, [x6, #64]     // Load third pair (32B)
    ldp q18, q19, [x6, #96]     // Load fourth pair (32B)
    eor v0.16b, v0.16b, v4.16b  // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b  // Accumulate q5 into v1
    eor v2.16b, v2.16b, v6.16b  // Accumulate q6 into v2
    eor v3.16b, v3.16b, v7.16b  // Accumulate q7 into v3
    eor v0.16b, v0.16b, v16.16b // Accumulate q16 into v0
    eor v1.16b, v1.16b, v17.16b // Accumulate q17 into v1
    eor v2.16b, v2.16b, v18.16b // Accumulate q18 into v2
    eor v3.16b, v3.16b, v19.16b // Accumulate q19 into v3
    add x6, x6, #128            // Advance source pointer by 128B
    sub x5, x5, #128            // Decrement remaining count by 128B

read_cleanup_64:               // 64B chunk
    cmp x5, #64                // Check if >= 64 bytes remain
    b.lt read_cleanup_32        // If less, handle smaller chunks
    ldp q4, q5, [x6, #0]       // Load first pair (32B)
    ldp q6, q7, [x6, #32]      // Load second pair (32B)
    eor v0.16b, v0.16b, v4.16b // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b // Accumulate q5 into v1
    eor v2.16b, v2.16b, v6.16b // Accumulate q6 into v2
    eor v3.16b, v3.16b, v7.16b // Accumulate q7 into v3
    add x6, x6, #64            // Advance source pointer by 64B
    sub x5, x5, #64            // Decrement remaining count by 64B

read_cleanup_32:               // 32B chunk
    cmp x5, #32                // Check if >= 32 bytes remain
    b.lt read_cleanup_byte      // If less, handle byte tail
    ldp q4, q5, [x6, #0]       // Load pair (32B)
    eor v0.16b, v0.16b, v4.16b  // Accumulate q4 into v0
    eor v1.16b, v1.16b, v5.16b  // Accumulate q5 into v1
    add x6, x6, #32            // Advance source pointer by 32B
    sub x5, x5, #32            // Decrement remaining count by 32B

read_cleanup_byte:             // Byte tail (<32B)
    cmp x5, #0                 // Check if any bytes remain
    b.le read_loop_combine_sum // If none, combine checksums
    ldrb w13, [x6], #1         // Load byte, post-increment source
    eor x12, x12, x13          // XOR byte into accumulator
    subs x5, x5, #1            // Decrement remaining count
    b.gt read_cleanup_byte     // Loop if more bytes remain

read_loop_combine_sum:         // Final reduction + result write-back
    // Combine accumulators v0-v3 -> v0
    eor v0.16b, v0.16b, v1.16b  // Combine v0 and v1 into v0
    eor v2.16b, v2.16b, v3.16b  // Combine v2 and v3 into v2
    eor v0.16b, v0.16b, v2.16b  // Combine v0 and v2 into v0 (final vector)

    // Combine byte checksum (x12) into final result (x0 from v0)
    umov x0, v0.d[0]            // Extract lower 64 bits of combined vector sum
    eor x0, x0, x12             // Combine with byte checksum accumulator

    ret                         // Return checksum in x0