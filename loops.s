// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)
// Provides hot inner loops for:
//   - Memory copy (bandwidth) using 512‑byte vector blocks
//   - Memory read (bandwidth) with XOR reduction to keep loads live
//   - Memory write (bandwidth) writing zeros using non‑temporal (STNP) hint
//   - Memory latency (pointer chasing) with an 8x unrolled dependent chain
//
// Design Goals:
//   * Saturate memory subsystem while minimizing loop overhead
//   * Use large (512B) blocks for steady‑state bandwidth, tiered cleanup for tail
//   * Keep a simple, predictable register footprint (no spills / stack usage)
//   * Avoid introducing control dependencies that reduce ILP (besides latency test)
//
// Notes:
//   * STNP is used as a non‑temporal hint; actual behavior is micro‑architecture dependent.
//   * All routines assume pointers are valid and properly aligned for at least 16‑byte accesses.
//   * No explicit prefetching is performed; modern Apple cores prefetch aggressively.
//   * No stack frame / callee‑saved registers used (leaf functions).
//   * These functions are intended to be called from C++ with matching prototypes.
//
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
//   x3‑x8 (temporaries), q0‑q31 (data vectors)
// Assumptions / Guarantees:
//   * Undefined behavior if regions overlap (not a memmove replacement).
//   * Tail handled with progressively smaller vector tiers then byte loop (<32B).
// -----------------------------------------------------------------------------
.global _memory_copy_loop_asm
.align 4
_memory_copy_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes

copy_loop_start_nt512:      // Main 512B block loop
    subs x5, x2, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt copy_loop_cleanup  // If less, handle remaining bytes

    // Calculate current addresses
    add x6, x1, x3          // src_addr = base + offset
    add x7, x0, x3          // dst_addr = base + offset

    // Load 512 bytes from source (16 ldp pairs, 32B each)
    ldp q0,  q1,  [x6, #0]      // Load pair 0 (offset 0)
    ldp q2,  q3,  [x6, #32]     // Load pair 1 (offset 32)
    ldp q4,  q5,  [x6, #64]     // Load pair 2 (offset 64)
    ldp q6,  q7,  [x6, #96]     // Load pair 3 (offset 96)
    ldp q8,  q9,  [x6, #128]    // Load pair 4 (offset 128)
    ldp q10, q11, [x6, #160]    // Load pair 5 (offset 160)
    ldp q12, q13, [x6, #192]    // Load pair 6 (offset 192)
    ldp q14, q15, [x6, #224]    // Load pair 7 (offset 224)
    ldp q16, q17, [x6, #256]    // Load pair 8 (offset 256)
    ldp q18, q19, [x6, #288]    // Load pair 9 (offset 288)
    ldp q20, q21, [x6, #320]    // Load pair 10 (offset 320)
    ldp q22, q23, [x6, #352]    // Load pair 11 (offset 352)
    ldp q24, q25, [x6, #384]    // Load pair 12 (offset 384)
    ldp q26, q27, [x6, #416]    // Load pair 13 (offset 416)
    ldp q28, q29, [x6, #448]    // Load pair 14 (offset 448)
    ldp q30, q31, [x6, #480]    // Load pair 15 (offset 480)


    // Store 512 bytes (non-temporal) - minimize cache pollution (16 stnp pairs)
    stnp q0,  q1,  [x7, #0]      // Store pair 0 (offset 0, non-temporal)
    stnp q2,  q3,  [x7, #32]     // Store pair 1 (offset 32, non-temporal)
    stnp q4,  q5,  [x7, #64]     // Store pair 2 (offset 64, non-temporal)
    stnp q6,  q7,  [x7, #96]     // Store pair 3 (offset 96, non-temporal)
    stnp q8,  q9,  [x7, #128]    // Store pair 4 (offset 128, non-temporal)
    stnp q10, q11, [x7, #160]    // Store pair 5 (offset 160, non-temporal)
    stnp q12, q13, [x7, #192]    // Store pair 6 (offset 192, non-temporal)
    stnp q14, q15, [x7, #224]    // Store pair 7 (offset 224, non-temporal)
    stnp q16, q17, [x7, #256]    // Store pair 8 (offset 256, non-temporal)
    stnp q18, q19, [x7, #288]    // Store pair 9 (offset 288, non-temporal)
    stnp q20, q21, [x7, #320]    // Store pair 10 (offset 320, non-temporal)
    stnp q22, q23, [x7, #352]    // Store pair 11 (offset 352, non-temporal)
    stnp q24, q25, [x7, #384]    // Store pair 12 (offset 384, non-temporal)
    stnp q26, q27, [x7, #416]    // Store pair 13 (offset 416, non-temporal)
    stnp q28, q29, [x7, #448]    // Store pair 14 (offset 448, non-temporal)
    stnp q30, q31, [x7, #480]    // Store pair 15 (offset 480, non-temporal)

    add x3, x3, x4          // offset += step
    b copy_loop_start_nt512 // Loop again

copy_loop_cleanup:          // Tail handling when <512B remain
    cmp x3, x2              // offset == count?
    b.ge copy_loop_end      // If done, exit

    subs x5, x2, x3         // Recalc remaining (x5)
    add x6, x1, x3          // src_addr = src + offset
    add x7, x0, x3          // dst_addr = dst + offset

    // Handle 256B chunks (8 ldp/stnp pairs)
    cmp x5, #256              // Check if >= 256 bytes remain
    b.lt cleanup_128          // If less, handle smaller chunks
    ldp q0, q1, [x6, #0]      // Load first pair (32B)
    ldp q2, q3, [x6, #32]     // Load second pair (32B)
    ldp q4, q5, [x6, #64]     // Load third pair (32B)
    ldp q6, q7, [x6, #96]     // Load fourth pair (32B)
    ldp q8, q9, [x6, #128]    // Load fifth pair (32B)
    ldp q10, q11, [x6, #160]  // Load sixth pair (32B)
    ldp q12, q13, [x6, #192]  // Load seventh pair (32B)
    ldp q14, q15, [x6, #224]  // Load eighth pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal)
    stnp q8, q9, [x7, #128]   // Store fifth pair (non-temporal)
    stnp q10, q11, [x7, #160] // Store sixth pair (non-temporal)
    stnp q12, q13, [x7, #192] // Store seventh pair (non-temporal)
    stnp q14, q15, [x7, #224] // Store eighth pair (non-temporal)
    add x6, x6, #256          // Advance source pointer by 256B
    add x7, x7, #256          // Advance destination pointer by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

cleanup_128:                  // Handle 128B chunks (4 ldp + 4 stnp)
    cmp x5, #128              // Check if >= 128 bytes remain
    b.lt cleanup_64           // If less, handle smaller chunks
    ldp q0, q1, [x6, #0]      // Load first pair (32B)
    ldp q2, q3, [x6, #32]     // Load second pair (32B)
    ldp q4, q5, [x6, #64]     // Load third pair (32B)
    ldp q6, q7, [x6, #96]     // Load fourth pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal)
    add x6, x6, #128          // Advance source pointer by 128B
    add x7, x7, #128          // Advance destination pointer by 128B
    sub x5, x5, #128          // Decrement remaining count by 128B

cleanup_64:                   // Handle 64B chunks (2 ldp + 2 stnp)
    cmp x5, #64               // Check if >= 64 bytes remain
    b.lt cleanup_32           // If less, handle smaller chunks
    ldp q0, q1, [x6, #0]      // Load first pair (32B)
    ldp q2, q3, [x6, #32]     // Load second pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    add x6, x6, #64           // Advance source pointer by 64B
    add x7, x7, #64           // Advance destination pointer by 64B
    sub x5, x5, #64           // Decrement remaining count by 64B

cleanup_32:                   // Handle 32B chunk (1 ldp + 1 stnp)
    cmp x5, #32               // Check if >= 32 bytes remain
    b.lt copy_cleanup_byte    // If less, handle byte tail
    ldp q0, q1, [x6, #0]      // Load pair (32B)
    stnp q0, q1, [x7, #0]     // Store pair (non-temporal)
    add x6, x6, #32           // Advance source pointer by 32B
    add x7, x7, #32           // Advance destination pointer by 32B
    sub x5, x5, #32           // Decrement remaining count by 32B

copy_cleanup_byte:           // Byte tail for final <32B
    cmp x5, #0               // Check if any bytes remain
    b.le copy_loop_end        // If none, exit
    ldrb w8, [x6], #1        // Load byte, post-increment source
    strb w8, [x7], #1        // Store byte, post-increment destination
    subs x5, x5, #1          // Decrement remaining count
    b.gt copy_cleanup_byte    // Loop if more bytes remain

copy_loop_end:              // Return to caller
    ret                     // Return

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
//   x2‑x7, x12‑x13, q0‑q31 (data + accumulators)
// Implementation Notes:
//   * 512B main loop mirrors copy routine structure.
//   * Distributes XOR into four accumulators (v8‑v11) to reduce dependency depth.
//   * Tail path mirrors tiered size reductions (256/128/64/32/bytes).
// -----------------------------------------------------------------------------

.global _memory_read_loop_asm
.align 4
_memory_read_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes
    mov x12, xzr            // Zero byte cleanup checksum accumulator

    // Zero accumulators (v8-v11) using XOR self-operation
    eor v8.16b, v8.16b, v8.16b   // Zero accumulator 0
    eor v9.16b, v9.16b, v9.16b   // Zero accumulator 1
    eor v10.16b, v10.16b, v10.16b // Zero accumulator 2
    eor v11.16b, v11.16b, v11.16b // Zero accumulator 3

read_loop_start_512:        // Main 512B block loop
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt read_loop_cleanup  // If less, handle remaining bytes

    add x6, x0, x3          // src_addr = base + offset

    // Load 512 bytes from source (16 * 32B = 512B) using pair loads
    ldp q0,  q1,  [x6, #0]        // Load pair 0 (offset 0)
    ldp q2,  q3,  [x6, #32]       // Load pair 1 (offset 32)
    ldp q4,  q5,  [x6, #64]       // Load pair 2 (offset 64)
    ldp q6,  q7,  [x6, #96]       // Load pair 3 (offset 96)
    // Store loaded values from q8-q11 temporarily in v24-v27 (these are available)
    ldp q24, q25, [x6, #128]      // Load pair 4 to temps (offset 128, avoid overwriting accumulators)
    ldp q26, q27, [x6, #160]      // Load pair 5 to temps (offset 160, avoid overwriting accumulators)
    ldp q12, q13, [x6, #192]      // Load pair 6 (offset 192)
    ldp q14, q15, [x6, #224]      // Load pair 7 (offset 224)
    ldp q16, q17, [x6, #256]      // Load pair 8 (offset 256)
    ldp q18, q19, [x6, #288]      // Load pair 9 (offset 288)
    ldp q20, q21, [x6, #320]      // Load pair 10 (offset 320)
    ldp q22, q23, [x6, #352]      // Load pair 11 (offset 352)
    // Repurpose upper registers that we already processed
    ldp q28, q29, [x6, #384]      // Load pair 12 (offset 384)
    ldp q30, q31, [x6, #416]      // Load pair 13 (offset 416)
    // Last two loads directly to free scratch registers (reuse q0-q3 after accumulation)
    ldp q0,  q1,  [x6, #448]      // Load pair 14 (offset 448, reuse q0-q1)
    ldp q2,  q3,  [x6, #480]      // Load pair 15 (offset 480, reuse q2-q3)

    // XOR fold loaded vectors into 4 accumulators (v8-v11)
    // First 8 vectors (q0-q7) into accumulators
    eor v8.16b,  v8.16b,  v0.16b    // Accumulate q0 into v8
    eor v9.16b,  v9.16b,  v1.16b    // Accumulate q1 into v9
    eor v10.16b, v10.16b, v2.16b    // Accumulate q2 into v10
    eor v11.16b, v11.16b, v3.16b    // Accumulate q3 into v11
    eor v8.16b,  v8.16b,  v4.16b    // Accumulate q4 into v8
    eor v9.16b,  v9.16b,  v5.16b    // Accumulate q5 into v9
    eor v10.16b, v10.16b, v6.16b    // Accumulate q6 into v10
    eor v11.16b, v11.16b, v7.16b    // Accumulate q7 into v11
    // Next 4 vectors (q24-q27, loaded to temps) into accumulators
    eor v8.16b,  v8.16b,  v24.16b   // Accumulate q24 into v8
    eor v9.16b,  v9.16b,  v25.16b   // Accumulate q25 into v9
    eor v10.16b, v10.16b, v26.16b   // Accumulate q26 into v10
    eor v11.16b, v11.16b, v27.16b   // Accumulate q27 into v11
    // Next 8 vectors (q12-q23) into accumulators
    eor v8.16b,  v8.16b,  v12.16b   // Accumulate q12 into v8
    eor v9.16b,  v9.16b,  v13.16b   // Accumulate q13 into v9
    eor v10.16b, v10.16b, v14.16b   // Accumulate q14 into v10
    eor v11.16b, v11.16b, v15.16b   // Accumulate q15 into v11
    eor v8.16b,  v8.16b,  v16.16b   // Accumulate q16 into v8
    eor v9.16b,  v9.16b,  v17.16b   // Accumulate q17 into v9
    eor v10.16b, v10.16b, v18.16b   // Accumulate q18 into v10
    eor v11.16b, v11.16b, v19.16b   // Accumulate q19 into v11
    eor v8.16b,  v8.16b,  v20.16b   // Accumulate q20 into v8
    eor v9.16b,  v9.16b,  v21.16b   // Accumulate q21 into v9
    eor v10.16b, v10.16b, v22.16b   // Accumulate q22 into v10
    eor v11.16b, v11.16b, v23.16b   // Accumulate q23 into v11
    // Next 4 vectors (q28-q31) into accumulators
    eor v8.16b,  v8.16b,  v28.16b   // Accumulate q28 into v8
    eor v9.16b,  v9.16b,  v29.16b   // Accumulate q29 into v9
    eor v10.16b, v10.16b, v30.16b   // Accumulate q30 into v10
    eor v11.16b, v11.16b, v31.16b   // Accumulate q31 into v11
    // Final 4 vectors (reused q0-q3 from second load) into accumulators
    eor v8.16b,  v8.16b,  v0.16b    // Accumulate reused q0 into v8
    eor v9.16b,  v9.16b,  v1.16b    // Accumulate reused q1 into v9
    eor v10.16b, v10.16b, v2.16b    // Accumulate reused q2 into v10
    eor v11.16b, v11.16b, v3.16b    // Accumulate reused q3 into v11

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
    ldp q0, q1, [x6, #0]      // Load first pair (32B)
    ldp q2, q3, [x6, #32]     // Load second pair (32B)
    ldp q4, q5, [x6, #64]     // Load third pair (32B)
    ldp q6, q7, [x6, #96]     // Load fourth pair (32B)
    ldp q16, q17, [x6, #128]  // Load fifth pair (32B) to temp registers
    ldp q18, q19, [x6, #160]  // Load sixth pair (32B) to temp registers
    ldp q12, q13, [x6, #192]  // Load seventh pair (32B)
    ldp q14, q15, [x6, #224]  // Load eighth pair (32B)
    // XOR fold into accumulators (use temp registers q16-q19 to avoid overwriting v8-v11)
    eor v8.16b, v8.16b, v0.16b    // Accumulate q0 into v8
    eor v9.16b, v9.16b, v1.16b    // Accumulate q1 into v9
    eor v10.16b, v10.16b, v2.16b  // Accumulate q2 into v10
    eor v11.16b, v11.16b, v3.16b  // Accumulate q3 into v11
    eor v8.16b, v8.16b, v4.16b    // Accumulate q4 into v8
    eor v9.16b, v9.16b, v5.16b    // Accumulate q5 into v9
    eor v10.16b, v10.16b, v6.16b  // Accumulate q6 into v10
    eor v11.16b, v11.16b, v7.16b  // Accumulate q7 into v11
    eor v8.16b, v8.16b, v16.16b   // Accumulate q16 into v8
    eor v9.16b, v9.16b, v17.16b   // Accumulate q17 into v9
    eor v10.16b, v10.16b, v18.16b // Accumulate q18 into v10
    eor v11.16b, v11.16b, v19.16b // Accumulate q19 into v11
    eor v8.16b, v8.16b, v12.16b   // Accumulate q12 into v8
    eor v9.16b, v9.16b, v13.16b   // Accumulate q13 into v9
    eor v10.16b, v10.16b, v14.16b // Accumulate q14 into v10
    eor v11.16b, v11.16b, v15.16b // Accumulate q15 into v11
    add x6, x6, #256          // Advance source pointer by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

read_cleanup_128:              // 128B chunk
    cmp x5, #128               // Check if >= 128 bytes remain
    b.lt read_cleanup_64        // If less, handle smaller chunks
    ldp q0, q1, [x6, #0]        // Load first pair (32B)
    ldp q2, q3, [x6, #32]       // Load second pair (32B)
    ldp q4, q5, [x6, #64]       // Load third pair (32B)
    ldp q6, q7, [x6, #96]       // Load fourth pair (32B)
    eor v8.16b, v8.16b, v0.16b  // Accumulate q0 into v8
    eor v9.16b, v9.16b, v1.16b  // Accumulate q1 into v9
    eor v10.16b, v10.16b, v2.16b // Accumulate q2 into v10
    eor v11.16b, v11.16b, v3.16b // Accumulate q3 into v11
    eor v8.16b, v8.16b, v4.16b  // Accumulate q4 into v8
    eor v9.16b, v9.16b, v5.16b  // Accumulate q5 into v9
    eor v10.16b, v10.16b, v6.16b // Accumulate q6 into v10
    eor v11.16b, v11.16b, v7.16b // Accumulate q7 into v11
    add x6, x6, #128            // Advance source pointer by 128B
    sub x5, x5, #128            // Decrement remaining count by 128B

read_cleanup_64:               // 64B chunk
    cmp x5, #64                // Check if >= 64 bytes remain
    b.lt read_cleanup_32        // If less, handle smaller chunks
    ldp q0, q1, [x6, #0]       // Load first pair (32B)
    ldp q2, q3, [x6, #32]      // Load second pair (32B)
    eor v8.16b, v8.16b, v0.16b // Accumulate q0 into v8
    eor v9.16b, v9.16b, v1.16b // Accumulate q1 into v9
    eor v10.16b, v10.16b, v2.16b // Accumulate q2 into v10
    eor v11.16b, v11.16b, v3.16b // Accumulate q3 into v11
    add x6, x6, #64            // Advance source pointer by 64B
    sub x5, x5, #64            // Decrement remaining count by 64B

read_cleanup_32:               // 32B chunk
    cmp x5, #32                // Check if >= 32 bytes remain
    b.lt read_cleanup_byte      // If less, handle byte tail
    ldp q0, q1, [x6, #0]       // Load pair (32B)
    eor v8.16b, v8.16b, v0.16b  // Accumulate q0 into v8
    eor v9.16b, v9.16b, v1.16b  // Accumulate q1 into v9
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
    // Combine accumulators v8-v11 -> v8
    eor v8.16b, v8.16b, v9.16b  // Combine v8 and v9 into v8
    eor v10.16b, v10.16b, v11.16b // Combine v10 and v11 into v10
    eor v8.16b, v8.16b, v10.16b // Combine v8 and v10 into v8 (final vector)

    // Combine byte checksum (x12) into final result (x0 from v8)
    umov x0, v8.d[0]            // Extract lower 64 bits of combined vector sum
    eor x0, x0, x12             // Combine with byte checksum accumulator

    ret                         // Return checksum in x0
    
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
//   x3‑x7, q0‑q31 (zero vectors)
// Implementation Notes:
//   * Zero vectors are materialized once (movi) then reused.
//   * Tiered tail mirrors copy/read for consistency.
// -----------------------------------------------------------------------------

.global _memory_write_loop_asm
.align 4
_memory_write_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes

    // Zero out data registers v0-v31 (full 128-bit vectors for consistent zero writes)
    movi v0.16b, #0             // Zero vector 0
    movi v1.16b, #0             // Zero vector 1
    movi v2.16b, #0             // Zero vector 2
    movi v3.16b, #0             // Zero vector 3
    movi v4.16b, #0             // Zero vector 4
    movi v5.16b, #0             // Zero vector 5
    movi v6.16b, #0             // Zero vector 6
    movi v7.16b, #0             // Zero vector 7
    movi v8.16b, #0             // Zero vector 8
    movi v9.16b, #0             // Zero vector 9
    movi v10.16b, #0            // Zero vector 10
    movi v11.16b, #0            // Zero vector 11
    movi v12.16b, #0            // Zero vector 12
    movi v13.16b, #0            // Zero vector 13
    movi v14.16b, #0            // Zero vector 14
    movi v15.16b, #0            // Zero vector 15
    movi v16.16b, #0            // Zero vector 16
    movi v17.16b, #0            // Zero vector 17
    movi v18.16b, #0            // Zero vector 18
    movi v19.16b, #0            // Zero vector 19
    movi v20.16b, #0            // Zero vector 20
    movi v21.16b, #0            // Zero vector 21
    movi v22.16b, #0            // Zero vector 22
    movi v23.16b, #0            // Zero vector 23
    movi v24.16b, #0            // Zero vector 24
    movi v25.16b, #0            // Zero vector 25
    movi v26.16b, #0            // Zero vector 26
    movi v27.16b, #0            // Zero vector 27
    movi v28.16b, #0            // Zero vector 28
    movi v29.16b, #0            // Zero vector 29
    movi v30.16b, #0            // Zero vector 30
    movi v31.16b, #0            // Zero vector 31


write_loop_start_nt512:     // Main 512B block loop
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt write_loop_cleanup // If less, handle remaining bytes

    add x7, x0, x3          // dst_addr = base + offset

    // Store 512B zeros (non-temporal) (16 stnp pairs)
    stnp q0,  q1,  [x7, #0]       // Store pair 0 (offset 0, non-temporal)
    stnp q2,  q3,  [x7, #32]      // Store pair 1 (offset 32, non-temporal)
    stnp q4,  q5,  [x7, #64]      // Store pair 2 (offset 64, non-temporal)
    stnp q6,  q7,  [x7, #96]      // Store pair 3 (offset 96, non-temporal)
    stnp q8,  q9,  [x7, #128]     // Store pair 4 (offset 128, non-temporal)
    stnp q10, q11, [x7, #160]     // Store pair 5 (offset 160, non-temporal)
    stnp q12, q13, [x7, #192]     // Store pair 6 (offset 192, non-temporal)
    stnp q14, q15, [x7, #224]     // Store pair 7 (offset 224, non-temporal)
    stnp q16, q17, [x7, #256]     // Store pair 8 (offset 256, non-temporal)
    stnp q18, q19, [x7, #288]     // Store pair 9 (offset 288, non-temporal)
    stnp q20, q21, [x7, #320]     // Store pair 10 (offset 320, non-temporal)
    stnp q22, q23, [x7, #352]     // Store pair 11 (offset 352, non-temporal)
    stnp q24, q25, [x7, #384]     // Store pair 12 (offset 384, non-temporal)
    stnp q26, q27, [x7, #416]     // Store pair 13 (offset 416, non-temporal)
    stnp q28, q29, [x7, #448]     // Store pair 14 (offset 448, non-temporal)
    stnp q30, q31, [x7, #480]     // Store pair 15 (offset 480, non-temporal)

    add x3, x3, x4          // offset += step
    b write_loop_start_nt512 // Loop again

write_loop_cleanup:         // Tail handling when <512B remain
    cmp x3, x1              // offset == count?
    b.ge write_loop_end     // If done, exit

    subs x5, x1, x3         // Recalc remaining (x5)
    add x7, x0, x3          // dst_addr = dst + offset

    // Handle 256B chunks (8 stnp pairs)
    cmp x5, #256              // Check if >= 256 bytes remain
    b.lt write_cleanup_128    // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal, 32B)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal, 32B)
    stnp q8, q9, [x7, #128]   // Store fifth pair (non-temporal, 32B)
    stnp q10, q11, [x7, #160] // Store sixth pair (non-temporal, 32B)
    stnp q12, q13, [x7, #192] // Store seventh pair (non-temporal, 32B)
    stnp q14, q15, [x7, #224] // Store eighth pair (non-temporal, 32B)
    add x7, x7, #256          // Advance destination pointer by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

write_cleanup_128:            // 128B chunk
    cmp x5, #128              // Check if >= 128 bytes remain
    b.lt write_cleanup_64      // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal, 32B)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal, 32B)
    add x7, x7, #128          // Advance destination pointer by 128B
    sub x5, x5, #128          // Decrement remaining count by 128B

write_cleanup_64:             // 64B chunk
    cmp x5, #64               // Check if >= 64 bytes remain
    b.lt write_cleanup_32      // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    add x7, x7, #64           // Advance destination pointer by 64B
    sub x5, x5, #64           // Decrement remaining count by 64B

write_cleanup_32:             // 32B chunk
    cmp x5, #32               // Check if >= 32 bytes remain
    b.lt write_cleanup_byte    // If less, handle byte tail
    stnp q0, q1, [x7, #0]     // Store pair (non-temporal, 32B)
    add x7, x7, #32           // Advance destination pointer by 32B
    sub x5, x5, #32           // Decrement remaining count by 32B

write_cleanup_byte:            // Byte tail (<32B)
    cmp x5, #0                // Check if any bytes remain
    b.le write_loop_end        // If none, exit
    strb wzr, [x7], #1         // Store zero byte, post-increment destination
    subs x5, x5, #1           // Decrement remaining count
    b.gt write_cleanup_byte    // Loop if more bytes remain

write_loop_end:             // Return to caller
    ret                     // Return

// -----------------------------------------------------------------------------
// memory_latency_chase_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Purpose:
//   Measure load‑to‑use latency via dependent pointer chasing. Each load feeds
//   the address of the next, forming a serialized chain. Loop is unrolled by 8
//   to reduce branch impact while preserving dependency.
// Arguments:
//   x0 = start_pointer (uintptr_t*)
//   x1 = count (number of pointer dereferences)
// Returns:
//   x0 = final pointer value (acts as a sink to prevent DCE)
// Clobbers:
//   x2‑x5 (temporaries), x4 reused for iteration counts
// Implementation Notes:
//   * Uses barriers (dsb/isb/dmb) to isolate measurement window and reduce
//     speculative side effects across the unrolled section.
//   * No prefetching: intentional to keep strictly serialized accesses.
// -----------------------------------------------------------------------------

.global _memory_latency_chase_asm
.align 4
_memory_latency_chase_asm:
    mov x2, x1              // Save count in x2 (unused, can be removed)
    mov x3, x0              // Preserve original pointer in x3 (unused, can be removed)
    
    // Pre-touch to ensure TLB entries are loaded
    ldr x4, [x0]                // Prime the first access but don't use result yet
    dsb ish                     // Data synchronization barrier (inner shareable)
    isb                         // Instruction synchronization barrier

    // Compute unrolled iterations and remainder for exact count handling
    lsr x4, x1, #3              // x4 = count / 8 (unrolled iterations)
    and x5, x1, #7              // x5 = count % 8 (remainder)

    cbz x4, latency_remainder   // Skip unrolled loop if no full groups

    dmb sy                      // Memory barrier to prevent reordering before measurement

latency_loop_unrolled:          // 8 dependent loads per iteration
    ldr x0, [x0]                // Load 1: x0 = *x0
    ldr x0, [x0]                // Load 2: x0 = *x0 (dependent on load 1)
    ldr x0, [x0]                // Load 3: x0 = *x0 (dependent on load 2)
    ldr x0, [x0]                // Load 4: x0 = *x0 (dependent on load 3)
    ldr x0, [x0]                // Load 5: x0 = *x0 (dependent on load 4)
    ldr x0, [x0]                // Load 6: x0 = *x0 (dependent on load 5)
    ldr x0, [x0]                // Load 7: x0 = *x0 (dependent on load 6)
    ldr x0, [x0]                // Load 8: x0 = *x0 (dependent on load 7)
    
    subs x4, x4, #1             // Decrement unrolled iteration count
    b.gt latency_loop_unrolled  // Loop if more groups remain

    dmb sy                      // Memory barrier after unrolled section to ensure completion

latency_remainder:              // Handle remaining (0-7) dereferences
    cbz x5, latency_end         // Skip if no remainder
latency_single:                 // Single dependent dereference loop
    ldr x0, [x0]                // Load: x0 = *x0 (dependent chain)
    subs x5, x5, #1             // Decrement remainder count
    b.gt latency_single         // Loop if more dereferences remain

latency_end:                    // Return final pointer value
    ret                         // Return final pointer to prevent optimizations