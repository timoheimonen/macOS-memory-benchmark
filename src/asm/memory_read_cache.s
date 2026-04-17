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
// memory_read_cache_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_cache_loop_asm(const void* src, size_t byteCount);
// Purpose:
//   Read 'byteCount' bytes from 'src' for cache-bandwidth paths (L1/L2/custom).
//   This kernel is intentionally independent from main-memory read kernel so cache
//   tuning can evolve without coupling to DRAM path behavior.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t)
// Returns:
//   x0 = 64-bit XOR checksum
// Clobbers:
//   x2-x7, x12-x13, q0-q7, q16-q31
// ABI Notes:
//   * AAPCS64 callee-saved registers are preserved (x19-x28, v8-v15 untouched).
//   * Checksum exists to keep loads observable (anti-DCE), not as a data-integrity primitive.
// Implementation Notes:
//   * Uses pointer+remaining loop state (x6/x5) to avoid per-iteration offset math.
//   * Processes 512B per iteration (16 pair loads) for high cache-path throughput.
//   * Distributes XOR into four accumulators (v0-v3) to reduce dependency depth.
//   * Tail path uses size-bit tests (tbz for 256/128/64/32), then byte cleanup.
//   * Main loop label is 64-byte aligned to keep the unrolled body on a single
//     I-cache line boundary for steady run-to-run timing on Apple Silicon.
// Control-Flow Map:
//   main 512B loop -> tiered tail (256/128/64/32) -> byte tail -> final reduction
// Timing Contract:
//   Caller must emit `dsb ish; isb` before reading the start-of-measurement
//   timestamp and another `dsb ish; isb` before reading the end-of-measurement
//   timestamp. This kernel emits no internal fences; barrier discipline is the
//   caller's responsibility for reproducible timing.
// -----------------------------------------------------------------------------

.global _memory_read_cache_loop_asm
.align 4
_memory_read_cache_loop_asm:
    // x6 = current source pointer, x5 = remaining bytes.
    // x12 accumulates XOR for byte cleanup (<32B tail).
    mov x6, x0
    mov x5, x1
    mov x12, xzr

    // Zero vector accumulators once; they fold all wide reads.
    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b

    // Align hot loop entry to 64B so the unrolled 512B body always lands on a
    // predictable I-cache line. Reduces first-iteration fetch-boundary jitter.
    .p2align 6
cache_read_loop_start_512:    // Main 512B loop
    // If <512B remain, switch to tail handling.
    cmp x5, #512
    b.lo cache_read_loop_cleanup

    // Load first 256B (8x 32B pairs) using caller-saved SIMD regs.
    ldp q4,  q5,  [x6, #0]
    ldp q6,  q7,  [x6, #32]
    ldp q16, q17, [x6, #64]
    ldp q18, q19, [x6, #96]
    ldp q20, q21, [x6, #128]
    ldp q22, q23, [x6, #160]
    ldp q24, q25, [x6, #192]
    ldp q26, q27, [x6, #224]

    // Fold first 256B into four independent accumulators (v0-v3).
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

    // Load second 256B (offsets 256..480).
    ldp q4,  q5,  [x6, #256]
    ldp q6,  q7,  [x6, #288]
    ldp q16, q17, [x6, #320]
    ldp q18, q19, [x6, #352]
    ldp q20, q21, [x6, #384]
    ldp q22, q23, [x6, #416]
    ldp q24, q25, [x6, #448]
    ldp q26, q27, [x6, #480]

    // Fold second 256B into the same accumulators.
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

    // Advance pointers/counters for next 512B block.
    add x6, x6, #512
    sub x5, x5, #512
    b cache_read_loop_start_512

cache_read_loop_cleanup:      // Tail handling when <512B remain
    cbz x5, cache_read_loop_combine_sum

    // Tiered tail: bits in x5 encode optional chunks.
    // bit8=256B, bit7=128B, bit6=64B, bit5=32B.
    tbz x5, #8, cache_read_cleanup_128
    // 256B chunk: load+fold exactly like half of main loop.
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

cache_read_cleanup_128:       // Optional 128B chunk
    tbz x5, #7, cache_read_cleanup_64
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

cache_read_cleanup_64:        // Optional 64B chunk
    tbz x5, #6, cache_read_cleanup_32
    ldp q4, q5, [x6, #0]
    ldp q6, q7, [x6, #32]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    eor v2.16b, v2.16b, v6.16b
    eor v3.16b, v3.16b, v7.16b
    add x6, x6, #64
    sub x5, x5, #64

cache_read_cleanup_32:        // Optional 32B chunk
    tbz x5, #5, cache_read_cleanup_byte
    ldp q4, q5, [x6, #0]
    eor v0.16b, v0.16b, v4.16b
    eor v1.16b, v1.16b, v5.16b
    add x6, x6, #32
    sub x5, x5, #32

cache_read_cleanup_byte:      // Final byte tail (<32B)
    // Byte tail keeps checksum exact for non-vectorizable remainder.
    cbz x5, cache_read_loop_combine_sum
    ldrb w13, [x6], #1
    eor x12, x12, x13
    subs x5, x5, #1
    b.ne cache_read_cleanup_byte

cache_read_loop_combine_sum:  // Final reduction and return value
    // Reduce four vector accumulators into one.
    eor v0.16b, v0.16b, v1.16b
    eor v2.16b, v2.16b, v3.16b
    eor v0.16b, v0.16b, v2.16b
    // Extract low 64b lane and fold byte-tail checksum (x12) into return value.
    umov x0, v0.d[0]
    eor x0, x0, x12
    ret
