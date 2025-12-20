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
// memory_copy_reverse_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_copy_reverse_loop_asm(void* dst, const void* src, size_t byteCount);
// Purpose:
//   Copy 'byteCount' bytes from 'src' to 'dst' in reverse order (backwards) using
//   wide (128‑bit) vector loads and pair stores, processing 512 bytes per iteration
//   to maximize throughput with reverse sequential access pattern.
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
// -----------------------------------------------------------------------------
.global _memory_copy_reverse_loop_asm
.align 4
_memory_copy_reverse_loop_asm:
    mov x3, xzr             // offset = 0 (tracked as end pointers)
    add x3, x0, x2          // dst_end = dst + byteCount
    add x4, x1, x2          // src_end = src + byteCount
    // 512B blocks chosen as sweet spot: large enough for high throughput,
    // small enough to fit in L2 cache, avoids TLB pressure on large regions
    mov x5, #512            // step = 512 bytes (optimal block size)

copy_reverse_loop_start_nt512:      // Main 512B block loop (reverse direction)
    // Loop invariants: x5=512B block size, x0/x1 are base pointers.
    // Only x3 (dst_end), x4 (src_end) and x6 (remaining) change per iteration.
    cmp x3, x0              // dst_end <= dst?
    b.le copy_reverse_loop_end      // If done, exit
    
    sub x6, x3, x0          // remaining = dst_end - dst
    cmp x6, x5              // remaining < step?
    b.lt copy_reverse_cleanup  // If less, handle remaining bytes
    
    // Calculate current addresses (working backwards from end)
    sub x7, x3, #512        // dst_block_start = dst_end - 512
    sub x8, x4, #512        // src_block_start = src_end - 512

    // Load 512 bytes from source (16 ldp pairs, 32B each)
    // Use caller-saved registers q0-q7,q16-q31 only (avoid q8-q15 per AAPCS64).
    // This ensures no callee-saved state corruption and follows ARM64 calling convention.
    // Load first 8 pairs (0-7) into q0-q7 and q16-q23
    ldp q0,  q1,  [x8, #0]      // Load pair 0 (offset 0)
    ldp q2,  q3,  [x8, #32]     // Load pair 1 (offset 32)
    ldp q4,  q5,  [x8, #64]     // Load pair 2 (offset 64)
    ldp q6,  q7,  [x8, #96]     // Load pair 3 (offset 96)
    ldp q16, q17, [x8, #128]    // Load pair 4 (offset 128)
    ldp q18, q19, [x8, #160]    // Load pair 5 (offset 160)
    ldp q20, q21, [x8, #192]    // Load pair 6 (offset 192)
    ldp q22, q23, [x8, #224]    // Load pair 7 (offset 224)
    
    // Store first 8 pairs before loading next 8 (to avoid register pressure).
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0,  q1,  [x7, #0]     // Store pair 0 (offset 0, non-temporal hint for bandwidth)
    stnp q2,  q3,  [x7, #32]    // Store pair 1 (offset 32, non-temporal)
    stnp q4,  q5,  [x7, #64]    // Store pair 2 (offset 64, non-temporal)
    stnp q6,  q7,  [x7, #96]    // Store pair 3 (offset 96, non-temporal)
    stnp q16, q17, [x7, #128]   // Store pair 4 (offset 128, non-temporal)
    stnp q18, q19, [x7, #160]   // Store pair 5 (offset 160, non-temporal)
    stnp q20, q21, [x7, #192]   // Store pair 6 (offset 192, non-temporal)
    stnp q22, q23, [x7, #224]   // Store pair 7 (offset 224, non-temporal)
    
    // Load next 8 pairs (8-15) into q24-q31 and q0-q7 (reusing q0-q7)
    ldp q24, q25, [x8, #256]    // Load pair 8 (offset 256)
    ldp q26, q27, [x8, #288]    // Load pair 9 (offset 288)
    ldp q28, q29, [x8, #320]    // Load pair 10 (offset 320)
    ldp q30, q31, [x8, #352]    // Load pair 11 (offset 352)
    ldp q0,  q1,  [x8, #384]    // Load pair 12 (offset 384, reuse q0-q1)
    ldp q2,  q3,  [x8, #416]    // Load pair 13 (offset 416, reuse q2-q3)
    ldp q4,  q5,  [x8, #448]    // Load pair 14 (offset 448, reuse q4-q5)
    ldp q6,  q7,  [x8, #480]    // Load pair 15 (offset 480, reuse q6-q7)
    
    // Store next 8 pairs (8-15)
    stnp q24, q25, [x7, #256]   // Store pair 8 (offset 256, non-temporal)
    stnp q26, q27, [x7, #288]   // Store pair 9 (offset 288, non-temporal)
    stnp q28, q29, [x7, #320]   // Store pair 10 (offset 320, non-temporal)
    stnp q30, q31, [x7, #352]   // Store pair 11 (offset 352, non-temporal)
    stnp q0,  q1,  [x7, #384]   // Store pair 12 (offset 384, non-temporal)
    stnp q2,  q3,  [x7, #416]   // Store pair 13 (offset 416, non-temporal)
    stnp q4,  q5,  [x7, #448]   // Store pair 14 (offset 448, non-temporal)
    stnp q6,  q7,  [x7, #480]   // Store pair 15 (offset 480, non-temporal)

    sub x3, x3, x5          // dst_end -= step (move backwards)
    sub x4, x4, x5          // src_end -= step (move backwards)
    b copy_reverse_loop_start_nt512 // Loop again

copy_reverse_cleanup:          // Tail handling when <512B remain
    // Tail handling: Process remaining bytes in 256B→128B→64B→32B→byte tiers.
    // This minimizes branches while ensuring exact byte count handling.
    // Larger chunks first for better performance on unaligned remainders.
    cmp x3, x0              // dst_end <= dst?
    b.ge copy_reverse_loop_end      // If done, exit

    subs x6, x3, x0         // Recalc remaining (x6)
    sub x7, x3, x6           // dst_block_start = dst_end - remaining
    sub x8, x4, x6           // src_block_start = src_end - remaining

    // Handle 256B chunks (8 ldp/stnp pairs)
    cmp x6, #256              // Check if >= 256 bytes remain
    b.lt copy_reverse_cleanup_128          // If less, handle smaller chunks
    sub x7, x3, #256          // dst_block_start = dst_end - 256
    sub x8, x4, #256          // src_block_start = src_end - 256
    ldp q0, q1, [x8, #0]      // Load first pair (32B)
    ldp q2, q3, [x8, #32]     // Load second pair (32B)
    ldp q4, q5, [x8, #64]     // Load third pair (32B)
    ldp q6, q7, [x8, #96]     // Load fourth pair (32B)
    ldp q16, q17, [x8, #128]  // Load fifth pair (32B)
    ldp q18, q19, [x8, #160]  // Load sixth pair (32B)
    ldp q20, q21, [x8, #192]  // Load seventh pair (32B)
    ldp q22, q23, [x8, #224]  // Load eighth pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal)
    stnp q16, q17, [x7, #128] // Store fifth pair (non-temporal)
    stnp q18, q19, [x7, #160] // Store sixth pair (non-temporal)
    stnp q20, q21, [x7, #192] // Store seventh pair (non-temporal)
    stnp q22, q23, [x7, #224] // Store eighth pair (non-temporal)
    sub x3, x3, #256          // Advance dst_end backwards by 256B
    sub x4, x4, #256          // Advance src_end backwards by 256B
    sub x6, x6, #256          // Decrement remaining count by 256B

copy_reverse_cleanup_128:                  // Handle 128B chunks (4 ldp + 4 stnp)
    cmp x6, #128              // Check if >= 128 bytes remain
    b.lt copy_reverse_cleanup_64           // If less, handle smaller chunks
    sub x7, x3, #128          // dst_block_start = dst_end - 128
    sub x8, x4, #128          // src_block_start = src_end - 128
    ldp q0, q1, [x8, #0]      // Load first pair (32B)
    ldp q2, q3, [x8, #32]     // Load second pair (32B)
    ldp q4, q5, [x8, #64]     // Load third pair (32B)
    ldp q6, q7, [x8, #96]     // Load fourth pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal)
    sub x3, x3, #128          // Advance dst_end backwards by 128B
    sub x4, x4, #128          // Advance src_end backwards by 128B
    sub x6, x6, #128          // Decrement remaining count by 128B

copy_reverse_cleanup_64:                   // Handle 64B chunks (2 ldp + 2 stnp)
    cmp x6, #64               // Check if >= 64 bytes remain
    b.lt copy_reverse_cleanup_32           // If less, handle smaller chunks
    sub x7, x3, #64           // dst_block_start = dst_end - 64
    sub x8, x4, #64           // src_block_start = src_end - 64
    ldp q0, q1, [x8, #0]      // Load first pair (32B)
    ldp q2, q3, [x8, #32]     // Load second pair (32B)
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal)
    sub x3, x3, #64           // Advance dst_end backwards by 64B
    sub x4, x4, #64           // Advance src_end backwards by 64B
    sub x6, x6, #64           // Decrement remaining count by 64B

copy_reverse_cleanup_32:                   // Handle 32B chunk (1 ldp + 1 stnp)
    cmp x6, #32               // Check if >= 32 bytes remain
    b.lt copy_reverse_cleanup_byte    // If less, handle byte tail
    sub x7, x3, #32           // dst_block_start = dst_end - 32
    sub x8, x4, #32           // src_block_start = src_end - 32
    ldp q0, q1, [x8, #0]      // Load pair (32B)
    stnp q0, q1, [x7, #0]     // Store pair (non-temporal)
    sub x3, x3, #32           // Advance dst_end backwards by 32B
    sub x4, x4, #32           // Advance src_end backwards by 32B
    sub x6, x6, #32           // Decrement remaining count by 32B

copy_reverse_cleanup_byte:           // Byte tail for final <32B
    // Final byte-by-byte copy for <32B remainder. Expected to be rare in
    // bandwidth benchmarks but ensures correctness for any input size.
    cmp x6, #0               // Check if any bytes remain
    b.le copy_reverse_loop_end        // If none, exit
    // Initialize pointers to last byte (one before end)
    sub x7, x3, #1           // dst_addr = dst_end - 1
    sub x8, x4, #1           // src_addr = src_end - 1
    // Use pre-indexed addressing to auto-decrement pointers, eliminating
    // redundant address calculations. This reduces loop overhead from 5
    // instructions (sub x7, sub x8, ldrb, strb, sub x3/x4) to 3 instructions.
copy_reverse_cleanup_byte_loop:
    ldrb w9, [x8, #-1]!      // Pre-decrement src_addr, then load byte
    strb w9, [x7, #-1]!      // Pre-decrement dst_addr, then store byte
    subs x6, x6, #1          // Decrement remaining count, set flags
    b.gt copy_reverse_cleanup_byte_loop // Loop if more bytes remain

copy_reverse_loop_end:              // Return to caller
    ret                     // Return

