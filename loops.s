// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)
// Optimized loops for memory copy, read, write bandwidth tests (512B blocks),
// and latency test.
//
// Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>
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

.global _memory_copy_loop_asm // Make function visible
.align 4                      // Align to 16 bytes

// --- Memory Copy Function (512B Non-Temporal Stores) ---
// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// Args: x0=dst, x1=src, x2=byteCount
// Uses q0-q31 for data
_memory_copy_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes

copy_loop_start_nt512:      // Main loop start (512B blocks)
    subs x5, x2, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt copy_loop_cleanup  // If less, handle remaining bytes

    // Calculate current addresses
    add x6, x1, x3          // src_addr = base + offset
    add x7, x0, x3          // dst_addr = base + offset

    // Load 512 bytes from source (16 ldp pairs)
    ldp q0,  q1,  [x6, #0]
    ldp q2,  q3,  [x6, #32]
    ldp q4,  q5,  [x6, #64]
    ldp q6,  q7,  [x6, #96]
    ldp q8,  q9,  [x6, #128]
    ldp q10, q11, [x6, #160]
    ldp q12, q13, [x6, #192]
    ldp q14, q15, [x6, #224]
    ldp q16, q17, [x6, #256]
    ldp q18, q19, [x6, #288]
    ldp q20, q21, [x6, #320]
    ldp q22, q23, [x6, #352]
    ldp q24, q25, [x6, #384]
    ldp q26, q27, [x6, #416]
    ldp q28, q29, [x6, #448]
    ldp q30, q31, [x6, #480]


    // Store 512 bytes (non-temporal) - minimize cache pollution (16 stnp pairs)
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q8,  q9,  [x7, #128]
    stnp q10, q11, [x7, #160]
    stnp q12, q13, [x7, #192]
    stnp q14, q15, [x7, #224]
    stnp q16, q17, [x7, #256]
    stnp q18, q19, [x7, #288]
    stnp q20, q21, [x7, #320]
    stnp q22, q23, [x7, #352]
    stnp q24, q25, [x7, #384]
    stnp q26, q27, [x7, #416]
    stnp q28, q29, [x7, #448]
    stnp q30, q31, [x7, #480]

    add x3, x3, x4          // offset += step
    b copy_loop_start_nt512 // Loop again

copy_loop_cleanup:          // Cleanup for <512B: vectorized for efficiency
    cmp x3, x2              // offset == count?
    b.ge copy_loop_end      // If done, exit

    subs x5, x2, x3         // Recalc remaining (x5)
    add x6, x1, x3          // src_addr = src + offset
    add x7, x0, x3          // dst_addr = dst + offset

    // Handle 256B chunks (8 ldp/stnp pairs)
    cmp x5, #256
    b.lt cleanup_128
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    ldp q8, q9, [x6, #128]
    ldp q10, q11, [x6, #160]
    ldp q12, q13, [x6, #192]
    ldp q14, q15, [x6, #224]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q8, q9, [x7, #128]
    stnp q10, q11, [x7, #160]
    stnp q12, q13, [x7, #192]
    stnp q14, q15, [x7, #224]
    add x6, x6, #256
    add x7, x7, #256
    sub x5, x5, #256

cleanup_128:               // Handle 128B chunks (4 ldp/stnp pairs)
    cmp x5, #128
    b.lt cleanup_64
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

cleanup_64:                // Handle 64B chunks (2 ldp/stnp pairs)
    cmp x5, #64
    b.lt cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x6, x6, #64
    add x7, x7, #64
    sub x5, x5, #64

cleanup_32:                // Handle 32B chunks (1 ldp/stnp pair)
    cmp x5, #32
    b.lt copy_cleanup_byte
    ldp q0, q1, [x6, #0]
    stnp q0, q1, [x7, #0]
    add x6, x6, #32
    add x7, x7, #32
    sub x5, x5, #32

copy_cleanup_byte:          // Byte-by-byte for final <32B
    cmp x5, #0
    b.le copy_loop_end
    ldrb w8, [x6], #1       // Load byte, post-inc
    strb w8, [x7], #1       // Store byte, post-inc
    subs x5, x5, #1
    b.gt copy_cleanup_byte

copy_loop_end:              // Function end
    ret                     // Return

// --- Memory Read Function (Bandwidth Test - 512B block) ---
// Reads memory, calculates XOR sum to prevent optimization.

.global _memory_read_loop_asm // Make function visible
.align 4                      // Align to 16 bytes

// C++ Signature: uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
// Args: x0=src, x1=byteCount
// Returns: x0=XOR checksum (lower 64 bits)
// Uses: q0-q31=data, q8-q11=accumulators (reused indices, but ok)
_memory_read_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes
    mov x12, xzr            // Zero byte cleanup checksum accumulator

    // Zero accumulators (v8-v11)
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b

read_loop_start_512:        // Main loop start (512B blocks)
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt read_loop_cleanup  // If less, handle remaining bytes

    add x6, x0, x3          // src_addr = base + offset

    // Load 512 bytes from source (16 ldp pairs, q0-q31)
    ldp q0,  q1,  [x6, #0]
    ldp q2,  q3,  [x6, #32]
    ldp q4,  q5,  [x6, #64]
    ldp q6,  q7,  [x6, #96]
    // Store loaded values from q8-q11 temporarily in v24-v27 (these are available)
    ldp q24, q25, [x6, #128]  // Load to temps instead of directly to accumulators
    ldp q26, q27, [x6, #160]  // Load to temps instead of directly to accumulators  
    ldp q12, q13, [x6, #192]
    ldp q14, q15, [x6, #224]
    ldp q16, q17, [x6, #256]
    ldp q18, q19, [x6, #288]
    ldp q20, q21, [x6, #320]
    ldp q22, q23, [x6, #352]
    // Repurpose upper registers that we already processed
    ldp q28, q29, [x6, #384]
    ldp q30, q31, [x6, #416]
    // Last two loads directly to free scratch registers
    ldp q0,  q1,  [x6, #448]  // Reuse q0,q1 as we already accumulated them
    ldp q2,  q3,  [x6, #480]  // Reuse q2,q3 as we already accumulated them

    // Accumulate XOR sum into v8-v11 (sink operation)
    // Distribute q0-q31 across the 4 accumulators
    eor v8.16b,  v8.16b,  v0.16b    ; eor v9.16b,  v9.16b,  v1.16b
    eor v10.16b, v10.16b, v2.16b   ; eor v11.16b, v11.16b, v3.16b
    eor v8.16b,  v8.16b,  v4.16b    ; eor v9.16b,  v9.16b,  v5.16b
    eor v10.16b, v10.16b, v6.16b   ; eor v11.16b, v11.16b, v7.16b
    eor v8.16b,  v8.16b,  v24.16b   ; eor v9.16b,  v9.16b,  v25.16b  // Use temps for v8/v9
    eor v10.16b, v10.16b, v26.16b  ; eor v11.16b, v11.16b, v27.16b  // Use temps for v10/v11
    eor v8.16b,  v8.16b,  v12.16b   ; eor v9.16b,  v9.16b,  v13.16b
    eor v10.16b, v10.16b, v14.16b  ; eor v11.16b, v11.16b, v15.16b
    eor v8.16b,  v8.16b,  v16.16b   ; eor v9.16b,  v9.16b,  v17.16b
    eor v10.16b, v10.16b, v18.16b  ; eor v11.16b, v11.16b, v19.16b
    eor v8.16b,  v8.16b,  v20.16b   ; eor v9.16b,  v9.16b,  v21.16b
    eor v10.16b, v10.16b, v22.16b  ; eor v11.16b, v11.16b, v23.16b
    eor v8.16b,  v8.16b,  v28.16b   ; eor v9.16b,  v9.16b,  v29.16b
    eor v10.16b, v10.16b, v30.16b  ; eor v11.16b, v11.16b, v31.16b
    eor v8.16b,  v8.16b,  v0.16b    ; eor v9.16b,  v9.16b,  v1.16b   // Reused q0-q3
    eor v10.16b, v10.16b, v2.16b   ; eor v11.16b, v11.16b, v3.16b   // Reused q0-q3

    add x3, x3, x4          // offset += step
    b read_loop_start_512   // Loop again

read_loop_cleanup:          // Cleanup for <512B: vectorized for efficiency
    cmp x3, x1              // offset == count?
    b.ge read_loop_combine_sum // If done, combine sums

    subs x5, x1, x3         // Recalc remaining (x5)
    add x6, x0, x3          // src_addr = src + offset

    // Handle 256B chunks (8 ldp pairs + eor)
    cmp x5, #256
    b.lt read_cleanup_128
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    ldp q8, q9, [x6, #128]
    ldp q10, q11, [x6, #160]
    ldp q12, q13, [x6, #192]
    ldp q14, q15, [x6, #224]
    eor v8.16b, v8.16b, v0.16b ; eor v9.16b, v9.16b, v1.16b
    eor v10.16b, v10.16b, v2.16b ; eor v11.16b, v11.16b, v3.16b
    eor v8.16b, v8.16b, v4.16b ; eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b ; eor v11.16b, v11.16b, v7.16b
    eor v8.16b, v8.16b, v8.16b ; eor v9.16b, v9.16b, v9.16b  // Note: Reusing loaded registers for eor
    eor v10.16b, v10.16b, v10.16b ; eor v11.16b, v11.16b, v11.16b
    eor v8.16b, v8.16b, v12.16b ; eor v9.16b, v9.16b, v13.16b
    eor v10.16b, v10.16b, v14.16b ; eor v11.16b, v11.16b, v15.16b
    add x6, x6, #256
    sub x5, x5, #256

read_cleanup_128:          // Handle 128B chunks
    cmp x5, #128
    b.lt read_cleanup_64
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    eor v8.16b, v8.16b, v0.16b ; eor v9.16b, v9.16b, v1.16b
    eor v10.16b, v10.16b, v2.16b ; eor v11.16b, v11.16b, v3.16b
    eor v8.16b, v8.16b, v4.16b ; eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b ; eor v11.16b, v11.16b, v7.16b
    add x6, x6, #128
    sub x5, x5, #128

read_cleanup_64:           // Handle 64B chunks
    cmp x5, #64
    b.lt read_cleanup_32
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    eor v8.16b, v8.16b, v0.16b ; eor v9.16b, v9.16b, v1.16b
    eor v10.16b, v10.16b, v2.16b ; eor v11.16b, v11.16b, v3.16b
    add x6, x6, #64
    sub x5, x5, #64

read_cleanup_32:           // Handle 32B chunks
    cmp x5, #32
    b.lt read_cleanup_byte
    ldp q0, q1, [x6, #0]
    eor v8.16b, v8.16b, v0.16b ; eor v9.16b, v9.16b, v1.16b
    add x6, x6, #32
    sub x5, x5, #32

read_cleanup_byte:          // Byte-by-byte for final <32B
    cmp x5, #0
    b.le read_loop_combine_sum
    ldrb w13, [x6], #1      // Load byte, post-inc
    eor x12, x12, x13       // XOR to accumulator
    subs x5, x5, #1
    b.gt read_cleanup_byte

read_loop_combine_sum:      // Combine final checksum
    // Combine accumulators v8-v11 -> v8
    eor v8.16b, v8.16b, v9.16b
    eor v10.16b, v10.16b, v11.16b
    eor v8.16b, v8.16b, v10.16b

    // Combine byte checksum (x12) into final result (x0 from v8)
    umov x0, v8.d[0]        // x0 = lower 64 bits of combined vector sum
    eor x0, x0, x12         // Combine with byte checksum

    ret                     // Return checksum
    
// --- Memory Write Function (Bandwidth Test - 512B Non-Temporal Stores) ---
// Writes zeros to memory using non-temporal stores.

.global _memory_write_loop_asm // Make function visible
.align 4                       // Align to 16 bytes

// C++ Signature: void memory_write_loop_asm(void* dst, size_t byteCount);
// Args: x0=dst, x1=byteCount
// Note: Writes zeros. Uses q0-q31.
_memory_write_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes

    // Zero out data registers v0-v31 (full 128-bit vectors for consistent zero writes)
    movi v0.16b, #0 ; movi v1.16b, #0
    movi v2.16b, #0 ; movi v3.16b, #0
    movi v4.16b, #0 ; movi v5.16b, #0
    movi v6.16b, #0 ; movi v7.16b, #0
    movi v8.16b, #0 ; movi v9.16b, #0
    movi v10.16b, #0 ; movi v11.16b, #0
    movi v12.16b, #0 ; movi v13.16b, #0
    movi v14.16b, #0 ; movi v15.16b, #0
    movi v16.16b, #0 ; movi v17.16b, #0
    movi v18.16b, #0 ; movi v19.16b, #0
    movi v20.16b, #0 ; movi v21.16b, #0
    movi v22.16b, #0 ; movi v23.16b, #0
    movi v24.16b, #0 ; movi v25.16b, #0
    movi v26.16b, #0 ; movi v27.16b, #0
    movi v28.16b, #0 ; movi v29.16b, #0
    movi v30.16b, #0 ; movi v31.16b, #0


write_loop_start_nt512:     // Main loop start (512B blocks)
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt write_loop_cleanup // If less, handle remaining bytes

    add x7, x0, x3          // dst_addr = base + offset

    // Store 512B zeros (non-temporal) (16 stnp pairs)
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q8,  q9,  [x7, #128]
    stnp q10, q11, [x7, #160]
    stnp q12, q13, [x7, #192]
    stnp q14, q15, [x7, #224]
    stnp q16, q17, [x7, #256]
    stnp q18, q19, [x7, #288]
    stnp q20, q21, [x7, #320]
    stnp q22, q23, [x7, #352]
    stnp q24, q25, [x7, #384]
    stnp q26, q27, [x7, #416]
    stnp q28, q29, [x7, #448]
    stnp q30, q31, [x7, #480]

    add x3, x3, x4          // offset += step
    b write_loop_start_nt512 // Loop again

write_loop_cleanup:         // Cleanup for <512B: vectorized for efficiency
    cmp x3, x1              // offset == count?
    b.ge write_loop_end     // If done, exit

    subs x5, x1, x3         // Recalc remaining (x5)
    add x7, x0, x3          // dst_addr = dst + offset

    // Handle 256B chunks (8 stnp pairs)
    cmp x5, #256
    b.lt write_cleanup_128
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q8, q9, [x7, #128]
    stnp q10, q11, [x7, #160]
    stnp q12, q13, [x7, #192]
    stnp q14, q15, [x7, #224]
    add x7, x7, #256
    sub x5, x5, #256

write_cleanup_128:         // Handle 128B chunks
    cmp x5, #128
    b.lt write_cleanup_64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_cleanup_64:          // Handle 64B chunks
    cmp x5, #64
    b.lt write_cleanup_32
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_cleanup_32:          // Handle 32B chunks
    cmp x5, #32
    b.lt write_cleanup_byte
    stnp q0, q1, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_cleanup_byte:         // Byte-by-byte for final <32B
    cmp x5, #0
    b.le write_loop_end
    strb wzr, [x7], #1      // Store zero byte, post-inc
    subs x5, x5, #1
    b.gt write_cleanup_byte

write_loop_end:             // Function end
    ret                     // Return

// --- Memory Latency Function (Pointer Chasing) ---

.global _memory_latency_chase_asm
.align 4

// C++ Signature: uint64_t memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Args: x0=ptr_addr, x1=count
// Returns: x0=result (to prevent optimization)
_memory_latency_chase_asm:
    mov x2, x1              // Save count in x2 for unrolled loop
    mov x3, x0              // Preserve original pointer in x3
    
    // Pre-touch to ensure TLB entries are loaded
    ldr x4, [x0]            // Prime the first access but don't use result yet
    dsb ish                 // Data synchronization barrier
    isb                     // Instruction synchronization barrier

    // Compute unrolled iterations and remainder for exact count handling
    lsr x4, x1, #3          // x4 = count / 8 (unrolled iterations)
    and x5, x1, #7          // x5 = count % 8 (remainder)

    cbz x4, latency_remainder // Skip unrolled if no full groups

    dmb sy                  // Barrier to prevent reordering before measurement

latency_loop_unrolled:  // Unrolled loop for multiples of 8
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    ldr x0, [x0]
    
    subs x4, x4, #1     // Decrement unrolled count
    b.gt latency_loop_unrolled // Loop if more groups

    dmb sy                  // Barrier after unrolled to ensure completion

latency_remainder:      // Handle remaining accesses (0-7)
    cbz x5, latency_end // Skip if no remainder
latency_single:
    ldr x0, [x0]
    subs x5, x5, #1
    b.gt latency_single

latency_end:
    ret                 // Return final pointer to prevent optimizations