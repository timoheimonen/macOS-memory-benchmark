// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)
// Optimized loops for memory copy, read, write bandwidth tests, and latency test.
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

// --- Memory Copy Function (128B Non-Temporal Stores) ---
// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// Args: x0=dst, x1=src, x2=byteCount
// Regs: x3=offset, x4=step, x5=remaining, x6=src_addr, x7=dst_addr, w5=byte_tmp, q0-q7=data
_memory_copy_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #128            // step = 128 bytes

copy_loop_start_nt128:      // Main loop start (128B blocks)
    subs x5, x2, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt copy_loop_cleanup  // If less, handle remaining bytes

    // Calculate current addresses
    add x6, x1, x3          // src_addr = base + offset
    add x7, x0, x3          // dst_addr = base + offset

    // Load 128 bytes from source
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]

    // Store 128 bytes (non-temporal) - minimize cache pollution
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]

    add x3, x3, x4          // offset += step
    b copy_loop_start_nt128 // Loop again

copy_loop_cleanup:          // Cleanup loop start (handles < 128 bytes)
    cmp x3, x2              // offset == count?
    b.ge copy_loop_end      // If done, exit

copy_cleanup_byte:          // Byte-by-byte copy loop
    ldrb w5, [x1, x3]       // Load 1 byte
    strb w5, [x0, x3]       // Store 1 byte
    add x3, x3, #1          // offset++
    cmp x3, x2              // offset == count?
    b.lt copy_cleanup_byte  // If not done, loop again

copy_loop_end:              // Function end
    ret                     // Return

// --- Memory Read Function (Bandwidth Test) ---
// Reads memory, calculates XOR sum to prevent optimization.

.global _memory_read_loop_asm // Make function visible
.align 4                      // Align to 16 bytes

// C++ Signature: uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
// Args: x0=src, x1=byteCount
// Returns: x0=XOR checksum (lower 64 bits)
// Regs: x3=offset, x4=step, x5=remaining/byte_tmp, x6=src_addr, q0-q7=data, q8-q11=accumulators
_memory_read_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #128            // step = 128 bytes

    // Zero accumulators (q8-q11)
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b

read_loop_start_128:        // Main loop start (128B blocks)
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt read_loop_cleanup  // If less, handle remaining bytes

    add x6, x0, x3          // src_addr = base + offset

    // Load 128 bytes from source
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]

    // Accumulate XOR sum into q8-q11 (sink operation)
    eor v8.16b, v8.16b, v0.16b   ; eor v9.16b, v9.16b, v1.16b
    eor v10.16b, v10.16b, v2.16b ; eor v11.16b, v11.16b, v3.16b
    eor v8.16b, v8.16b, v4.16b   ; eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b ; eor v11.16b, v11.16b, v7.16b

    add x3, x3, x4          // offset += step
    b read_loop_start_128   // Loop again

read_loop_cleanup:          // Cleanup loop start (handles < 128 bytes)
    cmp x3, x1              // offset == count?
    b.ge read_loop_combine_sum // If done, combine sums

read_cleanup_byte:          // Byte-by-byte read loop
    ldrb w5, [x0, x3]       // Load 1 byte
    eor x8, x8, x5          // checksum ^= byte (using lower 64b of q8)
    add x3, x3, #1          // offset++
    cmp x3, x1              // offset == count?
    b.lt read_cleanup_byte  // If not done, loop again

read_loop_combine_sum:      // Combine final checksum
    // Combine accumulators q8-q11 -> q8
    eor v8.16b, v8.16b, v9.16b
    eor v10.16b, v10.16b, v11.16b
    eor v8.16b, v8.16b, v10.16b

    // Return lower 64 bits of final XOR sum
    umov x0, v8.d[0]        // x0 = lower 64 bits of q8
    ret                     // Return checksum

// --- Memory Write Function (Bandwidth Test - 128B Non-Temporal Stores) ---
// Writes zeros to memory using non-temporal stores.

.global _memory_write_loop_asm // Make function visible
.align 4                       // Align to 16 bytes

// C++ Signature: void memory_write_loop_asm(void* dst, size_t byteCount);
// Args: x0=dst, x1=byteCount
// Note: Writes zeros.
// Regs: x3=offset, x4=step, x5=remaining/byte_tmp, x7=dst_addr, q0-q7=zero_data
_memory_write_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #128            // step = 128 bytes

    // Zero out data registers q0-q7
    fmov d0, xzr ; fmov d1, xzr
    fmov d2, xzr ; fmov d3, xzr
    fmov d4, xzr ; fmov d5, xzr
    fmov d6, xzr ; fmov d7, xzr

write_loop_start_nt128:     // Main loop start (128B blocks)
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt write_loop_cleanup // If less, handle remaining bytes

    add x7, x0, x3          // dst_addr = base + offset

    // Store 128B zeros (non-temporal)
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]

    add x3, x3, x4          // offset += step
    b write_loop_start_nt128 // Loop again

write_loop_cleanup:         // Cleanup loop start (handles < 128 bytes)
    cmp x3, x1              // offset == count?
    b.ge write_loop_end     // If done, exit

write_cleanup_byte:         // Byte-by-byte write loop
    strb wzr, [x0, x3]      // Store 1 zero byte (wzr = zero reg)
    add x3, x3, #1          // offset++
    cmp x3, x1              // offset == count?
    b.lt write_cleanup_byte // If not done, loop again

write_loop_end:             // Function end
    ret                     // Return

// --- Memory Latency Function (Pointer Chasing) ---
// Measures latency via dependent loads.

.global _memory_latency_chase_asm // Make function visible
.align 4                       // Align to 16 bytes

// C++ Signature: void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Args: x0=ptr_addr, x1=count
_memory_latency_chase_asm:
latency_loop:               // Loop start
    ldr x0, [x0]            // Load next ptr addr from current ptr addr (dependent)
    subs x1, x1, #1         // count--
    b.ne latency_loop       // If count != 0, loop again
ret                         // Return