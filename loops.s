// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)

.global _memory_copy_loop_asm // Make visible to linker
.align 4                      // Align function entry (16 bytes)

// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// Args: x0=dst_base, x1=src_base, x2=byteCount
// Regs: x3=offset, x4=step, x5=remaining, x6=src_addr, x7=dst_addr, w5=byte_tmp, q0-q7=data
_memory_copy_loop_asm:
    mov x3, xzr     // x3 = offset = 0
    mov x4, #128    // x4 = step (128 bytes)
copy_loop_start_nt128:
    subs x5, x2, x3 // remaining = byteCount - offset
    cmp x5, x4      // if remaining < step, goto cleanup
    b.lt copy_loop_cleanup
    add x6, x1, x3  // src_addr = src_base + offset
    add x7, x0, x3  // dst_addr = dst_base + offset
    // Load 128B using normal paired loads
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    // Store 128B using non-temporal paired stores (minimize cache pollution)
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x3, x3, x4  // offset += 128
    b copy_loop_start_nt128
copy_loop_cleanup:
    cmp x3, x2      // Done?
    b.ge copy_loop_end
copy_cleanup_byte: // Simple byte cleanup loop
    ldrb w5, [x1, x3]
    strb w5, [x0, x3]
    add x3, x3, #1
    cmp x3, x2
    b.lt copy_cleanup_byte
copy_loop_end:
    ret

// --- Memory Read Function (Bandwidth Test) ---

.global _memory_read_loop_asm // Make visible to linker
.align 4                      // Align function entry

// C++ Signature: uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
// Args: x0=src_base, x1=byteCount
// Returns: x0=XOR checksum (dummy value to prevent read optimization)
// Regs: x3=offset, x4=step, x5=remaining/byte_tmp, x6=src_addr, q0-q7=data, q8-q11=accumulators
_memory_read_loop_asm:
    mov x3, xzr     // x3 = offset = 0
    mov x4, #128    // x4 = step (128 bytes)
    // Init XOR checksum accumulators (q8-q11) to zero
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b

read_loop_start_128:
    subs x5, x1, x3 // remaining = byteCount - offset
    cmp x5, x4      // if remaining < step, goto cleanup
    b.lt read_loop_cleanup
    add x6, x0, x3  // src_addr = src_base + offset
    // Load 128B using normal paired loads
    ldp q0, q1, [x6, #0]
    ldp q2, q3, [x6, #32]
    ldp q4, q5, [x6, #64]
    ldp q6, q7, [x6, #96]
    // Accumulate XOR sum into q8-q11 (sink operation)
    eor v8.16b, v8.16b, v0.16b   ; eor v9.16b, v9.16b, v1.16b
    eor v10.16b, v10.16b, v2.16b ; eor v11.16b, v11.16b, v3.16b
    eor v8.16b, v8.16b, v4.16b   ; eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b ; eor v11.16b, v11.16b, v7.16b
    add x3, x3, x4  // offset += 128
    b read_loop_start_128
read_loop_cleanup:
    cmp x3, x1      // Done?
    b.ge read_loop_combine_sum
read_cleanup_byte: // Simple byte cleanup loop
    ldrb w5, [x0, x3] // Load byte
    eor x8, x8, x5    // XOR lower 64 bits of accumulator x8 with byte
    add x3, x3, #1    // offset++
    cmp x3, x1
    b.lt read_cleanup_byte

read_loop_combine_sum:
    // Combine accumulators q8-q11 -> q8
    eor v8.16b, v8.16b, v9.16b
    eor v10.16b, v10.16b, v11.16b
    eor v8.16b, v8.16b, v10.16b
    // Return lower 64 bits of final XOR sum
    umov x0, v8.d[0]
    ret

// --- Memory Write Function (Bandwidth Test - Optimized with 128B Non-Temporal Stores) ---

.global _memory_write_loop_asm // Make visible to linker
.align 4                       // Align function entry

// C++ Signature: void memory_write_loop_asm(void* dst, size_t byteCount);
// Args: x0=dst_base, x1=byteCount
// Note: Writes zeros.
// Regs: x3=offset, x4=step, x5=remaining/byte_tmp, x7=dst_addr, q0-q7=zero_data
_memory_write_loop_asm:
    mov x3, xzr     // x3 = offset = 0
    mov x4, #128    // x4 = step (128 bytes)
    // Zero out registers q0-q7 using NEON fmov (fastest way?)
    fmov d0, xzr ; fmov d1, xzr
    fmov d2, xzr ; fmov d3, xzr
    fmov d4, xzr ; fmov d5, xzr
    fmov d6, xzr ; fmov d7, xzr

write_loop_start_nt128:
    subs x5, x1, x3 // remaining = byteCount - offset
    cmp x5, x4      // if remaining < step, goto cleanup
    b.lt write_loop_cleanup
    add x7, x0, x3  // dst_addr = dst_base + offset
    // Store 128B zeros using non-temporal paired stores
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    add x3, x3, x4  // offset += 128
    b write_loop_start_nt128
write_loop_cleanup:
    cmp x3, x1      // Done?
    b.ge write_loop_end
write_cleanup_byte: // Simple byte cleanup loop
    strb wzr, [x0, x3] // Store zero byte (wzr is zero register for 32-bit)
    add x3, x3, #1    // offset++
    cmp x3, x1
    b.lt write_cleanup_byte
write_loop_end:
    ret

// --- Memory Latency Function (Pointer Chasing) ---

.global _memory_latency_chase_asm
.align 4

// C++ Signature: void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Args: x0=ptr_addr, x1=count
_memory_latency_chase_asm:
latency_loop:
    ldr x0, [x0]      // The dependent load
    subs x1, x1, #1   // count--
    b.ne latency_loop // Loop if count != 0
ret