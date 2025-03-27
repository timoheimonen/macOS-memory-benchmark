// loops.s
// Contains assembly implementations for memory operations (ARM64 macOS)

// Make the symbol _memory_copy_loop_asm visible to the linker
.global _memory_copy_loop_asm
// Align the function start to a 16-byte boundary (performance)
.align 4

// Function callable from C++:
// void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// Arguments according to AAPCS64 calling convention:
// x0: dst (destination address)
// x1: src (source address)
// x2: byteCount (number of bytes to copy)

_memory_copy_loop_asm:
    // --- Loop start ---
    // Use x3 as a counter/offset, start from zero
    mov x3, xzr
    // Use x4 for the step size (e.g., 16 bytes for NEON or 32/64/128 if unrolling)
    mov x4, #16

loop_start:
    // Compare offset (x3) with total count (x2)
    cmp x3, x2
    // If offset >= total count, jump to loop end (unsigned compare)
    b.ge loop_end

    // --- Core operation: Copy e.g., 16 bytes at a time using NEON register q0 ---
    ldr q0, [x1, x3]        // Load 16 bytes from [source + offset] into register q0
    str q0, [x0, x3]        // Store 16 bytes from register q0 to [destination + offset]

    // COULD ADD LOOP UNROLLING HERE:
    // Load and store more data (e.g., q1, q2, q3) per iteration
    // Example for 64 bytes per iteration:
    // ldr q1, [x1, x3, #16]
    // str q1, [x0, x3, #16]
    // ldr q2, [x1, x3, #32]
    // str q2, [x0, x3, #32]
    // ldr q3, [x1, x3, #48]
    // str q3, [x0, x3, #48]
    // add x3, x3, #64       // Increment offset by 64

    // Increment offset by the step size (16 in this case)
    add x3, x3, x4
    // Jump back to the start of the loop
    b loop_start

loop_end:
    // Return to C++ code
    ret
