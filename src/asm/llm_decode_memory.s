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
// llm_decode_memory_asm
// -----------------------------------------------------------------------------
// C++ prototype:
//   extern "C" void llm_decode_memory_asm(
//       const LlmLayerDescriptor* layers,
//       const LlmKvSequenceDescriptor* sequences,
//       uint64_t layer_count,
//       uint64_t step_count,
//       uint64_t scenario_flags,
//       uint64_t scenario_seed,
//       LlmWorkerChecksum* output) noexcept;
//
// Arguments:
//   x0 = layer descriptor array
//   x1 = KV sequence descriptor array
//   x2 = layer descriptor count
//   x3 = task-local synthetic step count
//   x4 = scenario flags: bit 0 = weights, bit 1 = KV (valid values 1..3)
//   x5 = scenario-domain seed used as the append-affine base
//   x6 = 16-byte-aligned worker checksum output
//
// Frozen descriptor ABI (llm-memory-descriptor-abi-v1):
//   LayerDescriptor, stride 48:
//      0 weight_ptr                 8 weight_bytes
//     16 first_sequence_index     24 sequence_count
//     32 layer_index              40 reserved_zero
//   KvSequenceDescriptor, stride 80:
//      0 k_visible_ptr              8 k_visible_bytes
//     16 v_visible_ptr             24 v_visible_bytes
//     32 k_append_ptr              40 k_append_bytes
//     48 v_append_ptr              56 v_append_bytes
//     64 batch_sequence_index      72 append_record_byte_offset
//
// Frozen output ABI (three consecutive 32-byte components):
//   weight offsets 0/8/16/24, K offsets 32/40/48/56,
//   V offsets 64/72/80/88; fields are state_a, state_b,
//   exact_bytes_read, and span_count.
//
// Traversal and memory semantics:
//   * Each step visits layers in descriptor order. A mixed layer reads its
//     weight span first, then visits its sequence descriptors in increasing
//     array order. Each sequence performs K append, V append, K read, V read.
//   * The read hot loop consumes exactly 32 bytes with one `ldp q` and treats
//     lanes [word0, word2] as even and [word1, word3] as odd. Full-word and
//     1-7-byte tails never cross the supplied span. Word parity restarts at
//     even for every non-empty span.
//   * Append stores are ordinary temporal `stp`/`str` operations. A canonical
//     record offset that starts inside a 64-bit word is peeled with exact
//     4/2/1-byte stores; the implementation never performs read-modify-write
//     and never touches bytes owned by an adjacent worker.
//   * Unaligned Normal-memory vector/scalar accesses are intentional. Planner
//     boundaries prefer 32-byte alignment but exact layer and record tails do
//     not promise it.
//   * Worker ranges are disjoint, so the kernel emits no internal barrier.
//     The executor owns start/completion barriers and all timestamp fences.
//     There is no timestamp read, software prefetch, cache maintenance, or
//     signal polling in this hot path.
//
// AAPCS64 and register use:
//   * x19-x30 are saved and restored; q8-q15 and platform register x18 are
//     untouched. Caller-saved q0-q7 hold loaded data and append vectors.
//   * x19-x22 = weight checksum, x23-x26 = K checksum,
//     x27-x30 = V checksum, in field order.
//   * x8-x17 hold immutable bases and traversal state. The span/append macros
//     clobber only x0-x7 and q0-q7.
//   * This is a leaf function. Its incoming x30 is saved before x30 is reused
//     for the V span counter.
//
// Safe direct-boundary behavior:
//   A null output returns immediately. With a valid output, all components are
//   initialized before validation; null top-level descriptor pointers, invalid
//   scenario flags, zero layers, or zero steps return the initialized result.
//   Materialized inner descriptors are trusted executor input.
// -----------------------------------------------------------------------------

// Load one exact 64-bit literal without a literal-pool memory access.
.macro LLM_LOAD_U64 reg, half0, half1, half2, half3
    movz \reg, #\half0
    movk \reg, #\half1, lsl #16
    movk \reg, #\half2, lsl #32
    movk \reg, #\half3, lsl #48
.endm

// Absorb the non-empty span in x0/x1 into the supplied persistent component.
// x14 permanently holds the 0x9E3779B97F4A7C15 ordinal multiplier.
.macro LLM_ABSORB_SPAN state_a, state_b, exact_bytes, span_count
    cbz x1, Lllm_absorb_done\@

    mov x6, x1                  // Preserve exact span bytes for the state fold.
    mov x2, xzr                 // span_even
    mov x3, xzr                 // span_odd

Lllm_absorb_vector_loop\@:
    cmp x1, #32
    b.lo Lllm_absorb_16\@
    ldp q0, q1, [x0], #32
    // q0=[word0,word1], q1=[word2,word3]. Unzip by checksum parity.
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x4, d4
    fmov x5, d5
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #32
    b Lllm_absorb_vector_loop\@

Lllm_absorb_16\@:
    tbz x1, #4, Lllm_absorb_8\@
    ldp x4, x5, [x0], #16
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #16

Lllm_absorb_8\@:
    tbz x1, #3, Lllm_absorb_even_byte_tail\@
    ldr x4, [x0], #8
    add x2, x2, x4
    sub x1, x1, #8
    cbz x1, Lllm_absorb_fold\@

    // An 8-byte remainder preceded this partial word, so it is odd.
    mov x4, xzr
    mov x5, xzr
Lllm_absorb_odd_byte_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lllm_absorb_odd_byte_loop\@
    add x3, x3, x4
    b Lllm_absorb_fold\@

Lllm_absorb_even_byte_tail\@:
    cbz x1, Lllm_absorb_fold\@
    // No 8-byte remainder preceded this partial word, so it is even.
    mov x4, xzr
    mov x5, xzr
Lllm_absorb_even_byte_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lllm_absorb_even_byte_loop\@
    add x2, x2, x4

Lllm_absorb_fold\@:
    // span_count is the old zero-based ordinal; x7 is ordinal + 1.
    add x7, \span_count, #1
    mul x0, x7, x14
    add \state_a, \state_a, x2
    add \state_a, \state_a, x0
    ror \state_a, \state_a, #47  // rotl64(..., 17)

    LLM_LOAD_U64 x0, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    mul x0, x7, x0
    add \state_b, \state_b, x3
    add \state_b, \state_b, x6
    add \state_b, \state_b, x0
    ror \state_b, \state_b, #35  // rotl64(..., 29)

    add \exact_bytes, \exact_bytes, x6
    mov \span_count, x7

Lllm_absorb_done\@:
.endm

// Write one K or V append subrange. x8 is the sequence descriptor and x7 is
// the precomputed seed + step + layer + batch affine base. The macro preserves
// both registers. A full 32-byte body is stored through q6/q7; leading and
// trailing partial canonical words use only exact scalar stores.
.macro LLM_WRITE_APPEND pointer_offset, bytes_offset, domain0, domain1, domain2, domain3
    ldr x1, [x8, #\bytes_offset]
    cbz x1, Lllm_append_done\@
    ldr x0, [x8, #\pointer_offset]
    ldr x2, [x8, #72]           // Canonical append-record byte offset.

    LLM_LOAD_U64 x4, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    lsr x3, x2, #3
    add x3, x3, #1
    mul x3, x3, x4
    add x5, x7, x3
    LLM_LOAD_U64 x3, \domain0, \domain1, \domain2, \domain3
    add x5, x5, x3             // First canonical 64-bit append word.

    and x2, x2, #7
    cbz x2, Lllm_append_vector_loop\@

    // Peel bytes from the middle of the first canonical word. x6 is the exact
    // owned byte count in that word; no store covers an adjacent worker byte.
    mov x3, #8
    sub x6, x3, x2
    cmp x1, x6
    csel x6, x1, x6, lo
    add x3, x5, x4             // Preserve the next full canonical word.
    lsl x2, x2, #3
    lsrv x5, x5, x2

    tbz x6, #2, Lllm_append_lead_2\@
    str w5, [x0], #4
    lsr x5, x5, #32
Lllm_append_lead_2\@:
    tbz x6, #1, Lllm_append_lead_1\@
    strh w5, [x0], #2
    lsr x5, x5, #16
Lllm_append_lead_1\@:
    tbz x6, #0, Lllm_append_lead_complete\@
    strb w5, [x0], #1
Lllm_append_lead_complete\@:
    sub x1, x1, x6
    cbz x1, Lllm_append_done\@
    mov x5, x3

Lllm_append_vector_loop\@:
    cmp x1, #32
    b.lo Lllm_append_16\@
    add x3, x5, x4
    add x6, x3, x4
    add x2, x6, x4
    ins v6.d[0], x5
    ins v6.d[1], x3
    ins v7.d[0], x6
    ins v7.d[1], x2
    stp q6, q7, [x0], #32
    add x5, x2, x4
    sub x1, x1, #32
    b Lllm_append_vector_loop\@

Lllm_append_16\@:
    tbz x1, #4, Lllm_append_8\@
    add x3, x5, x4
    ins v6.d[0], x5
    ins v6.d[1], x3
    str q6, [x0], #16
    add x5, x3, x4
    sub x1, x1, #16

Lllm_append_8\@:
    tbz x1, #3, Lllm_append_tail\@
    str x5, [x0], #8
    add x5, x5, x4
    sub x1, x1, #8

Lllm_append_tail\@:
    // The remaining 0-7 bytes are the exact low prefix of the next word.
    tbz x1, #2, Lllm_append_tail_2\@
    str w5, [x0], #4
    lsr x5, x5, #32
Lllm_append_tail_2\@:
    tbz x1, #1, Lllm_append_tail_1\@
    strh w5, [x0], #2
    lsr x5, x5, #16
Lllm_append_tail_1\@:
    tbz x1, #0, Lllm_append_done\@
    strb w5, [x0]

Lllm_append_done\@:
.endm

.text
.p2align 4
.global _llm_decode_memory_asm
_llm_decode_memory_asm:
    // Without an output location there is no safe result to initialize.
    cbz x6, Lllm_return_without_frame

    // 96 bytes preserve x19-x30; the remaining slots retain immutable bases.
    stp x19, x20, [sp, #-128]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    str x0, [sp, #96]           // Layer base.
    str x1, [sp, #104]          // Sequence base.
    str x6, [sp, #112]          // Checksum output.

    // llm-read-checksum-v1 component initial states.
    LLM_LOAD_U64 x19, 0x57E2, 0xCDF7, 0x23CF, 0x737A
    LLM_LOAD_U64 x20, 0xD275, 0x4BC4, 0xD375, 0x6A5E
    mov x21, xzr
    mov x22, xzr

    LLM_LOAD_U64 x23, 0x57E2, 0xC4E7, 0x38CD, 0x6F60
    LLM_LOAD_U64 x24, 0xD275, 0x44B4, 0xDC73, 0x5E78
    mov x25, xzr
    mov x26, xzr

    LLM_LOAD_U64 x27, 0x57E2, 0xC4E7, 0x38CD, 0x7260
    LLM_LOAD_U64 x28, 0xD275, 0x44B4, 0xDC73, 0x6978
    mov x29, xzr
    mov x30, xzr

    // Preserve the remaining top-level inputs in registers not clobbered by
    // the span/append macros, then reject only safe top-level boundary cases.
    mov x9, x0                   // Layer base.
    mov x10, x1                  // Sequence base until the first KV layer.
    mov x12, x3                  // Steps remaining.
    mov x13, x4                  // Scenario flags.
    mov x15, x5                  // Becomes the current step affine base.
    cbz x9, Lllm_store_output
    cbz x2, Lllm_store_output
    cbz x12, Lllm_store_output
    cbz x13, Lllm_store_output
    cmp x13, #3
    b.hi Lllm_store_output
    tbz x13, #1, Lllm_top_level_valid
    cbz x10, Lllm_store_output

Lllm_top_level_valid:
    // Layer end = layer base + layer_count * 48. Valid materialized plans have
    // already checked all size/count arithmetic before entering the timer.
    lsl x7, x2, #4
    add x7, x7, x2, lsl #5
    add x11, x9, x7

    // task_affine(step 0) = scenario_seed + step_multiplier * (0 + 1).
    LLM_LOAD_U64 x14, 0x7C15, 0x7F4A, 0x79B9, 0x9E37
    add x15, x15, x14

Lllm_step_loop:
    mov x16, x9                  // Current layer descriptor.

Lllm_layer_loop:
    cmp x16, x11
    b.eq Lllm_step_complete

    tbz x13, #0, Lllm_layer_kv
    ldp x0, x1, [x16, #0]
    LLM_ABSORB_SPAN x19, x20, x21, x22

Lllm_layer_kv:
    tbz x13, #1, Lllm_next_layer
    ldp x0, x1, [x16, #16]     // First sequence index and count.
    cbz x1, Lllm_next_layer

    // Sequence pointer = base + first_index * 80; end uses count * 80.
    ldr x3, [sp, #104]
    lsl x2, x0, #4
    add x2, x2, x0, lsl #6
    add x8, x3, x2
    lsl x2, x1, #4
    add x2, x2, x1, lsl #6
    add x10, x8, x2

Lllm_sequence_loop:
    // Common affine base excludes the K/V domain and canonical word index.
    ldr x0, [x16, #32]
    add x0, x0, #1
    LLM_LOAD_U64 x2, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x7, x0, x2, x15
    ldr x1, [x8, #64]
    add x1, x1, #1
    LLM_LOAD_U64 x2, 0x11EB, 0x1331, 0x49BB, 0x94D0
    madd x7, x1, x2, x7

    // Temporal append order is K then V. Domains are little-endian constants.
    LLM_WRITE_APPEND 32, 40, 0x4B4B, 0x4B4B, 0x4B4B, 0x4B4B
    LLM_WRITE_APPEND 48, 56, 0x5656, 0x5656, 0x5656, 0x5656

    ldp x0, x1, [x8, #0]
    LLM_ABSORB_SPAN x23, x24, x25, x26
    ldp x0, x1, [x8, #16]
    LLM_ABSORB_SPAN x27, x28, x29, x30

    add x8, x8, #80
    cmp x8, x10
    b.lo Lllm_sequence_loop

Lllm_next_layer:
    add x16, x16, #48
    b Lllm_layer_loop

Lllm_step_complete:
    // Incrementing the affine base is exactly step_multiplier modulo 2^64.
    add x15, x15, x14
    subs x12, x12, #1
    b.ne Lllm_step_loop

Lllm_store_output:
    ldr x6, [sp, #112]
    stp x19, x20, [x6, #0]
    stp x21, x22, [x6, #16]
    stp x23, x24, [x6, #32]
    stp x25, x26, [x6, #48]
    stp x27, x28, [x6, #64]
    stp x29, x30, [x6, #80]

    ldp x29, x30, [sp, #80]
    ldp x27, x28, [sp, #64]
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #128

Lllm_return_without_frame:
    ret
