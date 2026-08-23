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
// llm_prefill_memory_asm
// -----------------------------------------------------------------------------
// C++ prototype:
//   extern "C" void llm_prefill_memory_asm(
//       const LlmPrefillLayerDescriptor* layers,
//       const LlmPrefillKvSequenceDescriptor* sequences,
//       uint64_t layer_count,
//       uint64_t operation_count,
//       uint64_t scenario_flags,
//       uint64_t scenario_seed,
//       LlmWorkerChecksum* output) noexcept;
//
// Arguments:
//   x0 = layer descriptor array
//   x1 = scenario-selected KV owner descriptor array
//   x2 = layer descriptor count
//   x3 = task-local prefill operation count
//   x4 = scenario flags: bit 0 = weights, bit 1 = KV (valid values 1..3)
//   x5 = scenario-domain seed for the affine prompt pattern
//   x6 = 16-byte-aligned worker checksum output
//
// Frozen descriptor ABI (llm-memory-prefill-contiguous-descriptor-abi-v1):
//   LlmPrefillLayerDescriptor, stride 48:
//      0 weight_ptr                 8 weight_bytes
//     16 first_sequence_index     24 sequence_count
//     32 layer_index              40 reserved_zero
//   LlmPrefillKvSequenceDescriptor, stride 80:
//      0 k_owned_ptr                8 v_owned_ptr
//     16 first_token               24 owned_token_count
//     32 prompt_tokens             40 attention_query_tile_tokens
//     48 record_bytes              56 layer_index
//     64 batch_sequence_index      72 reserved_zero
//
// Frozen output ABI (three consecutive 32-byte components):
//   weight offsets 0/8/16/24, K offsets 32/40/48/56,
//   V offsets 64/72/80/88; fields are state_a, state_b,
//   exact_bytes_read, and span_count.
//
// Traversal, write, and checksum semantics:
//   * Each operation visits layers in descriptor order. A weight-enabled layer
//     absorbs its shard exactly once before the layer's batch descriptors.
//   * A non-empty owner descriptor walks its tokens in increasing order and
//     writes each complete K record followed by that token's complete V record
//     with llm-prefill-kv-affine64-v1. It then advances tile ends by
//     min(Q, P - current_end). Every non-empty owner/prefix intersection is
//     scanned K first and V second.
//   * The K/V state lanes implement llm-prefill-affine64-parity-sum-v1 over
//     the bytes actually loaded by every tile scan. state_a is the sum of
//     canonical even logical words and state_b the canonical odd-word sum.
//     A partial canonical word contributes only loaded bytes in their original
//     little-endian bit positions; parity never restarts at a descriptor span.
//   * exact_bytes_read and span_count describe those same non-empty tile scan
//     visits. Prompt writes do not add a separate checksum contribution.
//   * Vector bodies use exact 32-byte ldp/stp operations. Unaligned Normal-
//     memory access is intentional; leading and trailing fragments use exact
//     scalar byte/word operations and never touch an adjacent owner range.
//   * Worker ranges are disjoint, so the kernel has no internal barrier. The
//     executor owns task barriers, timing fences, stop handling, and excluded
//     final-byte validation for operation ordinal T - 1.
//
// AAPCS64 and register use:
//   * x19-x30 are saved and restored; q8-q15 and platform register x18 are
//     untouched. Caller-saved q0-q7 hold load/store vectors.
//   * x19-x22 = weight checksum, x23-x26 = K checksum,
//     x27-x30 = V checksum, in output-field order.
//   * x9-x17 retain loop state. The span/write macros clobber only x0-x8 and
//     q0-q7. The 192-byte frame is 16-byte aligned.
//   * This is a leaf function. Incoming x30 is saved before x30 is reused for
//     the V span counter.
//
// Safe direct-boundary behavior:
//   A null output returns immediately. With a valid output, all components are
//   initialized before top-level validation. Null layers, invalid flags, zero
//   layers, or zero operations return the initialized result. weights_only
//   never dereferences the KV descriptor pointer; KV-active work requires it.
//   Materialized inner descriptors are trusted executor input.
// -----------------------------------------------------------------------------

.macro LLM_PREFILL_LOAD_U64 reg, half0, half1, half2, half3
    movz \reg, #\half0
    movk \reg, #\half1, lsl #16
    movk \reg, #\half2, lsl #32
    movk \reg, #\half3, lsl #48
.endm

// Absorb one static weight span with the frozen llm-read-checksum-v1 fold.
// x14 permanently holds the span-ordinal multiplier.
.macro LLM_PREFILL_ABSORB_WEIGHT state_a, state_b, exact_bytes, span_count
    cbz x1, Lprefill_weight_done\@

    mov x6, x1
    mov x2, xzr
    mov x3, xzr

Lprefill_weight_vector\@:
    cmp x1, #32
    b.lo Lprefill_weight_16\@
    ldp q0, q1, [x0], #32
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x4, d4
    fmov x5, d5
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #32
    b Lprefill_weight_vector\@

Lprefill_weight_16\@:
    tbz x1, #4, Lprefill_weight_8\@
    ldp x4, x5, [x0], #16
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #16

Lprefill_weight_8\@:
    tbz x1, #3, Lprefill_weight_even_tail\@
    ldr x4, [x0], #8
    add x2, x2, x4
    sub x1, x1, #8
    cbz x1, Lprefill_weight_fold\@
    mov x4, xzr
    mov x5, xzr
Lprefill_weight_odd_tail_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lprefill_weight_odd_tail_loop\@
    add x3, x3, x4
    b Lprefill_weight_fold\@

Lprefill_weight_even_tail\@:
    cbz x1, Lprefill_weight_fold\@
    mov x4, xzr
    mov x5, xzr
Lprefill_weight_even_tail_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lprefill_weight_even_tail_loop\@
    add x2, x2, x4

Lprefill_weight_fold\@:
    add x7, \span_count, #1
    mul x0, x7, x14
    add \state_a, \state_a, x2
    add \state_a, \state_a, x0
    ror \state_a, \state_a, #47

    LLM_PREFILL_LOAD_U64 x0, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    mul x0, x7, x0
    add \state_b, \state_b, x3
    add \state_b, \state_b, x6
    add \state_b, \state_b, x0
    ror \state_b, \state_b, #35

    add \exact_bytes, \exact_bytes, x6
    mov \span_count, x7

Lprefill_weight_done\@:
.endm

// Write one token record. Frame slots 144/152/160/184 contain the common
// affine base, owner canonical first byte, record bytes, and current physical
// byte offset within the owner range. x17 is the owner descriptor. Only exact
// bytes are stored; no read checksum is changed.
.macro LLM_PREFILL_WRITE_OWNED pointer_offset, domain0, domain1, domain2, domain3
    ldr x1, [sp, #160]
    cbz x1, Lprefill_write_done\@
    ldr x0, [x17, #\pointer_offset]
    ldr x6, [sp, #184]
    add x0, x0, x6
    ldr x2, [sp, #152]
    add x2, x2, x6
    ldr x7, [sp, #144]
    LLM_PREFILL_LOAD_U64 x4, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    lsr x3, x2, #3
    add x3, x3, #1
    madd x3, x3, x4, x7
    LLM_PREFILL_LOAD_U64 x5, \domain0, \domain1, \domain2, \domain3
    add x3, x3, x5

    and x2, x2, #7
    cbz x2, Lprefill_write_vector\@
    mov x5, #8
    sub x5, x5, x2
    cmp x1, x5
    csel x5, x1, x5, lo
    add x6, x3, x4
    lsl x2, x2, #3
    lsrv x3, x3, x2
    tbz x5, #2, Lprefill_write_lead_2\@
    str w3, [x0], #4
    lsr x3, x3, #32
Lprefill_write_lead_2\@:
    tbz x5, #1, Lprefill_write_lead_1\@
    strh w3, [x0], #2
    lsr x3, x3, #16
Lprefill_write_lead_1\@:
    tbz x5, #0, Lprefill_write_lead_complete\@
    strb w3, [x0], #1
Lprefill_write_lead_complete\@:
    sub x1, x1, x5
    cbz x1, Lprefill_write_done\@
    mov x3, x6

Lprefill_write_vector\@:
    cmp x1, #32
    b.lo Lprefill_write_16\@
    add x5, x3, x4
    add x6, x5, x4
    add x7, x6, x4
    ins v6.d[0], x3
    ins v6.d[1], x5
    ins v7.d[0], x6
    ins v7.d[1], x7
    stp q6, q7, [x0], #32
    add x3, x7, x4
    sub x1, x1, #32
    b Lprefill_write_vector\@

Lprefill_write_16\@:
    tbz x1, #4, Lprefill_write_8\@
    add x5, x3, x4
    ins v6.d[0], x3
    ins v6.d[1], x5
    str q6, [x0], #16
    add x3, x5, x4
    sub x1, x1, #16

Lprefill_write_8\@:
    tbz x1, #3, Lprefill_write_tail\@
    str x3, [x0], #8
    add x3, x3, x4
    sub x1, x1, #8

Lprefill_write_tail\@:
    tbz x1, #2, Lprefill_write_tail_2\@
    str w3, [x0], #4
    lsr x3, x3, #32
Lprefill_write_tail_2\@:
    tbz x1, #1, Lprefill_write_tail_1\@
    strh w3, [x0], #2
    lsr x3, x3, #16
Lprefill_write_tail_1\@:
    tbz x1, #0, Lprefill_write_done\@
    strb w3, [x0]

Lprefill_write_done\@:
.endm

// Scan the current non-empty tile intersection. Frame slot 152 is the
// canonical first byte and slot 176 the exact span length. Canonical word
// parity is retained across leading/tail fragments instead of restarting at
// the physical span pointer.
.macro LLM_PREFILL_SCAN_OWNED pointer_offset, state_even, state_odd, exact_bytes, span_count
    ldr x1, [sp, #176]
    cbz x1, Lprefill_scan_done\@
    add \exact_bytes, \exact_bytes, x1
    add \span_count, \span_count, #1
    ldr x0, [x17, #\pointer_offset]
    ldr x2, [sp, #152]
    lsr x4, x2, #3
    and x3, x2, #7
    cbz x3, Lprefill_scan_vector\@

    mov x7, #8
    sub x7, x7, x3
    cmp x1, x7
    csel x7, x1, x7, lo
    mov x8, x7
Lprefill_scan_leading_loop\@:
    ldrb w5, [x0], #1
    lsl x6, x3, #3
    lslv x5, x5, x6
    tbnz x4, #0, Lprefill_scan_leading_odd\@
    add \state_even, \state_even, x5
    b Lprefill_scan_leading_added\@
Lprefill_scan_leading_odd\@:
    add \state_odd, \state_odd, x5
Lprefill_scan_leading_added\@:
    add x3, x3, #1
    subs x8, x8, #1
    b.ne Lprefill_scan_leading_loop\@
    sub x1, x1, x7
    cbz x1, Lprefill_scan_done\@
    add x4, x4, #1

Lprefill_scan_vector\@:
    cmp x1, #32
    b.lo Lprefill_scan_16\@
    ldp q0, q1, [x0], #32
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x5, d4
    fmov x6, d5
    tbnz x4, #0, Lprefill_scan_vector_odd\@
    add \state_even, \state_even, x5
    add \state_odd, \state_odd, x6
    b Lprefill_scan_vector_added\@
Lprefill_scan_vector_odd\@:
    add \state_even, \state_even, x6
    add \state_odd, \state_odd, x5
Lprefill_scan_vector_added\@:
    add x4, x4, #4
    sub x1, x1, #32
    b Lprefill_scan_vector\@

Lprefill_scan_16\@:
    tbz x1, #4, Lprefill_scan_8\@
    ldp x5, x6, [x0], #16
    tbnz x4, #0, Lprefill_scan_16_odd\@
    add \state_even, \state_even, x5
    add \state_odd, \state_odd, x6
    b Lprefill_scan_16_added\@
Lprefill_scan_16_odd\@:
    add \state_even, \state_even, x6
    add \state_odd, \state_odd, x5
Lprefill_scan_16_added\@:
    add x4, x4, #2
    sub x1, x1, #16

Lprefill_scan_8\@:
    tbz x1, #3, Lprefill_scan_tail\@
    ldr x5, [x0], #8
    tbnz x4, #0, Lprefill_scan_8_odd\@
    add \state_even, \state_even, x5
    b Lprefill_scan_8_added\@
Lprefill_scan_8_odd\@:
    add \state_odd, \state_odd, x5
Lprefill_scan_8_added\@:
    add x4, x4, #1
    sub x1, x1, #8

Lprefill_scan_tail\@:
    cbz x1, Lprefill_scan_done\@
    mov x3, xzr
Lprefill_scan_tail_loop\@:
    ldrb w5, [x0], #1
    lsl x6, x3, #3
    lslv x5, x5, x6
    tbnz x4, #0, Lprefill_scan_tail_odd\@
    add \state_even, \state_even, x5
    b Lprefill_scan_tail_added\@
Lprefill_scan_tail_odd\@:
    add \state_odd, \state_odd, x5
Lprefill_scan_tail_added\@:
    add x3, x3, #1
    subs x1, x1, #1
    b.ne Lprefill_scan_tail_loop\@

Lprefill_scan_done\@:
.endm

.text
.p2align 4
.global _llm_prefill_memory_asm
_llm_prefill_memory_asm:
    cbz x6, Lprefill_return_without_frame

    stp x19, x20, [sp, #-192]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    str x0, [sp, #96]
    str x1, [sp, #104]
    str x6, [sp, #112]

    // Weight retains llm-read-checksum-v1. K/V use raw canonical parity sums.
    LLM_PREFILL_LOAD_U64 x19, 0x57E2, 0xCDF7, 0x23CF, 0x737A
    LLM_PREFILL_LOAD_U64 x20, 0xD275, 0x4BC4, 0xD375, 0x6A5E
    mov x21, xzr
    mov x22, xzr
    mov x23, xzr
    mov x24, xzr
    mov x25, xzr
    mov x26, xzr
    mov x27, xzr
    mov x28, xzr
    mov x29, xzr
    mov x30, xzr

    mov x9, x0
    mov x10, x1
    mov x12, x3
    mov x13, x4
    mov x15, x5
    cbz x9, Lprefill_store_output
    cbz x2, Lprefill_store_output
    cbz x12, Lprefill_store_output
    cbz x13, Lprefill_store_output
    cmp x13, #3
    b.hi Lprefill_store_output
    tbz x13, #1, Lprefill_top_level_valid
    cbz x10, Lprefill_store_output

Lprefill_top_level_valid:
    lsl x7, x2, #4
    add x7, x7, x2, lsl #5
    add x11, x9, x7

    LLM_PREFILL_LOAD_U64 x14, 0x7C15, 0x7F4A, 0x79B9, 0x9E37
    LLM_PREFILL_LOAD_U64 x7, 0x4C31, 0x494C, 0x4546, 0x5052
    add x15, x15, x7
    add x15, x15, x14

Lprefill_operation_loop:
    mov x16, x9

Lprefill_layer_loop:
    cmp x16, x11
    b.eq Lprefill_operation_complete

    tbz x13, #0, Lprefill_layer_kv
    ldp x0, x1, [x16, #0]
    LLM_PREFILL_ABSORB_WEIGHT x19, x20, x21, x22

Lprefill_layer_kv:
    // The KV flag is tested before any descriptor-base dereference.
    tbz x13, #1, Lprefill_next_layer
    ldp x0, x1, [x16, #16]
    cbz x1, Lprefill_next_layer
    lsl x2, x0, #4
    add x2, x2, x0, lsl #6
    add x17, x10, x2
    lsl x2, x1, #4
    add x2, x2, x1, lsl #6
    add x2, x17, x2
    str x2, [sp, #128]

Lprefill_sequence_loop:
    ldr x0, [x17, #24]
    cbz x0, Lprefill_next_sequence
    str x0, [sp, #136]
    ldr x1, [x17, #16]
    ldr x2, [x17, #48]
    mul x3, x1, x2
    str x3, [sp, #152]
    str x2, [sp, #160]
    str xzr, [sp, #184]

    // Common affine base excludes only the K/V domain and logical word.
    ldr x0, [x17, #56]
    add x0, x0, #1
    LLM_PREFILL_LOAD_U64 x2, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x7, x0, x2, x15
    ldr x1, [x17, #64]
    add x1, x1, #1
    LLM_PREFILL_LOAD_U64 x2, 0x11EB, 0x1331, 0x49BB, 0x94D0
    madd x7, x1, x2, x7
    str x7, [sp, #144]

    // Source-audit anchors: tokens increase monotonically and each token's K
    // record is complete before its V record; all writes precede tile scans.
Lprefill_owner_write_phase:
Lprefill_owner_token_write_loop:
    LLM_PREFILL_WRITE_OWNED 0, 0x4B4B, 0x4B4B, 0x4B4B, 0x4B4B
    LLM_PREFILL_WRITE_OWNED 8, 0x5656, 0x5656, 0x5656, 0x5656
    ldr x0, [sp, #184]
    ldr x1, [sp, #160]
    add x0, x0, x1
    str x0, [sp, #184]
    ldr x0, [sp, #136]
    subs x0, x0, #1
    str x0, [sp, #136]
    b.ne Lprefill_owner_token_write_loop

    ldr x0, [x17, #16]
    ldr x1, [x17, #24]
    add x0, x0, x1
    str x0, [sp, #136]

    // Skip tiles whose prefix cannot intersect this owner. The first executed
    // iteration is floor(first_token / Q), calculated without a growing
    // multiplication that could overflow.
    ldr x0, [x17, #16]
    ldr x1, [x17, #40]
    udiv x2, x0, x1
    mul x2, x2, x1
    str x2, [sp, #168]

Lprefill_tile_loop:
    ldr x0, [x17, #32]
    ldr x2, [sp, #168]
    sub x3, x0, x2
    ldr x1, [x17, #40]
    cmp x3, x1
    csel x3, x3, x1, lo
    add x2, x2, x3
    str x2, [sp, #168]

    ldr x0, [x17, #16]
    cmp x2, x0
    b.ls Lprefill_next_tile
    ldr x1, [sp, #136]
    cmp x2, x1
    csel x2, x2, x1, lo
    sub x2, x2, x0
    ldr x3, [x17, #48]
    mul x2, x2, x3
    str x2, [sp, #176]

    // Source-audit anchors: each tile scans the complete K intersection before
    // restarting the same intersection for V.
Lprefill_tile_k_scan:
    LLM_PREFILL_SCAN_OWNED 0, x23, x24, x25, x26
Lprefill_tile_v_scan:
    LLM_PREFILL_SCAN_OWNED 8, x27, x28, x29, x30

Lprefill_next_tile:
    ldr x2, [sp, #168]
    ldr x0, [x17, #32]
    cmp x2, x0
    b.lo Lprefill_tile_loop

Lprefill_next_sequence:
    add x17, x17, #80
    ldr x2, [sp, #128]
    cmp x17, x2
    b.lo Lprefill_sequence_loop

Lprefill_next_layer:
    add x16, x16, #48
    b Lprefill_layer_loop

Lprefill_operation_complete:
    add x15, x15, x14
    subs x12, x12, #1
    b.ne Lprefill_operation_loop

Lprefill_store_output:
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
    ldp x19, x20, [sp], #192

Lprefill_return_without_frame:
    ret
