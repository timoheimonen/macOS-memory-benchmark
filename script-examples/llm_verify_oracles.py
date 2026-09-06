"""Independent bounded arithmetic for LLM schema 2 (no producer imports).

All modular arithmetic is explicit. Ranges are summed from affine words, not
materialized buffers; the caller supplies a shared operation budget.
"""

from functools import lru_cache

U64 = (1 << 64) - 1
U32 = (1 << 32) - 1
A = 0x9E3779B97F4A7C15
B = 0xBF58476D1CE4E5B9
C = 0x94D049BB133111EB
D = 0xD6E8FEB86659FD93
DOMAINS = (0x5745494748545F31, 0x4B5F524541445F31, 0x565F524541445F31)
SEED_DOMAINS = (
    0x4C4C4D5745494748,
    0x4C4C4D4B42554631,
    0x4C4C4D5642554631,
    0x4C4C4D5357454947,
    0x4C4C4D534B564F4E,
    0x4C4C4D534D495845,
)


def splitmix(value):
    value = (value + A) & U64
    value = ((value ^ (value >> 30)) * B) & U64
    value = ((value ^ (value >> 27)) * C) & U64
    return value ^ (value >> 31)


def permutation(seed, count, budget):
    budget(count)
    values = list(range(count))
    state = splitmix(seed ^ 0x4C4C4D4B56504731)
    for bound in range(count, 1, -1):
        while True:
            draw = splitmix(state)
            state = (state + A) & U64
            if draw >= (1 << 64) % bound:
                break
            budget(1)
        j = draw % bound
        values[bound - 1], values[j] = values[j], values[bound - 1]
    return values


def rot(value, shift):
    value &= U64
    return ((value << shift) | (value >> (64 - shift))) & U64


def partition(offset, size, workers):
    """Nearest absolute 32-byte boundary, ties down, preserving nonempty owners."""
    active = min(size, workers)
    if not active:
        return [(0, 0)] * workers
    boundaries = [0]
    for rank in range(1, active):
        target = size // active * rank + min(rank, size % active)
        lo, hi = boundaries[-1] + 1, size - (active - rank)
        aligned_lo, aligned_hi = ((offset + lo + 31) // 32) * 32, ((offset + hi) // 32) * 32
        if aligned_lo <= aligned_hi:
            down = max(aligned_lo, min(aligned_hi, (offset + target) // 32 * 32))
            up = min(aligned_hi, down + 32) if down < offset + target else down
            target = min((down, up), key=lambda x: (abs(x - offset - target), x)) - offset
        boundaries.append(target)
    boundaries.append(size)
    return [(offset + a, b - a) for a, b in zip(boundaries, boundaries[1:])] + [(0, 0)] * (workers - active)


def affine_slice(base, step, start, count, bits=64, parity_origin=None):
    """Sum clipped canonical words; parity_origin=None means absolute word parity.

    With parity_origin set, assemble little-endian words starting at that byte
    boundary. Only unaligned spans need bounded scalar reads (caller budgets).
    """
    word_bytes = bits // 8
    mask = (1 << bits) - 1
    sums = [0, 0]
    end = start + count
    origin = start if parity_origin is not None else start // word_bytes * word_bytes
    if parity_origin is not None and start % word_bytes:
        for pos in range(start, end, word_bytes):
            value = 0
            for b in range(min(word_bytes, end - pos)):
                absolute = pos + b
                word = (base + step * (absolute // word_bytes + 1)) & mask
                value |= ((word >> (8 * (absolute % word_bytes))) & 255) << (8 * b)
            sums[((pos - start) // word_bytes) % 2] += value
        return [v & mask for v in sums]
    first, last = (start + word_bytes - 1) // word_bytes, end // word_bytes
    for parity in range(2):
        first_p = first + ((parity - first) % 2)
        n = max(0, (last - first_p + 1) // 2)
        total = n * base + step * (n * (first_p + 1) + n * (n - 1))
        destination = (parity - (origin // word_bytes if parity_origin is not None else 0)) % 2
        sums[destination] += total
    for word in set((start // word_bytes, end // word_bytes)):
        low, high = max(start, word * word_bytes), min(end, (word + 1) * word_bytes)
        if low >= high or (low == word * word_bytes and high == (word + 1) * word_bytes):
            continue
        value = (base + step * (word + 1)) & mask
        byte_mask = ((1 << ((high - low) * 8)) - 1) << ((low - word * word_bytes) * 8)
        value &= byte_mask
        destination = (word - (origin // word_bytes if parity_origin is not None else 0)) % 2
        sums[destination] += value
    return [v & mask for v in sums]


def absorb(state, even, odd, count):
    if count:
        state[0] = rot(state[0] + even + A * (state[3] + 1), 17)
        state[1] = rot(state[1] + odd + count + D * (state[3] + 1), 29)
        state[2] += count
        state[3] += 1


def lookup(state, logical, physical, kind, work):
    term = (logical + 1) * (physical + 1) + (kind + 1) * C + (work + 1) * B
    state[0] = rot(state[0] + term + A, 13)
    state[1] = rot(state[1] ^ ((term + D) & U64), 31)


def fold(workers):
    a, b, ordinal = 0x6A09E667F3BCC909, 0xBB67AE8584CAA73B, 0
    for worker in workers:
        for x, y, count, spans in worker:
            a = rot(a + x + A * (ordinal + 1), 23)
            b = rot(b + y + count + D * spans + C * (ordinal + 1), 41)
            ordinal += 1
    return {"state_a_uint64_decimal": str(a), "state_b_uint64_decimal": str(b)}


def cpu_checksum(model, scenario, work, budget):
    """Reconstruct worker checksums once per canonical scenario/T plan."""
    g = model
    workers, layers, batch, record, tokens = (
        g[k] for k in ("workers", "layers", "batch", "record", "tokens")
    )
    weights = []
    offset = 0
    for layer in range(layers):
        size = g["weight"] // layers + (layer < g["weight"] % layers)
        weights.append(partition(offset, size, workers))
        offset += size
    budget(
        work
        * workers
        * layers
        * (
            1
            + batch
            * (
                (g["blocks"] * len(g["tile_ends"]) + g["blocks"])
                if g["prefill"] and g["paged"]
                else tokens if g["prefill"] else g["blocks"] or 1
            )
        )
    )
    states = [
        [[0x243F6A8885A308D3 ^ d, (0x13198A2E03707344 + d) & U64, 0, 0] for d in DOMAINS]
        for _ in range(workers)
    ]
    if g["prefill"]:
        for state in states:
            state[1:] = [[0, 0, 0, 0], [0, 0, 0, 0]]
    seed = g["scenario_seeds"][scenario]

    @lru_cache(maxsize=4096)
    def reference(base, step, first, size):
        if first % 8:
            budget((size + 7) // 8)
        return affine_slice(base, step, first, size, parity_origin=first)

    def read_decode(state, base, step, first, size, append_start, append_base):
        # Exact replacement of only the current record, preserving span parity.
        even, odd = reference(base, step, first, size)
        lo, hi = max(first, append_start), min(first + size, append_start + record)
        if lo < hi:
            budget((hi - lo + 7) // 8 + 2)
            # Only intersecting span words need replacement, including unaligned tails.
            for pos in range(first + ((lo - first) // 8) * 8, hi, 8):
                old = new = 0
                for byte in range(min(8, first + size - pos)):
                    address = pos + byte
                    value = ((base + step * (address // 8 + 1)) & U64) >> (8 * (address % 8)) & 255
                    replacement = value
                    if append_start <= address < append_start + record:
                        local = address - append_start
                        replacement = ((append_base + D * (local // 8 + 1)) & U64) >> (8 * (local % 8)) & 255
                    old |= value << (8 * byte)
                    new |= replacement << (8 * byte)
                if ((pos - first) // 8) % 2:
                    odd = (odd + new - old) & U64
                else:
                    even = (even + new - old) & U64
        absorb(state, even, odd, size)

    for w, state in enumerate(states):
        for unit in range(work):
            for layer in range(layers):
                start, size = weights[layer][w]
                if scenario != "kv_only" and size:
                    even, odd = reference(g["buffer_seeds"][0], A, start, size)
                    absorb(state[0], even, odd, size)
                if scenario == "weights_only":
                    continue
                for b in range(batch):
                    row = layer * batch + b
                    if g["prefill"]:
                        first, count = g["ownership"][scenario][row][w]
                        if not count:
                            continue
                        first_token = first * g["block_tokens"] if g["paged"] else first
                        last_token = (
                            min(tokens, (first + count) * g["block_tokens"]) if g["paged"] else first + count
                        )
                        if g["paged"]:
                            for block in range(first, first + count):
                                logical = b * g["blocks"] + block
                                lookup(state[1], logical, g["table"][logical], 0, unit)
                        for end in g["tile_ends"]:
                            if end <= first_token:
                                continue
                            for pool in (1, 2):
                                base = (
                                    seed
                                    + 0x50524546494C4C31
                                    + A * (unit + 1)
                                    + B * (layer + 1)
                                    + C * (b + 1)
                                    + (0x4B4B4B4B4B4B4B4B if pool == 1 else 0x5656565656565656)
                                )
                                ranges = [
                                    (first_token * record, (min(end, last_token) - first_token) * record)
                                ]
                                if g["paged"]:
                                    ranges = []
                                    for block in range(
                                        first,
                                        min(
                                            first + count, (end + g["block_tokens"] - 1) // g["block_tokens"]
                                        ),
                                    ):
                                        logical = b * g["blocks"] + block
                                        ranges.append(
                                            (
                                                block * g["block_bytes"],
                                                min(
                                                    g["block_bytes"], end * record - block * g["block_bytes"]
                                                ),
                                            )
                                        )
                                for pos, length in ranges:
                                    if g["paged"]:
                                        logical = b * g["blocks"] + pos // g["block_bytes"]
                                        lookup(state[pool], logical, g["table"][logical], pool, unit)
                                    even, odd = affine_slice(base, D, pos, length)
                                    state[pool][0] = (state[pool][0] + even) & U64
                                    state[pool][1] = (state[pool][1] + odd) & U64
                                    state[pool][2] += length
                                    state[pool][3] += 1
                    elif g["paged"]:
                        first, count = g["ownership"][scenario][row][w]
                        if not count:
                            continue
                        if first + count == g["blocks"]:
                            logical = b * g["blocks"] + g["blocks"] - 1
                            lookup(state[1], logical, g["table"][logical], 0, unit)
                        for pool in (1, 2):
                            for block in range(first, first + count):
                                logical = b * g["blocks"] + block
                                physical = g["table"][logical]
                                lookup(state[pool], logical, physical, pool, unit)
                                size = min(g["block_tokens"], tokens - block * g["block_tokens"]) * record
                                base = (
                                    g["buffer_seeds"][pool]
                                    + 0xA24BAED4963EE407 * (layer + 1)
                                    + 0x9FB21C651E98DF25 * (physical + 1)
                                )
                                append_base = (
                                    seed
                                    + A * (unit + 1)
                                    + B * (layer + 1)
                                    + C * (b + 1)
                                    + (0x4B4B4B4B4B4B4B4B if pool == 1 else 0x5656565656565656)
                                )
                                read_decode(
                                    state[pool],
                                    base,
                                    0xC13FA9A902A6328F,
                                    0,
                                    size,
                                    size - record if block == g["blocks"] - 1 else size,
                                    append_base,
                                )
                    else:
                        first, size = partition(row * tokens * record, tokens * record, workers)[w]
                        for pool in (1, 2):
                            append_base = (
                                seed
                                + A * (unit + 1)
                                + B * (layer + 1)
                                + C * (b + 1)
                                + (0x4B4B4B4B4B4B4B4B if pool == 1 else 0x5656565656565656)
                            )
                            read_decode(
                                state[pool],
                                g["buffer_seeds"][pool],
                                A,
                                first,
                                size,
                                (row * tokens + tokens - 1) * record,
                                append_base,
                            )
    keys = (
        "state_a_uint64_decimal",
        "state_b_uint64_decimal",
        "exact_bytes_read",
        "span_count_uint64_decimal",
    )
    output = [
        dict(
            worker_index=i,
            **{name: dict(zip(keys, map(str, values))) for name, values in zip(("weight", "k", "v"), worker)}
        )
        for i, worker in enumerate(states)
    ]
    return output, fold(states)


def metal_checksum(g, scenario, work, budget):
    """Scalar semantic visits with O(1) affine sums per range; no GPU helpers."""
    profile = {
        (False, False): 0x4D444331,
        (False, True): 0x4D445031,
        (True, False): 0x4D504331,
        (True, True): 0x4D505031,
    }[g["prefill"], g["paged"]]
    seed = g["scenario_seeds"][scenario]
    states = [[0, 0] for _ in range(3)]
    pools = (0x57474854, 0x4B455943, 0x56414C43)
    budget(work * g["layers"] * (1 + g["batch"] * (g["tokens"] + len(g["tile_ends"]) * (g["blocks"] or 1))))

    def domain(pool, visit, u, l, b, tile, mask, logical=None, physical=None, address=False):
        v = (
            profile
            + (seed & U32)
            + 0xA24BAED5 * (seed >> 32)
            + pools[pool]
            + visit
            + 0xC2B2AE3D * (u + 1)
            + 0x27D4EB35 * (l + 1)
            + 0x165667C5 * (b + 1)
            + 0xD1B54A35 * tile
            + 0xD3A2646D * mask
        )
        if logical is not None:
            v += (
                0x7F4A7C15 * (logical + 1)
                + 0x94D049BB * (physical + 1)
                + 0x369DEA0F * (logical + 1) * (physical + 1)
            )
            if address:
                global_block = l * g["batch"] * g["blocks"] + physical
                per_segment = (256 * 1024 * 1024) // g["block_bytes"]
                segment = global_block // per_segment
                local = (global_block % per_segment) * g["block_bytes"]
                absolute = global_block * g["block_bytes"]
                token = (
                    0x41444452
                    + 0xDB4F0B91 * (global_block & U32)
                    + 0xBBE05633 * (global_block >> 32)
                    + 0xA0F2EC75 * segment
                    + 0x89E18285 * (local & U32)
                    + 0xC6D1D6C9 * (local >> 32)
                    + 0xB492B66F * (absolute & U32)
                    + 0x9AE16A3B * (absolute >> 32)
                ) & U32
                v += token + 0xD6E8FEB9 * (logical + 1) * token
        return v & U32

    def mix(pool, value, indices, domains):
        states[pool][0] = (states[pool][0] + value + domains) & U32
        states[pool][1] = (
            states[pool][1] + value * 0x9E3779B1 + indices * 0x85EBCA77 + domains * 0x7FEB352D
        ) & U32

    def ranged(
        pool, base, step, start, length, u, l, b, visit, tile=0, logical=None, physical=None, index_base=0
    ):
        if not length:
            return
        first, last = (start + 3) // 4, (start + length) // 4
        count = max(0, last - first)
        indexsum = count * first + count * (count - 1) // 2
        values = count * base + step * (indexsum + count)
        mix(
            pool,
            values,
            indexsum + index_base * count,
            count * domain(pool, visit, u, l, b, tile, 15, logical, physical),
        )
        for i in set((start // 4, (start + length) // 4)):
            lo, hi = max(start, 4 * i), min(start + length, 4 * i + 4)
            if lo >= hi or (lo == 4 * i and hi == 4 * i + 4):
                continue
            mask = ((1 << (hi - lo)) - 1) << (lo - 4 * i)
            byte_mask = sum(255 << (8 * j) for j in range(4) if mask & (1 << j))
            value = ((base + step * (i + 1)) & U32) & byte_mask
            mix(pool, value, i + index_base, domain(pool, visit, u, l, b, tile, mask, logical, physical))

    def table_visit(pool, kind, u, l, b, block, tile=0):
        logical = b * g["blocks"] + block
        physical = g["table"][logical]
        mix(pool, physical, logical, domain(pool, kind, u, l, b, tile, 0, logical, physical, g["prefill"]))

    for u in range(work):
        weight_start = 0
        for l in range(g["layers"]):
            length = g["weight"] // g["layers"] + (l < g["weight"] % g["layers"])
            if scenario != "kv_only":
                ranged(0, g["buffer_seeds"][0] & U32, 0x9E3779B9, weight_start, length, u, l, 0, 0x57524541)
            weight_start += length
            if scenario == "weights_only":
                continue
            for b in range(g["batch"]):
                row = l * g["batch"] + b
                for pool in (1, 2):
                    write_domain = (
                        (0x504B5731 if pool == 1 else 0x50565731)
                        if g["prefill"]
                        else (0x4B455931 if pool == 1 else 0x56414C31)
                    )
                    write_base = (
                        (seed & U32)
                        + 0x85EBCA6B * (u + 1)
                        + 0xC2B2AE35 * (l + 1)
                        + 0x27D4EB2F * (b + 1)
                        + write_domain
                    )
                    if g["prefill"]:
                        if g["paged"]:
                            # Paired population has one K-owned lookup per block.
                            for block in range(g["blocks"]):
                                logical = b * g["blocks"] + block
                                table_visit(pool, 0x5050574C, u, l, b, block)
                                start = (row * g["tokens"] + block * g["block_tokens"]) * g["record"]
                                size = (
                                    min(g["block_tokens"], g["tokens"] - block * g["block_tokens"])
                                    * g["record"]
                                )
                                ranged(pool, write_base, 0x165667B1, start, size, u, l, b, 0x50575254)
                            for tile, end in enumerate(g["tile_ends"], 1):
                                for block in range((end + g["block_tokens"] - 1) // g["block_tokens"]):
                                    table_visit(
                                        pool, 0x50504B4C if pool == 1 else 0x5050564C, u, l, b, block, tile
                                    )
                                    start = (row * g["tokens"] + block * g["block_tokens"]) * g["record"]
                                    size = (
                                        min(g["block_tokens"], end - block * g["block_tokens"]) * g["record"]
                                    )
                                    ranged(
                                        pool, write_base, 0x165667B1, start, size, u, l, b, 0x4B565244, tile
                                    )
                        else:
                            for token in range(g["tokens"]):
                                ranged(
                                    pool,
                                    write_base,
                                    0x165667B1,
                                    (row * g["tokens"] + token) * g["record"],
                                    g["record"],
                                    u,
                                    l,
                                    b,
                                    0x50575254,
                                )
                            for tile, end in enumerate(g["tile_ends"], 1):
                                ranged(
                                    pool,
                                    write_base,
                                    0x165667B1,
                                    row * g["tokens"] * g["record"],
                                    end * g["record"],
                                    u,
                                    l,
                                    b,
                                    0x4B565244,
                                    tile,
                                )
                    else:
                        block_range = range(g["blocks"]) if g["paged"] else range(1)
                        for block in block_range:
                            logical = b * g["blocks"] + block if g["paged"] else None
                            physical = g["table"][logical] if g["paged"] else None
                            start = 0 if g["paged"] else row * g["tokens"] * g["record"]
                            length = (
                                min(g["block_tokens"], g["tokens"] - block * g["block_tokens"])
                                if g["paged"]
                                else g["tokens"]
                            ) * g["record"]
                            last = not g["paged"] or block == g["blocks"] - 1
                            index_base = logical * ((g["block_bytes"] + 3) // 4) if g["paged"] else 0
                            initial = g["buffer_seeds"][pool] & U32
                            step = 0x9E3779B9
                            if g["paged"]:
                                initial += 0xA24BAED5 * (l + 1) + 0x9FB21C65 * (physical + 1)
                                step = 0xC13FA9A9
                                if last:
                                    table_visit(pool, 0x50414C55, u, l, b, block)
                                table_visit(pool, 0x504B4C55 if pool == 1 else 0x50564C55, u, l, b, block)
                            ranged(
                                pool,
                                initial,
                                step,
                                start,
                                length,
                                u,
                                l,
                                b,
                                0x4B565244,
                                logical=logical,
                                physical=physical,
                                index_base=index_base,
                            )
                            if last:
                                append = start + length - g["record"]
                                ranged(
                                    pool,
                                    write_base,
                                    0x165667B1,
                                    append,
                                    g["record"],
                                    u,
                                    l,
                                    b,
                                    0x41505044,
                                    logical=logical,
                                    physical=physical,
                                    index_base=index_base,
                                )
                                # Replace just the appended bytes in the scan content sum.
                                old = sum(affine_slice(initial, step, append, g["record"], 32))
                                new = sum(affine_slice(write_base, 0x165667B1, append, g["record"], 32))
                                mix(pool, new - old, 0, 0)
    return None, {
        name: {"a_uint32_decimal": str(v[0]), "b_uint32_decimal": str(v[1])}
        for name, v in zip(("weight", "k", "v"), states)
    }
