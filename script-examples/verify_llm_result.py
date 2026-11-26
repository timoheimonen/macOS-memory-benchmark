#!/usr/bin/env python3
"""Verify one LLM schema-2 artifact or schema-1 LLM sweep envelope independently of the C++ producer.

Usage: python3 script-examples/verify_llm_result.py result.json [--binary memory_benchmark]
Exit 0: consistent accepted artifact; 1: inconsistent or unaccepted; 2: unsupported.
See documents/API.md for evidence limits. No benchmark or executable is invoked.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

from llm_verify_oracles import U64, U32, SEED_DOMAINS, splitmix, permutation, partition, cpu_checksum
from llm_verify_profiles import PROFILES

SCENARIOS = ("weights_only", "kv_only", "mixed")
METRICS = (
    "synthetic_work_unit_latency_seconds",
    "synthetic_memory_work_units_per_second",
    "effective_model_payload_gb_s",
)
SWEEP_PARAM_CONFIG_KEYS = {
    "context-tokens": "visible_context_tokens",
    "kv-block-tokens": "kv_block_tokens",
    "prompt-tokens": "prompt_tokens",
    "attention-query-tile-tokens": "attention_query_tile_tokens",
}
MAX_INPUT = 64 * 1024 * 1024
MAX_WORK = 1000000


class Rejected(Exception):
    def __init__(self, reason, unsupported=False):
        self.reason, self.unsupported = reason, unsupported


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def equal(actual, expected, reason):
    require(type(actual) is type(expected), reason)
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), reason)
        for key in expected:
            equal(actual[key], expected[key], reason)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), reason)
        for a, b in zip(actual, expected):
            equal(a, b, reason)
    else:
        require(actual == expected, reason)


def integer(value, decimal=False, maximum=U64):
    if decimal:
        require(
            isinstance(value, str) and len(value) <= 20 and re.fullmatch(r"0|[1-9][0-9]*", value),
            "integer-encoding",
        )
        number = int(value)
    else:
        require(type(value) is int, "integer-encoding")
        number = value
    require(0 <= number <= maximum, "integer-range")
    return number


def close(actual, expected, reason="rate-mismatch"):
    require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15),
        reason,
    )


class Budget:
    def __init__(self):
        self.remaining = MAX_WORK

    def __call__(self, amount):
        self.remaining -= amount
        if self.remaining < 0:
            raise Rejected("unsupported-work-limit", True)


def identity_fields(text):
    """Parse both documented length-prefix forms; retain repeated field names."""
    require(isinstance(text, str) and len(text) <= 1024 * 1024, "identity-encoding")
    parts = []
    pos = text.find("|")
    require(pos > 0, "identity-encoding")
    prefix = text[:pos]
    while pos < len(text):
        match = re.match(r"\|([a-zA-Z0-9_]+)=", text[pos:])
        require(match is not None, "identity-encoding")
        key = match[1]
        pos += len(match[0])
        length = re.match(r"(0|[1-9][0-9]*):", text[pos:])
        if length:
            size = int(length[1])
            pos += len(length[0])
            value = text[pos : pos + size]
            pos += size
            require(len(value) == size, "identity-length")
        elif parts and parts[-1][0] == key + "_size":
            size = integer(parts[-1][1], True)
            value = text[pos : pos + size]
            pos += size
            require(len(value) == size, "identity-length")
        else:
            end = text.find("|", pos)
            if end < 0:
                end = len(text)
            value = text[pos:end]
            pos = end
        parts.append((key, value))
    return prefix, parts


def fields(text):
    prefix, pairs = identity_fields(text)
    require(len(dict(pairs)) == len(pairs), "identity-duplicate-field")
    return prefix, dict(pairs)


def pipe(prefix, pairs):
    return prefix + "".join("|" + k + "=" + str(v) for k, v in pairs)


def check_geometry(doc, budget):
    r = doc["resolved_plan"]
    c = doc["configuration"]
    g = r["geometry"]
    layout = r["layout"]
    backend, phase, kind = (doc[k] for k in ("backend", "phase", "kv_layout"))
    profile = "-".join((backend, phase, kind))
    if profile not in PROFILES:
        raise Rejected("unsupported-profile", True)
    if doc["methodology_version"] != "llm-memory-v2-" + profile:
        raise Rejected("unsupported-methodology", True)
    components = r["component_identities"]
    profile_components = dict(PROFILES[profile])
    if backend == "metal" and not r["resources"]["metal"]["valid"]:
        require(not any(row["attempted"] for row in doc["measurements"]), "unresolved-execution")
        profile_components.update(msl_revision=None, msl_source_sha256=None)
    for key, value in profile_components.items():
        equal(components[key], value, "component-mismatch")
    component_identity = "llm-memory-components-v1" + "".join(
        "|" + k + "=" + ("null" if v is None else str(len(v)) + ":" + v)
        for k, v in profile_components.items()
        if k != "run_policy_version"
    )
    equal(components["identity"], component_identity, "component-identity-mismatch")
    for obj in (r, c, r["model_work_plan"]):
        for key in ("backend", "phase", "kv_layout"):
            equal(obj[key], doc[key], "profile-mismatch")
    require(r["valid"] is True and g["valid"] is True, "model-invalid")
    ints = {
        k: integer(g[k], k == "kv_element_bytes", (1 << 53) - 1)
        for k in (
            "layer_count",
            "batch_size",
            "query_head_count",
            "kv_head_count",
            "head_dimension",
            "kv_element_bytes",
        )
    }
    for k, v in ints.items():
        require(v > 0, "geometry-range")
        equal(c[k], g[k], "configuration-geometry-mismatch")
    l, b, h, d, e = (
        ints[k] for k in ("layer_count", "batch_size", "kv_head_count", "head_dimension", "kv_element_bytes")
    )
    require(e in (1, 2, 4) and ints["query_head_count"] % h == 0, "geometry-range")
    prefill = phase == "prefill"
    paged = kind == "paged"
    n = integer(
        g["prefill"]["prompt_tokens"] if prefill else g["decode"]["visible_context_tokens"],
        maximum=(1 << 53) - 1,
    )
    require(n > 0, "geometry-range")
    equal(c["prompt_tokens" if prefill else "visible_context_tokens"], n, "configuration-geometry-mismatch")
    q = integer(g["prefill"]["attention_query_tile_tokens"]) if prefill else n
    require(0 < q <= n, "geometry-range")
    tiles = (n + q - 1) // q
    budget(tiles + l * b)
    ends = [min(n, (i + 1) * q) for i in range(tiles)] if prefill else [n]
    if prefill:
        equal(c["attention_query_tile_tokens"], q, "configuration-geometry-mismatch")
        equal(g["prefill"]["tile_count"], str(tiles), "geometry-mismatch")
        equal(g["prefill"]["attention_prefix_token_visits_per_sequence"], str(sum(ends)), "geometry-mismatch")
    if paged:
        block = integer(c["kv_block_tokens"], maximum=U32)
        require(block > 0 and block & (block - 1) == 0, "geometry-range")
    else:
        require(c["kv_block_tokens"] is None, "configuration-geometry-mismatch")
        block = 0
    blocks = (n + block - 1) // block if paged else 0
    record = h * d * e
    weight = integer(g["active_weight_bytes_per_work_unit"], True)
    require(weight > 0, "geometry-range")
    equal(weight, integer(c["weight_size_mb"]) * 1048576, "configuration-geometry-mismatch")
    logical = l * b * n * record
    physical = l * b * blocks * block * record if paged else logical
    read = 2 * l * b * record * sum(ends)
    write = 2 * l * b * record * (n if prefill else 1)
    derived = {
        "kv_vector_bytes": d * e,
        "k_or_v_record_bytes_per_layer": record,
        "kv_record_bytes_per_layer": 2 * record,
        "kv_bytes_per_visible_token": 2 * l * record,
        "k_or_v_sequence_visible_bytes": n * record,
        "k_mapping_bytes": physical,
        "v_mapping_bytes": physical,
        "kv_capacity_bytes": 2 * physical,
        "weight_read_bytes_per_work_unit": weight,
        "kv_read_bytes_per_work_unit": read,
        "kv_write_bytes_per_work_unit": write,
        "kv_only_effective_model_payload_bytes_per_work_unit": read + write,
        "mixed_effective_model_payload_bytes_per_work_unit": weight + read + write,
        "total_data_mapping_bytes": weight + 2 * physical,
    }
    for key, value in derived.items():
        equal(g[key], str(value), "geometry-mismatch")
    equal(g["query_heads_per_kv_head"], ints["query_head_count"] // h, "geometry-mismatch")
    equal(
        g["attention_kind"],
        (
            "mqa"
            if h == 1 and ints["query_head_count"] > 1
            else "mha" if h == ints["query_head_count"] else "gqa"
        ),
        "geometry-mismatch",
    )
    resource = r["resources"]
    for key, value in {
        "weight_logical_bytes": weight,
        "k_logical_bytes": logical,
        "v_logical_bytes": logical,
        "k_physical_length_bytes": physical,
        "v_physical_length_bytes": physical,
        "k_layout_padding_bytes": physical - logical,
        "v_layout_padding_bytes": physical - logical,
    }.items():
        equal(resource[key], str(value), "resource-bytes-mismatch")
    model = dict(
        backend=backend,
        prefill=prefill,
        paged=paged,
        layers=l,
        batch=b,
        record=record,
        tokens=n,
        q=q,
        tile_ends=ends,
        blocks=blocks,
        block_tokens=block,
        block_bytes=block * record,
        weight=weight,
        workers=integer(r["model_work_plan"]["effective_workers"]) if backend == "cpu" else 0,
    )
    budget(model["workers"] * l * b)
    if backend == "cpu":
        mw = r["model_work_plan"]
        requested = integer(c["requested_workers"])
        available = integer(c["available_workers"])
        max_kv = (
            blocks
            if paged and prefill
            else min(l * b * blocks, l * b + blocks - 1) if paged else n if prefill else n * record
        )
        equal(
            model["workers"], min(requested, available, (weight + l - 1) // l, max_kv), "worker-plan-mismatch"
        )
        sequence_count = l * b * (3 if prefill else 1)
        for key, value in dict(
            layer_descriptors_per_worker=l,
            sequence_descriptors_per_worker=sequence_count,
            total_layer_descriptors=l * model["workers"],
            total_sequence_descriptors=sequence_count * model["workers"],
            worker_plan_count=model["workers"],
        ).items():
            equal(mw[key], value, "descriptor-count-mismatch")
        descriptor_bytes = model["workers"] * (
            l * 48 + sequence_count * (112 if paged and prefill else 96 if paged else 80)
        )
        equal(mw["descriptor_bytes"], str(descriptor_bytes), "descriptor-bytes-mismatch")
    seed = integer(doc["seeds"]["base_seed_uint64_decimal"], True)
    equal(c["base_seed_uint64_decimal"], str(seed), "seed-mismatch")
    seeds = [splitmix(seed ^ domain) for domain in SEED_DOMAINS]
    for key, value in zip(("weight_uint64_decimal", "k_uint64_decimal", "v_uint64_decimal"), seeds):
        equal(doc["seeds"]["buffer_domain_seeds"][key], str(value), "seed-mismatch")
    for key, value in zip(SCENARIOS, seeds[3:]):
        equal(doc["seeds"]["scenario_domain_seeds"][key], str(value), "seed-mismatch")
    model.update(buffer_seeds=seeds[:3], scenario_seeds=dict(zip(SCENARIOS, seeds[3:])), base_seed=seed)
    layout_geometry_present = False
    permutation_present = False
    if paged:
        layout_geometry_keys = (
            "kv_block_tokens",
            "blocks_per_sequence",
            "physical_blocks_per_layer",
            "total_physical_blocks",
            "block_bytes",
            "last_block_tokens",
            "last_block_valid_bytes",
            "decode_append_offset_in_last_block",
            "block_table_entries",
            "block_table_bytes",
            "layout_geometry_identity",
        )
        permutation_keys = (
            "layout_identity",
            "permutation_domain_uint64_hex",
            "permutation_seed_uint64_decimal",
            "permutation_algorithm_version",
            "permutation_entry_count",
            "permutation_sha256",
            "permutation_identity",
        )
        expected_append_offset = ((n - 1) % block) * record
        unattempted_metal_failure = (
            backend == "metal"
            and not any(row["attempted"] for row in doc["measurements"])
            and doc["status"] in ("unsupported", "failed")
        )
        if unattempted_metal_failure and not r["resources"]["metal"]["valid"] and layout["kv_block_tokens"] is None:
            for key in layout_geometry_keys:
                require(layout[key] is None, "layout-mismatch")
        else:
            layout_geometry_present = True
            equal(layout["kv_block_tokens"], block, "configuration-geometry-mismatch")
            for key, value in {
                "blocks_per_sequence": blocks,
                "physical_blocks_per_layer": b * blocks,
                "total_physical_blocks": l * b * blocks,
                "block_bytes": block * record,
                "last_block_tokens": n - (blocks - 1) * block,
                "last_block_valid_bytes": (n - (blocks - 1) * block) * record,
                "block_table_entries": b * blocks,
                "block_table_bytes": 4 * b * blocks,
            }.items():
                equal(layout[key], str(value), "layout-mismatch")
            if prefill:
                require(layout["decode_append_offset_in_last_block"] is None, "layout-mismatch")
            else:
                equal(layout["decode_append_offset_in_last_block"], str(expected_append_offset), "layout-mismatch")
        if unattempted_metal_failure and layout["permutation_sha256"] is None:
            for key in permutation_keys:
                require(layout[key] is None, "layout-mismatch")
        else:
            require(layout_geometry_present, "layout-mismatch")
            permutation_present = True
            table = permutation(seed, b * blocks, budget)
            digest = hashlib.sha256(b"".join(v.to_bytes(4, "little") for v in table)).hexdigest()
            equal(
                layout["permutation_seed_uint64_decimal"],
                str(splitmix(seed ^ 0x4C4C4D4B56504731)),
                "seed-mismatch",
            )
            equal(layout["permutation_sha256"], digest, "permutation-digest-mismatch")
            equal(layout["permutation_entry_count"], str(len(table)), "layout-mismatch")
            model["table"] = table
    model["layout_geometry_present"] = layout_geometry_present
    model["permutation_present"] = permutation_present
    model["read"], model["write"] = read, write
    return model


def ownership(model, doc, budget):
    """Independently balance exact prefix costs with rational targets, ties down."""
    m = model
    workers = m["workers"]
    result = {s: [] for s in SCENARIOS}
    if not workers:
        return result
    if not m["paged"] and not m["prefill"]:
        return result
    n = m["blocks"] if m["paged"] else m["tokens"]
    active = min(n, workers)
    budget(n * len(m["tile_ends"]) + workers * m["layers"] * m["batch"])
    payload = [0]
    lookups = [0]
    for unit in range(n):
        start = unit * m["block_tokens"] if m["paged"] else unit
        end = min(m["tokens"], start + (m["block_tokens"] if m["paged"] else 1))
        if m["prefill"]:
            visits = sum(max(0, min(t, end) - start) for t in m["tile_ends"])
            payload.append(payload[-1] + 2 * m["record"] * (end - start + visits))
            lookups.append(
                lookups[-1] + (1 + 2 * sum(t > start for t in m["tile_ends"]) if m["paged"] else 0)
            )
        else:
            payload.append(payload[-1] + 2 * m["record"] * (end - start + (unit == n - 1)))
            lookups.append(lookups[-1] + 2 + (unit == n - 1))
    costs = [p + 4 * k for p, k in zip(payload, lookups)]
    offset = 0
    for layer in range(m["layers"]):
        size = m["weight"] // m["layers"] + (layer < m["weight"] % m["layers"])
        shards = [x[1] for x in partition(offset, size, workers)]
        offset += size
        for b in range(m["batch"]):
            row = layer * m["batch"] + b
            rotation = row % workers
            for scenario in SCENARIOS:
                weights = shards if m["prefill"] and scenario == "mixed" and b == 0 else [0] * workers
                total = costs[-1] + sum(weights[(rank + rotation) % workers] for rank in range(active))
                boundaries = [0]
                preceding = 0
                for rank in range(1, active):
                    preceding += weights[(rank - 1 + rotation) % workers]
                    target = max(0, rank * total - preceding * active)
                    lo = boundaries[-1] + 1
                    hi = n - (active - rank)
                    # Monotone search avoids work proportional to virtual memory sizes.
                    left, right = lo, hi
                    while left < right:
                        mid = (left + right) // 2
                        if costs[mid] * active >= target:
                            right = mid
                        else:
                            left = mid + 1
                    candidates = (max(lo, left - 1), left)
                    boundaries.append(min(candidates, key=lambda k: (abs(costs[k] * active - target), k)))
                boundaries.append(n)
                ranges = [(0, 0)] * workers
                for rank, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
                    ranges[(rank + rotation) % workers] = (first, last - first)
                result[scenario].append(ranges)
    return result


def check_identities(doc, m, budget):
    r = doc["resolved_plan"]
    g = r["geometry"]
    layout = r["layout"]
    cpu = doc["backend_evidence"]["cpu"]
    prefix, values = fields(r["plan_identity"])
    equal(prefix, "llm-memory-work-plan-v1", "model-identity-mismatch")
    aliases = {
        "weight": "active_weight_bytes_per_work_unit",
        "layers": "layer_count",
        "query_heads": "query_head_count",
        "kv_heads": "kv_head_count",
        "head_dim": "head_dimension",
        "batch": "batch_size",
        "kv_only_payload_bytes_per_work_unit": "kv_only_effective_model_payload_bytes_per_work_unit",
        "mixed_payload_bytes_per_work_unit": "mixed_effective_model_payload_bytes_per_work_unit",
    }
    expected = {k: v for k, v in g.items() if type(v) in (str, int) and k not in ("reason_code",)}
    expected.update({k: g[v] for k, v in aliases.items()})
    expected.update(
        backend=m["backend"],
        phase=doc["phase"],
        kv_layout=doc["kv_layout"],
        methodology=doc["methodology_version"],
        component_identity=r["component_identities"]["identity"],
        component_identity_size=len(r["component_identities"]["identity"]),
        range_alignment=32,
        weight_passes_per_work_unit=1,
        kv_replay_factor=1,
        base_seed=m["base_seed"],
        weight_buffer_seed=m["buffer_seeds"][0],
        k_buffer_seed=m["buffer_seeds"][1],
        v_buffer_seed=m["buffer_seeds"][2],
    )
    expected.update({s + "_scenario_seed": v for s, v in m["scenario_seeds"].items()})
    expected.update(
        k_logical_bytes=r["resources"]["k_logical_bytes"],
        v_logical_bytes=r["resources"]["v_logical_bytes"],
        k_layout_padding_bytes=r["resources"]["k_layout_padding_bytes"],
        v_layout_padding_bytes=r["resources"]["v_layout_padding_bytes"],
    )
    if m["paged"]:
        last_block_tokens = m["tokens"] - (m["blocks"] - 1) * m["block_tokens"]
        entries = m["batch"] * m["blocks"]
        expected.update(
            kv_block_tokens=m["block_tokens"],
            kv_blocks_per_sequence=m["blocks"],
            physical_blocks_per_layer=entries,
            total_physical_blocks=m["layers"] * entries,
            kv_block_bytes=m["block_bytes"],
            last_block_tokens=last_block_tokens,
            last_block_valid_bytes=last_block_tokens * m["record"],
            decode_append_offset_in_last_block=(
                0 if m["prefill"] else ((m["tokens"] - 1) % m["block_tokens"]) * m["record"]
            ),
            block_table_entries=entries,
            block_table_bytes=4 * entries,
        )
    else:
        for key in (
            "kv_block_tokens",
            "kv_blocks_per_sequence",
            "physical_blocks_per_layer",
            "total_physical_blocks",
            "kv_block_bytes",
            "last_block_tokens",
            "last_block_valid_bytes",
            "decode_append_offset_in_last_block",
            "block_table_entries",
            "block_table_bytes",
        ):
            expected[key] = 0
    expected["layout_metadata_lookups_per_layer_sequence_per_work_unit"] = (
        (
            m["blocks"] + 2 * sum((t + m["block_tokens"] - 1) // m["block_tokens"] for t in m["tile_ends"])
            if m["prefill"]
            else 1 + 2 * m["blocks"]
        )
        if m["paged"]
        else 0
    )
    expected["traffic_crossover_numerator"] = 0 if m["prefill"] else m["weight"]
    expected["traffic_crossover_denominator"] = (
        0 if m["prefill"] else 2 * m["layers"] * m["batch"] * m["record"]
    )
    if m["prefill"]:
        p = m["tokens"]
        pairs = p * (p + 1) // 2
        attention = pairs * m["layers"] * m["batch"] * g["query_head_count"]
        context = r["model_context"]["prefill"]
        for key, value in {
            "causal_token_pairs_per_sequence": pairs,
            "logical_attention_pairs": attention,
            "logical_attention_fma_terms": attention * g["head_dimension"],
        }.items():
            equal(context[key], str(value) if value <= U64 else None, "model-context-mismatch")
            equal(
                context[key + "_reason_code"],
                "valid" if value <= U64 else "arithmetic-overflow",
                "model-context-mismatch",
            )
        expected.update(
            prefill_planner_version="llm-prefill-planner-v1",
            prefill_cpu_partition_version="llm-prefill-cpu-accounted-prefix-balanced-v1",
            prefill_prompt_tokens=p,
            prefill_query_tile_tokens=m["q"],
            prefill_tile_count=len(m["tile_ends"]),
            prefill_prefix_token_visits_per_sequence=sum(m["tile_ends"]),
            prefill_causal_token_pairs=pairs,
            prefill_logical_attention_pairs=attention,
            prefill_logical_attention_fma_terms=attention * g["head_dimension"],
            prefill_prefix_block_visits_per_sequence=(
                sum((t + m["block_tokens"] - 1) // m["block_tokens"] for t in m["tile_ends"])
                if m["paged"]
                else 0
            ),
        )
    else:
        expected["decode_context"] = m["tokens"]
    if m["backend"] == "cpu":
        for key in (
            "requested_workers",
            "effective_workers",
            "layer_descriptors_per_worker",
            "sequence_descriptors_per_worker",
            "total_layer_descriptors",
            "total_sequence_descriptors",
            "descriptor_bytes",
        ):
            expected[key] = r["model_work_plan"][key]
        if m["prefill"]:
            expected["prefill_execution_identity"] = cpu["prefill"]["identity"]
            expected["prefill_execution_identity_size"] = len(cpu["prefill"]["identity"])
        if m["paged"]:
            expected.update(
                paged_layout_identity=cpu["paged"]["layout_identity"],
                paged_layout_identity_size=len(cpu["paged"]["layout_identity"]),
                paged_execution_identity=cpu["paged"]["execution_identity"],
                paged_execution_identity_size=len(cpu["paged"]["execution_identity"]),
            )
            equal(cpu["paged"]["layout_identity"], layout["layout_identity"], "layout-identity-mismatch")
    else:
        identity = r["resources"]["metal"]["execution_identity"]
        resolved = r["resources"]["metal"]["valid"]
        expected.update(
            metal_execution_resolved=int(resolved),
            metal_execution_identity_size=len(identity) if resolved else 0,
            metal_execution_identity_sha256=(
                hashlib.sha256(identity.encode()).hexdigest() if resolved else "0" * 64
            ),
        )
    for key, value in values.items():
        require(key in expected, "unknown-model-identity-field")
        actual = expected[key]
        if type(actual) is int and actual > U64:
            actual = "unavailable:arithmetic-overflow"
        equal(value, str(actual), "model-identity-mismatch")
    # Mandatory scalar fields must not disappear even when every downstream ref is changed.
    required = {
        "backend",
        "phase",
        "kv_layout",
        "work_unit_kind",
        "methodology",
        "component_identity",
        "component_identity_size",
        "weight",
        "layers",
        "query_heads",
        "kv_heads",
        "head_dim",
        "kv_element_bytes",
        "batch",
        "base_seed",
        "weight_buffer_seed",
        "k_buffer_seed",
        "v_buffer_seed",
        "weights_only_scenario_seed",
        "kv_only_scenario_seed",
        "mixed_scenario_seed",
    }
    required.update(
        "range_alignment weight_passes_per_work_unit kv_replay_factor query_heads_per_kv_head attention_kind kv_vector_bytes k_or_v_record_bytes_per_layer kv_record_bytes_per_layer kv_bytes_per_visible_token k_or_v_sequence_visible_bytes kv_block_tokens kv_blocks_per_sequence physical_blocks_per_layer total_physical_blocks kv_block_bytes last_block_tokens last_block_valid_bytes decode_append_offset_in_last_block k_logical_bytes v_logical_bytes k_layout_padding_bytes v_layout_padding_bytes block_table_entries block_table_bytes layout_metadata_lookups_per_layer_sequence_per_work_unit k_mapping_bytes v_mapping_bytes kv_capacity_bytes weight_read_bytes_per_work_unit kv_read_bytes_per_work_unit kv_write_bytes_per_work_unit kv_only_payload_bytes_per_work_unit mixed_payload_bytes_per_work_unit total_data_mapping_bytes traffic_crossover_numerator traffic_crossover_denominator".split()
    )
    if m["prefill"]:
        required.update(
            "prefill_planner_version prefill_prompt_tokens prefill_query_tile_tokens prefill_tile_count prefill_prefix_token_visits_per_sequence prefill_causal_token_pairs prefill_logical_attention_pairs prefill_logical_attention_fma_terms prefill_prefix_block_visits_per_sequence".split()
        )
    else:
        required.add("decode_context")
    if m["backend"] == "cpu":
        required.update(
            "requested_workers effective_workers layer_descriptors_per_worker sequence_descriptors_per_worker total_layer_descriptors total_sequence_descriptors descriptor_bytes".split()
        )
        if m["prefill"]:
            required.update(
                "prefill_cpu_partition_version prefill_execution_identity_size prefill_execution_identity".split()
            )
        if m["paged"]:
            required.update(
                "paged_layout_identity_size paged_layout_identity paged_execution_identity_size paged_execution_identity".split()
            )
    else:
        required.update(
            "metal_execution_resolved metal_execution_identity_size metal_execution_identity_sha256".split()
        )
    require(required == values.keys(), "model-identity-field-set")
    if m["paged"]:
        entries = m["blocks"] * m["batch"]
        if m["layout_geometry_present"]:
            lp, lv = fields(layout["layout_geometry_identity"])
            equal(lp, "llm-kv-layout-geometry-v1", "layout-identity-mismatch")
            mapping = dict(
                expected,
                sequence_tokens=m["tokens"],
                layer_count=m["layers"],
                batch_size=m["batch"],
                blocks_per_sequence=m["blocks"],
                block_bytes=m["block_bytes"],
                decode_append_offset_in_last_block=((m["tokens"] - 1) % m["block_tokens"]) * m["record"],
                k_physical_bytes=g["k_mapping_bytes"],
                v_physical_bytes=g["v_mapping_bytes"],
                kv_layout_version=r["component_identities"]["kv_layout_version"],
            )
            table_bytes = 4 * entries
            bits = (entries + 7) // 8
            resident = int(g["kv_capacity_bytes"]) + table_bytes
            mapping.update(
                validation_bitset_bytes=bits,
                transient_peak_bytes=table_bytes + bits,
                resident_layout_bytes=resident,
                known_owned_peak_bytes=resident + bits,
                permutation_iterations=max(0, entries - 1),
                validation_entries=entries,
                hash_entries=entries,
                upload_bytes=table_bytes,
            )
            for key, value in lv.items():
                require(key in mapping, "unknown-layout-identity-field")
                equal(value, str(mapping[key]), "layout-identity-mismatch")
        else:
            require(layout["layout_geometry_identity"] is None, "layout-identity-mismatch")
        if m["permutation_present"]:
            _, lv = fields(layout["layout_identity"])
            mapping = dict(
                geometry_identity=layout["layout_geometry_identity"],
                permutation_algorithm_version="splitmix64-fisher-yates-rejection-v1",
                permutation_domain=0x4C4C4D4B56504731,
                permutation_domain_uint64_hex="0x4c4c4d4b56504731",
                permutation_seed=splitmix(m["base_seed"] ^ 0x4C4C4D4B56504731),
                permutation_entry_count=entries,
                permutation_sha256=layout["permutation_sha256"],
            )
            equal(lv, {k: str(v) for k, v in mapping.items()}, "layout-identity-mismatch")
            _, pv = fields(layout["permutation_identity"])
            equal(
                pv,
                dict(
                    domain=str(0x4C4C4D4B56504731),
                    domain_uint64_hex="0x4c4c4d4b56504731",
                    resolved_seed=str(mapping["permutation_seed"]),
                    entry_count=str(entries),
                    sha256=layout["permutation_sha256"],
                ),
                "permutation-identity-mismatch",
            )
        else:
            require(
                layout["layout_identity"] is None and layout["permutation_identity"] is None,
                "layout-identity-mismatch",
            )
    # Validate nested identity hashes against reconstructed/published canonical content.
    strings = set()
    stack = [r, doc["backend_evidence"]]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and "|" in value:
            strings.add(value)
    by_hash = {hashlib.sha256(s.encode()).hexdigest(): s for s in strings}
    for text in strings:
        budget(1)
        _, items = identity_fields(text)
        for key, value in items:
            if key.endswith("identity_sha256"):
                if (
                    key == "metal_execution_identity_sha256"
                    and value == "0" * 64
                    and not r["resources"]["metal"]["valid"]
                ):
                    continue
                require(value in by_hash, "identity-digest-mismatch")
    if m["backend"] == "cpu" and m["prefill"]:
        for sc in cpu["prefill"]["scenarios"]:
            scenario = sc["scenario"]
            for row, text in enumerate(sc["scope_identities"]):
                _, scope = fields(text)
                for key, value in {
                    "scenario": scenario,
                    "worker_count": m["workers"],
                    "prompt_tokens": m["tokens"],
                    "query_tile_tokens": m["q"],
                    "layer_count": m["layers"],
                    "batch_size": m["batch"],
                    "k_or_v_record_bytes_per_layer": m["record"],
                }.items():
                    equal(scope[key], str(value), "ownership-identity-mismatch")
                if scenario == "weights_only":
                    continue
                observed = [(0, 0)] * m["workers"]
                count = integer(scope["assignment_count"], True)
                require(count <= m["workers"], "ownership-assignment-count")
                for i in range(count):
                    name = "assignment_" + str(i) + "_"
                    w = integer(scope[name + "worker_index"], True)
                    require(w < m["workers"], "ownership-worker")
                    observed[w] = (
                        integer(scope[name + "first_unit"], True),
                        integer(scope[name + "unit_count"], True),
                    )
                equal(observed, m["ownership"][scenario][row], "ownership-assignment-mismatch")


def check_prefill_scopes(doc, m, budget):
    if m["backend"] != "cpu" or not m["prefill"]:
        return
    prefill = doc["backend_evidence"]["cpu"]["prefill"]
    g = doc["resolved_plan"]["geometry"]
    workers = m["workers"]
    offsets = []
    offset = 0
    for layer in range(m["layers"]):
        size = m["weight"] // m["layers"] + (layer < m["weight"] % m["layers"])
        offsets.append([v[1] for v in partition(offset, size, workers)])
        offset += size

    def cost(first, count):
        start = first * m["block_tokens"] if m["paged"] else first
        end = min(m["tokens"], (first + count) * m["block_tokens"]) if m["paged"] else first + count
        reads = sum(max(0, min(t, end) - start) for t in m["tile_ends"])
        payload = 2 * m["record"] * (end - start + reads)
        lookups = (
            count
            + 2
            * sum(
                max(0, min(first + count, (t + m["block_tokens"] - 1) // m["block_tokens"]) - first)
                for t in m["tile_ends"]
            )
            if m["paged"]
            else 0
        )
        return payload, lookups, payload + 4 * lookups

    execution = [("sequence_descriptors_per_scenario_per_worker", m["layers"] * m["batch"])]
    for index, sc in enumerate(prefill["scenarios"]):
        scenario = SCENARIOS[index]
        equal(sc["scenario"], scenario, "prefill-scenario-order")
        scope_count = m["layers"] if index == 0 else m["layers"] * m["batch"]
        equal(len(sc["scope_identities"]), scope_count, "scope-count")
        totals = [0] * workers
        for row, text in enumerate(sc["scope_identities"]):
            layer = row if index == 0 else row // m["batch"]
            batch = 0 if index == 0 else row % m["batch"]
            rotation = row % workers
            include = index == 0 or (index == 2 and batch == 0)
            weights = offsets[layer] if include else [0] * workers
            ranges = [(0, 0)] * workers if index == 0 else m["ownership"][scenario][row]
            quantities = [cost(first, count) if count else (0, 0, 0) for first, count in ranges]
            accounted = [w + x[2] for w, x in zip(weights, quantities)]
            total_payload = sum(x[0] for x in quantities)
            total_lookups = sum(x[1] for x in quantities)
            active = sum(bool(count) for first, count in ranges)
            scope = dict(
                planner_version="llm-prefill-planner-v1",
                schedule_version="llm-prefill-owner-local-write-then-tile-k-v-v1",
                unit_kind="paged_block" if m["paged"] else "contiguous_token",
                scenario=scenario,
                active_weight_bytes=m["weight"],
                prompt_tokens=m["tokens"],
                query_tile_tokens=m["q"],
                tile_count=len(m["tile_ends"]),
                layer_count=m["layers"],
                batch_size=m["batch"],
                query_head_count=g["query_head_count"],
                head_dimension=g["head_dimension"],
                k_or_v_record_bytes_per_layer=m["record"],
                paged=int(m["paged"]),
                kv_block_tokens=m["block_tokens"],
                blocks_per_sequence=m["blocks"],
                worker_count=workers,
                worker_rotation=rotation,
                logical_unit_count=0 if index == 0 else m["blocks"] if m["paged"] else m["tokens"],
                active_worker_count=active,
                weight_shards_included=int(include),
                total_weight_shard_bytes=sum(weights),
                total_kv_model_payload_bytes=total_payload,
                total_layout_metadata_lookup_count=total_lookups,
                total_layout_metadata_read_bytes=4 * total_lookups,
                total_kv_accounted_bytes=total_payload + 4 * total_lookups,
                total_scenario_accounted_bytes=sum(accounted),
                minimum_worker_accounted_bytes=min(accounted),
                maximum_worker_accounted_bytes=max(accounted),
                worker_accounted_imbalance_bytes=max(accounted) - min(accounted),
                worker_vector_count=workers,
            )
            for w, (weight, (payload, lookups, cost_value)) in enumerate(zip(weights, quantities)):
                for k, v in dict(
                    weight_shard_bytes=weight,
                    kv_model_payload_bytes=payload,
                    layout_metadata_lookup_count=lookups,
                    layout_metadata_read_bytes=4 * lookups,
                    scenario_accounted_bytes=weight + cost_value,
                ).items():
                    scope[f"worker_{w}_{k}"] = v
                totals[w] += weight + cost_value
            scope["assignment_count"] = active
            for rank in range(active):
                w = (rotation + rank) % workers
                first, count = ranges[w]
                payload, lookups, cost_value = quantities[w]
                for k, v in dict(
                    range_rank=rank,
                    worker_index=w,
                    first_unit=first,
                    unit_count=count,
                    kv_model_payload_bytes=payload,
                    layout_metadata_lookup_count=lookups,
                    layout_metadata_read_bytes=4 * lookups,
                    kv_accounted_bytes=cost_value,
                ).items():
                    scope[f"assignment_{rank}_{k}"] = v
            equal(
                text,
                pipe("llm-prefill-cpu-accounted-prefix-balanced-v1", scope.items()),
                "ownership-identity-mismatch",
            )
        common = [("scenario", scenario), ("scope_count", scope_count)]
        for text in sc["scope_identities"]:
            common.extend(
                [
                    ("scope_identity_size", len(text)),
                    ("scope_identity_sha256", hashlib.sha256(text.encode()).hexdigest()),
                ]
            )
        tail = (
            [("worker_count", workers)]
            + [("worker_accounted_bytes", v) for v in totals]
            + [
                ("minimum_worker_accounted_bytes", min(totals)),
                ("maximum_worker_accounted_bytes", max(totals)),
                ("worker_accounted_imbalance_bytes", max(totals) - min(totals)),
            ]
        )
        equal(
            sc["identity"],
            pipe("llm-prefill-cpu-scenario-execution-v1", common + tail),
            "scenario-execution-identity-mismatch",
        )
        equal(sc["worker_accounted_bytes_per_work_unit"], list(map(str, totals)), "worker-cost-mismatch")
        execution.extend(
            [("scenario_index", index)]
            + common
            + [
                ("scenario_identity_size", len(sc["identity"])),
                ("scenario_identity_sha256", hashlib.sha256(sc["identity"].encode()).hexdigest()),
            ]
            + tail
        )
    equal(
        prefill["identity"],
        pipe("llm-prefill-cpu-execution-v1", execution),
        "prefill-execution-identity-mismatch",
    )


def check_plans(doc, m, budget):
    from llm_verify_oracles import metal_checksum

    r = doc["resolved_plan"]
    plans = r["scenario_plans"]
    if len(plans) > 1024:
        raise Rejected("unsupported-plan-limit", True)
    keys = []
    for p in plans:
        scenario = p["scenario"]
        require(scenario in SCENARIOS, "scenario-mismatch")
        work = integer(p["work_units"], maximum=1000000000)
        require(work > 0, "work-mismatch")
        equal(p["model_ref"], "resolved_plan", "plan-reference")
        equal(p["scenario_seed_uint64_decimal"], str(m["scenario_seeds"][scenario]), "seed-mismatch")
        equal(p["work_unit_kind"], "prefill_operation" if m["prefill"] else "decode_step", "work-mismatch")
        equal(
            p["kv_write_kind"],
            (
                "none"
                if scenario == "weights_only"
                else "full_prompt_population" if m["prefill"] else "current_token_append"
            ),
            "work-mismatch",
        )
        require(type(p["explicit_iterations"]) is bool, "work-mismatch")
        equal(
            p["work_policy"],
            "explicit_fixed_work" if p["explicit_iterations"] else "automatic_calibration",
            "work-mismatch",
        )
        weight = m["weight"] if scenario != "kv_only" else 0
        read = m["read"] if scenario != "weights_only" else 0
        write = m["write"] if scenario != "weights_only" else 0
        lookups = 0
        if m["paged"] and scenario != "weights_only":
            lookups = (
                m["layers"]
                * m["batch"]
                * (
                    m["blocks"]
                    + 2 * sum((t + m["block_tokens"] - 1) // m["block_tokens"] for t in m["tile_ends"])
                    if m["prefill"]
                    else 1 + 2 * m["blocks"]
                )
            )
        per = {
            "weight_read_bytes": weight,
            "kv_read_bytes": read,
            "kv_write_bytes": write,
            "effective_model_payload_bytes": weight + read + write,
            "layout_metadata_lookup_count": lookups,
            "layout_metadata_read_bytes": 4 * lookups,
        }
        for key, value in per.items():
            equal(p[key + "_per_work_unit"], str(value), "plan-bytes-mismatch")
            equal(p[key], str(value * work), "plan-bytes-mismatch")
        accounted = weight + read + write + 4 * lookups
        equal(p["accounted_bytes_per_work_unit"], str(accounted), "plan-bytes-mismatch")
        equal(p["task_accounted_bytes"], str(accounted * work), "plan-bytes-mismatch")
        guard = 68719476736 // accounted
        cap = 65536 if m["backend"] == "metal" else 1000000000
        if m["backend"] == "metal" and m["prefill"] and (not m["paged"] or scenario == "weights_only"):
            visits = (m["layers"] if weight else 0) + (
                2 * m["layers"] * m["batch"] * (m["tokens"] + len(m["tile_ends"])) if read else 0
            )
            cap = min(cap, 1048576 // visits)
        if m["backend"] == "metal" and m["paged"] and lookups:
            cap = min(cap, 268435456 // lookups)
        equal(p["maximum_work_units_by_work_unit_cap"], cap, "work-cap-mismatch")
        equal(p["effective_maximum_work_units"], min(cap, guard), "work-cap-mismatch")
        equal(p["maximum_work_units_by_guardrail"], guard, "work-cap-mismatch")
        # Metal prefill additionally constrains serial visits; published effective cap is checked below.
        require(work <= min(guard, p["effective_maximum_work_units"], 1000000000), "work-cap-mismatch")
        prefix, pairs = identity_fields(p["plan_identity"])
        equal(prefix, "llm-memory-work-plan-v1", "plan-identity-mismatch")
        values = dict(pairs)
        equal(len(values), len(pairs), "plan-identity-mismatch")
        equal(values["model_plan_identity"], r["plan_identity"], "plan-identity-mismatch")
        equal(values["model_plan_identity_size"], str(len(r["plan_identity"])), "plan-identity-mismatch")
        mapped = {
            k: str(v)
            for k, v in p.items()
            if type(v) in (int, str)
            and k
            not in (
                "model_ref",
                "plan_identity",
                "scenario_seed_uint64_decimal",
                "work_policy",
                "reason_code",
            )
        }
        mapped.update(
            scenario_seed=p["scenario_seed_uint64_decimal"],
            explicit=str(int(p["explicit_iterations"])),
            model_plan_identity=r["plan_identity"],
            model_plan_identity_size=str(len(r["plan_identity"])),
        )
        equal(values, mapped, "plan-identity-mismatch")
        expected = p["expected_checksum"]
        if expected["status"] == "available":
            if m["paged"]:
                require("table" in m, "layout-mismatch")
            oracle = cpu_checksum if m["backend"] == "cpu" else metal_checksum
            workers, run = oracle(m, scenario, work, budget)
            equal(expected["expected_worker_checksums"], workers, "expected-worker-checksum-mismatch")
            equal(expected["expected_run_checksum"], run, "expected-run-checksum-mismatch")
        else:
            equal(expected["status"], "unavailable", "checksum-status")
            require(
                expected["expected_worker_checksums"] is None and expected["expected_run_checksum"] is None,
                "checksum-null",
            )
        keys.append((SCENARIOS.index(scenario), work, p["explicit_iterations"]))
    equal(keys, sorted(set(keys)), "canonical-plan-order")
    for scenario, ref in r["frozen_plan_refs"].items():
        if ref is not None:
            integer(ref)
            require(ref < len(plans) and plans[ref]["scenario"] == scenario, "plan-reference")
            if doc["configuration"]["iterations"] is not None:
                equal(plans[ref]["work_units"], doc["configuration"]["iterations"], "frozen-work-mismatch")
    return plans


def timing(execution, elapsed, backend):
    raw = execution.get("timing", {}).get("cpu_raw")
    if backend == "metal":
        require(raw is None, "timing-backend-mismatch")
        metal = execution.get("metal")
        if metal and metal["timing"]["valid"] is True:
            t = metal["timing"]
            start = t["gpu_start_seconds"]
            stop = t["gpu_end_seconds"]
            require(
                type(start) in (float, int) and type(stop) in (float, int) and 0 < start < stop,
                "gpu-timing-invalid",
            )
            if elapsed is not None:
                close(elapsed, stop - start, "elapsed-mismatch")
        return "gpu-timestamps"
    if raw is None:
        return "elapsed-only"
    start, stop, delta = (integer(raw[k], True) for k in ("start_ticks", "stop_ticks", "delta_ticks"))
    numer, denom = (integer(raw[k], maximum=U32) for k in ("timebase_numer", "timebase_denom"))
    equal(delta, (stop - start) & U64, "tick-delta-mismatch")
    reconstructed = float(delta) * float(numer) / float(denom) / 1e9 if denom else 0.0
    if elapsed is not None:
        close(elapsed, reconstructed, "elapsed-mismatch")
    return "cpu-raw-ticks"


def check_checksum(observed, expected, backend, compact=False):
    status = observed["status"]
    if status == "unavailable" or status == "not_evaluated":
        require(
            observed["checksum_valid"] is None and observed["actual_run_checksum"] is None, "checksum-null"
        )
        if not compact:
            require(observed["actual_worker_checksums"] is None, "checksum-null")
        return False
    require(status in ("valid", "invalid"), "checksum-status")
    require(expected["status"] == "available", "expected-checksum-unavailable")
    matches = observed["actual_run_checksum"] == expected["expected_run_checksum"]
    if not compact:
        matches = matches and observed["actual_worker_checksums"] == expected["expected_worker_checksums"]
        if backend == "cpu":
            from llm_verify_oracles import fold

            states = []
            for i, worker in enumerate(observed["actual_worker_checksums"]):
                equal(worker["worker_index"], i, "actual-worker-index")
                states.append(
                    [
                        [
                            integer(worker[p][key], True)
                            for key in (
                                "state_a_uint64_decimal",
                                "state_b_uint64_decimal",
                                "exact_bytes_read",
                                "span_count_uint64_decimal",
                            )
                        ]
                        for p in ("weight", "k", "v")
                    ]
                )
            equal(observed["actual_run_checksum"], fold(states), "actual-run-fold-mismatch")
    equal(observed["checksum_valid"], matches, "checksum-valid-mismatch")
    equal(status, "valid" if matches else "invalid", "checksum-status")
    return matches


def check_measurements(doc, m, plans):
    rows = doc["measurements"]
    count = integer(doc["configuration"]["loop_count"])
    if len(rows) > 100000:
        raise Rejected("unsupported-measurement-limit", True)
    equal(len(rows), 3 * count, "measurement-count-mismatch")
    require(len(doc["loop_records"]) == count, "loop-count-mismatch")
    levels = []
    accepted = {s: [] for s in SCENARIOS}
    stopped = False
    for i, row in enumerate(rows):
        integer(row["measurement_id"])
        equal(row["measurement_id"], i, "measurement-id-mismatch")
        loop, pos = divmod(i, 3)
        scenario = SCENARIOS[(loop + pos) % 3]
        equal(row["loop_index"], loop, "measurement-order-mismatch")
        equal(row["order_position"], pos, "measurement-order-mismatch")
        equal(row["scenario"], scenario, "measurement-order-mismatch")
        status = row["status"]
        require(status in ("measured", "invalid", "failed", "interrupted", "not_run"), "measurement-status")
        require(type(row["attempted"]) is bool, "measurement-attempted")
        if row["attempted"]:
            require(not stopped, "invalid-execution-prefix")
        if status != "measured":
            stopped = True
        ref = row["plan_ref"]
        p = None
        if ref is not None:
            integer(ref)
            require(ref < len(plans) and plans[ref]["scenario"] == scenario, "plan-reference")
            p = plans[ref]
            equal(ref, doc["resolved_plan"]["frozen_plan_refs"][scenario], "frozen-plan-reference")
        ex = row["execution"]
        elapsed = (
            row["elapsed_seconds"] if status == "measured" else ex["timing"]["diagnostic_elapsed_seconds"]
        )
        if row["attempted"]:
            require(p is not None, "plan-reference")
            checksum_ok = check_checksum(row["checksum"], p["expected_checksum"], m["backend"])
            levels.append(timing(ex, elapsed, m["backend"]))
        else:
            require(row["elapsed_seconds"] is None, "metric-null")
            checksum_ok = False
        named_checks(ex, m, scenario, status == "measured")
        for check in ex["validation"]["checks"]:
            if check["applicable"] is False:
                require(check["evaluated"] is None and check["valid"] is None, "validation-null")
            else:
                require(check["applicable"] is True, "validation-applicability")
                if status == "measured":
                    require(
                        check["evaluated"] is True and check["valid"] is True, "required-validation-failed"
                    )
        if status == "measured":
            require(
                row["attempted"]
                and checksum_ok
                and ex["valid"] is True
                and ex["timing"]["evaluated"] is True
                and ex["timing"]["valid"] is True,
                "measurement-valid-mismatch",
            )
            require(
                type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0, "elapsed-invalid"
            )
            equal(row["completed_work_units"], p["work_units"], "completion-mismatch")
            for name in (
                "effective_model_payload_bytes",
                "layout_metadata_lookup_count",
                "layout_metadata_read_bytes",
                "task_accounted_bytes",
            ):
                equal(row["completed_" + name], p[name], "completion-mismatch")
            t = p["work_units"]
            payload = integer(p["effective_model_payload_bytes"], True)
            for key, value in zip(METRICS, (elapsed / t, t / elapsed, payload / elapsed / 1e9)):
                close(row[key], value)
            if m["backend"] == "cpu":
                equal(row["completion_derivation"], "accepted-plan-derived", "completion-derivation")
                equal(row["requested_workers"], doc["configuration"]["requested_workers"], "worker-lifecycle")
                equal(row["effective_workers"], m["workers"], "worker-lifecycle")
                for key in ("requested_workers", "created_workers", "completed_workers"):
                    equal(ex[key], m["workers"], "worker-lifecycle")
                require(
                    ex["timer_started"] is True
                    and ex["timer_stopped"] is True
                    and ex["kernel_succeeded"] is True
                    and ex["worker_startup_failed"] is False,
                    "worker-lifecycle",
                )
            accepted[scenario].append(i)
        else:
            require(all(row[k] is None for k in METRICS + ("elapsed_seconds",)), "metric-null")
    for loop, record in enumerate(doc["loop_records"]):
        equal(record["loop_index"], loop, "loop-order-mismatch")
        order = [SCENARIOS[(loop + i) % 3] for i in range(3)]
        equal(record["planned_order"], order, "loop-order-mismatch")
        attempted = [rows[3 * loop + i]["scenario"] for i in range(3) if rows[3 * loop + i]["attempted"]]
        equal(record["realized_order"], attempted, "loop-prefix-mismatch")
        equal(record["realized_order_count"], len(attempted), "loop-prefix-mismatch")
        equal(
            record["measurement_indexes"], list(range(3 * loop, 3 * loop + 3)), "loop-measurement-reference"
        )
    return accepted, levels


def stats(values):
    n = len(values)
    if not n:
        return None
    s = sorted(values)
    mean = statistics.fmean(s)
    median = statistics.median(s)

    def percentile(p):
        x = (n - 1) * p
        lo = int(x)
        return s[lo] + (s[min(n - 1, lo + 1)] - s[lo]) * (x - lo)

    sd = statistics.stdev(s) if n > 1 else 0.0
    return dict(
        sample_count=n,
        average=mean,
        min=s[0],
        max=s[-1],
        median=median,
        p90=percentile(0.9),
        p95=percentile(0.95),
        p99=percentile(0.99),
        stddev=sd,
        coefficient_of_variation_pct=sd / mean * 100,
        median_absolute_deviation=statistics.median(abs(v - median) for v in s),
    )


def check_aggregates(doc, accepted):
    for scenario, ids in accepted.items():
        a = doc["aggregates"]["scenarios"][scenario]
        equal(a["accepted_measurement_ids"], ids, "aggregate-population-mismatch")
        n = len(ids)
        equal(
            a["status"],
            "unavailable" if not n else "complete" if n == doc["configuration"]["loop_count"] else "partial",
            "aggregate-status-mismatch",
        )
        cv = stats([doc["measurements"][i]["effective_model_payload_gb_s"] for i in ids])
        classification = (
            "insufficient-samples"
            if n < 3
            else "above-threshold" if cv["coefficient_of_variation_pct"] > 5 else "below-threshold"
        )
        equal(a["observed_cv_classification"], classification, "aggregate-cv-mismatch")
        close(a["cv_warning_threshold_pct"], 5.0, "aggregate-cv-mismatch")
        for metric in METRICS:
            values = [doc["measurements"][i][metric] for i in ids]
            expected = stats(values)
            got = a[metric]
            equal(got["sample_count"], n, "aggregate-count-mismatch")
            equal(
                got["headline_semantics"],
                "unavailable" if not n else "single_measurement" if n == 1 else "median_p50",
                "aggregate-headline-semantics",
            )
            if expected is None:
                require(got["headline"] is None and got["statistics"] is None, "aggregate-null")
                continue
            close(got["headline"], expected["median"], "aggregate-headline-mismatch")
            for key, value in expected.items():
                if key == "sample_count":
                    equal(got["statistics"][key], value, "aggregate-count-mismatch")
                else:
                    close(got["statistics"][key], value, "aggregate-statistic-mismatch")


def check_run_state(doc, accepted, plans):
    rows = doc["measurements"]
    count = doc["configuration"]["loop_count"]
    c = doc["counters"]
    measured = sum(map(len, accepted.values()))
    attempted = sum(r["attempted"] for r in rows)
    expected = dict(
        planned_loops=count,
        attempted_loops=len({r["loop_index"] for r in rows if r["attempted"]}),
        completed_loops=sum(
            all(r["status"] == "measured" for r in rows[i : i + 3]) for i in range(0, len(rows), 3)
        ),
        planned_measurements=len(rows),
        attempted_measurements=attempted,
        terminal_measurements=sum(r["status"] != "not_run" for r in rows),
        measured_measurements=measured,
    )
    for key, value in expected.items():
        equal(c[key], value, "counter-mismatch")
    for name in (
        "work_units",
        "effective_model_payload_bytes",
        "layout_metadata_lookup_count",
        "layout_metadata_read_bytes",
        "task_accounted_bytes",
    ):
        completed = sum(integer(r["completed_" + name], name != "work_units") for r in rows)
        equal(c["completed_" + name], str(completed), "counter-mismatch")
        refs = doc["resolved_plan"]["frozen_plan_refs"]
        planned = (
            sum(integer(plans[refs[s]][name], name != "work_units") for s in SCENARIOS) * count
            if all(refs[s] is not None for s in SCENARIOS)
            else 0
        )
        equal(c["planned_" + name], str(planned), "counter-mismatch")
    status = doc["status"]
    require(
        status in ("complete", "partial", "interrupted", "failed", "unsupported", "not_started"), "run-status"
    )
    require(type(doc["interruption_requested"]) is bool, "interruption-flag")
    if status == "complete":
        require(measured == len(rows) and len(rows) > 0, "run-status-mismatch")
    elif status == "partial":
        require(
            0 < attempted and measured < len(rows) and not doc["interruption_requested"],
            "run-status-mismatch",
        )
    elif status == "interrupted":
        require(doc["interruption_requested"] and measured < len(rows), "run-status-mismatch")
    elif status in ("not_started", "unsupported"):
        require(attempted == 0, "run-status-mismatch")
    if any(r["status"] == "invalid" for r in rows):
        require(status == "failed", "run-status-mismatch")
    for row in rows:
        if row["status"] == "interrupted":
            require(doc["interruption_requested"], "interruption-flag")
        if not row["attempted"]:
            require(
                row["checksum"]["checksum_valid"] is None
                and row["checksum"]["actual_run_checksum"] is None
                and row["checksum"]["actual_worker_checksums"] is None,
                "checksum-null",
            )
            require(row["execution"]["valid"] is None, "execution-null")
            require(
                row["completed_work_units"] == 0
                and all(
                    row["completed_" + k] == "0"
                    for k in (
                        "effective_model_payload_bytes",
                        "layout_metadata_lookup_count",
                        "layout_metadata_read_bytes",
                        "task_accounted_bytes",
                    )
                ),
                "completion-mismatch",
            )
    life = doc["checkpoint_lifecycle"]
    equal(life["snapshot_interval_loops"], max(1, (count + 7) // 8), "checkpoint-policy")
    equal(life["checkpoint_policy"], "bounded-loop-snapshots", "checkpoint-policy")
    require(life["current_persistence_success"] is None, "checkpoint-current-persistence")
    require(type(life["checkpoint_failed"]) is bool, "checkpoint-failure")
    if life["checkpoint_failed"]:
        require(status == "failed", "run-status-mismatch")


def named_checks(ex, m, scenario, measured):
    writes = scenario != "weights_only"
    padding = m["paged"] and m["tokens"] % m["block_tokens"] != 0
    cpu = m["backend"] == "cpu"
    kinds = [
        "post-validation-structure",
        (
            "kv-prefill-final-samples"
            if m["prefill"]
            else "kv-append-final" if writes or not cpu else "kv-append-unchanged"
        ),
        "kv-padding-canary",
    ]
    applies = [
        m["paged"] or m["prefill"] or writes if cpu else writes,
        writes or (m["paged"] and not m["prefill"]) if cpu else writes,
        padding if cpu else writes and padding,
    ]
    checks = ex["validation"]["checks"]
    equal(len(checks), 3, "validation-count")
    for check, kind, applicable in zip(checks, kinds, applies):
        equal(check["kind"], kind, "validation-kind")
        equal(check["applicable"], applicable, "validation-applicability")
        if not applicable:
            require(check["evaluated"] is None and check["valid"] is None, "validation-null")
        else:
            require(type(check["evaluated"]) is bool, "validation-evaluated")
            if check["evaluated"]:
                require(type(check["valid"]) is bool, "validation-valid")
            else:
                require(check["valid"] is None, "validation-null")
            if measured:
                require(check["evaluated"] and check["valid"], "required-validation-failed")


def check_manifest(manifest):
    equal(manifest["manifest_version"], 1, "manifest-version")
    require(manifest["status"] in ("available", "partial", "unavailable"), "manifest-status")
    if manifest["binary_sha256"] is not None:
        require(
            isinstance(manifest["binary_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", manifest["binary_sha256"]) is not None,
            "manifest-hash-encoding",
        )
    if manifest["git_commit"] is not None:
        require(
            isinstance(manifest["git_commit"], str)
            and re.fullmatch("[0-9a-f]{40,64}", manifest["git_commit"]) is not None,
            "manifest-commit-encoding",
        )
    require(manifest["git_dirty"] is None or type(manifest["git_dirty"]) is bool, "manifest-dirty-encoding")
    if manifest["status"] == "available":
        require(
            all(
                manifest[k] is not None
                for k in (
                    "binary_sha256",
                    "git_commit",
                    "git_dirty",
                    "compiler",
                    "compile_flags",
                    "link_flags",
                    "target_arch",
                    "sdk",
                    "min_os",
                )
            ),
            "manifest-availability",
        )


def hash_binary(binary):
    digest = hashlib.sha256()
    with open(binary, "rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def verify_single(doc, binary=None, require_raw=False, budget=None, binary_digest=None):
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != 2:
        raise Rejected("unsupported-schema", True)
    if doc.get("mode") != "llm_memory":
        raise Rejected("unsupported-mode", True)
    if budget is None:
        budget = Budget()
    work_before = budget.remaining
    m = check_geometry(doc, budget)
    m["ownership"] = ownership(m, doc, budget)
    check_identities(doc, m, budget)
    check_prefill_scopes(doc, m, budget)
    plans = check_plans(doc, m, budget)
    accepted, levels = check_measurements(doc, m, plans)
    check_aggregates(doc, accepted)
    check_run_state(doc, accepted, plans)
    for scenario, attempts in doc["calibration"]["attempts"].items():
        for index, attempt in enumerate(attempts):
            equal(attempt["attempt_index"], index, "calibration-index")
            ref = attempt["plan_ref"]
            if ref is None:
                require(
                    not attempt["terminal"]
                    and attempt["valid"] is False
                    and attempt["execution"]["valid"] is None
                    and attempt["execution"]["elapsed_seconds"] is None,
                    "calibration-placeholder",
                )
                continue
            integer(ref)
            require(ref < len(plans) and plans[ref]["scenario"] == scenario, "plan-reference")
            ex = attempt["execution"]
            ok = check_checksum(ex["checksum"], plans[ref]["expected_checksum"], m["backend"], True)
            if attempt["valid"]:
                require(
                    ok and ex["valid"] is True and ex["elapsed_seconds"] is not None,
                    "calibration-valid-mismatch",
                )
            named_checks(ex, m, scenario, attempt["valid"])
            levels.append(timing(ex, ex["elapsed_seconds"], m["backend"]))
    measured = sum(map(len, accepted.values()))
    total = len(doc["measurements"])
    complete = total > 0 and measured == total
    equal(doc["results_complete"], complete, "results-complete-mismatch")
    accepted_run = (
        doc["status"] == "complete" and complete and not doc["checkpoint_lifecycle"]["checkpoint_failed"]
    )
    equal(doc["run_accepted"], accepted_run, "run-acceptance-mismatch")
    equal(
        doc["scenario_order_balance_complete"],
        complete and doc["configuration"]["loop_count"] % 3 == 0,
        "order-balance-mismatch",
    )
    if require_raw and m["backend"] == "cpu" and ("elapsed-only" in levels or not levels):
        raise Rejected("unsupported-missing-raw-timing", True)
    manifest = doc["build_manifest"]
    build = "manifest-only" if doc["build_manifest"]["status"] != "unavailable" else "unavailable"
    check_manifest(manifest)
    if binary_digest is None and binary is not None:
        binary_digest = hash_binary(binary)
    if binary_digest is not None:
        require(manifest["binary_sha256"] is not None, "binary-hash-unavailable")
        equal(binary_digest, manifest["binary_sha256"], "binary-hash-mismatch")
        build = "binary-bound"
    return dict(
        artifact_consistent=True,
        run_accepted=accepted_run,
        reason_code="valid" if accepted_run else "run-not-accepted",
        checks=dict(
            checksum=(
                "independent-oracle"
                if any(p["expected_checksum"]["status"] == "available" for p in plans)
                else "unavailable"
            ),
            timing=(
                "elapsed-only"
                if "elapsed-only" in levels
                else (
                    ("cpu-raw-ticks" if m["backend"] == "cpu" else "gpu-timestamps")
                    if levels
                    else "unavailable"
                )
            ),
            build=build,
        ),
        oracle_work_used=work_before - budget.remaining,
    )


def verify_sweep(doc, binary=None, require_raw=False):
    cfg = doc.get("configuration")
    require(isinstance(cfg, dict), "sweep-configuration-invalid")
    if cfg.get("mode") != "sweep" or cfg.get("base_mode") != "llm_memory":
        raise Rejected("unsupported-mode", True)
    if type(cfg.get("sweep_schema_version")) is not int or cfg["sweep_schema_version"] != 1:
        raise Rejected("unsupported-sweep-schema", True)

    run_count = integer(cfg.get("run_count"), maximum=(1 << 53) - 1)
    sweep_max_runs = integer(cfg.get("sweep_max_runs"), maximum=(1 << 53) - 1)
    planned_runs = integer(doc.get("planned_runs"), maximum=(1 << 53) - 1)
    attempted_runs = integer(doc.get("attempted_runs"), maximum=(1 << 53) - 1)
    completed_runs = integer(doc.get("completed_runs"), maximum=(1 << 53) - 1)
    require(run_count > 0 and run_count == planned_runs and run_count <= sweep_max_runs, "sweep-plan-count-mismatch")

    sweep_params = cfg.get("sweep_parameters")
    require(isinstance(sweep_params, dict) and len(sweep_params) > 0, "sweep-parameters-invalid")
    param_keys = []
    parsed_params = {}
    product = 1
    for key, values in sweep_params.items():
        require(key in SWEEP_PARAM_CONFIG_KEYS, "sweep-parameter-unsupported")
        require(isinstance(values, list) and len(values) > 0, "sweep-parameters-invalid")
        parsed_values = []
        for value in values:
            parsed = integer(value, maximum=(1 << 53) - 1)
            require(parsed > 0, "sweep-parameter-value-invalid")
            if key == "kv-block-tokens":
                require(parsed <= U32 and (parsed & (parsed - 1)) == 0, "sweep-parameter-value-invalid")
            parsed_values.append(parsed)
        product *= len(parsed_values)
        if product > planned_runs or product > sweep_max_runs:
            raise Rejected("sweep-plan-count-mismatch")
        param_keys.append(key)
        parsed_params[key] = parsed_values
    require(product == planned_runs, "sweep-plan-count-mismatch")
    if "context-tokens" in parsed_params and (
        "prompt-tokens" in parsed_params or "attention-query-tile-tokens" in parsed_params
    ):
        raise Rejected("sweep-parameters-invalid")
    if "prompt-tokens" in parsed_params and "attention-query-tile-tokens" in parsed_params:
        require(
            max(parsed_params["attention-query-tile-tokens"]) <= min(parsed_params["prompt-tokens"]),
            "sweep-parameter-value-invalid",
        )

    status = doc.get("status")
    require(status in ("complete", "partial", "interrupted", "failed"), "sweep-status-invalid")
    status_reason = doc.get("status_reason")
    require(
        status_reason is None or (isinstance(status_reason, str) and len(status_reason) > 0),
        "sweep-status-invalid",
    )
    conclusions_valid = doc.get("conclusions_valid")
    require(type(conclusions_valid) is bool, "sweep-conclusions-invalid")

    elapsed_sec = doc.get("execution_time_sec")
    require(
        type(elapsed_sec) in (int, float) and math.isfinite(elapsed_sec) and elapsed_sec >= 0,
        "sweep-metadata-invalid",
    )
    timestamp = doc.get("timestamp")
    require(isinstance(timestamp, str) and len(timestamp) > 0, "sweep-metadata-invalid")
    version = doc.get("version")
    require(isinstance(version, str) and len(version) > 0, "sweep-metadata-invalid")

    runs = doc.get("runs")
    require(isinstance(runs, list) and len(runs) <= planned_runs, "sweep-runs-invalid")
    equal(attempted_runs, len(runs), "sweep-attempted-runs-mismatch")

    budget = Budget()
    budget(len(runs))
    binary_digest = hash_binary(binary) if binary is not None else None

    swept_config_keys = {SWEEP_PARAM_CONFIG_KEYS[k] for k in param_keys}
    fixed_config_keys = (
        "weight_size_mb",
        "layer_count",
        "query_head_count",
        "kv_head_count",
        "head_dimension",
        "kv_element_bytes",
        "batch_size",
        "requested_workers",
        "available_workers",
        "worker_source",
        "iterations",
        "work_policy",
        "loop_count",
        "base_seed_uint64_decimal",
        "seed_source",
        "argv",
        "resolved_sources",
    ) + tuple(k for k in SWEEP_PARAM_CONFIG_KEYS.values() if k not in swept_config_keys)

    # Match Cartesian prefix order independently of JSON object key order in sweep_parameters.
    multi_keys = [k for k in param_keys if len(parsed_params[k]) > 1]
    fixed_keys = tuple(k for k in param_keys if len(parsed_params[k]) == 1)
    candidate_orders = [tuple(order) + fixed_keys for order in itertools.permutations(multi_keys)]
    active_candidates = [
        (order, itertools.islice(itertools.product(*(parsed_params[k] for k in order)), len(runs)))
        for order in candidate_orders
    ]
    param_key_set = set(param_keys)

    run_verdicts = []
    observed_completed = 0
    observed_profile = None
    observed_base_config = None
    observed_manifest = None
    checksum_levels = set()
    timing_levels = set()
    build_levels = set()
    stopped = False

    for index, run in enumerate(runs):
        require(not stopped, "sweep-invalid-execution-prefix")
        require(isinstance(run, dict), "sweep-run-invalid")
        equal(run.get("index"), index, "sweep-run-index-mismatch")
        run_params = run.get("parameters")
        require(
            isinstance(run_params, dict)
            and run_params.keys() == param_key_set
            and all(type(v) is int for v in run_params.values()),
            "sweep-parameters-mismatch",
        )
        budget(len(active_candidates))
        next_candidates = []
        for order, iterator in active_candidates:
            combo = next(iterator, None)
            if combo is not None and all(run_params[k] == v for k, v in zip(order, combo)):
                next_candidates.append((order, iterator))
        active_candidates = next_candidates
        require(len(active_candidates) > 0, "sweep-parameters-mismatch")

        run_status = run.get("status")
        require(run_status in ("complete", "partial", "interrupted", "failed"), "sweep-run-status-invalid")
        run_reason = run.get("status_reason")
        if run_status == "complete":
            require(run_reason is None, "sweep-run-status-mismatch")
        else:
            require(isinstance(run_reason, str) and len(run_reason) > 0, "sweep-run-status-mismatch")
            stopped = True

        nested = run.get("result")
        if nested is None:
            require(run_status != "complete", "sweep-run-status-mismatch")
            run_verdicts.append(
                dict(
                    run_index=index,
                    status=run_status,
                    artifact_consistent=True,
                    run_accepted=False,
                    reason_code="nested-result-null",
                )
            )
            continue

        require(isinstance(nested, dict), "sweep-run-result-invalid")
        nested_verdict = verify_single(
            nested, require_raw=require_raw, budget=budget, binary_digest=binary_digest
        )
        nested_profile = (
            nested["backend"],
            nested["phase"],
            nested["kv_layout"],
            nested["methodology_version"],
        )
        if observed_profile is None:
            observed_profile = nested_profile
        else:
            equal(nested_profile, observed_profile, "sweep-profile-mismatch")

        nested_cfg = nested["configuration"]
        equal(nested_cfg.get("output_file"), "", "sweep-nested-output-file-mismatch")
        base_config = {k: nested_cfg.get(k) for k in fixed_config_keys}
        if observed_base_config is None:
            observed_base_config = base_config
            observed_manifest = nested["build_manifest"]
        else:
            equal(base_config, observed_base_config, "sweep-base-config-mismatch")
            equal(nested["build_manifest"], observed_manifest, "sweep-base-config-mismatch")

        for key, expected_value in run_params.items():
            equal(nested_cfg.get(SWEEP_PARAM_CONFIG_KEYS[key]), expected_value, "sweep-nested-parameter-mismatch")

        nested_status = nested["status"]
        if run_status == "complete":
            require(nested_verdict["run_accepted"] is True, "sweep-run-status-mismatch")
            observed_completed += 1
        elif run_status == "interrupted":
            require(nested_status == "interrupted", "sweep-run-status-mismatch")
        elif run_status == "failed":
            require(
                nested_status in ("failed", "unsupported") or not nested_verdict["run_accepted"],
                "sweep-run-status-mismatch",
            )
        elif run_status == "partial":
            require(not nested_verdict["run_accepted"], "sweep-run-status-mismatch")

        checksum_levels.add(nested_verdict["checks"]["checksum"])
        timing_levels.add(nested_verdict["checks"]["timing"])
        build_levels.add(nested_verdict["checks"]["build"])
        run_verdicts.append(
            dict(
                run_index=index,
                status=run_status,
                artifact_consistent=nested_verdict["artifact_consistent"],
                run_accepted=nested_verdict["run_accepted"],
                reason_code=nested_verdict["reason_code"],
            )
        )

    equal(completed_runs, observed_completed, "sweep-completed-runs-mismatch")
    all_complete = planned_runs > 0 and len(runs) == planned_runs and completed_runs == planned_runs
    last_status = runs[-1]["status"] if runs else None
    if status == "complete":
        require(all_complete and conclusions_valid is True and status_reason is None, "sweep-status-mismatch")
    else:
        require(
            conclusions_valid is False and isinstance(status_reason, str) and len(status_reason) > 0,
            "sweep-status-mismatch",
        )
        if status == "failed":
            if status_reason != "checkpoint-write-failed":
                require(last_status == "failed", "sweep-status-mismatch")
        elif status == "interrupted":
            require(last_status != "failed", "sweep-status-mismatch")
        elif status == "partial":
            require(len(runs) > 0 and last_status in ("partial", "complete"), "sweep-status-mismatch")
            require(last_status == "partial" or len(runs) < planned_runs, "sweep-status-mismatch")

    envelope_accepted = status == "complete" and conclusions_valid is True and all_complete

    def summarize_level(levels, preferred_order):
        for candidate in preferred_order:
            if candidate in levels:
                return candidate
        return "unavailable"

    return dict(
        artifact_consistent=True,
        run_accepted=envelope_accepted,
        reason_code="valid" if envelope_accepted else "sweep-not-accepted",
        artifact_kind="llm_sweep",
        planned_runs=planned_runs,
        completed_runs=completed_runs,
        verified_runs=len(runs),
        run_verdicts=run_verdicts,
        checks=dict(
            checksum=summarize_level(checksum_levels, ("unavailable", "independent-oracle")),
            timing=summarize_level(timing_levels, ("elapsed-only", "unavailable", "cpu-raw-ticks", "gpu-timestamps")),
            build=summarize_level(build_levels, ("unavailable", "manifest-only", "binary-bound")),
        ),
        oracle_work_used=MAX_WORK - budget.remaining,
    )


def verify(doc, binary=None, require_raw=False):
    require(isinstance(doc, dict), "invalid-input")
    cfg = doc.get("configuration")
    if isinstance(cfg, dict) and cfg.get("mode") == "sweep" and cfg.get("base_mode") == "llm_memory":
        return verify_sweep(doc, binary=binary, require_raw=require_raw)
    return verify_single(doc, binary=binary, require_raw=require_raw)


def load(path):
    with open(path, "rb") as f:
        data = f.read(MAX_INPUT + 1)
    if len(data) > MAX_INPUT:
        raise Rejected("unsupported-input-limit", True)

    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate-json-key")
            result[k] = v
        return result

    doc = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(Rejected("nonfinite-json-number")),
    )
    stack = [(doc, 0)]
    count = 0
    while stack:
        obj, depth = stack.pop()
        count += 1
        if depth > 64 or count > 1000000:
            raise Rejected("unsupported-structure-limit", True)
        if isinstance(obj, dict):
            stack.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            stack.extend((v, depth + 1) for v in obj)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--require-raw-timing", action="store_true", help="Require original timing evidence for CPU tasks"
    )
    args = parser.parse_args()
    try:
        verdict = verify(load(args.result), args.binary, args.require_raw_timing)
        code = 0 if verdict["run_accepted"] else 1
    except Rejected as error:
        verdict = dict(
            artifact_consistent=None if error.unsupported else False,
            run_accepted=False,
            reason_code=error.reason,
        )
        code = 2 if error.unsupported else 1
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RecursionError,
        ArithmeticError,
        AttributeError,
    ) as error:
        verdict = dict(
            artifact_consistent=False,
            run_accepted=False,
            reason_code="invalid-input",
            detail=str(error)[:200],
        )
        code = 1
    print(json.dumps(verdict, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
