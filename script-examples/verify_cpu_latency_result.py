#!/usr/bin/env python3
"""Verify one CPU latency schema-1 artifact or schema-1 sweep envelope
independently of the C++ producer.

Usage:
  python3 script-examples/verify_cpu_latency_result.py result.json [--binary memory_benchmark] [--require-raw-timing] [--max-chain-nodes N]
Exit 0: consistent accepted artifact; 1: inconsistent or unaccepted; 2: unsupported.
"""

import hashlib
import math
import sys

from cpu_verify_common import (
    MAX_SAFE_INT,
    U32,
    VALID_MEASUREMENT_STATUSES,
    Budget,
    Rejected,
    check_checkpoint_lifecycle,
    check_buffer_evidence,
    check_environment,
    check_manifest,
    check_memory_budget_block,
    check_quality_warnings,
    check_resolved_sources,
    check_run_acceptance,
    check_statistics_block,
    check_validation_checks,
    close,
    compute_statistics,
    enforce_require_raw_timing,
    equal,
    fisher_yates_rejection_shuffle,
    hash_binary,
    integer,
    require,
    run_verifier_cli,
    splitmix64_mix,
    verify_cpu_raw_timing,
    verify_sweep_envelope,
)

SCHEMA_VERSION = 1
MODE_NAME = "cpu_latency"
METHODOLOGY_VERSION = "cpu-latency-v1-arm64-calibrated-seeded-balanced"
DEFAULT_MAX_CHAIN_NODES = 8_388_608

EXPECTED_COMPONENTS = {
    "version": "cpu-latency-components-v1",
    "work_plan_version": "cpu-latency-work-plan-v1",
    "chain_generator_version": "splitmix64-fisher-yates-rejection-v1",
    "chain_version": "cpu-latency-chain-v1",
    "run_policy_version": "bounded-loop-snapshots-v1",
}

EXPECTED_RESOLVED_SOURCE_TOKENS = {
    "buffer_size": ("explicit", "default", "sweep"),
    "cache_size": ("explicit", "default", "sweep"),
    "iterations": ("explicit", "auto"),
    "latency_samples": ("explicit", "default"),
    "latency_stride_bytes": ("explicit", "default", "sweep"),
    "latency_chain_mode": ("explicit", "default", "sweep"),
    "latency_tlb_locality_kb": ("explicit", "default", "sweep"),
    "seed": ("explicit", "generated"),
}

LAT_FLAG_MAP = {
    "buffer_size": ("-b", "--buffer-size"),
    "cache_size": ("-k", "--cache-size"),
    "iterations": ("-i", "--iterations"),
    "latency_samples": ("-n", "--latency-samples"),
    "latency_stride_bytes": ("-s", "--latency-stride-bytes"),
    "latency_chain_mode": ("-m", "--latency-chain-mode"),
    "latency_tlb_locality_kb": ("-l", "--latency-tlb-locality-kb"),
    "seed": ("--seed",),
}

EXPECTED_COMPARABILITY = [
    "same-methodology-version",
    "same-frozen-work-plan-and-targets",
    "same-software-version",
    "sufficiently-similar-hardware",
    "sufficiently-similar-thermal-and-power-state",
]

TARGET_DOMAINS = {
    "l1": 0x434C41545F4C315F,
    "l2": 0x434C41545F4C325F,
    "custom": 0x434C41545F435553,
    "main_memory": 0x434C41545F4D4149,
}
LOCALITY_16K_DOMAIN = 0x434C41545F4C3136
LOCALITY_GLOBAL_DOMAIN = 0x434C41545F474C4F

CANONICAL_TARGETS = ("l1", "l2", "custom", "main_memory")

MAX_ACCESSES = 4_000_000_000
CV_WARNING_THRESHOLD_PCT = 7.5
LOCALITY_16K_BYTES = 16 * 1024


def normalize_chain_mode(token):
    """Normalize latency chain mode string matching src/core/memory/memory_utils.cpp."""
    require(isinstance(token, str) and len(token) > 0, "chain-mode-invalid")
    norm = token.replace("-", "_").replace(" ", "_").lower()
    mapping = {
        "auto": "auto",
        "global": "global-random",
        "global_random": "global-random",
        "random_box": "random-box",
        "random_in_box_random_box": "random-box",
        "same_random_in_box": "same-random-in-box",
        "same_random_in_box_increasing_box": "same-random-in-box",
        "diff_random_in_box": "diff-random-in-box",
        "diff_random_in_box_increasing_box": "diff-random-in-box",
    }
    require(norm in mapping, "chain-mode-invalid")
    return mapping[norm]


def resolve_chain_mode(mode, tlb_locality_bytes):
    """Resolve 'auto' based on tlb_locality_bytes."""
    canonical = normalize_chain_mode(mode)
    if canonical == "auto":
        return "global-random" if tlb_locality_bytes == 0 else "random-box"
    if canonical != "global-random":
        require(tlb_locality_bytes > 0, "chain-mode-requires-locality")
    return canonical


def build_latency_chain(
    base_seed,
    domain_id,
    node_count,
    effective_mode,
    locality_nodes,
    budget=None,
    max_chain_nodes=DEFAULT_MAX_CHAIN_NODES,
):
    """Reconstruct canonical cpu-latency-chain-v1 permutation and SHA-256 digest."""
    require(node_count >= 2, "chain-geometry-invalid")
    if node_count > max_chain_nodes:
        raise Rejected("unsupported-chain-node-limit", True)
    if effective_mode != "global-random":
        require(locality_nodes >= 2, "chain-geometry-invalid")
    if locality_nodes > max_chain_nodes:
        raise Rejected("unsupported-chain-node-limit", True)
    if budget is not None:
        budget(1)
    chain_seed = splitmix64_mix(base_seed ^ domain_id)
    state = chain_seed

    if effective_mode == "global-random":
        indices = list(range(node_count))
        fisher_yates_rejection_shuffle(indices, state, budget)
    else:
        box_count = (node_count + locality_nodes - 1) // locality_nodes
        box_order = list(range(box_count))
        if effective_mode == "random-box":
            state = fisher_yates_rejection_shuffle(box_order, state, budget)
        indices = []
        if effective_mode == "same-random-in-box":
            shared = list(range(locality_nodes))
            state = fisher_yates_rejection_shuffle(shared, state, budget)
            for box_id in box_order:
                start = box_id * locality_nodes
                end = min(start + locality_nodes, node_count)
                for off in shared:
                    if start + off < end:
                        indices.append(start + off)
        else:
            for box_id in box_order:
                start = box_id * locality_nodes
                end = min(start + locality_nodes, node_count)
                box = list(range(start, end))
                state = fisher_yates_rejection_shuffle(box, state, budget)
                indices.extend(box)

    pos0 = indices.index(0)
    indices = indices[pos0:] + indices[:pos0]
    require(len(indices) == node_count and indices[0] == 0, "chain-hamiltonian-invalid")

    h = hashlib.sha256()
    for i in range(0, node_count, 4096):
        h.update(b"".join(v.to_bytes(8, "little") for v in indices[i : i + 4096]))
    return chain_seed, indices, h.hexdigest()


def _validation_specs(has_samples):
    return [
        ("post-validation-structure", True),
        ("chain-hamiltonian-prevalidation", True),
        ("chain-readonly-protection", True),
        ("main-final-pointer", True),
        ("sample-window-end-pointers", has_samples),
    ]


def _check_chain_identity(
    ci,
    base_seed,
    domain_id,
    target_bytes,
    stride_bytes,
    effective_mode,
    tlb_locality_kb,
    budget,
    max_chain_nodes,
):
    require(isinstance(ci, dict), "chain-identity-invalid")
    node_count = integer(ci.get("node_count"), maximum=MAX_SAFE_INT)
    expected_nodes = target_bytes // stride_bytes
    equal(node_count, expected_nodes, "chain-geometry-invalid")
    equal(ci.get("stride_bytes"), stride_bytes, "chain-geometry-invalid")
    equal(ci.get("chain_mode"), effective_mode, "chain-mode-mismatch")
    equal(ci.get("tlb_locality_kb"), tlb_locality_kb, "chain-geometry-invalid")
    locality_nodes = (tlb_locality_kb * 1024) // stride_bytes if effective_mode != "global-random" else 0
    chain_seed, indices, digest = build_latency_chain(
        base_seed,
        domain_id,
        node_count,
        effective_mode,
        locality_nodes,
        budget=budget,
        max_chain_nodes=max_chain_nodes,
    )
    equal(ci.get("chain_seed_uint64_decimal"), str(chain_seed), "seed-mismatch")
    equal(ci.get("chain_permutation_sha256"), digest, "chain-digest-mismatch")
    return node_count, indices


def _check_sample_windows(sw, requested_samples, completed_accesses, cpu_raw, indices, status):
    if requested_samples == 0:
        require(sw is None, "sample-window-null")
        return []
    if status != "measured":
        if sw is None:
            return []
        require(isinstance(sw, dict), "sample-window-invalid")
        val = sw.get("validation")
        require(isinstance(val, dict), "sample-window-invalid")
        if status == "invalid":
            require(
                val.get("valid") is False or val.get("all_end_pointers_valid") is False,
                "sample-window-validation-failed",
            )
        return []

    require(isinstance(sw, dict), "sample-window-invalid")
    effective_samples = min(requested_samples, completed_accesses)
    equal(sw.get("sample_count"), effective_samples, "sample-count-mismatch")
    tb = sw.get("timebase")
    require(isinstance(tb, dict), "sample-timebase-mismatch")
    numer = integer(tb.get("timebase_numer"), maximum=U32)
    denom = integer(tb.get("timebase_denom"), maximum=U32)
    require(numer > 0 and denom > 0, "sample-timebase-mismatch")
    if cpu_raw is not None:
        equal(numer, integer(cpu_raw.get("timebase_numer"), maximum=U32), "sample-timebase-mismatch")
        equal(denom, integer(cpu_raw.get("timebase_denom"), maximum=U32), "sample-timebase-mismatch")

    ticks = sw.get("sample_delta_ticks")
    ns_list = sw.get("samples_ns")
    require(isinstance(ticks, list) and isinstance(ns_list, list), "sample-window-invalid")
    equal(len(ticks), effective_samples, "sample-count-mismatch")
    equal(len(ns_list), effective_samples, "sample-count-mismatch")

    val = sw.get("validation")
    require(isinstance(val, dict), "sample-window-validation-failed")
    require(
        val.get("valid") is True and val.get("all_end_pointers_valid") is True,
        "sample-window-validation-failed",
    )
    end_nodes = val.get("end_node_indices")
    if end_nodes is not None:
        require(isinstance(end_nodes, list) and len(end_nodes) == effective_samples, "sample-window-validation-failed")

    base_step, rem = divmod(completed_accesses, effective_samples)
    node_count = len(indices)
    cum_steps = 0
    for idx in range(effective_samples):
        accesses_i = base_step + (1 if idx < rem else 0)
        cum_steps += accesses_i
        dt = integer(ticks[idx], decimal=True)
        expected_ns = (float(dt) * float(numer) / float(denom)) / float(accesses_i)
        close(ns_list[idx], expected_ns, "sample-timing-mismatch")
        if end_nodes is not None:
            equal(end_nodes[idx], indices[cum_steps % node_count], "sample-window-end-pointer-mismatch")
    return list(ns_list)


def verify_single(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    binary_digest=None,
    max_chain_nodes=DEFAULT_MAX_CHAIN_NODES,
):
    """Verify a single cpu_latency schema-1 artifact."""
    require(isinstance(doc, dict), "invalid-input")
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != SCHEMA_VERSION:
        raise Rejected("unsupported-schema", True)
    if doc.get("mode") != MODE_NAME:
        raise Rejected("unsupported-mode", True)
    if doc.get("methodology_version") != METHODOLOGY_VERSION:
        raise Rejected("unsupported-methodology", True)

    if budget is None:
        budget = Budget()
    work_before = budget.remaining

    cfg = doc.get("configuration")
    require(isinstance(cfg, dict), "configuration-invalid")
    buf_mb = integer(cfg.get("buffer_size_mb"), maximum=MAX_SAFE_INT)
    cache_kb = cfg.get("custom_cache_size_kb")
    if cache_kb is not None:
        cache_kb = integer(cache_kb, maximum=1048576)
        require(cache_kb == 0 or cache_kb >= 16, "configuration-invalid")
    iterations = cfg.get("iterations")
    if iterations is not None:
        iterations = integer(iterations, maximum=MAX_ACCESSES)
        require(iterations >= 1, "configuration-invalid")
    equal(
        cfg.get("work_policy"),
        "explicit_fixed_work" if iterations is not None else "automatic_calibration",
        "configuration-invalid",
    )
    loop_count = integer(cfg.get("loop_count"), maximum=MAX_SAFE_INT)
    require(loop_count >= 1, "configuration-invalid")
    req_samples = integer(cfg.get("latency_samples"), maximum=U32)
    stride = integer(cfg.get("latency_stride_bytes"), maximum=U32)
    require(stride >= 8 and stride % 8 == 0, "configuration-invalid")
    tlb_locality_kb = integer(cfg.get("latency_tlb_locality_kb"), maximum=MAX_SAFE_INT)
    canonical_mode = normalize_chain_mode(cfg.get("latency_chain_mode"))
    compare_locality = cfg.get("compare_locality")
    require(type(compare_locality) is bool, "configuration-invalid")
    require(type(cfg.get("non_cacheable")) is bool, "configuration-invalid")
    require(cfg.get("seed_source") in ("user", "generated"), "seed-mismatch")
    require(isinstance(cfg.get("output_file"), str), "configuration-invalid")
    require(isinstance(cfg.get("argv"), list) and all(isinstance(a, str) for a in cfg["argv"]), "configuration-invalid")
    check_resolved_sources(
        cfg.get("resolved_sources"),
        cfg,
        EXPECTED_RESOLVED_SOURCE_TOKENS,
        LAT_FLAG_MAP,
    )

    env = doc.get("environment")
    check_environment(env)
    page_size = integer(env["page_size_bytes"], decimal=True)

    seeds = doc.get("seeds")
    require(isinstance(seeds, dict), "seed-mismatch")
    base_seed = integer(seeds.get("base_seed_uint64_decimal"), decimal=True)
    equal(cfg.get("base_seed_uint64_decimal"), str(base_seed), "seed-mismatch")
    equal(seeds.get("source"), cfg.get("seed_source"), "seed-mismatch")
    target_domain_seeds = seeds.get("target_domain_seeds")
    require(isinstance(target_domain_seeds, dict), "seed-mismatch")

    rp = doc.get("resolved_plan")
    require(isinstance(rp, dict), "plan-invalid")
    rp_valid = rp.get("valid")
    require(type(rp_valid) is bool, "plan-invalid")
    rp_reason = rp.get("reason_code")
    require(isinstance(rp_reason, str) and len(rp_reason) > 0, "plan-invalid")
    if rp_valid:
        equal(rp_reason, "valid", "plan-invalid")
        effective_mode = resolve_chain_mode(cfg.get("latency_chain_mode"), tlb_locality_kb * 1024)
        if compare_locality:
            require(
                buf_mb > 0 and stride <= 8192 and (LOCALITY_16K_BYTES % page_size) == 0,
                "configuration-invalid",
            )
        if effective_mode != "global-random":
            require((tlb_locality_kb * 1024) % page_size == 0, "locality-not-page-multiple")
    else:
        if canonical_mode == "auto":
            effective_mode = "global-random" if tlb_locality_kb == 0 else "random-box"
        else:
            effective_mode = canonical_mode
        require(rp_reason != "valid", "plan-invalid")
    equal(rp.get("component_identities"), EXPECTED_COMPONENTS, "component-mismatch")
    require(
        isinstance(rp.get("plan_identity"), str)
        and rp["plan_identity"].startswith("cpu-latency-work-plan-v1|"),
        "plan-identity-mismatch",
    )

    rows = doc.get("measurements")
    require(isinstance(rows, list), "measurement-invalid")
    if len(rows) > 100_000:
        raise Rejected("unsupported-measurement-limit", True)
    require(all(isinstance(r, dict) for r in rows), "measurement-invalid")
    attempted_any = any(r.get("attempted") is True for r in rows)
    doc_status = doc.get("status")
    if not rp_valid:
        require(doc_status == "failed" and not attempted_any, "plan-invalid")
        equal(doc.get("reason_code"), rp_reason, "plan-invalid")
        if rp_reason == "locality-not-page-multiple":
            require(
                effective_mode != "global-random" and (tlb_locality_kb * 1024) % page_size != 0,
                "plan-invalid",
            )
        elif rp_reason == "compare-locality-invalid":
            require(
                compare_locality
                and (buf_mb == 0 or stride > 8192 or (LOCALITY_16K_BYTES % page_size) != 0),
                "plan-invalid",
            )
        elif rp_reason == "chain-mode-requires-locality":
            require(
                effective_mode != "global-random" and tlb_locality_kb == 0,
                "plan-invalid",
            )

    targets_obj = rp.get("targets")
    chain_identities = rp.get("chain_identities")
    require(isinstance(targets_obj, dict) and isinstance(chain_identities, dict), "target-geometry-mismatch")

    active_targets = []
    target_info = {}
    min_committed_bytes = 0

    for tname in CANONICAL_TARGETS:
        tdata = targets_obj.get(tname)
        if tdata is None:
            continue
        require(isinstance(tdata, dict) and type(tdata.get("enabled")) is bool, "target-geometry-mismatch")
        if not tdata["enabled"]:
            continue
        tbytes = integer(tdata.get("bytes"), decimal=True)
        require(tbytes > 0, "target-geometry-mismatch")
        if tname == "main_memory":
            require(buf_mb > 0, "target-geometry-mismatch")
            equal(tbytes, buf_mb * 1048576, "target-geometry-mismatch")
        elif tname == "custom":
            require(cache_kb is not None and cache_kb > 0, "target-geometry-mismatch")
            equal(tbytes, cache_kb * 1024, "target-geometry-mismatch")
        else:
            require(cache_kb is None, "target-geometry-mismatch")

        expected_tseed = splitmix64_mix(base_seed ^ TARGET_DOMAINS[tname])
        equal(target_domain_seeds.get(tname), str(expected_tseed), "seed-mismatch")
        if rp_valid:
            node_count, indices = _check_chain_identity(
                chain_identities.get(tname),
                base_seed,
                TARGET_DOMAINS[tname],
                tbytes,
                stride,
                effective_mode,
                tlb_locality_kb,
                budget,
                max_chain_nodes,
            )
        else:
            node_count = tbytes // stride
            indices = []
        active_targets.append(tname)
        target_info[tname] = {
            "bytes": tbytes,
            "node_count": node_count,
            "indices": indices,
        }
        rounded = ((tbytes + page_size - 1) // page_size) * page_size
        min_committed_bytes += rounded

    if rp_valid:
        require(len(active_targets) > 0, "target-omission")
        if buf_mb > 0:
            require("main_memory" in target_info, "target-geometry-mismatch")
        else:
            require("main_memory" not in target_info, "target-geometry-mismatch")
        if cache_kb == 0:
            require(all(t not in target_info for t in ("l1", "l2", "custom")), "target-geometry-mismatch")
        elif cache_kb is not None and cache_kb > 0:
            require("custom" in target_info and "l1" not in target_info and "l2" not in target_info, "target-geometry-mismatch")
        else:
            require(("l1" in target_info or "l2" in target_info) and "custom" not in target_info, "target-geometry-mismatch")

    resources_obj = rp.get("resources")
    require(isinstance(resources_obj, dict), "resource-evidence-invalid")
    lat_buffer_evidences = []
    require_target_ro = (doc_status == "complete" or attempted_any)
    allow_unalloc = (not attempted_any and doc_status in ("failed", "interrupted", "unsupported", "not_started"))
    for key, entry in resources_obj.items():
        if key == "compare_buffer":
            require(compare_locality and "main_memory" in target_info, "resource-evidence-invalid")
            check_buffer_evidence(
                entry,
                target_info["main_memory"]["bytes"],
                cfg["non_cacheable"],
                page_size,
                require_ro=(doc_status == "complete"),
                allow_unallocated=allow_unalloc,
            )
            lat_buffer_evidences.append(entry)
        else:
            require(key in target_info and isinstance(entry, dict), "resource-evidence-invalid")
            require("buffer" in entry, "resource-evidence-invalid")
            check_buffer_evidence(
                entry["buffer"],
                target_info[key]["bytes"],
                cfg["non_cacheable"],
                page_size,
                require_ro=require_target_ro,
                allow_unallocated=allow_unalloc,
            )
            lat_buffer_evidences.append(entry["buffer"])
    if doc_status == "complete" or attempted_any:
        expected_res_keys = set(active_targets)
        if compare_locality:
            expected_res_keys.add("compare_buffer")
        equal(set(resources_obj.keys()), expected_res_keys, "resource-evidence-invalid")

    locality_chains = {}
    if compare_locality and rp_valid:
        main_bytes = target_info["main_memory"]["bytes"]
        equal(
            target_domain_seeds.get("locality_16k"),
            str(splitmix64_mix(base_seed ^ LOCALITY_16K_DOMAIN)),
            "seed-mismatch",
        )
        equal(
            target_domain_seeds.get("locality_global"),
            str(splitmix64_mix(base_seed ^ LOCALITY_GLOBAL_DOMAIN)),
            "seed-mismatch",
        )
        for loc_key, domain_id, loc_mode, loc_kb in (
            ("locality_16k", LOCALITY_16K_DOMAIN, "random-box", 16),
            ("locality_global", LOCALITY_GLOBAL_DOMAIN, "global-random", 0),
        ):
            n_nodes, loc_indices = _check_chain_identity(
                chain_identities.get(loc_key),
                base_seed,
                domain_id,
                main_bytes,
                stride,
                loc_mode,
                loc_kb,
                budget,
                max_chain_nodes,
            )
            locality_chains[loc_key] = {"node_count": n_nodes, "indices": loc_indices}
        min_committed_bytes += ((main_bytes + page_size - 1) // page_size) * page_size

    check_memory_budget_block(
        doc.get("memory_budget"),
        min_committed_bytes=min_committed_bytes,
        require_valid_when_active=attempted_any,
    )

    target_plans = rp.get("target_plans")
    require(isinstance(target_plans, list), "plan-invalid")
    if len(target_plans) > 4096:
        raise Rejected("unsupported-plan-limit", True)

    for idx, plan in enumerate(target_plans):
        require(isinstance(plan, dict), "plan-invalid")
        equal(plan.get("plan_ref"), idx, "plan-reference")
        tname = plan.get("target")
        require(tname in target_info, "plan-invalid")
        cycles = integer(plan.get("complete_chain_cycles"), maximum=MAX_ACCESSES)
        require(cycles >= 1, "work-mismatch")
        accesses = integer(plan.get("access_count"), maximum=MAX_ACCESSES)
        equal(accesses, cycles * target_info[tname]["node_count"], "plan-bytes-mismatch")
        require(type(plan.get("explicit_iterations")) is bool, "work-mismatch")

    frozen_refs = rp.get("frozen_plan_refs")
    require(isinstance(frozen_refs, dict), "plan-reference")
    for tname in active_targets:
        ref = frozen_refs.get(tname)
        if ref is not None:
            integer(ref, maximum=MAX_SAFE_INT)
            require(ref < len(target_plans) and target_plans[ref]["target"] == tname, "plan-reference")
            if iterations is not None:
                equal(target_plans[ref]["complete_chain_cycles"], iterations, "frozen-work-mismatch")

    levels = []
    calibration = doc.get("calibration")
    require(isinstance(calibration, dict), "calibration-invalid")
    equal(calibration.get("excluded_from_results"), True, "calibration-excluded-mismatch")
    cal_attempts = calibration.get("attempts")
    require(isinstance(cal_attempts, dict), "calibration-invalid")
    for tname, attempts in cal_attempts.items():
        require(tname in target_info and isinstance(attempts, list), "calibration-invalid")
        for idx, attempt in enumerate(attempts):
            require(isinstance(attempt, dict), "calibration-invalid")
            equal(attempt.get("attempt_index"), idx, "calibration-index")
            ref = attempt.get("plan_ref")
            ex = attempt.get("execution")
            require(isinstance(ex, dict), "calibration-invalid")
            if ref is None:
                require(
                    not attempt.get("terminal")
                    and attempt.get("valid") is False
                    and ex.get("valid") is None
                    and ex.get("elapsed_seconds") is None,
                    "calibration-placeholder",
                )
                continue
            integer(ref, maximum=MAX_SAFE_INT)
            require(ref < len(target_plans) and target_plans[ref]["target"] == tname, "plan-reference")
            require(type(attempt.get("valid")) is bool, "calibration-invalid")
            if attempt.get("valid"):
                require(ex.get("valid") is True and ex.get("elapsed_seconds") is not None, "calibration-valid-mismatch")
            val_obj = ex.get("validation")
            require(isinstance(val_obj, dict), "validation-invalid")
            check_validation_checks(
                val_obj.get("checks"),
                _validation_specs(False),
                bool(attempt.get("valid")),
            )
            cal_timing = ex.get("timing")
            require(isinstance(cal_timing, dict), "timing-invalid")
            levels.append(verify_cpu_raw_timing(cal_timing.get("cpu_raw"), ex.get("elapsed_seconds")))

    num_targets = len(active_targets)
    equal(len(rows), loop_count * num_targets, "measurement-count-mismatch")

    loop_records = doc.get("loop_records")
    require(isinstance(loop_records, list) and len(loop_records) == loop_count, "loop-count-mismatch")

    accepted = {t: [] for t in active_targets}
    pooled_samples = {t: [] for t in active_targets}
    stopped = False

    for i, row in enumerate(rows):
        budget(1)
        require(isinstance(row, dict), "measurement-invalid")
        integer(row.get("measurement_id"), maximum=MAX_SAFE_INT)
        equal(row.get("measurement_id"), i, "measurement-id-mismatch")
        loop, t_pos = divmod(i, num_targets)
        expected_target = active_targets[(loop + t_pos) % num_targets]
        equal(row.get("loop_index"), loop, "measurement-order-mismatch")
        equal(row.get("target_order_position"), t_pos, "measurement-order-mismatch")
        equal(row.get("target"), expected_target, "measurement-order-mismatch")

        status = row.get("status")
        require(
            status in ("measured", "invalid", "failed", "interrupted", "not_run"),
            "measurement-status",
        )
        attempted = row.get("attempted")
        require(type(attempted) is bool, "measurement-attempted")
        if attempted:
            require(not stopped, "invalid-execution-prefix")
        if status != "measured":
            stopped = True

        ref = row.get("plan_ref")
        plan = None
        if ref is not None:
            integer(ref, maximum=MAX_SAFE_INT)
            require(ref < len(target_plans) and target_plans[ref]["target"] == expected_target, "plan-reference")
            plan = target_plans[ref]
            equal(ref, frozen_refs[expected_target], "frozen-plan-reference")

        ex = row.get("execution")
        require(isinstance(ex, dict), "execution-invalid")
        timing_obj = ex.get("timing")
        require(isinstance(timing_obj, dict), "timing-invalid")
        elapsed = (
            timing_obj.get("elapsed_seconds")
            if status == "measured"
            else timing_obj.get("diagnostic_elapsed_seconds")
        )
        cpu_raw = timing_obj.get("cpu_raw")
        fpv = row.get("final_pointer_verification")
        require(isinstance(fpv, dict), "final-pointer-mismatch")

        if attempted:
            require(plan is not None, "plan-reference")
            levels.append(verify_cpu_raw_timing(cpu_raw, elapsed))
        else:
            require(timing_obj.get("elapsed_seconds") is None, "metric-null")
            require(ex.get("valid") is None, "execution-null")
            require(row.get("latency_ns") is None, "metric-null")
            equal(row.get("completed_accesses"), 0, "completion-mismatch")
            equal(row.get("completed_chain_cycles"), 0, "completion-mismatch")
            require(row.get("sample_windows") is None, "sample-window-null")
            require(fpv.get("valid") is None and fpv.get("actual_node") is None, "final-pointer-null")

        val_obj = ex.get("validation")
        require(isinstance(val_obj, dict), "validation-invalid")
        check_validation_checks(
            val_obj.get("checks"),
            _validation_specs(req_samples > 0),
            status == "measured",
        )

        if status == "measured":
            require(
                attempted
                and ex.get("valid") is True
                and timing_obj.get("evaluated") is True
                and timing_obj.get("valid") is True,
                "measurement-valid-mismatch",
            )
            require(
                type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                "elapsed-invalid",
            )
            equal(row.get("completed_chain_cycles"), plan["complete_chain_cycles"], "completion-mismatch")
            equal(row.get("completed_accesses"), plan["access_count"], "completion-mismatch")
            equal(fpv.get("expected_node"), 0, "final-pointer-mismatch")
            equal(fpv.get("actual_node"), 0, "final-pointer-mismatch")
            require(fpv.get("valid") is True, "final-pointer-mismatch")
            expected_latency_ns = (elapsed * 1e9) / float(plan["access_count"])
            close(row.get("latency_ns"), expected_latency_ns, "latency-mismatch")
            row_samples = _check_sample_windows(
                row.get("sample_windows"),
                req_samples,
                plan["access_count"],
                cpu_raw,
                target_info[expected_target]["indices"],
                status,
            )
            pooled_samples[expected_target].extend(row_samples)
            accepted[expected_target].append(i)
        else:
            require(row.get("latency_ns") is None, "metric-null")
            if attempted:
                _check_sample_windows(
                    row.get("sample_windows"),
                    req_samples,
                    plan["access_count"],
                    cpu_raw,
                    target_info[expected_target]["indices"],
                    status,
                )

    for loop, record in enumerate(loop_records):
        require(isinstance(record, dict), "loop-order-mismatch")
        equal(record.get("loop_index"), loop, "loop-order-mismatch")
        planned_order = [active_targets[(loop + k) % num_targets] for k in range(num_targets)]
        equal(record.get("planned_order"), planned_order, "loop-order-mismatch")
        start_idx = loop * num_targets
        attempted_order = [
            r["target"] for r in rows[start_idx : start_idx + num_targets] if r["attempted"]
        ]
        equal(record.get("realized_order"), attempted_order, "loop-prefix-mismatch")
        equal(record.get("realized_order_count"), len(attempted_order), "loop-prefix-mismatch")
        equal(
            record.get("measurement_indexes"),
            list(range(start_idx, start_idx + num_targets)),
            "loop-measurement-reference",
        )

    loc_16k_values = []
    loc_global_values = []
    loc_comp = doc.get("locality_comparison")
    doc_status = doc.get("status")
    if not compare_locality:
        require(loc_comp is None, "locality-unexpected")
    else:
        require(isinstance(loc_comp, dict), "locality-missing")
        rounds = loc_comp.get("rounds")
        require(isinstance(rounds, list) and len(rounds) <= loop_count * 3, "locality-rounds-invalid")
        if doc_status == "complete":
            require(len(rounds) == loop_count * 3, "locality-rounds-invalid")
        for r_idx, rnd in enumerate(rounds):
            budget(1)
            require(isinstance(rnd, dict), "locality-rounds-invalid")
            for key, dest in (("locality_16k", loc_16k_values), ("locality_global", loc_global_values)):
                sub = rnd.get(key)
                require(isinstance(sub, dict), "locality-rounds-invalid")
                sub_status = sub.get("status")
                require(sub_status in VALID_MEASUREMENT_STATUSES, "locality-rounds-invalid")
                sub_attempted = sub.get("attempted")
                require(type(sub_attempted) is bool, "locality-rounds-invalid")
                if doc_status == "complete":
                    require(
                        sub_status == "measured" and sub_attempted is True,
                        "locality-rounds-invalid",
                    )
                if sub_status == "measured":
                    require(sub_attempted is True, "locality-rounds-invalid")
                if sub_attempted:
                    ex = sub.get("execution")
                    require(isinstance(ex, dict), "locality-rounds-invalid")
                    t_obj = ex.get("timing")
                    require(isinstance(t_obj, dict), "timing-invalid")
                    el = (
                        t_obj.get("elapsed_seconds")
                        if sub_status == "measured"
                        else t_obj.get("diagnostic_elapsed_seconds")
                    )
                    levels.append(verify_cpu_raw_timing(t_obj.get("cpu_raw"), el))
                    if sub_status == "measured":
                        require(type(el) in (int, float) and math.isfinite(el) and el > 0, "elapsed-invalid")
                        acc = integer(sub.get("completed_accesses"), maximum=MAX_ACCESSES)
                        require(acc > 0, "locality-rounds-invalid")
                        fpv = sub.get("final_pointer_verification")
                        require(isinstance(fpv, dict), "final-pointer-mismatch")
                        equal(fpv.get("expected_node"), 0, "final-pointer-mismatch")
                        equal(fpv.get("actual_node"), 0, "final-pointer-mismatch")
                        require(fpv.get("valid") is True, "final-pointer-mismatch")
                        close(sub.get("latency_ns"), (el * 1e9) / float(acc), "latency-mismatch")
                        dest.append(sub["latency_ns"])

    aggs = doc.get("aggregates")
    require(isinstance(aggs, dict) and isinstance(aggs.get("targets"), dict), "aggregate-invalid")
    equal(set(aggs["targets"].keys()), set(active_targets), "aggregate-population-mismatch")

    for tname in active_targets:
        a = aggs["targets"][tname]
        require(isinstance(a, dict), "aggregate-invalid")
        ids = accepted[tname]
        equal(a.get("accepted_measurement_ids"), ids, "aggregate-population-mismatch")
        values = [rows[idx]["latency_ns"] for idx in ids]
        expected_stats = compute_statistics(values)
        n = len(ids)
        if expected_stats is None:
            require(a.get("value") is None and a.get("statistics") is None, "aggregate-null")
        else:
            close(a.get("value"), expected_stats["median"], "aggregate-headline-mismatch")
            check_statistics_block(a.get("statistics"), expected_stats)

        q = a.get("quality")
        require(isinstance(q, dict), "aggregate-quality-invalid")
        equal(q.get("repeatability_assessment_available"), n >= 2, "aggregate-quality-mismatch")
        equal(q.get("run_complete"), n == loop_count, "aggregate-quality-mismatch")
        close(q.get("cv_warning_threshold_pct"), CV_WARNING_THRESHOLD_PCT, "aggregate-cv-mismatch")
        cv_warn = (
            n >= 3
            and expected_stats is not None
            and expected_stats["coefficient_of_variation_pct"] is not None
            and expected_stats["coefficient_of_variation_pct"] > CV_WARNING_THRESHOLD_PCT
        )
        equal(q.get("cv_warning"), cv_warn, "aggregate-cv-mismatch")

        psd = a.get("pooled_sample_distribution")
        if req_samples == 0 or n == 0:
            if req_samples == 0:
                require(psd is None, "pooled-sample-null")
            elif psd is not None:
                require(isinstance(psd, dict) and psd.get("statistics") is None, "pooled-sample-null")
        else:
            require(isinstance(psd, dict), "pooled-sample-invalid")
            equal(
                psd.get("semantics"),
                "pooled-separate-sample-window-distribution",
                "pooled-sample-invalid",
            )
            expected_pooled = pooled_samples[tname]
            got_vals = psd.get("values_ns")
            if got_vals is not None:
                require(isinstance(got_vals, list) and len(got_vals) == len(expected_pooled), "pooled-sample-statistic-mismatch")
                for gv, ev in zip(got_vals, expected_pooled):
                    close(gv, ev, "pooled-sample-statistic-mismatch")
            expected_pooled_stats = compute_statistics(expected_pooled)
            check_statistics_block(
                psd.get("statistics"),
                expected_pooled_stats,
                count_reason="pooled-sample-statistic-mismatch",
                stat_reason="pooled-sample-statistic-mismatch",
            )

    agg_loc = aggs.get("locality_comparison")
    if not compare_locality:
        require(agg_loc is None, "locality-unexpected")
    else:
        require(isinstance(agg_loc, dict), "locality-missing")
        s16 = compute_statistics(loc_16k_values)
        sg = compute_statistics(loc_global_values)
        if doc_status == "complete":
            require(
                s16 is not None
                and sg is not None
                and len(loc_16k_values) == loop_count * 3
                and len(loc_global_values) == loop_count * 3,
                "locality-rounds-invalid",
            )
        if s16 is None or sg is None:
            require(
                agg_loc.get("locality_16k_latency_ns") is None
                and agg_loc.get("global_random_latency_ns") is None
                and agg_loc.get("locality_latency_delta_ns") is None,
                "locality-delta-mismatch",
            )
        else:
            close(agg_loc.get("locality_16k_latency_ns"), s16["median"], "locality-delta-mismatch")
            close(agg_loc.get("global_random_latency_ns"), sg["median"], "locality-delta-mismatch")
            close(
                agg_loc.get("locality_latency_delta_ns"),
                sg["median"] - s16["median"],
                "locality-delta-mismatch",
            )

    counters = doc.get("counters")
    require(isinstance(counters, dict), "counter-mismatch")
    measured_total = sum(len(accepted[t]) for t in active_targets)
    attempted_total = sum(1 for r in rows if r["attempted"])
    completed_loops = (
        sum(
            all(r["status"] == "measured" for r in rows[k : k + num_targets])
            for k in range(0, len(rows), num_targets)
        )
        if num_targets > 0
        else 0
    )
    expected_counters = {
        "planned_loops": loop_count,
        "attempted_loops": len({r["loop_index"] for r in rows if r["attempted"]}),
        "completed_loops": completed_loops,
        "planned_measurements": len(rows),
        "attempted_measurements": attempted_total,
        "terminal_measurements": sum(1 for r in rows if r["status"] != "not_run"),
        "measured_measurements": measured_total,
    }
    for key, val in expected_counters.items():
        equal(counters.get(key), val, "counter-mismatch")

    completed_accesses_total = sum(
        integer(r.get("completed_accesses"), maximum=MAX_SAFE_INT) for r in rows
    )
    equal(counters.get("completed_accesses"), str(completed_accesses_total), "counter-mismatch")
    if all(frozen_refs[t] is not None for t in active_targets):
        planned_accesses_total = (
            sum(target_plans[frozen_refs[t]]["access_count"] for t in active_targets) * loop_count
        )
    else:
        planned_accesses_total = 0
    equal(counters.get("planned_accesses"), str(planned_accesses_total), "counter-mismatch")

    check_checkpoint_lifecycle(doc.get("checkpoint_lifecycle"), loop_count, doc.get("status"))
    accepted_run = check_run_acceptance(doc, measured_total, len(rows), attempted_total, rows)
    complete = len(rows) > 0 and measured_total == len(rows)
    equal(
        doc.get("target_order_balance_complete"),
        complete and num_targets > 0 and loop_count % num_targets == 0,
        "order-balance-mismatch",
    )

    expected_interp = {
        "dram_residency": "unverified",
        "cache_residency": "unverified",
        "non_cacheable_requested": cfg["non_cacheable"],
        "non_cacheable_is_best_effort_madvise": True,
        "sample_windows_run_as_separate_pass_after_main_measurement": True,
        "comparability_requires": EXPECTED_COMPARABILITY,
    }
    equal(doc.get("interpretation"), expected_interp, "interpretation-invalid")

    check_quality_warnings(
        doc.get("quality_warnings"),
        env=env,
        cfg=cfg,
        buffer_evidences=lat_buffer_evidences,
        all_targets_have_resources=(set(active_targets).issubset(set(resources_obj.keys())) and len(active_targets) > 0),
        active_targets=active_targets,
        operations=None,
        agg_targets=aggs["targets"],
        attempted_any=attempted_any,
        iterations=iterations,
        rows=rows,
        duration_window=(0.100, 0.300),
    )

    enforce_require_raw_timing(require_raw, levels)
    manifest = doc.get("build_manifest")
    if binary_digest is None and binary is not None:
        binary_digest = hash_binary(binary)
    build_level = check_manifest(manifest, binary_digest=binary_digest)

    return dict(
        artifact_consistent=True,
        run_accepted=accepted_run,
        reason_code="valid" if accepted_run else "run-not-accepted",
        checks=dict(
            checksum="independent-oracle",
            timing=(
                "elapsed-only"
                if "elapsed-only" in levels
                else ("cpu-raw-ticks" if levels else "unavailable")
            ),
            build=build_level,
        ),
        oracle_work_used=work_before - budget.remaining,
    )


def _validate_lat_buffer_size(v):
    return integer(v, maximum=MAX_SAFE_INT)


def _validate_lat_cache_size(v):
    n = integer(v, maximum=1048576)
    require(n == 0 or n >= 16, "sweep-parameter-value-invalid")
    return n


def _validate_lat_locality_kb(v):
    return integer(v, maximum=MAX_SAFE_INT)


def _validate_lat_stride_bytes(v):
    n = integer(v, maximum=U32)
    require(n >= 8 and n % 8 == 0, "sweep-parameter-value-invalid")
    return n


def _validate_lat_chain_mode(v):
    try:
        return normalize_chain_mode(v)
    except Rejected as err:
        raise Rejected("sweep-parameter-value-invalid") from err


SWEEP_PARAM_SPECS = {
    "buffer-size": ("buffer_size_mb", _validate_lat_buffer_size),
    "cache-size": ("custom_cache_size_kb", _validate_lat_cache_size),
    "latency-tlb-locality-kb": ("latency_tlb_locality_kb", _validate_lat_locality_kb),
    "latency-stride-bytes": ("latency_stride_bytes", _validate_lat_stride_bytes),
    "latency-chain-mode": ("latency_chain_mode", _validate_lat_chain_mode),
}

SWEEP_RESOLVED_SOURCE_MAP = {
    "buffer-size": "buffer_size",
    "cache-size": "cache_size",
    "latency-tlb-locality-kb": "latency_tlb_locality_kb",
    "latency-stride-bytes": "latency_stride_bytes",
    "latency-chain-mode": "latency_chain_mode",
}

FIXED_CONFIG_KEYS = (
    "iterations",
    "work_policy",
    "loop_count",
    "latency_samples",
    "compare_locality",
    "non_cacheable",
    "base_seed_uint64_decimal",
    "seed_source",
    "argv",
    "resolved_sources",
)


def verify_sweep(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    max_chain_nodes=DEFAULT_MAX_CHAIN_NODES,
):
    """Verify a cpu_latency schema-1 sweep envelope."""
    return verify_sweep_envelope(
        doc,
        base_mode=MODE_NAME,
        param_specs=SWEEP_PARAM_SPECS,
        verify_single_fn=verify_single,
        fixed_config_keys=FIXED_CONFIG_KEYS,
        binary=binary,
        require_raw=require_raw,
        budget=budget,
        extra_verify_kwargs={"max_chain_nodes": max_chain_nodes},
        sweep_resolved_source_map=SWEEP_RESOLVED_SOURCE_MAP,
    )


def verify(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    max_chain_nodes=DEFAULT_MAX_CHAIN_NODES,
):
    """Verify either a direct cpu_latency artifact or a cpu_latency sweep envelope."""
    require(isinstance(doc, dict), "invalid-input")
    cfg = doc.get("configuration")
    if isinstance(cfg, dict) and cfg.get("mode") == "sweep" and cfg.get("base_mode") == MODE_NAME:
        return verify_sweep(
            doc,
            binary=binary,
            require_raw=require_raw,
            budget=budget,
            max_chain_nodes=max_chain_nodes,
        )
    return verify_single(
        doc,
        binary=binary,
        require_raw=require_raw,
        budget=budget,
        max_chain_nodes=max_chain_nodes,
    )


def _extra_args(parser):
    parser.add_argument(
        "--max-chain-nodes",
        type=int,
        default=DEFAULT_MAX_CHAIN_NODES,
        help=f"Maximum chain node count per chain (default: {DEFAULT_MAX_CHAIN_NODES})",
    )


def _verify_cli(doc, args):
    require(args.max_chain_nodes >= 2, "invalid-input")
    return verify(
        doc,
        binary=args.binary,
        require_raw=args.require_raw_timing,
        max_chain_nodes=args.max_chain_nodes,
    )


def main():
    return run_verifier_cli(_verify_cli, __doc__, extra_args_fn=_extra_args)


if __name__ == "__main__":
    sys.exit(main())
