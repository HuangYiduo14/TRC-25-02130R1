"""Frozen active-only baseline and deterministic fresh-seed proportional packs."""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd

from . import MODEL_CLASS
from .common import (canonical, digest, arrays_hash, input_hash, read_json, read_npz,
                     write_json, write_npz, write_frame, sha, Monitor, event)
from .registry import region_dir, quota_target
from .oracle import Oracle
from . import kernels as K

PACK = (["RADIAL_RANDOM"] * 4 + ["PAIR_REWEIGHT"] * 8 + ["SOURCE_REWEIGHT"] * 4
        + ["TARGET_REWEIGHT"] * 4 + ["BOTTLENECK_RANDOM"] * 4
        + ["ACTIVE_MIX_RANDOM"] * 4 + ["ACTIVE_SPARSE_RANDOM"] * 4)


def rng_for(region, round_index, family, slot, redraw=0, *, support_hash):
    fields = [20260906, MODEL_CLASS, support_hash, str(region), int(round_index), family, int(slot), int(redraw)]
    h = hashlib.sha256(canonical(fields)).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h, "big"))), h.hex()


def composition(m, rng, sparse=False):
    count = max(1, (m + 1) // 2) if sparse else min(256, m)
    indices = rng.choice(m, count, replace=False)
    v = np.zeros(m); v[indices] = rng.uniform(.25, 1.75, count)
    return v / v.sum()


def reweight(w, src, tgt, family, rng, eta=.5):
    if family == "PAIR_REWEIGHT": noise = rng.uniform(-eta, eta, len(w))
    else:
        _, inv = np.unique(src if family == "SOURCE_REWEIGHT" else tgt, return_inverse=True)
        noise = rng.uniform(-eta, eta, int(inv.max()) + 1)[inv]
    v = w * np.exp(noise)
    return v / v.sum()


def draw(low, src, tgt, family, rng, masks):
    if not len(low) or (low <= 0).any(): raise ValueError("Only frozen LOW-active coordinates")
    total = math.fsum(low); w = low / total; tag = ""
    if family == "RADIAL_RANDOM": return rng.uniform(.1, 1.5) * low, tag
    if family in ["PAIR_REWEIGHT", "SOURCE_REWEIGHT", "TARGET_REWEIGHT"]:
        v = reweight(w, src, tgt, family, rng); gamma = rng.uniform(.75, 1.25)
    elif family == "BOTTLENECK_RANDOM":
        if len(masks):
            v = w.copy(); v[masks[int(rng.integers(len(masks)))]] *= rng.uniform(1.25, 2.5)
            gamma = rng.uniform(.75, 1.5)
        else:
            v = reweight(w, src, tgt, "PAIR_REWEIGHT", rng); gamma = rng.uniform(.75, 1.25)
            tag = "NO_USED_ARC_SEPARATE_PAIR_REWEIGHT_SEED"
    elif family in ["ACTIVE_MIX_RANDOM", "ACTIVE_SPARSE_RANDOM"]:
        v = composition(len(w), rng, family == "ACTIVE_SPARSE_RANDOM")
        if family == "ACTIVE_MIX_RANDOM":
            rho = rng.uniform(.05, .5); v = (1 - rho) * w + rho * v
        gamma = rng.uniform(.75, 1.5)
    else: raise ValueError("Unknown family")
    x = gamma * total * (v / v.sum())
    if (x < 0).any() or not np.isfinite(x).all(): raise ValueError("Nonfinite constructed input")
    return x, tag


def scales_and_masks(a, low):
    o = Oracle(a); m = len(low); total = math.fsum(low)
    unit = np.full(m, 1 / m)
    abar = K.load_routes(unit, o.free_offsets, o.free_arcs, len(a["tail"]))
    used = abar > 0
    qstruct = float(np.min(.25 * a["capacity"][used] / abar[used])) if used.any() else 1.
    scal = np.maximum(np.maximum(low, total / m), 1e-12)
    vscale = max(float(o.c0 @ low), total * float(o.c0.mean()), 1.)
    load = K.load_routes(low, o.free_offsets, o.free_arcs, len(a["tail"]))
    used_arcs = np.flatnonzero(load > 0)
    rank = np.lexsort((a["physical_arc_id"][used_arcs], -(load[used_arcs] / a["capacity"][used_arcs])))
    selected = used_arcs[rank[:4]]
    masks = np.zeros((len(selected), m), bool)
    for j, arc in enumerate(selected):
        for k in range(m): masks[j, k] = arc in o.free_arcs[o.free_offsets[k]:o.free_offsets[k + 1]]
    if not np.isfinite(qstruct) or qstruct <= 0 or not np.isfinite(vscale):
        raise ValueError("Invalid physical sampling scale")
    result = dict(input_scale=scal, c0=o.c0, Qbase=np.array(total), Q_struct=np.array(qstruct),
                  Vscale=np.array(vscale), bottleneck_masks=masks, bottleneck_arc_local_ids=selected)
    return result


def baseline(desc, low, src, tgt, scales):
    m = len(low)
    if not m: return np.empty((0, 0)), []
    total = math.fsum(low); w = low / total; rows = []; xs = []; hashes = {}
    def add(x, split, family, slot, seed="ANALYTIC", **extra):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (m,) or (x < 0).any() or not np.isfinite(x).all(): raise ValueError("Bad baseline input")
        ih = input_hash(desc["model_hash"], desc["support_hash"], x)
        owner = hashes.setdefault(ih, {"slot": len(rows), "split": split})
        rows.append(dict(region_id=desc["region_id"], model_hash=desc["model_hash"],
            support_hash=desc["support_hash"], input_hash=ih, input_index=len(xs), role_index=len(rows),
            split=split, family=family, family_slot=slot, seed=seed, round_index=0,
            owner_role_index=owner["slot"], owner_split=owner["split"],
            independent_role=owner["slot"] == len(rows), alias=owner["slot"] != len(rows),
            target_gap=1e-6 if family == "LOW" else 1e-5, **extra))
        xs.append(x)
    add(np.zeros(m), "TRAIN", "ZERO", 0); add(low.copy(), "TRAIN", "LOW", 0)
    for j, gamma in enumerate([.1, .25, .5, .75, 1.25, 1.5]):
        add(gamma * low, "TRAIN", "RADIAL_TEMPLATE", j)
    for family, count in [("PAIR_REWEIGHT", 8), ("SOURCE_REWEIGHT", 4), ("TARGET_REWEIGHT", 4)]:
        for slot in range(count):
            rng, seed = rng_for(desc["region_id"], 0, "TRAIN_" + family, slot, support_hash=desc["support_hash"])
            v = reweight(w, src, tgt, family, rng); gamma = rng.uniform(.75, 1.25)
            add(gamma * total * v, "TRAIN", family, slot, seed)
    masks = scales["bottleneck_masks"]
    for slot, gamma in enumerate([.75, 1., 1.25, 1.5]):
        rng, seed = rng_for(desc["region_id"], 0, "TRAIN_BOTTLENECK", slot, support_hash=desc["support_hash"])
        if len(masks):
            v = w.copy(); v[masks[slot % len(masks)]] *= 2; v /= v.sum(); tag = ""
        else:
            v = reweight(w, src, tgt, "PAIR_REWEIGHT", rng); gamma = rng.uniform(.75, 1.25); tag = "NO_USED_ARC"
        add(gamma * total * v, "TRAIN", "BOTTLENECK_GROUP", slot, seed, degeneracy=tag)
    for slot, (rho, gamma) in enumerate(zip([.01, .025, .05, .1], [.75, 1, 1.25, 1.5])):
        rng, seed = rng_for(desc["region_id"], 0, "TRAIN_ACTIVE_COMPOSITION_MIX", slot, support_hash=desc["support_hash"])
        v = (1 - rho) * w + rho * composition(m, rng)
        add(gamma * total * (v / v.sum()), "TRAIN", "ACTIVE_COMPOSITION_MIX", slot, seed)
    # Two disjoint transfer directions where possible; dimension-one uses insertion.
    permutation, _ = rng_for(desc["region_id"], 0, "VALID_DIRECTIONS", 0, support_hash=desc["support_hash"])
    order = permutation.permutation(m)
    for direction_index in range(2):
        d = np.zeros(m)
        if m >= 2:
            indices = order[2 * direction_index:2 * direction_index + 2]
            if len(indices) != 2: indices = order[:2]
            d[indices[0]] = .5; d[indices[1]] = -.5
        else: d[0] = 1
        h = 1e-3 * total
        if np.any(low + h * d < 0) and np.any(low - h * d < 0):
            # Both transfers fail at this prescribed step. An insertion probe is
            # feasible and records the actual retained-coordinate direction.
            d[:] = 0.; d[order[direction_index % m]] = 1.
        elif np.any(low + h * d < 0):
            d = -d
        central = np.all(low - h * d >= 0) and np.all(low + h * d >= 0)
        steps = [-h, h] if central else [h / 2, h]
        for j, step in enumerate(steps):
            x = low + step * d
            add(x, "VALID", "FINITE_DIFFERENCE", direction_index * 2 + j,
                direction_index=direction_index, signed_step=float(step), central=bool(central),
                direction=d.tolist())
    for slot, (rho, gamma) in enumerate(zip([.02, .08, .02, .08], [.8, 1.2, 1.1, .9])):
        rng, seed = rng_for(desc["region_id"], 0, "VALID_MIX", slot, support_hash=desc["support_hash"])
        v = reweight(w, src, tgt, "SOURCE_REWEIGHT" if slot % 2 == 0 else "TARGET_REWEIGHT", rng)
        v = (1 - rho) * v + rho * composition(m, rng)
        add(gamma * total * v / v.sum(), "VALID", "MIXED_PERTURBATION", slot, seed)
    for slot, gamma in enumerate([.6, .9, 1.1, 1.4]):
        rng, seed = rng_for(desc["region_id"], 0, "LOW_TEST_PERTURBATION", slot, support_hash=desc["support_hash"])
        v = reweight(w, src, tgt, ["PAIR_REWEIGHT", "SOURCE_REWEIGHT", "TARGET_REWEIGHT", "PAIR_REWEIGHT"][slot], rng, eta=.75)
        if slot in [2, 3]: v = .95 * v + .05 * composition(m, rng)
        add(gamma * total * v / v.sum(), "LOW_TEST", "PERTURBATION", slot, seed)
    for slot, gamma in enumerate([1.75, 2.]): add(gamma * low, "LOW_TEST", "RADIAL_STRESS", slot)
    for slot, gamma in enumerate([1., 2.]):
        add(np.full(m, gamma * float(scales["Q_struct"]) / m), "LOW_TEST", "CAPACITY_STRESS", slot)
    if len(rows) != 48 or sum(r["split"] == "TRAIN" for r in rows) != 32:
        raise AssertionError("Incorrect baseline roles")
    return np.asarray(xs), rows


def prepare_all_low(c):
    root = Path(c["output_root"])
    registry = pd.read_parquet(root / "registry/regions.parquet")
    rows = []
    with Monitor(c, "P1_LOW_SAMPLE_FREEZE") as monitor:
        for desc in registry.to_dict("records"):
            if not desc["m_active"]: continue
            monitor.check(); rd = region_dir(c, desc["region_id"])
            a = read_npz(rd / "local_graph.npz"); low = read_npz(rd / "support_map.npz")["LOW_ACTIVE"]
            pairs = pd.read_parquet(rd / "active_pairs.parquet")
            scales = scales_and_masks(a, low)
            scale_hash = arrays_hash(scales)
            x, plan = baseline(desc, low, pairs.source_terminal_index.to_numpy(), pairs.target_terminal_index.to_numpy(), scales)
            for r in plan: r["scale_hash"] = scale_hash; r["generator"] = "PCG64"; r["numpy_version"] = np.__version__
            write_npz(rd / "scales.npz", **scales)
            write_npz(rd / "baseline_inputs_LOW_frozen.npz", x=x)
            write_frame(rd / "baseline_LOW_plan.parquet", pd.DataFrame(plan))
            write_json(rd / "LOW_freeze.json", dict(scale_hash=scale_hash,
                native_inputs_hash=arrays_hash({"x": x}), planned_roles=48,
                unique_inputs=len(set(r["input_hash"] for r in plan)), high_used=False))
            rows.extend(plan)
            if len(rows) % 1200 == 0: print("LOW plans", len(rows), flush=True)
        write_frame(root / "samples/LOW_frozen_plan.parquet", pd.DataFrame(rows))
        record = dict(status="PASS", roles=len(rows), active_regions=int((registry.m_active > 0).sum()),
            plan_sha256=sha(root / "samples/LOW_frozen_plan.parquet"), high_used_for_selection=False,
            generator="PCG64", numpy_version=np.__version__, seed_fields=[20260906, MODEL_CLASS, "support_hash", "region", "round", "family", "slot", "redraw"],
            native_hash_contract="SHA256(canonical model/support/shape + little-endian binary64); signed zero canonicalized",
            proportional_proxy="65*m+1121", target="32*max(2,ceil(P/1024))", no_unobserved_activation=True)
        write_json(root / "samples/LOW_PLAN_FROZEN.json", record)
        return record


def extra_candidate(desc, low, src, tgt, scales, round_index, slot, existing, protected):
    family = PACK[slot % 32]
    ledger = []
    for redraw in range(9):
        seed_family = family
        if family == "BOTTLENECK_RANDOM" and not len(scales["bottleneck_masks"]):
            seed_family += "_NO_USED_ARC_PAIR_REWEIGHT"
        rng, seed = rng_for(desc["region_id"], round_index, seed_family, slot, redraw,
                            support_hash=desc["support_hash"])
        x, tag = draw(low, src, tgt, family, rng, scales["bottleneck_masks"])
        ih = input_hash(desc["model_hash"], desc["support_hash"], x)
        reason = "HELDOUT_COLLISION" if ih in protected else "TRAIN_DUPLICATE" if ih in existing else "NEW"
        record = dict(region_id=desc["region_id"], round_index=round_index, family=family, slot=slot,
                      redraw=redraw, seed=seed, input_hash=ih, disposition=reason, degeneracy=tag)
        ledger.append(record)
        if reason == "NEW": return x, record, ledger
    return None, {**ledger[-1], "disposition": "NO_UNIQUE_INPUT"}, ledger
