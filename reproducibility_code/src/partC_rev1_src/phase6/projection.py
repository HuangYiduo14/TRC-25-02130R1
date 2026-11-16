"""Post-freeze HIGH projection with exact retained-coordinate inputs and loss."""
from pathlib import Path
import numpy as np
import pandas as pd

from .common import read_json, read_npz, write_json, write_npz, write_frame, input_hash, sha, Monitor
from .inputs import load_native, read_tier
from .registry import region_dir, native_lookup, prove_centroid_exclusion


def project_high(c):
    root = Path(c["output_root"])
    frozen = read_json(root / "samples/LOW_PLAN_FROZEN.json")
    if not frozen["status"] == "PASS": raise ValueError("LOW must freeze before projection")
    regions = pd.read_parquet(root / "registry/regions.parquet")
    with Monitor(c, "P1_HIGH_PROJECTION") as monitor:
        g, t, frames, sources, p4 = load_native(c)
        _, high = read_tier(c, g, t, sources, "high_benchmark")
        hk = high["current__coarse_regional_pair_flow__keys"]
        hv = high["current__coarse_regional_pair_flow__values"]
        hs = (hk >> np.uint64(32)).astype(np.int64); ht = (hk & np.uint64(0xffffffff)).astype(np.int64)
        hreg = t.terminal_region_local[hs]
        if not np.array_equal(hreg, t.terminal_region_local[ht]): raise ValueError("HIGH cross-region key")
        terms = pd.read_parquet(root / "registry/coarse_terminals.parquet")
        tg = terms.groupby("region_id").indices
        ledger = []; plans = []; unknown_proofs = []
        for desc in regions.to_dict("records"):
            monitor.check(); rid = desc["region_id"]; rd = region_dir(c, rid); ri = desc["region_index"]
            maps = read_npz(rd / "support_map.npz")
            full = maps["full_keys"]; active = full[maps["active_indices"]]
            sel = hreg == ri; regionkeys = hk[sel]; regionvalues = hv[sel]
            infull = np.isin(regionkeys, full); inact = np.isin(regionkeys, active)
            missing = regionkeys[~infull]
            cross = missing[(missing >> np.uint64(32)) != (missing & np.uint64(0xffffffff))]
            if len(cross):
                a = read_npz(rd / "local_graph.npz"); fm = read_npz(rd / "fine_members.npz")
                localmembers = {int(k): fm["local_nodes"][fm["offsets"][j]:fm["offsets"][j+1]] for j,k in enumerate(fm["terminal_indices"])}
                tr = terms.iloc[tg[rid]]
                for key in cross:
                    unknown_proofs.append(prove_centroid_exclusion(g, t, rid, tr, localmembers, a["global_node_index"],
                        a["global_arc_index"], int(key >> np.uint64(32)), int(key & np.uint64(0xffffffff))))
            projected = native_lookup(hk, hv, active)
            ledger.append(dict(region_id=rid, m_active=len(active), native_HIGH_keys=len(regionkeys),
                native_HIGH_exposure=float(regionvalues.sum()), projected_HIGH_exposure=float(projected.sum()),
                excluded_inactive_keys=int((infull & ~inact).sum()), excluded_inactive_exposure=float(regionvalues[infull & ~inact].sum()),
                outside_interface_keys=int((~infull).sum()), outside_interface_exposure=float(regionvalues[~infull].sum()),
                excluded_total_exposure=float(regionvalues[~inact].sum()), projected_nonzero=int((projected > 0).sum()),
                redistribution=False, background_flow=False, support_changed=False,
                no_active_skip=not bool(len(active))))
            if not len(active): continue
            plan = pd.read_parquet(rd / "baseline_LOW_plan.parquet")
            xs = read_npz(rd / "baseline_inputs_LOW_frozen.npz")["x"]
            ih = input_hash(desc["model_hash"], desc["support_hash"], projected)
            prior = plan.loc[plan.input_hash.eq(ih)]
            owner = int(prior.iloc[0].owner_role_index) if len(prior) else 48
            owner_split = str(prior.iloc[0].owner_split) if len(prior) else "HIGH_PROJECTED"
            highrow = dict(region_id=rid, model_hash=desc["model_hash"], support_hash=desc["support_hash"],
                input_hash=ih, input_index=48, role_index=48, split="HIGH_PROJECTED", family="HIGH_PROJECTED",
                family_slot=0, seed="NATIVE_HIGH_CURRENT_X200_PROJECTED", round_index=0,
                owner_role_index=owner, owner_split=owner_split, independent_role=owner==48, alias=owner!=48,
                target_gap=1e-6, scale_hash=read_json(rd / "LOW_freeze.json")["scale_hash"],
                generator="NATIVE_PROJECTION", numpy_version=np.__version__)
            plan = pd.concat([plan, pd.DataFrame([highrow])], ignore_index=True)
            write_npz(rd / "baseline_inputs.npz", x=np.vstack([xs, projected]))
            write_frame(rd / "baseline_plan.parquet", plan)
            write_json(rd / "BASELINE_PLAN_FROZEN.json", dict(planned_roles=49, unique_inputs=plan.input_hash.nunique(),
                input_sha256=sha(rd / "baseline_inputs.npz"), plan_sha256=sha(rd / "baseline_plan.parquet"),
                LOW_freeze_sha256=sha(rd / "LOW_freeze.json"), all_HIGH_loss_explicit=True,
                heldout_hashes=sorted(set(plan.loc[plan.split.ne("TRAIN"), "input_hash"]))))
            plans.append(plan)
        loss = pd.DataFrame(ledger)
        if not np.isclose(loss.projected_HIGH_exposure.sum()+loss.excluded_total_exposure.sum(), hv.sum(), rtol=1e-13, atol=1e-8):
            raise ValueError("HIGH exposure conservation failed")
        write_frame(root / "registry/high_projection_loss.parquet", loss)
        write_json(root / "registry/high_centroid_exclusion_proofs.json", unknown_proofs)
        allplan = pd.concat(plans, ignore_index=True)
        write_frame(root / "samples/baseline_plan.parquet", allplan)
        result = dict(status="PASS", planned_roles=len(allplan), unique_inputs=allplan.input_hash.nunique(),
            regions=len(regions), active_regions=len(plans), native_HIGH_exposure=float(hv.sum()),
            projected_HIGH_exposure=float(loss.projected_HIGH_exposure.sum()),
            excluded_HIGH_exposure=float(loss.excluded_total_exposure.sum()),
            frozen_LOW_plan_sha256=frozen["plan_sha256"], no_HIGH_selection=True,
            no_renormalization=True, all_heldout_hashes_frozen=True)
        write_json(root / "samples/BASELINE_PLAN_FROZEN.json", result)
        return result
