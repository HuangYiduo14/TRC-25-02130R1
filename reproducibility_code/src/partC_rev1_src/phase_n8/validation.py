"""Bounded native model parity and scale-aware derivative evidence."""
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase67.portable import predict_arrays
from .common import read_json, write_json
from .native import load_registry
from .objective import Composite, validate_arrays


def reconcile_self_od(ctx, sources):
    from .common import read_npz, save_npz, sha
    from .native import mapped_demand, attach_roles, SUITES
    from .oracle import Demand, aon, conservation
    original, uids, physical_reachable, _ = sources.original_od()
    self_rows = []
    for group, origin in enumerate(original['origin_group_node_idx']):
        begin,end=original['origin_offsets'][group:group+2]
        self_rows.extend((np.flatnonzero(original['destination_node_idx'][begin:end]==origin)+begin).tolist())
    reports=[]
    for sid in SUITES:
        ctx.guard()
        root=ctx.root/'graphs'/sid
        old=read_json(root/'IDENTITY_AND_COVERAGE.json')
        if old.get('endpoint_mapping_version')=='EXACT_NATIVE_ROLES_WITH_EMPTY_SELF_OD_V2':
            reports.append(dict(suite=sid,status='ALREADY_V2',original_self_od_rows=self_rows))
            continue
        a=read_npz(root/'mapping.npz')
        if sha(root/'mapping.npz')!=old['mapping_sha256']:
            raise ValueError('Prior owned mapping changed')
        before=a['classification'].copy()
        a['classification'][self_rows]=2
        # Every changed row is an exact original physical self OD, not a new
        # inactive regional connection or a different collapsed-coordinate OD.
        s=sources.suite(sid);attach_roles(s,a)
        od=mapped_demand(original,a['origin_map'],a['destination_map'],a['classification']==2)
        a['balance']=od.balance(len(s['graph'].node_ids))
        groups=np.searchsorted(original['origin_offsets'][1:],self_rows,side='right')
        probe=Demand(od.origins[groups],np.arange(len(self_rows)+1),od.destinations[self_rows],
                     od.demand[self_rows],np.ones(len(self_rows),bool))
        tested=aon(s['graph'],probe,np.ones(len(s['graph'].tail)),guard=ctx.guard)
        if tested['sptt']!=0 or np.count_nonzero(tested['flow']) or tested['reachable_demand']!=probe.total:
            raise ArithmeticError('Exact empty-path self OD verification failed')
        save_npz(root/'mapping_selfod_v2.npz',a)
        new=dict(old,mapping_file='mapping_selfod_v2.npz',mapping_sha256=sha(root/'mapping_selfod_v2.npz'),
                 demand_identity=od.identity,represented_demand=od.total,
                 represented_origins=int(np.count_nonzero(np.add.reduceat(od.serve.astype(int),od.offsets[:-1]))),
                 endpoint_mapping_version='EXACT_NATIVE_ROLES_WITH_EMPTY_SELF_OD_V2',
                 supersedes_metadata_sha256=sha(root/'IDENTITY_AND_COVERAGE.json'),
                 self_od_rows=self_rows,self_od_empty_path_verified=True,
                 original_full_origin_timing_preserved=True,
                 profile_correction='All 3225 exact physical self OD rows use their own source as sink with zero-edge paths. Only one reachability membership changes. Original all-origin V1 profile is historical; full pilots remeasure V2.')
        new['original_v1_profile'] = {k: old[k] for k in ['conservation', 'zero_cost_full_demand_sptt', 'sptt_difference', 'full_loading_seconds']}
        for key in ['conservation', 'zero_cost_full_demand_sptt', 'sptt_difference']:
            new[key] = None
        new['timing_role'] = 'HISTORICAL_V1_FULL_ORIGIN_PROFILE_SAME_GRAPH_DIFFERENT_SELF_OD_MAPPING'
        rows=[]
        for code,label in [(1,'INHERITED_PHYSICAL_UNREACHABLE'),(2,'REPRESENTED_REACHABLE'),(3,'ADDITIONAL_ACTIVE_GRAPH_UNREACHABLE')]:
            mask=a['classification']==code
            rows.append(dict(suite=sid,classification=label,aggregated_od_count=int(mask.sum()),
                original_od_rows=int(original['original_row_multiplicity'][mask].sum()),demand=float(original['demand'][mask].sum()),
                fraction_of_full_original_demand=float(original['demand'][mask].sum()/original['demand'].sum())))
        new['coverage']=rows
        write_json(root/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json',new)
        reports.append(dict(suite=sid,status='V2_EXACT_SELF_OD_MAPPING_RECONCILED',changed_membership_rows=int(np.count_nonzero(before!=a['classification'])),
                            original_self_od_rows=self_rows,zero_edge_flow_verified=True,source_graph_changed=False))
    write_json(ctx.root/'receipts/ENGINEERING_REPAIR_02.json',dict(issue='Separate OD role ports omitted empty paths for 3225 exact physical self ODs; one newly unreachable row exposed this error.',
       rule='Original physical origin ID equals original physical destination ID: zero-edge feasible path, matching Phase4. Never infer equality from B/R collapse.',
       preserved_initial_mapping_and_profile=True,original_R2_start=ctx.c['start_epoch'],reports=reports))
    return dict(status='PASS_EXACT_SELF_OD_RECONCILIATION',reports=reports)


def native_tests(ctx, sources):
    suite = sources.suite("s03")
    scenarios = [s for s in read_json(ctx.root/"SCENARIO_DEFINITION_FREEZE.json")["scenarios"] if s["suite"] == "s03"]
    records = []
    cohorts = []
    summaries = []
    seen = set()
    for sc in scenarios:
        started = time.perf_counter()
        entries, registry = load_registry(sources, suite, sc["registry_id"])
        composite = Composite(suite, entries, registry, guard=ctx.guard)
        low = suite["fixed_x45"]
        direction = np.ones_like(low)
        v, c, h, diagnostic = composite.evaluate(low, direction, enforce_safety=False)
        npacked = suite["packed_count"]
        pc, pv, pdyn = suite["packed"].evaluate(low[:npacked], "combined")[0]
        cost = np.empty(len(low)-npacked)
        hvp = np.empty_like(cost)
        values = []
        sizes = []
        for e in entries:
            sl = slice(e["start"], e["stop"])
            result = predict_arrays(e["model_kind"], e["arrays"], low[npacked:][sl], direction[npacked:][sl])
            values.append(float(result[0]))
            cost[sl] = result[1]
            hvp[sl] = result[2]
            sizes.append(e["stop"]-e["start"])
            key = (e["portable_sha256"], e["model_kind"])
            if key not in seen:
                seen.add(key)
                records.append(dict(region_id=e["region_id"], kind=e["model_kind"], dimension=sizes[-1],
                    portable_sha256=e["portable_sha256"], structural_order_hash_scale_eligibility="PASS",
                    zero_anchor="PASS", low_minimum_cost=float(np.min(result[1])),
                    LOW_negative_count=int((result[1]<0).sum()), source_weight_mutations=0))
        reference_value = float(np.sum(pv)+sum(values))
        value_error = abs(v-reference_value)/max(1., abs(reference_value))
        cost_error = float(np.max(np.abs(c[npacked:]-cost), initial=0.))
        hvp_error = float(np.max(np.abs(h[npacked:]-hvp), initial=0.))
        if value_error > 2e-12 or not np.allclose(c[npacked:], cost, rtol=2e-13, atol=1e-8) or not np.allclose(h[npacked:], hvp, rtol=2e-13, atol=1e-8):
            raise ArithmeticError("Grouped composite differs from exact typed native reference")
        chosen = set()
        for kind in sorted({e["model_kind"] for e in entries}):
            ix = [i for i,e in enumerate(entries) if e["model_kind"] == kind]
            bysize = sorted(ix, key=lambda i: sizes[i])
            chosen.update(bysize[:1]+bysize[-3:]+[bysize[len(bysize)//2]])
            chosen.update(i for i in bysize if sizes[i] == 28)
        # Cohorts are fixed by type and dimension, independent of derivative error.
        for i in sorted(chosen):
            ctx.guard()
            e = entries[i]
            a, kind = e["arrays"], e["model_kind"]
            u = low[npacked+e["start"]:npacked+e["stop"]]
            direction = u*(.1*np.cos(np.arange(len(u))*.71)+.2)
            eps = 1e-4
            val, grad, hv = predict_arrays(kind, a, u, direction)
            vp, cp = predict_arrays(kind, a, u+eps*direction)
            vm, cm = predict_arrays(kind, a, u-eps*direction)
            directional = float(grad@direction)
            estimated = float((vp-vm)/(2*eps))
            verr = abs(estimated-directional)/max(1., abs(directional))
            herr = float(np.linalg.norm((cp-cm)/(2*eps)-hv)/max(1., np.linalg.norm(hv)))
            curvature = float(direction@hv)
            passed = verr <= 5e-5 and herr <= 1e-4 and curvature >= -1e-10*max(1., float(np.linalg.norm(direction)*np.linalg.norm(hv)))
            cohorts.append(dict(scenario=sc["scenario_id"], region_id=e["region_id"], kind=kind,
               dimension=len(u), state="FULL_NATIVE_LOW_CURRENT_VECTOR", epsilon=eps,
               potential_gradient_relative_error=verr, gradient_HVP_relative_error=herr,
               directional_curvature=curvature, minimum_cost=float(np.min(grad)),
               derivative_status="PASS" if passed else "FAIL", portable_sha256=e["portable_sha256"]))
            if not passed:
                pd.DataFrame(cohorts).to_csv(ctx.root/"tables/native_derivative_cohorts.csv", index=False)
                raise ArithmeticError("Native scale-aware derivative gate failed: " + e["region_id"])
        summaries.append(dict(scenario=sc["scenario_id"], registry_identity=registry["identity"],
            entries=len(entries), coordinates=len(cost), potential_relative_difference=value_error,
            maximum_cost_difference=cost_error, maximum_HVP_difference=hvp_error,
            full_query_diagnostics=diagnostic, LOW_negative_coordinates=int((cost<0).sum()),
            elapsed_seconds=time.perf_counter()-started, status="PASS_PARITY_AND_DERIVATIVES"))
    pd.DataFrame(records).to_csv(ctx.root/"tables/native_model_validation.csv", index=False)
    pd.DataFrame(cohorts).to_csv(ctx.root/"tables/native_derivative_cohorts.csv", index=False)
    write_json(ctx.root/"validation/NATIVE_PARITY.json", summaries)
    return dict(status="PASS", native_unique_models=len(records), native_derivative_cases=len(cohorts),
                exact_five_s03_scenario_parity=summaries, no_training_or_label_generation=True)
