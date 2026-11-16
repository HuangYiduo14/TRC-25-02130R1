"""Explicit P-ACNHN Stage2 controller. Importing this module launches no work.

Regional concurrency is chosen by a bounded correctness/throughput gate,
within the approved maximum of four workers.
All immutable scientific checkpoint files are retained; only a documented finite
pool remains eligible for selection. No TEST metric influences selection/schedule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import shutil

import numpy as np
import psutil
import torch

from .control import Context, array_hash, read_json, save_npz, sha, write_json
from .models import (MonotoneICNN, Power4Model, cost_wmape, fit_power4_cost,
                     fit_xgboost_cost, load_model, save_model, update_icnn,
                     calibrate_icnn_train, ICNN_NATIVE, ICNN_SCALED, XGB_UNCENTERED, XGB_CENTERED)
from .oracle import Network, aon
from .regional import (load_region_graph, sample_requests, generate_labels,
                       load_admitted_dataset)

FAMILIES = ("ICNN", "XGBOOST")
CONFIG_SCHEMA = "PACNHN_TRAINING_CONFIG_V1"
OPTIMIZER_SCHEMA = "PACNHN_ICNN_OPTIMIZER_V2"


def _atomic_torch_save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _new_json(path, value):
    if Path(path).exists():
        raise FileExistsError(path)
    write_json(path, value)


def _tree_bytes(path, *, exclude=None):
    total=0
    for directory,subdirectories,files in os.walk(path):
        if exclude is not None:
            subdirectories[:]=[name for name in subdirectories if Path(directory)/name!=exclude]
        for name in files:
            total+=(Path(directory)/name).stat().st_size
    return total


def label_solver_settings(config):
    """Resolve versioned generation policy; all historical attempt slots count."""
    first = config.get("label_solver_method", "FW")
    retry = config.get("label_retry_solver_method", first)
    if first not in {"FW", "CFW"} or retry not in {"FW", "CFW"}:
        raise ValueError("Local label methods must be FW or CFW")
    iterations = config["label_max_iterations"]
    retry_iterations = config.get("label_retry_max_iterations", iterations)
    maximum = config["max_attempts_per_input"]
    if any(type(v) is not int or v <= 0 for v in (iterations, retry_iterations, maximum)):
        raise ValueError("Label iteration and global attempt caps must be positive integers")
    bound = config.get("label_cfw_reset_on_bound", False)
    retry_bound = config.get("label_retry_cfw_reset_on_bound", bound)
    if type(bound) is not bool or type(retry_bound) is not bool:
        raise ValueError("Label bound-reset policies must be explicit booleans")
    selection = config.get("label_iterate_selection", "LAST_CERTIFIED")
    retry_selection = config.get("label_retry_iterate_selection", selection)
    allowed = {"LAST_CERTIFIED", "BEST_CERTIFIED_RG"}
    if type(selection) is not str or type(retry_selection) is not str or selection not in allowed or retry_selection not in allowed:
        raise ValueError("Explicit label iterate selection must be LAST_CERTIFIED or BEST_CERTIFIED_RG")
    return dict(solver_method=first, retry_solver_method=retry,
        max_iterations=iterations, retry_max_iterations=retry_iterations,
        cfw_reset_on_bound=bound, retry_cfw_reset_on_bound=retry_bound,
        max_attempts_per_input=maximum, iterate_selection=selection, retry_iterate_selection=retry_selection)


def read_training_config(path):
    c = read_json(path)
    if c.get("schema") != CONFIG_SCHEMA or c.get("phase_id") != "PACNHN_V1":
        raise ValueError("Explicit current training configuration required")
    positive = ("stage_seconds", "final_evaluation_reserve_seconds", "initial_train", "fixed_valid", "blind_test",
        "train_growth", "icnn_updates_per_block", "xgb_rounds_per_block", "eligible_candidate_cap_per_family",
        "icnn_blocks_before_next_width", "region_slice_seconds", "per_label_seconds", "label_max_iterations",
        "max_attempts_per_input", "audit_batch_rows", "icnn_batch_rows")
    if any(not math.isfinite(c[k]) or c[k] <= 0 for k in positive):
        raise ValueError("Nonpositive/incomplete explicit training budgets")
    if c["stage_seconds"] != 86400 or not 1<=c["regional_workers"]<=4 or c["torch_threads"] != 1 or c["xgboost_threads"] != 1:
        raise ValueError("Require approved 24h, at most4 regional workers and one compute thread each")
    if c["eligible_candidate_cap_per_family"] < len(c["icnn_widths"])+2 or not 0 < c["label_fraction_per_slice"] < 1:
        raise ValueError("Invalid candidate pool or slice policy")
    if c.get('icnn_coefficient_parameterization',ICNN_NATIVE) not in {ICNN_NATIVE,ICNN_SCALED}:
        raise ValueError('Unknown ICNN coefficient parameterization')
    if c.get('icnn_coefficient_parameterization')==ICNN_SCALED and not 0<c.get('icnn_initial_neural_cost_share',0)<1:
        raise ValueError('Explicit TRAIN neural initialization share required')
    if c.get('xgb_log_center_policy',XGB_UNCENTERED) not in {XGB_UNCENTERED,XGB_CENTERED}:
        raise ValueError('Unknown XGBoost log-center policy')
    label_solver_settings(c)
    return c


def stage_clock(ctx, config):
    root = ctx.root/"training"; root.mkdir(parents=True, exist_ok=True)
    path = root/"TRAINING_CLOCK.json"
    maximum = min(ctx.c["maximum_cumulative_stage_deadlines"][1], ctx.c["hard_deadline_epoch"])
    if path.exists():
        clock = read_json(path)
        expected = min(clock["stage_start_epoch"]+config["stage_seconds"], maximum)
        if (clock["original_campaign_start_epoch"] != ctx.c["original_start_epoch"]
                or clock["stage_deadline_epoch"] != expected
                or not ctx.c["original_start_epoch"] <= clock["stage_start_epoch"] <= time.time()
                or clock["stage_seconds"] != config["stage_seconds"]):
            raise ValueError("Original training clock changed; reset prohibited")
    else:
        started = getattr(ctx,'started',time.time())
        clock = dict(original_campaign_start_epoch=ctx.c["original_start_epoch"], stage_start_epoch=started,
            stage_seconds=config["stage_seconds"], stage_deadline_epoch=min(started+config["stage_seconds"], maximum),
            output_bytes_before_training=_tree_bytes(ctx.root,exclude=root))
        _new_json(path, clock)
    return clock


def gate_inputs(ctx):
    support_path = ctx.root/"SUPPORT_FREEZE.json"; witness_path = ctx.root/"COMPLETE_WITNESS_AUDIT.json"
    support = read_json(support_path); witness = read_json(witness_path)
    if support.get("status") != "PASS" or witness.get("status") != "PASS":
        raise ValueError("Support and complete witness hard gates must PASS before any labels/training")
    if witness.get("support_freeze_sha256") != sha(support_path):
        raise ValueError("Complete witness gate is not bound to these exact frozen support bytes")
    checks = {r["scenario"]: r for r in witness.get("checks", [])}
    if set(checks) != {"open", "i95"} or not all(r.get("all_five_resolutions") and r.get("all_directed_P_arcs_and_masks_valid")
            and r.get("all_segments_in_frozen_support") for r in checks.values()):
        raise ValueError("Incomplete open/closed witness hard gate")
    if not all(r.get("final_certificate_fresh") for r in support.get("source_references", [])) or len(support.get("source_references", [])) != 2:
        raise ValueError("Two frozen state-consistent P references required")
    ctx.bind(support_path, "FROZEN_NATIVE_REGIONAL_SUPPORT")
    ctx.bind(witness_path, "COMPLETE_OPEN_CLOSED_WITNESS_GATE")
    regions = []
    for suite in support["suites"]:
        sid = suite["suite_id"]
        if sid == "s01" or sid not in ctx.c["suite_ids"] or not suite.get("support_frozen_before_labels"):
            raise ValueError("Invalid/unauthorized suite support")
        if not suite["active_regions"] or len(suite["region_records"]) != suite["active_regions"]:
            raise ValueError("Empty/incomplete regional inventory")
        if len(suite["coverage"]) != 2 or not all(r["exact_OD_membership"] for r in suite["coverage"]):
            raise ValueError("Exact scenario OD-membership gate missing")
        for row in suite["region_records"]:
            if row["suite_id"] != sid or not row["all_pairs_lift"] or row["pairs"] < 1:
                raise ValueError("Invalid active-region lift record")
            path = Path(row["path"])
            metadata_path = path.parent/"REGION.json"
            if read_json(metadata_path) != row:
                raise ValueError("REGION metadata differs from support freeze")
            ctx.bind(metadata_path, "REGIONAL_NATIVE_IDENTITY")
            regions.append(dict(row, key=f'{sid}:{row["region_index"]:06d}',graph_file_bytes=path.stat().st_size))
    if set(r["suite_id"] for r in regions) != set(ctx.c["suite_ids"]):
        raise ValueError("Frozen support must inventory all five authorized suites")
    if len({r["key"] for r in regions}) != len(regions):
        raise ValueError("Duplicate regional identities")
    return support, sorted(regions, key=lambda r: (r["suite_id"], r["region_index"]))


def choose_device(root, config):
    """Bounded synthetic FP64 correctness/timing gate; never a GPU assumption."""
    path = Path(root)/"DEVICE_GATE.json"
    if path.exists():
        return read_json(path)["selected_device"]
    result = dict(selected_device="cpu", torch_version=torch.__version__, policy=config["gpu_policy"],
                  cuda_available=bool(torch.cuda.is_available()), correct=False)
    if config["gpu_policy"] == "pilot_then_use_only_if_faster" and result["cuda_available"]:
        try:
            dimensions=1024
            q = np.random.default_rng(20260925).uniform(.1, 2., (128, dimensions))
            target = 2.+.2*q**4; timings = {}; costs = {}; models = {}
            for device in ("cpu", "cuda"):
                model = MonotoneICNN(np.full(dimensions, 2.), np.ones(dimensions), width=32, seed=11, device=device)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                update_icnn(model, optimizer, q, target, train_cost_mean=float(target.mean()))
                if device == "cuda":
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(6):
                    update_icnn(model, optimizer, q, target, train_cost_mean=float(target.mean()))
                if device == "cuda":
                    torch.cuda.synchronize()
                timings[device] = time.perf_counter()-started
                costs[device] = model.cost(q); models[device] = model
            np.testing.assert_allclose(costs["cpu"], costs["cuda"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(models["cpu"].hess_vec(q[0], np.ones(dimensions)),
                models["cuda"].hess_vec(q[0], np.ones(dimensions)), rtol=1e-9, atol=1e-10)
            memory = torch.cuda.max_memory_allocated()
            result.update(correct=True, measured_seconds=timings, peak_gpu_bytes=memory,
                          device_name=torch.cuda.get_device_name(0),dimensions=dimensions,
                          routing='At most one CUDA worker, only regions with at least1024 coordinates; other regional workers CPU')
            if memory <= config["gpu_memory_limit_bytes"] and timings["cuda"] < config["gpu_speed_ratio_required"]*timings["cpu"]:
                result["selected_device"] = "cuda"
            del models
            torch.cuda.empty_cache()
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc), selected_device="cpu")
    _new_json(path, result)
    return result["selected_device"]


def _state(path):
    return read_json(path) if path.exists() else dict(visits=0, dataset_identity=None,
        eligible={f: [] for f in FAMILIES}, selected={}, optimizers={}, blocks_by_width={},
        xgb_latest=None, candidate_counter=0, retired=[], failures=[])


def compact_state(state, region_root):
    keys=("visits","dataset_identity","valid_dataset_identity","selected","power4","status","training_rows","valid_rows",
          'audited_request_index_sha256','requires_current_dataset_refresh','error_type','error','label_counts',
          "train_cost_mean","last_visit_epoch","persisted_region_bytes","monotonic_conflict_audit","last_timing","fixed_role_progress")
    compact={k:state[k] for k in keys if k in state}
    compact["region_state_path"]=str(Path(region_root)/"STATE.json")
    compact["retired_candidates"]=len(state['retired']) if 'retired' in state else state.get('retired_candidates',0)
    compact["failure_count"]=len(state['failures']) if 'failures' in state else state.get('failure_count',0)
    if state.get("label_summary"):
        compact["label_counts"]=state["label_summary"].get("admitted_by_role",{})
    return compact


def regional_worker(payload):
    """Spawn-safe worker; owns only its distinct region and per-PID resources."""
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:raise
    ctx=Context(payload["config_path"],step_id="training")
    os.environ["NUMBA_CACHE_DIR"]=str(ctx.root/"cache/numba")
    with threadpool_limits(limits=1):
        state=region_slice(ctx,payload["record"],Path(payload["region_root"]),payload["settings"],payload["device"],
            payload["science_deadline"],payload["slice_seconds"])
    write_json(Path(payload['region_root'])/'SOURCE_BINDINGS.json',list(ctx.bindings.values()))
    return compact_state(state,payload["region_root"])


def _worker_payload(ctx,record,root,config,device,deadline,seconds):
    bounded=dict(config)
    if 'xgb_raw_budget_bytes' in record:
        bounded['xgb_max_projected_model_bytes']=min(config['xgb_max_projected_model_bytes'],record['xgb_raw_budget_bytes'])
    return dict(config_path=str(ctx.path),record=record,region_root=str(root),settings=bounded,
                device=device,science_deadline=deadline,slice_seconds=seconds)


def _stop_own_pool(pool):
    """Python3.12 has no public terminate_workers; terminate only owned handles."""
    processes=list(getattr(pool,'_processes',{}).values())
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:
        process.join(timeout=2.)
        if process.is_alive():process.kill();process.join(timeout=2.)
    pool.shutdown(wait=True,cancel_futures=True)


def freeze_xgb_memory_budget(ctx,regions,config):
    """Frozen per-suite budget: each national query loads one full suite."""
    path=ctx.root/'training'/'XGB_REGISTRY_MEMORY_BUDGET.json'
    totals={sid:sum(r['pairs'] for r in regions if r['suite_id']==sid) for sid in ctx.c['suite_ids']}
    if path.exists():
        record=read_json(path)
        if record['suite_pair_totals']!=totals or record['support_freeze_sha256']!=sha(ctx.root/'SUPPORT_FREEZE.json'):
            raise ValueError('Frozen XGBoost registry memory allocation changed support')
    else:
        available=int(psutil.virtual_memory().available)
        multiplier=max(3.,config['xgb_resident_multiplier_floor'])
        usable=min(ctx.c['resources']['tree_rss_bytes'],available-ctx.c['resources']['minimum_available_bytes'])
        raw=int(min(config['xgb_suite_raw_booster_cap_bytes'],
            (usable-config['xgb_graph_runtime_reserve_bytes'])/multiplier))
        if raw<=0:raise MemoryError('No positive complete-registry booster allowance after physical/runtime reserves')
        record=dict(recorded_epoch=time.time(),available_system_bytes=available,tree_limit_bytes=ctx.c['resources']['tree_rss_bytes'],
            minimum_available_bytes=ctx.c['resources']['minimum_available_bytes'],graph_runtime_reserve_bytes=config['xgb_graph_runtime_reserve_bytes'],
            resident_multiplier=multiplier,multiplier_basis='DECLARED_CONSERVATIVE_FLOOR_NOT_MEASURED_RSS',
            per_suite_raw_booster_bytes=raw,suite_pair_totals=totals,support_freeze_sha256=sha(ctx.root/'SUPPORT_FREEZE.json'),
            scope='One complete suite per national query; proportional all-coordinate allocation; no output dropping')
        _new_json(path,record)
    floor=int(config.get('xgb_region_minimum_allocation_bytes',0))
    counts={sid:sum(r['suite_id']==sid for r in regions) for sid in totals}
    if floor and any(floor*n>=record['per_suite_raw_booster_bytes'] for n in counts.values()):
        raise MemoryError('Minimum regional allocations exceed the unchanged frozen suite budget')
    for region in regions:
        sid=region['suite_id'];remaining=record['per_suite_raw_booster_bytes']-floor*counts[sid]
        region['xgb_raw_budget_bytes']=max(1,min(config['xgb_max_projected_model_bytes'],
            floor+int(remaining*region['pairs']/totals[sid])))
    if floor:
        allocation_path=ctx.root/'training/XGB_REGISTRY_ALLOCATION_V2.json'
        allocation=dict(schema='PACNHN_XGB_ALLOCATION_V2',frozen_budget_sha256=sha(path),
            support_freeze_sha256=record['support_freeze_sha256'],floor_bytes=floor,
            maximum_per_region=config['xgb_max_projected_model_bytes'],per_suite_budget=record['per_suite_raw_booster_bytes'],
            suite_region_counts=counts,suite_pair_totals=totals,
            per_suite_allocated={sid:sum(r['xgb_raw_budget_bytes'] for r in regions if r['suite_id']==sid) for sid in totals},
            unused_per_suite_bytes={sid:record['per_suite_raw_booster_bytes']-sum(r['xgb_raw_budget_bytes'] for r in regions if r['suite_id']==sid) for sid in totals},
            regions={r['key']:r['xgb_raw_budget_bytes'] for r in regions},
            policy='min(regional cap,192KiB floor + floor((frozen suite budget-N*floor)*pairs/suite_pairs)); no RAM allowance increase')
        if any(v>record['per_suite_raw_booster_bytes'] for v in allocation['per_suite_allocated'].values()):
            raise ArithmeticError('Regional allocations exceed suite cap')
        if allocation_path.exists():
            if read_json(allocation_path)!=allocation:raise ValueError('Frozen v2 XGBoost allocation changed')
        else:_new_json(allocation_path,allocation)
        record=dict(record,allocation_path=str(allocation_path),allocation_sha256=sha(allocation_path))
    return record


def concurrency_gate(ctx,root,config,deadline):
    """Synthetic distinct-directory spawn reliability and steady throughput pilot."""
    gate_path=root/"CPU_CONCURRENCY_GATE.json"
    if gate_path.exists():return read_json(gate_path)["selected_workers"]
    gate_root=root/"cpu_concurrency_gate";gate_root.mkdir(parents=True,exist_ok=True)
    graph_path=gate_root/"graph.npz"
    if not graph_path.exists():
        save_npz(graph_path,tail=np.array([0],np.int64),head=np.array([1],np.int64),node_ids=np.array(['P0','P1']),
            edge_ids=np.array(['toy']),base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0],np.int64),
            pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([1.]),baseline_i95=np.array([2.]),c0=np.array([2.]))
    bounded=dict(config,initial_train=4,fixed_valid=2,blind_test=2,train_growth=2,icnn_updates_per_block=2,
        icnn_widths=[8],xgb_rounds_per_block=2,xgb_max_depth=1,xgb_depth_candidates=[1],
        per_label_seconds=5.,region_slice_seconds=30.)
    measurements=[];reference=None
    for workers in [w for w in (1,2,4) if w<=config['regional_workers']]:
        if time.time()+20>=deadline:break
        start=time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                stages=[]
                for phase in ('warmup','measured'):
                    tick=time.perf_counter();futures=[]
                    for index in range(4):
                        record=dict(path=str(graph_path),sha256=sha(graph_path),key=f's03:{index:06d}',suite_id='s03',region_index=index)
                        destination=gate_root/f'w{workers}'/phase/f'{index:06d}'
                        futures.append(pool.submit(regional_worker,_worker_payload(ctx,record,destination,bounded,'cpu',deadline,30.)))
                    pending=set(futures)
                    while pending:
                        ctx.guard(deadline=deadline)
                        _,pending=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
                    outputs=[f.result() for f in futures]
                    if not all(all(f in s.get('selected',{}) for f in FAMILIES) for s in outputs):
                        raise RuntimeError('Spawn pilot did not produce both audited model families in every distinct region')
                    for state in outputs:
                        signature={family:load_model(state['selected'][family]['model_path']).cost(np.array([[0.],[1.],[3.]])) for family in FAMILIES}
                        if reference is None:reference=signature
                        else:
                            for family in FAMILIES:np.testing.assert_allclose(signature[family],reference[family],rtol=1e-12,atol=1e-12)
                    stages.append(dict(phase=phase,seconds=time.perf_counter()-tick))
            measurements.append(dict(workers=workers,status='PASS',stages=stages,complete_seconds=time.perf_counter()-start,
                parent_process_tree_peak_bytes=ctx.peak_rss))
        except Exception as exc:
            measurements.append(dict(workers=workers,status='FAILED',error_type=type(exc).__name__,error=str(exc)))
    passing=[m for m in measurements if m['status']=='PASS']
    if not passing:raise RuntimeError('No bounded CPU spawn configuration passed; no regional campaign authorized')
    best=min(passing,key=lambda m:(m['stages'][-1]['seconds'],m['workers']))
    _new_json(gate_path,dict(selected_workers=best['workers'],measurements=measurements,scope='SYNTHETIC_LOCAL_GRAPH_NO_TEST_OPENING',
        semantics='Warmup/import overhead and measured steady four-region batch both recorded; identical model predictions, distinct output ownership.'))
    return best['workers']


def fixed_role_progress(root,config,index=None):
    """Resolve sealed initial cohorts before enrichment; never read TEST arrays."""
    path=Path(root)/'labels'/'REQUEST_INDEX.json'
    if index is None:index=read_json(path) if path.exists() else {}
    expected={'TRAIN':config['initial_train'],'VALID':config['fixed_valid'],'TEST':config['blind_test']}
    roles={}
    for role,n in expected.items():
        # The append-only index seals all initial roles before any label. Later
        # TRAIN blocks append after these first n role members.
        rows=[r for r in index.values() if r['role']==role][:n]
        admitted=exhausted=pending=unfinished=0
        for row in rows:
            if row['selected_label'] is not None:
                admitted+=1;continue
            cohort='blind_TEST' if role=='TEST' else role
            attempts=list((Path(root)/'labels'/'labels'/cohort/row['sample_id']).glob('attempt_*'))
            unfinished+=sum(not (p/'RESULT.json').exists() and not (p/'FAILED.json').exists() for p in attempts)
            if len(attempts)>=config['max_attempts_per_input']:exhausted+=1
            else:pending+=1
        roles[role]=dict(expected=n,frozen=len(rows),admitted=admitted,exhausted=exhausted,
            pending=pending,not_yet_frozen=max(0,n-len(rows)),unfinished_attempt_directories=unfinished)
    complete=all(r['frozen']==r['expected'] and r['pending']==0 and r['admitted']+r['exhausted']==r['expected'] for r in roles.values())
    return dict(complete=complete,expected_total=sum(expected.values()),admitted_total=sum(r['admitted'] for r in roles.values()),
        exhausted_total=sum(r['exhausted'] for r in roles.values()),roles=roles,
        exhaustion_semantics='Explicit maximum attempt slots; interrupted directories without terminal receipts remain separately counted, never called admitted labels',
        TEST_arrays_opened=False)


def _requests_for_region(region, root, config, *, grow, reference_scale):
    index_path = root/"labels"/"REQUEST_INDEX.json"
    index = read_json(index_path) if index_path.exists() else {}
    requests = []
    counts = {role: sum(r["role"] == role for r in index.values()) for role in ("TRAIN", "VALID", "TEST")}
    targets = {"TRAIN": config["initial_train"], "VALID": config["fixed_valid"], "TEST": config["blind_test"]}
    if grow and fixed_role_progress(root,config,index)['complete']:
        targets["TRAIN"] = counts["TRAIN"]+config["train_growth"]
    existing = set(index)
    for role in ("TRAIN", "VALID", "TEST"):
        for row in index.values():
            if row["role"] != role:
                continue
            metadata = read_json(row["request_metadata"])
            attempts_dir = root/"labels"/"labels"/("blind_TEST" if role == "TEST" else role)/row["sample_id"]
            if row["selected_label"] is not None or len(list(attempts_dir.glob("attempt_*"))) >= config["max_attempts_per_input"]:
                continue
            with np.load(row["input_file"], allow_pickle=False) as z:
                requests.append(dict(metadata, u=z["u"].copy()))
        needed = max(0, targets[role]-counts[role])
        if needed:
            old_indices = [read_json(r["request_metadata"])["sample_index"] for r in index.values() if r["role"] == role]
            new = sample_requests(region, role=role, count=needed, seed=config["seed"],
                start_index=max(old_indices, default=-1)+1, existing_input_ids=existing, reference_scale=reference_scale)
            requests.extend(new); existing.update(r["input_id"] for r in new)
    initial_train_ids=[key for key,row in index.items() if row['role']=='TRAIN']
    initial_train_ids.extend(r['input_id'] for r in requests if r['role']=='TRAIN' and r['input_id'] not in index)
    initial_train_ids=set(initial_train_ids[:config['initial_train']])
    # Pending fixed cohorts, including retries, always precede later TRAIN.
    # Within each group new requests precede retries in stable role/index order.
    def priority(r):
        cohort = "blind_TEST" if r["role"] == "TEST" else r["role"]
        prior = len(list((root/"labels"/"labels"/cohort/r["sample_id"]).glob("attempt_*")))
        later_train=r['role']=='TRAIN' and r['input_id'] not in initial_train_ids
        return (later_train,prior > 0, ("TRAIN", "VALID", "TEST").index(r["role"]), r["sample_index"])
    return sorted(requests, key=priority)


def _evaluate(model, dataset, batch, check):
    numerator = denominator = 0.; count = 0; inference_seconds = 0.
    for start in range(0, len(dataset["u"]), batch):
        check(); u = dataset["u"][start:start+batch]; tau = dataset["tau"][start:start+batch]
        started = time.perf_counter(); prediction = model.cost(u); inference_seconds += time.perf_counter()-started
        if prediction.shape != tau.shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Invalid full-coordinate native prediction")
        numerator += float(np.abs(prediction-tau).sum()); denominator += float(tau.sum()); count += tau.size
    if not denominator > 0:
        raise ValueError("Regional cost WMAPE undefined for zero total true cost")
    return dict(WMAPE_percent=100*numerator/denominator, absolute_error_sum=numerator,
                true_cost_sum=denominator, coordinates=count, unique_inputs=len(dataset["u"]),
                batch_inference_seconds=inference_seconds, amortized_seconds_per_query=inference_seconds/len(dataset["u"]))


def native_scale(region, record, config):
    baseline=np.maximum(region.baseline_open,region.baseline_i95)
    positive=baseline[baseline>0]
    if len(positive):
        reference=float(np.median(positive)); source="MEDIAN_POSITIVE_FINAL_OPEN_I95_REGIONAL_DEMAND"
    else:
        reference=record.get("physical_capacity_reference")
        if reference is None or not math.isfinite(reference) or reference<=0 or not record.get("physical_capacity_reference_provenance"):
            raise ValueError("All-zero baseline requires a documented actual physical_capacity_reference")
        source="RECORDED_PHYSICAL_CAPACITY_REFERENCE"
    floor=config["native_scale_positive_median_fraction"]*reference
    scale=np.maximum(baseline,floor)
    return scale,dict(policy="MAX_OPEN_I95_AND_0.01_POSITIVE_MEDIAN_OR_PHYSICAL_CAPACITY",units="native demand/flow units",
        reference=reference,reference_source=source,floor=floor,fraction=config["native_scale_positive_median_fraction"],
        physical_capacity_reference_provenance=record.get("physical_capacity_reference_provenance"),graph_sha256=region.graph_sha256)


def monotonic_conflicts(train, scale, pair_keys):
    """Valid L1 lower bound using disjoint ordered query pairs; not exhaustive."""
    u,tau=train["u"],train["tau"]
    order=np.argsort(np.sum(u/scale,axis=1),kind="stable")
    violations=[]; lower=0.; comparisons=0
    for at in range(0,len(order)-1,2):
        i,j=int(order[at]),int(order[at+1])
        if not np.all(u[i]<=u[j]):continue
        comparisons+=1; differences=tau[i]-tau[j]; bad=np.flatnonzero(differences>0.)
        if len(bad):
            lower+=float(differences[bad].sum())
            violations.append(dict(lower_row=i,upper_row=j,pair_keys=[str(pair_keys[k]) for k in bad],
                native_cost_decrease=differences[bad],lower_input_id=train["records"][i]["input_identity"],
                upper_input_id=train["records"][j]["input_identity"]))
    denom=float(tau.sum())
    return dict(train_dataset_identity=train["dataset_identity"],screen="DISJOINT_ADJACENT_SORTED_QUERY_PAIRS_NOT_EXHAUSTIVE",
        unique_rows=len(u),ordered_comparisons=comparisons,violating_disjoint_pairs=len(violations),
        fixed_label_L1_error_lower_bound=lower,fixed_label_WMAPE_percent_lower_bound=100*lower/denom if denom>0 else None,
        denominator_native_cost_sum=denom,violations=violations,
        scope="Lower bound for fitting these recorded targets; no coordinatewise true-equilibrium label-error certificate")


def _retain_pool(candidates, selected_id, cap):
    """Deterministic bounded eligibility; this never deletes historical artifacts."""
    required = [selected_id, candidates[-1]["candidate_id"]] if candidates else []
    for width in sorted({r.get("width", 0) for r in candidates}):
        matching = [r for r in candidates if r.get("width", 0) == width]
        required.append(matching[-1]["candidate_id"])
    ordered = required+[r["candidate_id"] for r in sorted(candidates,
        key=lambda r: (r["audit"]["VALID"]["WMAPE_percent"], r["candidate_id"]))]
    keep = list(dict.fromkeys(ordered))[:cap]
    return [r for r in candidates if r["candidate_id"] in keep], [r for r in candidates if r["candidate_id"] not in keep]


def audit_and_select(root, family, state, train, valid, config, check):
    started = time.perf_counter(); audited = []
    for candidate in state["eligible"][family]:
        hash_started=time.perf_counter()
        if sha(candidate["model_path"]) != candidate["model_sha256"]:
            raise ValueError("Eligible immutable model artifact changed")
        hash_seconds=time.perf_counter()-hash_started
        if (candidate.get("audit", {}).get("train_dataset_identity") == train["dataset_identity"]
                and candidate["audit"].get("valid_dataset_identity") == valid["dataset_identity"]):
            audited.append(candidate); continue
        audit_path = root/"audits"/family/(candidate["candidate_id"]+"_"+train["dataset_identity"]+"_"+valid["dataset_identity"]+".json")
        if audit_path.exists():
            audit=read_json(audit_path)
            if (audit["model_sha256"]!=candidate["model_sha256"] or audit["train_dataset_identity"]!=train["dataset_identity"]
                    or audit["valid_dataset_identity"]!=valid["dataset_identity"]):
                raise ValueError("Preserved audit identity mismatch")
            candidate["audit"]=audit;candidate["audit_path"]=str(audit_path);audited.append(candidate);continue
        load_started=time.perf_counter();model = load_model(candidate["model_path"])
        load_seconds=time.perf_counter()-load_started
        audit = dict(train_dataset_identity=train["dataset_identity"], valid_dataset_identity=valid["dataset_identity"],
            TRAIN=_evaluate(model, train, config["audit_batch_rows"], check),
            VALID=_evaluate(model, valid, config["audit_batch_rows"], check), recorded_epoch=time.time(),model_sha256=candidate["model_sha256"],
            model_hash_seconds=hash_seconds,model_load_seconds=load_seconds)
        audit["target_met"] = audit["TRAIN"]["WMAPE_percent"] < config["train_target_wmape_percent"]
        if family=="ICNN":
            audit['neural_branch'],audit['neural_branch_inference_seconds']=_neural_branch_audit(
                model,train,config['audit_batch_rows'],check)
        _new_json(audit_path, audit)
        candidate["audit"] = audit; candidate["audit_path"] = str(audit_path)
        audited.append(candidate)
    if not audited:
        return 0.
    qualified = [r for r in audited if r["audit"]["target_met"]]
    selected = min(qualified or audited, key=lambda r: (r["audit"]["VALID"]["WMAPE_percent"], r["candidate_id"]))
    state["selected"][family] = copy.deepcopy(selected)
    retained, retired = _retain_pool(audited, selected["candidate_id"], config["eligible_candidate_cap_per_family"])
    state["eligible"][family] = retained
    for row in retired:
        retirement = dict(family=family, candidate_id=row["candidate_id"], model_path=row["model_path"],
            retired_epoch=time.time(), reason="PREDECLARED_FINITE_ELIGIBILITY_POOL", preserved=True)
        state["retired"].append(retirement)
    return time.perf_counter()-started


def _candidate(root, state, family, model, *, width=0, **extra):
    if shutil.disk_usage(root).free<5*1024**3:
        raise OSError("Checkpoint refused by output free-space guard")
    # A process interruption can leave immutable output ahead of STATE.json.
    # Preserve it and advance the counter; never overwrite or silently select it.
    while True:
        state["candidate_counter"] += 1
        identity = f'{family}_{state["candidate_counter"]:08d}'
        directory = root/"checkpoints"/identity
        if not directory.exists():break
        state.setdefault('interrupted_checkpoint_artifacts',[]).append(str(directory))
    directory.mkdir(parents=True, exist_ok=False)
    receipt = save_model(model, directory/"model.npz")
    row = dict(candidate_id=identity, family=family, width=width, model_path=receipt["path"],
        model_sha256=receipt["sha256"], model_kind=receipt["kind"], created_epoch=time.time(), **extra)
    state["eligible"][family].append(row)
    return row, directory


def _initialize_icnn(train,scale,config,device,width):
    eta=fit_power4_cost(train['u'],train['tau'],train['c0'],scale).eta
    model=MonotoneICNN(train['c0'],scale,eta=eta,width=width,seed=config['seed'],device=device,
        amplitude=max(1.,float(train['tau'].mean())*float(scale.mean())))
    if config.get('icnn_coefficient_parameterization',ICNN_NATIVE)==ICNN_SCALED:
        calibrate_icnn_train(model,train['u'],train['tau'],
            delta=config['icnn_initial_neural_cost_share'],train_dataset_identity=train['dataset_identity'])
    return model


def _icnn_block(root, state, train, scale, config, device, stop_epoch, check):
    blocks = sum(state["blocks_by_width"].values())
    width_index = min(blocks//config["icnn_blocks_before_next_width"], len(config["icnn_widths"])-1)
    width = config["icnn_widths"][width_index]; key = str(width)
    parameterization=config.get('icnn_coefficient_parameterization',ICNN_NATIVE)
    previous=state['optimizers'].get(key)
    # A resumed candidate restores its original units; current TRAIN only changes
    # the loss denominator and audits, never that candidate's parameterization.
    model=(MonotoneICNN(train['c0'],scale,width=width,seed=config['seed'],device=device)
           if previous else _initialize_icnn(train,scale,config,device,width))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["icnn_learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"]+width)
    total_updates = 0
    if previous:
        if sha(previous["path"]) != previous["sha256"]:
            raise ValueError("Resumable ICNN optimizer checkpoint changed")
        saved = torch.load(previous["path"], map_location="cpu", weights_only=True)
        if (saved.get('schema')!=OPTIMIZER_SCHEMA or saved.get('coefficient_parameterization')!=parameterization):
            raise ValueError('Optimizer schema/parameterization mismatch; implicit legacy migration prohibited')
        if (saved["width"] != width or not np.array_equal(saved["scale"].numpy(), scale)
                or not np.array_equal(saved['model']['c0'].numpy(),train['c0'])):
            raise ValueError("Optimizer native scaling/architecture changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        model.coefficient_parameterization=parameterization;model.calibration=saved.get('TRAIN_calibration')
        generator.set_state(saved["generator"]); total_updates = saved["total_updates"]
    updates = 0; started = time.perf_counter(); last = None
    for _ in range(config["icnn_updates_per_block"]):
        if time.time() >= stop_epoch:
            break
        check()
        indices = torch.randint(len(train["u"]), (min(config["icnn_batch_rows"], len(train["u"])),), generator=generator).numpy()
        last = update_icnn(model, optimizer, train["u"][indices], train["tau"][indices], train_cost_mean=train["train_cost_mean"])
        updates += 1
    if not updates:
        return dict(seconds=time.perf_counter()-started, updates=0)
    candidate, directory = _candidate(root, state, "ICNN", model, width=width,
        updates_in_block=updates, total_updates=total_updates+updates, device=device,
        trained_dataset_identity=train["dataset_identity"], last_batch=last,training_block_index=blocks+1,
        coefficient_parameterization=parameterization,TRAIN_calibration=model.calibration)
    state_path = directory/"optimizer.pt"
    _atomic_torch_save(state_path, dict(schema=OPTIMIZER_SCHEMA,coefficient_parameterization=parameterization,
        TRAIN_calibration=model.calibration,model=model.state_dict(), optimizer=optimizer.state_dict(),
        generator=generator.get_state(), width=width, scale=torch.tensor(scale), total_updates=total_updates+updates))
    state["optimizers"][key] = dict(path=str(state_path), sha256=sha(state_path), candidate_id=candidate["candidate_id"])
    state["blocks_by_width"][key] = state["blocks_by_width"].get(key, 0)+1
    _new_json(directory/"CHECKPOINT.json", dict(candidate, optimizer=state["optimizers"][key]))
    return dict(seconds=time.perf_counter()-started, updates=updates)


def _structural_zeros(region):
    """Proof: a directed route consisting only of identically zero-cost arcs."""
    zero_edges = (region.base == 0) & (region.k4 == 0)
    if not zero_edges.any():
        return region.pair_source == region.pair_target
    od, order = region.demand(np.zeros(region.m))
    result = aon(region.network, od, np.zeros(len(region.base)), enabled=zero_edges, workers=1, hash_capacity=16)
    zero = np.empty(region.m, bool); zero[order] = np.isfinite(result["distances"])
    return zero


def _xgb_block(root, state, train, scale, region, config, device, stop_epoch, check):
    previous = state["xgb_latest"]
    if previous and sha(previous["model_path"]) != previous["model_sha256"]:
        raise ValueError("Resumable XGBoost checkpoint changed")
    initial = load_model(previous["model_path"]) if previous else None
    block_rounds=min(config['xgb_rounds_per_block'],state.get('xgb_resource_block_rounds',config['xgb_rounds_per_block']))
    restart_reason=None;depth=config.get('xgb_depth_candidates',[config['xgb_max_depth']])[0]
    if initial is not None:
        depth=previous.get('max_depth',depth)
        rounds=int(initial.metadata["boosting_rounds_total"])
        prior_bytes=previous.get('uncompressed_model_bytes',previous.get('uncompressed_booster_bytes',Path(previous['model_path']).stat().st_size))
        if config.get('xgb_adaptive_quota_blocks',False):
            quota_rounds=max(0,math.floor(rounds*config['xgb_max_projected_model_bytes']/(1.5*max(1,prior_bytes)))-rounds)
            block_rounds=min(block_rounds,max(0,config['xgb_max_rounds_per_candidate']-rounds),quota_rounds)
        projected=prior_bytes*(rounds+block_rounds)/max(1,rounds)*1.5
        if block_rounds==0 or rounds+block_rounds>config["xgb_max_rounds_per_candidate"] or projected>config["xgb_max_projected_model_bytes"]:
            ladder=config.get('xgb_depth_candidates',[depth]);next_depth=ladder[min(ladder.index(depth)+1,len(ladder)-1)]
            requested=min(config['xgb_rounds_per_block'],state.get('xgb_resource_block_rounds',config['xgb_rounds_per_block']))
            proposed_bytes=prior_bytes*requested/max(1,rounds)*2**(next_depth-depth)*1.5
            next_depth=next_depth if proposed_bytes<=config['xgb_max_projected_model_bytes'] else depth
            if (config.get('xgb_adaptive_quota_blocks',False) and next_depth==depth
                    and previous.get('trained_dataset_identity')==train['dataset_identity']):
                return dict(seconds=0.,rounds=0,status='QUOTA_OR_COMPLEXITY_BOUND_PRESERVED_CURRENT_MODEL',
                    raw_model_bytes=prior_bytes,budget_bytes=config['xgb_max_projected_model_bytes'],
                    reason='No admissible continuation or new depth; identical same-TRAIN restart suppressed')
            initial=None;restart_reason="PREDECLARED_COMPLEXITY_LIMIT_RETRAIN_ON_CURRENT_TRAIN"
            depth=next_depth;block_rounds=requested
    started = time.perf_counter(); zeros = _structural_zeros(region)
    model = fit_xgboost_cost(train["u"], train["tau"], train["c0"], scale,
        n_estimators=block_rounds, max_depth=depth,
        learning_rate=config["xgb_learning_rate"], seed=config["seed"], device="cpu", n_jobs=1,
        structural_zero_mask=zeros, structural_zero_evidence="ZERO_BASE_ZERO_K4_DIRECTED_PATH:"+region.identity if zeros.any() else None,
        initial_model=initial,deadline_epoch=stop_epoch,guard=check,
        log_center_policy=config.get('xgb_log_center_policy',XGB_UNCENTERED),train_dataset_identity=train['dataset_identity'])
    rounds_before=0 if initial is None else int(initial.metadata["boosting_rounds_total"])
    if model.metadata["boosting_rounds_total"]<=rounds_before and not np.all(zeros):
        return dict(seconds=time.perf_counter()-started,rounds=0,status="TIME_LIMIT_BEFORE_NEW_TREE")
    raw_bytes=0 if model.booster is None else len(model.booster.save_raw(raw_format="ubj"))
    parameter_bytes=sum(a.nbytes for a in (model.c0,model.scale,model.log_center,model.structural_zero_mask))
    model_bytes=raw_bytes+parameter_bytes
    if model_bytes>config["xgb_max_projected_model_bytes"]:
        candidate,directory=_candidate(root,state,'XGBOOST',model,trained_dataset_identity=train['dataset_identity'],
            disposition='RESOURCE_REJECTED_NOT_ELIGIBLE',uncompressed_booster_bytes=raw_bytes,
            uncompressed_model_bytes=model_bytes,native_parameter_bytes=parameter_bytes,
            raw_booster_budget_bytes=config['xgb_max_projected_model_bytes'],max_depth=depth)
        state['eligible']['XGBOOST'].remove(candidate)
        reduced=max(1,int(block_rounds*config['xgb_max_projected_model_bytes']/model_bytes*.8))
        state['xgb_resource_block_rounds']=min(block_rounds,reduced)
        _new_json(directory/'CHECKPOINT.json',candidate)
        raise MemoryError('Candidate preserved but excluded: raw booster exceeds proportional registry budget; subsequent blocks reduced to '+str(reduced))
    tree_count=leaf_count=0
    if model.booster is not None:
        def count_nodes(node):
            if "leaf" in node:return 1
            return sum(count_nodes(child) for child in node["children"])
        for tree in model.booster.get_dump(dump_format="json"):
            tree_count+=1;leaf_count+=count_nodes(json.loads(tree))
    candidate, directory = _candidate(root, state, "XGBOOST", model,
        trained_dataset_identity=train["dataset_identity"], boosting_rounds_total=model.metadata["boosting_rounds_total"],
        restart_reason=restart_reason,uncompressed_booster_bytes=raw_bytes,tree_count=tree_count,leaf_count=leaf_count,max_depth=depth,
        uncompressed_model_bytes=model_bytes,native_parameter_bytes=parameter_bytes,
        log_center_policy=model.metadata['log_center_policy'],log_center_sha256=model.metadata['log_center_sha256'],
        log_center_train_dataset_identity=model.metadata['log_center_train_dataset_identity'],
        raw_booster_budget_bytes=config['xgb_max_projected_model_bytes'],
        training_block_index=state.get('xgb_blocks',0)+1)
    state['xgb_blocks']=state.get('xgb_blocks',0)+1
    state["xgb_latest"] = copy.deepcopy(candidate)
    _new_json(directory/"CHECKPOINT.json", candidate)
    return dict(seconds=time.perf_counter()-started, rounds=model.metadata['boosting_rounds_total']-rounds_before, device="cpu",
                GPU_policy="Torch GPU pilot does not authorize a different XGBoost device")


def region_slice(ctx, record, root, config, device, science_deadline, slice_seconds):
    complete_started=time.time();input_started=time.perf_counter()
    root.mkdir(parents=True, exist_ok=True); state_path = root/"STATE.json"; state = _state(state_path)
    completed_visits=[int(p.stem.split('_')[-1]) for p in (root/'timing').glob('visit_*.json')]
    state['visits']=max([state['visits']]+completed_visits)
    if state.get("status")=="BLOCKED_REGIONAL_INVARIANT":
        return state
    ctx.bind(record["path"], "CURRENT_DETAILED_REGIONAL_LABEL_GRAPH", record["sha256"])
    region = load_region_graph(record["path"], expected_sha256=record["sha256"])
    started = complete_started; slice_end = min(started+slice_seconds, science_deadline)
    def check():
        ctx.guard(deadline=science_deadline)
        if time.time() >= science_deadline:
            raise TimeoutError("Training scientific deadline reached")
    timing = dict(input_load=time.perf_counter()-input_started,graph_file_bytes=Path(record['path']).stat().st_size,
        labels=0., training_icnn=0., training_xgboost=0., power4=0., evaluation=0.,dataset_load=0.)
    state["visits"] += 1; state["last_visit_epoch"] = started
    try:
        check()
        scale_path = root/"NATIVE_SCALE.json"
        if scale_path.exists():
            scale_record=read_json(scale_path)
            if scale_record["graph_sha256"]!=region.graph_sha256:raise ValueError("Frozen scale graph changed")
            scale = np.asarray(scale_record["scale"], np.float64)
        else:
            scale,scale_record=native_scale(region,record,config)
            _new_json(scale_path,dict(scale=scale,**scale_record))
        current_index = root/"labels"/"REQUEST_INDEX.json"
        entries = read_json(current_index) if current_index.exists() else {}
        fixed=fixed_role_progress(root,config,entries)
        # Growth must not starve sealed VALID/TEST behind fresh TRAIN blocks.
        grow = fixed['complete'] and all(f in state["selected"] for f in FAMILIES)
        requests = _requests_for_region(region, root, config, grow=grow,reference_scale=scale)
        if requests:
            t = time.perf_counter()
            label_deadline = min(slice_end, started+slice_seconds*config["label_fraction_per_slice"])
            label_config = dict(phase_id="PACNHN_V1", run_id=ctx.c["run_id"], region_id=record["key"],
                graph_path=record["path"], graph_sha256=record["sha256"], deadline_epoch=label_deadline,
                per_label_seconds=config["per_label_seconds"],
                relative_gap_tolerance=config["label_target_relative_gap"], **label_solver_settings(config))
            generation = generate_labels(label_config, root/"labels", requests=requests, guard=check)
            timing["labels"] += time.perf_counter()-t
            state["label_summary"] = generation
            if generation["status"] == "BLOCKED_REGIONAL_INVARIANT":
                state["status"] = "BLOCKED_REGIONAL_INVARIANT"; return state
        data_started=time.perf_counter();train = load_admitted_dataset(root/"labels", role="TRAIN")
        valid = load_admitted_dataset(root/"labels", role="VALID")
        timing['dataset_load']=time.perf_counter()-data_started
        state["dataset_identity"] = train["dataset_identity"]
        state["valid_dataset_identity"] = valid["dataset_identity"]
        state['audited_request_index_sha256']=sha(root/'labels/REQUEST_INDEX.json')
        state['requires_current_dataset_refresh']=False
        state["training_rows"] = train["unique_rows"]; state["valid_rows"] = valid["unique_rows"]
        state["train_cost_mean"] = train["train_cost_mean"]
        conflict_path=root/"monotonic_conflicts"/(train["dataset_identity"]+".json")
        if not conflict_path.exists():_new_json(conflict_path,monotonic_conflicts(train,scale,region.pair_keys))
        state["monotonic_conflict_audit"]=str(conflict_path)
        t = time.perf_counter()
        power = fit_power4_cost(train["u"], train["tau"], train["c0"], scale)
        ppath = root/"power4"/(train["dataset_identity"]+".npz")
        if not ppath.exists():
            save_model(power, ppath, metadata={"trained_dataset_identity":train["dataset_identity"]})
        state["power4"] = dict(model_path=str(ppath), model_sha256=sha(ppath),
            model_kind=power.kind, train_dataset_identity=train["dataset_identity"],
            valid_dataset_identity=valid["dataset_identity"],
            TRAIN=_evaluate(power,train,config["audit_batch_rows"],check), VALID=_evaluate(power,valid,config["audit_batch_rows"],check))
        timing["power4"] += time.perf_counter()-t
        # Each family gets one resumable block per visit; neither has a terminal cap.
        for family in FAMILIES:
            try:
                check()
                if family == "ICNN":
                    budget_end = min(slice_end, time.time()+max(1., (slice_end-time.time())/2))
                    block = _icnn_block(root,state,train,scale,config,device,budget_end,check)
                    timing["training_icnn"] += block["seconds"]
                elif time.time() < slice_end:
                    block = _xgb_block(root,state,train,scale,region,config,device,min(slice_end,science_deadline),check)
                    timing["training_xgboost"] += block["seconds"]
                timing["evaluation"] += audit_and_select(root,family,state,train,valid,config,check)
                write_json(state_path,state)
            except TimeoutError:
                raise
            except Exception as exc:
                failure = dict(family=family, visit=state["visits"], epoch=time.time(), error_type=type(exc).__name__, error=str(exc))
                state["failures"].append(failure)
                _new_json(root/"failures"/f'{state["visits"]:06d}_{family}.json',dict(failure,traceback=traceback.format_exc()))
        state["status"] = "TRAINING_CONTINUES"
    except Exception as exc:
        state["status"] = "TIME_LIMIT" if isinstance(exc,TimeoutError) else "INCOMPLETE_REGION_SLICE"
        failure = dict(epoch=time.time(),visit=state["visits"],error_type=type(exc).__name__,error=str(exc))
        state["failures"].append(failure)
        _new_json(root/"failures"/f'{state["visits"]:06d}_SLICE.json',dict(failure,traceback=traceback.format_exc()))
    finally:
        state['fixed_role_progress']=fixed_role_progress(root,config)
        write_json(root/'FIXED_ROLE_PROGRESS.json',state['fixed_role_progress'])
        timing["complete_slice_seconds"] = time.time()-started
        state['last_timing']=timing
        _new_json(root/"timing"/f'visit_{state["visits"]:06d}.json',timing)
        state["persisted_region_bytes"]=_tree_bytes(root)
        write_json(state_path,state)
    return state


def qualified(state):
    return bool(state and not state.get('requires_current_dataset_refresh',False) and state.get("dataset_identity") and all(f in state.get("selected", {})
        and state["selected"][f].get("audit", {}).get("train_dataset_identity") == state["dataset_identity"]
        and state["selected"][f]["audit"].get("valid_dataset_identity") == state.get("valid_dataset_identity")
        and state["selected"][f]["audit"]["target_met"] for f in FAMILIES))


def pilot_checkpoint_ready(state):
    """Pilot completion needs both current-data fits, independently of accuracy."""
    if (not state or not state.get('dataset_identity') or not state.get('valid_dataset_identity')
            or state.get('status')=='BLOCKED_REGIONAL_INVARIANT' or not state.get('fixed_role_progress',{}).get('complete')):
        return False
    for family in FAMILIES:
        candidate=state.get('selected',{}).get(family,{})
        audit=candidate.get('audit',{})
        if (not candidate.get('model_path') or not candidate.get('model_sha256')
                or audit.get('train_dataset_identity')!=state['dataset_identity']
                or audit.get('valid_dataset_identity')!=state['valid_dataset_identity']):
            return False
        for role in ('TRAIN','VALID'):
            metric=audit.get(role,{})
            value=metric.get('WMAPE_percent')
            if metric.get('unique_inputs',0)<1 or value is None or not math.isfinite(value):
                return False
    return True


def schedule_pilot_region(regions,states,attempts,*,exclude_keys=(),now,deadline):
    """Fair bounded revisits; a returned slice is not a completed pilot region."""
    if now>=deadline:return None
    excluded=set(exclude_keys)
    remaining=[r for r in regions if r['key'] not in excluded
        and not pilot_checkpoint_ready(states.get(r['key']))
        and states.get(r['key'],{}).get('status')!='BLOCKED_REGIONAL_INVARIANT']
    if not remaining:return None
    order={r['key']:i for i,r in enumerate(regions)}
    return min(remaining,key=lambda r:(attempts.get(r['key'],0),order[r['key']]))


def pilot_progress(regions,states,attempts):
    keys=[r['key'] for r in regions]
    attempted=[key for key in keys if attempts.get(key,0)>0]
    completed=[key for key in keys if pilot_checkpoint_ready(states.get(key))]
    return dict(regions_required=len(keys),regions_attempted=len(attempted),regions_completed=len(completed),
        required_region_keys=keys,attempted_region_keys=attempted,completed_region_keys=completed,
        incomplete_region_keys=[key for key in keys if key not in completed],slice_attempts=dict(attempts),
        fixed_role_progress={key:states.get(key,{}).get('fixed_role_progress') for key in keys},
        usable_counts={key:dict(TRAIN=states.get(key,{}).get('training_rows',0),VALID=states.get(key,{}).get('valid_rows',0),
            TEST=states.get(key,{}).get('fixed_role_progress',{}).get('roles',{}).get('TEST',{}).get('admitted',0)) for key in keys},
        all_required_complete=bool(keys) and len(completed)==len(keys))


def schedule_region(regions, states, scheduler, config, *, exclude_keys=()):
    excluded=set(exclude_keys)
    primary = [r for r in regions if r["suite_id"] == "s03"]
    if not primary:raise ValueError('No active s03 regions in scheduling inventory')
    def available(r):
        return r['key'] not in excluded and states.get(r['key'],{}).get('status')!='BLOCKED_REGIONAL_INVARIANT'
    def unresolved(r):
        state=states.get(r['key'],{})
        return state.get('requires_current_dataset_refresh',False) or not state.get('fixed_role_progress',{}).get('complete',False)
    # Apply active/blocked exclusions before choosing a priority tier. A pending
    # fixed-cohort region already in flight must not idle other usable workers.
    usable_primary=[r for r in primary if available(r)]
    unvisited = [r for r in usable_primary if not states.get(r["key"], {}).get("visits")]
    if unvisited:
        return unvisited[0]
    if not scheduler.get("first_all_s03_qualified"):
        all_ready=all(not unresolved(r) and qualified(states.get(r['key']))
            and states.get(r['key'],{}).get('status')!='BLOCKED_REGIONAL_INVARIANT' for r in primary)
        if all_ready:
            # Drain outstanding regional mutations before recording an all-current
            # dataset milestone. Their returned audits may reveal new shortfalls.
            if any(r['key'] in excluded for r in primary):return None
            scheduler["first_all_s03_qualified"] = time.time()
    if not scheduler.get("first_all_s03_qualified"):
        pool = ([r for r in usable_primary if unresolved(r)]
                or [r for r in usable_primary if not qualified(states.get(r["key"]))]
                or usable_primary)
    elif scheduler.get("postqualification_slices",0)%4 == 0:
        pool = usable_primary
    else:
        pool = []
        for sid in config["additional_suite_order"]:
            candidates = [r for r in regions if r["suite_id"] == sid and available(r)]
            incomplete = ([r for r in candidates if unresolved(r)]
                or [r for r in candidates if not qualified(states.get(r["key"]))])
            if incomplete:
                pool = incomplete; break
        if not pool:
            pool = [r for r in regions if r["suite_id"] != "s03" and available(r)]
    if not pool:
        if excluded:return None
        pool=usable_primary
    if not pool:raise RuntimeError("All eligible primary regions are blocked by invariant failures")
    if scheduler.get('first_all_s03_qualified'):
        scheduler['postqualification_slices']=scheduler.get('postqualification_slices',0)+1
    return min(pool,key=lambda r:(states.get(r["key"],{}).get("last_visit_epoch",0.),r["key"]))


def finalization_forecast(regions,states,config):
    load=[s.get('last_timing',{}).get('input_load',0.) for s in states.values()]
    load=[x for x in load if x>0]
    rates=[s['last_timing']['input_load']/s['last_timing']['graph_file_bytes'] for s in states.values()
        if s.get('last_timing',{}).get('graph_file_bytes',0)>0]
    graph_seconds=(float(np.median(load))*len(regions)) if load else 0.
    if rates:graph_seconds=max(graph_seconds,float(np.percentile(rates,90))*sum(r.get('graph_file_bytes',0) for r in regions))
    label_io=predictions=model_io=0.;test_rows=0
    reassess_prediction=reassess_branch=reassess_model_io=reassess_data_io=0.
    for state in states.values():
        counts=state.get('label_counts',{})
        n=counts.get('TEST',0);test_rows+=n
        source_rows=max(1,state.get('training_rows',0)+state.get('valid_rows',0))
        data_seconds=state.get('last_timing',{}).get('dataset_load',0.)
        label_io+=data_seconds*n/source_rows
        sources=dict(state.get('selected',{}))
        if state.get('power4'):sources['POWER4']=state['power4']
        if sources:reassess_data_io+=data_seconds+state.get('last_timing',{}).get('input_load',0.)
        peer_model_io=max((r.get('audit',{}).get('model_load_seconds',0.)+2*r.get('audit',{}).get('model_hash_seconds',0.)
            for r in state.get('selected',{}).values()),default=0.)
        for family,candidate in sources.items():
            audit=candidate if family=='POWER4' else candidate.get('audit',{})
            predictions+=audit.get('TRAIN',{}).get('amortized_seconds_per_query',0.)*n
            measured_io=audit.get('model_load_seconds',0.)+2*audit.get('model_hash_seconds',0.)
            if family=='POWER4' and measured_io==0.:
                # Older power4 records have no load timer. Its measured entire
                # fit/save/evaluate block (or the peer model I/O maximum) is an
                # explicit conservative proxy, not a fabricated measured load.
                measured_io=max(peer_model_io,state.get('last_timing',{}).get('power4',0.))
            model_io+=measured_io;reassess_model_io+=measured_io
            for role,state_key in [('TRAIN','training_rows'),('VALID','valid_rows')]:
                prior=audit.get(role,{})
                rows=max(counts.get(role,0),state.get(state_key,0),prior.get('unique_inputs',0))
                reassess_prediction+=prior.get('amortized_seconds_per_query',0.)*rows
            if family=='ICNN':
                prior=audit.get('TRAIN',{});rows=max(counts.get('TRAIN',0),state.get('training_rows',0),prior.get('unique_inputs',0))
                # Legacy audits lack a branch timer: use the measured complete
                # ICNN TRAIN prediction pass as an explicit proxy, not zero.
                branch=audit.get('neural_branch_inference_seconds',prior.get('batch_inference_seconds',0.))
                reassess_branch+=branch*rows/max(1,prior.get('unique_inputs',state.get('training_rows',1)))
    current_data_authentication=2*sum(s.get('last_timing',{}).get('dataset_load',0.) for s in states.values())
    base=graph_seconds+label_io+predictions+model_io+current_data_authentication
    protected=max(config['finalization_minimum_reserve_seconds'],1.5*base+config['region_slice_seconds'])
    if not load:protected=max(protected,config['final_evaluation_reserve_seconds'])
    reassessment=1.5*(reassess_data_io+reassess_prediction+reassess_branch+reassess_model_io)
    return dict(reserve_seconds=protected+reassessment,protected_freeze_and_TEST_seconds=protected,
        selected_reassessment_allowance_seconds=reassessment,
        projected_reassessment_data_and_graph_load_seconds=reassess_data_io,
        projected_reassessment_TRAIN_VALID_prediction_seconds=reassess_prediction,
        projected_reassessment_ICNN_branch_seconds=reassess_branch,
        projected_reassessment_model_load_and_hash_seconds=reassess_model_io,
        reassessment_scope='All existing selected checkpoints as conservative upper coverage; stale-only execution; missing ICNN branch timer uses full TRAIN prediction; missing power4 I/O timer uses maximum of measured power4 fit/save/evaluate block and peer model I/O; 50% explicit allowance margin',
        projected_graph_input_seconds=graph_seconds,projected_test_IO_seconds=label_io,
        projected_current_TRAIN_VALID_authentication_seconds=current_data_authentication,
        projected_selected_prediction_seconds=predictions,projected_model_load_and_hash_seconds=model_io,admitted_TEST_rows=test_rows,
        basis='Measured regional loading and batch inference; explicit additional selected-only reassessment allowance precedes protected freeze/TEST reserve.',
        nominal_initial_reserve_seconds=config['final_evaluation_reserve_seconds'],measured_regions=len(load))


def _neural_branch_audit(model,train,batch,check):
    started=time.perf_counter();total=0.;maximum=0.;nonzero=0
    for start in range(0,len(train['u']),batch):
        check();u=train['u'][start:start+batch]
        delta=model.amplitude*model.unit_neural_cost(u)
        if delta.shape!=u.shape or not np.isfinite(delta).all() or np.any(delta<0):
            raise ValueError('Invalid selected ICNN native neural increment')
        total+=float(np.abs(delta).sum());maximum=max(maximum,float(np.max(delta,initial=0.)))
        nonzero+=int(np.count_nonzero(delta));check()
    return dict(amplitude=model.amplitude,structurally_active=model.amplitude>0,
        train_dataset_identity=train['dataset_identity'],observed_increment_absolute_sum=total,
        maximum_native_cost_increment=maximum,nonzero_prediction_coordinates=nonzero,
        interpretation='POWER4_PLUS_MONOTONE_ICNN' if model.amplitude>0 else 'POWER4_WITH_INACTIVE_ICNN_BRANCH'),time.perf_counter()-started


def _selected_snapshot(states):
    selected={}
    for key,state in states.items():
        sources={f:copy.deepcopy(v) for f,v in state.get('selected',{}).items() if f in FAMILIES}
        if state.get('power4'):sources['POWER4']=copy.deepcopy(state['power4'])
        if sources:selected[key]=sources
    return selected


def _reuse_selected_reassessment(ctx,receipt_dir,intent_sha,source,record,label_root):
    """Authenticate a completed receipt by bytes only; no inference after cutoff."""
    for path in sorted(receipt_dir.glob('*.json')):
        ctx.guard()
        saved=read_json(path)
        if not (saved.get('status')=='PASS' and saved.get('selection_intent_sha256')==intent_sha
                and saved.get('candidate_id')==source.get('candidate_id')
                and saved.get('model_path')==source['model_path'] and saved.get('model_sha256')==source['model_sha256']
                and saved.get('graph_sha256')==record['sha256']):continue
        population=saved.get('current_population',{})
        index=label_root/'REQUEST_INDEX.json';identity=label_root/'REGION_IDENTITY.json'
        if ((sha(index) if index.exists() else None)!=population.get('request_index_sha256')
                or (sha(identity) if identity.exists() else None)!=population.get('region_identity_sha256')):continue
        if sha(source['model_path'])!=source['model_sha256'] or sha(record['path'])!=record['sha256']:
            raise ValueError('Completed selected reassessment model/graph bytes changed')
        for artifact,digest in population.get('source_files',{}).items():
            ctx.guard()
            if not Path(artifact).exists() or sha(artifact)!=digest:
                raise ValueError('Completed selected reassessment admitted source changed')
        audit=saved['audit']
        if (audit['model_sha256']!=source['model_sha256']
                or audit['train_dataset_identity']!=population['TRAIN']['dataset_identity']
                or audit['valid_dataset_identity']!=population['VALID']['dataset_identity']):
            raise ValueError('Completed selected reassessment receipt population mismatch')
        return path,saved
    return None


def _reassessed_source(source,family,receipt,path):
    replacement=copy.deepcopy(source)
    if receipt['status']=='PASS':
        if family=='POWER4':
            replacement.update(receipt['audit'])
            replacement['fit_train_dataset_identity']=receipt['fit_train_dataset_identity']
        else:replacement['audit']=receipt['audit']
    replacement.update(final_reassessment_status=receipt['status'],
        reassessment_receipt_path=str(path),reassessment_receipt_sha256=sha(path))
    return replacement


def reassess_selected_before_freeze(ctx,root,regions,states,support_sha,config,deadline):
    """Reassess exact prior selections after worker shutdown; never fit or select.

    The immutable selection journal also makes interruption resume finalization
    only. Original regional STATE/checkpoints/audits and all admitted labels stay
    untouched; only a copied state feeds the later registry freeze.
    """
    root=Path(root)
    if (root/'REGISTRY_FREEZE.json').exists() or (root/'TEST_OPENING_AUTHORIZATION.json').exists():
        raise ValueError('Selected reassessment requires unfrozen selection and sealed TEST')
    clock_path=root/'TRAINING_CLOCK.json';clock=read_json(clock_path)
    if deadline!=clock['stage_deadline_epoch']:raise ValueError('Reassessment must use the original Stage2 deadline')
    if sha(ctx.root/'SUPPORT_FREEZE.json')!=support_sha:raise ValueError('Reassessment support identity changed')
    selection=_selected_snapshot(states);intent_path=root/'SELECTED_REASSESSMENT_SELECTION.json'
    config_identity=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if intent_path.exists():
        intent=read_json(intent_path)
        if (intent['selected']!=selection or intent['support_freeze_sha256']!=support_sha
                or intent['training_clock_sha256']!=sha(clock_path) or intent['config_identity']!=config_identity):
            raise ValueError('Frozen prior selection/config/clock changed during reassessment resume')
    else:
        forecast=finalization_forecast(regions,states,config)
        intent=dict(schema='PACNHN_SELECTED_REASSESSMENT_V1',selected=selection,
            support_freeze_sha256=support_sha,training_clock_sha256=sha(clock_path),config_identity=config_identity,
            stage_deadline_epoch=deadline,started_epoch=time.time(),source_sha256=sha(__file__),
            forecast=forecast,work_deadline_epoch=deadline-forecast['protected_freeze_and_TEST_seconds'],
            policy='Exact previously selected IDs and bytes only; no fitting, candidate scan, selection, replacement, label exclusion or TEST access')
        _new_json(intent_path,intent)
    intent_sha=sha(intent_path);working=copy.deepcopy(states);outcomes=[]
    def check():
        if time.time()>=min(deadline,intent['work_deadline_epoch']):
            raise TimeoutError('Selected reassessment budget exhausted; protected freeze/TEST reserve retained')
        ctx.guard(deadline=min(deadline,intent['work_deadline_epoch']))
    inventory={r['key']:r for r in regions}
    if not set(selection).issubset(inventory):raise ValueError('Selected region absent from frozen support')
    for key,sources in selection.items():
        record=inventory[key];regional_root=root/'regions'/record['suite_id']/f"{record['region_index']:06d}"
        index_path=regional_root/'labels/REQUEST_INDEX.json'
        index_sha=sha(index_path) if index_path.exists() else None
        state=states.get(key,{})
        # Exact unchanged admitted-index commitment plus a matching prior audit
        # needs no reassessment. Freeze still authenticates every source byte.
        pending_sources={}
        for family,source in sources.items():
            prior=source if family=='POWER4' else source['audit']
            if (index_sha is not None and state.get('audited_request_index_sha256')==index_sha
                    and state.get('dataset_identity') is not None and state.get('valid_dataset_identity') is not None
                    and prior.get('train_dataset_identity')==state['dataset_identity']
                    and prior.get('valid_dataset_identity')==state['valid_dataset_identity']):continue
            pending_sources[family]=source
        for family,source in list(pending_sources.items()):
            receipt_dir=root/'selected_reassessment'/record['suite_id']/f"{record['region_index']:06d}"/family
            try:reused=_reuse_selected_reassessment(ctx,receipt_dir,intent_sha,source,record,regional_root/'labels')
            except (ValueError,OSError):reused=None  # Main bounded audit records an explicit failure; no fallback.
            if reused is None:continue
            receipt_path,receipt=reused;replacement=_reassessed_source(source,family,receipt,receipt_path)
            if family=='POWER4':working[key]['power4']=replacement
            else:working[key].setdefault('selected',{})[family]=replacement
            outcomes.append(dict(region_key=key,family=family,status='PASS',path=str(receipt_path),sha256=sha(receipt_path),reused=True))
            pending_sources.pop(family)
        if not pending_sources:continue
        data_started=time.perf_counter();current=None;datasets={};region=None;data_error=None
        try:
            check();region=load_region_graph(record['path'],expected_sha256=record['sha256']);check()
            current,datasets=current_region_datasets(regional_root/'labels',region,check,include_data=True)
            check()
        except Exception as exc:data_error=exc
        data_seconds=time.perf_counter()-data_started
        for family,source in pending_sources.items():
            prior=source if family=='POWER4' else source['audit']
            if (data_error is None and current['TRAIN']['dataset_identity'] is not None
                    and current['VALID']['dataset_identity'] is not None
                    and prior.get('train_dataset_identity')==current['TRAIN']['dataset_identity']
                    and prior.get('valid_dataset_identity')==current['VALID']['dataset_identity']):
                continue  # Already current; do not evaluate or select it again.
            started=time.time();receipt=dict(schema='PACNHN_SELECTED_REASSESSMENT_RECEIPT_V1',
                selection_intent_sha256=intent_sha,region_key=key,family=family,
                candidate_id=source.get('candidate_id'),model_path=source['model_path'],
                model_sha256=source['model_sha256'],prior_audit=prior,
                prior_audit_path=source.get('audit_path'),
                prior_audit_sha256=sha(source['audit_path']) if source.get('audit_path') and Path(source['audit_path']).exists() else None,
                fit_train_dataset_identity=source.get('fit_train_dataset_identity',source.get('trained_dataset_identity',prior.get('train_dataset_identity'))),
                current_population=current,support_freeze_sha256=support_sha,graph_path=record['path'],graph_sha256=record['sha256'],
                stage_deadline_epoch=deadline,work_deadline_epoch=intent['work_deadline_epoch'],started_epoch=started,
                graph_and_current_data_seconds=data_seconds,TEST_arrays_opened=False,model_fitted=False,selection_changed=False)
            receipt_dir=root/'selected_reassessment'/record['suite_id']/f"{record['region_index']:06d}"/family
            receipt_path=None
            try:
                if data_error is not None:raise data_error
                check()
                if set(datasets)!={'TRAIN','VALID'}:raise ValueError('Current TRAIN and VALID must both be nonempty')
                train,valid=datasets['TRAIN'],datasets['VALID']
                receipt.update(graph_identity=region.identity,support_identity=region.support_identity,
                    c0_identity=array_hash({'c0':region.c0}),pair_keys_identity=array_hash({'pair_keys':region.pair_keys}))
                hash_started=time.perf_counter();before=sha(source['model_path'])
                if before!=source['model_sha256']:raise ValueError('Selected immutable model bytes changed before reassessment')
                hash_seconds=time.perf_counter()-hash_started;check()
                audit=None
                if audit is None:
                    load_started=time.perf_counter();model=load_model(source['model_path'])
                    load_seconds=time.perf_counter()-load_started;check()
                    if model.kind!=source['model_kind'] or not np.array_equal(model.c0,region.c0):
                        raise ValueError('Selected model family/native c0 differs from frozen support')
                    audit=dict(train_dataset_identity=train['dataset_identity'],valid_dataset_identity=valid['dataset_identity'],
                        TRAIN=_evaluate(model,train,config['audit_batch_rows'],check),
                        VALID=_evaluate(model,valid,config['audit_batch_rows'],check),
                        model_sha256=before,model_hash_seconds=hash_seconds,model_load_seconds=load_seconds,
                        evaluation_kind='FINAL_SELECTED_ONLY_REASSESSMENT',recorded_epoch=time.time())
                    if family=='ICNN':
                        audit['neural_branch'],audit['neural_branch_inference_seconds']=_neural_branch_audit(model,train,config['audit_batch_rows'],check)
                    audit['target_met']=audit['TRAIN']['WMAPE_percent']<config['train_target_wmape_percent']
                    del model;check()
                    after=sha(source['model_path'])
                    if after!=before or sha(record['path'])!=record['sha256']:
                        raise ValueError('Selected model or graph bytes changed during reassessment')
                    if sha(regional_root/'labels/REQUEST_INDEX.json')!=current['request_index_sha256']:
                        raise ValueError('Current admitted population changed during reassessment')
                    check();receipt.update(status='PASS',audit=audit,model_sha256_before=before,model_sha256_after=after,
                        ended_epoch=time.time(),elapsed_seconds=time.time()-started)
                replacement=copy.deepcopy(source)
                if family=='POWER4':
                    replacement.update(audit)
                    replacement['fit_train_dataset_identity']=receipt['fit_train_dataset_identity']
                else:replacement['audit']=audit
                replacement['final_reassessment_status']='PASS'
            except Exception as exc:
                receipt.update(status='UNAVAILABLE_REASSESSMENT_TIMEOUT' if isinstance(exc,TimeoutError) else 'UNAVAILABLE_REASSESSMENT_FAILED',
                    error_type=type(exc).__name__,error=str(exc),ended_epoch=time.time(),elapsed_seconds=time.time()-started)
                replacement=copy.deepcopy(source);replacement['final_reassessment_status']=receipt['status']
            if receipt_path is None:
                receipt_path=receipt_dir/f'{time.time_ns()}.json';_new_json(receipt_path,receipt)
            replacement['reassessment_receipt_path']=str(receipt_path);replacement['reassessment_receipt_sha256']=sha(receipt_path)
            if family=='POWER4':working[key]['power4']=replacement
            else:working[key].setdefault('selected',{})[family]=replacement
            outcomes.append(dict(region_key=key,family=family,status=receipt['status'],path=str(receipt_path),sha256=sha(receipt_path)))
        del datasets
    summary=dict(selection_intent_path=str(intent_path),selection_intent_sha256=intent_sha,
        completed_epoch=time.time(),stage_deadline_epoch=deadline,outcomes=outcomes,TEST_arrays_opened=False,
        selected_ids_and_model_bytes_preserved=True)
    _new_json(root/'selected_reassessment'/f'SUMMARY_{time.time_ns()}.json',summary)
    return working


def selected_learning_metadata(family,source=None,audit=None):
    """Flatten measured branch evidence and explicit optimizer/tree counters."""
    fields=('observed_increment_absolute_sum','maximum_native_cost_increment','nonzero_prediction_coordinates',
        'structurally_active','neural_contribution_train_dataset_identity','selected_optimizer_updates',
        'selected_is_initialization','selected_boosting_rounds')
    result={field:None for field in fields}
    if source is None:return result
    audit={} if audit is None else audit
    if family=='ICNN':
        updates=source.get('total_updates')
        if updates is not None:
            if isinstance(updates,bool) or int(updates)!=updates or updates<0:
                raise ValueError('Selected ICNN optimizer counter must be a nonnegative integer')
            result.update(selected_optimizer_updates=int(updates),selected_is_initialization=updates==0)
        branch=audit.get('neural_branch')
        if branch is not None:
            identity=branch.get('train_dataset_identity',audit.get('train_dataset_identity'))
            if identity!=audit.get('train_dataset_identity'):
                raise ValueError('Neural contribution audit uses another TRAIN dataset')
            for field in fields[:4]:result[field]=branch.get(field)
            result['neural_contribution_train_dataset_identity']=identity
    elif family=='XGBOOST':
        rounds=source.get('boosting_rounds_total')
        if rounds is not None:
            if isinstance(rounds,bool) or int(rounds)!=rounds or rounds<0:
                raise ValueError('Selected XGBoost round counter must be a nonnegative integer')
            result['selected_boosting_rounds']=int(rounds)
    return result


def current_region_datasets(label_root,region,check,*,include_data=False):
    """Authenticate actual admitted data, not potentially stale worker STATE."""
    label_root=Path(label_root);index_path=label_root/'REQUEST_INDEX.json'
    identity_path=label_root/'REGION_IDENTITY.json'
    index_sha=sha(index_path) if index_path.exists() else None
    entries=read_json(index_path) if index_sha is not None else {}
    if any(r.get('role') not in {'TRAIN','VALID','TEST'} for r in entries.values()):
        raise ValueError('Unknown role in current admitted population')
    counts={role:sum(r['role']==role and r.get('selected_label') is not None for r in entries.values())
        for role in ('TRAIN','VALID','TEST')}
    result=dict(request_index_path=str(index_path),request_index_sha256=index_sha,
        region_identity_path=str(identity_path),region_identity_sha256=sha(identity_path) if identity_path.exists() else None,
        admitted_counts=counts,source_files={},TEST_arrays_opened=False)
    datasets={}
    for role in ('TRAIN','VALID'):
        check()
        result[role]=dict(dataset_identity=None,unique_rows=0)
        if not counts[role]:continue
        data=load_admitted_dataset(label_root,role=role)
        if (data['identity']['graph_sha256']!=region.graph_sha256 or data['identity']['graph_identity']!=region.identity
                or data['identity']['support_identity']!=region.support_identity
                or data['u'].shape[1]!=region.m or not np.array_equal(data['c0'],region.c0)
                or data['unique_rows']!=counts[role]):
            raise ValueError('Current admitted dataset differs from frozen native graph/support/c0')
        result[role]=dict(dataset_identity=data['dataset_identity'],unique_rows=data['unique_rows'])
        if include_data:datasets[role]=data
        del data
        for row in entries.values():
            selected=row.get('selected_label')
            if row['role']==role and selected is not None:
                # The authenticated loader just verified these exact byte hashes.
                result['source_files'][selected['path']]=selected['sha256']
                result['source_files'][selected['receipt']]=selected['receipt_sha256']
    if (sha(index_path) if index_path.exists() else None)!=index_sha:
        raise ValueError('Admitted population changed during registry authentication')
    return (result,datasets) if include_data else result


def validate_registry_current_data(root,registry,*,check=None):
    """Reject stale/legacy commitments before resuming any frozen TEST access."""
    commitment=registry.get('current_dataset_commitment')
    if not isinstance(commitment,dict) or commitment.get('schema')!='PACNHN_CURRENT_DATASET_COMMITMENT_V1':
        raise ValueError('Frozen registry lacks current admitted dataset commitments; TEST remains sealed')
    populations=commitment.get('regions',{})
    expected={f'{r["suite_id"]}:{r["region_index"]:06d}' for r in registry['models']}
    if set(populations)!=expected:raise ValueError('Frozen dataset commitment does not cover every registry region')
    for key,population in populations.items():
        if check is not None:check()
        sid,index=key.split(':');label_root=Path(root)/'regions'/sid/index/'labels'
        for name,path_key,hash_key in [('REQUEST_INDEX.json','request_index_path','request_index_sha256'),
                ('REGION_IDENTITY.json','region_identity_path','region_identity_sha256')]:
            path=label_root/name
            if Path(population[path_key]).resolve()!=path.resolve():raise ValueError('Frozen regional population path differs')
            actual=sha(path) if path.exists() else None
            if actual!=population[hash_key]:raise ValueError('Frozen registry current admitted population changed: '+key)
        for path,digest in population['source_files'].items():
            if check is not None:check()
            if not Path(path).exists() or sha(path)!=digest:
                raise ValueError('Frozen registry admitted label/receipt bytes changed: '+str(path))
    for row in registry['models']:
        if row['status']!='AVAILABLE' or row['family']=='FFTT':continue
        current=populations[f'{row["suite_id"]}:{row["region_index"]:06d}']
        if (current['TRAIN']['dataset_identity'] is None or current['VALID']['dataset_identity'] is None
                or row.get('train_dataset_identity')!=current['TRAIN']['dataset_identity']
                or row.get('valid_dataset_identity')!=current['VALID']['dataset_identity']):
            raise ValueError('Available selected model does not match committed current TRAIN/VALID')
    return True


def freeze_registry(ctx, root, regions, states, support_sha, config, *, deadline=None):
    path = root/"REGISTRY_FREEZE.json"
    if path.exists():
        registry=read_json(path);validate_registry_current_data(root,registry,check=lambda:ctx.guard());return registry
    records = [];populations={}
    for region_record in regions:
        ctx.guard(deadline=deadline)
        region = load_region_graph(region_record["path"],expected_sha256=region_record["sha256"])
        state = states.get(region_record["key"],{})
        current=current_region_datasets(root/'regions'/region_record['suite_id']/f'{region_record["region_index"]:06d}'/'labels',
            region,lambda:ctx.guard(deadline=deadline))
        populations[region_record['key']]=current
        native = dict(suite_id=region_record["suite_id"],region_index=region_record["region_index"],
            graph_sha256=region.graph_sha256,pair_keys_identity=array_hash({"pair_keys":region.pair_keys}),pairs=region.m,
            train_rows=current['TRAIN']['unique_rows'],valid_rows=current['VALID']['unique_rows'],
            test_rows=current['admitted_counts']['TEST'],test_opened=False,
            current_train_dataset_identity=current['TRAIN']['dataset_identity'],
            current_valid_dataset_identity=current['VALID']['dataset_identity'],
            label_origin='NEW_DETAILED_REGIONAL_FW_ON_FROZEN_P',neural_amplitude=None)
        native.update(selected_learning_metadata(None))
        if region.c0 is None:
            raise ValueError("FFTT registry requires the support-audited c0 artifact")
        fpath = root/"fftt"/region_record["suite_id"]/f'{region_record["region_index"]:06d}.npz'
        if not fpath.exists():
            save_model(Power4Model(region.c0,np.ones(region.m),np.zeros(region.m)),fpath,
                       metadata={"family":"FFTT","graph_sha256":region.graph_sha256})
        records.append(dict(native,family="FFTT",status="AVAILABLE",model_path=str(fpath),model_sha256=sha(fpath),
            artifact_hashes={str(fpath):sha(fpath)},model_kind="DIRECT_COST_POWER4",target_met=None))
        for family in ("POWER4",)+FAMILIES:
            row = dict(native,family=family,status="MISSING",target_met=False)
            source = state.get("power4") if family == "POWER4" else state.get("selected",{}).get(family)
            if source:
                audit = source if family == "POWER4" else source["audit"]
                row.update(reassessment_receipt_path=source.get('reassessment_receipt_path'),
                    reassessment_receipt_sha256=source.get('reassessment_receipt_sha256'),
                    fit_train_dataset_identity=source.get('fit_train_dataset_identity',
                        source.get('trained_dataset_identity',audit.get('train_dataset_identity'))))
                if (source.get('final_reassessment_status','PASS')=='PASS'
                        and current['TRAIN']['dataset_identity'] is not None and current['VALID']['dataset_identity'] is not None
                        and audit["train_dataset_identity"] == current['TRAIN']['dataset_identity']
                        and audit.get("valid_dataset_identity") == current['VALID']['dataset_identity']):
                    if sha(source["model_path"]) != source["model_sha256"]:
                        raise ValueError("Selected model bytes changed before freeze")
                    row.update(status="AVAILABLE",model_path=source["model_path"],model_sha256=source["model_sha256"],
                        artifact_hashes={source["model_path"]:source["model_sha256"]},model_kind=source["model_kind"],
                        candidate_id=source.get("candidate_id"),train_dataset_identity=audit["train_dataset_identity"],
                        valid_dataset_identity=audit['valid_dataset_identity'],
                        TRAIN_WMAPE_percent=audit["TRAIN"]["WMAPE_percent"],VALID_WMAPE_percent=audit["VALID"]["WMAPE_percent"],
                        target_met=audit["TRAIN"]["WMAPE_percent"] < config["train_target_wmape_percent"])
                    row.update(model_file_bytes=Path(source["model_path"]).stat().st_size,
                        tree_count=source.get("tree_count",0),leaf_count=source.get("leaf_count",0),
                        uncompressed_booster_bytes=source.get('uncompressed_booster_bytes',0),
                        uncompressed_model_bytes=source.get('uncompressed_model_bytes',source.get('uncompressed_booster_bytes',0)),
                        native_parameter_bytes=source.get('native_parameter_bytes',0),
                        raw_booster_budget_bytes=source.get('raw_booster_budget_bytes'),
                        observed_amortized_inference_seconds_per_query=audit["TRAIN"].get("amortized_seconds_per_query"),
                        neural_branch=audit.get("neural_branch"))
                    row.update(selected_checkpoint=source.get('candidate_id'),
                        parameters=dict(width=source.get('width'),total_updates=source.get('total_updates'),
                            boosting_rounds=source.get('boosting_rounds_total'),max_depth=source.get('max_depth'),
                            coefficient_parameterization=source.get('coefficient_parameterization'),
                            log_center_policy=source.get('log_center_policy'),log_center_sha256=source.get('log_center_sha256'),
                            pair_count=region.m,dtype='float64 native API; XGBoost float32 internals' if family=='XGBOOST' else 'float64'),
                        neural_amplitude=audit.get('neural_branch',{}).get('amplitude'))
                    row.update(selected_learning_metadata(family,source,audit))
                else:
                    row.update(status=source.get('final_reassessment_status') if source.get('final_reassessment_status','PASS')!='PASS' else 'UNAVAILABLE_STALE_CURRENT_DATASET',target_met=False,
                        unavailability_reason=source.get('final_reassessment_status','Selected audit differs from actual current admitted TRAIN or VALID; no late reselection'),
                        preserved_model_path=source['model_path'],preserved_model_sha256=source['model_sha256'],
                        prior_audited_train_dataset_identity=audit.get('train_dataset_identity'),
                        prior_audited_valid_dataset_identity=audit.get('valid_dataset_identity'),
                        prior_TRAIN_WMAPE_percent=audit.get('TRAIN',{}).get('WMAPE_percent'),
                        prior_VALID_WMAPE_percent=audit.get('VALID',{}).get('WMAPE_percent'))
            records.append(row)
    coverage = []
    for sid in ctx.c["suite_ids"]:
        for family in ("FFTT","POWER4")+FAMILIES:
            rows = [r for r in records if r["suite_id"] == sid and r["family"] == family]
            available = sum(r["status"] == "AVAILABLE" for r in rows)
            coverage.append(dict(suite_id=sid,family=family,expected=len(rows),available=available,
                qualified=sum(r.get("target_met") is True for r in rows),complete_family=bool(rows) and available==len(rows)))
    result = dict(schema="PACNHN_REGISTRY_V1",support_freeze_sha256=support_sha,
        selection_frozen_epoch=time.time(),models=records,coverage=coverage,
        eligibility_policy=config["eligible_pool_policy"],test_opened_before_selection=False)
    result['current_dataset_commitment']=dict(schema='PACNHN_CURRENT_DATASET_COMMITMENT_V1',regions=populations,
        identity_basis='Authenticated actual admitted TRAIN/VALID arrays at freeze; STATE is not authoritative',
        TEST_arrays_opened=False)
    result['learning_field_semantics']=dict(
        neural_contribution='Measured native cost increment relative to the selected ICNN own power4 component on the named current full unique TRAIN dataset; no comparison against a separately fitted power4 model',
        selected_optimizer_updates='Recorded source.total_updates for the selected ICNN checkpoint; never derived from candidate identifiers',
        selected_is_initialization='True only for an explicitly recorded zero optimizer-update checkpoint',
        selected_boosting_rounds='Recorded source.boosting_rounds_total for the selected XGBoost model',
        missing='JSON null / empty CSV for other families or unavailable evidence; measured zero/False values are retained')
    result["selected_model_storage_and_inference_forecast"]={family:dict(
        selected_model_bytes=sum(r.get("model_file_bytes",0) for r in records if r["family"]==family),
        trees=sum(r.get("tree_count",0) for r in records if r["family"]==family),
        leaves=sum(r.get("leaf_count",0) for r in records if r["family"]==family),
        sum_observed_amortized_query_seconds=sum(r.get("observed_amortized_inference_seconds_per_query") or 0. for r in records if r["family"]==family),
        limitations="Batch-amortized measured inference sum; not measured full-registry singleton latency or RSS. Includes every available coordinate, no output dropping.") for family in FAMILIES}
    memory_path=ctx.root/'training'/'XGB_REGISTRY_MEMORY_BUDGET.json'
    if memory_path.exists():
        memory=read_json(memory_path)
        result['XGB_memory_budget']=memory
        result['XGB_per_suite_memory_forecast']={sid:dict(
            raw_booster_bytes=sum(r.get('uncompressed_booster_bytes',0) for r in records if r['suite_id']==sid and r['family']=='XGBOOST'),
            raw_model_bytes_including_native_parameters=sum(r.get('uncompressed_model_bytes',0) for r in records if r['suite_id']==sid and r['family']=='XGBOOST'),
            forecast_resident_booster_bytes=memory['resident_multiplier']*sum(r.get('uncompressed_model_bytes',0)
                for r in records if r['suite_id']==sid and r['family']=='XGBOOST'),
            scope='CONSERVATIVE_MULTIPLIER_FORECAST_NOT_MEASURED_COMPLETE_REGISTRY_RSS') for sid in ctx.c['suite_ids']}
    _new_json(path,result)
    return result


def export_learning_csv(root,registry):
    tests={}
    for path in (root/'final_test').glob('*/*.json'):
        record=read_json(path)
        for model in record['models']:
            tests[(record['suite_id'],record['region_index'],model['family'])]=dict(model,rows=record['admitted_test_inputs'])
    fields=['suite_id','region_index','family','status','target_met','train_rows','valid_rows','test_rows','test_opened',
        'TRAIN_WMAPE_percent','VALID_WMAPE_percent','TEST_WMAPE_percent','TEST_status','selected_checkpoint','label_origin',
        'parameters','neural_amplitude','observed_increment_absolute_sum','maximum_native_cost_increment',
        'nonzero_prediction_coordinates','structurally_active','neural_contribution_train_dataset_identity',
        'selected_optimizer_updates','selected_is_initialization','selected_boosting_rounds',
        'model_path','model_sha256','graph_sha256','train_dataset_identity','valid_dataset_identity',
        'current_train_dataset_identity','current_valid_dataset_identity','unavailability_reason',
        'preserved_model_path','preserved_model_sha256','prior_audited_train_dataset_identity','prior_audited_valid_dataset_identity',
        'pairs','model_file_bytes','tree_count','leaf_count','fit_train_dataset_identity',
        'reassessment_receipt_path','reassessment_receipt_sha256']
    path=root/'LEARNING_RESULTS.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in registry['models']:
            out={k:row.get(k) for k in fields};test=tests.get((row['suite_id'],row['region_index'],row['family']))
            out['parameters']=json.dumps(row.get('parameters',{}),sort_keys=True)
            out['TEST_status']='AVAILABLE_AFTER_FREEZE' if test else 'UNAVAILABLE_NO_VALID_FROZEN_TEST_METRIC'
            out['test_opened']=bool(test)
            if test:out.update(TEST_WMAPE_percent=test['WMAPE_percent'],test_rows=test['rows'])
            writer.writerow(out)
    return str(path)


def export_curves(root,clock):
    fields=['suite_id','region_index','family','block','elapsed_seconds','train_wmape','valid_wmape','candidate_id',
        'train_dataset_identity','valid_dataset_identity','evaluation_kind','audit_path']
    count=0;new_count=0;path=root/'TRAINING_CURVES.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for region_root in sorted((root/'regions').glob('*/*')):
            checkpoints={}
            for p in (region_root/'checkpoints').glob('*/CHECKPOINT.json'):
                record=read_json(p);checkpoints[record['candidate_id']]=record
            for p in sorted((region_root/'audits').glob('*/*.json')):
                audit=read_json(p);candidate=p.name.split('_'+audit['train_dataset_identity'])[0]
                checkpoint=checkpoints.get(candidate)
                if checkpoint is None:continue
                is_new=checkpoint.get('trained_dataset_identity')==audit['train_dataset_identity']
                writer.writerow(dict(suite_id=region_root.parent.name,region_index=int(region_root.name),family=p.parent.name,
                    block=checkpoint.get('training_block_index'),elapsed_seconds=audit['recorded_epoch']-clock['stage_start_epoch'],
                    train_wmape=audit['TRAIN']['WMAPE_percent'],valid_wmape=audit['VALID']['WMAPE_percent'],candidate_id=candidate,
                    train_dataset_identity=audit['train_dataset_identity'],valid_dataset_identity=audit['valid_dataset_identity'],
                    evaluation_kind='NEW_CHECKPOINT' if is_new else 'REEVALUATION_ON_GROWING_TRAIN',audit_path=str(p)))
                count+=1;new_count+=int(is_new)
    write_json(root/'TRAINING_CURVES_METADATA.json',dict(rows=count,new_checkpoint_rows=new_count,
        metric_units='percent ratio of native cost absolute-error sums',elapsed_clock='original Stage2 start',
        disposition='MEASURED' if count else 'NO_COMPLETED_CURRENT_DATA_CANDIDATE_AUDITS_HEADER_ONLY',
        note='Reevaluated historical candidates are explicitly tagged; these are not additional optimizer blocks.'))
    return str(path)


def final_test_audit(ctx, root, registry, deadline, *, _population_verified=False):
    """The only TEST reader: binds a persisted frozen registry before array access."""
    registry_path = root/"REGISTRY_FREEZE.json"; registry_sha = sha(registry_path)
    if read_json(registry_path) != registry:
        raise ValueError("Final registry argument differs from persisted selection")
    if time.time()>=deadline:raise TimeoutError('Original Stage2 deadline reached before TEST access')
    if not _population_verified:
        validate_registry_current_data(root,registry,check=lambda:ctx.guard(deadline=deadline))
    entry_path = root/"TEST_OPENING_AUTHORIZATION.json"
    if entry_path.exists():
        if read_json(entry_path)["registry_sha256"] != registry_sha:
            raise ValueError("TEST is already bound to another frozen registry")
    else:
        _new_json(entry_path,dict(registry_sha256=registry_sha,opened_epoch=time.time(),
            policy="ONCE_AFTER_FINAL_SELECTION_NO_FEEDBACK",selection_frozen_epoch=registry["selection_frozen_epoch"]))
    grouped = {}
    for row in registry["models"]:
        if row["status"] == "AVAILABLE":
            grouped.setdefault((row["suite_id"],row["region_index"]),[]).append(row)
    outcomes = []
    for (sid,index),models in grouped.items():
        out = root/"final_test"/sid/f"{index:06d}.json"
        if out.exists():
            prior = read_json(out)
            if prior["registry_sha256"] != registry_sha:
                raise ValueError("Existing final TEST audit belongs to different registry")
            outcomes.append(prior); continue
        if time.time() >= deadline:
            break
        ctx.guard(deadline=deadline)
        data_root = root/"regions"/sid/f"{index:06d}"/"labels"
        request_index = data_root/"REQUEST_INDEX.json"
        entries = read_json(request_index) if request_index.exists() else {}
        inputs=[]; truths=[]
        for record in entries.values():
            selected=record["selected_label"]
            if record["role"] != "TEST" or selected is None:
                continue
            if sha(selected["path"]) != selected["sha256"] or sha(selected["receipt"]) != selected["receipt_sha256"]:
                raise ValueError("Blind TEST artifact changed")
            receipt=read_json(selected["receipt"])
            if receipt["role"] != "TEST" or not receipt["admitted"]:
                raise ValueError("Blind TEST label role/admission invalid")
            with np.load(selected["path"],allow_pickle=False) as z:
                inputs.append(z["u"].copy());truths.append(z["tau"].copy())
        evidence=dict(suite_id=sid,region_index=index,registry_sha256=registry_sha,
            admitted_test_inputs=len(inputs),requested_test_inputs=sum(r["role"]=="TEST" for r in entries.values()),models=[])
        if inputs:
            dataset=dict(u=np.stack(inputs),tau=np.stack(truths))
            for model_row in models:
                if sha(model_row["model_path"]) != model_row["model_sha256"]:
                    raise ValueError("Frozen TEST model changed")
                metric=_evaluate(load_model(model_row["model_path"]),dataset,256,lambda:ctx.guard(deadline=deadline))
                evidence["models"].append(dict(family=model_row["family"],model_sha256=model_row["model_sha256"],**metric))
        _new_json(out,evidence);outcomes.append(evidence)
    write_json(root/"FINAL_TEST_SUMMARY.json",dict(registry_sha256=registry_sha,regions_audited=len(outcomes),
        regions_available=len(grouped),complete=len(outcomes)==len(grouped),results=outcomes,
        test_results_used_for_selection=False))
    return outcomes


def finalize_test_with_receipt(ctx,root,registry,deadline,*,_population_verified=False):
    """Preserve a registry-bound disposition even when held-out time expires."""
    registry_path=root/'REGISTRY_FREEZE.json'
    registry_sha=sha(registry_path)
    if not _population_verified:
        validate_registry_current_data(root,registry,check=None if ctx is None else lambda:ctx.guard())
    summary_path=root/'FINAL_TEST_SUMMARY.json'
    if summary_path.exists():
        summary=read_json(summary_path)
        if summary.get('registry_sha256')!=registry_sha:raise ValueError('Existing TEST summary belongs to another registry')
        if summary.get('complete') is True:return summary.get('results',[])
    try:
        if time.time()>=deadline:raise TimeoutError('Original Stage2 deadline reached before final TEST completion')
        return final_test_audit(ctx,root,registry,deadline,_population_verified=True)
    except TimeoutError as exc:
        path=root/'REGISTRY_FREEZE.json'
        if read_json(path)!=registry:raise ValueError('Timeout disposition must bind the exact persisted registry')
        write_json(root/'FINAL_TEST_TIMEOUT.json',dict(status='INCOMPLETE_TIME_LIMIT',error=str(exc),
            registry_sha256=sha(path),selection_frozen_epoch=registry['selection_frozen_epoch'],
            recorded_epoch=time.time(),deadline_epoch=deadline,VALID_not_substituted=True,
            TEST_opening_authorization_exists=(root/'TEST_OPENING_AUTHORIZATION.json').exists()))
        return None


def ensure_finalization_clock(root,clock,registry=None):
    """Journal before the first freeze; never reset this interval on restart."""
    path=root/'FINALIZATION_CLOCK.json'
    if path.exists():
        record=read_json(path)
        if (record['stage_start_epoch']!=clock['stage_start_epoch'] or record['stage_deadline_epoch']!=clock['stage_deadline_epoch']
                or not clock['stage_start_epoch']<=record['finalization_started_epoch']<=time.time()):
            raise ValueError('Persisted finalization clock conflicts with the original Stage2 clock')
        return record
    started=time.time() if registry is None else registry['selection_frozen_epoch']
    if not clock['stage_start_epoch']<=started<=time.time():raise ValueError('Invalid finalization start evidence')
    record=dict(stage_start_epoch=clock['stage_start_epoch'],stage_deadline_epoch=clock['stage_deadline_epoch'],
        finalization_started_epoch=started,recorded_epoch=time.time(),
        start_basis='PERSISTED_BEFORE_FIRST_REGISTRY_FREEZE' if registry is None else 'LEGACY_SELECTION_FREEZE_EPOCH_PRE_FREEZE_FINALIZATION_TIME_UNKNOWN',
        pre_freeze_finalization_interval_known=registry is None)
    _new_json(path,record);return record


def complete_frozen_training(ctx,root,registry,clock,*,controller=None,source_intent=None,resumed=False):
    """Fresh and resumed finalization emit one identical bound Stage2 handoff."""
    invoked=time.time();registry_path=root/'REGISTRY_FREEZE.json';registry_sha=sha(registry_path)
    if read_json(registry_path)!=registry:raise ValueError('Stage2 handoff requires the exact persisted registry')
    validate_registry_current_data(root,registry,check=None if ctx is None else lambda:ctx.guard())
    result_path=root/'RESULT.json'
    if result_path.exists():
        prior=read_json(result_path)
        if (prior.get('status')!='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE' or prior.get('registry_sha256')!=registry_sha
                or prior.get('clock')!=clock):raise ValueError('Existing Stage2 completion receipt conflicts with the frozen registry or clock')
        return prior
    journal=ensure_finalization_clock(root,clock,registry)
    finalize_test_with_receipt(ctx,root,registry,clock['stage_deadline_epoch'],_population_verified=True)
    export_learning_csv(root,registry);export_curves(root,clock)
    dispositions=[]
    for name in ('FINAL_TEST_SUMMARY.json','FINAL_TEST_TIMEOUT.json'):
        path=root/name
        if path.exists():
            record=read_json(path)
            if record.get('registry_sha256')!=registry_sha:raise ValueError('Stage2 final TEST disposition binds another registry')
            dispositions.append(dict(path=str(path),sha256=sha(path),kind=name,
                complete=record.get('complete',False),status=record.get('status','SUMMARY')))
    if not dispositions:raise ValueError('Stage2 completion lacks a bound final TEST disposition')
    selected=next((r for r in dispositions if r['kind']=='FINAL_TEST_SUMMARY.json' and r['complete']),
        next((r for r in dispositions if r['kind']=='FINAL_TEST_TIMEOUT.json'),dispositions[0]))
    controller={} if controller is None else controller
    ended=time.time();started=journal['finalization_started_epoch']
    result=dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry=str(registry_path),registry_sha256=registry_sha,
        coverage=registry['coverage'],clock=clock,slices=controller.get('slices'),
        first_all_s03_qualified=controller.get('scheduler',{}).get('first_all_s03_qualified'),
        source_intent=source_intent or controller.get('source_intent'),TEST_opened_only_after_freeze=True,
        test_disposition=selected,test_dispositions=dispositions,
        finalization_clock_path=str(root/'FINALIZATION_CLOCK.json'),finalization_clock_sha256=sha(root/'FINALIZATION_CLOCK.json'),
        finalization_started_epoch=started,finalization_ended_epoch=ended,finalization_elapsed_seconds=ended-started,
        finalization_start_basis=journal['start_basis'],pre_freeze_finalization_interval_known=journal['pre_freeze_finalization_interval_known'],
        finalization_current_invocation_seconds=ended-invoked,finalization_resumed=resumed,
        finalization_wall_includes_restart_gaps=resumed,completed_after_stage_deadline=ended>clock['stage_deadline_epoch'],
        learning_csv_sha256=sha(root/'LEARNING_RESULTS.csv'),training_curves_sha256=sha(root/'TRAINING_CURVES.csv'))
    write_json(result_path,result);return result


def owned_stop_request(ctx,root,controller,clock):
    """A run/PID-bound administrative request; never signal unrelated processes."""
    path=root/'STOP_REQUEST.json'
    if not path.exists():return None
    request=read_json(path)
    if request.get('target_pid')!=os.getpid():return None
    if (request.get('phase_id')!='PACNHN_V1' or request.get('run_id')!=ctx.c['run_id']
            or request.get('action')!='PAUSE_FOR_REPAIR' or not request.get('request_id') or not request.get('reason')
            or request.get('stage_deadline_epoch')!=clock['stage_deadline_epoch']):
        raise ValueError('Owned stop request must bind this exact run, action and original clock')
    digest=sha(path)
    if digest in controller.get('handled_stop_requests',[]):return None
    archived=root/'pause_requests'/(digest+'.json')
    if archived.exists():
        if read_json(archived)!=request:raise ValueError('Preserved stop request differs')
    else:_new_json(archived,request)
    return dict(request=request,request_sha256=digest,archived_request=str(archived),acknowledged_epoch=time.time())


def pause_without_finalization(root,controller,clock,request,intent_path):
    """Administrative pause receipt only: no registry freeze or blind evaluation."""
    controller['status']='PAUSED_FOR_REPAIR';controller['stop_reason']='OWNED_STOP_REQUEST'
    controller.setdefault('handled_stop_requests',[]).append(request['request_sha256'])
    write_json(root/'CONTROLLER_STATE.json',controller)
    result=dict(status='PAUSED_FOR_REPAIR',pause_request=request,clock=clock,
        source_intent=str(intent_path),completed_epoch=time.time(),slices=controller['slices'],
        regions_with_persisted_state=len(controller['states']),TEST_opened=False,registry_frozen=False)
    _new_json(root/'pause_receipts'/f'{time.time_ns()}.json',result)
    return result


def recovered_current_data_state(recovered,region_root):
    """Cheap restart screen; stale/legacy saved states must revisit real data."""
    result=compact_state(recovered,region_root)
    index=Path(region_root)/'labels/REQUEST_INDEX.json'
    current=sha(index) if index.exists() else None
    if (recovered.get('audited_request_index_sha256')!=current
            or (current is None and recovered.get('dataset_identity'))):
        result['requires_current_dataset_refresh']=True
        result['current_request_index_sha256']=current
    return result


def recover_controller_states(root,regions,controller):
    """Recover only persisted regional metadata; no fitting or label-array load."""
    states=controller['states']
    for record in regions:
        region_root=root/'regions'/record['suite_id']/f'{record["region_index"]:06d}'
        regional_state=region_root/'STATE.json'
        chosen=states.get(record['key'])
        if regional_state.exists():
            recovered=read_json(regional_state)
            if recovered.get('visits',0)>=(chosen or {}).get('visits',0):chosen=recovered
        if chosen is not None:
            states[record['key']]=recovered_current_data_state(chosen,region_root)


def resource_pause_receipt(ctx,root,clock,exc):
    """Persist a resource pause only after the controller pool has unwound."""
    root=Path(root);path=root/'CONTROLLER_STATE.json'
    controller=read_json(path) if path.exists() else dict(states={},scheduler={},slices=0)
    cleanup=getattr(exc,'pacnhn_owned_cleanup',None)
    if cleanup is None and getattr(ctx,'_training_owned_pool_active',False):
        raise RuntimeError('Resource exception occurred before owned-pool cleanup completed; pause not certified') from exc
    if cleanup is None:
        cleanup=dict(completed=True,completed_epoch=time.time(),mode='NO_ACTIVE_POOL_AT_EXCEPTION_BOUNDARY',owned_pids=[])
    if not cleanup.get('completed'):raise RuntimeError('Resource pause cannot claim unfinished owned-worker cleanup')
    intent=getattr(exc,'pacnhn_source_intent',getattr(ctx,'_training_current_intent',None))
    intent_path=Path(intent) if intent and Path(intent).exists() else None
    prior=controller.get('source_intent')
    prior_path=Path(prior) if prior and Path(prior).exists() else None
    current_config=getattr(ctx,'_training_current_config_path',None)
    receipt_path=root/'resource_pause_receipts'/f'{time.time_ns()}.json'
    controller.update(status='RESOURCE_PAUSED',stop_reason='PARENT_MEMORY_GUARD',
        resource_error_type=type(exc).__name__,resource_error=str(exc),resource_pause_receipt_path=str(receipt_path))
    write_json(path,controller)
    record=dict(schema='PACNHN_RESOURCE_PAUSE_V1',status='RESOURCE_PAUSED',phase_id=ctx.c['phase_id'],
        run_id=ctx.c['run_id'],pid=os.getpid(),context_start_epoch=ctx.started,clock=clock,
        training_clock_path=str(ctx.root/'training/TRAINING_CLOCK.json'),training_clock_sha256=sha(ctx.root/'training/TRAINING_CLOCK.json'),
        source_intent=str(intent_path) if intent_path else None,
        source_intent_sha256=sha(intent_path) if intent_path else None,
        source_intent_scope='CURRENT_INVOCATION_WRITTEN_INTENT' if intent_path else 'NO_CURRENT_INVOCATION_INTENT_CREATED',
        prior_controller_source_intent=str(prior_path) if prior_path else None,
        prior_controller_source_intent_sha256=sha(prior_path) if prior_path else None,
        prior_controller_source_intent_scope='RETAINED_CONTROLLER_METADATA_NOT_CURRENT_PROCESS_ATTRIBUTION',
        current_invocation_source_sha256=sha(__file__),
        current_training_config_path=current_config,current_training_config_sha256=sha(current_config) if current_config else None,
        reason=str(exc),error_type=type(exc).__name__,owned_worker_cleanup=cleanup,
        controller_state_path=str(path),controller_state_sha256=sha(path),
        guard=dict(peak_rss_bytes=getattr(ctx,'peak_rss',None),
            minimum_available_bytes=None if getattr(ctx,'minimum_available',None)==2**63-1 else getattr(ctx,'minimum_available',None),
            limits=ctx.c.get('resources',{})),
        TEST_opened=(root/'TEST_OPENING_AUTHORIZATION.json').exists(),registry_frozen=(root/'REGISTRY_FREEZE.json').exists(),
        no_new_registry_or_TEST_during_pause=True,completed_epoch=time.time(),
        regions_with_persisted_state=len(controller.get('states',{})),slices=controller.get('slices',0))
    _new_json(receipt_path,record)
    print('RESOURCE_PAUSED '+str(receipt_path),flush=True)
    return record


def expired_finalization_receipt(ctx,root,clock,reason):
    """Metadata-only expiry; never manufacture a registry or Stage2 success."""
    root=Path(root);path=root/'CONTROLLER_STATE.json'
    controller=read_json(path) if path.exists() else dict(states={},scheduler={},slices=0)
    hard=time.time()>=ctx.c['hard_deadline_epoch']
    registry=root/'REGISTRY_FREEZE.json'
    status='CAMPAIGN_HARD_DEADLINE_EXPIRED' if hard else ('STAGE2_EXPIRED_BEFORE_REGISTRY_FREEZE' if not registry.exists() else 'STAGE2_FINALIZATION_INCOMPLETE_TIME_LIMIT')
    record=dict(schema='PACNHN_FINALIZATION_EXPIRY_V1',status=status,phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        pid=os.getpid(),clock=clock,training_clock_sha256=sha(root/'TRAINING_CLOCK.json'),
        hard_deadline_epoch=ctx.c['hard_deadline_epoch'],reason=str(reason),recorded_epoch=time.time(),
        registry_frozen=registry.exists(),registry_sha256=sha(registry) if registry.exists() else None,
        TEST_opening_authorization_exists=(root/'TEST_OPENING_AUTHORIZATION.json').exists(),
        metadata_only=True,new_registry_created=False,new_model_availability_claimed=False,new_scientific_evaluation=False,
        completed_stage2=False,regions_with_persisted_state=len(controller.get('states',{})))
    receipt=root/'finalization_expiry_receipts'/f'{time.time_ns()}.json';_new_json(receipt,record)
    controller.update(status=status,stop_reason='ORIGINAL_DEADLINE_EXPIRED',finalization_expiry_receipt_path=str(receipt))
    write_json(path,controller)
    print(status+' '+str(receipt),flush=True)
    return record


def finalize_retained_controller(ctx,root,regions,controller,config,config_path,clock):
    """Same-clock late restart: authenticate retained selections, no new workers."""
    if controller.get('config_sha256')!=sha(config_path):
        raise ValueError('Retained controller configuration differs from finalization configuration')
    original_intent=controller.get('source_intent')
    if not original_intent or not Path(original_intent).exists():
        raise ValueError('Retained controller lacks its original execution intent')
    prior=read_json(original_intent);support_sha=sha(ctx.root/'SUPPORT_FREEZE.json')
    if (prior.get('phase_id')!='PACNHN_V1' or prior.get('run_id')!=ctx.c['run_id'] or prior.get('clock')!=clock
            or prior.get('config_sha256')!=sha(config_path) or prior.get('support_freeze_sha256')!=support_sha):
        raise ValueError('Retained execution intent clock/config/support identity differs')
    recover_controller_states(root,regions,controller)
    if not _selected_snapshot(controller['states']):
        raise ValueError('Late finalization requires retained selected checkpoints; empty recovery is not success')
    ctx.guard(deadline=clock['stage_deadline_epoch'])
    intent=dict(phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],step_id='FINALIZATION_ONLY_RECOVERY',pilot=False,
        clock=clock,original_source_intent=original_intent,original_source_intent_sha256=sha(original_intent),
        config_path=str(config_path),config_sha256=sha(config_path),support_freeze_sha256=support_sha,
        source_sha256=sha(__file__),started_epoch=time.time(),
        tasks=['AUTHENTICATE_RETAINED_SELECTIONS','BOUNDED_SELECTED_REASSESSMENT','REGISTRY_FREEZE','BOUNDED_BLIND_TEST'],
        new_training_or_labels=False,new_device_or_concurrency_pilots=False)
    intent_path=root/'finalization_recovery_intents'/f'{time.time_ns()}.json';_new_json(intent_path,intent)
    ctx._training_current_intent=str(intent_path)
    controller.update(status='FINALIZATION_ONLY_RECOVERY',finalization_recovery_intent=str(intent_path))
    write_json(root/'CONTROLLER_STATE.json',controller)
    ensure_finalization_clock(root,clock)
    states=reassess_selected_before_freeze(ctx,root,regions,controller['states'],support_sha,config,clock['stage_deadline_epoch'])
    registry=freeze_registry(ctx,root,regions,states,support_sha,config,deadline=clock['stage_deadline_epoch'])
    return complete_frozen_training(ctx,root,registry,clock,controller=controller,source_intent=original_intent,resumed=True)


def run(ctx, training_config_path, *, pilot=False, pilot_regions=(), pilot_seconds=600., pilot_id=None):
    """Resource pause and expiry boundary around the owned-pool controller."""
    if pilot:
        if (not pilot_id or pilot_id in {'.','..'} or '/' in pilot_id or '\\' in pilot_id
                or ':' in pilot_id or not 1<=len(pilot_regions)<=3 or not math.isfinite(pilot_seconds) or pilot_seconds<=0):
            raise ValueError('Pilot requires a safe explicit ID,1-3 regions and positive budget')
        pilot_root=ctx.root/'training/pilots'/pilot_id
        if pilot_root.exists():raise FileExistsError('Pilot output is immutable: '+str(pilot_root))
    ctx._training_current_intent=None
    ctx._training_current_config_path=str(Path(training_config_path).resolve())
    config=read_training_config(training_config_path)
    clock=stage_clock(ctx,config)
    root=ctx.root/'training'
    # Expired recovery writes diagnostics only; no preflight input reloading,
    # device pilot, model prediction or held-out access can occur here.
    if not pilot and time.time()>=ctx.c['hard_deadline_epoch']:
        return expired_finalization_receipt(ctx,root,clock,'Original hard campaign deadline reached before recovery')
    if not pilot and time.time()>=clock['stage_deadline_epoch'] and not (root/'REGISTRY_FREEZE.json').exists():
        return expired_finalization_receipt(ctx,root,clock,'Original Stage2 deadline reached before registry freeze')
    try:
        return _run_training(ctx,training_config_path,pilot=pilot,pilot_regions=pilot_regions,pilot_seconds=pilot_seconds,pilot_id=pilot_id)
    except MemoryError as exc:
        # _run_training has already executed its owned-pool finally cleanup.
        # Pilot outputs remain isolated, while their clock is the shared one.
        if pilot:root=pilot_root
        return resource_pause_receipt(ctx,root,clock,exc)
    except TimeoutError as exc:
        if not pilot and time.time()>=min(clock['stage_deadline_epoch'],ctx.c['hard_deadline_epoch']):
            return expired_finalization_receipt(ctx,root,clock,exc)
        raise


def _run_training(ctx, training_config_path, *, pilot=False, pilot_regions=(), pilot_seconds=600., pilot_id=None):
    from .gates import require_component
    config=read_training_config(training_config_path)
    ctx.bind(training_config_path,"EXPLICIT_TRAINING_CONFIGURATION")
    clock=stage_clock(ctx,config)
    require_component(ctx,"training",extra_config=training_config_path)
    support,regions=gate_inputs(ctx)
    if pilot and (not 1<=len(pilot_regions)<=3 or not pilot_id or not math.isfinite(pilot_seconds) or pilot_seconds<=0):
        raise ValueError("Pilot requires explicit id, 1–3 region keys and positive wall budget")
    root=ctx.root/"training"
    if pilot:
        selected=set(pilot_regions)
        if len(selected)!=len(pilot_regions):raise ValueError('Pilot region keys must be distinct')
        if not selected.issubset({r["key"] for r in regions}):
            raise ValueError("Unknown pilot region")
        inventory={r['key']:r for r in regions}
        regions=[inventory[key] for key in pilot_regions]
        root=root/"pilots"/pilot_id
        root.mkdir(parents=True,exist_ok=False)
    else:
        root.mkdir(parents=True,exist_ok=True)
        if (root/"REGISTRY_FREEZE.json").exists():
            registry=read_json(root/'REGISTRY_FREEZE.json')
            controller=read_json(root/'CONTROLLER_STATE.json') if (root/'CONTROLLER_STATE.json').exists() else {}
            return complete_frozen_training(ctx,root,registry,clock,controller=controller,resumed=True)
    if not pilot and (root/'SELECTED_REASSESSMENT_SELECTION.json').exists():
        # Once exact selections are journaled, restart cannot resume fitting or
        # alter candidate IDs. Continue only the original bounded finalization.
        controller=read_json(root/'CONTROLLER_STATE.json')
        support_sha=sha(ctx.root/'SUPPORT_FREEZE.json')
        states=reassess_selected_before_freeze(ctx,root,regions,controller['states'],support_sha,config,clock['stage_deadline_epoch'])
        registry=freeze_registry(ctx,root,regions,states,support_sha,config,deadline=clock['stage_deadline_epoch'])
        return complete_frozen_training(ctx,root,registry,clock,controller=controller,resumed=True)
    torch.set_num_threads(1)
    deadline=min(clock["stage_deadline_epoch"],time.time()+pilot_seconds) if pilot else clock["stage_deadline_epoch"]
    science=deadline-5 if pilot else deadline-config["final_evaluation_reserve_seconds"]
    if not pilot and (root/'CONTROLLER_STATE.json').exists():
        retained=read_json(root/'CONTROLLER_STATE.json')
        if retained.get('config_sha256')!=sha(training_config_path):
            raise ValueError('Retained controller configuration changed')
        recover_controller_states(root,regions,retained)
        current_cutoff=deadline-max(config['final_evaluation_reserve_seconds'],
            finalization_forecast(regions,retained['states'],config)['reserve_seconds'])
        if time.time()>=current_cutoff:
            return finalize_retained_controller(ctx,root,regions,retained,config,training_config_path,clock)
    if time.time()>=science:
        raise TimeoutError("No scientific training budget remains and no retained controller is available")
    registry_memory=freeze_xgb_memory_budget(ctx,regions,config)
    source_files=[Path(__file__),Path(__file__).with_name("models.py"),Path(__file__).with_name("regional.py")]
    intent=dict(phase_id="PACNHN_V1",step_id="training",run_id=ctx.c["run_id"],pilot=pilot,
        pilot_regions=list(pilot_regions),config_path=str(training_config_path),config_sha256=sha(training_config_path),
        source_hashes={str(p):sha(p) for p in source_files},support_freeze_sha256=sha(ctx.root/"SUPPORT_FREEZE.json"),
        witness_gate_sha256=sha(ctx.root/"COMPLETE_WITNESS_AUDIT.json"),clock=clock,science_deadline_epoch=science,
        actual_deadline_epoch=deadline,resources=ctx.c["resources"],settings=config)
    intent['xgb_registry_memory_budget']=registry_memory
    intent.update(task_list=['FIXED_ROLE_LABELS','RESUMABLE_ICNN_XGB_DIRECT_COST_POWER4','CURRENT_DATA_SELECTION','FINAL_FROZEN_TEST'],
        expected_outputs=['REGISTRY_FREEZE.json','LEARNING_RESULTS.csv','TRAINING_CURVES.csv','FINAL_TEST_SUMMARY.json'],acceptance_tests=ctx.c['acceptance_tests'])
    intent_path=root/"intents"/f'{time.time_ns()}.json';_new_json(intent_path,intent)
    ctx._training_current_intent=str(intent_path)
    device=choose_device(root,config)
    workers=concurrency_gate(ctx,root,config,science)
    state_path=root/"CONTROLLER_STATE.json"
    controller=read_json(state_path) if state_path.exists() else dict(states={},scheduler={},slices=0)
    if controller.get("config_sha256",sha(training_config_path)) != sha(training_config_path):
        raise ValueError("Controller configuration changed without a new versioned execution plan")
    controller["config_sha256"]=sha(training_config_path)
    controller['source_intent']=str(intent_path)
    controller['status']='TRAINING_CONTINUES'
    if controller.get('stop_reason') in {'OWNED_STOP_REQUEST','PARENT_MEMORY_GUARD'}:controller.pop('stop_reason')
    states=controller["states"]; scheduler=controller["scheduler"]
    recover_controller_states(root,regions,controller)
    controller.setdefault("other_existing_training_bytes",_tree_bytes(root)-sum(s.get("persisted_region_bytes",0) for s in states.values()))
    pilot_attempts=controller.setdefault('pilot_slice_attempts',{}) if pilot else {}
    pending={};pause_request=None;resource_exception=None
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    ctx._training_owned_pool_active=True
    try:
        while time.time()<science or pending:
            ctx.guard(deadline=deadline)
            if not pilot and pause_request is None:
                pause_request=owned_stop_request(ctx,root,controller,clock)
                if pause_request is not None:
                    controller['pause_request_acknowledged']=pause_request
                    controller['status']='PAUSING_FOR_REPAIR'
                    write_json(state_path,controller)
            if not pilot:
                forecast=finalization_forecast(regions,states,config);science=deadline-forecast['reserve_seconds']
                write_json(root/'FINALIZATION_FORECAST.json',dict(forecast,science_deadline_epoch=science,stage_deadline_epoch=deadline,recorded_epoch=time.time()))
            if pending and time.time()>=science+config['worker_stop_grace_seconds']:
                controller['stop_reason']='SCIENTIFIC_WORKER_GRACE_EXHAUSTED'
                break
            accounted=clock.get("output_bytes_before_training",0)+controller["other_existing_training_bytes"]+sum(s.get("persisted_region_bytes",0) for s in states.values())
            # Leave a bounded write allowance before starting another label/model slice.
            if accounted+config["active_write_and_finalization_reserve_bytes"]>=ctx.c["resources"]["added_disk_bytes"]:
                controller["stop_reason"]="ADDED_DISK_BUDGET";science=min(science,time.time())
            while pause_request is None and len(pending)<workers and time.time()<science:
                active={v['record']['key'] for v in pending.values()}
                if pilot:
                    record=schedule_pilot_region(regions,states,pilot_attempts,exclude_keys=active,
                        now=time.time(),deadline=science)
                else:record=schedule_region(regions,states,scheduler,config,exclude_keys=active)
                if record is None:break
                milestone=root/'S03_FIRST_ALL_QUALIFIED.json'
                if scheduler.get('first_all_s03_qualified') and not milestone.exists():
                    _new_json(milestone,dict(epoch=scheduler['first_all_s03_qualified'],support_sha256=intent['support_freeze_sha256'],
                        regions={r['key']:dict(train_dataset_identity=states[r['key']]['dataset_identity'],
                            selected={f:states[r['key']]['selected'][f]['model_sha256'] for f in FAMILIES}) for r in regions if r['suite_id']=='s03'}))
                    scheduler['qualification_milestone_path']=str(milestone);scheduler['qualification_milestone_sha256']=sha(milestone)
                unvisited=sum(r['suite_id']=='s03' and not states.get(r['key'],{}).get('visits') for r in regions)
                seconds=min(config['region_slice_seconds'],max(2.,(science-time.time())*.8*workers/max(1,unvisited)))
                regional_root=root/'regions'/record['suite_id']/f'{record["region_index"]:06d}'
                used_device='cuda' if device=='cuda' and record['pairs']>=1024 and not any(v['device']=='cuda' for v in pending.values()) else 'cpu'
                payload=_worker_payload(ctx,record,regional_root,config,used_device,science,seconds)
                future=pool.submit(regional_worker,payload)
                pending[future]=dict(record=record,device=used_device,submitted_epoch=time.time(),region_root=str(regional_root))
                if pilot:pilot_attempts[record['key']]=pilot_attempts.get(record['key'],0)+1
            if not pending:break
            done,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
            if not done:continue
            for future in done:
                info=pending.pop(future);record=info['record']
                try:state=future.result()
                except Exception as exc:
                    state=dict(states.get(record['key'],{}),status='WORKER_FAILED',visits=states.get(record['key'],{}).get('visits',0)+1,
                        last_visit_epoch=time.time(),error_type=type(exc).__name__,error=str(exc))
                    _new_json(root/'worker_failures'/f'{time.time_ns()}.json',dict(info,error_type=type(exc).__name__,error=str(exc)))
                state=recovered_current_data_state(state,info['region_root'])
                states[record['key']]=state;controller['slices']+=1
                print(f'TRAIN {record["key"]} visit={state["visits"]} qualified={qualified(state)}',flush=True)
            if pilot:controller['pilot_progress']=pilot_progress(regions,states,pilot_attempts)
            write_json(state_path,controller)
    except MemoryError as exc:
        resource_exception=exc
        controller.update(status='RESOURCE_PAUSING',stop_reason='PARENT_MEMORY_GUARD',resource_error=str(exc))
        raise
    except TimeoutError:
        controller["stop_reason"]="SCIENCE_DEADLINE"
    finally:
        owned_pids=[p.pid for p in getattr(pool,'_processes',{}).values() if p.pid is not None]
        cleanup_mode='TERMINATED_OWNED_POOL' if pending else 'JOINED_OWNED_POOL'
        if pending:
            _stop_own_pool(pool)
            for info in pending.values():
                regional_state=Path(info['region_root'])/'STATE.json'
                if regional_state.exists():
                    states[info['record']['key']]=recovered_current_data_state(read_json(regional_state),info['region_root'])
                _new_json(root/'worker_interruptions'/f'{time.time_ns()}.json',dict(info,
                    reason=controller.get('stop_reason','PARENT_GUARD_OR_EXCEPTION'),epoch=time.time(),
                    persisted_state_recovered=regional_state.exists(),all_completed_artifacts_preserved=True))
        else:pool.shutdown(wait=True,cancel_futures=True)
        ctx._training_owned_pool_active=False
        write_json(state_path,controller);ctx.flush_bindings()
        if resource_exception is not None:
            resource_exception.pacnhn_owned_cleanup=dict(completed=True,completed_epoch=time.time(),mode=cleanup_mode,owned_pids=owned_pids)
            resource_exception.pacnhn_source_intent=str(intent_path)
    if pause_request is not None:
        return pause_without_finalization(root,controller,clock,pause_request,intent_path)
    if pilot:
        progress=pilot_progress(regions,states,pilot_attempts)
        result=dict(status="PILOT_PASS" if progress['all_required_complete'] else "PILOT_INCOMPLETE",
            regions=progress['regions_attempted'],states=states,**progress,
            completion_criterion='All initial fixed roles admitted or explicitly attempt-exhausted, plus both families have immutable selected checkpoints audited on current nonempty TRAIN and VALID; usable counts and target attainment reported separately',
            TEST_opened=False,clock=clock,deadline_epoch=deadline,pilot_output=str(root))
        write_json(root/"PILOT_RESULT.json",result);return result
    ensure_finalization_clock(root,clock)
    states=reassess_selected_before_freeze(ctx,root,regions,states,intent['support_freeze_sha256'],config,deadline)
    registry=freeze_registry(ctx,root,regions,states,intent["support_freeze_sha256"],config,deadline=deadline)
    return complete_frozen_training(ctx,root,registry,clock,controller=controller,source_intent=str(intent_path))


def conditioning_probe(ctx,training_config_path,*,source_pilot_id,probe_id,region_keys,seconds):
    """Explicit fresh-fit repair evidence on authenticated pilot TRAIN/VALID only.

    No labels are generated, no previous optimizer is imported and the initial
    zero-update model is a diagnostic only, never an eligible trained success.
    """
    from .gates import require_component,component_snapshot
    from threadpoolctl import threadpool_limits
    config=read_training_config(training_config_path)
    if config.get('icnn_coefficient_parameterization')!=ICNN_SCALED or config.get('xgb_log_center_policy')!=XGB_CENTERED:
        raise ValueError('Conditioning probe requires explicit v2 ICNN and XGBoost policies')
    if (not probe_id or not source_pilot_id or any(Path(k).name!=k for k in (probe_id,source_pilot_id))
            or not 1<=len(region_keys)<=3 or len(set(region_keys))!=len(region_keys)
            or not math.isfinite(seconds) or not 0<seconds<=config['conditioning_probe_seconds']):
        raise ValueError('Explicit safe probe/source ids, 1-3 regions and bounded seconds required')
    clock_path=ctx.root/'training/TRAINING_CLOCK.json'
    if not clock_path.exists():raise ValueError('Repair cannot create/reset the Stage2 clock')
    clock=stage_clock(ctx,config);clock_sha=sha(clock_path)
    deadline=min(time.time()+seconds,clock['stage_deadline_epoch'])
    ctx.guard(deadline=deadline)
    require_component(ctx,'training',extra_config=training_config_path)
    _,regions=gate_inputs(ctx);freeze_xgb_memory_budget(ctx,regions,config)
    inventory={r['key']:r for r in regions}
    if not set(region_keys)<=set(inventory):raise ValueError('Unknown frozen support region')
    source=ctx.root/'training/pilots'/source_pilot_id
    source_result=read_json(source/'PILOT_RESULT.json')
    if source_result.get('status')!='PILOT_PASS' or source_result.get('TEST_opened') is not False:
        raise ValueError('Completed, blind original pilot required')
    root=ctx.root/'training/conditioning_probes'/probe_id;root.mkdir(parents=True,exist_ok=False)
    before={}
    def bind(path,role,expected=None):
        digest=ctx.bind(path,role,expected);before[str(Path(path))]=digest;return digest
    bind(training_config_path,'CONDITIONING_V2_CONFIGURATION')
    bind(clock_path,'ORIGINAL_STAGE2_CLOCK',clock_sha)
    bind(source/'PILOT_RESULT.json','IMMUTABLE_ORIGINAL_PILOT')
    audit_manifest=ctx.root/'audits/training_pilot_v1/ARTIFACT_MANIFEST.json'
    if audit_manifest.exists():bind(audit_manifest,'ORIGINAL_PILOT_READINESS_AUDIT')
    intent=dict(phase_id=ctx.c['phase_id'],step_id='training',action='conditioning-probe',run_id=ctx.c['run_id'],
        probe_id=probe_id,source_pilot_id=source_pilot_id,regions=list(region_keys),
        tasks=['AUTHENTICATE_TRAIN_VALID','FRESH_CALIBRATED_ICNN_INITIALIZATION','BOUNDED_ICNN_UPDATES',
               'FRESH_CENTERED_XGBOOST_BLOCK','NATIVE_COST_AND_PORTABLE_AUDITS'],
        expected_outputs=['RESULT.json','INPUT_BINDINGS.json','regions/*/INITIALIZATION.json','regions/*/RESULT.json'],
        acceptance_tests=['same_native_cost_model','initial_increment_bound','finite_positive_native_outputs',
            'c0_and_zero_anchor','portable_prediction_reproduction','current_full_TRAIN_fixed_VALID','no_TEST_opening'],
        config_path=str(training_config_path),config_sha256=sha(training_config_path),
        source_snapshot=component_snapshot(ctx,'training'),started_epoch=time.time(),deadline_epoch=deadline,
        original_training_clock=clock,resources=ctx.c['resources'],old_optimizer_reuse=False,
        TEST_opened=False,initialization_is_eligible=False)
    _new_json(root/'INTENT.json',intent)
    rows=[]
    def check():ctx.guard(deadline=deadline)
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            for key in region_keys:
                check();record=inventory[key];tick=time.time()
                previous=source/'regions'/record['suite_id']/f'{record["region_index"]:06d}'
                target=root/'regions'/record['suite_id']/f'{record["region_index"]:06d}';target.mkdir(parents=True)
                state=_state(target/'STATE.json')
                try:
                    bind(record['path'],'EXACT_CURRENT_REGIONAL_GRAPH',record['sha256'])
                    region=load_region_graph(record['path'],expected_sha256=record['sha256'])
                    for name in ('NATIVE_SCALE.json','STATE.json','labels/REGION_IDENTITY.json','labels/REQUEST_INDEX.json'):
                        bind(previous/name,'IMMUTABLE_ORIGINAL_PILOT_METADATA')
                    index=read_json(previous/'labels/REQUEST_INDEX.json')
                    for entry in index.values():
                        if entry['role'] not in {'TRAIN','VALID'} or entry['selected_label'] is None:continue
                        selected=entry['selected_label']
                        bind(entry['input_file'],'FROZEN_'+entry['role']+'_INPUT',entry['input_file_sha256'])
                        bind(entry['request_metadata'],'FROZEN_'+entry['role']+'_ROLE',entry['request_metadata_sha256'])
                        bind(selected['path'],'FROZEN_'+entry['role']+'_LABEL',selected['sha256'])
                        bind(selected['receipt'],'FROZEN_'+entry['role']+'_RECEIPT',selected['receipt_sha256'])
                    _new_json(target/'INPUT_BINDINGS.json',before)
                    train=load_admitted_dataset(previous/'labels',role='TRAIN')
                    valid=load_admitted_dataset(previous/'labels',role='VALID')
                    source_state=read_json(previous/'STATE.json');scale_record=read_json(previous/'NATIVE_SCALE.json')
                    if (train['dataset_identity']!=source_state['dataset_identity']
                            or valid['dataset_identity']!=source_state['valid_dataset_identity']
                            or train['unique_rows']!=config['initial_train'] or valid['unique_rows']!=config['fixed_valid']
                            or scale_record['graph_sha256']!=region.graph_sha256):
                        raise ValueError('Exact original full TRAIN/fixed VALID or scale identity mismatch')
                    for data in (train,valid):
                        if (data['identity']['graph_sha256']!=region.graph_sha256
                                or data['identity']['graph_identity']!=region.identity
                                or data['identity']['support_identity']!=region.support_identity
                                or not np.array_equal(data['c0'],region.c0)):
                            raise ValueError('Pilot labels differ from frozen native graph/support/c0')
                    scale=np.asarray(scale_record['scale'],np.float64)
                    state.update(dataset_identity=train['dataset_identity'],valid_dataset_identity=valid['dataset_identity'],
                        training_rows=train['unique_rows'],valid_rows=valid['unique_rows'])
                    power=fit_power4_cost(train['u'],train['tau'],train['c0'],scale)
                    baseline=dict(TRAIN=_evaluate(power,train,config['audit_batch_rows'],check),
                        VALID=_evaluate(power,valid,config['audit_batch_rows'],check))
                    initial=_initialize_icnn(train,scale,config,'cpu',config['icnn_widths'][0])
                    initial_cost=initial.cost(train['u']);native=initial.export_model()
                    np.testing.assert_allclose(native.cost(train['u']),initial_cost,rtol=2e-12,atol=1e-10)
                    np.testing.assert_array_equal(native.cost(np.zeros(region.m)),region.c0)
                    if native.value_cost(np.zeros(region.m))[0]!=0:raise ValueError('Initial ICNN zero anchor failed')
                    initial_receipt=save_model(initial,target/'initialization.npz')
                    loaded=load_model(initial_receipt['path'])
                    np.testing.assert_array_equal(loaded.cost(train['u']),native.cost(train['u']))
                    initialization=dict(total_updates=0,eligible_for_trained_selection=False,model=initial_receipt,
                        TRAIN=_evaluate(native,train,config['audit_batch_rows'],check),
                        VALID=_evaluate(native,valid,config['audit_batch_rows'],check),
                        TRAIN_calibration=initial.calibration,coefficient_units_identity=array_hash({
                            'eta_units':initial.eta_units.cpu().numpy(),'amplitude_unit':initial.amplitude_unit.cpu().numpy()}))
                    if initialization['TRAIN']['WMAPE_percent']>baseline['TRAIN']['WMAPE_percent']+100*config['icnn_initial_neural_cost_share']+1e-8:
                        raise ValueError('Declared TRAIN initial neural cost-share bound failed')
                    _new_json(target/'INITIALIZATION.json',initialization)
                    timings=[]
                    for _ in range(config['conditioning_probe_icnn_blocks']):
                        check();timings.append(dict(family='ICNN',**_icnn_block(target,state,train,scale,config,'cpu',deadline,check)))
                        audit_and_select(target,'ICNN',state,train,valid,config,check)
                        write_json(target/'STATE.json',state)
                    limited=dict(config,xgb_max_projected_model_bytes=record['xgb_raw_budget_bytes'])
                    for _ in range(config['conditioning_probe_xgb_blocks']):
                        check();timings.append(dict(family='XGBOOST',**_xgb_block(target,state,train,scale,region,limited,'cpu',
                            min(deadline,time.time()+config['conditioning_probe_xgb_seconds_per_region']),check)))
                        audit_and_select(target,'XGBOOST',state,train,valid,limited,check)
                        write_json(target/'STATE.json',state)
                    if not all(f in state['selected'] for f in FAMILIES):raise ValueError('Probe has a missing trained family')
                    checks=[]
                    for family in FAMILIES:
                        candidate=state['selected'][family];model=load_model(candidate['model_path'])
                        prediction=model.cost(train['u'])
                        if not np.isfinite(prediction).all() or np.any(prediction<0):raise ValueError('Invalid portable costs')
                        check_path=target/(family+'_roundtrip.npz');save_model(model,check_path)
                        np.testing.assert_array_equal(load_model(check_path).cost(train['u']),prediction)
                        checks.append(dict(family=family,prediction_sha256=array_hash({'cost':prediction}),
                            portable_roundtrip_exact=True,minimum_native_cost=float(prediction.min()),
                            roundtrip_path=str(check_path),roundtrip_sha256=sha(check_path)))
                    row=dict(key=key,status='PASS',TRAIN_rows=train['unique_rows'],VALID_rows=valid['unique_rows'],
                        power4=baseline,initialization=initialization,selected=state['selected'],
                        all_trained_checkpoint_audits=[r['audit'] for family in FAMILIES for r in state['eligible'][family]],
                        timings=timings,portable_checks=checks,TEST_opened=False,elapsed_seconds=time.time()-tick,
                        train_dataset_identity=train['dataset_identity'],valid_dataset_identity=valid['dataset_identity'])
                except Exception as exc:
                    row=dict(key=key,status='FAILED',error_type=type(exc).__name__,error=str(exc),
                        traceback=traceback.format_exc(),TEST_opened=False,elapsed_seconds=time.time()-tick)
                finally:write_json(target/'STATE.json',state)
                _new_json(target/'RESULT.json',row);rows.append(row)
                print('CONDITIONING '+key+' '+row['status'],flush=True)
    finally:
        source_stable=all(sha(path)==digest for path,digest in before.items())
        code_stable=component_snapshot(ctx,'training')==intent['source_snapshot']
        config_stable=sha(training_config_path)==intent['config_sha256']
        _new_json(root/'INPUT_BINDINGS.json',before);ctx.flush_bindings()
        result=dict(status='PASS' if len(rows)==len(region_keys) and all(r['status']=='PASS' for r in rows) and source_stable and code_stable and config_stable else 'INCOMPLETE',
            rows=rows,TEST_opened=False,source_bindings_unchanged=source_stable,
            source_snapshot_unchanged=code_stable,configuration_unchanged=config_stable,
            original_training_clock_sha256=clock_sha,started_epoch=intent['started_epoch'],ended_epoch=time.time(),
            deadline_epoch=deadline,accuracy_target_success_is_separate=True,intent_sha256=sha(root/'INTENT.json'))
        _new_json(root/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--training-config",required=True)
    parser.add_argument("--action",choices=("train","pilot","conditioning-probe"),required=True)
    parser.add_argument("--region",action="append",default=[])
    parser.add_argument("--pilot-id")
    parser.add_argument("--pilot-seconds",type=float,default=600.)
    parser.add_argument('--source-pilot-id')
    args=parser.parse_args()
    ctx=Context(args.config,step_id="training")
    if args.action=='conditioning-probe':
        result=conditioning_probe(ctx,args.training_config,source_pilot_id=args.source_pilot_id,
            probe_id=args.pilot_id,region_keys=args.region,seconds=args.pilot_seconds)
        if result['status']!='PASS':raise RuntimeError('Conditioning probe incomplete; see preserved exact evidence')
    else:
        run(ctx,args.training_config,pilot=args.action=="pilot",pilot_regions=args.region,
            pilot_seconds=args.pilot_seconds,pilot_id=args.pilot_id)


if __name__=="__main__":
    main()
