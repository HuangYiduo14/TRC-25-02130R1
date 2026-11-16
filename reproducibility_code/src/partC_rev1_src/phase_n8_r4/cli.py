"""Explicit bounded R4 child tasks. No default scientific execution."""
from __future__ import annotations
import argparse
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from partC_rev1_src.phase_n8.common import read_json,write_json,save_npz,array_hash,sha
from .context import Context,sources,reopen_suite,code_identity


def unit_tests(ctx):
    import pytest
    authority_tests=ctx.authority/'scripts'
    sys.path.insert(0,str(authority_tests))
    paths=[str(authority_tests/name) for name in ('test_safety_bound.py','test_round_clock.py','test_bootstrap_intake.py')]
    paths.append(str(ctx.repo/'partC_rev1_src/phase_n8/tests'))
    status=pytest.main(paths+['-q','-p','no:cacheprovider','--basetemp='+str(ctx.root/'validation'/'local_pytest')])
    import scipy,pandas,psutil,torch
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),
        numpy=np.__version__,scipy=scipy.__version__,pandas=pandas.__version__,torch=torch.__version__,
        torch_cuda_available=torch.cuda.is_available(),torch_cuda_devices=torch.cuda.device_count(),
        cpu_logical_count=psutil.cpu_count(),cpu_physical_count=psutil.cpu_count(logical=False),
        memory_total=psutil.virtual_memory().total,numerical_workers=1,nested_threads=1,gpu_used=False,
        package_installs=0)
    write_json(ctx.root/'receipts'/'R4_ENVIRONMENT.json',env)
    return dict(status='PASS' if status==0 else 'FAIL',pytest_exit_code=int(status),environment=env)


def assignment(ctx,args):
    from .safety import load_safe,SafeComposite
    from partC_rev1_src.phase_n8.solver import solve
    src=sources(ctx);s,od,balance,coverage=reopen_suite(ctx,src,'s03')
    entries,registry=load_safe(ctx,src,s,args.scenario)
    objective=SafeComposite(s,entries,registry,guard=ctx.guard)
    ch,members,inherited=code_identity(ctx.repo)
    identity=dict(graph=s['graph'].identity,demand=od.identity,objective=objective.identity,
        support=s['support_identity'],registry=registry['identity'],code=ch,scenario=args.scenario,solver='FW',tolerance=.001)
    clocks=dict(R4_start_epoch=ctx.c['start_epoch'],R4_deadline_epoch=ctx.c['deadline_epoch'],
        run_start_epoch=args.job_start,run_deadline_epoch=args.job_deadline)
    root=ctx.root/'runs'/args.job_id;root.mkdir(exist_ok=False)
    write_json(root/'PRE_RUN_FREEZE.json',dict(identity=identity,clocks=clocks,coverage=coverage,
        registry_file=str(ctx.root/'registries/bound_safe_v1'/(args.scenario+'.json')),code_members=members,
        inherited_core_code_sha256=inherited,job_id=args.job_id,initialization='COLD_FEASIBLE_AON',workers=1))
    loading=time.time()-args.job_start;begin=time.time()
    result=solve(s['graph'],od,objective,balance,root,identity=identity,clocks=clocks,ctx=ctx,tol=.001)
    result.update(scenario=args.scenario,suite='s03',job_id=args.job_id,coverage=coverage,
        source_load_and_admission_seconds=loading,solver_seconds=time.time()-begin,
        phase_id='N8',step_id='N8R4_NUM',run_id=ctx.c['run_id'],new_mathematical_variant=True)
    write_json(root/'RESULT.json',result);write_json(root/'INFERENCE_TIMINGS.json',objective.timings)
    return dict(status='EXECUTED',result=result)


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','task','job-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--job-start',type=float,required=True);p.add_argument('--job-deadline',type=float,required=True)
    p.add_argument('--scenario');args=p.parse_args()
    ctx=Context(args.config,args.task,args.run_id,deadline=args.job_deadline)
    try:
        if args.task=='B0_UNIT_TESTS':result=unit_tests(ctx)
        elif args.task=='B1_SAFETY_FREEZE':
            from .safety import freeze
            result=freeze(ctx)
        elif args.task=='B2_SAFE_ASSIGNMENT':
            if not args.scenario:raise ValueError('Explicit safe scenario required')
            result=assignment(ctx,args)
        elif args.task=='C0_PPS_FREEZE':
            from .sampling import freeze
            result=freeze(ctx)
        elif args.task=='C1_PHYSICAL_TESTS':
            import pytest
            from .code_dependencies import physical_code_bindings
            tested=physical_code_bindings(ctx.repo)
            (ctx.root/'validation/physical_pytest').mkdir(parents=True,exist_ok=True)
            code=pytest.main([str(ctx.repo/'partC_rev1_src/phase_n8_r4/tests'),'-q','-p','no:cacheprovider',
                '--basetemp='+str(ctx.root/'validation/physical_pytest'/args.job_id)])
            result=dict(status='PASS' if code==0 else 'FAIL',pytest_exit_code=int(code))
            if tested!=physical_code_bindings(ctx.repo):raise ValueError('Physical dependency bytes changed during tests')
            write_json(ctx.root/'validation/CURRENT_PHYSICAL_TESTS.json',dict(**result,job_id=args.job_id,tested_code=tested,
                pytest_scope='All current phase_n8_r4 tests; physical gate binds transitive local dependencies and dynamic fixture'))
        elif args.task=='C2_PHYSICAL_CONTROL':
            from .physical import run
            result=run(ctx,args)
        elif args.task=='C3_PATH_METRICS':
            from .path_metrics import run
            result=run(ctx,args)
        elif args.task=='C0_PHYSICAL_POPULATION_FREEZE':
            from .physical_population import freeze
            result=freeze(ctx,args)
        elif args.task=='C4_PHYSICAL_FLOW_COMPARE':
            from .physical_population import compare
            result=compare(ctx,args)
        elif args.task=='E1_GEOGRAPHY_CLOSURE_AUDIT':
            from .geography import run
            result=run(ctx,args)
        elif args.task=='B4_CURRENT_REGIONAL_METRICS':
            from .regional_metrics import run
            result=run(ctx,args)
        elif args.task=='E0_PARTITION_STABILITY':
            from .stability import run
            result=run(ctx,args)
        elif args.task=='E2_BOUNDARY_PATH_DIAGNOSTICS':
            from .boundary_diagnostics import run
            result=run(ctx,args)
        elif args.task=='E3_OFFLINE_ACCOUNTING':
            from .accounting import run
            result=run(ctx,args)
        elif args.task=='E4_REGIONAL_GEOGRAPHY':
            from .regional_geography import run
            result=run(ctx,args)
        elif args.task=='D0_LEARNING_PREPARE':
            from .learning_data import prepare
            result=prepare(ctx)
        elif args.task=='B3_ORIGINAL_WARM':
            from .warm import run
            result=run(ctx,args)
        elif args.task=='D1_LEARNING_STUDY':
            from .learning import study
            result=study(ctx)
        elif args.task=='F1_BUILD_RELEASE':
            from .release_pipeline import build
            result=build(ctx,args)
        elif args.task=='F2_RENDER_RELEASE':
            from .release_pipeline import render
            result=render(ctx,args)
        elif args.task=='F3_COMPILE_PDF':
            from .release_pipeline import compile_pdf
            result=compile_pdf(ctx,args)
        elif args.task in ('REPORT_PROGRESS','REPORT_FINAL'):
            from .reporting import refresh
            result=refresh(ctx,args,final=args.task=='REPORT_FINAL')
        elif args.task=='REPORT_DOCUMENTATION':
            from .documentation import sync
            result=sync(ctx,args)
        elif args.task=='VERIFY_FINAL_SOURCES':
            from .handoff import source_verification
            result=source_verification(ctx,args)
        elif args.task=='H0_ARTIFACT_INDEX':
            from .handoff import artifact_index
            result=artifact_index(ctx,args)
        elif args.task=='H1_PACKAGE':
            from .handoff import package
            result=package(ctx,args)
        else:raise ValueError('Unknown explicit R4 task: '+args.task)
        ctx.finish(args.job_id,result)
        return 1 if result.get('status')=='FAIL' else 0
    except Exception as exc:
        detail=dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print(detail['traceback'],flush=True);ctx.finish(args.job_id,detail)
        return 1


if __name__=='__main__':sys.exit(main())
