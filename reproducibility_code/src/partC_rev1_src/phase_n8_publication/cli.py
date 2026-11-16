"""Explicit, supervised publication amendment tasks; never default execution."""
from __future__ import annotations
import argparse
import sys
import traceback
from .context import Context, code_identity
from partC_rev1_src.phase_n8.common import write_json


def unit_tests(ctx):
    import pytest
    import numpy, scipy, pandas, psutil
    paths=[str(ctx.authority/'tests'),str(ctx.repo/'partC_rev1_src/phase_n8/tests'),
           str(ctx.repo/'partC_rev1_src/phase_n8_r4/tests'),str(ctx.repo/'partC_rev1_src/phase_n8_publication/tests')]
    status=pytest.main(paths+['-q','-p','no:cacheprovider','--basetemp='+str(ctx.root/'validation'/'pytest')])
    result=dict(status='PASS' if status==0 else 'FAIL',pytest_exit_code=int(status),
        code_identity=code_identity(ctx.repo)[0],python=sys.version,numpy=numpy.__version__,
        scipy=scipy.__version__,pandas=pandas.__version__,cpu_logical=psutil.cpu_count(),
        cpu_physical=psutil.cpu_count(logical=False),memory_bytes=psutil.virtual_memory().total,
        numeric_workers=1,BLAS_threads=1)
    write_json(ctx.root/'validation/TESTS.json',result)
    if status:raise RuntimeError('Required tests failed')
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','task','job-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--job-start',type=float,required=True);p.add_argument('--job-deadline',type=float,required=True)
    p.add_argument('--scenario');args=p.parse_args()
    ctx=Context(args.config,args.task,args.run_id,deadline=args.job_deadline)
    try:
        if args.task=='TESTS':result=unit_tests(ctx)
        elif args.task=='REFERENCE_AUDIT':
            from .reference import run
            result=run(ctx,args)
        elif args.task=='REGISTRY_PREFLIGHT':
            if args.scenario=='ALL_RECORDED_WITNESSES':
                from .witness_audit import run
                result=run(ctx,args)
            else:
                from .registry import preflight
                result=preflight(ctx,args)
        elif args.task=='ASSIGNMENT':
            from .assignment import run
            result=run(ctx,args)
        elif args.task=='PHYSICAL_CLOSURE':
            from .physical_closure import run
            result=run(ctx,args)
        elif args.task=='REGIONAL_COSTS':
            from .regional_costs import run
            result=run(ctx,args)
        elif args.task=='LEARNING_COMPLETION':
            from .learning_completion import run
            result=run(ctx,args)
        elif args.task=='METRICS':
            from .metrics import run
            result=run(ctx,args)
        elif args.task=='PUBLICATION':
            from .publication_pipeline import build
            result=build(ctx,args)
        elif args.task=='VERIFY':
            from .publication_pipeline import verify
            result=verify(ctx,args)
        elif args.task=='HANDOFF':
            from .handoff import run
            result=run(ctx,args)
        else:raise ValueError('Authorized task has no implemented handler: '+args.task)
        ctx.finish(args.job_id,result)
        return 0
    except Exception as exc:
        ctx.finish(args.job_id,dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        return 1


if __name__=='__main__':sys.exit(main())
