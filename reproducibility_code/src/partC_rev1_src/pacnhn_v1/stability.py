"""Read-only certified-state sampling; never runs or alters an assignment."""
from pathlib import Path
import time
import numpy as np
from .control import read_json,write_json,save_npz,array_hash,sha


def capture(ctx,scenario, *,minimum_interval_seconds=300):
    if scenario not in {'open','i95'}:raise ValueError('Explicit reference scenario required')
    source=ctx.root/'references'/scenario
    out=ctx.root/'audits/reference_flow_stability'/scenario
    receipts=sorted(out.glob('state_*.json')) if out.exists() else []
    previous=read_json(receipts[-1]) if receipts else None
    if previous and time.time()-previous['captured_epoch']<minimum_interval_seconds:
        return dict(status='SAMPLING_INTERVAL_NOT_REACHED',last=previous['iteration'])
    # Atomic files can still describe adjacent updates. Accept only an exact
    # flow/certificate/history match, otherwise try again on the next call.
    with np.load(source/'certified_flow.npz',allow_pickle=False) as z:flow=z['flow']
    digest=array_hash({'flow':flow});certificate=read_json(source/'CERTIFICATE.json')
    if not certificate.get('fresh') or certificate['state_hash']!=digest:
        return dict(status='ADJACENT_ATOMIC_UPDATES_RETRY')
    matches=[r for r in read_json(source/'HISTORY.json') if r['state_hash']==digest]
    if not matches:return dict(status='HISTORY_WRITE_PENDING_RETRY')
    row=matches[-1];iteration=row['iteration']
    path=out/f'state_{iteration:06d}_{digest[:12]}.json'
    if path.exists():return dict(status='ALREADY_CAPTURED',iteration=iteration)
    vector=path.with_suffix('.npz');out.mkdir(parents=True,exist_ok=True)
    save_npz(vector,flow=flow)
    receipt=dict(status='CERTIFIED_SNAPSHOT',scenario=scenario,iteration=iteration,
        elapsed_seconds=row['elapsed_seconds'],captured_epoch=time.time(),
        certificate=certificate,flow_path=str(vector),flow_file_sha256=sha(vector),
        source_directory=str(source),sampling_role='READ_ONLY_NO_ORACLE_OR_ASSIGNMENT_UPDATE')
    write_json(path,receipt);return receipt


def summarize(ctx,scenario):
    source=ctx.root/'references'/scenario;result=read_json(source/'RESULT.json')
    with np.load(source/'final_flow.npz',allow_pickle=False) as z:final=z['flow']
    if not result['final_certificate_fresh'] or array_hash({'flow':final})!=result['certificate']['state_hash']:
        raise ValueError('Final reference is not freshly certified')
    out=ctx.root/'audits/reference_flow_stability'/scenario;rows=[]
    bcount=ctx.c['topology']['expected_backbone_count']
    for path in sorted(out.glob('state_*.json')):
        receipt=read_json(path)
        if sha(receipt['flow_path'])!=receipt['flow_file_sha256']:raise ValueError('Stability snapshot changed')
        with np.load(receipt['flow_path'],allow_pickle=False) as z:earlier=z['flow']
        if earlier.shape!=final.shape or array_hash({'flow':earlier})!=receipt['certificate']['state_hash']:
            raise ValueError('Stability flow identity differs')
        for population,count in [('BACKBONE',bcount),('COMPLETE_P',len(final))]:
            a,b=earlier[:count],final[:count];den=float(b.sum());absolute=float(abs(a-b).sum())
            rows.append(dict(earlier_iteration=receipt['iteration'],final_iteration=result['reported_state_iteration'],
                comparison_seconds=result['elapsed_seconds']-receipt['elapsed_seconds'],population=population,
                final_flow_sum=den,absolute_flow_change_sum=absolute,
                change_percent=100*absolute/den if den else None,
                earlier_RG_SPTT=receipt['certificate']['RG_SPTT'],final_RG_SPTT=result['certificate']['RG_SPTT'],
                snapshot_sha256=sha(path)))
    report=dict(status='MEASURED' if rows else 'NO_EARLIER_CERTIFIED_SNAPSHOT',scenario=scenario,
        reference_status=result['status'],reference_result_sha256=sha(source/'RESULT.json'),comparisons=rows,
        interpretation='Change between identified certified states; not a bound on equilibrium flow error or proof of convergence')
    write_json(out/'SUMMARY.json',report);return report
