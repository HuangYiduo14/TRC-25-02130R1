"""Three bounded Windows spawned-process trials against production-imported I/O."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

from . import common, shared_io


def payload(episode, trial, generation):
    body = dict(episode_id=episode, trial=trial, generation=generation,
                values=[generation, generation * 17, generation ^ 0x55AA, 'regional-protocol-' * 64])
    body['payload_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return body


def validate(value, episode, trial):
    expected = payload(episode, trial, value['generation'])
    if value != expected:
        raise AssertionError('Mixed payload, generation checksum or episode identity')


def process(role, index, path, episode, trial, barrier, stop, results, active_seconds):
    import psutil
    source = Path(shared_io.__file__)
    record = dict(role=role, index=index, pid=os.getpid(), creation_time=psutil.Process().create_time(),
        helper_path=str(source), helper_sha256=common.sha(source),
        read_callable_module=common.read_json.__module__, write_callable_module=common.write_json.__module__,
        native_writer='path-gated guarded SetFileInformationByHandle(FileRenameInfoEx); exclusive first publication',
        initial_counters=shared_io.counters(), operations=0, first_operation_unix=None, last_operation_unix=None)
    try:
        assert os.name == 'nt'
        assert common.read_json is shared_io.read_json and common.write_json is shared_io.write_json
        barrier.wait(timeout=30)
        started = time.monotonic()
        last_generation = 0
        while not stop.is_set() and time.monotonic() - started < active_seconds:
            if record['first_operation_unix'] is None:
                record['first_operation_unix'] = time.time()
            if role == 'writer':
                generation = record['operations'] + 1
                common.write_json(path, payload(episode, trial, generation))
                time.sleep(.0005)
            else:
                value = common.read_json(path)
                validate(value, episode, trial)
                assert value['generation'] >= last_generation
                last_generation = value['generation']
            record['operations'] += 1
            record['last_operation_unix'] = time.time()
        record.update(status='PASS',active_seconds=time.monotonic() - started,last_generation_observed=last_generation)
    except BaseException as error:
        stop.set()
        record.update(status='FAIL',error=repr(error),error_type=type(error).__name__,traceback=traceback.format_exc(),
            winerror=getattr(error,'winerror',None),pending_path=getattr(error,'pending_path',None),
            pending_sha256=getattr(error,'pending_sha256',None),io_attempts=getattr(error,'io_attempts',None),
            retry_loop_seconds=getattr(error,'replace_retry_elapsed_seconds',None))
    record['final_counters'] = shared_io.counters()
    results.put(record)


def trial_run(c, directory, trial):
    started = time.monotonic()
    path = directory / 'shared_forecast.json'
    directory.mkdir(parents=True)
    common.write_json(path, payload(c['episode_id'],trial,0))
    assert path.stat().st_dev == Path(c['output_root']).stat().st_dev
    ctx = mp.get_context('spawn')
    barrier, stop, results = ctx.Barrier(6), ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=process,args=(role,index,str(path),c['episode_id'],trial,barrier,stop,results,62.))
                for role,index in [('writer',0)]+[('reader',i) for i in range(4)]]
    records=[]
    timed_out=False
    for child in children:
        child.start()
    try:
        barrier.wait(timeout=30)
        while len(records) < 5 and time.monotonic() - started < 165:
            try:
                records.append(results.get(timeout=.2))
            except queue.Empty:
                if all(not child.is_alive() for child in children):
                    break
        timed_out = len(records) != 5
    except BaseException as error:
        records.append(dict(role='supervisor',status='FAIL',error=repr(error),traceback=traceback.format_exc()))
    finally:
        stop.set()
        for child in children:
            child.join(timeout=max(0.,min(1.,175-(time.monotonic()-started))))
        for child in children:
            if child.is_alive():
                child.terminate()  # Only the exact Process handles spawned above.
        for child in children:
            child.join(timeout=max(0.,178-(time.monotonic()-started)))
    writers=[r for r in records if r['role']=='writer']; readers=[r for r in records if r['role']=='reader']
    overlap=0.
    if len(records)==5 and all(r.get('first_operation_unix') and r.get('last_operation_unix') for r in records):
        overlap=min(r['last_operation_unix'] for r in records)-max(r['first_operation_unix'] for r in records)
    valid=(not timed_out and len(writers)==1 and len(readers)==4 and all(r['status']=='PASS' for r in records)
           and writers[0]['operations']>=1000 and all(r['operations']>=1000 for r in readers)
           and overlap>=60 and all(not child.is_alive() and child.exitcode==0 for child in children))
    final=common.read_json(path)
    validate(final,c['episode_id'],trial)
    valid=valid and final['generation']==writers[0]['operations'] and time.monotonic()-started<=180
    result=dict(status='PASS' if valid else 'FAIL',trial=trial,started_unix=time.time()-(time.monotonic()-started),
        ended_unix=time.time(),wall_seconds=time.monotonic()-started,simultaneous_activity_seconds=overlap,
        required_simultaneous_seconds=60,minimum_operations_per_process=1000,maximum_wall_seconds=180,
        source_sha256=common.sha(shared_io.__file__),records=records,final_generation=final['generation'],
        final_payload_sha256=common.sha(path),timed_out=timed_out,all_owned_children_joined=all(not p.is_alive() for p in children),
        child_exitcodes=[p.exitcode for p in children],scientific_labels_generated=0,optimizer_updates=0)
    common.write_json(directory/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['status','trial','wall_seconds','simultaneous_activity_seconds','final_generation','source_sha256']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--task',required=True,choices=['E1_WINDOWS_PROTOCOL_THREE_TRIALS'])
    args=parser.parse_args()
    c=common.read_json(args.config)
    assert c['episode_id']=='p67_postaudit_e2_0908' and c['repair_epoch']==8 and c['total_repair_ceiling']==8
    assert os.name=='nt' and time.time()+600<c['hard_deadline_unix']
    dest=Path(c['output_root'])/'validation'/('windows_spawn_probe_'+str(time.time_ns()))
    dest.mkdir(parents=True)
    before=common.sha(shared_io.__file__)
    results=[]
    for trial in range(1,4):
        result=trial_run(c,dest/f'trial_{trial:02d}',trial)
        results.append(result)
        if result['status']!='PASS':break
    passed=len(results)==3 and all(r['status']=='PASS' and r['source_sha256']==before for r in results)
    assert common.sha(shared_io.__file__)==before
    summary=dict(status='PASS' if passed else 'FAIL',task=args.task,run_id=c['run_id'],episode_id=c['episode_id'],
        original_start_unix=c['original_start_unix'],original_deadline_unix=c['hard_deadline_unix'],helper_sha256=before,
        trials=[dict(path=str(dest/f'trial_{r["trial"]:02d}'/'RESULT.json'),sha256=common.sha(dest/f'trial_{r["trial"]:02d}'/'RESULT.json'),status=r['status']) for r in results])
    common.write_json(dest/'RESULT.json',summary)
    if passed:common.write_json(Path(c['output_root'])/'validation/WINDOWS_PROTOCOL_3PASS.json',summary)
    print(json.dumps(summary),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    main()
