"""Single owned numerical child; original-clock wall/RSS guards and exit receipts."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import psutil
from partC_rev1_src.phase_n8.common import read_json,write_json,sha


def job_deadline(c, root, task, start, cap):
    """Enforce the shared learning cap even if a child misses its guards."""
    cutoff=c['deadline_epoch'] if task.startswith(('F','H','REPORT','VERIFY')) else c['compute_cutoff_epoch']
    deadline=min(start+cap,cutoff);binding=None
    if task=='D1_LEARNING_STUDY':
        clock_path=root/'learning/LEARNING_CLOCK.json'
        prep_start_path=root/'jobs/D00_LEARNING_PREPARE/START.json'
        prep_exit_path=root/'jobs/D00_LEARNING_PREPARE/EXIT.json'
        clock=read_json(clock_path);began=read_json(prep_start_path);ended=read_json(prep_exit_path)
        if (began.get('task'),began.get('run_id'),began.get('job_id'))!=('D0_LEARNING_PREPARE',c['run_id'],'D00_LEARNING_PREPARE'):
            raise ValueError('Learning preparation receipt identity differs')
        if ended.get('exit_code')!=0 or ended.get('supervisor_stop') is not None:
            raise ValueError('Learning preparation did not finish with a usable supervised handoff')
        if ended.get('start_epoch')!=began['start_epoch'] or ended.get('job_id')!=began['job_id']:
            raise ValueError('Learning preparation start/exit receipts disagree')
        if clock['cap_seconds']!=14400 or not c['start_epoch']<=began['start_epoch']<=clock['stage_start_epoch']<=ended['ended_epoch']<=start:
            raise ValueError('Invalid original learning-stage start')
        if clock['stage_deadline_epoch']!=min(clock['stage_start_epoch']+14400,c['compute_cutoff_epoch']):
            raise ValueError('Learning-stage deadline changed or reset')
        deadline=min(deadline,clock['stage_deadline_epoch'],began['start_epoch']+14400)
        binding=dict(clock_sha256=sha(clock_path),preparation_start_sha256=sha(prep_start_path),
            preparation_exit_sha256=sha(prep_exit_path),complete_stage_start_epoch=began['start_epoch'],
            hard_stage_deadline_epoch=deadline,cap_seconds=14400,
            scope='Entire supervised D00 plus D01, including startup and launch gap')
    return deadline,binding


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','task','job-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--cap-seconds',type=float,required=True)
    p.add_argument('--scenario')
    args=p.parse_args();c=read_json(args.config);root=Path(c['output_root'])
    if args.run_id!=c['run_id'] or args.cap_seconds<=0:raise ValueError('Explicit run/budget required')
    matrix_path=root/'run_matrix.csv'
    if not matrix_path.exists():raise ValueError('Explicit current run matrix is required')
    with matrix_path.open(encoding='utf-8',newline='') as matrix_file:matrix=list(csv.DictReader(matrix_file))
    matches=[row for row in matrix if row['job_id']==args.job_id]
    if len(matches)!=1:raise ValueError('Job absent or duplicated in current explicit matrix')
    task=matches[0]
    if (task['task'],task['run_id'],float(task['cap_seconds']),task['scenario'])!=(args.task,args.run_id,args.cap_seconds,args.scenario or ''):
        raise ValueError('Task/scenario/budget differs from explicit matrix')
    if not task['expected_outputs'] or not task['acceptance_policy']:raise ValueError('Job has no concrete expected outputs or acceptance policy')
    start=time.time();deadline,learning_clock_binding=job_deadline(c,root,args.task,start,args.cap_seconds)
    if deadline<=start:raise TimeoutError('Original clock exhausted')
    lock=root/'ACTIVE_SCIENCE_CHILD.json'
    if lock.exists():
        prior=read_json(lock)
        for key in ('supervisor_pid','child_pid'):
            if prior.get(key) and psutil.pid_exists(prior[key]):raise RuntimeError('Existing owned job still active')
        raise RuntimeError('Stale receipt requires explicit reconciliation, not silent replacement')
    job=root/'jobs'/args.job_id;job.mkdir(parents=True,exist_ok=False)
    # Keep exact scoped code bytes as well as hashes before launching the child.
    # Earlier jobs retain their original PRE_RUN_FREEZE code-member manifests.
    snapshot=[];repository=Path(c['repository'])
    for namespace in ('phase_n8','phase_n8_r4'):
        for source in sorted((repository/'partC_rev1_src'/namespace).rglob('*.py')):
            relative=source.relative_to(repository);destination=job/'code_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            snapshot.append(dict(relative_path=relative.as_posix(),sha256=sha(destination),size_bytes=destination.stat().st_size))
    write_json(job/'CODE_SNAPSHOT.json',dict(status='COPIED_BEFORE_CHILD_LAUNCH',initial_HEAD=c['initial_head'],
        unchanged_other_local_dependencies='Recorded repository commit plus final dependency snapshot; no source edits authorized',files=snapshot))
    command=[c['python_executable'],'-X','utf8','-B','-m','partC_rev1_src.phase_n8_r4.cli',
        '--config',str(Path(args.config).resolve()),'--run-id',args.run_id,'--task',args.task,'--job-id',args.job_id,
        '--job-start',str(start),'--job-deadline',str(deadline)]
    if args.scenario:command+=['--scenario',args.scenario]
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key]='1'
    env.update(PYTHONHASHSEED='0',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_NUM_THREADS='1',
        NUMBA_CACHE_DIR=str(root/'cache/numba'))
    record=dict(supervisor_pid=os.getpid(),child_pid=None,task=args.task,job_id=args.job_id,run_id=args.run_id,
        command=command,config_sha256=sha(args.config),start_epoch=start,deadline_epoch=deadline,
        original_R4_start_epoch=c['start_epoch'],numeric_workers=1,source_mutation=False)
    record['job_code_snapshot_manifest_sha256']=sha(job/'CODE_SNAPSHOT.json')
    if learning_clock_binding is not None:record['shared_learning_clock']=learning_clock_binding
    record['run_matrix_sha256']=sha(matrix_path);record['explicit_task']=task
    with lock.open('x',encoding='utf-8') as f:json.dump(record,f)
    child=None;reason=None;peak=0
    try:
        with (job/'stdout.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=c['repository'],env=env,stdout=log,stderr=subprocess.STDOUT)
            record['child_pid']=child.pid;write_json(lock,record);write_json(job/'START.json',record)
            print(json.dumps(record),flush=True)
            last_report=0
            while child.poll() is None:
                now=time.time()
                try:
                    cp=psutil.Process(child.pid);tree=[cp]+cp.children(recursive=True)
                    rss=sum(t.memory_info().rss for t in tree if t.is_running());peak=max(peak,rss)
                except psutil.NoSuchProcess:rss=0
                if now>=deadline:reason='SUPERVISOR_WALL_LIMIT'
                elif rss>=c['max_process_tree_RSS_bytes'] or psutil.virtual_memory().available<c['system_available_floor_bytes']:
                    reason='SUPERVISOR_MEMORY_LIMIT'
                if reason:
                    for t in reversed(tree):
                        try:t.terminate()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=15);break
                if now-last_report>=60:
                    write_json(job/'LIVE.json',dict(epoch=now,elapsed_seconds=now-start,rss_bytes=rss,peak_rss_bytes=peak))
                    print(f'{args.job_id}: active, elapsed={now-start:.1f}s, RSS={rss/1024**3:.2f} GiB',flush=True)
                    last_report=now
                time.sleep(1)
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=15)
        ended=time.time();record.update(ended_epoch=ended,elapsed_seconds=ended-start,peak_process_tree_rss_bytes=peak,
            exit_code=None if child is None else child.returncode,supervisor_stop=reason)
        write_json(job/'EXIT.json',record)
        if lock.exists() and read_json(lock).get('supervisor_pid')==os.getpid():lock.unlink()
    print(json.dumps(record),flush=True)
    return 0 if child is not None and child.returncode==0 and reason is None else 1


if __name__=='__main__':sys.exit(main())
