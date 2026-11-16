"""Measured process-tree guards and periodic actual Python stacks; no source walks."""
import os,time,threading,faulthandler,shutil,traceback,argparse
from pathlib import Path
import psutil
from .common import root,rd,read_json,write_json,append_json,elapsed,native_threads,sha,is_e2
from partC_rev1_src.phase6.stack_sampler import StackSampler

STACK_FILES=[]
STACK_SAMPLERS=[]

def start_stack_sampler(stream,interval=300):
    sampler=StackSampler(stream,interval=interval)
    sampler.sample();sampler.__enter__();STACK_SAMPLERS.append(sampler)
    return sampler
def worker_setup(c,role):
    native_threads()
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    p=root(c)/'control/stacks'/f'{role}_{os.getpid()}.txt';p.parent.mkdir(parents=True,exist_ok=True)
    f=p.open('a',encoding='utf-8');STACK_FILES.append(f);faulthandler.enable(f)
    # The native timed watchdog caused access violations while walking live
    # PyTorch frames on this host. Reuse the already implemented Python sampler.
    start_stack_sampler(f,interval=300)
    from . import shared_io
    assert read_json is shared_io.read_json and write_json is shared_io.write_json
    write_json(root(c)/'control/worker_runtime'/f'{os.getpid()}.json',dict(pid=os.getpid(),unix=time.time(),
        native_threads=1,periodic_stack_sampler='partC_rev1_src.phase6.stack_sampler.StackSampler',stack_seconds=300,
        stack_sampler_source_sha256=sha(Path(__file__).parent.parent/'phase6/stack_sampler.py'),native_timeout_watchdog=False,
        read_callable_module=read_json.__module__,write_callable_module=write_json.__module__,shared_io_sha256=sha(shared_io.__file__)))

def owned_bytes(c):
    # Only this new output and named patch evidence/code, never old result trees.
    repo=Path(c['execution']['repository_root']);prefixes=[root(c),repo/'partC_rev1_src/phase67',repo/'partC_doc_icnn/phase_7_work/p67_patch24h_0908',repo/c['execution']['authority_install_relative'],
        repo/'partC_doc_icnn/handoffs/phase_07/step_p67_patch24h_0908']
    if is_e2(c):
        prefixes=[root(c),repo/'partC_rev1_src/phase67',repo/('partC_doc_icnn/phase_7_work/'+c['step_id']),
                  repo/c['execution']['authority_install_relative'],repo/('partC_doc_icnn/handoffs/phase_07/step_'+c['step_id'])]
    total=c.get('parent_added_bytes_at_seal',0) if is_e2(c) else 0;files=0
    for prefix in prefixes:
        for base,dirs,names in os.walk(prefix):
            for name in names:
                try:total+=os.stat(os.path.join(base,name)).st_size;files+=1
                except FileNotFoundError:pass
    for p in (repo/'artifacts_icnn/handoff_bundles/phase_07').glob('phase07_step'+c['step_id']+'_*'):
        if p.is_file():total+=p.stat().st_size;files+=1
    for p in (repo/'tests').glob('test_phase67_*.py'):
        total+=p.stat().st_size;files+=1
    return total,files

class Monitor:
    def __init__(self,c,role):
        self.c=c;self.role=role;self.stop=threading.Event();self.peak=0;self.last_disk=0.;self.added=0;self.files=0
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def start(self):self.thread.start();return self
    def loop(self):
        while not self.stop.is_set():
            try:
                now=time.time();p=psutil.Process();procs=[p]+p.children(recursive=True);rss=0;cpu=0.
                for proc in procs:
                    try:rss+=proc.memory_info().rss;ct=proc.cpu_times();cpu+=ct.user+ct.system
                    except psutil.Error:pass
                self.peak=max(self.peak,rss)
                if now-self.last_disk>=300:self.added,self.files=owned_bytes(self.c);self.last_disk=now
                free=shutil.disk_usage(root(self.c)).free
                row=dict(unix=now,elapsed_seconds=elapsed(self.c),role=self.role,coordinator_pid=os.getpid(),pids=[p.pid for p in procs],
                    rss_bytes=rss,peak_rss_bytes=self.peak,process_tree_cpu_seconds=cpu,added_bytes=self.added,owned_files=self.files,free_bytes=free,
                    stop_admission=rss>=.8*self.c['execution']['rss_bytes'])
                append_json(root(self.c)/'control/resource_history.jsonl',row);write_json(root(self.c)/'control/heartbeat.json',row)
                if self.added>=self.c['execution']['added_disk_bytes']-2*1024**3:
                    write_json(root(self.c)/'control/STOP_GENERATION.json',dict(reason='ADDED_DATA_DISK_LIMIT_KEEP_TRAINING_VALID_DATA',**row))
                if rss>=self.c['execution']['rss_bytes'] or self.added>=self.c['execution']['added_disk_bytes'] or free<2*1024**3:
                    target=(root(self.c)/'control/stop_requests'/f'monitor_{os.getpid()}_{time.time_ns()}.json') if is_e2(self.c) else root(self.c)/'control/STOP_SCIENCE.json'
                    write_json(target,dict(reason='MEASURED_RESOURCE_GUARD',**row))
            except Exception:append_json(root(self.c)/'control/resource_errors.jsonl',dict(unix=time.time(),traceback=traceback.format_exc()))
            self.stop.wait(30)
    def close(self):self.stop.set();self.thread.join(timeout=5)

def guardian(c,pid):
    from . import shared_io
    assert read_json is shared_io.read_json and write_json is shared_io.write_json
    write_json(root(c)/'control/guardian_runtime.json',dict(pid=os.getpid(),role='guardian',unix=time.time(),
        read_callable_module=read_json.__module__,write_callable_module=write_json.__module__,shared_io_sha256=sha(shared_io.__file__)))
    identity=psutil.Process(pid).create_time()
    while psutil.pid_exists(pid):
        try:
            p=psutil.Process(pid)
            if p.create_time()!=identity:break
            procs=[p]+[q for q in p.children(recursive=True) if q.pid!=os.getpid()];rss=0
            for q in procs:
                try:rss+=q.memory_info().rss
                except psutil.Error:pass
            r=dict(pid=pid,unix=time.time(),rss_bytes=rss,elapsed_seconds=elapsed(c),free_bytes=shutil.disk_usage(root(c)).free)
            write_json(root(c)/'control/guardian_heartbeat.json',r)
            if rss>=c['execution']['rss_bytes'] or r['free_bytes']<1024**3 or elapsed(c)>=86340:
                write_json(root(c)/'control/GUARDIAN_FORCED_STOP.json',r)
                for q in reversed(procs):
                    try:q.terminate()
                    except psutil.Error:pass
                break
        except psutil.NoSuchProcess:break
        time.sleep(30)

if __name__=='__main__':
    from .common import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args();guardian(load_config(a.config),a.pid)
