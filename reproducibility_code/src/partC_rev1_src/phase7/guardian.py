"""Portable external process-tree resource guard for the original campaign clock."""
import argparse,os,shutil,time,traceback
from pathlib import Path
import psutil
from .common import load_config,root,read_json,write_json,append_json,elapsed

def tree_bytes(path):
    total=0
    for p in Path(path).rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass # A concurrent atomic rename is already accounted by its destination.
    return total

def resource_sample(c,pid):
    process=psutil.Process(pid);procs=[process]+process.children(recursive=True);rss=0
    for p in procs:
        try:rss+=p.memory_info().rss
        except psutil.Error:pass
    added=tree_bytes(root(c));free=shutil.disk_usage(root(c)).free
    stop=rss>=c['execution']['rss_bytes'] or added>=c['execution']['disk_new_bytes']-c['execution']['disk_seal_reserve_bytes'] or free<c['execution']['disk_seal_reserve_bytes']
    return dict(unix=time.time(),elapsed_seconds=elapsed(c),rss_bytes=rss,added_output_bytes=added,free_disk_bytes=free,resource_stop=bool(stop),pids=[p.pid for p in procs])

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--pid',type=int,required=True);args=a.parse_args();c=load_config(args.config)
    identity=psutil.Process(args.pid).create_time();peak=0
    while psutil.pid_exists(args.pid):
        try:
            if psutil.Process(args.pid).create_time()!=identity:break
            r=resource_sample(c,args.pid);peak=max(peak,r['rss_bytes']);r.update(peak_rss_bytes=peak,guardian_pid=os.getpid(),method='stdlib_disk_usage_and_psutil_process_tree')
            append_json(root(c)/'guardian_resource_timeline.jsonl',r);write_json(root(c)/'guardian_heartbeat.json',r)
            if r['resource_stop'] or elapsed(c)>=28740:
                write_json(root(c)/'GUARDIAN_FORCED_STOP.json',r)
                for pid in reversed(r['pids']):
                    try:psutil.Process(pid).terminate()
                    except psutil.Error:pass
                break
        except psutil.NoSuchProcess:break
        except Exception:append_json(root(c)/'logs/guardian_errors.jsonl',dict(unix=time.time(),traceback=traceback.format_exc()))
        time.sleep(30)
    write_json(root(c)/'GUARDIAN_EXIT.json',dict(unix=time.time(),peak_rss_bytes=peak,elapsed_seconds=elapsed(c)))
if __name__=='__main__':main()
