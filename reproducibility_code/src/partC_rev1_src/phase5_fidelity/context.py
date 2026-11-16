"""Explicit task identity, resource supervision and dependency-bound receipts."""
import faulthandler
import json
import os
from pathlib import Path
import threading
import time
import psutil
from partC_rev1_src.phase5.campaign_io import (
    sha, write_json, frame_write, npz_write, binding_hash,
    file_record, publish_receipt, verify_receipt, git,
)


def write_progress(path,payload):
    """Windows readers can briefly deny replacement of the live progress file."""
    for attempt in range(10):
        try:
            write_json(path,payload)
            return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.02*(attempt+1))


class Context:
    def __init__(self, path):
        self.config_path = Path(path).resolve()
        self.config = json.loads(self.config_path.read_text())
        c = self.config
        for key in ['phase_id','step_id','run_id','tasks','input_versions_file','expected_outputs_file','acceptance_tests','resources']:
            if not c.get(key):
                raise ValueError('Explicit run field missing: ' + key)
        if c['tasks'] != ['F0','F1','F2','F3','F4','F5','F6']:
            raise ValueError('Task list differs from authority')
        if c['permissions']['national_assignment'] or c['permissions']['phase6'] or c['permissions']['network']:
            raise ValueError('Forbidden permission')
        self.repo = Path(c['repository_root']); self.out = Path(c['output_root'])
        self.source = Path(c['source_output_root'])
        self.source_config = json.loads(Path(c['source_campaign_config']).read_text())
        self.code = {str(p.relative_to(self.repo)):sha(p) for p in (self.repo/'partC_rev1_src/phase5_fidelity').glob('*.py')}
        self.binding = binding_hash(dict(config=sha(path), code=self.code))
        self.last_progress = time.time(); self.activity = 'INITIALIZING'; self.violation = None
        self.stop_event = threading.Event()
        (self.out/'validation').mkdir(exist_ok=True,parents=True)
        self.stack_file = (self.out/'validation/stacks.log').open('a',encoding='utf-8')
        faulthandler.enable(file=self.stack_file)
        faulthandler.dump_traceback_later(300,repeat=True,file=self.stack_file)
        self.monitor = threading.Thread(target=self._monitor,daemon=True);self.monitor.start()

    def disk_bytes(self):
        roots = [self.out.parent, self.repo/'partC_doc_icnn/review_gates/phase_05/fidelity_0906',
                 self.repo/self.config['work_records_relative'], self.repo.parent/'p5f0906_input']
        return sum(p.stat().st_size for r in roots if r.exists() for p in r.rglob('*') if p.is_file())

    def _monitor(self):
        target = self.out/'validation/resource_history.csv'
        while True:
            try:
                parent = psutil.Process(os.getpid()); processes=[parent]+parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                disk = self.disk_bytes(); elapsed = time.time()-self.config['task_started_unix']
                r = self.config['resources']
                if rss >= r['rss_bytes'] or disk >= r['added_disk_bytes'] or elapsed >= r['base_seconds']-r['sealing_reserve_seconds']:
                    self.violation = f'Resource guard: elapsed={elapsed}, RSS={rss}, added_bytes={disk}'
                if time.time()-self.last_progress > r['no_progress_seconds']:
                    faulthandler.dump_traceback(file=self.stack_file,all_threads=True)
                    self.violation = 'No progress for 900 seconds: ' + self.activity
                new = not target.exists()
                with target.open('a',encoding='utf-8') as f:
                    if new:f.write('unix,elapsed_seconds,process_tree_rss_bytes,added_disk_bytes,activity\n')
                    f.write(f'{time.time()},{elapsed},{rss},{disk},{self.activity.replace(",",";")}\n')
                print(json.dumps(dict(heartbeat=True,activity=self.activity,elapsed_seconds=round(elapsed,1),rss_bytes=rss,added_bytes=disk)),flush=True)
            except Exception as exc:
                self.violation = 'Resource monitor exception: '+repr(exc)
            if self.stop_event.wait(30):break

    def mark(self, text):
        if self.violation:raise RuntimeError(self.violation)
        self.activity = text;self.last_progress = time.time()
        print(text,flush=True)
        write_progress(self.out/'validation/current_progress.json',dict(activity=text,unix=time.time()))

    def dependency(self, *paths, extra=None, modules=None):
        code = self.code if modules is None else {k:v for k,v in self.code.items() if Path(k).name in modules}
        return binding_hash(dict(config=sha(self.config_path), code=code,
            inputs={str(p):sha(p) for p in paths},extra=extra))

    def stage(self, name, function, dependency=None):
        receipt = self.out/'receipts'/f'{name}.json'
        dep = dependency or self.binding
        if receipt.exists():
            self.mark('Reusing valid '+name)
            return verify_receipt(self.out,receipt,dep)
        self.mark('Starting '+name); start=time.time()
        paths, extra = function()
        publish_receipt(self.out,receipt,name,dep,paths,start,**extra)
        self.mark('Completed '+name)
        return json.loads(receipt.read_text())

    def close(self):
        self.stop_event.set();self.monitor.join(timeout=5)
        faulthandler.cancel_dump_traceback_later();self.stack_file.close()
