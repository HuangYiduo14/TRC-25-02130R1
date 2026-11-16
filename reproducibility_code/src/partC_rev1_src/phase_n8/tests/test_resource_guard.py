import os
import sys
import time
import psutil
from partC_rev1_src.phase_n8.supervise import run_supervised


def test_owned_child_resource_limit_terminates_without_orphan(tmp_path):
    result=run_supervised([sys.executable,'-c','import time; time.sleep(30)'],os.environ.copy(),
        tmp_path/'resource_child.log',deadline=time.time()+5,rss_cap=1,available_floor=0)
    assert result['reason']=='WATCHDOG_RESOURCE_LIMIT'
    # Windows/Conda may introduce an owned helper process; all descendants must exit.
    assert result['no_orphan_child'] and result['pid'] in result['terminated_owned_pids']
    assert all(not psutil.pid_exists(pid) for pid in result['terminated_owned_pids'])
