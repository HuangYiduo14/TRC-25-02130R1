"""Run one standalone asset with a fail-closed file-access audit.

This is the verification harness, not a scientific plotting dependency. It reads
no campaign inputs. Its explicit arguments name the isolated local asset script
and an output audit receipt. The figure/table folder needs only its own script.
"""
import json
import os
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve(strict=True)
receipt = Path(sys.argv[2]).resolve()
root = script.parent
prefixes = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(sys.executable).resolve().parent}
windows = Path(os.environ.get("WINDIR", "C:/Windows")).resolve()
access = []


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def audit(event, args):
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "urllib.Request"}:
        raise RuntimeError("Network/process access prohibited in standalone renderer: "+event)
    if event != "open" or not args or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode = args[1] if len(args)>1 else "r"
    flags = args[2] if len(args)>2 and isinstance(args[2], int) else 0
    writing = (isinstance(mode, str) and any(v in mode for v in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if writing:
        allowed = inside(path, root)
    elif inside(path, root):
        allowed = path == script or (path.parent == root and path.suffix == ".csv") or inside(path, root/".mplconfig") or inside(path, root/".tmp")
    else:
        allowed = any(inside(path, prefix) for prefix in prefixes) or inside(path, windows)
    access.append({"path_class": "asset" if inside(path, root) else "runtime_or_system", "file": path.name,
                   "operation": "write" if writing else "read", "allowed": bool(allowed)})
    if not allowed:
        raise RuntimeError("CSV-only standalone file-access violation: "+str(path))


sys.addaudithook(audit)
runpy.run_path(str(script), run_name="__main__")
receipt.write_text(json.dumps({"status": "PASS", "scientific_inputs": "LOCAL_CSV_ONLY", "access": access}, indent=2)+"\n", encoding="utf-8")
