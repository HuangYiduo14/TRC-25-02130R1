"""Verify the distributed payload; downloaded/generated data are outside this manifest."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {'PACKAGE_MANIFEST.csv', 'CHECKSUMS.sha256'}

def excluded(relative):
    return (relative.startswith(('data/raw/', 'data/processed/', 'runs/', '.venv/', '.pytest_cache/'))
            or '__pycache__' in Path(relative).parts or relative.endswith(('.pyc','.nbc','.nbi')))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root/'PACKAGE_MANIFEST.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty package manifest')
    declared = {r['path'] for r in rows}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.relative_to(root).as_posix() not in CONTROLS
              and not excluded(p.relative_to(root).as_posix())}
    if declared != actual or len(declared) != len(rows):
        raise ValueError(f'Member closure mismatch; missing={declared-actual}, extra={actual-declared}')
    for row in rows:
        path = root/row['path']
        if path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
            raise ValueError(f'Payload changed: {row["path"]}')
        if row['role'] == 'mapping_metadata' and int(row['bytes']) > 10000000:
            raise ValueError(f'Data size limit exceeded: {row["path"]}')
    for line in (root/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        if relative not in declared | {'PACKAGE_MANIFEST.csv'} or sha(root/relative) != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
    total = sum(int(r['bytes']) for r in rows) + sum((root/p).stat().st_size for p in CONTROLS)
    if total >= 100000000:
        raise ValueError('Expanded distributed package exceeds100MB')
    print(json.dumps({'status':'PASS','payload_files':len(rows),'payload_bytes':total,
                      'max_mapping_metadata_bytes':max((int(r['bytes']) for r in rows if r['role']=='mapping_metadata'),default=0),
                      'runtime_data_excluded':True},indent=2))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'FAILED','error':str(exc)}),file=sys.stderr)
        raise SystemExit(1)
