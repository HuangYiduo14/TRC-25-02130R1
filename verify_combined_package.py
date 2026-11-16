"""Check all distributed code/data files; report optional generated files separately."""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parent
    with (root / 'PACKAGE_MANIFEST.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    declared = {r['path'] for r in rows}
    assert rows and len(declared) == len(rows), 'Empty or duplicate manifest'
    for row in rows:
        relative = PurePosixPath(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / row['path']
        assert path.is_file(), f'Missing: {row["path"]}'
        assert path.stat().st_size == int(row['bytes']) and digest(path) == row['sha256'], f'Changed: {row["path"]}'
    checksum_paths = set()
    for line in (root / 'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert relative in declared | {'PACKAGE_MANIFEST.csv'}
        assert relative not in checksum_paths
        assert digest(root / relative) == expected, f'Checksum mismatch: {relative}'
        checksum_paths.add(relative)
    assert checksum_paths == declared | {'PACKAGE_MANIFEST.csv'}, 'Incomplete checksums'
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    extras = actual - declared - {'PACKAGE_MANIFEST.csv', 'CHECKSUMS.sha256'}
    print(json.dumps({'status': 'PASS', 'verified_payload_files': len(rows), 'additional_generated_files': len(extras)}, indent=2))

if __name__ == '__main__':
    main()
