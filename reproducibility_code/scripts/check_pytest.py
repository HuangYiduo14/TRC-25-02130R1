"""Run only the portable package tests, with isolated source imports."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists():
        raise FileExistsError('Preserve previous check evidence; choose another run-id')
    output.mkdir(parents=True)
    os.environ['NUMBA_CACHE_DIR']=str(output/'numba_cache')
    import pytest
    return pytest.main([str(ROOT/'tests'),str(ROOT/'scripts/upstream/test_upstream.py'),
        str(ROOT/'src/partC_rev1_src/pacnhn_v1/tests/test_pacnhn_topology.py'),
        '-q','-p','no:cacheprovider','--basetemp',str(output/'fixtures'),
        '--junitxml',str(output/'junit.xml')])

if __name__=='__main__':
    raise SystemExit(main())
