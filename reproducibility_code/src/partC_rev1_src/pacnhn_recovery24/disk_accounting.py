"""Process-local, bounded-age measurements of the NEW output tree only.

The cache retains measurements, never resource-admission decisions. Every call
checks the current caller's limits. It neither writes files nor scans the old
source tree; the caller retains live RSS, available-RAM and free-disk guards.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import threading
import time

MAX_AGE_SECONDS = 600.


@dataclass(frozen=True)
class _Measurement:
    output_root: str
    bytes: int
    census_epoch: float
    scan_seconds: float


@dataclass
class _Entry:
    measurement: _Measurement
    last_observed_epoch: float


_CACHE: dict[str, _Entry] = {}
_LOCK = threading.RLock()


def _roots(output_root, source_root):
    output = Path(output_root).resolve(strict=True)
    source = Path(source_root).resolve(strict=True)
    if (output == source or output.is_relative_to(source) or source.is_relative_to(output)
            or not output.is_dir() or not source.is_dir()):
        raise ValueError('Disk accounting needs disjoint existing new-output and source directories')
    return output, source


def _scan_output_bytes(root):
    """Read metadata only. No alias may redirect traversal outside this tree.

    Files deleted between directory enumeration and stat are treated as absent,
    as in the previous live-output census. Other I/O errors propagate; a failed
    scan is never cached or silently replaced by an older measurement.
    """
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (stat.S_ISLNK(info.st_mode)
                        or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)):
                    raise OSError('Output census refuses symlink/reparse traversal: '+entry.path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise OSError('Unexpected nonregular output entry: '+entry.path)
    return total


def _current_limits(measured_bytes, resources):
    values = {}
    for name in ('new_output_bytes', 'baseline_run_bytes', 'added_disk_bytes'):
        value = resources[name]
        if type(value) is not int or value < 0:
            raise ValueError('Nonnegative integer disk limit required: '+name)
        values[name] = value
    if (measured_bytes > values['new_output_bytes']
            or measured_bytes + values['baseline_run_bytes'] > values['added_disk_bytes']):
        raise OSError('Recovery cumulative/new-output disk guard')


def check_output_disk(output_root, *, source_root, resources, now=None):
    """Return a census observation after enforcing THIS call's disk limits.

    ``census_epoch`` is the start of the actual scan. Cache hits preserve it;
    they cannot postpone the600-second expiry. Any backwards call timestamp
    forces a fresh scan. A lock avoids duplicate scans by concurrent callers in
    this process; separate processes intentionally retain separate caches.
    """
    observed = time.time() if now is None else now
    if type(observed) not in (int, float) or not math.isfinite(observed):
        raise ValueError('Finite census observation epoch required')
    output, _ = _roots(output_root, source_root)
    cache_key = os.path.normcase(str(output))
    with _LOCK:
        entry = _CACHE.get(cache_key)
        reused = bool(entry is not None
            and observed >= entry.last_observed_epoch
            and 0 <= observed-entry.measurement.census_epoch < MAX_AGE_SECONDS)
        if not reused:
            # Remove stale data before scanning. An exception cannot make it
            # reusable on a subsequent call with a different timestamp.
            _CACHE.pop(cache_key, None)
            tick = time.monotonic()
            measured = _scan_output_bytes(output)
            if type(measured) is not int or measured < 0:
                raise ValueError('Invalid census byte count')
            measurement = _Measurement(str(output), measured, float(observed), time.monotonic()-tick)
            entry = _Entry(measurement, float(observed))
            _CACHE[cache_key] = entry
        else:
            entry.last_observed_epoch = float(observed)
        result = entry.measurement
        _current_limits(result.bytes, resources)
        return dict(output_root=result.output_root, bytes=result.bytes,
            census_epoch=result.census_epoch, census_scan_seconds=result.scan_seconds,
            observed_epoch=float(observed), cache_reused=reused, maximum_age_seconds=MAX_AGE_SECONDS)
