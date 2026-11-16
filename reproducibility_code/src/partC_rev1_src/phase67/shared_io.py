"""Episode-scoped, rename-friendly Windows I/O with unchanged serialization bytes.

Only OPEN/READ and REPLACE sharing denials are retried. Failed publications retain
their unique pending payload; a target is never unlinked or written in place.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np

READ_DELAYS = tuple(.02 * n for n in range(1, 10))
WRITE_DELAYS = tuple(min(.02 * 2**min(n, 4), .2) for n in range(24))
_counters = Counter()
_lock = threading.Lock()


def _count(**values):
    with _lock:
        _counters.update(values)


def counters():
    with _lock:
        return dict(_counters)


def sharing_denial(error):
    return isinstance(error, PermissionError) or (
        os.name == 'nt' and getattr(error, 'winerror', None) in (5, 32, 33))


if os.name == 'nt':
    from ctypes import wintypes
    import msvcrt
    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _create = _kernel.CreateFileW
    _create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _create.restype = wintypes.HANDLE
    _close = _kernel.CloseHandle
    _close.argtypes = [wintypes.HANDLE]
    _close.restype = wintypes.BOOL
    _set_file_info = _kernel.SetFileInformationByHandle
    _set_file_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _set_file_info.restype = wintypes.BOOL
    _create_mutex = _kernel.CreateMutexW
    _create_mutex.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _create_mutex.restype = wintypes.HANDLE
    _wait = _kernel.WaitForSingleObject
    _wait.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _wait.restype = wintypes.DWORD
    _release_mutex = _kernel.ReleaseMutex
    _release_mutex.argtypes = [wintypes.HANDLE]
    _release_mutex.restype = wintypes.BOOL

    class _RenameInfo(ctypes.Structure):
        _fields_ = [('Flags', wintypes.DWORD), ('RootDirectory', wintypes.HANDLE),
                    ('FileNameLength', wintypes.DWORD), ('FileName', wintypes.WCHAR * 1)]


def _windows_name(path):
    name = os.path.abspath(os.fspath(path))  # Lexical: no resolve/stat walk per read.
    if not name.startswith('\\\\?\\') and len(name) >= 248:
        name = '\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name
    return name


@contextmanager
def _path_gate(path):
    """Coordinate controlled opens with the native rename, not snapshot reads.

    No persistent lock file or alternate storage is used. An already opened
    delete-sharing reader remains independent and may finish its old snapshot.
    Timeout/abandonment fails closed; it is not an open/read sharing retry.
    """
    if os.name != 'nt':
        yield
        return
    name = os.path.normcase(os.path.abspath(os.fspath(path)))
    name = 'Local\\p67_shared_io_' + hashlib.sha256(name.encode('utf-8')).hexdigest()
    handle = _create_mutex(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    owned = False
    started = time.monotonic()
    try:
        result = _wait(handle, 1000)
        _count(protocol_gate_wait_seconds=time.monotonic()-started, protocol_gate_waits=1)
        owned = result in (0, 0x80)
        if result == 0x102:
            _count(protocol_gate_timeouts=1)
            error = TimeoutError(errno.ETIMEDOUT, 'Shared I/O path gate timed out', str(path))
            error.wait_result = result
            raise error
        if result == 0x80:
            _count(protocol_gate_abandoned=1)
            error = OSError(errno.EIO, 'Shared I/O path gate was abandoned', str(path))
            error.wait_result = result
            raise error
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        yield
    finally:
        try:
            if owned and not _release_mutex(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _close(handle)


def _open_read(path):
    if os.name != 'nt':
        return Path(path).open('rb')
    raw = None
    try:
        with _path_gate(path):
            raw = _create(_windows_name(path), 0x80000000, 7, None, 3, 0x80, None)
            if raw == ctypes.c_void_p(-1).value:
                raw = None
                raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        if raw is not None:
            _close(raw)
        raise
    try:
        fd = msvcrt.open_osfhandle(int(raw), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _close(raw)
        raise
    try:
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)  # fd owns raw after open_osfhandle succeeds.
        raise


def _read_shared_bytes(path):
    with _open_read(path) as stream:
        return stream.read()


def _replace(pending, target):
    if os.name != 'nt':
        return os.replace(pending, target)
    with _path_gate(target):
        return _replace_windows(pending, target)


def _replace_windows(pending, target):
    # Hold DELETE access throughout replacement. This respects incompatible readers
    # and prevents a new non-delete-sharing reader from entering during the rename.
    guard = _create(_windows_name(target), 0x10000, 7, None, 3, 0x80, None)
    if guard == ctypes.c_void_p(-1).value:
        error = ctypes.WinError(ctypes.get_last_error())
        if error.winerror == 2:
            os.rename(pending, target)
            _count(exclusive_first_publications=1)
            return
        raise error
    handle = None
    try:
        handle = _create(_windows_name(pending), 0x10000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            handle = None
            raise ctypes.WinError(ctypes.get_last_error())
        name = _windows_name(target).encode('utf-16-le')
        # The Win32 wrapper also expects a terminated wide filename. Length binds
        # only the name bytes; include separate zeroed WCHAR storage for the NUL.
        buffer = ctypes.create_string_buffer(ctypes.sizeof(_RenameInfo) + len(name) + 2)
        header = _RenameInfo.from_buffer(buffer)
        header.Flags = 3  # REPLACE_IF_EXISTS | POSIX_SEMANTICS, with the guard above.
        header.RootDirectory = None
        header.FileNameLength = len(name)
        ctypes.memmove(ctypes.addressof(buffer) + _RenameInfo.FileName.offset, name, len(name))
        _count(native_rename_info_calls=1)
        if not _set_file_info(handle, 22, buffer, len(buffer)):  # FileRenameInfoEx
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if handle is not None:
            _close(handle)
        _close(guard)


def read_json(path, *, read_bytes=None, sleep=None):
    reader = _read_shared_bytes if read_bytes is None else read_bytes
    sleep = time.sleep if sleep is None else sleep
    _count(read_calls=1)
    for attempt in range(10):
        _count(read_attempts=1)
        try:
            payload = reader(Path(path))
        except OSError as error:
            if not sharing_denial(error) or attempt == 9:
                _count(read_failures=1)
                error.io_attempts = attempt + 1
                raise
            _count(read_retries=1, requested_read_sleep_seconds=READ_DELAYS[attempt])
            sleep(READ_DELAYS[attempt])
        else:
            return json.loads(payload.decode('utf-8'))


def _failure(error, pending, payload, attempt, started, monotonic):
    error.pending_path = str(pending)
    error.serialized_payload_sha256 = hashlib.sha256(payload).hexdigest()
    error.io_attempts = attempt
    error.replace_retry_elapsed_seconds = max(0., monotonic() - started)
    error.pending_sha256 = None
    if pending.is_file():
        try:
            with pending.open('rb') as stream:
                error.pending_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError:
            pass  # Annotation is optional; never replace the original exception.


def atomic_bytes(path, payload, *, replace=None, sleep=None, monotonic=None):
    if not isinstance(payload, bytes):
        raise TypeError('Atomic payload must be serialized bytes')
    replace = _replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    monotonic = time.monotonic if monotonic is None else monotonic
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f'.{target.name}.pending.{os.getpid()}.{uuid.uuid4().hex}')
    _count(write_calls=1)
    started = monotonic()
    try:
        with pending.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        _failure(error, pending, payload, 0, started, monotonic)
        _count(write_failures=1)
        raise
    started = monotonic()
    last_error = None
    for attempt in range(25):
        if attempt and monotonic() - started >= 5.:
            _failure(last_error, pending, payload, attempt, started, monotonic)
            _count(write_failures=1)
            raise last_error
        _count(replace_attempts=1)
        try:
            replace(pending, target)
        except OSError as error:
            last_error = error
            duration = monotonic() - started
            if not sharing_denial(error) or attempt == 24 or duration + WRITE_DELAYS[attempt] >= 5.:
                _failure(error, pending, payload, attempt + 1, started, monotonic)
                _count(write_failures=1, failed_replace_loop_seconds=duration)
                raise
            _count(write_retries=1, requested_write_sleep_seconds=WRITE_DELAYS[attempt])
            sleep(WRITE_DELAYS[attempt])
        else:
            duration = monotonic() - started
            _count(successful_writes=1, successful_replace_loop_seconds=duration)
            return dict(attempts=attempt + 1, sha256=hashlib.sha256(payload).hexdigest(),
                        bytes=len(payload), temporary_file=str(pending), retry_loop_seconds=duration)


def atomic(path, payload):
    atomic_bytes(path, payload)


def write_json(path, obj):
    # Byte-for-byte serializer used by phase7/common.py; keep NumPy scalar handling.
    payload = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False,
                         default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n').encode()
    atomic(path, payload)


def save_npz(path, **arrays):
    assert all(np.asarray(v).dtype.kind != 'O' for v in arrays.values())
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    atomic(path, payload.getvalue())
