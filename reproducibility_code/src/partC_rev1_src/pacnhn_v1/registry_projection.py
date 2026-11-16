"""Bounded, authenticated Stage 3 view of the frozen pretty-JSON registry.

This reads the indent=2 format emitted by control.write_json. It is deliberately
not a general JSON parser and never falls back to loading the entire document.
The caller must supply the SHA-256 from the completed training-stage receipt.
Excluded metadata remains authenticated by that exact whole-file digest.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


MODEL_FIELDS = frozenset((
    'suite_id', 'family', 'region_index', 'status', 'pair_keys_identity',
    'graph_sha256', 'artifact_hashes', 'model_path', 'model_sha256', 'model_kind',
))
MAX_LINE_BYTES = 2 * 1024**2
MAX_VALUE_BYTES = 4 * 1024**2
_FIELD = re.compile(r'^  ("(?:[^"\\]|\\.)*"): (.*)$')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate registry JSON property: ' + key)
        result[key] = value
    return result


def _decode(value):
    return json.loads(value, object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError('Nonfinite registry JSON constant: ' + value)))


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class _Lines:
    def __init__(self, stream, guard):
        self.stream = stream
        self.guard = guard
        self.digest = hashlib.sha256()
        self.bytes = 0
        self.next_guard = 0

    def next(self):
        if self.bytes >= self.next_guard:
            self.guard()
            self.next_guard = self.bytes + 8 * 1024**2
        raw = self.stream.readline(MAX_LINE_BYTES + 1)
        if len(raw) > MAX_LINE_BYTES:
            raise ValueError('Registry line exceeds bounded reader limit')
        if not raw:
            raise ValueError('Truncated frozen registry')
        self.digest.update(raw)
        self.bytes += len(raw)
        return raw.decode('utf-8').rstrip('\r\n')


def _value(lines, first, *, keep):
    """Consume one root property; nested JSON never has indent two closers."""
    if first not in ('[', '{'):
        return (_decode(first.removesuffix(',')) if keep else None,
                first.endswith(','))
    closing = '  ]' if first == '[' else '  }'
    block = [first] if keep else None
    size = len(first)
    while True:
        line = lines.next()
        if line in (closing, closing + ','):
            if keep:
                block.append(line.removesuffix(','))
                return _decode('\n'.join(block)), line.endswith(',')
            return None, line.endswith(',')
        # Every nested container/member is indented deeper by the producer.
        # Strings cannot contain literal newlines in JSON. An incompatible
        # source format fails closed instead of changing projection boundaries.
        if not line.startswith('    '):
            raise ValueError('Unsupported registry nested indentation')
        if keep:
            size += len(line) + 1
            if size > MAX_VALUE_BYTES:
                raise ValueError('Registry metadata value exceeds bounded reader limit')
            block.append(line)


def _models(lines, first):
    if first in ('[]', '[],'):
        return [], first.endswith(',')
    if first != '[':
        raise ValueError('Registry models must use the frozen array format')
    rows = []
    keys = set()
    need_record = True
    while True:
        line = lines.next()
        if line in ('  ]', '  ],'):
            if need_record and rows:
                raise ValueError('Trailing comma in registry model array')
            return rows, line.endswith(',')
        if not need_record or line != '    {':
            raise ValueError('Unsupported registry model record format')
        block = ['{']
        size = 1
        while True:
            line = lines.next()
            if line in ('    }', '    },'):
                block.append('}')
                need_record = line.endswith(',')
                break
            if not line.startswith('      '):
                raise ValueError('Unsupported registry model indentation')
            size += len(line) + 1
            if size > MAX_VALUE_BYTES:
                raise ValueError('Registry model exceeds bounded reader limit')
            block.append(line)
        record = _decode('\n'.join(block))
        key = (record['suite_id'], record['family'], record['region_index'])
        if (not isinstance(key[0], str) or not isinstance(key[1], str)
                or type(key[2]) is not int or key[2] < 0
                or not isinstance(record['status'], str)):
            raise ValueError('Invalid frozen model identity/status')
        if key in keys:
            raise ValueError('Duplicate frozen model identity')
        keys.add(key)
        # Keep every row, including unavailable/FFTT rows. Additional scalar
        # identities are cheap and retain exact dataset/support provenance.
        rows.append({k: v for k, v in record.items() if k in MODEL_FIELDS or
            ((k.endswith('_identity') or k.endswith('_sha256')) and
             (v is None or isinstance(v, (str, int, float, bool))))})


def _validate_coverage(registry):
    if 'models' not in registry and 'coverage' not in registry:
        return  # Older minimal gate fixtures; numerical consumers require both.
    if 'models' not in registry or 'coverage' not in registry:
        raise ValueError('Registry model/coverage fields must occur together')
    counts = Counter((r['suite_id'], r['family']) for r in registry['models'])
    available = Counter((r['suite_id'], r['family']) for r in registry['models']
                        if r['status'] == 'AVAILABLE')
    seen = set()
    for row in registry['coverage']:
        key = row['suite_id'], row['family']
        if key in seen:
            raise ValueError('Duplicate registry coverage identity')
        seen.add(key)
        if (type(row['expected']) is not int or type(row['available']) is not int
                or row['expected'] != counts[key] or row['available'] != available[key]
                or row['complete_family'] is not (row['expected'] == row['available'])):
            raise ValueError('Registry coverage differs from retained model rows')
    if seen != set(counts):
        raise ValueError('Registry coverage omits model groups')


def read_stage3_registry(path, expected_sha256, *, guard=lambda: None):
    """Return the existing dict interface after exact full-source authentication.

    Only current_dataset_commitment and unused per-model training diagnostics
    are omitted. No model, availability row, runtime hash or cost is substituted.
    Parsing is bounded per line/value/model; metadata rows are retained once.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Completed training must supply the exact registry SHA-256')
    path = Path(path)
    before = _signature(path.stat())
    registry = {}
    seen = set()
    guard()
    with path.open('rb') as stream:
        import os
        if _signature(os.fstat(stream.fileno())) != before:
            raise ValueError('Frozen registry changed before reading')
        lines = _Lines(stream, guard)
        if lines.next() != '{':
            raise ValueError('Unsupported frozen registry root format')
        more = True
        while more:
            line = lines.next()
            field = _FIELD.fullmatch(line)
            if field is None:
                raise ValueError('Unsupported frozen registry root property')
            key, first = _decode(field[1]), field[2]
            if key in seen:
                raise ValueError('Duplicate registry root property: ' + key)
            seen.add(key)
            if key == 'models':
                value, more = _models(lines, first)
            else:
                value, more = _value(lines, first, keep=key != 'current_dataset_commitment')
            if key != 'current_dataset_commitment':
                registry[key] = value
        if lines.next() != '}' or stream.read(1):
            raise ValueError('Unexpected frozen registry trailing bytes')
        after = _signature(os.fstat(stream.fileno()))
    if before != after or before != _signature(path.stat()):
        raise ValueError('Frozen registry changed while reading')
    if lines.bytes != before[2] or lines.digest.hexdigest() != expected_sha256:
        raise ValueError('Completed training registry SHA-256 differs')
    if registry.get('schema') != 'PACNHN_REGISTRY_V1':
        raise ValueError('Unsupported frozen registry schema')
    _validate_coverage(registry)
    guard()
    return registry
