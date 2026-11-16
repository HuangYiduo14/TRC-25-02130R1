"""Local logical-CSV reader, embedded verbatim into each generated renderer.

A single pandas parser sees the ordered original records across all parts, so
type inference and floating-point parsing never depend on shard boundaries.
This module uses only sibling CSV files and has no execution side effects.
"""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

CSV_PART_COLUMNS = ["logical_file", "part_file", "part_index", "row_start", "row_count",
    "total_rows", "size_bytes", "sha256", "header_bytes", "header_sha256", "columns_json", "text_columns_json"]


def csv_leaf(name):
    if not isinstance(name, str) or not name or Path(name).name != name or not name.endswith(".csv") or any(c in name for c in '/\\:<>"|?*') or name in {".", ".."}:
        raise ValueError("Logical CSV inputs must be sibling CSV filenames")
    return name


def csv_file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def csv_part_ledger(root, reads=None):
    path = Path(root)/"csv_parts.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame) != CSV_PART_COLUMNS or not len(frame):
        raise ValueError("Missing or invalid logical-CSV part ledger")
    if frame.part_file.duplicated().any():
        raise ValueError("A physical CSV part belongs to more than one logical dataset")
    for name in list(frame.logical_file.unique())+list(frame.part_file):
        csv_leaf(name)
        if name == "csv_parts.csv":
            raise ValueError("The part ledger cannot contain its own circular hash")
    if reads is not None:
        reads["csv_parts.csv"] = {"file": "csv_parts.csv", "sha256": csv_file_sha(path), "rows": len(frame)}
    return frame


def csv_logical_names(root, prefix="", reads=None):
    frame = csv_part_ledger(root, reads)
    return sorted(name for name in frame.logical_file.unique() if name.startswith(prefix))


def csv_logical_parts(root, name, reads=None):
    csv_leaf(name)
    rows = csv_part_ledger(root, reads)
    rows = rows.loc[rows.logical_file.eq(name)].to_dict("records")
    if not rows:
        raise ValueError("Logical scientific CSV is absent from the local part ledger: "+name)
    position = 0
    identities = ["total_rows", "header_bytes", "header_sha256", "columns_json", "text_columns_json"]
    for index, row in enumerate(rows, 1):
        for key in ["part_index", "row_start", "row_count", "total_rows", "size_bytes", "header_bytes"]:
            value = int(row[key])
            if value < 0 or str(value) != row[key]:
                raise ValueError("Part ledger counts must be canonical nonnegative integers")
            row[key] = value
        if row["part_index"] != index or row["row_start"] != position or row["header_bytes"] <= 0:
            raise ValueError("CSV parts are not in complete contiguous source-row order")
        if index > 1 and any(row[key] != rows[0][key] for key in identities):
            raise ValueError("Logical CSV parts disagree on total rows or schema")
        if row["row_count"] == 0 and (len(rows) != 1 or row["total_rows"] != 0):
            raise ValueError("Empty CSV part is allowed only for an empty logical dataset")
        position += row["row_count"]
    if position != rows[0]["total_rows"]:
        raise ValueError("CSV part row counts do not cover the declared logical population")
    return rows


def csv_expected_files(root, logical_names):
    physical = {"csv_parts.csv"}
    for name in logical_names:
        physical.update(row["part_file"] for row in csv_logical_parts(root, name))
    return physical


class _CSVPartsStream(io.RawIOBase):
    """Present complete first header plus all ordered part records as one file."""
    def __init__(self, root, rows):
        super().__init__()
        self.root = Path(root); self.rows = rows; self.index = 0; self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        target = memoryview(buffer); used = 0
        while used < len(target):
            if self.current is None:
                if self.index >= len(self.rows):
                    break
                row = self.rows[self.index]
                self.current = (self.root/row["part_file"]).open("rb")
                if self.index:
                    self.current.seek(row["header_bytes"])
                self.index += 1
            count = self.current.readinto(target[used:])
            if count:
                used += count
            else:
                self.current.close(); self.current = None
        return used

    def close(self):
        if self.current is not None:
            self.current.close(); self.current = None
        super().close()


def read_logical_csv(root, name, reads, dtype=None):
    rows = csv_logical_parts(root, name, reads)
    columns = json.loads(rows[0]["columns_json"])
    text_columns = json.loads(rows[0]["text_columns_json"])
    if not isinstance(columns, list) or len(columns) != len(set(columns)) or not set(text_columns).issubset(columns):
        raise ValueError("Invalid logical CSV column schema")
    for row in rows:
        path = Path(root)/row["part_file"]
        if path.is_symlink() or path.resolve().parent != Path(root).resolve() or not path.is_file():
            raise ValueError("CSV part must be an actual local sibling file")
        if path.stat().st_size != row["size_bytes"] or csv_file_sha(path) != row["sha256"]:
            raise ValueError("CSV part changed or was omitted: "+row["part_file"])
        with path.open("rb") as stream:
            if hashlib.sha256(stream.read(row["header_bytes"])).hexdigest() != row["header_sha256"]:
                raise ValueError("CSV part header differs from its logical schema")
        reads[row["part_file"]] = {"file": row["part_file"], "sha256": row["sha256"], "rows": row["row_count"]}
    options = {"keep_default_na": False, "float_precision": "round_trip", "dtype": dtype if dtype is not None else {name: str for name in text_columns}}
    with io.BufferedReader(_CSVPartsStream(root, rows)) as stream:
        frame = pd.read_csv(stream, **options)
    if list(frame) != columns or len(frame) != rows[0]["total_rows"]:
        raise ValueError("Reconstructed logical CSV schema/row count differs")
    return frame
