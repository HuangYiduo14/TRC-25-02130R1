"""Tiny deterministic CSV-part tests; no real data, rendering or model work."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1.publication import write_csv_parts, asset_folder, static_renderer_check
from partC_rev1_src.pacnhn_v1.publication_templates.csv_support import (
    CSV_PART_COLUMNS, csv_expected_files, csv_logical_names, read_logical_csv,
)


def ledger(folder, records):
    pd.DataFrame(records, columns=CSV_PART_COLUMNS).to_csv(folder/"csv_parts.csv", index=False, lineterminator="\n")


def original_bytes(frame):
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def reconstructed_bytes(folder, records):
    return b"".join((folder/row["part_file"]).read_bytes()[0 if i == 0 else row["header_bytes"]:]
                    for i, row in enumerate(records))


def observations():
    return pd.DataFrame({"link_id": [f"{i:05d}" for i in range(31)],
        "flow": np.linspace(0., 1234.987654321, 31),
        "text": ["quoted, comma", 'a "quote"', "two\nlines", "Unicode: Δ 中文", ""]*6+["last"],
        "closed": [False, True]*15+[False]})


def test_many_parts_are_byte_lossless_ordered_and_roundtrip_exact(tmp_path):
    frame = observations()
    records = write_csv_parts(tmp_path/"observations_links.csv", frame, target_bytes=190, batch_rows=11)
    ledger(tmp_path, records)
    assert len(records) > 4
    assert max(row["size_bytes"] for row in records) <= 190
    assert sum(row["row_count"] for row in records) == len(frame)
    assert reconstructed_bytes(tmp_path, records) == original_bytes(frame)
    assert not (tmp_path/"observations_links.csv").exists()
    reads = {}; actual = read_logical_csv(tmp_path, "observations_links.csv", reads)
    pd.testing.assert_frame_equal(actual, frame)
    assert actual.link_id.iloc[0] == "00000"
    assert set(reads) == {"csv_parts.csv"}|{r["part_file"] for r in records}
    assert csv_logical_names(tmp_path, "observations_") == ["observations_links.csv"]
    assert csv_expected_files(tmp_path, ["observations_links.csv"]) == set(reads)


def test_part_boundaries_do_not_change_mixed_column_inference(tmp_path):
    frame = pd.DataFrame({"id": ["0001", "0002", "0003", "0004"],
        "value": [1., 2., np.nan, 4.], "mixed": ["1", "2", "word", "4"]})
    rows = write_csv_parts(tmp_path/"mixed.csv", frame, target_bytes=45, batch_rows=2); ledger(tmp_path, rows)
    actual = read_logical_csv(tmp_path, "mixed.csv", {})
    expected = pd.read_csv(io.BytesIO(original_bytes(frame)), keep_default_na=False,
                           float_precision="round_trip", dtype={"id": str, "mixed": str})
    pd.testing.assert_frame_equal(actual, expected)


def test_deterministic_records_and_csv_bytes_across_independent_folders(tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"; a.mkdir(); b.mkdir(); frame = observations()
    first = write_csv_parts(a/"links.csv", frame, target_bytes=220, batch_rows=7)
    second = write_csv_parts(b/"links.csv", frame, target_bytes=220, batch_rows=7)
    assert first == second
    assert all((a/r["part_file"]).read_bytes() == (b/r["part_file"]).read_bytes() for r in first)


@pytest.mark.parametrize("empty", [False, True])
def test_small_or_empty_data_remains_one_named_valid_csv(tmp_path, empty):
    frame = pd.DataFrame({"id": ["001"], "flow": [1.25]})
    if empty: frame = frame.iloc[:0]
    rows = write_csv_parts(tmp_path/"small.csv", frame); ledger(tmp_path, rows)
    assert len(rows) == 1 and rows[0]["part_file"] == "small.csv"
    assert rows[0]["row_count"] == len(frame)
    actual = read_logical_csv(tmp_path, "small.csv", {}, dtype=str)
    assert len(actual) == len(frame) and list(actual) == list(frame)


@pytest.mark.parametrize("mutation", ["missing", "changed", "reordered", "row_count", "escaped"])
def test_missing_modified_or_incomplete_parts_fail_closed(tmp_path, mutation):
    rows = write_csv_parts(tmp_path/"links.csv", observations(), target_bytes=190, batch_rows=8)
    if mutation == "missing": (tmp_path/rows[-1]["part_file"]).unlink()
    if mutation == "changed": (tmp_path/rows[-1]["part_file"]).write_bytes(b"wrong CSV\n")
    if mutation == "reordered": rows = list(reversed(rows))
    if mutation == "row_count": rows[-1]["row_count"] += 1
    if mutation == "escaped": rows[0]["part_file"] = "../outside.csv"
    ledger(tmp_path, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_logical_csv(tmp_path, "links.csv", {})


def test_oversized_single_observation_is_rejected_never_truncated(tmp_path):
    with pytest.raises(ValueError, match="never truncate"):
        write_csv_parts(tmp_path/"large.csv", pd.DataFrame({"text": ["x"*200]}), target_bytes=40)
    assert not (tmp_path/"csv_parts.csv").exists()


def test_existing_logical_file_never_overwritten(tmp_path):
    path = tmp_path/"x.csv"; path.write_bytes(b"keep\n")
    with pytest.raises(ValueError, match="already exists"):
        write_csv_parts(path, pd.DataFrame({"x": [1]}))
    assert path.read_bytes() == b"keep\n"


def test_generated_renderer_embeds_reader_without_helper_imports(tmp_path):
    data = pd.DataFrame({"value": [1., 2.]})
    folder = asset_folder(tmp_path, "figure3", {"values.csv": data}, [],
        panels=[dict(panel_id="a", title="Synthetic fixture", kind="hist", source="values.csv", x="value")])
    source = (folder/"plot.py").read_text(encoding="utf-8")
    assert "EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE" not in source
    assert "def read_logical_csv(" in source
    assert "from csv_support" not in source
    assert static_renderer_check(folder/"plot.py")["status"] == "PASS"
    provenance = json.loads((folder/"provenance.json").read_text())
    assert provenance["rows_retained_without_sampling"] is True
    assert provenance["csv_part_target_bytes"] == 20_000_000
    pd.testing.assert_frame_equal(read_logical_csv(folder, "values.csv", {}), data.astype({"value": "int64"}))
