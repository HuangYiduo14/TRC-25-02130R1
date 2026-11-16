"""Synthetic table extraction only; no files, figures, or campaign measurements."""
from pathlib import Path

import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1 import publication as p


@pytest.mark.parametrize("physical_suite", [None, ""])
def test_table4_preserves_all_eight_physical_rows_with_empty_suite(monkeypatch, physical_suite):
    data = {key: pd.DataFrame(columns=columns) for key, columns in p.SCHEMAS.items()}
    network = []
    for suite in ("P", "s03"):
        row = {key: 1 for key in p.SCHEMAS["network"]}
        row.update(suite_id=suite, resolution=1., active_pairs=2, status="SYNTHETIC")
        network.append(row)
    data["network"] = pd.DataFrame(network)
    rows = []
    physical_ids = set()
    for solver in ("CFW", "FW", "MSA", "REFERENCE"):
        for suffix, scenario in (("open", "OPEN"), ("i95", "REPRESENTED_I95")):
            case_id = f"P_{solver}_{suffix}"
            physical_ids.add(case_id)
            reference = solver == "REFERENCE"
            row = {key: "" for key in p.SCHEMAS["methods"]}
            row.update(case_id=case_id, method_id=f"P/{solver}", network="P", suite_id=physical_suite,
                scenario=scenario, status="REFERENCE_SELF" if reference else "CONVERGED", solver=solver,
                RG_SPTT=0. if reference else 1e-4, elapsed_seconds=7.,
                wmape_backbone_percent=0. if reference else 1.25,
                wmape_key_percent=0. if reference else 2.5,
                reference_case_id=f"P_REFERENCE_{suffix}",
                assignment_role="REFERENCE_SELF" if reference else "INDEPENDENT_CONTROL")
            rows.append(row)
    regional = dict(rows[0], case_id="s03_FFTT_open", method_id="s03/FFTT", network="ACNHN",
        suite_id="s03", assignment_role="SURROGATE", wmape_backbone_percent=3.75)
    rows.append(regional)
    data["methods"] = pd.DataFrame(rows)
    before = data["methods"].copy(deep=True)
    assets = {}
    def capture(root, asset_id, frames, records, **kwargs):
        assets[asset_id] = (frames, kwargs["table"].copy(deep=True))
        return Path("not-written")
    monkeypatch.setattr(p, "asset_folder", capture)
    p.build_tables(Path("unused"), data, [])
    frames, table = assets["table4"]
    physical = table.loc[table.suite.str.startswith("P / ")]
    assert len(physical) == 8
    actual_ids = physical.case.str.split(" / ").str[0]
    assert set(actual_ids) == physical_ids
    assert not physical.status.eq("NO_METHOD_ROW_IN_EXPORT").any()
    assert (physical.status == "REFERENCE_SELF").sum() == 2
    assert (physical.status == "CONVERGED").sum() == 6
    assert not actual_ids.eq("s03_FFTT_open").any()
    regional_table = table.loc[table.suite.str.startswith("s03 / ")]
    assert list(regional_table.case) == ["s03_FFTT_open / OPEN"]
    for _, output in physical.iterrows():
        original = before.loc[before.case_id.eq(output["case"].split(" / ")[0])].iloc[0]
        assert output.B == p.finite_text(original.wmape_backbone_percent)
        assert output.key == p.finite_text(original.wmape_key_percent)
    pd.testing.assert_frame_equal(frames["observations_methods.csv"], before)
    pd.testing.assert_frame_equal(data["methods"], before)
