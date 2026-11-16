from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "upstream"))
import build_inputs as adapter
from vectorized_capacity import attach_link_aadt_defaults, enrich_directed_arcs_capacity
from partC_rev1_src.partC_rev1_capacity_enrichment import (
    partC_rev1_attach_link_aadt_defaults, partC_rev1_enrich_directed_arcs_capacity)


def test_supervisor_run_id_overrides_config_and_direct_fallback(monkeypatch):
    path = BASE / "configs/upstream.json"
    configured_run_id = json.loads(path.read_text())["run_id"]
    monkeypatch.setenv("REPRODUCTION_RUN_ID", "supervised_test_campaign")
    control, cfg, _, _ = adapter.config_for_stage(path)
    assert control["run_id"] == "supervised_test_campaign"
    assert cfg["project"]["run_id"] == "supervised_test_campaign"
    monkeypatch.delenv("REPRODUCTION_RUN_ID")
    control, cfg, _, _ = adapter.config_for_stage(path)
    assert control["run_id"] == configured_run_id
    assert cfg["project"]["run_id"] == configured_run_id


def test_capacity_exact_scalar_equivalence():
    control, cfg, _, _ = adapter.config_for_stage(BASE / "configs/upstream.json")
    links = pd.DataFrame({
        "partC_rev1_link_uid": np.arange(7), "Class": [11, 12, 13, 14, 50, 41, 18],
        "AB_Lanes": [2, 1, 3, 0, np.nan, 1, 1], "BA_Lanes": [1, 2, 1, 1, 0, 1, 1],
        "AB_FreeFlowTime": [2, 0, np.nan, 0, 0, 0, 0], "BA_FreeFlowTime": [3, 0, np.nan, 0, 0, 0, 0],
        "AB_FinalSpeed": [65, 60, 50, np.nan, np.nan, 20, np.nan],
        "BA_FinalSpeed": [60, 55, 45, np.nan, np.nan, 25, np.nan],
        "partC_rev1_ntad_aadt_total": [40_000, 50_000, 0, np.nan, 1000, 1e8, 2000],
        "partC_rev1_ntad_aadt_combination": [3000, np.nan, 0, np.nan, 80, 3e6, -10],
        "partC_rev1_ntad_aadt_single": [200, np.nan, 0, np.nan, 20, 2e6, np.nan],
        "partC_rev1_ntad_speed_limit_mph": [60, 55, 40, 35, np.nan, np.nan, np.nan],
        "Speed_Limit": [65, 60, 45, np.nan, np.nan, np.nan, np.nan],
    })
    old_links = partC_rev1_attach_link_aadt_defaults(links, cfg)
    new_links = attach_link_aadt_defaults(links, cfg)
    pd.testing.assert_frame_equal(old_links, new_links)
    arcs = pd.DataFrame({"partC_rev1_link_uid": np.repeat(np.arange(7), 2),
                         "direction": ["AB", "BA"] * 7, "length_miles": np.repeat([2, 1, 2, 3, 0, 1, 2], 2),
                         "class_code": np.repeat(links["Class"], 2),
                         "free_time_min": [1, 2, np.nan, 0, 0, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0],
                         "capacity": 2000.0, "partC_rev1_default_lane_count": np.nan})
    old, old_preserve, old_report = partC_rev1_enrich_directed_arcs_capacity(arcs, old_links, cfg)
    new, new_preserve, new_report = enrich_directed_arcs_capacity(arcs, new_links, cfg)
    for name in old:
        if name in new and old[name].dtype.kind in "fiu":
            np.testing.assert_allclose(old[name], new[name], rtol=0, atol=0, equal_nan=True, err_msg=name)
    for name in ["partC_rev1_fftt_source", "partC_rev1_lanes_source", "partC_rev1_capacity_source", "partC_rev1_enrichment_warning"]:
        pd.testing.assert_series_equal(old[name], new[name])
    pd.testing.assert_frame_equal(old_preserve, new_preserve)
    pd.testing.assert_frame_equal(old_report, new_report)


def make_raw_fixture(tmp_path, cfg):
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    raw = tmp_path / "raw"
    raw.mkdir()
    points = [(-88, 41), (-87.99, 41), (-87.98, 41)]
    nodes = gpd.GeoDataFrame({"ID": ["1", "2", "3"], "DATA": ["FAF"] * 3,
                             "Centroid": [1, 0, 1], "CentroidID": ["17001", "", "17003"], "FAFID": ["1", "1", "2"]},
                             geometry=[Point(*p) for p in points], crs=4269)
    links = gpd.GeoDataFrame({"ID": ["10", "11"], "LENGTH": [0.0, .5], "Class": [50, 11], "STRAHNET": ["", "S"],
                             "AB_Lanes": [1, 2], "BA_Lanes": [1, 2], "AB_FreeFlowTime": [1, 1], "BA_FreeFlowTime": [1, 1],
                             "AB_FinalSpeed": [30, 30], "BA_FinalSpeed": [30, 30], "Status": [1, 1], "Truck": ["", ""],
                             "Road_Name": ["TEST", "TEST"], "Sign_Rte": ["I 95", "I 95"], "NHS": [1, 1], "NHFN": [1, 1]},
                             geometry=[LineString(points[:2]), LineString(points[1:])], crs=4269)
    network = raw / "network.gpkg"
    nodes.to_file(network, layer="FAF5_Nodes", driver="GPKG")
    links.to_file(network, layer="FAF5_Links", driver="GPKG")
    pd.DataFrame({"dms_orig": [1, 2, 1], "dms_dest": [2, 1, 2], "dms_mode": [1, 1, 2],
                  "sctg2": [1, 1, 1], "tons_2017": [100, 30, 900]}).to_csv(raw / "demand.csv", index=False)
    pd.DataFrame({"dms_orig": [1, 2], "dms_orig_cnty": [17001, 17003], "sctgG5": ["sctg0109"] * 2, "f_orig": [1, 1]}).to_csv(raw / "origin.csv", index=False)
    pd.DataFrame({"dms_dest": [1, 2], "dms_dest_cnty": [17001, 17003], "sctgG5": ["sctg0109"] * 2, "f_dest": [1, 1]}).to_csv(raw / "destination.csv", index=False)
    ntad = gpd.GeoDataFrame({"OBJECTID": [1], "LNAME": ["TEST"], "SIGN1": ["I 95"], "SIGNT1": ["I"], "SIGNN1": [95],
                            "AADT": [50000], "AADT_COM": [5000], "AADT_SINGL": [1000], "SPEED_LIMI": [65], "DIR": [0]},
                            geometry=[LineString(points)], crs=4269)
    for name in ["strahnet", "nhs", "nn"]:
        ntad.to_file(raw / (name + ".gpkg"), layer=name, driver="GPKG")
    cfg["raw_root"] = str(raw)
    cfg["output_root"] = str(tmp_path / "new_outputs")
    cfg["raw_paths"] = {"network_gdb_path": "network.gpkg", "od_path": "demand.csv", "origin_factors_path": "origin.csv",
                        "destination_factors_path": "destination.csv", "strahnet_gpkg_path": "strahnet.gpkg", "nhs_gpkg_path": "nhs.gpkg", "nn_gpkg_path": "nn.gpkg"}
    return cfg


def test_raw_to_portable_arrays(tmp_path):
    control = json.loads((BASE / "configs/upstream.json").read_text())
    control = make_raw_fixture(tmp_path, control)
    config_path = tmp_path / "control.json"
    config_path.write_text(json.dumps(control))
    for stage in ["validate-inputs", "clean-network", "disaggregate-demand", "enrich-network", "snapshots"]:
        adapter.main(["--config", str(config_path), "--step", stage])
    output = Path(control["output_root"])
    import os
    expected_run_id = os.environ.get("REPRODUCTION_RUN_ID", control["run_id"])
    for stage in ["validate-inputs", "clean-network", "disaggregate-demand", "enrich-network", "snapshots"]:
        assert json.loads((output / (stage + "_status.json")).read_text())["run_id"] == expected_run_id
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["graph_node_count"] == 3
    assert receipt["graph_arc_count"] == 4
    assert receipt["raw_od_rows"] == 2
    assert receipt["raw_od_annual_ktons"] == 130
    assert all(Path(value).is_file() for value in receipt["graph_paths"].values())
    assert all(Path(value).is_file() for value in receipt["od_paths"].values())
    with np.load(receipt["graph_paths"]["npz"]) as graph:
        assert np.all(graph["free_time"] > 0)
        assert np.all(graph["capacity"] > 0)
        assert int((graph["length"] == 0).sum()) == 2  # Both zero-length directions retained.
        assert sorted(graph["capacity_multiplier"]) == sorted([100, 100, 2/3, 2/3])
    # Interface consumers reopen the generated files rather than depending on
    # this fixture's in-memory producer objects.
    from partC_rev1_src.phase4.inputs import load_phase4_graph, load_or_build_aggregated_od
    frozen = load_phase4_graph({"input_versions": receipt["input_versions"]})
    aggregate = load_or_build_aggregated_od({"input_versions": receipt["input_versions"]}, output_root=output / "phase4")
    assert len(frozen.tail) == 4
    assert float(aggregate.demand.sum()) == 130
