"""Configuration loading and validation for partC_rev1."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .partC_rev1_constants import (
    PARTC_REV1_DEFAULT_ALPHA,
    PARTC_REV1_DEFAULT_BETA,
    PARTC_REV1_INF_COST,
    PARTC_REV1_REQUIRED_CONFIG_SECTIONS,
    PARTC_REV1_SOLVER_ALGORITHMS,
    PARTC_REV1_SURROGATE_MODEL_TYPES,
)
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json
from .partC_rev1_paths import (
    partC_rev1_assert_prefixed_path_name,
    partC_rev1_ensure_configured_folders,
    partC_rev1_resolve_path,
)


_PARTC_REV1_DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "run_id": "partC_rev1_default",
        "schema_version": "partC_rev1",
        "random_seed": 42,
        "strict_inputs": False,
        "source_code_dir": "partC_rev1_src",
        "tests_dir": "partC_rev1_tests",
        "config_dir": "partC_rev1_configs",
        "cache_dir": "partC_rev1_cache",
        "output_dir": "partC_rev1_outputs",
        "docs_dir": "partC_rev1_docs",
    },
    "inputs": {
        "network_gdb_path": "./faf_assignment_result/FAF5_Model_Highway_Network/Networks/Geodatabase Format/FAF5Network.gdb",
        "network_dictionary_pdf": "./faf_assignment_result/FAF5_Model_Highway_Network/Networks/Geodatabase Format/FAF5 Network Data Dictionary.pdf",
        "links_layer": "FAF5_Links",
        "nodes_layer": "FAF5_Nodes",
        "od_path": "./faf_assignment_result/FAF5.6.1.csv",
        "faf_metadata_path": "./faf_assignment_result/FAF5_metadata.xlsx",
        "origin_factors_path": "./faf_assignment_result/All_Experimental_Disaggregation_Factors/truck_origin_factors.csv",
        "destination_factors_path": "./faf_assignment_result/All_Experimental_Disaggregation_Factors/truck_destination_factors.csv",
        "ground_truth_flow_dir": "./faf_assignment_result/FAF5_2017_HighwayAssignmentResults_04_07_2022/Assignment Flow Tables/CSV Format",
        "ground_truth_flow_path": "",
        "ground_truth_dictionary_pdf": "./faf_assignment_result/FAF5_2017_HighwayAssignmentResults_04_07_2022/Assignment Flow Tables/CSV Format/FAF5 Assignment Flow Tables Data Dictionary.pdf",
    },
    "inspection_defaults": {
        "gdb_driver": "OpenFileGDB",
        "gdb_crs": "EPSG:4269",
        "link_layer_geometry_type": "MultiLineString",
        "node_layer_geometry_type": "Point",
        "link_feature_count_observed": 487394,
        "node_feature_count_observed": 974788,
        "node_unique_source_id_count_observed": 348495,
        "node_duplicate_source_ids_observed": True,
        "link_from_to_columns_observed": False,
        "requires_endpoint_snapping": True,
    },
    "input_schema": {
        "link_id_col": "ID",
        "node_id_col": "ID",
        "node_source_data_col": "DATA",
        "link_from_col_candidates": ["A", "Anode", "A_Node", "FromNodeID", "FROM_ID", "from_node", "tail", "Tail"],
        "link_to_col_candidates": ["B", "Bnode", "B_Node", "ToNodeID", "TO_ID", "to_node", "head", "Head"],
        "dir_col": "DIR",
        "length_col": "LENGTH",
        "class_col": "Class",
        "class_description_col": "Class_Description",
        "nhs_col": "NHS",
        "nhfn_col": "NHFN",
        "road_name_col": "Road_Name",
        "sign_route_col": "Sign_Rte",
        "route_type_col": "Rte_Type",
        "route_number_col": "Rte_Number",
        "truck_col": "Truck",
        "status_col": "Status",
        "ctfips_col": "CTFIPS",
        "fafzone_col": "FAFZONE",
        "ab_time_col": "AB_FreeFlowTime",
        "ba_time_col": "BA_FreeFlowTime",
        "ab_speed_col": "AB_FinalSpeed",
        "ba_speed_col": "BA_FinalSpeed",
        "ab_lanes_col": "AB_Lanes",
        "ba_lanes_col": "BA_Lanes",
        "node_lon_col": "",
        "node_lat_col": "",
        "node_geometry_source": "geometry",
        "centroid_col": "Centroid",
        "centroid_id_col": "CentroidID",
        "fafid_col": "FAFID",
        "node_state_col": "State",
        "node_county_col": "County",
    },
    "preprocess": {
        "cache_sqlite": "partC_rev1_cache/partC_rev1_network.sqlite",
        "deduplicate_nodes": True,
        "deduplicate_links": True,
        "duplicate_policy": "drop_exact_duplicate_rows_only",
        "node_dedupe_keys": ["ID", "DATA", "geometry"],
        "link_dedupe_keys": "all_columns_including_geometry",
        "complete_directed_graph_after_cleaning": True,
        "preserve_existing_rows_when_completing_graph": True,
        "assign_new_link_index": True,
        "assign_new_node_index": True,
        "snap_after_clean_physical_graph": True,
        "capacity_per_lane": 2000.0,
        "default_speed_mph": 45.0,
        "default_lanes": 1.0,
        "default_alpha": PARTC_REV1_DEFAULT_ALPHA,
        "default_beta": PARTC_REV1_DEFAULT_BETA,
        "large_cost": PARTC_REV1_INF_COST,
        "snap_tolerance_meters": 100.0,
        "snap_projected_crs": "EPSG:5070",
        "exclude_truck_prohibited": True,
        "require_status_open": True,
        "unknown_dir_as_two_way": False,
        "fail_on_unsnapped_endpoint": True,
        "fail_on_duplicate_link_ids": False,
    },
    "data_cleaning": {
        "drop_exact_duplicate_nodes": True,
        "drop_exact_duplicate_links": True,
        "report_same_geometry_nonidentical_nodes": True,
        "report_same_geometry_nonidentical_links": True,
        "geometry_comparison_precision": 8,
        "deduplicate_geometry_using_wkb": True,
    },
    "direction_inference": {
        "method": "lanes_primary",
        "ab_lanes_col": "AB_Lanes",
        "ba_lanes_col": "BA_Lanes",
        "positive_lane_threshold": 0.0,
        "fallback_to_dir_when_lanes_missing": False,
        "assume_missing_lanes_bidirectional": True,
        "missing_lane_default_lanes": 1.0,
        "missing_lane_direction_case": "DEFAULT_AB_BA_1_LANE",
        "append_missing_lane_arcs_after_lane_inferred": True,
        "dir_col": "Dir",
        "dir_two_way_values": [0, "0"],
        "dir_ab_only_values": [1, "1"],
        "dir_ba_only_values": [],
        "unknown_dir_policy": "report_and_ignore_if_no_lane_info",
    },
    "unit_conversion": {
        "solver_flow_unit": "annual_ktons",
        "od_raw_unit": "annual_ktons",
        "ground_truth_preferred_unit": "annual_ktons",
        "annual_ktons_to_annual_tons_factor": 1000.0,
        "annual_operating_days": 365.0,
        "average_payload_tons_per_truck": 18.0,
        "allow_unit_conversion": True,
        "strict_no_unit_mixing": True,
    },
    "direction": {
        "prefer_ab_ba_lanes_when_available": True,
        "lane_direction_policy": "both_positive_bidirectional_one_positive_unidirectional_no_info_ignore",
        "no_lane_info_policy": "ignore_link",
        "dir_two_way_values": [0, "0"],
        "dir_ab_only_values": [1, "1"],
        "dir_ba_only_values": [-1, "-1"],
        "dir_negative_policy": "ba_only",
        "flag_negative_dir_for_review": True,
    },
    "od": {
        "baseline_year": 2017,
        "mode": "faf_dms_to_county_to_centroid",
        "truck_mode_codes": [1, "1", "Truck", "truck"],
        "demand_col_candidates": ["tons_2017", "Tons_2017", "tons2017", "tons_17", "tons"],
        "demand_col": "tons_2017",
        "demand_units": "annual_ktons",
        "origin_col": "dms_orig",
        "destination_col": "dms_dest",
        "mode_col_candidates": ["dms_mode", "mode", "Mode"],
        "mode_col": "dms_mode",
        "commodity_col": "sctg2",
        "commodity_group_col": "sctgG5",
        "origin_factor_region_col": "dms_orig",
        "origin_factor_county_col": "dms_orig_cnty",
        "origin_factor_col": "f_orig",
        "destination_factor_region_col": "dms_dest",
        "destination_factor_county_col": "dms_dest_cnty",
        "destination_factor_col": "f_dest",
        "normalize_factors": False,
        "county_fips_width": 5,
        "split_county_to_centroids": "equal",
        "drop_zero_demand": True,
        "sctg2_to_sctgG5": {
            "1-9": "sctg0109",
            "10-14": "sctg1014",
            "15-19": "sctg1519",
            "20-33": "sctg2033",
            "34-99": "sctg3499",
        },
    },
    "ground_truth": {
        "candidate_files": [
            "FAF5 Domestic Truck Flows by Commodity_2017.csv",
            "FAF5 Export Truck Flows by Commodity_2017.csv",
            "FAF5 Import Truck Flows by Commodity_2017.csv",
            "FAF5 Total CU Truck Flows by Commodity_2017.csv",
            "FAF5 Total SU Truck Flows by Commodity_2017.csv",
            "FAF5 Total Truck Flows by Commodity_2017.csv",
        ],
        "recommended_file": "FAF5 Total Truck Flows by Commodity_2017.csv",
        "link_id_col": "ID",
        "default_total_flow_col": "TOT Tons_17 All",
        "default_ab_flow_col": "AB Tons_17 All",
        "default_ba_flow_col": "BA Tons_17 All",
        "default_total_trips_col": "TOT Trips_17 All",
        "normalized_flow_col": "flow",
        "flow_units": "annual_ktons",
        "validation_join": "left_join_outputs_to_ground_truth_by_ID",
        "missing_ground_truth_flow_value": 0.0,
        "preferred_year_suffix": "17",
        "preferred_type_suffix": "All",
        "preferred_flow_unit": "annual_ktons",
        "preferred_flow_col_candidates": [
            "TOT Tons_17 All",
            "TOT_Tons_17_All",
            "TOT Tons_2017 All",
            "TOT_Tons_2017_All",
            "flow",
            "Flow",
        ],
        "id_col_candidates": ["ID", "id", "LinkID", "link_id"],
        "missing_ground_truth_flow_default_zero": True,
        "add_ground_truth_to_clean_links": True,
        "ground_truth_clean_col": "partC_rev1_ground_truth_flow",
    },
    "clean_geodatabase": {
        "write_clean_geodatabase": True,
        "preferred_format": "gpkg",
        "output_gpkg_path": "partC_rev1_cache/partC_rev1_clean_network.gpkg",
        "attempt_filegdb_if_supported": False,
        "output_filegdb_path": "partC_rev1_cache/partC_rev1_clean_network.gdb",
        "layers": {
            "clean_nodes": "partC_rev1_clean_physical_nodes",
            "clean_links": "partC_rev1_clean_physical_links",
            "directed_arcs": "partC_rev1_directed_arcs",
        },
    },
    "ntad": {
        "strahnet_gpkg_path": "./faf_assignment_result/NTAD_Strategic_Highway_Network_2899699978735006194.gpkg",
        "nhs_gpkg_path": "./faf_assignment_result/NTAD_National_Highway_System_583647460914501285.gpkg",
        "nhs_gpkg_glob": "./faf_assignment_result/*National_Highway_System*.gpkg",
        "nn_gpkg_path": "./faf_assignment_result/NTAD_National_Network_4561358941476452483.gpkg",
        "nn_gpkg_glob": "./faf_assignment_result/*National_Network*.gpkg",
        "strahnet_layer": "",
        "nhs_layer": "",
        "nn_layer": "",
        "projected_crs": "EPSG:5070",
    },
    "matching": {
        "enabled": True,
        "matching_algorithm_version": "buffer_geo_v3",
        "source_priority": ["STRAHNET", "NHS", "NN"],
        "allow_spatial_only_if_no_name": True,
        "collect_lower_priority_candidates_for_diagnostics": False,
        "geo_buffer_radius_miles": 0.08,
        "geo_buffer_radius_m": None,
        "score_weight_buffer_overlap": 0.35,
        "score_weight_line_distance": 0.25,
        "score_weight_angle_parallel": 0.20,
        "score_weight_endpoint_distance": 0.10,
        "score_weight_semantic": 0.07,
        "score_weight_fips": 0.03,
        "close_competitor_ratio_threshold": 0.95,
        "initial_search_radius_m": 150.0,
        "fallback_search_radius_m": 500.0,
        "max_search_radius_m": 1500.0,
        "endpoint_distance_scale_m": 100.0,
        "line_distance_scale_m": 100.0,
        "angle_tolerance_deg": 30.0,
        "max_angle_diff_deg": 30.0,
        "max_match_distance_m": 500.0,
        "min_name_or_route_similarity": 0.65,
        "match_score_threshold": 0.62,
        "ambiguity_ratio_threshold": 0.95,
        "spatial_only_match_score_threshold": 0.72,
        "spatial_only_max_angle_diff_deg": 20.0,
        "spatial_only_max_match_distance_m": 120.0,
        "spatial_only_min_overlap_score": 0.10,
        "spatial_only_max_ambiguity_ratio": 0.90,
        "overlap_buffer_m": 50.0,
        "inherit_opposite_direction_matches": True,
        "max_candidates_per_link": 30,
        "max_geometry_candidates_per_link": 120,
        "keep_candidate_rows_per_source": 2,
        "max_faf_links": None,
        "max_ntad_candidates_per_source": None,
        "reuse_cached_best_matches": False,
        "dry_run": False,
    },
    "capacity_enrichment": {
        "capacity_units": "annual_ktons",
        "payload_tons_per_truck": 18.0,
        "truck_pce": 2.0,
        "default_truck_ratio": 0.12,
        "direction_split": 0.5,
        "k_factor": 0.10,
        "d_factor": 0.55,
        "urban_phf": 0.90,
        "rural_phf": 0.88,
        "fallback_phf": 0.90,
        "min_remaining_capacity_fraction": 0.05,
        "default_speed_mph": 45.0,
        "default_speed_by_faf_class": {
            "11": 65,
            "12": 60,
            "13": 55,
            "14": 45,
            "15": 30,
            "16": 25,
            "17": 20,
            "18": 20,
            "19": 20,
            "21": 35,
            "22": 35,
            "23": 45,
            "33": 60,
            "36": 15,
            "41": 15,
            "50": 20,
        },
        "default_capacity_vphpl_by_faf_class": {
            "11": 2400,
            "12": 2200,
            "13": 2000,
            "14": 1800,
            "15": 900,
            "16": 900,
            "17": 800,
            "18": 800,
            "19": 800,
            "21": 1600,
            "22": 1600,
            "23": 2000,
            "33": 2200,
            "36": 600,
            "41": 300,
            "50": 600,
        },
        "default_aadt_by_faf_class": {
            "11": 50000,
            "12": 35000,
            "13": 25000,
            "14": 15000,
            "15": 3000,
            "16": 2000,
            "17": 5000,
            "18": 5000,
            "19": 1500,
            "21": 12000,
            "22": 12000,
            "23": 20000,
            "33": 30000,
            "36": 500,
            "41": 300,
            "50": 1000,
        },
    },
    "assignment_capacity_multipliers": {
        "enabled": True,
        "centroid_connector_multiplier": 100.0,
        "overall_capacity_multiplier": 0.3,
        "centroid_connector_class_codes": [50],
    },
    "backbone": {
        "use_nhs": True,
        "nhs_include_codes": [1, 3, 4, 7, 8, 9, 10, 12],
        "nhs_exclude_codes": [11],
        "use_nhfn": True,
        "nhfn_include_codes": [1, 2, 12],
        "nhfn_exclude_codes": [11],
        "use_class": True,
        "class_include_codes": [11, 12, 13, 21, 22, 23, 33],
        "status_open_values": [1, "1"],
        "truck_prohibited_values": ["Prohibited"],
    },
    "backbone_partition": {
        "main_class_codes": [11, 12, 13, 21, 22, 23, 33],
        "use_strahnet_nonempty": True,
        "do_not_exclude_nhs_11": True,
        "do_not_exclude_nhfn_11": True,
        "use_supernodes": False,
        "seed_lcc": {
            "enabled": True,
            "component_selection": "largest_weak_component",
        },
        "exclude_centroid_connectors_from_expansion": True,
        "centroid_connector_class_codes": [50],
        "direct_connectors": {
            "enabled": True,
            "max_length_m": 300.0,
            "max_degree": 1,
            "exclude_centroid_connectors": True,
        },
        "candidate_expansion": {
            "enabled": False,
            "undirected": True,
            "geo_radius_m": 5000.0,
            "max_path_fftt_min": 5.0,
            "max_path_length_miles": 3.0,
            "max_hops": 10,
            "max_internal_candidate_degree": 4,
            "max_paths_per_interface": 20,
        },
        "supernodes": {
            "enabled": False,
            "undirected_reachability": True,
            "geo_radius_m": 200.0,
            "max_reachable_fftt_min": 1.0,
            "max_cluster_diameter_m": 500.0,
            "max_cluster_fftt_min": 2.0,
            "max_cluster_nodes": 100,
            "no_distance_only_merge": True,
        },
        "parallel_backbone_arcs": {
            "merge": True,
            "group_by_route_key_if_available": False,
            "capacity_aggregation": "sum",
            "aadt_aggregation": "sum",
            "fftt_aggregation": "capacity_weighted_average",
            "alpha_beta_conflict_policy": "capacity_weighted_average_with_warning",
        },
        "partition": {
            "method": "igraph_leiden",
            "graph_directed": False,
            "objective_function": "modularity",
            "resolution": 1.0,
            "beta": 0.01,
            "n_iterations": -1,
            "edge_weight_mode": "capacity_over_fftt",
            "random_seed": 42,
            "exclude_backbone_nodes_as_internal_region_vertices": True,
        },
        "stability": {
            "enabled": True,
            "run_on_real_data_in_prompt3B_only": True,
            "vary_backbone_labels": False,
            "resolution_values": [0.5, 0.75, 1.0, 1.25, 1.5],
            "random_seeds": [42, 43, 44],
            "compute_pairwise_similarity": True,
            "compute_complexity_proxies": True,
            "reuse_default_physical_graph_for_all_runs": True,
        },
    },
    "partition": {
        "method": "connected_components",
        "existing_region_col": "",
        "fallback_methods": ["FAFZONE", "CTFIPS"],
        "resolution_values": [0.5, 1.0, 1.5],
        "random_seed": 42,
        "require_every_non_backbone_arc_assigned": True,
    },
    "portal": {
        "cluster_method": "radius",
        "max_geo_radius_miles": 5.0,
        "max_portals_per_region": 12,
        "split_entry_exit_nodes": True,
        "split_origin_destination_centroids": True,
        "never_cluster_different_neighbor_regions": True,
        "forbid_backbone_to_portal_out": True,
        "forbid_portal_in_to_backbone": True,
        "forbid_centroid_dest_outgoing": True,
        "forbid_centroid_origin_incoming": True,
        "forbid_portal_out_to_portal_in_same_region": True,
    },
    "regional_training": {
        "enabled": False,
        "num_samples_per_region": 100,
        "max_regions_for_debug": 3,
        "sample_strategy": "lognormal",
        "od_sampler": "legacy_sparse_ipf_lognormal",
        "regional_solver": "msa",
        "store_training_samples": False,
    },
    "model": {
        "model_type": "freeflow",
        "icnn_hidden_units": [64, 64],
        "icnn_mu": 0.001,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
    },
    "solver": {
        "algorithm": "portal_overlay_fw",
        "max_iter": 50,
        "relative_gap_tol": 0.001,
        "line_search": "bisection",
        "line_search_tol": 0.000001,
        "num_threads": 1,
        "one_tree_per_origin": True,
        "reverse_load_destinations": True,
        "materialize_regional_cliques": False,
        "regional_transfer_storage": "flat_arrays_grouped_by_source",
    },
    "benchmarks": {
        "run_physical_msa": True,
        "run_physical_fw": True,
        "run_physical_cfw": False,
        "max_iter": 50,
        "do_not_claim_bar_gera_or_dial": True,
    },
    "experiments": {
        "run_validation": True,
        "run_partition_stability": False,
        "run_learning_curves": False,
        "run_i95_closure": False,
    },
    "outputs": {
        "output_dir": "partC_rev1_outputs",
        "docs_dir": "partC_rev1_docs",
        "physical_flow_filename": "partC_rev1_physical_flow.csv",
        "virtual_flow_filename": "partC_rev1_virtual_flow.csv",
        "flow_column_name": "partC_rev1_flow",
        "ordinary_virtual_id_prefix": "ord_",
        "regional_transfer_id_prefix": "reg_",
        "overwrite": False,
        "preserve_original_physical_link_columns": True,
    },
    "testing": {
        "use_toy_data": True,
        "national_scale_in_unit_tests": False,
        "require_unit_tests_for_every_module": True,
    },
}


def partC_rev1_default_config() -> dict[str, Any]:
    return copy.deepcopy(_PARTC_REV1_DEFAULT_CONFIG)


def _partC_rev1_require_mapping(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid config section: {section}")
    return value


def _partC_rev1_positive(section: Mapping[str, Any], key: str, *, section_name: str, min_value: float = 0.0) -> None:
    value = section.get(key)
    if value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section_name}.{key} must be numeric") from exc
    if numeric <= min_value:
        raise ValueError(f"{section_name}.{key} must be > {min_value}: {value}")


def _partC_rev1_nonnegative(section: Mapping[str, Any], key: str, *, section_name: str) -> None:
    value = section.get(key)
    if value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section_name}.{key} must be numeric") from exc
    if numeric < 0:
        raise ValueError(f"{section_name}.{key} must be nonnegative: {value}")


def _partC_rev1_require_list(section: Mapping[str, Any], key: str, *, section_name: str) -> None:
    if not isinstance(section.get(key), list):
        raise ValueError(f"{section_name}.{key} must be a list")


def _partC_rev1_validate_created_artifact_names(config: Mapping[str, Any]) -> None:
    project = _partC_rev1_require_mapping(config, "project")
    outputs = _partC_rev1_require_mapping(config, "outputs")
    preprocess = _partC_rev1_require_mapping(config, "preprocess")

    for key in ("source_code_dir", "tests_dir", "config_dir", "cache_dir", "output_dir", "docs_dir"):
        if key in project:
            partC_rev1_assert_prefixed_path_name(str(project[key]), label=f"project.{key}")
    for key in ("output_dir", "docs_dir", "physical_flow_filename", "virtual_flow_filename"):
        if key in outputs:
            partC_rev1_assert_prefixed_path_name(str(outputs[key]), label=f"outputs.{key}")
    if "cache_sqlite" in preprocess:
        cache_path = Path(str(preprocess["cache_sqlite"]))
        partC_rev1_assert_prefixed_path_name(cache_path.name, label="preprocess.cache_sqlite")
        if cache_path.parent != Path("."):
            partC_rev1_assert_prefixed_path_name(cache_path.parent.name, label="preprocess.cache_sqlite parent")


def _partC_rev1_validate_input_paths(config: dict[str, Any], strict_inputs: bool) -> None:
    inputs = _partC_rev1_require_mapping(config, "inputs")
    required_path_keys = (
        "network_gdb_path",
        "od_path",
        "origin_factors_path",
        "destination_factors_path",
        "ground_truth_flow_dir",
    )
    warnings = config.setdefault("_partC_rev1_warnings", [])
    for key in required_path_keys:
        value = inputs.get(key, "")
        if value is None or str(value).strip() == "":
            message = f"inputs.{key} is empty"
            if strict_inputs:
                raise FileNotFoundError(message)
            warnings.append(message)
            continue
        resolved = partC_rev1_resolve_path(str(value))
        if not resolved.exists():
            message = f"inputs.{key} does not exist: {resolved}"
            if strict_inputs:
                raise FileNotFoundError(message)
            warnings.append(message)


def _partC_rev1_validate_backbone_partition(cfg: dict[str, Any]) -> None:
    section = _partC_rev1_require_mapping(cfg, "backbone_partition")
    seed_lcc = _partC_rev1_require_mapping(section, "seed_lcc")
    direct_connectors = _partC_rev1_require_mapping(section, "direct_connectors")
    expansion = section.get("candidate_expansion", {})
    if expansion is not None and not isinstance(expansion, dict):
        raise ValueError("backbone_partition.candidate_expansion must be a mapping when present")
    supernodes = section.get("supernodes", {})
    if supernodes is not None and not isinstance(supernodes, dict):
        raise ValueError("backbone_partition.supernodes must be a mapping when present")
    parallel = _partC_rev1_require_mapping(section, "parallel_backbone_arcs")
    partition = _partC_rev1_require_mapping(section, "partition")
    stability = _partC_rev1_require_mapping(section, "stability")
    warnings = cfg.setdefault("_partC_rev1_warnings", [])

    _partC_rev1_require_list(section, "main_class_codes", section_name="backbone_partition")
    _partC_rev1_require_list(section, "centroid_connector_class_codes", section_name="backbone_partition")
    if seed_lcc.get("component_selection", "largest_weak_component") != "largest_weak_component":
        raise ValueError("backbone_partition.seed_lcc.component_selection must be 'largest_weak_component'")
    _partC_rev1_positive(direct_connectors, "max_length_m", section_name="backbone_partition.direct_connectors")
    _partC_rev1_positive(direct_connectors, "max_degree", section_name="backbone_partition.direct_connectors", min_value=0)
    int(direct_connectors.get("max_degree", 1))
    if bool(section.get("use_supernodes", False)):
        warnings.append("backbone_partition.use_supernodes is ignored for Prompt 3A/3B; physical nodes are partitioned directly.")
    if bool(expansion.get("enabled", False)):
        warnings.append("backbone_partition.candidate_expansion is obsolete and ignored; Prompt 3 uses direct_connectors only.")
    if bool(supernodes.get("enabled", False)):
        warnings.append("backbone_partition.supernodes is obsolete and ignored; Prompt 3 uses physical nodes directly.")

    if parallel.get("capacity_aggregation") != "sum":
        raise ValueError("backbone_partition.parallel_backbone_arcs.capacity_aggregation must be 'sum'")
    if parallel.get("aadt_aggregation") != "sum":
        raise ValueError("backbone_partition.parallel_backbone_arcs.aadt_aggregation must be 'sum'")
    if parallel.get("fftt_aggregation") != "capacity_weighted_average":
        raise ValueError("backbone_partition.parallel_backbone_arcs.fftt_aggregation must be 'capacity_weighted_average'")

    if partition.get("method") != "igraph_leiden":
        raise ValueError("backbone_partition.partition.method must be 'igraph_leiden'")
    if bool(partition.get("graph_directed", False)):
        raise ValueError("backbone_partition.partition.graph_directed must be false for Prompt 3A")
    if partition.get("objective_function") not in {"modularity", "CPM"}:
        raise ValueError("backbone_partition.partition.objective_function must be 'modularity' or 'CPM'")
    if partition.get("edge_weight_mode") != "capacity_over_fftt":
        raise ValueError("backbone_partition.partition.edge_weight_mode must be 'capacity_over_fftt'")
    if bool(partition.get("exclude_backbone_supernodes_as_internal_region_vertices", False)):
        warnings.append("backbone_partition.partition.exclude_backbone_supernodes_as_internal_region_vertices is obsolete; using physical backbone nodes instead.")
    if not bool(partition.get("exclude_backbone_nodes_as_internal_region_vertices", True)):
        raise ValueError("backbone_partition.partition.exclude_backbone_nodes_as_internal_region_vertices must be true")
    _partC_rev1_positive(partition, "resolution", section_name="backbone_partition.partition")
    _partC_rev1_positive(partition, "beta", section_name="backbone_partition.partition", min_value=0.0)
    try:
        int(partition.get("n_iterations", -1))
        int(partition.get("random_seed", 42))
    except (TypeError, ValueError) as exc:
        raise ValueError("backbone_partition.partition n_iterations and random_seed must be integers") from exc

    _partC_rev1_require_list(stability, "resolution_values", section_name="backbone_partition.stability")
    _partC_rev1_require_list(stability, "random_seeds", section_name="backbone_partition.stability")
    if bool(stability.get("vary_backbone_labels", False)):
        raise ValueError("backbone_partition.stability.vary_backbone_labels must be false for Prompt 3A")
    if "reuse_default_backbone_abstraction_for_all_runs" in stability:
        warnings.append(
            "backbone_partition.stability.reuse_default_backbone_abstraction_for_all_runs is obsolete; use reuse_default_physical_graph_for_all_runs."
        )
    reuse_physical_graph = stability.get(
        "reuse_default_physical_graph_for_all_runs",
        stability.get("reuse_default_backbone_abstraction_for_all_runs", True),
    )
    if not bool(reuse_physical_graph):
        raise ValueError("backbone_partition.stability.reuse_default_physical_graph_for_all_runs must be true")
    for value in stability.get("resolution_values", []):
        if float(value) <= 0:
            raise ValueError("backbone_partition.stability.resolution_values must be positive")
    for value in stability.get("random_seeds", []):
        int(value)
    if "label_variants" in stability:
        warnings.append("backbone_partition.stability.label_variants is obsolete for Prompt 3A and will be ignored.")


def validate_partC_rev1_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("partC_rev1 config must be a dictionary")

    cfg = copy.deepcopy(config)
    missing = [section for section in PARTC_REV1_REQUIRED_CONFIG_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"Missing required top-level config sections: {missing}")

    project = _partC_rev1_require_mapping(cfg, "project")
    preprocess = _partC_rev1_require_mapping(cfg, "preprocess")
    backbone = _partC_rev1_require_mapping(cfg, "backbone")
    _partC_rev1_require_mapping(cfg, "backbone_partition")
    portal = _partC_rev1_require_mapping(cfg, "portal")
    regional_training = _partC_rev1_require_mapping(cfg, "regional_training")
    model = _partC_rev1_require_mapping(cfg, "model")
    solver = _partC_rev1_require_mapping(cfg, "solver")
    benchmarks = _partC_rev1_require_mapping(cfg, "benchmarks")
    outputs = _partC_rev1_require_mapping(cfg, "outputs")
    matching = cfg.get("matching", {})
    capacity_enrichment = cfg.get("capacity_enrichment", {})
    assignment_capacity = cfg.get("assignment_capacity_multipliers", {})

    _partC_rev1_validate_created_artifact_names(cfg)
    partC_rev1_ensure_configured_folders(cfg)
    _partC_rev1_validate_backbone_partition(cfg)

    algorithm = solver.get("algorithm")
    if algorithm not in PARTC_REV1_SOLVER_ALGORITHMS:
        raise ValueError(f"Unsupported solver.algorithm {algorithm!r}; expected one of {PARTC_REV1_SOLVER_ALGORITHMS}")

    model_type = model.get("model_type")
    if model_type not in PARTC_REV1_SURROGATE_MODEL_TYPES:
        raise ValueError(f"Unsupported model.model_type {model_type!r}; expected one of {PARTC_REV1_SURROGATE_MODEL_TYPES}")

    flow_col = outputs.get("flow_column_name")
    if not isinstance(flow_col, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", flow_col):
        raise ValueError(f"outputs.flow_column_name must be a valid column identifier: {flow_col!r}")

    for key in ("nhs_include_codes", "nhs_exclude_codes", "nhfn_include_codes", "nhfn_exclude_codes", "class_include_codes"):
        _partC_rev1_require_list(backbone, key, section_name="backbone")
    if isinstance(matching, Mapping) and "source_priority" in matching:
        priority = list(matching.get("source_priority", []))
        if priority != ["STRAHNET", "NHS", "NN"]:
            raise ValueError("matching.source_priority must be ['STRAHNET', 'NHS', 'NN']")

    _partC_rev1_positive(preprocess, "capacity_per_lane", section_name="preprocess")
    _partC_rev1_positive(preprocess, "default_speed_mph", section_name="preprocess")
    _partC_rev1_positive(preprocess, "default_lanes", section_name="preprocess")
    _partC_rev1_nonnegative(preprocess, "default_alpha", section_name="preprocess")
    _partC_rev1_positive(preprocess, "default_beta", section_name="preprocess")
    _partC_rev1_positive(preprocess, "large_cost", section_name="preprocess")
    _partC_rev1_positive(preprocess, "snap_tolerance_meters", section_name="preprocess")
    _partC_rev1_positive(portal, "max_geo_radius_miles", section_name="portal")
    _partC_rev1_positive(portal, "max_portals_per_region", section_name="portal")
    _partC_rev1_positive(regional_training, "num_samples_per_region", section_name="regional_training")
    _partC_rev1_positive(model, "learning_rate", section_name="model")
    _partC_rev1_positive(model, "epochs", section_name="model")
    _partC_rev1_positive(model, "batch_size", section_name="model")
    _partC_rev1_nonnegative(model, "icnn_mu", section_name="model")
    _partC_rev1_positive(solver, "max_iter", section_name="solver")
    _partC_rev1_positive(solver, "relative_gap_tol", section_name="solver")
    _partC_rev1_positive(solver, "line_search_tol", section_name="solver")
    _partC_rev1_positive(solver, "num_threads", section_name="solver")
    _partC_rev1_positive(benchmarks, "max_iter", section_name="benchmarks")
    if isinstance(matching, Mapping):
        for key in (
            "initial_search_radius_m",
            "fallback_search_radius_m",
            "max_search_radius_m",
            "endpoint_distance_scale_m",
            "line_distance_scale_m",
            "angle_tolerance_deg",
            "max_angle_diff_deg",
            "max_match_distance_m",
            "match_score_threshold",
            "spatial_only_match_score_threshold",
            "overlap_buffer_m",
            "geo_buffer_radius_miles",
            "geo_buffer_radius_m",
            "score_weight_buffer_overlap",
            "score_weight_line_distance",
            "score_weight_angle_parallel",
            "score_weight_endpoint_distance",
            "score_weight_semantic",
            "score_weight_fips",
            "close_competitor_ratio_threshold",
            "max_geometry_candidates_per_link",
        ):
            _partC_rev1_positive(matching, key, section_name="matching")
    if isinstance(capacity_enrichment, Mapping):
        for key in ("payload_tons_per_truck", "truck_pce", "direction_split", "k_factor", "d_factor", "fallback_phf"):
            _partC_rev1_positive(capacity_enrichment, key, section_name="capacity_enrichment")
    if "assignment_capacity_multipliers" in cfg and isinstance(assignment_capacity, Mapping):
        for key in ("centroid_connector_multiplier", "overall_capacity_multiplier"):
            _partC_rev1_positive(assignment_capacity, key, section_name="assignment_capacity_multipliers")
        _partC_rev1_require_list(assignment_capacity, "centroid_connector_class_codes", section_name="assignment_capacity_multipliers")
        for value in assignment_capacity.get("centroid_connector_class_codes", []):
            int(float(value))
    if "od_sampler" in regional_training:
        sampler = str(regional_training.get("od_sampler"))
        allowed_samplers = {"legacy_sparse_ipf_lognormal", "benchmark_flow_gravity", "benchmark_flow_gravity_additive"}
        if sampler not in allowed_samplers:
            raise ValueError(f"Unsupported regional_training.od_sampler {sampler!r}; expected one of {sorted(allowed_samplers)}")

    strict_inputs = bool(project.get("strict_inputs", False))
    _partC_rev1_validate_input_paths(cfg, strict_inputs)
    return cfg


def load_partC_rev1_config(config_path: str) -> dict[str, Any]:
    return validate_partC_rev1_config(partC_rev1_read_json(config_path))


def write_partC_rev1_default_config(path: str) -> None:
    partC_rev1_write_json(partC_rev1_default_config(), path)
