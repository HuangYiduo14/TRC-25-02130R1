"""Prompt 3B partition-stability experiment over Leiden resolution and seed only."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_partition_igraph import partC_rev1_assign_physical_arc_regions, partC_rev1_run_leiden_partition
from .partC_rev1_paths import partC_rev1_output_file


def partC_rev1_nonbackbone_graph_hash(nonbackbone_arcs: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "directed_arc_id",
            "partC_rev1_link_uid",
            "tail_node_uid",
            "head_node_uid",
            "from_node_uid",
            "to_node_uid",
            "capacity",
            "partC_rev1_capacity_annual_ktons_dir",
            "free_time_min",
            "partC_rev1_fftt_min",
        )
        if column in nonbackbone_arcs.columns
    ]
    frame = nonbackbone_arcs[columns].copy() if columns else pd.DataFrame(index=nonbackbone_arcs.index)
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values(columns, kind="mergesort").reset_index(drop=True) if columns else frame
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partC_rev1_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {f"mean_{prefix}": 0.0, f"median_{prefix}": 0.0, f"p95_{prefix}": 0.0, f"max_{prefix}": 0.0}
    return {
        f"mean_{prefix}": float(numeric.mean()),
        f"median_{prefix}": float(numeric.median()),
        f"p95_{prefix}": float(numeric.quantile(0.95)),
        f"max_{prefix}": float(numeric.max()),
    }


def _partC_rev1_fixed_metrics(partition_state: dict[str, pd.DataFrame], arc_regions: pd.DataFrame) -> dict[str, int]:
    arcs = partition_state.get("arcs", pd.DataFrame())
    final = partition_state.get("final_backbone_arcs", pd.DataFrame())
    expansion_directed = 0
    if not final.empty and "backbone_stage" in final.columns:
        expansion_directed = int(final["backbone_stage"].astype(str).eq("added_direct_seed_lcc_link").sum())
    seed_candidate = int(arcs.get("is_seed_candidate_backbone", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not arcs.empty else 0
    seed_lcc = int(arcs.get("is_seed_backbone", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not arcs.empty else len(partition_state.get("backbone_seed_arcs", pd.DataFrame()))
    return {
        "seed_candidate_directed_arcs": seed_candidate,
        "seed_lcc_directed_arcs": seed_lcc,
        "direct_connector_added_directed_arcs": expansion_directed,
        "final_backbone_directed_arcs": int(len(final)),
        "final_backbone_nodes": int(len(partition_state.get("backbone_seed_nodes", pd.DataFrame()))),
        "nonbackbone_physical_directed_arcs": int(len(partition_state.get("nonbackbone_physical_arcs", pd.DataFrame()))),
        "boundary_connector_candidates": int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).str.contains("boundary_connector", na=False).sum()),
    }


def _partC_rev1_region_metrics(regions: pd.DataFrame, arc_regions: pd.DataFrame) -> dict[str, float]:
    internal = arc_regions.loc[arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("regional_internal")]
    internal_counts = internal.groupby("region_id").size() if not internal.empty else pd.Series(dtype=float)
    metrics = {"num_regions": int(len(regions))}
    metrics.update(_partC_rev1_stats(regions.get("node_count", pd.Series(dtype=float)), "region_nodes"))
    metrics.update(_partC_rev1_stats(internal_counts, "region_internal_arcs"))
    metrics["regional_internal_arcs"] = int(len(internal))
    metrics["backbone_region_boundary_connectors"] = int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("backbone_region_boundary_connector").sum())
    metrics["region_region_boundary_connectors"] = int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("region_region_boundary_connector").sum())
    metrics["post_partition_backbone_arcs"] = int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("backbone_network").sum())
    return metrics


def _partC_rev1_complexity_proxy(regions: pd.DataFrame, arc_regions: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=["region_id", "boundary_interface_count", "m_r_est", "m_r_est_cubed"])
    region_ids = regions["region_id"].astype(str)
    arc_region_id = arc_regions.get("region_id", pd.Series(dtype=str)).astype(str)
    arc_type = arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str)
    backbone_counts = (
        arc_regions.loc[arc_type.eq("backbone_region_boundary_connector")]
        .assign(region_id=arc_region_id.loc[arc_type.eq("backbone_region_boundary_connector")])
        .groupby("region_id")["backbone_node_uid"]
        .nunique()
        if {"backbone_node_uid", "region_id"}.issubset(arc_regions.columns)
        else pd.Series(dtype=int)
    )
    neighbor_counts = (
        arc_regions.loc[arc_type.eq("region_region_boundary_connector")]
        .assign(region_id=arc_region_id.loc[arc_type.eq("region_region_boundary_connector")])
        .groupby("region_id")["neighbor_region_id"]
        .nunique()
        if {"neighbor_region_id", "region_id"}.issubset(arc_regions.columns)
        else pd.Series(dtype=int)
    )
    proxy = pd.DataFrame({"region_id": region_ids})
    proxy["boundary_interface_count"] = (
        proxy["region_id"].map(backbone_counts).fillna(0).astype(int)
        + proxy["region_id"].map(neighbor_counts).fillna(0).astype(int)
        + 1
    )
    proxy["m_r_est"] = proxy["boundary_interface_count"] * (proxy["boundary_interface_count"] - 1) + 2 * proxy["boundary_interface_count"]
    proxy["m_r_est_cubed"] = proxy["m_r_est"] ** 3
    return proxy


def _partC_rev1_complexity_metrics(proxy: pd.DataFrame, backbone_arc_count: int) -> dict[str, float]:
    m = pd.to_numeric(proxy.get("m_r_est", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    burden = pd.to_numeric(proxy.get("m_r_est_cubed", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "sum_m_r_est": float(m.sum()),
        "median_m_r_est": float(m.median()) if len(m) else 0.0,
        "p95_m_r_est": float(m.quantile(0.95)) if len(m) else 0.0,
        "max_m_r_est": float(m.max()) if len(m) else 0.0,
        "sum_m_r_est_cubed": float(burden.sum()),
        "proxy_physical_label_graph_size": float(backbone_arc_count + m.sum()),
    }


def _partC_rev1_similarity(run_partitions: dict[tuple[float, int], pd.DataFrame], resolutions: list[float], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def compare(left_key: tuple[float, int], right_key: tuple[float, int], comparison_type: str) -> None:
        if left_key not in run_partitions or right_key not in run_partitions:
            return
        left = run_partitions[left_key].rename(columns={"region_id": "left_region_id"})
        right = run_partitions[right_key].rename(columns={"region_id": "right_region_id"})
        merged = left[["partC_rev1_node_uid", "left_region_id"]].merge(
            right[["partC_rev1_node_uid", "right_region_id"]],
            on="partC_rev1_node_uid",
            how="inner",
        )
        nmi = float(normalized_mutual_info_score(merged["left_region_id"], merged["right_region_id"])) if not merged.empty else np.nan
        ari = float(adjusted_rand_score(merged["left_region_id"], merged["right_region_id"])) if not merged.empty else np.nan
        rows.append(
            {
                "comparison_type": comparison_type,
                "left_resolution": left_key[0],
                "left_seed": left_key[1],
                "right_resolution": right_key[0],
                "right_seed": right_key[1],
                "common_nodes": int(len(merged)),
                "normalized_mutual_info_score": nmi,
                "adjusted_rand_score": ari,
                "nmi": nmi,
                "ari": ari,
            }
        )

    for resolution in resolutions:
        for i, seed in enumerate(seeds):
            for other in seeds[i + 1 :]:
                compare((resolution, seed), (resolution, other), "same_resolution_different_seed")
    for seed in seeds:
        for left, right in zip(resolutions, resolutions[1:]):
            compare((left, seed), (right, seed), "same_seed_adjacent_resolution")
    return pd.DataFrame(rows)


def _partC_rev1_recommend_resolution(runs: pd.DataFrame, pairwise: pd.DataFrame) -> float | None:
    if runs.empty:
        return None
    seed_similarity = pairwise.loc[pairwise.get("comparison_type", pd.Series(dtype=str)).eq("same_resolution_different_seed")]
    complexity = runs.groupby("resolution", as_index=False)["proxy_physical_label_graph_size"].median()
    if seed_similarity.empty:
        preferred = runs.loc[runs["resolution"].sub(1.0).abs().idxmin(), "resolution"]
        return float(preferred)
    scores = (
        seed_similarity.groupby("left_resolution", as_index=False)[["normalized_mutual_info_score", "adjusted_rand_score"]]
        .mean()
        .rename(columns={"left_resolution": "resolution"})
    )
    scores = scores.merge(complexity, on="resolution", how="left")
    scores = scores.sort_values(
        ["normalized_mutual_info_score", "adjusted_rand_score", "proxy_physical_label_graph_size"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return float(scores.iloc[0]["resolution"])


def partC_rev1_compute_partition_stability(
    partition_state: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    write_outputs: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the full Prompt 3B stability grid over Leiden resolution and seed only."""

    stability = config.get("backbone_partition", {}).get("stability", {})
    resolutions = [float(value) for value in stability.get("resolution_values", [0.5, 0.75, 1.0, 1.25, 1.5])]
    seeds = [int(value) for value in stability.get("random_seeds", [42, 43, 44])]
    partition_nodes = partition_state["partition_nodes"]
    nonbackbone = partition_state["nonbackbone_physical_arcs"]
    graph_hash = partC_rev1_nonbackbone_graph_hash(nonbackbone)
    run_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    proxy_frames: list[pd.DataFrame] = []
    run_partitions: dict[tuple[float, int], pd.DataFrame] = {}

    for resolution in resolutions:
        for seed in seeds:
            run_id = f"partC_rev1_stability_res_{str(resolution).replace('.', 'p')}_seed_{seed}"
            total_start = time.perf_counter()
            try:
                partition = partC_rev1_run_leiden_partition(partition_nodes, nonbackbone, config, resolution=resolution, random_seed=seed)
                assign_start = time.perf_counter()
                arc_regions = partC_rev1_assign_physical_arc_regions(partition_state["arcs"], partition_nodes, partition["node_partition"])
                assignment_seconds = time.perf_counter() - assign_start
                metrics_start = time.perf_counter()
                proxy = _partC_rev1_complexity_proxy(partition["regions"], arc_regions)
                proxy = proxy.assign(resolution=resolution, seed=seed, random_seed=seed, run_id=run_id, graph_hash=graph_hash)
                fixed = _partC_rev1_fixed_metrics(partition_state, arc_regions)
                region_metrics = _partC_rev1_region_metrics(partition["regions"], arc_regions)
                complexity = _partC_rev1_complexity_metrics(proxy, int(fixed["final_backbone_directed_arcs"]))
                metric_seconds = time.perf_counter() - metrics_start
                total_seconds = time.perf_counter() - total_start
                run_rows.append(
                    {
                        "resolution": resolution,
                        "seed": seed,
                        "random_seed": seed,
                        "run_id": run_id,
                        "graph_hash": graph_hash,
                        **fixed,
                        **region_metrics,
                        **complexity,
                        "leiden_time_seconds": float(partition["leiden_seconds"]),
                        "partition_assignment_time_seconds": assignment_seconds,
                        "metric_computation_time_seconds": metric_seconds,
                        "total_time_seconds": total_seconds,
                    }
                )
                proxy_frames.append(proxy)
                run_partitions[(resolution, seed)] = partition["node_partition"].copy()
            except Exception as exc:
                incomplete_rows.append(
                    {
                        "resolution": resolution,
                        "seed": seed,
                        "run_id": run_id,
                        "graph_hash": graph_hash,
                        "reason": str(exc),
                    }
                )

    runs = pd.DataFrame(run_rows)
    complexity_proxy = pd.concat(proxy_frames, ignore_index=True) if proxy_frames else pd.DataFrame()
    pairwise = _partC_rev1_similarity(run_partitions, resolutions, seeds)
    recommended = _partC_rev1_recommend_resolution(runs, pairwise)
    summary = pd.DataFrame(
        [
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "variation_dimensions": "resolution,seed",
                "vary_backbone_labels": False,
                "expected_run_count": len(resolutions) * len(seeds),
                "completed_run_count": len(runs),
                "incomplete_run_count": len(incomplete_rows),
                "resolution_values": "|".join(str(value) for value in resolutions),
                "random_seeds": "|".join(str(value) for value in seeds),
                "graph_hash": graph_hash,
                "recommended_default_resolution": recommended,
            }
        ]
    )
    outputs = {
        "runs": runs,
        "complexity_proxy": complexity_proxy,
        "pairwise_similarity": pairwise,
        "summary": summary,
        "incomplete_runs": pd.DataFrame(incomplete_rows),
    }
    if write_outputs:
        partC_rev1_write_stability_outputs(outputs, config)
    return outputs


def _partC_rev1_metric_triplet(df: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"min={values.min():.4f}, median={values.median():.4f}, mean={values.mean():.4f}"


def _partC_rev1_stability_summary_md(outputs: dict[str, pd.DataFrame]) -> str:
    runs = outputs["runs"]
    pairwise = outputs["pairwise_similarity"]
    summary = outputs["summary"].iloc[0]
    incomplete = outputs.get("incomplete_runs", pd.DataFrame())
    seed_pairwise = pairwise.loc[pairwise.get("comparison_type", pd.Series(dtype=str)).eq("same_resolution_different_seed")]
    adjacent = pairwise.loc[pairwise.get("comparison_type", pd.Series(dtype=str)).eq("same_seed_adjacent_resolution")]
    lines = [
        "# partC_rev1 Partition Stability Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Experiment",
        "",
        "- variation_dimensions: resolution, seed",
        "- vary_backbone_labels: false",
        "- model: physical graph labels, no supernodes, no abstract graph",
        f"- completed_runs: {summary['completed_run_count']} / {summary['expected_run_count']}",
        f"- graph_hash: {summary['graph_hash']}",
        f"- recommended_default_resolution: {summary['recommended_default_resolution']}",
        "",
        "## Same-Resolution Seed Similarity",
        "",
        f"- NMI: {_partC_rev1_metric_triplet(seed_pairwise, 'normalized_mutual_info_score')}",
        f"- ARI: {_partC_rev1_metric_triplet(seed_pairwise, 'adjusted_rand_score')}",
        "",
        "## Adjacent-Resolution Similarity",
        "",
        f"- NMI: {_partC_rev1_metric_triplet(adjacent, 'normalized_mutual_info_score')}",
        f"- ARI: {_partC_rev1_metric_triplet(adjacent, 'adjusted_rand_score')}",
        "",
        "## Sensitivity",
        "",
    ]
    if not runs.empty:
        region_by_resolution = runs.groupby("resolution")["num_regions"].agg(["min", "median", "mean", "max"]).reset_index()
        transfer_by_resolution = runs.groupby("resolution")["sum_m_r_est"].agg(["min", "median", "mean", "max"]).reset_index()
        lines.append("### Region Count By Resolution")
        lines.append(region_by_resolution.to_string(index=False))
        lines.append("")
        lines.append("### Transfer-Pair Proxy By Resolution")
        lines.append(transfer_by_resolution.to_string(index=False))
        lines.append("")
    lines.extend(
        [
            "## Manuscript Readiness",
            "",
            "The experiment isolates Leiden resolution and random seed while reusing one fixed physical backbone and physical graph labeling. Downstream surrogate accuracy and assignment runtime remain later-prompt validation tasks.",
            "",
            "## Incomplete Runs",
            "",
        ]
    )
    if incomplete.empty:
        lines.append("None.")
    else:
        lines.append(incomplete.to_string(index=False))
    return "\n".join(lines) + "\n"


def partC_rev1_write_stability_outputs(outputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> None:
    partC_rev1_write_table(outputs["runs"], partC_rev1_output_file(config, "partC_rev1_partition_stability_runs.csv"), index=False)
    partC_rev1_write_table(outputs["pairwise_similarity"], partC_rev1_output_file(config, "partC_rev1_partition_stability_pairwise_similarity.csv"), index=False)
    partC_rev1_write_table(outputs["complexity_proxy"], partC_rev1_output_file(config, "partC_rev1_partition_stability_complexity_proxy.csv"), index=False)
    incomplete = outputs.get("incomplete_runs", pd.DataFrame())
    if not incomplete.empty:
        partC_rev1_write_table(incomplete, partC_rev1_output_file(config, "partC_rev1_partition_stability_partial_run_report.csv"), index=False)
    partC_rev1_output_file(config, "partC_rev1_partition_stability_summary.md").write_text(_partC_rev1_stability_summary_md(outputs), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        runs = outputs["runs"]
        if not runs.empty:
            ax = runs.groupby("resolution")["num_regions"].mean().plot(marker="o")
            ax.set_ylabel("Mean region count")
            ax.figure.tight_layout()
            ax.figure.savefig(partC_rev1_output_file(config, "partC_rev1_partition_stability_region_count.png"))
            plt.close(ax.figure)

            ax = runs.groupby("resolution")["sum_m_r_est"].mean().plot(marker="o")
            ax.set_ylabel("Mean estimated transfer pairs")
            ax.figure.tight_layout()
            ax.figure.savefig(partC_rev1_output_file(config, "partC_rev1_partition_stability_transfer_pairs.png"))
            plt.close(ax.figure)

            pairwise = outputs["pairwise_similarity"]
            if not pairwise.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                matrix = pairwise.pivot_table(index="left_resolution", columns="right_resolution", values="normalized_mutual_info_score", aggfunc="mean")
                image = ax.imshow(matrix.fillna(0.0).to_numpy(), vmin=0.0, vmax=1.0)
                ax.set_xticks(range(len(matrix.columns)), [str(value) for value in matrix.columns])
                ax.set_yticks(range(len(matrix.index)), [str(value) for value in matrix.index])
                ax.set_xlabel("Right resolution")
                ax.set_ylabel("Left resolution")
                fig.colorbar(image, ax=ax, label="NMI")
                fig.tight_layout()
                fig.savefig(partC_rev1_output_file(config, "partC_rev1_partition_stability_similarity_heatmap.png"))
                plt.close(fig)
    except Exception:
        return
