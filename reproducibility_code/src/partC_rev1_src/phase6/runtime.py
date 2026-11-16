"""Untrained active-only data interface and exact regional edge enumeration."""
from pathlib import Path
import numpy as np
import pandas as pd

from .common import read_json, read_npz, sha, input_hash, arrays_hash, digest, write_json, write_frame
from .registry import region_dir
from .resume_policy import require_readiness


class ActiveTopology:
    def __init__(self, root):
        self.root=Path(root)
        self.contract=read_json(self.root/"registry/runtime_adjacency.json")
        p=self.root/"registry"/self.contract["active_pairs_path"]
        if sha(p)!=self.contract["active_pairs_sha256"]: raise ValueError("Active edges changed")
        self.edges=pd.read_parquet(p)
        if not self.edges.runtime_connected.all() or not self.edges.LOW_CURRENT.gt(0).all(): raise ValueError("Inactive regional edge")
        if self.edges.pair_id.duplicated().any(): raise ValueError("Duplicate regional edge")
        self.lookup={(r.region_id,r.source_terminal_id,r.target_terminal_id):r.pair_id for r in self.edges.itertuples()}
        if len(self.lookup)!=len(self.edges): raise ValueError("Duplicate regional terminal pair")

    def regional_edge(self, region_id, source_terminal_id, target_terminal_id):
        return self.lookup.get((region_id,source_terminal_id,target_terminal_id))

    def enumerate_region(self, region_id):
        return self.edges.loc[self.edges.region_id.eq(region_id)].copy()


class RegionalDataset:
    """Streaming native/normalized label rows; never constructs any optimizer or ICNN."""
    def __init__(self, root, region_id, *, grades=("A","B"), split="TRAIN", allow_incomplete=False):
        self.root=Path(root)
        self.directory=self.root/"regions"/region_id.removeprefix("partC_p2_region_")
        self.desc=read_json(self.directory/"descriptor.json")
        if set(grades) not in [{"A","B"},{"A","B","C"}]: raise ValueError("Only explicit AB or ABC views")
        self.grades=set(grades); self.split=split
        self.dimension=int(self.desc["m_active"])
        self.model_required=self.dimension>0
        if self.dimension and not allow_incomplete:
            if (self.root/"control/resume_identity.json").exists():
                require_readiness(self.root,region_id)
                return
            blocker=self.root/"validation/BLOCKING_DIAGNOSIS.json"
            gate=self.root/"validation/BASELINE_COMPLETE.json"
            if blocker.exists() and read_json(blocker)["status"]=="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT":
                raise ValueError("Dataset blocked by mandatory source/path-contract review")
            if not gate.exists() or not read_json(gate).get("baseline_complete",False):
                raise ValueError("Mandatory baseline is incomplete; internal audit access must be explicit")

    def iter_labels(self, *, normalized=False):
        if not self.dimension: return
        scales=read_npz(self.directory/"scales.npz") if normalized else None
        for p in sorted((self.directory/"labels/receipts").glob("*.json")):
            receipt=read_json(p)
            if receipt["scalars"]["quality"] not in self.grades: continue
            if self.split is not None and receipt["metadata"]["split"]!=self.split: continue
            if receipt["model_hash"]!=self.desc["model_hash"] or receipt["support_hash"]!=self.desc["support_hash"]:
                raise ValueError("Dataset model or active mask changed")
            payload=self.directory/"labels"/receipt["payload_path"]
            if sha(payload)!=receipt["payload_sha256"]: raise ValueError("Dataset payload changed")
            a=read_npz(payload)
            if a["demand"].shape!=(self.dimension,) or a["tau"].shape!=(self.dimension,): raise ValueError("Padding or omitted active gradient")
            if input_hash(self.desc["model_hash"],self.desc["support_hash"],a["demand"])!=receipt["input_hash"]: raise ValueError("Input identity differs")
            if normalized:
                x=a["normalized_input"]; gradient=a["normalized_gradient"]; scalars=receipt["normalized_scalars"]
            else:
                x=a["demand"]; gradient=a["tau"]; scalars=receipt["scalars"]
            yield {"input":x,"gradient":gradient,"value_upper":scalars["value_upper"],
                   "value_lower":scalars["value_lower"],"input_hash":receipt["input_hash"],
                   "quality":receipt["scalars"]["quality"],"metadata":receipt["metadata"]}


def validate_topology(c, *, output_name="P5_RUNTIME_INTERFACE_PASS.json"):
    root=Path(c["output_root"]); topology=ActiveTopology(root)
    regions=pd.read_parquet(root/"registry/regions.parquet")
    excluded=pd.read_parquet(root/"registry/excluded_zero_low_pairs.parquet")
    owned=pd.read_parquet(root/"registry/owned_arcs.parquet")
    if len(regions)!=1847 or regions.D_full.sum()!=888787 or len(owned)!=327757 or owned.physical_arc_index.duplicated().any():
        raise ValueError("All-owner topology closure differs")
    if len(topology.edges)+len(excluded)!=888787 or set(topology.edges.pair_id)&set(excluded.pair_id): raise ValueError("Active/excluded closure differs")
    if excluded.runtime_connected.any() or not excluded.LOW_CURRENT.eq(0).all(): raise ValueError("Excluded positive or connected pair")
    for b in topology.contract["static_bindings"].values():
        p=Path(b["absolute_path"])
        if p.stat().st_size!=b["size_bytes"] or sha(p)!=b["sha256"]: raise ValueError("Frozen static source changed")
    rows=[]; modeled=set()
    for d in regions.to_dict("records"):
        rd=region_dir(c,d["region_id"])
        a=read_npz(rd/"local_graph.npz"); mp=read_npz(rd/"support_map.npz")
        full=pd.read_parquet(rd/"full_pair_audit.parquet"); pairs=pd.read_parquet(rd/"active_pairs.parquet")
        edges=topology.enumerate_region(d["region_id"])
        if len(pairs)!=d["m_active"] or not np.array_equal(pairs.pair_id,edges.pair_id): raise ValueError("Runtime selected edges differ")
        if not np.array_equal(np.flatnonzero(full.LOW_CURRENT.to_numpy()>0),mp["active_indices"]): raise ValueError("Strict native LOW support selection differs")
        if not np.array_equal(mp["full_keys"][mp["active_indices"]],pairs.coarse_key): raise ValueError("Forward active mapping differs")
        if not np.array_equal(mp["full_to_active"][mp["active_indices"]],np.arange(d["m_active"])): raise ValueError("Reverse active mapping differs")
        if len(a["global_arc_index"])!=d["physical_arcs"] or len(a["global_node_index"])!=d["physical_nodes"]: raise ValueError("Owned physical arc pruning")
        if len(a["choice_offsets"])-1!=d["m_active"]: raise ValueError("Full model padding in runtime")
        if arrays_hash({"full_keys":mp["full_keys"],"active_mask":full.LOW_CURRENT.to_numpy()>0,"LOW_CURRENT":full.LOW_CURRENT.to_numpy()})!=d["support_hash"]:
            raise ValueError("Frozen model support hash differs")
        ds=RegionalDataset(root,d["region_id"],allow_incomplete=True)
        if ds.dimension!=d["m_active"]: raise ValueError("Dataset dimension differs")
        if not d["m_active"] and list(ds.iter_labels()): raise ValueError("Inactive training tensors")
        modeled.update(pairs.pair_id)
        rows.append(dict(region_id=d["region_id"],m_active=d["m_active"],D_full=d["D_full"],
            physical_arcs=d["physical_arcs"],model_required=ds.model_required,status="PASS"))
    if len(modeled)!=len(topology.edges): raise ValueError("Global edge bijection differs")
    record=dict(status="PASS",regions=len(regions),regional_edges=len(modeled),
        excluded_no_edge_pairs=len(excluded),static_links=topology.contract["static_links"],
        total_logical_links=len(modeled)+topology.contract["static_links"],online_nodes=72724,
        inactive_regions=int(regions.m_active.eq(0).sum()),all_models_reopened=True,
        no_per_sample_edge_reselection=True,no_fallback=True,ICNN_constructions=0,training_performed=False)
    write_frame(root/"validation"/output_name.replace(".json","_regions.parquet"),pd.DataFrame(rows))
    write_json(root/"validation"/output_name,record)
    return record
