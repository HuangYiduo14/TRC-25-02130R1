"""Immutable native labels and append-only shared path catalogs; no training."""
from pathlib import Path
import time
import numpy as np

from .common import read_json, read_npz, write_json, write_npz, sha, input_hash, arrays_hash, digest
from .oracle import Oracle, Result, normalized


class Store:
    def __init__(self, directory, desc, arrays, *, namespace="labels"):
        self.root = Path(directory) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.desc = desc
        self.code_binding = {name:sha(Path(__file__).with_name(name)) for name in ["oracle.py","kernels.py","storage.py"]}
        self.oracle = Oracle(arrays)
        self.index_file = self.root / "catalog_index.json"
        self.segments = []
        self.saved_paths = 0
        if self.index_file.exists():
            index = read_json(self.index_file)
            if index["model_hash"] != desc["model_hash"]: raise ValueError("Catalog wrong model")
            self.segments = index["segments"]
            for seg in self.segments:
                path = self.root / seg["relative_path"]
                if sha(path) != seg["sha256"]: raise ValueError("Catalog bytes differ")
                self.oracle.import_catalog(read_npz(path))
            self.saved_paths = int(index["path_count"])
            if self.saved_paths != len(self.oracle.paths): raise ValueError("Catalog count differs")

    def exists(self, ih):
        return (self.root / "receipts" / (ih + ".json")).exists()

    def hashes(self):
        return {p.stem for p in (self.root / "receipts").glob("*.json")}

    def get(self, ih, *, recertify=False):
        receipt = read_json(self.root / "receipts" / (ih + ".json"))
        if receipt["model_hash"] != self.desc["model_hash"] or receipt["support_hash"] != self.desc["support_hash"]:
            raise ValueError("Label wrong model or support")
        p = self.root / receipt["payload_path"]
        if sha(p) != receipt["payload_sha256"]: raise ValueError("Label bytes differ")
        a = read_npz(p)
        if input_hash(self.desc["model_hash"], self.desc["support_hash"], a["demand"]) != ih:
            raise ValueError("Native input identity differs")
        result = Result(**{**receipt["scalars"], **{k: a[k] for k in ["demand", "z", "tau", "path_ids", "weights"]}})
        if recertify:
            h = np.zeros(len(self.oracle.paths)); h[result.path_ids] = result.weights
            again = self.oracle.certify(result.demand, h, result.iterations, time.perf_counter(),
                                        "FRESH_REOPEN", result.exchanges, self.oracle.lmo_count)
            for field in ["z", "tau"]:
                if not np.allclose(getattr(result, field), getattr(again, field), rtol=2e-12, atol=1e-10):
                    raise ValueError("Fresh label vector recertification differs: " + field)
            for field in ["value_upper", "value_lower", "gap", "supporting_intercept"]:
                if not np.isclose(getattr(result, field), getattr(again, field), rtol=2e-10, atol=1e-7):
                    raise ValueError("Fresh scalar recertification differs: " + field)
            if result.quality != again.quality: raise ValueError("Fresh quality differs")
        return result, receipt

    def publish(self, result, metadata, scales=None):
        if result.quality not in {"A", "B", "C"}: raise ValueError("Uncertified result cannot be published")
        ih = input_hash(self.desc["model_hash"], self.desc["support_hash"], result.demand)
        if self.exists(ih):
            existing, receipt = self.get(ih)
            return receipt
        if len(self.oracle.paths) > self.saved_paths:
            segment = self.oracle.catalog(self.saved_paths)
            name = f"catalog/{self.saved_paths:09d}_{len(self.oracle.paths):09d}.npz"
            p = self.root / name
            if p.exists():
                prior = read_npz(p)
                if arrays_hash(prior) != arrays_hash(segment): raise ValueError("Orphan catalog differs")
            else: write_npz(p, **segment)
            self.segments.append(dict(relative_path=name, sha256=sha(p), size_bytes=p.stat().st_size))
            self.saved_paths = len(self.oracle.paths)
            write_json(self.index_file, dict(model_hash=self.desc["model_hash"],
                path_count=self.saved_paths, segments=self.segments), new=False)
        data = {k: getattr(result, k) for k in ["demand", "z", "tau", "path_ids", "weights"]}
        pair, source, target = self.oracle.pool_arrays()
        selected = result.path_ids
        fine_keys = np.column_stack([pair[selected], source[selected], target[selected]])
        unique, inverse = np.unique(fine_keys, axis=0, return_inverse=True)
        data.update(fine_pair=unique[:, 0], fine_source=unique[:, 1], fine_target=unique[:, 2],
                    fine_amount=np.bincount(inverse, weights=result.weights, minlength=len(unique)))
        normalized_scalars = None
        if scales is not None:
            norm = normalized(result, scales["input_scale"], scales["c0"], float(scales["Vscale"]))
            data["normalized_input"] = norm.pop("input"); data["normalized_gradient"] = norm.pop("gradient")
            normalized_scalars = norm
        p = self.root / "payloads" / (ih + ".npz")
        if p.exists():
            if arrays_hash(read_npz(p)) != arrays_hash(data): raise ValueError("Orphan payload differs")
        else: write_npz(p, **data)
        receipt = dict(schema="partC.phase6.active-label.v1", model_class=self.desc["model_class"],
            model_hash=self.desc["model_hash"], support_hash=self.desc["support_hash"],
            region_id=self.desc["region_id"], input_hash=ih, m_active=len(result.demand),
            payload_path=str(p.relative_to(self.root)).replace("\\", "/"), payload_sha256=sha(p),
            payload_bytes=p.stat().st_size, scalars=result.scalars(), normalized_scalars=normalized_scalars,
            metadata=metadata, migration="NEW_ACTIVE_MODEL_LABEL", finalized_full_LMO=True,
            implementation_sources=self.code_binding,
            source_binding={k:self.desc.get(k) for k in ["graph_hash","fiber_hash","source_interface_hash","source_snapshot_sha256"]},
            native_float64=True, published_unix=time.time(), catalog_path_count=self.saved_paths)
        write_json(self.root / "receipts" / (ih + ".json"), receipt)
        return receipt
