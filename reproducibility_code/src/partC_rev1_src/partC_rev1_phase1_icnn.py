"""Exact anchored float64 ICNN for bounded synthetic Phase 1 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .partC_rev1_phase1_campaign_contract import (
    partC_rev1_load_campaign_config,
    partC_rev1_resolve_campaign_output,
    partC_rev1_verify_approval_bindings,
)


_RUN_MATRIX_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RUN_MATRIX.json"
)
_CHECKPOINT_SCHEMA_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/schemas/phase1_icnn_checkpoint.schema.json"
)


class PartCRev1ICNNError(ValueError):
    """Raised for a mathematical, identity, or synthetic-scope violation."""


class PartCRev1ICNNCheckpointError(PartCRev1ICNNError):
    """Raised before loading or writing a mismatched checkpoint."""


def _float64_vector(value: Sequence[float] | np.ndarray | Tensor, label: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise PartCRev1ICNNError(f"{label} must be a nonempty finite float64 vector")
    return tensor.detach().clone()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_order_hash(pair_ids: Sequence[Any]) -> str:
    values = [str(value) for value in pair_ids]
    if not values or any(not value.strip() for value in values) or len(values) != len(set(values)):
        raise PartCRev1ICNNError("Ordered pair IDs must be nonempty, unique, and nonblank")
    return _canonical_hash(values)


@dataclass(frozen=True, slots=True)
class PartCRev1PositiveScale:
    values: Tensor

    def __init__(self, values: Sequence[float] | np.ndarray | Tensor) -> None:
        tensor = _float64_vector(values, "Positive scale")
        if not bool((tensor > 0).all()):
            raise PartCRev1ICNNError("Scale values must be strictly positive")
        object.__setattr__(self, "values", tensor)

    @property
    def dimension(self) -> int:
        return int(self.values.numel())

    def _aligned(self, value: Tensor, label: str) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim < 1 or tensor.shape[-1] != self.dimension or not bool(torch.isfinite(tensor).all()):
            raise PartCRev1ICNNError(f"{label} must be finite with last dimension {self.dimension}")
        return tensor

    def normalize_flow(self, physical_flow: Tensor) -> Tensor:
        return self._aligned(physical_flow, "Physical flow") / self.values

    def denormalize_flow(self, normalized_flow: Tensor) -> Tensor:
        return self._aligned(normalized_flow, "Normalized flow") * self.values

    def normalize_cost_target(self, physical_cost: Tensor) -> Tensor:
        return self._aligned(physical_cost, "Physical cost") * self.values

    def physical_cost(self, normalized_gradient: Tensor) -> Tensor:
        return self._aligned(normalized_gradient, "Normalized gradient") / self.values

    def to_list(self) -> list[float]:
        return [float(value) for value in self.values.tolist()]


class PartCRev1AnchoredICNN(nn.Module):
    """One-hidden-layer convex potential with computed exact anchors."""

    def __init__(
        self,
        input_dimension: int,
        tau0: Sequence[float] | np.ndarray | Tensor,
        *,
        hidden_width: int = 16,
        beta: float = 2.0,
        mu: float = 0.001,
        w_row_l2_max: float = 4.0,
        v_l2_max: float = 4.0,
        bias_abs_max: float = 8.0,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        dimension = int(input_dimension)
        width = int(hidden_width)
        if dimension <= 0 or width <= 0:
            raise PartCRev1ICNNError("ICNN dimensions must be positive")
        anchor = _float64_vector(tau0, "Free-flow anchor")
        if anchor.numel() != dimension:
            raise PartCRev1ICNNError("Free-flow anchor dimension mismatch")
        settings = (float(beta), float(mu), float(w_row_l2_max), float(v_l2_max), float(bias_abs_max))
        if any(not math.isfinite(value) for value in settings) or beta <= 0 or mu < 0 or min(settings[2:]) <= 0:
            raise PartCRev1ICNNError("ICNN beta/bounds must be positive and mu must be nonnegative")
        self.input_dimension = dimension
        self.hidden_width = width
        self.beta = float(beta)
        self.mu = float(mu)
        self.w_row_l2_max = float(w_row_l2_max)
        self.v_l2_max = float(v_l2_max)
        self.bias_abs_max = float(bias_abs_max)
        self.register_buffer("tau0", anchor)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.W = nn.Parameter(torch.randn(width, dimension, generator=generator, dtype=torch.float64) * 0.2)
        self.b = nn.Parameter(torch.randn(width, generator=generator, dtype=torch.float64) * 0.1)
        self.v = nn.Parameter(torch.rand(width, generator=generator, dtype=torch.float64) * 0.05)
        self.to(dtype=torch.float64)
        partC_rev1_project_icnn_parameters(self)

    def computed_anchors(self) -> tuple[Tensor, Tensor]:
        effective_v = self.v
        softplus_b = F.softplus(self.b, beta=self.beta)
        derivative_b = torch.sigmoid(self.beta * self.b)
        c = -torch.dot(effective_v, softplus_b)
        a = self.tau0 - self.W.transpose(0, 1).matmul(effective_v * derivative_b)
        return c, a

    def forward(self, flow: Tensor) -> Tensor:
        return partC_rev1_potential(self, flow)

    def analytic_cost(self, flow: Tensor) -> Tensor:
        x = torch.as_tensor(flow, dtype=torch.float64)
        if x.ndim < 1 or x.shape[-1] != self.input_dimension:
            raise PartCRev1ICNNError("Flow last dimension does not match the ICNN")
        _, a = self.computed_anchors()
        z = x.matmul(self.W.transpose(0, 1)) + self.b
        return a + torch.sigmoid(self.beta * z).matmul(self.W * self.v[:, None]) + self.mu * x

    def analytic_hessian(self, flow: Tensor) -> Tensor:
        x = torch.as_tensor(flow, dtype=torch.float64)
        if x.ndim != 1 or x.shape[0] != self.input_dimension:
            raise PartCRev1ICNNError("Analytic Hessian requires one flow vector")
        z = self.W.matmul(x) + self.b
        sigma = torch.sigmoid(self.beta * z)
        weights = self.v * self.beta * sigma * (1.0 - sigma)
        return self.W.transpose(0, 1).matmul(weights[:, None] * self.W) + self.mu * torch.eye(
            self.input_dimension, dtype=torch.float64, device=x.device
        )

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "dtype": "float64",
            "input_dimension": self.input_dimension,
            "output_dimension": self.input_dimension,
            "hidden_layers": 1,
            "hidden_width": self.hidden_width,
            "activation": "softplus",
            "beta": self.beta,
            "mu": self.mu,
            "w_constraint": "unrestricted_signed_projected_row_l2",
            "v_constraint": "nonnegative_projected_l2",
            "anchor_policy": "computed_c_and_a",
            "batch_normalization": False,
        }

    def norm_controls(self) -> dict[str, float]:
        return {
            "w_row_l2_max": self.w_row_l2_max,
            "v_l2_max": self.v_l2_max,
            "bias_abs_max": self.bias_abs_max,
        }


def _flow(model: PartCRev1AnchoredICNN, flow: Tensor, *, requires_grad: bool = False) -> Tensor:
    x = torch.as_tensor(flow, dtype=torch.float64)
    if x.ndim < 1 or x.shape[-1] != model.input_dimension or not bool(torch.isfinite(x).all()):
        raise PartCRev1ICNNError(f"Flow must be finite with last dimension {model.input_dimension}")
    if requires_grad and not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)
    return x


def partC_rev1_potential(model: PartCRev1AnchoredICNN, flow: Tensor) -> Tensor:
    x = _flow(model, flow)
    c, a = model.computed_anchors()
    hidden = F.softplus(x.matmul(model.W.transpose(0, 1)) + model.b, beta=model.beta)
    return c + x.matmul(a) + hidden.matmul(model.v) + 0.5 * model.mu * torch.sum(x * x, dim=-1)


def partC_rev1_cost(
    model: PartCRev1AnchoredICNN, flow: Tensor, create_graph: bool = False
) -> Tensor:
    x = _flow(model, flow, requires_grad=True)
    potential = partC_rev1_potential(model, x)
    return torch.autograd.grad(potential.sum(), x, create_graph=bool(create_graph))[0]


def _constraint_metrics(model: PartCRev1AnchoredICNN) -> dict[str, float]:
    return {
        "max_w_row_l2": float(torch.linalg.vector_norm(model.W, dim=1).max()),
        "v_l2": float(torch.linalg.vector_norm(model.v)),
        "min_v": float(model.v.min()),
        "max_abs_b": float(model.b.abs().max()),
    }


def partC_rev1_project_icnn_parameters(model: PartCRev1AnchoredICNN) -> dict[str, float]:
    with torch.no_grad():
        row_norms = torch.linalg.vector_norm(model.W, dim=1, keepdim=True)
        model.W.mul_(torch.clamp(model.w_row_l2_max / torch.clamp(row_norms, min=1e-30), max=1.0))
        model.v.clamp_(min=0.0)
        v_norm = torch.linalg.vector_norm(model.v)
        if float(v_norm) > model.v_l2_max:
            model.v.mul_(model.v_l2_max / v_norm)
        model.b.clamp_(min=-model.bias_abs_max, max=model.bias_abs_max)
    maxima = _constraint_metrics(model)
    if (
        maxima["max_w_row_l2"] > model.w_row_l2_max + 1e-12
        or maxima["v_l2"] > model.v_l2_max + 1e-12
        or maxima["min_v"] < -1e-15
        or maxima["max_abs_b"] > model.bias_abs_max + 1e-12
    ):
        raise PartCRev1ICNNError("ICNN parameter projection failed")
    return maxima


def partC_rev1_validate_pair_order(
    pair_ids: Sequence[Any], declared_hash: str, *, expected_dimension: int | None = None
) -> str:
    observed = _pair_order_hash(pair_ids)
    declared = str(declared_hash).lower()
    if len(declared) != 64 or any(value not in "0123456789abcdef" for value in declared):
        raise PartCRev1ICNNCheckpointError("Declared pair-order hash must be a lowercase SHA-256")
    if expected_dimension is not None and len(pair_ids) != int(expected_dimension):
        raise PartCRev1ICNNCheckpointError("Ordered pair count differs from checkpoint input dimension")
    if observed != declared:
        raise PartCRev1ICNNCheckpointError(
            f"Ordered pair identity mismatch: expected {declared}, observed {observed}"
        )
    return observed


def _metadata(
    model: PartCRev1AnchoredICNN,
    scale: PartCRev1PositiveScale,
    pair_ids: Sequence[Any],
    *,
    physical_cost_units: str,
    config_sha256: str,
    code_commit: str,
    upstream_identities: Mapping[str, str],
) -> dict[str, Any]:
    pairs = [str(value) for value in pair_ids]
    if len(pairs) != model.input_dimension or scale.dimension != model.input_dimension:
        raise PartCRev1ICNNCheckpointError("Pair/scale/input dimensions must match")
    pair_hash = _pair_order_hash(pairs)
    architecture = model.architecture_dict()
    metadata = {
        "schema_id": "partC_rev1_phase1_icnn_checkpoint",
        "schema_version": "1.0",
        "dtype": "float64",
        "input_dimension": model.input_dimension,
        "output_dimension": model.input_dimension,
        "hidden_width": model.hidden_width,
        "beta": model.beta,
        "mu": model.mu,
        "input_scales": scale.to_list(),
        "physical_cost_units": str(physical_cost_units),
        "free_flow_anchor": [float(value) for value in model.tau0.tolist()],
        "free_flow_anchor_hash": _canonical_hash([float(value) for value in model.tau0.tolist()]),
        "ordered_pair_ids": pairs,
        "ordered_pair_ids_hash": pair_hash,
        "norm_controls": model.norm_controls(),
        "architecture": architecture,
        "architecture_hash": _canonical_hash(architecture),
        "config_sha256": str(config_sha256),
        "code_commit": str(code_commit),
        "training_code_sha256": _sha256(Path(__file__).resolve()),
        "upstream_identities": dict(upstream_identities),
        "synthetic_only": True,
    }
    return metadata


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / _CHECKPOINT_SCHEMA_RELATIVE).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    errors = sorted(validator.iter_errors(dict(metadata)), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(value) for value in first.absolute_path) or "$"
        raise PartCRev1ICNNCheckpointError(f"ICNN checkpoint schema failed at {location}: {first.message}")
    if metadata["architecture_hash"] != _canonical_hash(metadata["architecture"]):
        raise PartCRev1ICNNCheckpointError("ICNN checkpoint architecture hash mismatch")
    if metadata["free_flow_anchor_hash"] != _canonical_hash(metadata["free_flow_anchor"]):
        raise PartCRev1ICNNCheckpointError("ICNN checkpoint anchor hash mismatch")
    partC_rev1_validate_pair_order(
        metadata["ordered_pair_ids"],
        metadata["ordered_pair_ids_hash"],
        expected_dimension=int(metadata["input_dimension"]),
    )


def partC_rev1_save_icnn_checkpoint(
    path: str | Path,
    model: PartCRev1AnchoredICNN,
    scale: PartCRev1PositiveScale,
    pair_ids: Sequence[Any],
    *,
    physical_cost_units: str,
    config_sha256: str,
    code_commit: str,
    upstream_identities: Mapping[str, str],
) -> dict[str, Any]:
    destination = Path(path).resolve(strict=False)
    if not destination.parent.is_dir() or destination.exists():
        raise PartCRev1ICNNCheckpointError("Checkpoint parent must exist and checkpoint must be new")
    partC_rev1_project_icnn_parameters(model)
    metadata = _metadata(
        model,
        scale,
        pair_ids,
        physical_cost_units=physical_cost_units,
        config_sha256=config_sha256,
        code_commit=code_commit,
        upstream_identities=upstream_identities,
    )
    _validate_metadata(metadata)
    payload = {"metadata": metadata, "state_dict": model.state_dict()}
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {**metadata, "checkpoint_size_bytes": destination.stat().st_size, "checkpoint_sha256": _sha256(destination)}
    return metadata


def _compare_expected(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise PartCRev1ICNNCheckpointError(f"Expected mapping at {path}")
        for key, value in expected.items():
            if key not in actual:
                raise PartCRev1ICNNCheckpointError(f"Missing expected checkpoint value {path}.{key}")
            _compare_expected(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if list(actual) != list(expected):
            raise PartCRev1ICNNCheckpointError(f"Checkpoint sequence mismatch at {path}")
    elif actual != expected:
        raise PartCRev1ICNNCheckpointError(
            f"Checkpoint identity mismatch at {path}: expected {expected!r}, observed {actual!r}"
        )


def partC_rev1_load_icnn_checkpoint(
    path: str | Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[PartCRev1AnchoredICNN, PartCRev1PositiveScale, dict[str, Any]]:
    source = Path(path).resolve(strict=True)
    if not expected:
        raise PartCRev1ICNNCheckpointError("Checkpoint expectations must be explicit and nonempty")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PartCRev1ICNNCheckpointError(f"Cannot load ICNN checkpoint: {source}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
        raise PartCRev1ICNNCheckpointError("ICNN checkpoint payload structure is invalid")
    metadata = dict(payload["metadata"])
    _validate_metadata(metadata)
    _compare_expected(metadata, expected)
    architecture = metadata["architecture"]
    if int(metadata["input_dimension"]) != int(metadata["output_dimension"]):
        raise PartCRev1ICNNCheckpointError("ICNN checkpoint input/output dimensions differ")
    model = PartCRev1AnchoredICNN(
        int(metadata["input_dimension"]),
        metadata["free_flow_anchor"],
        hidden_width=int(metadata["hidden_width"]),
        beta=float(metadata["beta"]),
        mu=float(metadata["mu"]),
        w_row_l2_max=float(metadata["norm_controls"]["w_row_l2_max"]),
        v_l2_max=float(metadata["norm_controls"]["v_l2_max"]),
        bias_abs_max=float(metadata["norm_controls"]["bias_abs_max"]),
    )
    if model.architecture_dict() != architecture:
        raise PartCRev1ICNNCheckpointError("Reconstructed ICNN architecture differs from metadata")
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise PartCRev1ICNNCheckpointError("ICNN checkpoint tensor dimension/order mismatch") from exc
    maxima = _constraint_metrics(model)
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise PartCRev1ICNNCheckpointError(f"Nonfinite checkpoint parameter: {name}")
    if (
        maxima["max_w_row_l2"] > model.w_row_l2_max + 1e-12
        or maxima["v_l2"] > model.v_l2_max + 1e-12
        or maxima["min_v"] < -1e-15
        or maxima["max_abs_b"] > model.bias_abs_max + 1e-12
    ):
        raise PartCRev1ICNNCheckpointError("Checkpoint violates frozen ICNN norm constraints")
    scale = PartCRev1PositiveScale(metadata["input_scales"])
    return model, scale, metadata


def partC_rev1_run_synthetic_recovery(
    *,
    seed: int = 1729,
    sample_count: int = 128,
    max_steps: int = 600,
    learning_rate: float = 0.01,
    checkpoint_path: str | Path | None = None,
    config_sha256: str = "0" * 64,
    code_commit: str = "0" * 40,
) -> dict[str, Any]:
    """Run the frozen one-dimensional convex synthetic recovery."""

    if int(seed) != 1729 or int(sample_count) != 128 or not 1 <= int(max_steps) <= 600 or float(learning_rate) != 0.01:
        raise PartCRev1ICNNError("Synthetic recovery parameters differ from the frozen configuration")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    scale = PartCRev1PositiveScale([2.0])
    pair_ids = ["synthetic_pair_000"]
    tau0_physical = torch.tensor([1.25], dtype=torch.float64)
    model = PartCRev1AnchoredICNN(
        1,
        scale.normalize_cost_target(tau0_physical),
        hidden_width=16,
        beta=2.0,
        mu=0.001,
        w_row_l2_max=4.0,
        v_l2_max=4.0,
        bias_abs_max=8.0,
        seed=int(seed),
    )
    physical_flow = torch.linspace(0.0, 4.0, int(sample_count), dtype=torch.float64).reshape(-1, 1)
    normalized_flow = scale.normalize_flow(physical_flow)
    target_physical_cost = tau0_physical + 0.15 * physical_flow + 0.02 * physical_flow.square()
    target_normalized_cost = scale.normalize_cost_target(target_physical_cost)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, float | int]] = []
    initial_rmse = float("nan")
    maxima = partC_rev1_project_icnn_parameters(model)
    for step in range(1, int(max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = partC_rev1_cost(model, normalized_flow, create_graph=True)
        loss = torch.mean((predicted - target_normalized_cost).square())
        if not bool(torch.isfinite(loss)):
            raise PartCRev1ICNNError("Synthetic recovery produced nonfinite loss")
        physical_rmse = torch.sqrt(torch.mean((scale.physical_cost(predicted) - target_physical_cost).square()))
        if step == 1:
            initial_rmse = float(physical_rmse.detach())
        loss.backward()
        optimizer.step()
        maxima = partC_rev1_project_icnn_parameters(model)
        if step == 1 or step % 50 == 0 or step == int(max_steps):
            history.append(
                {
                    "step": step,
                    "normalized_mse": float(loss.detach()),
                    "physical_cost_rmse": float(physical_rmse.detach()),
                }
            )
    predicted = partC_rev1_cost(model, normalized_flow, create_graph=False).detach()
    final_rmse = float(torch.sqrt(torch.mean((scale.physical_cost(predicted) - target_physical_cost).square())))
    zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    phi_zero = float(partC_rev1_potential(model, zero).detach())
    cost_zero = partC_rev1_cost(model, zero, create_graph=False).detach()
    anchor_gradient_error = float(torch.max(torch.abs(cost_zero - model.tau0)))
    hessian_minimum = min(
        float(torch.linalg.eigvalsh(model.analytic_hessian(point)).min())
        for point in normalized_flow[::16]
    )
    anchor_potential_error = abs(phi_zero)
    if anchor_potential_error > 1e-12 or anchor_gradient_error > 1e-12:
        raise PartCRev1ICNNError("Synthetic ICNN anchor tolerance failed")
    if hessian_minimum < -1e-10 or final_rmse > 0.08:
        raise PartCRev1ICNNError("Synthetic ICNN convexity or recovery tolerance failed")
    checkpoint_metadata: dict[str, Any] | None = None
    reload_max_abs = 0.0
    if checkpoint_path is not None:
        checkpoint_metadata = partC_rev1_save_icnn_checkpoint(
            checkpoint_path,
            model,
            scale,
            pair_ids,
            physical_cost_units="synthetic_cost_units",
            config_sha256=config_sha256,
            code_commit=code_commit,
            upstream_identities={"scope": "synthetic_only", "real_label_hash": "NOT_LOADED"},
        )
        reloaded, reloaded_scale, _ = partC_rev1_load_icnn_checkpoint(
            checkpoint_path,
            expected={
                "synthetic_only": True,
                "ordered_pair_ids": pair_ids,
                "ordered_pair_ids_hash": _pair_order_hash(pair_ids),
                "input_dimension": 1,
                "config_sha256": config_sha256,
                "code_commit": code_commit,
            },
        )
        reloaded_cost = reloaded_scale.physical_cost(
            partC_rev1_cost(reloaded, normalized_flow, create_graph=False)
        ).detach()
        reload_max_abs = float(torch.max(torch.abs(reloaded_cost - scale.physical_cost(predicted))))
        if reload_max_abs > 1e-12:
            raise PartCRev1ICNNError("Synthetic checkpoint reload equality failed")
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "scope": "synthetic_only",
        "seed": int(seed),
        "sample_count": int(sample_count),
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "steps": int(max_steps),
        "input_dimension": 1,
        "hidden_width": 16,
        "dtype": "float64",
        "beta": 2.0,
        "mu": 0.001,
        "pair_ids": pair_ids,
        "pair_order_hash": _pair_order_hash(pair_ids),
        "input_scales": scale.to_list(),
        "initial_cost_rmse": initial_rmse,
        "final_cost_rmse": final_rmse,
        "cost_rmse_tolerance": 0.08,
        "anchor_potential_absolute_error": anchor_potential_error,
        "anchor_gradient_max_absolute_error": anchor_gradient_error,
        "sampled_minimum_hessian_eigenvalue": hessian_minimum,
        "norms": maxima,
        "history": history,
        "checkpoint": checkpoint_metadata,
        "checkpoint_reload_max_absolute_cost_error": reload_max_abs,
        "real_labels_loaded": False,
        "historical_checkpoints_loaded": False,
        "national_assignment_run": False,
        "phase_2_artifacts_created": 0,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PartCRev1ICNNError(f"Refusing to overwrite synthetic report: {path}")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_cli(config_path: Path, task_id: str, synthetic_only: bool) -> dict[str, Any]:
    if task_id != "S3-T02" or not synthetic_only:
        raise PartCRev1ICNNError("The approved CLI requires S3-T02 and --synthetic-only")
    config = partC_rev1_load_campaign_config(config_path)
    repository = Path(config["repository_root"]).resolve(strict=True)
    binding = partC_rev1_verify_approval_bindings(config_path, repository / _RUN_MATRIX_RELATIVE)
    icnn = config["icnn"]
    exact = {
        "scope": "synthetic_only",
        "dtype": "float64",
        "input_dimension": 1,
        "hidden_layers": 1,
        "hidden_width": 16,
        "softplus_beta": 2.0,
        "w_constraint": "unrestricted_signed",
        "v_constraint": "nonnegative_projected_after_each_optimizer_step",
        "fixed_mu": 0.001,
        "synthetic_seed": 1729,
        "synthetic_samples": 128,
        "synthetic_optimizer": "Adam",
        "synthetic_learning_rate": 0.01,
        "synthetic_max_steps": 600,
    }
    for name, expected in exact.items():
        if icnn.get(name) != expected:
            raise PartCRev1ICNNError(f"Frozen ICNN config mismatch for {name}")
    checkpoint = partC_rev1_resolve_campaign_output(
        config, "stage_03/synthetic_model_checkpoint.pt", create_parent=True
    )
    report_path = partC_rev1_resolve_campaign_output(config, "stage_03/synthetic_recovery.json")
    if report_path.exists() or checkpoint.exists():
        raise PartCRev1ICNNError("Synthetic CLI outputs must be new")
    result = partC_rev1_run_synthetic_recovery(
        seed=int(icnn["synthetic_seed"]),
        sample_count=int(icnn["synthetic_samples"]),
        max_steps=int(icnn["synthetic_max_steps"]),
        learning_rate=float(icnn["synthetic_learning_rate"]),
        checkpoint_path=checkpoint,
        config_sha256=binding["config_sha256"],
        code_commit=binding["current_head"],
    )
    result.update(
        {
            "campaign_id": config["campaign_id"],
            "run_id": config["run_id"],
            "stage_id": "step_1_5",
            "task_id": task_id,
            "checkpoint_path": checkpoint.as_posix(),
            "phase_2_authorized": False,
            "scientific_real_data_execution_performed": False,
        }
    )
    _atomic_json(report_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--synthetic-only", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _run_cli(Path(args.config).resolve(strict=True), str(args.task_id), bool(args.synthetic_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
