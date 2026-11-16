"""Exact physical-path occurrence classification and coarse projection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from .models import ProjectionOccurrence, ProjectionResult


class Phase3AProjectionError(ValueError):
    """Raised when a physical path cannot be classified exactly once."""


def project_physical_path(
    *,
    arc_indices: Sequence[int],
    arc_id: Sequence[str],
    tail: np.ndarray,
    head: np.ndarray,
    arc_fftt_s: np.ndarray,
    is_final_backbone_arc: np.ndarray,
    arc_region_id: Sequence[str],
    node_uid: Sequence[str],
    node_to_bacn: Mapping[str, str],
    bundle_by_arc_id: Mapping[str, str],
    terminal_by_region_node_role: Mapping[tuple[str, str, str], str],
    regional_pair_id: Mapping[tuple[str, str, str], str] | None = None,
    terminal_type_by_id: Mapping[str, str] | None = None,
    terminal_parent_object_by_id: Mapping[str, str] | None = None,
    internal_aug_patterns: Mapping[tuple[str, ...], str] | None = None,
    compiled_internal_aug_patterns: Mapping[
        str, Sequence[tuple[tuple[str, ...], str]]
    ] | None = None,
) -> ProjectionResult:
    """Project an ordered physical path into maximal frozen coarse occurrences."""

    path = tuple(int(value) for value in arc_indices)
    if not path:
        return ProjectionResult((), (), (), 0, 0, 0, 0, 0.0, 0.0, 0.0)
    arc_ids = tuple(str(value) for value in arc_id)
    tails = np.asarray(tail, dtype=np.int64)
    heads = np.asarray(head, dtype=np.int64)
    costs = np.asarray(arc_fftt_s, dtype=np.float64)
    backbone = np.asarray(is_final_backbone_arc, dtype=bool)
    regions = tuple(str(value) for value in arc_region_id)
    nodes = tuple(str(value) for value in node_uid)
    size = len(arc_ids)
    if any(len(value) != size for value in (tails, heads, costs, backbone, regions)):
        raise Phase3AProjectionError("Projection graph arrays have inconsistent lengths")
    if any(index < 0 or index >= size for index in path):
        raise Phase3AProjectionError("Physical path contains an out-of-range arc index")
    for previous, current in zip(path[:-1], path[1:], strict=True):
        if int(heads[previous]) != int(tails[current]):
            raise Phase3AProjectionError("Physical path arc endpoints are discontinuous")
    aug_patterns = (
        compile_internal_aug_patterns(internal_aug_patterns or {})
        if compiled_internal_aug_patterns is None
        else compiled_internal_aug_patterns
    )
    classifications = [""] * len(path)
    occurrences: list[ProjectionOccurrence] = []
    coarse_nodes: list[str] = []
    offset = 0
    while offset < len(path):
        aug_match = match_internal_aug_pattern(path, offset, arc_ids, aug_patterns)
        if aug_match is not None:
            pattern, bacn_id = aug_match
            end = offset + len(pattern)
            _append_occurrence(
                occurrences,
                coarse_nodes,
                classifications,
                path,
                arc_ids,
                costs,
                offset,
                end,
                kind="INTERNAL_AUG",
                object_id=bacn_id,
                region_id=None,
                coarse_tail=bacn_id,
                coarse_head=bacn_id,
                omitted=True,
            )
            offset = end
            continue
        arc = path[offset]
        stable_arc = arc_ids[arc]
        if backbone[arc]:
            tail_uid, head_uid = nodes[int(tails[arc])], nodes[int(heads[arc])]
            tail_bacn = node_to_bacn.get(tail_uid)
            head_bacn = node_to_bacn.get(head_uid)
            if tail_bacn is None or head_bacn is None:
                raise Phase3AProjectionError("Final-backbone path endpoint lacks B-ACN membership")
            if tail_bacn == head_bacn:
                kind, object_id, omitted = "INTERNAL_BACN", tail_bacn, True
            else:
                object_id = bundle_by_arc_id.get(stable_arc)
                if object_id is None:
                    raise Phase3AProjectionError("Inter-B-ACN final arc lacks a bundle")
                kind, omitted = "BUNDLE", False
            _append_occurrence(
                occurrences,
                coarse_nodes,
                classifications,
                path,
                arc_ids,
                costs,
                offset,
                offset + 1,
                kind=kind,
                object_id=str(object_id),
                region_id=None,
                coarse_tail=tail_bacn,
                coarse_head=head_bacn,
                omitted=omitted,
            )
            offset += 1
            continue
        region = regions[arc]
        if not region or region.lower() == "backbone":
            raise Phase3AProjectionError("Nonbackbone path arc lacks a retained-region identity")
        end = offset + 1
        while end < len(path):
            candidate = path[end]
            if backbone[candidate] or regions[candidate] != region:
                break
            if match_internal_aug_pattern(path, end, arc_ids, aug_patterns) is not None:
                break
            end += 1
        source_uid = nodes[int(tails[path[offset]])]
        target_uid = nodes[int(heads[path[end - 1]])]
        source_terminal = terminal_by_region_node_role.get((region, source_uid, "source"))
        target_terminal = terminal_by_region_node_role.get((region, target_uid, "target"))
        if source_terminal is None or target_terminal is None:
            raise Phase3AProjectionError(
                f"Regional segment endpoints lack directional terminals: {region} {source_uid}->{target_uid}"
            )
        if source_terminal == target_terminal:
            terminal_type = str((terminal_type_by_id or {}).get(source_terminal, "")).upper()
            parent_object = str((terminal_parent_object_by_id or {}).get(source_terminal, ""))
            if not terminal_type or not parent_object:
                source_bacn = node_to_bacn.get(source_uid)
                target_bacn = node_to_bacn.get(target_uid)
                if source_bacn is not None and source_bacn == target_bacn:
                    terminal_type, parent_object = "BACN", str(source_bacn)
            if terminal_type not in {"BACN", "RACN"} or not parent_object:
                raise Phase3AProjectionError(
                    "Same-terminal regional segment lacks a frozen internal aggregate identity"
                )
            kind = f"INTERNAL_{terminal_type}"
            object_id = parent_object
            occurrence_region: str | None = None
            coarse_tail = parent_object
            coarse_head = parent_object
            omitted = True
        else:
            pair_key = (region, source_terminal, target_terminal)
            object_id = (regional_pair_id or {}).get(
                pair_key,
                f"regional_pair:{region}:{source_terminal}:{target_terminal}",
            )
            kind = "REGIONAL_PAIR"
            occurrence_region = region
            coarse_tail = source_terminal
            coarse_head = target_terminal
            omitted = False
        _append_occurrence(
            occurrences,
            coarse_nodes,
            classifications,
            path,
            arc_ids,
            costs,
            offset,
            end,
            kind=kind,
            object_id=object_id,
            region_id=occurrence_region,
            coarse_tail=coarse_tail,
            coarse_head=coarse_head,
            omitted=omitted,
        )
        offset = end
    if any(not value for value in classifications):
        raise Phase3AProjectionError("At least one physical arc occurrence was not classified")
    if sum(item.arc_offset_end - item.arc_offset_begin for item in occurrences) != len(path):
        raise Phase3AProjectionError("Projection occurrences do not close over the physical path")
    concatenated = tuple(value for item in occurrences for value in item.physical_arc_ids)
    expected = tuple(arc_ids[index] for index in path)
    if concatenated != expected:
        raise Phase3AProjectionError("Projection witness slices do not reproduce the physical path")
    region_sequence = [item.region_id for item in occurrences if item.kind == "REGIONAL_PAIR"]
    counts = Counter(region_sequence)
    repeated = sum(max(0, value - 1) for value in counts.values())
    return ProjectionResult(
        occurrences=tuple(occurrences),
        coarse_sequence=tuple(coarse_nodes),
        classifications=tuple(classifications),
        accounted_arc_count=len(classifications),
        physical_arc_count=len(path),
        repeated_region_entry_count=int(repeated),
        regional_transition_count=len(region_sequence),
        omitted_suppressed_fftt_s=sum(item.physical_fftt_s for item in occurrences if item.kind == "INTERNAL_BACN"),
        omitted_aug_fftt_s=sum(item.physical_fftt_s for item in occurrences if item.kind == "INTERNAL_AUG"),
        omitted_racn_fftt_s=sum(item.physical_fftt_s for item in occurrences if item.kind == "INTERNAL_RACN"),
    )


def compile_internal_aug_patterns(
    patterns: Mapping[tuple[str, ...], str],
) -> dict[str, tuple[tuple[tuple[str, ...], str], ...]]:
    """Index frozen AUG patterns by first stable arc without changing priority."""

    grouped: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for raw_pattern, raw_bacn_id in patterns.items():
        pattern = tuple(map(str, raw_pattern))
        if not pattern:
            raise Phase3AProjectionError("Internal AUG pattern may not be empty")
        grouped.setdefault(pattern[0], []).append((pattern, str(raw_bacn_id)))
    return {
        first_arc: tuple(
            sorted(values, key=lambda item: (-len(item[0]), item[0], item[1]))
        )
        for first_arc, values in grouped.items()
    }


def match_internal_aug_pattern(
    path: tuple[int, ...],
    offset: int,
    arc_ids: Sequence[str],
    patterns: Mapping[str, Sequence[tuple[tuple[str, ...], str]]],
) -> tuple[tuple[str, ...], str] | None:
    if offset < 0 or offset >= len(path):
        return None
    first_arc = str(arc_ids[path[offset]])
    for pattern, bacn_id in patterns.get(first_arc, ()):
        if offset + len(pattern) <= len(path) and tuple(
            str(arc_ids[path[pos]]) for pos in range(offset, offset + len(pattern))
        ) == pattern:
            return pattern, bacn_id
    return None


def _append_occurrence(
    occurrences: list[ProjectionOccurrence],
    coarse_nodes: list[str],
    classifications: list[str],
    path: tuple[int, ...],
    arc_ids: tuple[str, ...],
    costs: np.ndarray,
    begin: int,
    end: int,
    *,
    kind: str,
    object_id: str,
    region_id: str | None,
    coarse_tail: str | None,
    coarse_head: str | None,
    omitted: bool,
) -> None:
    for index in range(begin, end):
        if classifications[index]:
            raise Phase3AProjectionError("A physical arc occurrence was classified more than once")
        classifications[index] = kind
    witness = tuple(arc_ids[path[index]] for index in range(begin, end))
    occurrence = ProjectionOccurrence(
        occurrence_index=len(occurrences),
        kind=kind,
        object_id=str(object_id),
        region_id=region_id,
        arc_offset_begin=begin,
        arc_offset_end=end,
        physical_arc_ids=witness,
        physical_fftt_s=float(sum(float(costs[path[index]]) for index in range(begin, end))),
        coarse_tail_id=coarse_tail,
        coarse_head_id=coarse_head,
        omitted_online=omitted,
    )
    occurrences.append(occurrence)
    for value in (coarse_tail, coarse_head):
        if value is not None and (not coarse_nodes or coarse_nodes[-1] != value):
            coarse_nodes.append(value)
