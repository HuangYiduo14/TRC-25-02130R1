"""Final-only compact evidence; never dispatches an optimizer or changes eligibility."""
from pathlib import Path
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from .common import read_json, read_npz, write_json, write_frame, sha, input_hash
from .artifacts import describe
from .registry import region_dir


def file_binding(path):
    path = Path(path).resolve()
    rows, schema = describe(path)
    stat = path.stat()
    return dict(absolute_path=str(path), sha256=sha(path), size_bytes=stat.st_size,
                modified_unix_ns=stat.st_mtime_ns, format=path.suffix,
                row_or_object_count=rows, schema_or_shape=schema)


def reopen_rejected(candidate_json, graph_path, model_hash, support_hash):
    """Reprice a saved feasible candidate; retain its failure grade without solving."""
    from .oracle import Oracle
    meta = read_json(candidate_json)
    arrays = read_npz(Path(candidate_json).with_suffix('.npz'))
    if input_hash(model_hash, support_hash, arrays['demand']) != meta['input_hash']:
        raise ValueError('Rejected native input identity changed')
    oracle = Oracle(read_npz(graph_path))
    oracle.import_catalog({k.removeprefix('catalog_'): v for k, v in arrays.items()
                           if k.startswith('catalog_')})
    h = np.bincount(arrays['path_ids'], weights=arrays['weights'], minlength=len(oracle.paths))
    result = oracle.certify(arrays['demand'], h, 0, time.perf_counter(),
                            'FINAL_REJECTED_CANDIDATE_REOPEN', 0, oracle.lmo_count)
    for key in ['z', 'tau']:
        if not np.allclose(arrays[key], getattr(result, key), rtol=2e-11, atol=1e-8):
            raise ValueError('Rejected candidate refreshed arrays differ: ' + key)
    for key in ['relative_gap', 'value_upper', 'value_lower']:
        if not np.isclose(meta['scalars'][key], getattr(result, key), rtol=2e-9, atol=1e-7):
            raise ValueError('Rejected candidate refreshed scalar differs: ' + key)
    return dict(status='PASS_NATIVE_CANDIDATE_REOPEN', optimizer_solves=0,
                quality=result.quality, relative_gap=result.relative_gap,
                requested_target=meta['target'], eligible=False,
                path_demand_max_absolute_residual=oracle.validate(result.demand, result.path_ids, result.weights),
                centroid_nontransit_and_fine_endpoint_checks='PASS',
                value_lower=result.value_lower, value_upper=result.value_upper)


def optional_docket(c, regions, states):
    """Read durable batches once, including rejected, degenerate and deferred slots."""
    root = Path(c['output_root']); dbpath = root/'control/enrichment.sqlite'
    records = []; repairs = []; examples = []
    if not dbpath.exists():
        return records, repairs, examples
    descriptors = regions.set_index('region_id')
    # Durable aggregate counts identify every wave containing a non-label. Avoid
    # reopening tens of thousands of already-certified batch JSONs just to find a
    # few optional misses; native acceptance itself is independently reopened in R5.
    db = sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        paths = [r[0] for r in db.execute('''SELECT b.result_path FROM completed_batches AS b
            JOIN waves AS w ON b.round_index=w.round_index AND b.wave=w.wave AND b.region_id=w.region_id
            WHERE w.quarantined>0 OR w.degenerate>0 OR
                  w.next_slot>w.completed+w.quarantined+w.degenerate+w.skipped
            ORDER BY b.round_index,b.wave,b.region_id,b.start_slot''')]
    finally:
        db.close()
    for result_path in paths:
        result_path = Path(result_path); batch = read_json(result_path)
        nonlabels = [row for row in batch['rows'] if row['result'] != 'CERTIFIED']
        if not nonlabels:
            continue
        rid = batch['region_id']; rd = region_dir(c, rid); desc = descriptors.loc[rid]
        plan_path = result_path.with_name(result_path.name.replace('_result.json', '_plan.json'))
        inputs_path = result_path.with_name(result_path.name.replace('_result.json', '_inputs.npz'))
        plan = read_json(plan_path)
        if plan['model_hash'] != desc.model_hash or plan['support_hash'] != desc.support_hash or sha(inputs_path) != plan['inputs_sha256']:
            raise ValueError('Non-label batch identity changed: ' + str(plan_path))
        bindings = dict(plan=file_binding(plan_path), inputs=file_binding(inputs_path), batch_result=file_binding(result_path),
                        graph=file_binding(rd/'local_graph.npz'), catalog_index=file_binding(rd/'labels/catalog_index.json'))
        for row in nonlabels:
            ih = row.get('input_hash')
            if row['result'] != 'NO_UNIQUE_INPUT' and ih and (rd/'labels/receipts'/(ih+'.json')).exists():
                raise ValueError('A recorded non-label was later published without reconciled disposition')
            rec = dict(region_id=rid, kind='OPTIONAL_'+row['result'], input_hash=ih,
                       model_hash=desc.model_hash, support_hash=desc.support_hash,
                       round_index=batch['round_index'], wave=batch['wave'], family=row.get('family'),
                       seed=row.get('seed'), absolute_slot=row['absolute_slot'], input_index=row['input_index'],
                       draw_disposition=row.get('disposition'), redraw=row.get('redraw'), degeneracy=row.get('degeneracy'),
                       state=states[rid]['state'], regional_failure=False, baseline_preserved=True,
                       eligible_AB=False, eligible_ABC=False, native_sources=bindings,
                       requested_target=1e-5, measured_solver_seconds=row.get('seconds'),
                       replay_command=None, replay_command_status='NO_AUTHORIZED_EXACT_OPTIONAL_REPLAY_ENTRYPOINT',
                       next_repair='Under a new hash-bound decision, add an explicit exact-input replay role; preserve this seed, input and model. Do not draw a replacement or rerun the campaign.')
            candidate = rd/'labels/quarantine'/((ih or '')+'_normal.json')
            if row['result'] not in {'NO_UNIQUE_INPUT', 'DEFERRED_DEADLINE'}:
                attempt = read_json(candidate)
                if attempt['metadata']['run_id'] != c['run_id'] or attempt['metadata']['plan_sha256'] != sha(plan_path):
                    raise ValueError('Optional rejection provenance mismatch')
                reopened = reopen_rejected(candidate, rd/'local_graph.npz', desc.model_hash, desc.support_hash)
                if reopened['quality'] in {'A', 'B', 'C'}:
                    raise ValueError('Rejected optional candidate is actually publishable; explicit reconciliation required')
                rec.update(candidate_json=file_binding(candidate), candidate_npz=file_binding(candidate.with_suffix('.npz')),
                           achieved_relative_gap=attempt['scalars']['relative_gap'], solver_status=attempt['scalars']['status'],
                           execution_head=attempt['metadata'].get('execution_head'), target=attempt['target'],
                           scalars=attempt['scalars'], native_reopen=reopened,
                           diagnosis='Finite feasible candidate stopped above the ABC envelope; deeper numerical cause is unproved.',
                           exception_and_stack='No exception: normal bounded optimizer termination',
                           per_attempt_memory='Not individually measured; campaign process-tree resources are separately bound.')
                examples.append(rec)
            else:
                rec.update(achieved_relative_gap=None, diagnosis='Explicit non-label state; not a failed active region.')
            records.append(rec)
            if row['result'] != 'NO_UNIQUE_INPUT':
                repairs.append(rec)
    return records, repairs, examples


def deterministic_sample(frame, count=6):
    if not len(frame):
        return frame.copy()
    columns = [key for key in ['region_id', 'input_hash', 'relative_path', 'absolute_path'] if key in frame]
    ordered = frame.sort_values(columns, kind='stable') if columns else frame
    positions = np.unique(np.linspace(0, len(ordered)-1, min(count, len(ordered)), dtype=np.int64))
    return ordered.iloc[positions].copy()


def compact_external_evidence(root, destination):
    """All full shards remain local; bind them and include bounded review samples."""
    root = Path(root); destination = Path(destination)
    bindings = []; samples = {}; populations = []; stats = []
    groups = {
        'full_file_manifests': sorted((root/'manifests').glob('*.parquet')),
        'published_label_population': sorted((root/'reports/label_population').glob('*.parquet')),
        'fresh_cumulative_label_scalars': sorted((root/'validation/cumulative_labels').glob('*.parquet')),
        'copied_file_lineage': sorted((root/'validation/copy_lineage').glob('*.parquet')),
    }
    excluded = set()
    for group, paths in groups.items():
        selected = []
        for path in paths:
            frame = pd.read_parquet(path)
            b = file_binding(path); b.update(group=group, relative_path=path.relative_to(root).as_posix(),
                                            storage_policy='EXTERNAL_FULL_HASH_BOUND')
            bindings.append(b); excluded.add(path.resolve())
            sample = deterministic_sample(frame, 6 if 'label' in group else 8)
            sample['source_shard'] = path.relative_to(root).as_posix(); selected.append(sample)
            if group == 'published_label_population':
                keys = ['region_id', 'baseline', 'split', 'quality', 'origin_run_id', 'round_index']
                populations.append(frame.groupby(keys, dropna=False).agg(labels=('input_hash', 'size'),
                    unique_inputs=('input_hash', 'nunique'), gap_min=('relative_gap', 'min'),
                    gap_max=('relative_gap', 'max'), AB_eligible=('eligible_AB', 'sum')).reset_index())
            if 'label' in group:
                for key in frame.select_dtypes(include=np.number).columns:
                    values = frame[key].to_numpy(dtype=float); finite = values[np.isfinite(values)]
                    stats.append(dict(source_shard=b['relative_path'], column=key, rows=len(values),
                                      nonfinite=int((~np.isfinite(values)).sum()),
                                      minimum=float(finite.min()) if len(finite) else None,
                                      maximum=float(finite.max()) if len(finite) else None,
                                      median=float(np.median(finite)) if len(finite) else None))
        if selected:
            target = destination/(group+'_sample.parquet')
            combined = pd.concat(selected, ignore_index=True)
            write_frame(target, combined); samples[group] = dict(path=target.name, rows=len(combined),
                method='Stable lexical key order; six evenly spaced rows per label shard, eight per other shard, including first and last.')
    if populations:
        write_frame(destination/'complete_region_label_population_summary.parquet', pd.concat(populations, ignore_index=True))
    if stats:
        write_frame(destination/'full_label_numeric_statistics.parquet', pd.DataFrame(stats))
    write_json(destination/'FULL_LOCAL_SHARD_BINDINGS.json', dict(status='COMPLETE_EXTERNAL_INDEX',
        shard_count=len(bindings), shards=bindings, samples=samples,
        producer='partC_rev1_src.phase6', consumer='current cumulative R5 review and explicitly gated data loader',
        full_region_manifest_included=True, raw_multi_round_dataset_embedded=False))
    return excluded, dict(shard_count=len(bindings), full_shard_bytes=sum(r['size_bytes'] for r in bindings), samples=samples)


def privacy_and_format_scan(source):
    from partC_rev1_src.phase5.handoff import _privacy_scan
    import xml.etree.ElementTree as ET
    scan = _privacy_scan(source); unresolved = []; reviewed = []
    for finding in scan['findings']:
        path = Path(source)/finding['path']
        if finding['pattern'] == 'unreviewed_binary_type' and path.suffix in {'.npz', '.parquet'}:
            rows, schema = describe(path)
            reviewed.append(dict(**finding, rows=rows, typed_format_reopen='PASS'))
        elif finding['pattern'] == 'unreviewed_binary_type' and path.suffix == '.xml':
            ET.parse(path); reviewed.append(dict(**finding, typed_format_reopen='PASS'))
        else:
            unresolved.append(finding)
    scan.update(findings=unresolved, status='PASS' if not unresolved else 'FAIL', typed_format_reopens=reviewed)
    if unresolved:
        raise ValueError('Unresolved package scan findings: ' + str(unresolved))
    return scan
