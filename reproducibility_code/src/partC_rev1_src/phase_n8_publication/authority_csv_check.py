#!/usr/bin/env python3
"""Recompute ACNHN backbone/key-link KPIs from explicit CSV exports only.

This script does NOT construct pi_s, infer semantic backbone roles, or recover
physical flows. The caller supplies a validated complete abstract-backbone registry.
Python standard library only. Run --help for input/output arguments.
"""
from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

IDENTITY = ('reference_id', 'graph_identity', 'mapping_identity')
REF_FIELDS = ('suite_id', 'abstract_link_id', 'lambda_reference') + IDENTITY
PRED_FIELDS = ('suite_id', 'method_id', 'state_id', 'abstract_link_id', 'lambda_estimate') + IDENTITY
KEY_FIELDS = ('suite_id', 'abstract_link_id')


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path.name}: empty input')
    for row in rows:
        if any(row.get(k) is None or not str(row[k]).strip() for k in required):
            raise ValueError(f'{path.name}: blank required value')
    return rows


def nonnegative(value: str, name: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f'{name}: flow must be finite and nonnegative')
    return x


def signed_bounded(prediction: float, reference: float) -> float:
    denominator = max(abs(prediction), abs(reference))
    return (prediction - reference) / denominator if denominator else 0.0


def aggregate(ref: list[float], pred: list[float]) -> dict[str, object]:
    if not ref or len(ref) != len(pred):
        raise ValueError('nonempty matched vectors required')
    delta = [p-r for r,p in zip(ref,pred)]
    den = math.fsum(ref)
    absolute = math.fsum(abs(d) for d in delta)
    n = len(ref)
    return {
        'link_count': n, 'reference_flow_sum': den,
        'absolute_error_sum': absolute,
        'wmape_percent': 100.0*absolute/den if den else '',
        'wmape_status': 'DEFINED' if den else 'UNDEFINED_ZERO_REFERENCE_TOTAL',
        'mae': absolute/n,
        'rmse': math.sqrt(math.fsum(d*d for d in delta)/n),
        'mean_signed_error': math.fsum(delta)/n,
        'reference_zero_link_count': sum(r == 0 for r in ref),
        'reference_zero_positive_prediction_count': sum(r == 0 and p > 0 for r,p in zip(ref,pred)),
    }


def compute(reference_rows, prediction_rows, key_rows, include_reference=True, reference_method_id="REFERENCE_X45"):
    if not reference_method_id:
        raise ValueError("reference method identity required")
    references = defaultdict(dict)
    identities = {}
    for row in reference_rows:
        s,e = row['suite_id'],row['abstract_link_id']
        if e in references[s]:
            raise ValueError(f'duplicate reference coordinate: {s}/{e}')
        ident = tuple(row[k] for k in IDENTITY)
        if s in identities and ident != identities[s]:
            raise ValueError(f'mixed reference/map/graph within suite: {s}')
        identities[s]=ident
        references[s][e]=nonnegative(row['lambda_reference'],f'{s}/{e}')
    if not references:
        raise ValueError('empty reference registry')
    keys=defaultdict(set)
    for row in key_rows:
        s,e=row['suite_id'],row['abstract_link_id']
        if s not in references or e not in references[s]:
            raise ValueError(f'key link not in reference backbone: {s}/{e}')
        if e in keys[s]:
            raise ValueError(f'duplicate key link: {s}/{e}')
        # Closed comparisons retain baseline-selected keys, including newly zero flows.
        if reference_method_id == "REFERENCE_X45" and references[s][e] <= 0:
            raise ValueError('selected high-flow key link has nonpositive reference')
        keys[s].add(e)
    if any(not keys[s] for s in references):
        raise ValueError('each suite requires a nonempty key-link set')
    groups=defaultdict(dict)
    states={}
    for row in prediction_rows:
        s,m,e=row['suite_id'],row['method_id'],row['abstract_link_id']
        if m == reference_method_id:
            raise ValueError(f"{reference_method_id} is reserved for the exact generated identity row")
        if s not in references or e not in references[s]:
            raise ValueError(f'prediction outside backbone reference: {s}/{e}')
        if tuple(row[k] for k in IDENTITY) != identities[s]:
            raise ValueError(f'prediction/reference identity mismatch: {s}/{m}')
        group=s,m
        if group in states and row['state_id'] != states[group]:
            raise ValueError(f'mixed prediction states: {s}/{m}')
        if e in groups[group]:
            raise ValueError(f'duplicate prediction coordinate: {s}/{m}/{e}')
        states[group]=row['state_id']
        groups[group][e]=nonnegative(row['lambda_estimate'],f'{s}/{m}/{e}')
    if not groups:
        raise ValueError('no admitted prediction groups')
    for (s,m),values in groups.items():
        missing=set(references[s])-set(values)
        if missing:
            raise ValueError(f'incomplete backbone coverage: {s}/{m}; missing {len(missing)} links')
    if include_reference:
        for s,values in references.items():
            groups[s,reference_method_id]=dict(values)
            states[s,reference_method_id]=identities[s][0]
    summaries=[]
    errors=[]
    for (s,m),values in sorted(groups.items()):
        ordered=sorted(references[s])
        ident=dict(zip(IDENTITY,identities[s]))
        base={'suite_id':s,'method_id':m,'state_id':states[s,m],**ident}
        den=math.fsum(references[s].values())
        for e in ordered:
            r,p=references[s][e],values[e]
            delta=p-r
            errors.append({**base,'abstract_link_id':e,'is_key_link':int(e in keys[s]),
                'lambda_reference':r,'lambda_estimate':p,'signed_error':delta,
                'absolute_error':abs(delta),'signed_bounded_error':signed_bounded(p,r),
                'signed_relative_error':delta/r if r else '',
                'absolute_relative_error':abs(delta)/r if r else '',
                'backbone_wmape_contribution_percent':100*abs(delta)/den if den else ''})
        for label,subset in [('BACKBONE',ordered),('KEY_LINKS',[e for e in ordered if e in keys[s]])]:
            ref=[references[s][e] for e in subset]
            pred=[values[e] for e in subset]
            result=aggregate(ref,pred)
            if m==reference_method_id:
                assert ref==pred
                for k in ['absolute_error_sum','mae','rmse','mean_signed_error']:
                    assert result[k]==0.0, 'reference identity failure'
                if result['wmape_status']=='DEFINED':
                    assert result['wmape_percent']==0.0, 'reference WMAPE failure'
            summaries.append({**base,'population':label,**result})
    return summaries,errors


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError('refusing to write an empty result')
    with path.open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path,help='Complete validated abstract-backbone reference CSV')
    parser.add_argument('--predictions',required=True,type=Path,help='One complete row per method and abstract backbone link')
    parser.add_argument('--keys',required=True,type=Path,help='Frozen baseline-only key-link membership CSV')
    parser.add_argument('--output',required=True,type=Path,help='New output directory; existing directory is rejected')
    parser.add_argument('--reference-method-id', default='REFERENCE_X45', help='Use a distinct identity for a matched closed reference')
    args=parser.parse_args()
    rows=read_csv(args.reference,REF_FIELDS)
    pred=read_csv(args.predictions,PRED_FIELDS)
    keys=read_csv(args.keys,KEY_FIELDS)
    metrics,errors=compute(rows,pred,keys,reference_method_id=args.reference_method_id)
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'summary_metrics.csv',metrics)
    write_csv(args.output/'per_link_errors.csv',errors)
    print(f'Wrote {len(metrics)} aggregate rows and {len(errors)} per-link rows to {args.output}')

if __name__=='__main__':
    main()
